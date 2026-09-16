"""Run creation and access rules shared by several routers."""
from datetime import datetime
from typing import Any, Optional

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import materials, stepcode, storage
from .routers import classroom
from .bus import SyncBus, cluster_capacity
from .config import settings
from .models import Run, Submission, User
from .auth import is_staff
from .registry import ExperimentSpec, Registry
from .schemas import UserOut


def user_out(db: Session, user: User) -> UserOut:
    """A user as the app sees them, including the experiments an admin has granted
    them. Every endpoint that returns an account goes through this: /auth/me told
    people they had none, which emptied their page."""
    out = UserOut.model_validate(user)
    out.self_experiments = sorted(classroom.self_allowed(db, user.id))
    return out


def visible_run(db: Session, run_id: int, user: User) -> Run:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    if not is_staff(user) and run.user_id != user.id:
        raise HTTPException(403, "This run belongs to someone else")
    return run


def visible_submission(db: Session, submission_id: int, user: User) -> Submission:
    sub = db.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(404, "Submission not found")
    if not is_staff(user) and sub.user_id != user.id:
        raise HTTPException(403, "This submission belongs to someone else")
    return sub


def check_capacity(db: Session, user: User, gpus: int) -> None:
    if is_staff(user):
        cap = cluster_capacity(SyncBus())
        # a single run stays on one machine, so the biggest node is the limit. With no GPU
        # anywhere the run goes to CPU, and with no worker checked in the size is unknown:
        # either way let it queue.
        if cap["largest_node"] and gpus > cap["largest_node"]:
            where = f" (largest of {cap['nodes']} nodes; {cap['total']} in total)" if cap["nodes"] > 1 else ""
            raise HTTPException(400, f"The biggest GPU node has {cap['largest_node']} GPUs{where}, and one run cannot span machines")
        return
    if gpus > settings.max_gpus_per_student_run:
        raise HTTPException(400, f"Students can use up to {settings.max_gpus_per_student_run} GPUs per run")
    active = db.scalar(select(func.count()).select_from(Run).where(Run.user_id == user.id, Run.status.in_(["queued", "running"]))) or 0
    if active >= settings.max_running_per_student:
        raise HTTPException(429, f"You already have {active} runs queued or running. Wait for one to finish or cancel it.")


def _material_value(key: str, p: dict[str, Any], v: Any) -> Optional[str]:
    """A path under materials/models or materials/datasets. Whether it exists is
    checked by the run itself: on a split install the API has no materials to look at."""
    if v in (None, ""):
        return None
    rel = str(v).replace("\\", "/").strip().strip("/")
    top = "models/" if p.get("kind") == "model" else "datasets/"
    if not rel.startswith(top) or any(part in ("", ".", "..") for part in rel.split("/")):
        raise HTTPException(400, f"Parameter {key!r} must be a folder under materials/{top}")
    return rel


def coerce_params(spec_params: list[dict[str, Any]], given: dict[str, Any], spec: Optional[ExperimentSpec] = None) -> dict[str, Any]:
    """Fill defaults and clamp numbers to the declared ranges. A material param must
    be one of the experiment's declared materials for it."""
    out: dict[str, Any] = {}
    for p in spec_params:
        key = p["key"]
        t = p.get("type", "text")
        v = given.get(key, p.get("default"))
        try:
            if t == "int" and v is not None:
                v = int(v)
            elif t == "float" and v is not None:
                v = float(v)
            elif t == "bool":
                v = bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "yes", "on")
            elif t == "run" and v not in (None, ""):
                v = int(v)
            elif t == "material":
                v = _material_value(key, p, v)
                if v is not None and spec is not None:
                    allowed = [m["path"].replace("\\", "/").strip("/") for m in materials.choices(spec, p)]
                    if v not in allowed:
                        raise HTTPException(400, f"Parameter {key!r} must be one of the materials this experiment offers: {', '.join(allowed) or 'none are declared'}")
            elif t == "multiselect":
                v = list(v or [])
        except (TypeError, ValueError):
            raise HTTPException(400, f"Parameter {key!r} has the wrong type")
        if t in ("int", "float") and v is not None:
            if "min" in p and v < p["min"]:
                v = p["min"]
            if "max" in p and v > p["max"]:
                v = p["max"]
        if t in ("select",) and p.get("options"):
            allowed = [o["value"] if isinstance(o, dict) else o for o in p["options"]]
            if v not in allowed:
                raise HTTPException(400, f"Parameter {key!r} must be one of {allowed}")
        out[key] = v
    return out


def resolve_run_refs(db: Session, user: User, spec_params: list[dict[str, Any]], params: dict[str, Any]) -> dict[str, Any]:
    """Params of type `run` point at another run (possibly in another experiment);
    hand its outputs to the script under inputs[<key>_run]."""
    refs: dict[str, Any] = {}
    for p in spec_params:
        if p.get("type") != "run":
            continue
        rid = params.get(p["key"])
        if not rid:
            continue
        ref = visible_run(db, int(rid), user)
        if ref.status != "succeeded":
            raise HTTPException(400, f"Run {rid} has not finished successfully")
        want_exp, want_step = p.get("experiment"), p.get("step")
        if want_exp and ref.experiment != want_exp:
            raise HTTPException(400, f"Parameter {p['key']!r} needs a run from the {want_exp} experiment")
        if want_step and ref.step != int(want_step):
            raise HTTPException(400, f"Parameter {p['key']!r} needs a step {want_step} run")
        refs[f"{p['key']}_run"] = {"id": ref.id, "experiment": ref.experiment, "step": ref.step, "run_dir": str(storage.run_dir(ref.id)), "outputs": ref.outputs}
    return refs


def create_step_run(db: Session, bus: SyncBus, user: User, spec: ExperimentSpec, step_no: int, params: dict[str, Any], parent_run_id: Optional[int], gpus: Optional[int], label: str, code_source: str = "standard") -> Run:
    step = spec.step(step_no)
    own = code_source == "own"
    if own and not step.own_code:
        raise HTTPException(400, f"Step {step_no} of {spec.slug} runs the standard code only")
    script = stepcode.code_path(user.id, spec.slug, step_no) if own else spec.dir / step.script
    if own and not script.exists():
        raise HTTPException(400, "Write and save your own code for this step first")
    allowed, why = classroom.may_run(db, user, spec.slug)
    if not allowed:
        raise HTTPException(403, why)
    # in class, whatever the teacher set the experiment to begin from wins: a student
    # has not done the experiment this one builds on, so they cannot pick its run
    session = classroom.active_session(db)
    in_class = session is not None and session.experiment == spec.slug and (user.id == session.started_by or (classroom.membership(db, session.id, user.id) or None) is not None)
    if session is not None and session.experiment == spec.slug and user.id != session.started_by and session.params:
        fixed = {k: v for k, v in session.params.items() if k in {p["key"] for p in step.params}}
        params = {**params, **fixed}
    params = coerce_params(step.params, params, spec)
    inputs: dict[str, Any] = {
        # which class this run belongs to, so class work and a student's own work on
        # the same experiment stay apart even though they share a page
        "class_session": session.id if in_class else None,
        "code_source": "own" if own else "standard",
        "materials_dir": str(settings.materials_dir),
        "workspace_dir": str(storage.workspace_dir(user.id, spec.slug)),
        "experiment_dir": str(spec.dir),
    }
    if step.needs_previous:
        if not parent_run_id:
            raise HTTPException(400, f"Step {step_no} needs a finished step {step_no - 1} run to build on")
        parent = visible_run(db, parent_run_id, user)
        if parent.experiment != spec.slug or parent.kind != "step" or parent.step != step_no - 1:
            raise HTTPException(400, f"Run {parent_run_id} is not a step {step_no - 1} run of this experiment")
        if parent.status != "succeeded":
            raise HTTPException(400, f"Run {parent_run_id} did not finish successfully")
        inputs.update(parent.outputs or {})
        inputs["parent_run_id"] = parent.id
        inputs["parent_run_dir"] = str(storage.run_dir(parent.id))
    inputs.update(resolve_run_refs(db, user, step.params, params))
    want_gpus = step.gpus if gpus is None else int(gpus)
    if want_gpus > step.gpus and not is_staff(user):
        want_gpus = step.gpus
    check_capacity(db, user, want_gpus)
    run = Run(
        user_id=user.id,
        experiment=spec.slug,
        kind="step",
        step=step_no,
        parent_run_id=parent_run_id,
        label=(label or f"{spec.title} · {step.title}") + (" · own code" if own else ""),
        params=params,
        inputs=inputs,
        gpus=want_gpus,
        timeout_min=step.timeout_min,
        command=f"{settings.python_bin} {script} --run-dir {{run_dir}}",
    )
    db.add(run)
    db.commit()
    _materialize(run, spec)
    bus.enqueue(run.id)
    bus.publish_event({"type": "run", "id": run.id, "status": run.status, "user_id": user.id, "experiment": spec.slug})
    return run


def create_command_run(db: Session, bus: SyncBus, user: User, spec: ExperimentSpec, kind: str, command: str, cwd: str, gpus: int, label: str, timeout_min: int, submission_id: Optional[int] = None, extra_inputs: Optional[dict[str, Any]] = None) -> Run:
    check_capacity(db, user, gpus)
    inputs = {
        "materials_dir": str(settings.materials_dir),
        "workspace_dir": cwd,
        "experiment_dir": str(spec.dir),
    }
    inputs.update(extra_inputs or {})
    run = Run(
        user_id=user.id,
        experiment=spec.slug,
        kind=kind,
        step=None,
        submission_id=submission_id,
        label=label,
        params={},
        inputs=inputs,
        gpus=gpus,
        timeout_min=timeout_min,
        command=command,
    )
    db.add(run)
    db.commit()
    _materialize(run, spec)
    bus.enqueue(run.id)
    bus.publish_event({"type": "run", "id": run.id, "status": run.status, "user_id": user.id, "experiment": spec.slug})
    return run


def _materialize(run: Run, spec: ExperimentSpec) -> None:
    d = storage.run_dir(run.id)
    d.mkdir(parents=True, exist_ok=True)
    storage.write_json(d / "params.json", run.params)
    storage.write_json(d / "inputs.json", run.inputs)
    storage.write_json(
        d / "meta.json",
        {"run_id": run.id, "experiment": spec.slug, "kind": run.kind, "step": run.step, "user_id": run.user_id, "created_at": datetime.utcnow().isoformat(), "command": run.command},
    )


def user_progress(db: Session, user: User, registry: Registry) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rows = db.execute(
        select(Run.experiment, func.max(Run.step), func.count()).where(Run.user_id == user.id, Run.kind == "step", Run.status == "succeeded").group_by(Run.experiment)
    ).all()
    steps = {exp: (best, n) for exp, best, n in rows}
    subs = dict(db.execute(select(Submission.experiment, func.count()).where(Submission.user_id == user.id).group_by(Submission.experiment)).all())
    last = {}
    for r in db.scalars(select(Run).where(Run.user_id == user.id).order_by(Run.id.desc()).limit(200)).all():
        last.setdefault(r.experiment, {"id": r.id, "status": r.status, "kind": r.kind, "step": r.step, "created_at": r.created_at.isoformat()})
    for spec in registry.list():
        best, n = steps.get(spec.slug, (0, 0))
        out[spec.slug] = {"best_step": best or 0, "succeeded_runs": n, "submissions": subs.get(spec.slug, 0), "last_run": last.get(spec.slug)}
    return out
