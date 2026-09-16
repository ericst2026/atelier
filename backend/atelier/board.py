"""Read models for the wall displays and the teacher dashboard.

Everything here is computed server-side and pushed to display clients over a
websocket, so a display never needs credentials or polling logic."""
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import metrics, storage
from .bus import SyncBus
from .config import settings
from . import stepcode
from .models import ClassSession, Display, Run, Submission, User
from .registry import Registry

DEFAULT_DISPLAYS = [
    (1, "Hardware", "grafana", {}),
    (2, "Live runs", "live", {}),
    (3, "Class progress", "progress", {}),
    (4, "Leaderboard", "leaderboard", {}),
    (5, "Spotlight", "message", {"title": "Atelier", "text": "The teacher can cast any run's results here."}),
]


def ensure_displays(db: Session) -> None:
    existing = {d.id for d in db.scalars(select(Display)).all()}
    changed = False
    for i, name, mode, payload in DEFAULT_DISPLAYS[: settings.display_count]:
        if i not in existing:
            db.add(Display(id=i, name=name, mode=mode, payload=payload))
            changed = True
    if changed:
        db.commit()


def _sole_node(gpus: dict[str, Any]) -> Optional[str]:
    """Runs recorded before nodes existed have no node; if there is only one, it is that one."""
    nodes = gpus.get("nodes") or []
    return nodes[0] if len(nodes) == 1 else None


def _gpus_across_nodes() -> dict[str, Any]:
    """This machine's GPUs if it has any, plus every node that published a sample.
    On a single machine this is exactly what it always was."""
    local = metrics.gpu_snapshot()
    out: list[dict[str, Any]] = []
    nodes: list[str] = []
    seen: set[str] = set()
    if local.get("gpus"):
        for g in local["gpus"]:
            g.setdefault("node", settings.node_name)
            g["uid"] = f"{g['node']}:{g['index']}"
        out += local["gpus"]
        nodes.append(settings.node_name)
        seen.add(settings.node_name)
    try:
        for snap in metrics.remote_snapshots(SyncBus()):
            node = snap.get("node") or "node"
            if node in seen:
                continue
            for g in snap.get("gpus") or []:
                g["node"] = node
                g["uid"] = f"{node}:{g['index']}"
            out += snap.get("gpus") or []
            nodes.append(node)
            seen.add(node)
    except Exception:
        pass
    return {"gpus": out, "nodes": nodes, "at": local.get("at")}


def _elapsed(run: Run) -> float:
    if not run.started_at:
        return 0.0
    end = run.finished_at or datetime.utcnow()
    return (end - run.started_at).total_seconds()


def live(db: Session, worker_state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    users = {u.id: u for u in db.scalars(select(User)).all()}
    running = db.scalars(select(Run).where(Run.status == "running").order_by(Run.started_at)).all()
    queued = db.scalars(select(Run).where(Run.status == "queued").order_by(Run.id)).all()
    since = datetime.utcnow() - timedelta(minutes=30)
    recent = db.scalars(
        select(Run).where(Run.status.in_(["succeeded", "failed", "cancelled"]), Run.finished_at >= since).order_by(Run.finished_at.desc()).limit(12)
    ).all()

    def row(r: Run) -> dict[str, Any]:
        u = users.get(r.user_id)
        return {
            "id": r.id,
            "user": u.name or u.username if u else "?",
            "experiment": r.experiment,
            "kind": r.kind,
            "step": r.step,
            "label": r.label,
            "status": r.status,
            "gpu_ids": r.gpu_ids,
            "gpus": r.gpus,
            "progress_pct": r.progress_pct,
            "progress_msg": r.progress_msg,
            "elapsed": _elapsed(r),
            "metrics": r.metrics[:4],
        }

    gpus = _gpus_across_nodes()
    # GPU 3 on one machine is not GPU 3 on another, so ownership is keyed by both
    owner: dict[tuple[str, int], dict[str, Any]] = {}
    for r in running:
        node = r.node or _sole_node(gpus)
        for g in r.gpu_ids or []:
            owner[(node, g)] = {"run_id": r.id, "user": row(r)["user"], "experiment": r.experiment, "node": node}
    for g in gpus["gpus"]:
        g["job"] = owner.get((g.get("node"), g["index"]))
    counts = {s: 0 for s in ("queued", "running", "succeeded", "failed", "cancelled")}
    for status, n in db.execute(select(Run.status, func.count()).group_by(Run.status)).all():
        counts[status] = n
    active_since = datetime.utcnow() - timedelta(minutes=15)
    active_users = db.scalar(select(func.count(func.distinct(Run.user_id))).where(Run.created_at >= active_since)) or 0
    metrics.update_run_gauges(counts, sum(r.gpus for r in running), active_users)
    return {
        "gpus": gpus,
        "gpu_count": len(gpus["gpus"]) or settings.gpu_count,
        "nodes": gpus.get("nodes", []),
        "running": [row(r) for r in running],
        "queued": [row(r) for r in queued],
        "recent": [row(r) for r in recent],
        "counts": counts,
        "worker": worker_state,
        "generated_at": datetime.utcnow().isoformat(),
    }


def progress(db: Session, registry: Registry) -> dict[str, Any]:
    students = db.scalars(select(User).where(User.role == "student", User.active == True)).all()  # noqa: E712
    student_ids = {u.id for u in students}
    done = db.execute(
        select(Run.user_id, Run.experiment, func.max(Run.step)).where(Run.kind == "step", Run.status == "succeeded").group_by(Run.user_id, Run.experiment)
    ).all()
    best: dict[tuple[int, str], int] = {(uid, exp): step for uid, exp, step in done if uid in student_ids}
    subs = db.execute(select(Submission.user_id, Submission.experiment, func.count()).group_by(Submission.user_id, Submission.experiment)).all()
    submitted: dict[tuple[int, str], int] = {(uid, exp): n for uid, exp, n in subs if uid in student_ids}
    out = []
    for spec in registry.list():
        at_step = [0] * 5
        for u in students:
            at_step[best.get((u.id, spec.slug), 0)] += 1
        out.append(
            {
                "slug": spec.slug,
                "title": spec.title,
                "students": len(students),
                "at_step": at_step,
                "submitted": sum(1 for (uid, exp) in submitted if exp == spec.slug),
                "steps": [s.title for s in spec.steps],
            }
        )
    return {"experiments": out, "students": len(students), "generated_at": datetime.utcnow().isoformat()}


def _metric_value(run_metrics: list[Any], key: str) -> Optional[float]:
    for m in run_metrics or []:
        if isinstance(m, dict) and m.get("key") == key:
            try:
                return float(m.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def leaderboard(db: Session, registry: Registry, experiment: Optional[str], metric: Optional[str] = None, include_unpublished: bool = False) -> dict[str, Any]:
    specs = registry.list()
    if experiment:
        specs = [s for s in specs if s.slug == experiment]
    users = {u.id: u for u in db.scalars(select(User)).all()}
    boards = []
    for spec in specs:
        key = metric or spec.leaderboard.get("metric") or "score"
        higher = bool(spec.leaderboard.get("higher_is_better", True))
        q = select(Submission).where(Submission.experiment == spec.slug)
        if not include_unpublished:
            q = q.where(Submission.published == True)  # noqa: E712
        subs = db.scalars(q).all()
        rows = []
        for s in subs:
            grade_run = db.get(Run, s.last_test_run_id) if s.last_test_run_id else None
            value = _metric_value(grade_run.metrics if grade_run else [], key)
            if value is None and key == "score":
                value = s.score if s.score is not None else s.auto_score
            u = users.get(s.user_id)
            rows.append(
                {
                    "submission_id": s.id,
                    "user": (u.name or u.username) if u else "?",
                    "value": value,
                    "score": s.score if s.score is not None else s.auto_score,
                    "submitted_at": s.created_at.isoformat(),
                    "metrics": grade_run.metrics[:5] if grade_run else [],
                }
            )
        rows = [r for r in rows if r["value"] is not None]
        rows.sort(key=lambda r: r["value"], reverse=higher)
        boards.append({"slug": spec.slug, "title": spec.title, "metric": key, "higher_is_better": higher, "rows": rows[:25]})
    return {"boards": boards, "generated_at": datetime.utcnow().isoformat()}


def run_view(db: Session, run_id: int) -> Optional[dict[str, Any]]:
    run = db.get(Run, run_id)
    if run is None:
        return None
    user = db.get(User, run.user_id)
    result = storage.read_json(storage.run_dir(run.id) / "result.json", None)
    live_series = storage.read_json(storage.run_dir(run.id) / "live.json", None)
    return {
        "run": {
            "id": run.id,
            "user": (user.name or user.username) if user else "?",
            "experiment": run.experiment,
            "kind": run.kind,
            "step": run.step,
            "label": run.label,
            "status": run.status,
            "progress_pct": run.progress_pct,
            "progress_msg": run.progress_msg,
            "elapsed": _elapsed(run),
            "metrics": run.metrics,
        },
        "result": result,
        "live": live_series,
        "log_tail": storage.last_lines(storage.run_dir(run.id) / "run.log", 12),
    }


def _standard_run(db: Session, slug: str, step_no: int, teacher_id: Optional[int] = None) -> Optional[Run]:
    """What the step produces when the code that ships with it is run: the teacher's
    latest such run, or anyone's if the teacher has not run it."""
    base = select(Run).where(Run.experiment == slug, Run.step == step_no, Run.kind == "step", Run.status == "succeeded").order_by(Run.id.desc())
    for q in ([base.where(Run.user_id == teacher_id)] if teacher_id else []) + [base]:
        for run in db.scalars(q).all():
            if (run.inputs or {}).get("code_source") != "own":
                return run
    return None


def _metric_pairs(mine: list[dict], standard: list[dict]) -> list[dict[str, Any]]:
    """The same figures side by side, in the order the step reports them."""
    by_key = {m.get("key"): m for m in standard or []}
    rows = []
    for m in mine or []:
        other = by_key.pop(m.get("key"), None)
        rows.append({"key": m.get("key"), "label": m.get("label"), "fmt": m.get("fmt"), "mine": m.get("value"), "standard": other.get("value") if other else None})
    for key, m in by_key.items():
        rows.append({"key": key, "label": m.get("label"), "fmt": m.get("fmt"), "mine": None, "standard": m.get("value")})
    return rows


def step_view(db: Session, registry: Registry, payload: dict[str, Any]) -> dict[str, Any]:
    """What displays 1-4 show while a class runs: what the step is for, what a
    student's own version of it has to do, and who has handed theirs in."""
    session_id = payload.get("session_id")
    session = db.get(ClassSession, int(session_id)) if session_id else None
    if session is not None and session.ended_at is not None:
        session = None
    slug = (session.experiment if session else payload.get("experiment")) or ""
    step_no = int(payload.get("step") or 1)
    if session is None and not slug:
        return {"no_class": True}
    try:
        spec = registry.get(slug)
        step = spec.step(step_no)
    except KeyError:
        return {"error": "that experiment is not loaded"}
    if session is None:
        return {"no_class": True, "title": spec.title, "step": step_no, "step_title": step.title}
    c = stepcode.contract(spec, step)
    seen: set[int] = set()
    handed = []
    for sub in db.scalars(select(Submission).where(Submission.experiment == slug, Submission.step == step_no).order_by(Submission.id.desc())).all():
        if sub.user_id in seen:
            continue
        seen.add(sub.user_id)
        u = db.get(User, sub.user_id)
        handed.append({"user_id": sub.user_id, "name": (u.name or u.username) if u else "?", "submission_id": sub.id, "at": sub.created_at.isoformat()})
    return {
        "experiment": slug,
        "title": spec.title,
        "step": step_no,
        "step_title": step.title,
        "summary": step.summary,
        "description": step.description,
        "own_code": step.own_code,
        "instructions": c,
        "handed_in": handed,
    }


def student_view(db: Session, registry: Registry, payload: dict[str, Any]) -> dict[str, Any]:
    """One student put up by the teacher: their code, or their results against what
    the standard code produced for the same step."""
    session_id = payload.get("session_id")
    session = db.get(ClassSession, int(session_id)) if session_id else None
    if session_id and (session is None or session.ended_at is not None):
        return {"no_class": True}  # the class it belonged to is over
    slug = (session.experiment if session else payload.get("experiment")) or ""
    step_no = int(payload.get("step") or 1)
    user_id, show = int(payload.get("user_id") or 0), (payload.get("show") or "results")
    u = db.get(User, user_id)
    out: dict[str, Any] = {"experiment": slug, "step": step_no, "show": show, "name": (u.name or u.username) if u else "?"}
    try:
        spec = registry.get(slug)
        out["title"], out["step_title"] = spec.title, spec.step(step_no).title
    except KeyError:
        return {"error": "that experiment is not loaded"}
    if show == "code":
        sub = db.scalar(select(Submission).where(Submission.experiment == slug, Submission.step == step_no, Submission.user_id == user_id).order_by(Submission.id.desc()))
        if sub is None:
            out["error"] = "they have not handed in this step"
            return out
        path = storage.submission_dir(sub.id) / f"step{step_no}.py"
        out["code"] = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        out["at"] = sub.created_at.isoformat()
        return out
    run = db.scalar(select(Run).where(Run.experiment == slug, Run.step == step_no, Run.user_id == user_id, Run.kind == "step", Run.status == "succeeded").order_by(Run.id.desc()))
    if run is None:
        out["error"] = "no finished run of this step yet"
        return out
    result = storage.read_json(storage.run_dir(run.id) / "result.json", {})
    out["run"] = {"id": run.id, "label": run.label, "metrics": run.metrics, "own_code": (run.inputs or {}).get("code_source") == "own"}
    out["result"] = result
    standard = _standard_run(db, slug, step_no, session.started_by if session else None)
    if standard is not None and standard.id != run.id:
        base = storage.read_json(storage.run_dir(standard.id) / "result.json", {})
        out["standard"] = {"run_id": standard.id, "result": base}
        out["compare"] = _metric_pairs(result.get("metrics") or [], base.get("metrics") or [])
    return out


def display_states(db: Session, registry: Registry, worker_state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Everything the display clients need, keyed by display id."""
    ensure_displays(db)
    displays = db.scalars(select(Display).order_by(Display.id)).all()
    shared: dict[str, Any] = {}
    out: dict[str, Any] = {}
    for d in displays:
        data: Any = None
        if d.mode == "live":
            shared.setdefault("live", live(db, worker_state))
            data = shared["live"]
        elif d.mode == "progress":
            shared.setdefault("progress", progress(db, registry))
            data = shared["progress"]
        elif d.mode == "leaderboard":
            data = leaderboard(db, registry, d.payload.get("experiment"), d.payload.get("metric"))
        elif d.mode == "run":
            data = run_view(db, int(d.payload.get("run_id", 0) or 0))
        elif d.mode == "grafana":
            data = {"url": d.payload.get("url") or settings.grafana_url}
        elif d.mode == "step":
            data = step_view(db, registry, d.payload or {})
        elif d.mode == "student":
            data = student_view(db, registry, d.payload)
        out[str(d.id)] = {"id": d.id, "name": d.name, "mode": d.mode, "payload": d.payload, "updated_at": d.updated_at.isoformat(), "data": data}
    return out
