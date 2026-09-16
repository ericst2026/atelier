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


# Which way is better for a figure, when the name says so plainly. Anything not
# recognised is shown without a winner rather than guessed at.
LOWER_IS_BETTER = ("loss", "perplexity", "ppl", "error", "err", "removed", "duplicate", "dup", "seconds", "sec", "ms", "latency", "time", "bytes", "memory", "mem", "cost")
HIGHER_IS_BETTER = ("accuracy", "acc", "score", "correct", "pass", "recall", "precision", "f1", "kept", "coverage", "throughput", "tokens_per", "speedup", "reward", "win")


def _better(key: str, label: str) -> int:
    """1 when more is better, -1 when less is, 0 when it is not for us to say."""
    text = f"{key} {label}".lower()
    for word in HIGHER_IS_BETTER:
        if word in text:
            return 1
    for word in LOWER_IS_BETTER:
        if word in text:
            return -1
    return 0


def _metric_pairs(mine: list[dict], standard: list[dict]) -> list[dict[str, Any]]:
    """The same figures side by side, with the better one marked where the figure
    says which way is better."""
    by_key = {m.get("key"): m for m in standard or []}
    rows = []
    for m in mine or []:
        other = by_key.pop(m.get("key"), None)
        row = {"key": m.get("key"), "label": m.get("label"), "fmt": m.get("fmt"), "mine": m.get("value"), "standard": other.get("value") if other else None}
        d = _better(str(row["key"] or ""), str(row["label"] or ""))
        if d and isinstance(row["mine"], (int, float)) and isinstance(row["standard"], (int, float)) and row["mine"] != row["standard"]:
            row["best"] = "mine" if (row["mine"] > row["standard"]) == (d > 0) else "standard"
        rows.append(row)
    for key, m in by_key.items():
        rows.append({"key": key, "label": m.get("label"), "fmt": m.get("fmt"), "mine": None, "standard": m.get("value")})
    return rows


def _handed_in(db: Session, slug: str, step_no: int) -> list[dict[str, Any]]:
    """Who has handed this step in, newest attempt each, in the order they did."""
    seen: set[int] = set()
    out = []
    for sub in db.scalars(select(Submission).where(Submission.experiment == slug, Submission.step == step_no).order_by(Submission.id.desc())).all():
        if sub.user_id in seen:
            continue
        seen.add(sub.user_id)
        u = db.get(User, sub.user_id)
        out.append({"user_id": sub.user_id, "name": (u.name or u.username) if u else "?", "submission_id": sub.id, "at": sub.created_at.isoformat(), "score": sub.score})
    return out


def _result_of(run_id: Optional[int]) -> dict[str, Any]:
    return storage.read_json(storage.run_dir(int(run_id)) / "result.json", {}) if run_id else {}


def step_view(db: Session, registry: Registry, payload: dict[str, Any]) -> dict[str, Any]:
    """What displays 1-4 show while a class runs: what the step is for, what a
    student's own version of it has to do, and who has handed theirs in."""
    session_id = payload.get("session_id")
    session = db.get(ClassSession, int(session_id)) if session_id else None
    # a class that has ended still shows if the teacher pointed a screen at it on
    # purpose; one a screen merely followed goes blank when the class is over
    if session is not None and session.ended_at is not None and not payload.get("pinned"):
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
        "class_name": session.name or "",
        "over": session.ended_at is not None,
        "step": step_no,
        "step_title": step.title,
        "summary": step.summary,
        "description": step.description,
        "own_code": step.own_code,
        "figure": f"/api/experiments/{slug}/figure/{step_no}",
        "instructions": c,
        "handed_in": handed,
    }


def standard_view(db: Session, registry: Registry, payload: dict[str, Any]) -> dict[str, Any]:
    """The code the experiment ships with, and what it produced on this step — the
    thing every student's own version is measured against."""
    session_id = payload.get("session_id")
    session = db.get(ClassSession, int(session_id)) if session_id else None
    if session_id and (session is None or (session.ended_at is not None and not payload.get("pinned"))):
        return {"no_class": True}
    slug = (session.experiment if session else payload.get("experiment")) or ""
    step_no = int(payload.get("step") or 1)
    try:
        spec = registry.get(slug)
        step = spec.step(step_no)
    except KeyError:
        return {"error": "that experiment is not loaded"}
    try:
        code = (spec.dir / step.script).read_text(encoding="utf-8", errors="replace")
    except OSError:
        code = ""
    out: dict[str, Any] = {
        "experiment": slug,
        "title": spec.title,
        "step": step_no,
        "step_title": step.title,
        "code": code,
        "handed_in": _handed_in(db, slug, step_no),
    }
    # what it produced: whatever this class last ran it for — a baseline queued by a
    # submission, or the teacher's own run of the step
    q = select(Run).where(Run.experiment == slug, Run.step == step_no, Run.status == "succeeded").order_by(Run.id.desc())
    for run in db.scalars(q).all():
        inputs = run.inputs or {}
        if inputs.get("code_source") == "own":
            continue
        if session is not None and inputs.get("class_session") not in (None, session.id) and run.user_id != session.started_by:
            continue
        out["result"] = _result_of(run.id)
        out["run"] = {"id": run.id, "by": (db.get(User, run.user_id).name or db.get(User, run.user_id).username) if db.get(User, run.user_id) else ""}
        break
    return out


def student_view(db: Session, registry: Registry, payload: dict[str, Any]) -> dict[str, Any]:
    """One person's handed-in work, put up by the teacher: their code, or what their
    code produced against the standard code on the same settings. The list of who has
    handed in stays beside it, with this one marked."""
    session_id = payload.get("session_id")
    session = db.get(ClassSession, int(session_id)) if session_id else None
    if session_id and (session is None or (session.ended_at is not None and not payload.get("pinned"))):
        return {"no_class": True}
    slug = (session.experiment if session else payload.get("experiment")) or ""
    step_no = int(payload.get("step") or 1)
    user_id, show = int(payload.get("user_id") or 0), (payload.get("show") or "results")
    u = db.get(User, user_id)
    out: dict[str, Any] = {"experiment": slug, "step": step_no, "show": show, "name": (u.name or u.username) if u else "?", "user_id": user_id}
    try:
        spec = registry.get(slug)
        out["title"], out["step_title"] = spec.title, spec.step(step_no).title
    except KeyError:
        return {"error": "that experiment is not loaded"}
    out["handed_in"] = _handed_in(db, slug, step_no)
    sub = db.scalar(select(Submission).where(Submission.experiment == slug, Submission.step == step_no, Submission.user_id == user_id).order_by(Submission.id.desc()))
    if sub is None:
        # the teacher's own work is not handed in to anybody, so it is read from the
        # last run they made of this step
        run = db.scalar(select(Run).where(Run.experiment == slug, Run.step == step_no, Run.user_id == user_id, Run.kind == "step", Run.status == "succeeded").order_by(Run.id.desc()))
        if run is None:
            out["error"] = "they have not handed this step in"
            return out
        out["their_own"] = True
        out["result"] = _result_of(run.id)
        out["run"] = {"id": run.id, "own_code": (run.inputs or {}).get("code_source") == "own"}
        return out
    out["submission_id"] = sub.id
    out["at"] = sub.created_at.isoformat()
    if show == "code":
        path = storage.submission_dir(sub.id) / f"step{step_no}.py"
        out["code"] = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        return out
    mine = _result_of(sub.own_run_id or sub.last_test_run_id)
    base = _result_of(sub.standard_run_id)
    out["result"] = mine
    out["compare"] = _metric_pairs(mine.get("metrics") or [], base.get("metrics") or [])
    if sub.standard_run_id and not base:
        run = db.get(Run, sub.standard_run_id)
        out["standard_state"] = run.status if run else "missing"
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
        elif d.mode == "standard":
            data = standard_view(db, registry, d.payload)
        out[str(d.id)] = {"id": d.id, "name": d.name, "mode": d.mode, "payload": d.payload, "updated_at": d.updated_at.isoformat(), "data": data}
    return out
