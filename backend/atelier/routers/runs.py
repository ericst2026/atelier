from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import services, storage
from ..auth import current_user, is_staff, require_teacher
from ..bus import SyncBus
from ..db import get_db
from ..deps import get_bus, get_registry
from ..models import ClassSession, Run, User
from ..registry import Registry
from ..schemas import RunOut, StepRunCreate

router = APIRouter(tags=["runs"])


def _out(run: Run, users: dict[int, User], classes: Optional[dict[int, ClassSession]] = None) -> RunOut:
    o = RunOut.model_validate(run)
    u = users.get(run.user_id)
    o.username = u.username if u else ""
    c = (classes or {}).get((run.inputs or {}).get("class_session"))
    if c is not None:
        o.class_name = c.name or c.experiment
        o.class_teacher = getattr(c, "teacher_name", "") or ""
    return o


def _classes(db: Session, runs: list[Run]) -> dict[int, ClassSession]:
    """The classes these runs belong to, each carrying its teacher's name."""
    ids = {cid for r in runs if (cid := (r.inputs or {}).get("class_session"))}
    if not ids:
        return {}
    sessions = list(db.scalars(select(ClassSession).where(ClassSession.id.in_(ids))).all())
    teachers = {u.id: u for u in db.scalars(select(User).where(User.id.in_([c.started_by for c in sessions]))).all()}
    for c in sessions:
        t = teachers.get(c.started_by)
        c.teacher_name = (t.name or t.username) if t else ""
    return {c.id: c for c in sessions}


@router.post("/experiments/{slug}/steps/{step_no}/runs", response_model=RunOut, status_code=201)
def start_step(slug: str, step_no: int, body: StepRunCreate, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    try:
        spec = registry.get(slug)
        spec.step(step_no)
    except KeyError:
        raise HTTPException(404, "Unknown experiment or step")
    run = services.create_step_run(db, bus, user, spec, step_no, body.params, body.parent_run_id, body.gpus, body.label, body.code_source)
    return _out(run, {user.id: user})


@router.get("/runs", response_model=list[RunOut])
def list_runs(
    experiment: Optional[str] = None,
    kind: Optional[str] = None,
    step: Optional[int] = None,
    status: Optional[str] = None,
    user_id: Optional[int] = None,
    mine: bool = False,
    limit: int = Query(100, le=1000),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    q = select(Run).order_by(Run.id.desc()).limit(limit)
    if not is_staff(user) or mine:
        q = q.where(Run.user_id == user.id)
    elif user_id:
        q = q.where(Run.user_id == user_id)
    if experiment:
        q = q.where(Run.experiment == experiment)
    if kind:
        q = q.where(Run.kind == kind)
    if step:
        q = q.where(Run.step == step)
    if status:
        q = q.where(Run.status.in_(status.split(",")))
    runs = db.scalars(q).all()
    users = {u.id: u for u in db.scalars(select(User).where(User.id.in_({r.user_id for r in runs}))).all()} if runs else {}
    return [_out(r, users, _classes(db, runs)) for r in runs]


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = services.visible_run(db, run_id, user)
    return _out(run, {run.user_id: db.get(User, run.user_id)})


@router.get("/runs/{run_id}/logs")
def run_logs(run_id: int, offset: int = -1, user: User = Depends(current_user), db: Session = Depends(get_db)):
    services.visible_run(db, run_id, user)
    return storage.tail_log(storage.run_dir(run_id) / "run.log", offset)


@router.get("/runs/{run_id}/result")
def run_result(run_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    run = services.visible_run(db, run_id, user)
    result = storage.read_json(storage.run_dir(run_id) / "result.json")
    if result is None:
        raise HTTPException(404, "No result yet" if run.status in ("queued", "running") else "This run produced no result.json")
    return result


@router.get("/runs/{run_id}/live")
def run_live(run_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    services.visible_run(db, run_id, user)
    return storage.read_json(storage.run_dir(run_id) / "live.json", {"series": {}})


@router.get("/runs/{run_id}/artifacts")
def run_artifacts(run_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    services.visible_run(db, run_id, user)
    return {"entries": storage.tree(storage.run_dir(run_id), max_depth=4)}


@router.get("/runs/{run_id}/artifacts/{path:path}")
def run_artifact(run_id: int, path: str, user: User = Depends(current_user), db: Session = Depends(get_db)):
    services.visible_run(db, run_id, user)
    try:
        p = storage.safe_join(storage.run_dir(run_id), path)
    except storage.StorageError:
        raise HTTPException(404, "Not found")
    if not p.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(p, filename=p.name)


@router.post("/runs/{run_id}/cancel", response_model=RunOut)
def cancel_run(run_id: int, user: User = Depends(current_user), db: Session = Depends(get_db), bus: SyncBus = Depends(get_bus)):
    run = services.visible_run(db, run_id, user)
    if run.status not in ("queued", "running"):
        raise HTTPException(400, f"Run is already {run.status}")
    bus.request_cancel(run.id)
    if run.status == "queued":
        run.status = "cancelled"
        run.error = "Cancelled before it started"
        db.commit()
        bus.publish_run(run.id, {"type": "status", "status": "cancelled"})
        bus.publish_event({"type": "run", "id": run.id, "status": "cancelled"})
    return _out(run, {run.user_id: db.get(User, run.user_id)})


@router.delete("/runs/{run_id}", dependencies=[Depends(require_teacher)])
def delete_run(run_id: int, db: Session = Depends(get_db)):
    import shutil

    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    if run.status in ("queued", "running"):
        raise HTTPException(400, "Cancel the run first")
    shutil.rmtree(storage.run_dir(run_id), ignore_errors=True)
    db.delete(run)
    db.commit()
    return {"ok": True}
