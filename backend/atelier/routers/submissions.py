from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import services, stepcode, storage
from ..auth import current_user, is_staff, require_teacher
from ..bus import SyncBus
from ..config import settings
from ..db import get_db
from ..deps import get_bus, get_registry
from ..models import Run, Submission, User
from ..registry import Registry
from ..schemas import GradeIn, RunOut, SubmissionCreate, SubmissionOut

router = APIRouter(tags=["submissions"])


def _out(sub: Submission, users: dict[int, User]) -> SubmissionOut:
    o = SubmissionOut.model_validate(sub)
    u = users.get(sub.user_id)
    if u:
        o.username, o.name = u.username, u.name
    return o


def _grade_run(db: Session, bus: SyncBus, actor: User, sub: Submission, registry: Registry) -> Optional[Run]:
    spec = registry.get(sub.experiment)
    grader = spec.grader
    if not grader.get("script"):
        return None
    cmd = f"{settings.python_bin} {spec.dir / grader['script']} --project {{workspace_dir}} --run-dir {{run_dir}}"
    run = services.create_command_run(
        db, bus, actor, spec, "grade", cmd, str(storage.submission_dir(sub.id)), int(grader.get("gpus", 0)), f"Grade · submission {sub.id}", int(grader.get("timeout_min", 30)), submission_id=sub.id, extra_inputs={"submission_id": sub.id, "submitted_by": sub.user_id}
    )
    sub.last_test_run_id = run.id
    db.commit()
    return run


@router.post("/experiments/{slug}/submissions", response_model=SubmissionOut, status_code=201)
def submit(slug: str, body: SubmissionCreate, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    """Hand in your own code for one step, for the teacher to read and mark."""
    try:
        spec = registry.get(slug)
    except KeyError:
        raise HTTPException(404, "Experiment not found")
    try:
        step = spec.step(int(body.step))
    except (KeyError, ValueError):
        raise HTTPException(400, "Say which step you are handing in")
    if not step.own_code:
        raise HTTPException(400, f"Step {step.index} runs the standard code only, so there is nothing to hand in")
    src = stepcode.code_path(user.id, slug, step.index)
    if not src.exists():
        raise HTTPException(400, "Write and save your own code for this step first")
    code = src.read_text(encoding="utf-8", errors="replace")
    if not code.strip():
        raise HTTPException(400, "Your code for this step is empty")
    # the run it last produced, so the teacher can see it working (or not)
    last = db.scalars(
        select(Run).where(Run.user_id == user.id, Run.experiment == slug, Run.step == step.index, Run.kind == "step").order_by(Run.id.desc()).limit(1)
    ).first()
    sub = Submission(user_id=user.id, experiment=slug, step=step.index, note=body.note[:4000])
    db.add(sub)
    db.commit()
    dest = storage.submission_dir(sub.id)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / src.name).write_text(code, encoding="utf-8", newline="")
    stats = {"files": 1, "bytes": len((dest / src.name).read_bytes())}
    if last is not None:
        sub.last_test_run_id = last.id
    sub.sha256 = storage.sha256_dir(dest)
    sub.file_count, sub.bytes = stats["files"], stats["bytes"]
    db.commit()
    # no automatic grading: a step submission is code for the teacher to read, and
    # the run it produced is linked above
    bus.publish_event({"type": "submission", "id": sub.id, "user_id": user.id, "experiment": slug})
    return _out(sub, {user.id: user})


@router.get("/submissions", response_model=list[SubmissionOut])
def list_submissions(experiment: Optional[str] = None, user_id: Optional[int] = None, limit: int = Query(200, le=2000), user: User = Depends(current_user), db: Session = Depends(get_db)):
    q = select(Submission).order_by(Submission.id.desc()).limit(limit)
    if not is_staff(user):
        q = q.where(Submission.user_id == user.id)
    elif user_id:
        q = q.where(Submission.user_id == user_id)
    if experiment:
        q = q.where(Submission.experiment == experiment)
    subs = db.scalars(q).all()
    users = {u.id: u for u in db.scalars(select(User)).all()}
    return [_out(s, users) for s in subs]


@router.get("/submissions/{submission_id}", response_model=SubmissionOut)
def get_submission(submission_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    sub = services.visible_submission(db, submission_id, user)
    return _out(sub, {sub.user_id: db.get(User, sub.user_id)})


@router.get("/submissions/{submission_id}/tree")
def submission_tree(submission_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    services.visible_submission(db, submission_id, user)
    return {"entries": storage.tree(storage.submission_dir(submission_id))}


@router.get("/submissions/{submission_id}/file")
def submission_file(submission_id: int, path: str = Query(...), user: User = Depends(current_user), db: Session = Depends(get_db)):
    services.visible_submission(db, submission_id, user)
    try:
        return dict(storage.read_text(storage.safe_join(storage.submission_dir(submission_id), path)), path=path)
    except storage.StorageError as exc:
        raise HTTPException(404, str(exc))


@router.get("/submissions/{submission_id}/download")
def submission_download(submission_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    sub = services.visible_submission(db, submission_id, user)
    return Response(storage.zip_dir(storage.submission_dir(sub.id)), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="submission-{sub.id}.zip"'})


@router.get("/submissions/{submission_id}/runs", response_model=list[RunOut])
def submission_runs(submission_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    services.visible_submission(db, submission_id, user)
    runs = db.scalars(select(Run).where(Run.submission_id == submission_id).order_by(Run.id.desc())).all()
    users = {u.id: u for u in db.scalars(select(User).where(User.id.in_({r.user_id for r in runs}))).all()} if runs else {}
    out = []
    for r in runs:
        o = RunOut.model_validate(r)
        o.username = users[r.user_id].username if r.user_id in users else ""
        out.append(o)
    return out


@router.post("/submissions/{submission_id}/test", response_model=RunOut, status_code=201)
def test_submission(submission_id: int, teacher: User = Depends(require_teacher), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    sub = db.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(404, "Submission not found")
    run = _grade_run(db, bus, teacher, sub, registry)
    if run is None:
        raise HTTPException(400, "This experiment has no grader")
    o = RunOut.model_validate(run)
    o.username = teacher.username
    return o


@router.post("/submissions/{submission_id}/grade", response_model=SubmissionOut)
def grade(submission_id: int, body: GradeIn, teacher: User = Depends(require_teacher), db: Session = Depends(get_db), bus: SyncBus = Depends(get_bus)):
    sub = db.get(Submission, submission_id)
    if sub is None:
        raise HTTPException(404, "Submission not found")
    if body.score is not None:
        sub.score = body.score
        sub.graded_by, sub.graded_at = teacher.id, datetime.utcnow()
    if body.feedback is not None:
        sub.feedback = body.feedback
        sub.graded_by, sub.graded_at = teacher.id, datetime.utcnow()
    if body.published is not None:
        sub.published = body.published
    db.commit()
    bus.publish_event({"type": "submission", "id": sub.id, "graded": True})
    return _out(sub, {sub.user_id: db.get(User, sub.user_id)})
