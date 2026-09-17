from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import board
from ..auth import optional_user, require_teacher
from ..bus import SyncBus
from ..config import settings
from ..db import get_db
from ..deps import get_bus, get_registry
from ..models import Display, Run, User
from ..registry import Registry
from ..schemas import DisplayIn, DisplayOut

router = APIRouter(prefix="/displays", tags=["displays"])
# step    the step's own code, and who has handed theirs in
# student one student's work on a step: their code, or the results they got
MODES = {"grafana", "live", "progress", "leaderboard", "run", "message", "step", "student", "standard"}


@router.get("", response_model=list[DisplayOut])
def list_displays(user: User | None = Depends(optional_user), db: Session = Depends(get_db)):
    if user is None and not settings.displays_public:
        raise HTTPException(401, "Sign in to continue")
    board.ensure_displays(db)
    return db.scalars(select(Display).order_by(Display.id)).all()


@router.get("/{n}")
def get_display(n: int, user: User | None = Depends(optional_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    if user is None and not settings.displays_public:
        raise HTTPException(401, "Sign in to continue")
    states = board.display_states(db, registry, bus.worker_state())
    if str(n) not in states:
        raise HTTPException(404, "No such display")
    return states[str(n)]


@router.put("/{n}", response_model=DisplayOut)
def set_display(n: int, body: DisplayIn, teacher: User = Depends(require_teacher), db: Session = Depends(get_db), bus: SyncBus = Depends(get_bus)):
    board.ensure_displays(db)
    d = db.get(Display, n)
    if d is None:
        raise HTTPException(404, "No such display")
    # the wall belongs to the class in the room: while one runs, only its teacher
    # (or an admin) decides what the screens show
    from .classroom import active_session

    running = active_session(db)
    if running is not None and teacher.role != "admin" and running.started_by != teacher.id:
        raise HTTPException(403, "The wall belongs to the class that is running, and that is not yours")
    if body.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {sorted(MODES)}")
    if body.mode == "run":
        rid = body.payload.get("run_id")
        if not rid or db.get(Run, int(rid)) is None:
            raise HTTPException(400, "payload.run_id must point at an existing run")
    d.mode, d.payload, d.updated_at = body.mode, body.payload, datetime.utcnow()
    if body.name is not None:
        d.name = body.name
    db.commit()
    bus.publish_event({"type": "display", "id": n})
    return d
