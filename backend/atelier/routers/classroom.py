"""The class session: one experiment at a time, driven by the teacher.

A class is one experiment the teacher has started. Students do not choose it —
they ask to join and the teacher admits them, and while the session runs that is
the experiment they can run steps of. A student the teacher has marked as allowed
to experiment on their own is not bound by any of this: they work outside the
classroom, on any experiment, whenever they like.

Starting a session also points the wall displays at it: displays 1-4 follow the
four steps and the last one keeps showing the hardware dashboard.
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import current_user, require_teacher
from ..bus import SyncBus
from ..db import get_db
from ..deps import get_bus, get_registry
from ..models import ClassSession, Display, SelfPermission, SessionMember, User
from ..registry import Registry

router = APIRouter(prefix="/class", tags=["class"])
STEP_DISPLAYS = 4  # displays 1-4 follow the steps; the last one stays on Grafana


def active_session(db: Session) -> Optional[ClassSession]:
    return db.scalar(select(ClassSession).where(ClassSession.ended_at.is_(None)).order_by(ClassSession.id.desc()))


def membership(db: Session, session_id: int, user_id: int) -> Optional[SessionMember]:
    return db.scalar(select(SessionMember).where(SessionMember.session_id == session_id, SessionMember.user_id == user_id))


def self_allowed(db: Session, user_id: int) -> list[str]:
    """The experiments this person may run on their own, as granted by an admin."""
    return list(db.scalars(select(SelfPermission.experiment).where(SelfPermission.user_id == user_id)).all())


def may_run(db: Session, user: User, experiment: str) -> tuple[bool, str]:
    """Whether this person may run a step of this experiment right now: because the
    class is running it and they are in the class, or because an admin has let them
    work on this one by themselves."""
    if user.role == "admin":
        return True, ""
    s = active_session(db)
    in_class = s is not None and s.experiment == experiment
    if in_class and user.role == "teacher":
        return True, ""
    if in_class:
        m = membership(db, s.id, user.id)
        if m is not None and m.admitted:
            return True, ""
    if experiment in self_allowed(db, user.id):
        return True, ""
    if s is None:
        return False, "No class is running, and you have not been allowed to run this experiment on your own. A teacher starts the class; an admin grants working on your own."
    if s.experiment != experiment:
        return False, f"The class is running {s.experiment}. That is the experiment to work on now."
    return False, "Ask your teacher to admit you to the class first."


def _state(db: Session, s: Optional[ClassSession], user: Optional[User], registry: Registry) -> dict:
    if s is None:
        return {"running": False, "experiment": None, "members": [], "me": None}
    members = db.scalars(select(SessionMember).where(SessionMember.session_id == s.id)).all()
    users = {u.id: u for u in db.scalars(select(User).where(User.id.in_([m.user_id for m in members] or [0]))).all()}
    title = ""
    try:
        title = registry.get(s.experiment).title
    except KeyError:
        pass
    me = None
    if user is not None:
        m = membership(db, s.id, user.id)
        me = {"asked": m is not None, "admitted": bool(m and m.admitted)} if m else {"asked": False, "admitted": False}
    return {
        "running": True,
        "id": s.id,
        "experiment": s.experiment,
        "title": title,
        "started_at": s.started_at.isoformat(),
        "members": [
            {"user_id": m.user_id, "username": users.get(m.user_id).username if users.get(m.user_id) else "?", "name": users.get(m.user_id).name if users.get(m.user_id) else "", "admitted": m.admitted}
            for m in members
        ],
        "me": me,
    }


@router.get("")
def get_class(user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    return _state(db, active_session(db), user, registry)


@router.post("")
def start_class(body: dict, teacher: User = Depends(require_teacher), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    """Start an experiment for the class. Only one runs at a time, so an earlier
    one is closed first."""
    slug = str(body.get("experiment") or "").strip()
    try:
        spec = registry.get(slug)
    except KeyError:
        raise HTTPException(404, "Unknown experiment")
    current = active_session(db)
    if current is not None:
        current.ended_at = datetime.utcnow()
    s = ClassSession(experiment=slug, started_by=teacher.id)
    db.add(s)
    db.commit()
    # the wall follows the class: one display per step, and the last one goes back
    # to the hardware dashboard, which is what it shows unless the teacher asks for
    # the leaderboard
    for d in db.scalars(select(Display).order_by(Display.id)).all():
        if d.id <= STEP_DISPLAYS:
            d.mode, d.payload = "step", {"experiment": slug, "step": d.id}
        elif d.mode not in ("grafana", "leaderboard"):
            d.mode, d.payload = "grafana", {}
        d.updated_at = datetime.utcnow()
    db.commit()
    bus.publish_event({"type": "class", "experiment": slug, "running": True})
    for n in range(1, STEP_DISPLAYS + 1):
        bus.publish_event({"type": "display", "id": n})
    return _state(db, s, teacher, registry)


@router.delete("")
def stop_class(teacher: User = Depends(require_teacher), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    s = active_session(db)
    if s is None:
        raise HTTPException(400, "No class is running")
    s.ended_at = datetime.utcnow()
    db.commit()
    bus.publish_event({"type": "class", "running": False})
    return {"running": False}


@router.post("/join")
def ask_to_join(user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    s = active_session(db)
    if s is None:
        raise HTTPException(400, "No class is running")
    if membership(db, s.id, user.id) is None:
        db.add(SessionMember(session_id=s.id, user_id=user.id, admitted=False))
        db.commit()
        bus.publish_event({"type": "class", "asked": user.id})
    return _state(db, s, user, registry)


@router.post("/members/{user_id}")
def admit(user_id: int, body: dict, teacher: User = Depends(require_teacher), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    """Let a student in, or take them back out."""
    s = active_session(db)
    if s is None:
        raise HTTPException(400, "No class is running")
    admitted = bool(body.get("admitted", True))
    m = membership(db, s.id, user_id)
    if m is None:
        m = SessionMember(session_id=s.id, user_id=user_id, admitted=admitted)
        db.add(m)
    else:
        m.admitted = admitted
    db.commit()
    bus.publish_event({"type": "class", "admitted": user_id})
    return _state(db, s, teacher, registry)
