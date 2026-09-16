"""The class: one experiment, one teacher, the whole room.

Only one class runs at a time in the building. A teacher starts it, for one of the
experiments an admin has granted them, and while it runs it is the class — another
teacher has to wait for it to end. The wall follows it: displays 1-4 show its four
steps, and each display can be pointed somewhere else by the teacher.

Students never pick the experiment. They see the class that is running, ask to join,
and the teacher lets them in; that is then what they can run. Working alone is
separate: an admin grants a person a list of experiments, and those they may run
whenever they like.
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
STEP_DISPLAYS = 4  # displays 1-4 follow the four steps of the class


def active_session(db: Session) -> Optional[ClassSession]:
    """The one class running now, whoever is teaching it. A paused class is not it:
    it has given the room up until someone resumes it."""
    return db.scalar(select(ClassSession).where(ClassSession.ended_at.is_(None), ClassSession.paused_at.is_(None)).order_by(ClassSession.id.desc()))


def paused_sessions(db: Session, teacher_id: Optional[int] = None) -> list[ClassSession]:
    """Classes that are stopped but not finished, newest first."""
    q = select(ClassSession).where(ClassSession.ended_at.is_(None), ClassSession.paused_at.is_not(None)).order_by(ClassSession.id.desc())
    if teacher_id is not None:
        q = q.where(ClassSession.started_by == teacher_id)
    return list(db.scalars(q).all())


def membership(db: Session, session_id: int, user_id: int) -> Optional[SessionMember]:
    return db.scalar(select(SessionMember).where(SessionMember.session_id == session_id, SessionMember.user_id == user_id))


def self_allowed(db: Session, user_id: int) -> list[str]:
    """The experiments this person may run on their own, as granted by an admin."""
    return list(db.scalars(select(SelfPermission.experiment).where(SelfPermission.user_id == user_id)).all())


def visible_experiments(db: Session, user: User) -> set[str]:
    """What this person may see: what an admin granted them, plus the class that is
    running. An admin sees everything, so callers skip this for them."""
    out = set(self_allowed(db, user.id))
    s = active_session(db)
    if s is not None:
        out.add(s.experiment)
    return out


def may_run(db: Session, user: User, experiment: str) -> tuple[bool, str]:
    """Whether this person may run a step of this experiment right now."""
    if user.role == "admin":
        return True, ""
    if experiment in self_allowed(db, user.id):
        return True, ""
    s = active_session(db)
    if s is None:
        return False, "No class is running, and you have not been allowed to run this on your own. A teacher starts the class; an admin grants working alone."
    if s.experiment != experiment:
        return False, f"The class is running {s.experiment}. That is the experiment to work on now."
    if user.role == "teacher" and s.started_by == user.id:
        return True, ""
    m = membership(db, s.id, user.id)
    if m is not None and m.admitted:
        return True, ""
    return False, "Ask your teacher to let you into the class first."


def _session_dict(db: Session, s: ClassSession, user: Optional[User], registry: Registry) -> dict:
    members = db.scalars(select(SessionMember).where(SessionMember.session_id == s.id)).all()
    people = {u.id: u for u in db.scalars(select(User).where(User.id.in_([m.user_id for m in members] + [s.started_by]))).all()}
    title = ""
    try:
        title = registry.get(s.experiment).title
    except KeyError:
        pass
    teacher = people.get(s.started_by)
    me = None
    if user is not None:
        m = membership(db, s.id, user.id)
        me = {"asked": m is not None, "admitted": bool(m and m.admitted), "mine": s.started_by == user.id}
    return {
        "id": s.id,
        "experiment": s.experiment,
        "name": s.name or "",
        "paused": s.paused_at is not None,
        "paused_at": s.paused_at.isoformat() if s.paused_at else None,
        "params": s.params or {},
        "title": title,
        "teacher": (teacher.name or teacher.username) if teacher else "?",
        "teacher_id": s.started_by,
        "started_at": s.started_at.isoformat(),
        "ended_at": s.ended_at.isoformat() if s.ended_at else None,
        "members": [
            {"user_id": m.user_id, "username": people[m.user_id].username if m.user_id in people else "?", "name": people[m.user_id].name if m.user_id in people else "", "admitted": m.admitted}
            for m in members
        ],
        "me": me,
    }


def _state(db: Session, user: User, registry: Registry) -> dict:
    s = active_session(db)
    mine_paused = paused_sessions(db, None if user.role == "admin" else user.id) if user.role in ("teacher", "admin") else []
    return {
        "running": s is not None,
        "session": _session_dict(db, s, user, registry) if s is not None else None,
        "paused": [_session_dict(db, x, user, registry) for x in mine_paused],
    }


@router.get("")
def get_class(user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    """The class running now, and where the caller stands in it."""
    return _state(db, user, registry)


@router.get("/history")
def history(limit: int = 50, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    """Classes that have been held: the ones a teacher ran, the ones a student took
    part in, and for an admin, every class there has been."""
    q = select(ClassSession).order_by(ClassSession.id.desc()).limit(min(limit, 200))
    if user.role == "teacher":
        q = q.where(ClassSession.started_by == user.id)
    elif user.role != "admin":
        attended = select(SessionMember.session_id).where(SessionMember.user_id == user.id, SessionMember.admitted.is_(True))
        q = q.where(ClassSession.id.in_(attended))
    return {"sessions": [_session_dict(db, s, user, registry) for s in db.scalars(q).all()], "all_teachers": user.role == "admin"}


@router.post("")
def start_class(body: dict, teacher: User = Depends(require_teacher), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    """Start the class. One runs at a time in the building, so another teacher's
    class has to end first, and it has to be an experiment an admin granted you."""
    slug = str(body.get("experiment") or "").strip()
    try:
        registry.get(slug)
    except KeyError:
        raise HTTPException(404, "Unknown experiment")
    if teacher.role != "admin" and slug not in self_allowed(db, teacher.id):
        raise HTTPException(403, "An admin has not given you this experiment to teach")
    current = active_session(db)
    if current is not None:
        # no switching: a class is stopped or paused deliberately, never replaced by
        # starting the next one
        who = db.get(User, current.started_by)
        whose = "Your class" if current.started_by == teacher.id else f"{(who.name or who.username) if who else 'Another teacher'}'s class"
        raise HTTPException(409, f"{whose} ({current.name or current.experiment}) is running. Pause or end it first.")
    name = str(body.get("name") or "").strip()[:120]
    s = ClassSession(experiment=slug, name=name, started_by=teacher.id, params=dict(body.get("params") or {}))
    db.add(s)
    db.commit()
    # the wall follows the class: one display per step, and any display showing a
    # student from the old class goes back to its step
    for d in db.scalars(select(Display).order_by(Display.id)).all():
        if d.id <= STEP_DISPLAYS:
            d.mode, d.payload, d.updated_at = "step", {"session_id": s.id, "experiment": slug, "step": d.id}, datetime.utcnow()
            bus.publish_event({"type": "display", "id": d.id})
        elif d.mode not in ("grafana", "leaderboard"):
            d.mode, d.payload, d.updated_at = "grafana", {}, datetime.utcnow()
            bus.publish_event({"type": "display", "id": d.id})
    db.commit()
    bus.publish_event({"type": "class", "session": s.id, "running": True})
    return _state(db, teacher, registry)


@router.delete("")
def stop_class(teacher: User = Depends(require_teacher), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    s = active_session(db)
    if s is None:
        raise HTTPException(400, "No class is running")
    if teacher.role != "admin" and s.started_by != teacher.id:
        raise HTTPException(403, "That is another teacher's class")
    s.ended_at = datetime.utcnow()
    db.commit()
    bus.publish_event({"type": "class", "session": s.id, "running": False})
    for d in db.scalars(select(Display)).all():
        bus.publish_event({"type": "display", "id": d.id})  # the wall now says no class is running
    return _state(db, teacher, registry)


@router.post("/pause")
def pause_class(teacher: User = Depends(require_teacher), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    """Put the class down without ending it: the room is free for another teacher,
    and this one keeps its roster and its settings until it is resumed."""
    s = active_session(db)
    if s is None:
        raise HTTPException(400, "No class is running")
    if teacher.role != "admin" and s.started_by != teacher.id:
        raise HTTPException(403, "That is another teacher's class")
    s.paused_at = datetime.utcnow()
    db.commit()
    bus.publish_event({"type": "class", "session": s.id, "paused": True})
    for d in db.scalars(select(Display)).all():
        bus.publish_event({"type": "display", "id": d.id})
    return _state(db, teacher, registry)


@router.post("/{session_id}/resume")
def resume_class(session_id: int, teacher: User = Depends(require_teacher), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    """Pick a paused class back up, if the room is free."""
    s = db.get(ClassSession, session_id)
    if s is None or s.ended_at is not None or s.paused_at is None:
        raise HTTPException(404, "That class is not paused")
    if teacher.role != "admin" and s.started_by != teacher.id:
        raise HTTPException(403, "That is another teacher's class")
    current = active_session(db)
    if current is not None:
        who = db.get(User, current.started_by)
        raise HTTPException(409, f"{(who.name or who.username) if who else 'Another teacher'} is running a class ({current.experiment}). It has to stop before this one can carry on.")
    s.paused_at = None
    db.commit()
    # the wall picks the class back up where it was
    for d in db.scalars(select(Display).order_by(Display.id)).all():
        if d.id <= STEP_DISPLAYS:
            d.mode, d.payload, d.updated_at = "step", {"session_id": s.id, "experiment": s.experiment, "step": d.id}, datetime.utcnow()
        bus.publish_event({"type": "display", "id": d.id})
    db.commit()
    bus.publish_event({"type": "class", "session": s.id, "running": True})
    return _state(db, teacher, registry)


@router.post("/join")
def ask_to_join(user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    s = active_session(db)
    if s is None:
        raise HTTPException(400, "No class is running")
    if membership(db, s.id, user.id) is None:
        db.add(SessionMember(session_id=s.id, user_id=user.id, admitted=False))
        db.commit()
        bus.publish_event({"type": "class", "session": s.id, "asked": user.id})
    return _state(db, user, registry)


@router.delete("/members/{user_id}")
def deny(user_id: int, teacher: User = Depends(require_teacher), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    """Turn a request down, or remove someone already in. They can ask again."""
    s = active_session(db)
    if s is None:
        raise HTTPException(400, "No class is running")
    if teacher.role != "admin" and s.started_by != teacher.id:
        raise HTTPException(403, "That is another teacher's class")
    m = membership(db, s.id, user_id)
    if m is not None:
        db.delete(m)
        db.commit()
        bus.publish_event({"type": "class", "session": s.id, "denied": user_id})
    return _state(db, teacher, registry)


@router.post("/members/{user_id}")
def admit(user_id: int, body: dict, teacher: User = Depends(require_teacher), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    """Let a student into the class, or take them back out."""
    s = active_session(db)
    if s is None:
        raise HTTPException(400, "No class is running")
    if teacher.role != "admin" and s.started_by != teacher.id:
        raise HTTPException(403, "That is another teacher's class")
    admitted = bool(body.get("admitted", True))
    m = membership(db, s.id, user_id)
    if m is None:
        db.add(SessionMember(session_id=s.id, user_id=user_id, admitted=admitted))
    else:
        m.admitted = admitted
    db.commit()
    bus.publish_event({"type": "class", "session": s.id, "admitted": user_id})
    return _state(db, teacher, registry)
