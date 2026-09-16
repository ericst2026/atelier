from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import board as svc
from ..auth import current_user, is_staff, optional_user
from ..bus import SyncBus
from ..config import settings
from ..db import get_db
from ..deps import get_bus, get_registry
from ..models import User
from ..registry import Registry

router = APIRouter(prefix="/board", tags=["board"])


def _public_or_user(user: Optional[User]) -> None:
    if user is None and not settings.displays_public:
        raise HTTPException(401, "Sign in to continue")


@router.get("/live")
def live(user: Optional[User] = Depends(optional_user), db: Session = Depends(get_db), bus: SyncBus = Depends(get_bus)):
    _public_or_user(user)
    return svc.live(db, bus.worker_state())


@router.get("/progress")
def progress(user: Optional[User] = Depends(optional_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    _public_or_user(user)
    return svc.progress(db, registry)


@router.get("/leaderboard")
def leaderboard(experiment: Optional[str] = None, metric: Optional[str] = None, user: Optional[User] = Depends(optional_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    _public_or_user(user)
    include_unpublished = is_staff(user)
    return svc.leaderboard(db, registry, experiment, metric, include_unpublished=include_unpublished)


@router.get("/run/{run_id}")
def run_view(run_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    from .. import services

    services.visible_run(db, run_id, user)
    return svc.run_view(db, run_id)
