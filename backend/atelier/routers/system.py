import platform

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import __version__, metrics
from ..auth import current_user, require_teacher
from ..bus import SyncBus, cluster_capacity
from .. import materials
from ..config import settings
from ..db import get_db
from ..deps import get_bus, get_registry
from ..models import Run, User
from ..registry import Registry

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health")
def health(bus: SyncBus = Depends(get_bus)):
    redis_ok = True
    try:
        bus.r.ping()
    except Exception:
        redis_ok = False
    worker = bus.worker_state() if redis_ok else None
    nodes = bus.worker_states() if redis_ok else []
    return {"ok": redis_ok, "redis": redis_ok, "worker": worker, "nodes": nodes is not None, "version": __version__}


@router.get("/gpus")
def gpus(user: User = Depends(current_user)):
    return metrics.gpu_snapshot()


@router.get("/info")
def info(user: User = Depends(current_user), registry: Registry = Depends(get_registry)):
    return {
        "version": __version__,
        "python": platform.python_version(),
        "gpu_count": cluster_capacity(bus)["total"],
        "gpu_capacity": cluster_capacity(bus),
        "runner_backend": settings.runner_backend,
        "display_count": settings.display_count,
        "limits": {"max_running_per_student": settings.max_running_per_student, "max_gpus_per_student_run": settings.max_gpus_per_student_run},
        "experiments": [s.slug for s in registry.list()],
        "registry_errors": registry.errors if user.role == "teacher" else {},
    }


@router.get("/queue", dependencies=[Depends(require_teacher)])
def queue(db: Session = Depends(get_db), bus: SyncBus = Depends(get_bus)):
    runs = db.scalars(select(Run).where(Run.status.in_(["queued", "running"])).order_by(Run.id)).all()
    return {"worker": bus.worker_state(), "materials": materials.node_inventories(bus), "runs": [{"id": r.id, "status": r.status, "user_id": r.user_id, "experiment": r.experiment, "kind": r.kind, "step": r.step, "gpus": r.gpus, "gpu_ids": r.gpu_ids} for r in runs]}
