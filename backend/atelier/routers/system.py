import platform

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import __version__, metrics
from ..auth import current_user, is_staff, require_teacher
from ..bus import SyncBus, cluster_capacity
from .. import materials
from ..config import settings
from ..db import get_db
from ..deps import get_bus, get_registry
from ..models import Run, User
from ..registry import Registry

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/materials/catalog")
def materials_catalog(kind: str = "model", formats: str = "", schemas: str = "", experiment: str = "", step: int = 0, param: str = "", refresh: bool = False, user: User = Depends(current_user), bus: SyncBus = Depends(get_bus), registry: Registry = Depends(get_registry)):
    """Prepared models or datasets under materials/, for a `material` step param.
    formats / schemas are comma-separated filters (atelier,hf / documents,qa,pairs).

    With experiment, step and param the list is that param's declared choices only:
    each one found on some machine, or listed with a problem when it is not there or
    does not have the format or fields the step reads."""
    cat = materials.catalog(bus, force=refresh and is_staff(user))
    if experiment:
        try:
            p = next(x for x in registry.get(experiment).step(step).params if x.get("key") == param and x.get("type") == "material")
        except Exception:
            raise HTTPException(404, "No such material param")
        kind, formats, schemas = p["kind"], ",".join(p.get("formats") or []), ",".join(p.get("schemas") or [])
    if kind == "dataset":
        want = {s for s in schemas.split(",") if s}
        fits = lambda d: not want or bool(want & set(d["schemas"]))
        found = cat["datasets"]
    else:
        want = {f for f in formats.split(",") if f}
        fits = lambda m: not want or m["format"] in want
        found = cat["models"]
    if not experiment:
        return {"items": [x for x in found if fits(x)]}
    by_path = {x["path"]: x for x in found}
    items = []
    for m in materials.choices(registry.get(experiment), p):
        rel = materials._clean(m["path"])
        if rel in by_path:
            item = dict(by_path[rel])
        else:
            there = materials.status(rel, bus)["available"]
            item = {"path": rel, "name": rel.rsplit("/", 1)[-1], "problem": "not in a form the steps read" if there else "not on any machine"}
        item.update({k: m[k] for k in ("name", "description") if m.get(k)})
        if rel in by_path and not fits(by_path[rel]):
            need = " or ".join(want)
            item["problem"] = f"not {need}" if kind == "model" else f"no {need} fields"
        items.append(item)
    return {"items": items}


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
        "registry_errors": registry.errors if is_staff(user) else {},
    }


@router.get("/queue", dependencies=[Depends(require_teacher)])
def queue(db: Session = Depends(get_db), bus: SyncBus = Depends(get_bus)):
    runs = db.scalars(select(Run).where(Run.status.in_(["queued", "running"])).order_by(Run.id)).all()
    return {"worker": bus.worker_state(), "materials": materials.node_inventories(bus), "runs": [{"id": r.id, "status": r.status, "user_id": r.user_id, "experiment": r.experiment, "kind": r.kind, "step": r.step, "gpus": r.gpus, "gpu_ids": r.gpu_ids} for r in runs]}
