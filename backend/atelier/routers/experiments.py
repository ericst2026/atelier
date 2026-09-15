from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import materials, services, storage
from ..auth import current_user
from ..db import get_db
from ..deps import get_bus, get_registry
from ..models import User
from ..registry import Registry

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _spec(registry: Registry, slug: str):
    try:
        return registry.get(slug)
    except KeyError:
        raise HTTPException(404, "Experiment not found")


@router.get("")
def list_experiments(user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    progress = services.user_progress(db, user, registry)
    return {"experiments": [dict(s.to_dict(), progress=progress.get(s.slug)) for s in registry.list()], "categories": registry.categories(), "errors": registry.errors if user.role == "teacher" else {}}


@router.get("/{slug}")
def get_experiment(slug: str, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    spec = _spec(registry, slug)
    d = spec.to_dict(full=True)
    bus = get_bus()
    for m in d["materials"]:
        st = materials.status(m.get("path", ""), bus)
        # "available" means a run will find it, which on a split install means a GPU
        # node has it — not this machine
        m["available"] = st["available"]
        m["local"] = st["local"]
        m["nodes"] = st["nodes"]
        if st["stats"]:
            m["stats"] = {k: v for k, v in st["stats"].items() if k != "dir"}
    d["materials_missing"] = materials.missing_for(spec, bus)
    d["progress"] = services.user_progress(db, user, registry).get(slug)
    d["readme"] = (spec.dir / "README.md").read_text(encoding="utf-8") if (spec.dir / "README.md").exists() else ""
    return d


@router.get("/{slug}/materials/{path:path}")
def download_material(slug: str, path: str, user: User = Depends(current_user), registry: Registry = Depends(get_registry)):
    spec = _spec(registry, slug)
    allowed = [m.get("path", "") for m in spec.materials]
    if not any(path == a or path.startswith(a.rstrip("/") + "/") for a in allowed):
        raise HTTPException(404, "Not a material of this experiment")
    p = storage.material_path(path)
    if p is None or not p.is_file():
        st = materials.status(path, get_bus())
        if st["nodes"]:
            raise HTTPException(409, f"This material is on {', '.join(st['nodes'])}, not on the machine serving this page. Runs can use it; browsing it here needs a copy on this machine (see docs/topologies.md).")
        raise HTTPException(404, "File not found on the server")
    return FileResponse(p, filename=p.name)


@router.get("/{slug}/materials-tree")
def material_tree(slug: str, path: str = Query(...), user: User = Depends(current_user), registry: Registry = Depends(get_registry)):
    spec = _spec(registry, slug)
    if path not in [m.get("path", "") for m in spec.materials]:
        raise HTTPException(404, "Not a material of this experiment")
    p = storage.material_path(path)
    if p is None:
        st = materials.status(path, get_bus())
        return {"available": st["available"], "local": False, "nodes": st["nodes"], "entries": [],
                "note": f"on {', '.join(st['nodes'])}" if st["nodes"] else "not found on any node"}
    return {"available": True, "entries": storage.tree(p, max_depth=3, max_entries=500) if p.is_dir() else [{"path": p.name, "type": "file", "bytes": p.stat().st_size}]}


@router.get("/{slug}/sample/tree")
def sample_tree(slug: str, user: User = Depends(current_user), registry: Registry = Depends(get_registry)):
    spec = _spec(registry, slug)
    return {"entries": storage.tree(spec.sample_dir)}


@router.get("/{slug}/sample/file")
def sample_file(slug: str, path: str = Query(...), user: User = Depends(current_user), registry: Registry = Depends(get_registry)):
    spec = _spec(registry, slug)
    try:
        p = storage.safe_join(spec.sample_dir, path)
        return dict(storage.read_text(p), path=path)
    except storage.StorageError as exc:
        raise HTTPException(404, str(exc))


@router.get("/{slug}/sample/download")
def sample_download(slug: str, user: User = Depends(current_user), registry: Registry = Depends(get_registry)):
    from fastapi.responses import Response

    spec = _spec(registry, slug)
    return Response(storage.zip_dir(spec.sample_dir), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{slug}-sample.zip"'})
