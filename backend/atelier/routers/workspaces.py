"""The student's project for one experiment. Lives on the server; edited in the
browser, uploaded as a zip, or synced from the student's own machine."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import services, storage
from ..auth import current_user
from ..bus import SyncBus
from ..config import settings
from ..db import get_db
from ..deps import get_bus, get_registry
from ..models import Run, User
from ..registry import Registry
from ..schemas import FileWrite, RunOut, WorkspaceRunCreate

router = APIRouter(prefix="/workspaces/{slug}", tags=["workspaces"])


def _ctx(slug: str, user: User, registry: Registry, db: Session, user_id: Optional[int]):
    try:
        spec = registry.get(slug)
    except KeyError:
        raise HTTPException(404, "Experiment not found")
    owner = user
    if user_id and user_id != user.id:
        if user.role != "teacher":
            raise HTTPException(403, "Teachers only")
        owner = db.get(User, user_id)
        if owner is None:
            raise HTTPException(404, "User not found")
    root = storage.workspace_dir(owner.id, slug)
    if not root.exists() and owner.id == user.id and spec.sample_dir.exists():
        storage.copy_tree(spec.sample_dir, root)
    root.mkdir(parents=True, exist_ok=True)
    return spec, owner, root


@router.get("")
def info(slug: str, user_id: Optional[int] = None, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    spec, owner, root = _ctx(slug, user, registry, db, user_id)
    return {"owner_id": owner.id, "read_only": owner.id != user.id, "stats": storage.dir_stats(root), "entries": storage.tree(root), "run_command": spec.project.get("run_command", "python project/train.py"), "entry": spec.project.get("entry")}


@router.get("/file")
def read_file(slug: str, path: str = Query(...), user_id: Optional[int] = None, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    _, _, root = _ctx(slug, user, registry, db, user_id)
    try:
        return dict(storage.read_text(storage.safe_join(root, path)), path=path)
    except storage.StorageError as exc:
        raise HTTPException(404, str(exc))


@router.put("/file")
def write_file(slug: str, body: FileWrite, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    _, _, root = _ctx(slug, user, registry, db, None)
    try:
        p = storage.safe_join(root, body.path)
    except storage.StorageError as exc:
        raise HTTPException(400, str(exc))
    if len(body.content.encode("utf-8")) > storage.TEXT_LIMIT:
        raise HTTPException(413, "File is too large to edit in the browser")
    storage.write_text(p, body.content)
    return {"ok": True, "path": body.path, "bytes": p.stat().st_size}


@router.delete("/file")
def delete_file(slug: str, path: str = Query(...), user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    import shutil

    _, _, root = _ctx(slug, user, registry, db, None)
    try:
        p = storage.safe_join(root, path)
    except storage.StorageError as exc:
        raise HTTPException(400, str(exc))
    if p == root.resolve():
        raise HTTPException(400, "Cannot delete the workspace root")
    if p.is_dir():
        shutil.rmtree(p)
    elif p.exists():
        p.unlink()
    return {"ok": True}


@router.post("/reset")
def reset(slug: str, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    spec, _, root = _ctx(slug, user, registry, db, None)
    if not spec.sample_dir.exists():
        raise HTTPException(400, "This experiment has no sample project")
    storage.copy_tree(spec.sample_dir, root)
    return {"ok": True, "entries": storage.tree(root)}


@router.post("/upload")
async def upload(slug: str, file: UploadFile, replace: bool = False, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    import shutil

    _, _, root = _ctx(slug, user, registry, db, None)
    data = await file.read()
    if replace:
        shutil.rmtree(root, ignore_errors=True)
    try:
        total = storage.unzip_to(data, root, settings.workspace_max_mb * 1024 * 1024)
    except storage.StorageError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        raise HTTPException(400, "Upload a .zip archive of your project")
    return {"ok": True, "bytes": total, "entries": storage.tree(root)}


@router.get("/download")
def download(slug: str, user_id: Optional[int] = None, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    _, owner, root = _ctx(slug, user, registry, db, user_id)
    return Response(storage.zip_dir(root), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{slug}-{owner.username}.zip"'})


@router.post("/run", response_model=RunOut, status_code=201)
def run_command(slug: str, body: WorkspaceRunCreate, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    spec, _, root = _ctx(slug, user, registry, db, None)
    run = services.create_command_run(db, bus, user, spec, "workspace", body.command, str(root), body.gpus, body.label or body.command[:80], spec.project.get("timeout_min", settings.default_timeout_min))
    out = RunOut.model_validate(run)
    out.username = user.username
    return out


@router.post("/test", response_model=RunOut, status_code=201)
def test_workspace(slug: str, user: User = Depends(current_user), db: Session = Depends(get_db), registry: Registry = Depends(get_registry), bus: SyncBus = Depends(get_bus)):
    spec, _, root = _ctx(slug, user, registry, db, None)
    grader = spec.grader
    if not grader.get("script"):
        raise HTTPException(400, "This experiment has no grader")
    cmd = f"{settings.python_bin} {spec.dir / grader['script']} --project {{workspace_dir}} --run-dir {{run_dir}}"
    run = services.create_command_run(db, bus, user, spec, "test", cmd, str(root), int(grader.get("gpus", 0)), f"Test · {spec.title}", int(grader.get("timeout_min", 30)))
    out = RunOut.model_validate(run)
    out.username = user.username
    return out


@router.get("/runs", response_model=list[RunOut])
def workspace_runs(slug: str, limit: int = 30, user: User = Depends(current_user), db: Session = Depends(get_db)):
    runs = db.scalars(select(Run).where(Run.user_id == user.id, Run.experiment == slug, Run.kind.in_(["workspace", "test"])).order_by(Run.id.desc()).limit(limit)).all()
    out = []
    for r in runs:
        o = RunOut.model_validate(r)
        o.username = user.username
        out.append(o)
    return out
