"""Endpoints the worker calls when it is on a different machine from the API.

Only used when ATELIER_STORAGE_MODE=upload. Authentication is a shared token
rather than a user JWT, because the caller is a machine, not a person."""
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from sqlalchemy.orm import Session

from .. import storage
from ..config import settings
from ..db import get_db
from ..models import Run

router = APIRouter(tags=["internal"], include_in_schema=False)


def _authorise(token: Optional[str]) -> None:
    if not settings.worker_token:
        raise HTTPException(503, "This API is not configured to accept worker uploads (set ATELIER_WORKER_TOKEN)")
    if token != settings.worker_token:
        raise HTTPException(401, "Bad worker token")


@router.get("/internal/ping")
def ping(x_atelier_token: Optional[str] = Header(None)) -> dict:
    """The worker calls this at startup so a misconfiguration is visible immediately."""
    _authorise(x_atelier_token)
    return {"ok": True, "storage_mode": settings.storage_mode, "role": settings.role}


@router.post("/internal/runs/{run_id}/file")
async def upload_file(
    run_id: int,
    request: Request,
    path: str = Query(..., description="path relative to the run directory"),
    append: bool = Query(False, description="append rather than replace, used for the log"),
    x_atelier_token: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Raw body upload. Deliberately not multipart: the worker uses urllib from the
    standard library, and this keeps that to a dozen lines."""
    _authorise(x_atelier_token)
    if db.get(Run, run_id) is None:
        raise HTTPException(404, "No such run")
    body = await request.body()
    limit = settings.upload_max_mb * 1024 * 1024
    if len(body) > limit:
        raise HTTPException(413, f"File larger than ATELIER_UPLOAD_MAX_MB ({settings.upload_max_mb} MB)")
    target = storage.safe_join(storage.run_dir(run_id), path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "ab" if append else "wb") as fh:
        fh.write(body)
    return {"ok": True, "bytes": len(body), "path": path, "appended": append}


@router.get("/internal/fingerprint")
def fingerprint(x_atelier_token: Optional[str] = Header(None)) -> dict:
    """A hash of the experiments directory. The worker compares it with its own copy
    and complains loudly if they differ, because a mismatch there produces failures
    that look like anything except the real cause."""
    _authorise(x_atelier_token)
    return {"experiments": storage.sha256_dir(settings.experiments_dir)[:16], "node": settings.node_name}
