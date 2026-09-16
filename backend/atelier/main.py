"""FastAPI application. Run with:  uvicorn atelier.main:app --host 0.0.0.0 --port 8000"""
import contextlib
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from . import __version__, deps, metrics
from .auth import hash_password
from .board import ensure_displays
from .config import settings
from .db import SessionLocal, init_db
from .events import Hub
from .models import User
from .routers import auth, board, classroom, displays, experiments, internal, runs, submissions, system, users, workspaces, ws

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("atelier")


def bootstrap() -> None:
    settings.ensure_dirs()
    init_db()
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.role == "admin")) is None:
            # an install from before admins existed already has this account as a
            # teacher: promote it rather than colliding with its username
            existing = db.scalar(select(User).where(User.username == settings.admin_username))
            if existing is not None:
                existing.role = "admin"
                log.info("promoted %r to admin", existing.username)
            else:
                db.add(User(username=settings.admin_username, name="Admin", role="admin", password_hash=hash_password(settings.admin_password)))
                log.info("created the admin account %r", settings.admin_username)
            db.commit()
        ensure_displays(db)
    deps.registry.reload(force=True)
    for name, err in deps.registry.errors.items():
        log.error("experiment %s could not be loaded: %s", name, err)
    log.info("loaded experiments: %s", [s.slug for s in deps.registry.list()])


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap()
    metrics.start_sampler()
    deps.hub = Hub(deps.async_bus, deps.registry)
    await deps.hub.start()
    yield
    await deps.hub.stop()


app = FastAPI(title="Atelier", version=__version__, lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

for r in (auth.router, users.router, experiments.router, runs.router, workspaces.router, submissions.router, displays.router, board.router, classroom.router, system.router, internal.router):
    app.include_router(r, prefix="/api")
app.include_router(ws.router)
app.mount("/metrics", metrics.metrics_app)
