"""Websocket endpoints.

    /ws/runs/{id}?token=      log lines, progress, status for one run (owner or teacher)
    /ws/displays/{n}          board snapshots for a wall display (public by default)
    /ws/teacher?token=        live board + every run/submission event (teachers)
"""
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import board, storage
from ..auth import user_from_token
from ..bus import run_channel
from ..config import settings
from ..db import SessionLocal
from ..deps import get_hub, get_registry, get_bus
from ..models import Run

router = APIRouter()


async def _keepalive(ws: WebSocket) -> None:
    """Consume client pings until the socket closes."""
    while True:
        await ws.receive_text()


@router.websocket("/ws/runs/{run_id}")
async def ws_run(ws: WebSocket, run_id: int, token: str = ""):
    hub = get_hub()
    with SessionLocal() as db:
        user = user_from_token(db, token)
        run = db.get(Run, run_id)
        if user is None or run is None or (user.role != "teacher" and run.user_id != user.id):
            await ws.close(code=4403)
            return
        snapshot = {"type": "snapshot", "status": run.status, "progress_pct": run.progress_pct, "progress_msg": run.progress_msg, "lines": storage.last_lines(storage.run_dir(run.id) / "run.log", settings.log_tail_lines), "live": storage.read_json(storage.run_dir(run.id) / "live.json", {"series": {}})}
    await ws.accept()
    channel = run_channel(run_id)
    hub.subscribe_run(channel, ws)
    try:
        await ws.send_text(json.dumps(snapshot, default=str))
        await _keepalive(ws)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe_run(channel, ws)


@router.websocket("/ws/displays/{n}")
async def ws_display(ws: WebSocket, n: int, token: str = ""):
    hub = get_hub()
    if not settings.displays_public:
        with SessionLocal() as db:
            if user_from_token(db, token) is None:
                await ws.close(code=4403)
                return
    await ws.accept()
    hub.subscribe_display(n, ws)
    try:
        with SessionLocal() as db:
            states = board.display_states(db, get_registry(), get_bus().worker_state())
        state = states.get(str(n))
        await ws.send_text(json.dumps({"type": "display", "display": state} if state else {"type": "error", "error": "No such display"}, default=str))
        await _keepalive(ws)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe_display(n, ws)


@router.websocket("/ws/teacher")
async def ws_teacher(ws: WebSocket, token: str = ""):
    hub = get_hub()
    with SessionLocal() as db:
        user = user_from_token(db, token)
        if user is None or user.role != "teacher":
            await ws.close(code=4403)
            return
        live = board.live(db, get_bus().worker_state())
    await ws.accept()
    hub.teacher_subs.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "live", "live": live}, default=str))
        await _keepalive(ws)
    except WebSocketDisconnect:
        pass
    finally:
        hub.teacher_subs.discard(ws)
