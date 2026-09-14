"""Websocket hub. Fans redis pub/sub messages out to browser sockets and pushes
board snapshots to the wall displays on a timer and on every run event."""
import asyncio
import contextlib
import json
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket

from . import board
from .bus import EVENTS, AsyncBus
from .config import settings
from .db import SessionLocal
from .registry import Registry

log = logging.getLogger("atelier.events")


class Hub:
    def __init__(self, bus: AsyncBus, registry: Registry) -> None:
        self.bus = bus
        self.registry = registry
        self.run_subs: dict[str, set[WebSocket]] = defaultdict(set)
        self.display_subs: dict[int, set[WebSocket]] = defaultdict(set)
        self.teacher_subs: set[WebSocket] = set()
        self._refresh = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._listen()), asyncio.create_task(self._board_loop())]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

    # --- subscriptions ----------------------------------------------------
    def subscribe_run(self, channel: str, ws: WebSocket) -> None:
        self.run_subs[channel].add(ws)

    def unsubscribe_run(self, channel: str, ws: WebSocket) -> None:
        self.run_subs[channel].discard(ws)

    def subscribe_display(self, n: int, ws: WebSocket) -> None:
        self.display_subs[n].add(ws)

    def unsubscribe_display(self, n: int, ws: WebSocket) -> None:
        self.display_subs[n].discard(ws)

    def request_refresh(self) -> None:
        self._refresh.set()

    # --- internals --------------------------------------------------------
    async def _listen(self) -> None:
        while True:
            try:
                ps = self.bus.pubsub()
                await ps.psubscribe("atelier:*")
                async for msg in ps.listen():
                    if msg.get("type") != "pmessage":
                        continue
                    channel = msg["channel"]
                    if channel == EVENTS:
                        self._refresh.set()
                        await self._broadcast(self.teacher_subs, msg["data"])
                        continue
                    subs = self.run_subs.get(channel)
                    if subs:
                        await self._broadcast(subs, msg["data"])
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # redis restart etc.
                log.warning("pubsub listener restarting: %s", exc)
                await asyncio.sleep(2)

    async def _broadcast(self, sockets: set[WebSocket], text: str) -> None:
        dead = []
        for ws in list(sockets):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            sockets.discard(ws)

    def _compute(self) -> dict[str, Any]:
        with SessionLocal() as db:
            return board.display_states(db, self.registry)

    async def push_boards(self) -> None:
        if not any(self.display_subs.values()) and not self.teacher_subs:
            return
        try:
            worker_state = await self.bus.worker_state()
            states = await asyncio.to_thread(self._compute_with_worker, worker_state)
        except Exception as exc:
            log.warning("board computation failed: %s", exc)
            return
        for n, sockets in list(self.display_subs.items()):
            state = states.get(str(n))
            if state and sockets:
                await self._broadcast(sockets, json.dumps({"type": "display", "display": state}, default=str))
        if self.teacher_subs:
            with SessionLocal() as db:
                live = await asyncio.to_thread(board.live, db, worker_state)
            await self._broadcast(self.teacher_subs, json.dumps({"type": "live", "live": live}, default=str))

    def _compute_with_worker(self, worker_state: Any) -> dict[str, Any]:
        with SessionLocal() as db:
            return board.display_states(db, self.registry, worker_state)

    async def _board_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._refresh.wait(), timeout=settings.board_interval_sec)
                await asyncio.sleep(0.8)  # debounce bursts of events
            except asyncio.TimeoutError:
                pass
            self._refresh.clear()
            await self.push_boards()
