"""Redis glue shared by the API and the worker.

    list   atelier:jobs            queue of run ids (LPUSH by API, LPOP by worker)
    set    atelier:cancel          run ids the user asked to cancel
    pubsub atelier:run:<id>        log / progress / status messages for one run
    pubsub atelier:events          coarse events (run status changes, display edits) → boards refresh
    key    atelier:worker:<node>   json heartbeat, one per GPU node, expiring
    key    atelier:lock:<name>      short-lived lock, so only one node does a
                                    cluster-wide job such as requeuing orphans
    key    atelier:gpu:snapshot    the worker's NVML sample, so an API on another
                                   machine can still draw the hardware strip
"""
import json
from typing import Any, Optional

import redis
import redis.asyncio as aredis

from .config import settings

JOBS = "atelier:jobs"
CANCEL = "atelier:cancel"
EVENTS = "atelier:events"
WORKER = "atelier:worker"          # prefix; the key is atelier:worker:<node>


def run_channel(run_id: int) -> str:
    return f"atelier:run:{run_id}"


def cluster_capacity(bus: "SyncBus") -> dict[str, int]:
    """How many GPUs the cluster actually has, according to the workers themselves.

    Every worker detects its own count and reports it in its heartbeat, and this reads
    them. Until one has checked in, the fallback is whatever GPUs this process can see
    itself: the real count on a single machine, 0 on an API with no GPUs.
    """
    try:
        state = bus.worker_state()
    except Exception:
        state = None
    if not state or not state.get("gpu_count"):
        return {"total": settings.gpu_count, "largest_node": settings.gpu_count, "nodes": 0, "source": "local"}
    return {"total": int(state["gpu_count"]), "largest_node": int(state.get("largest_node") or state["gpu_count"]), "nodes": len(state.get("nodes") or []), "source": "workers"}


def merge_worker_states(states: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Fold per-node heartbeats into one. GPU indices repeat across machines, so the
    flat lists use "node:index" when there is more than one node and plain integers
    when there is one — which keeps every existing reader working."""
    if not states:
        return None
    multi = len(states) > 1
    free, running, pending = [], {}, []
    for st in states:
        node = st.get("node") or "node"
        free += [f"{node}:{g}" if multi else g for g in st.get("free_gpus") or []]
        for rid, ids in (st.get("running") or {}).items():
            running[rid] = [f"{node}:{g}" if multi else g for g in ids]
        pending += st.get("pending") or []
    counts = [int(st.get("gpu_count") or 0) for st in states]
    return {
        "gpu_count": sum(counts),
        # a run cannot span machines, so the largest single node is the ceiling for
        # one job even when the cluster adds up to more
        "largest_node": max(counts) if counts else 0,
        "free_gpus": free,
        "running": running,
        "pending": sorted(set(pending)),
        "backend": states[0].get("backend"),
        "storage_mode": states[0].get("storage_mode"),
        "node": ", ".join(st.get("node") or "node" for st in states),
        "nodes": states,
        "at": max((st.get("at") or "") for st in states),
    }


class SyncBus:
    def __init__(self, url: Optional[str] = None) -> None:
        self.r = redis.from_url(url or settings.redis_url, decode_responses=True)

    def enqueue(self, run_id: int) -> None:
        self.r.rpush(JOBS, str(run_id))

    def pop(self) -> Optional[int]:
        v = self.r.lpop(JOBS)
        return int(v) if v is not None else None

    def publish_run(self, run_id: int, message: dict[str, Any]) -> None:
        self.r.publish(run_channel(run_id), json.dumps(message, default=str))

    def publish_event(self, message: dict[str, Any]) -> None:
        self.r.publish(EVENTS, json.dumps(message, default=str))

    def request_cancel(self, run_id: int) -> None:
        self.r.sadd(CANCEL, str(run_id))

    def is_cancelled(self, run_id: int) -> bool:
        return bool(self.r.sismember(CANCEL, str(run_id)))

    def clear_cancel(self, run_id: int) -> None:
        self.r.srem(CANCEL, str(run_id))

    def set_json(self, key: str, value: Any, ttl: int = 15) -> None:
        self.r.set(f"atelier:{key}", json.dumps(value, default=str), ex=ttl)

    def get_json(self, key: str) -> Optional[Any]:
        raw = self.r.get(f"atelier:{key}")
        return json.loads(raw) if raw else None

    def set_worker_state(self, state: dict[str, Any]) -> None:
        node = state.get("node") or "node"
        self.r.set(f"{WORKER}:{node}", json.dumps(state, default=str), ex=15)

    def worker_states(self) -> list[dict[str, Any]]:
        """Every node that has sent a heartbeat in the last fifteen seconds."""
        out = []
        for key in self.r.scan_iter(f"{WORKER}:*"):
            raw = self.r.get(key)
            if raw:
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    pass
        return sorted(out, key=lambda s: s.get("node") or "")

    def worker_state(self) -> Optional[dict[str, Any]]:
        """One dict for the whole cluster, in the shape a single worker used to
        return, so a one-node install sees no difference."""
        return merge_worker_states(self.worker_states())

    def queued_ids(self) -> list[int]:
        return [int(v) for v in self.r.lrange(JOBS, 0, -1)]

    def claim_lock(self, name: str, holder: str, ttl: int = 60) -> bool:
        """True for exactly one caller until the lock expires."""
        return bool(self.r.set(f"atelier:lock:{name}", holder, nx=True, ex=ttl))


class AsyncBus:
    def __init__(self, url: Optional[str] = None) -> None:
        self.r = aredis.from_url(url or settings.redis_url, decode_responses=True)

    async def enqueue(self, run_id: int) -> None:
        await self.r.rpush(JOBS, str(run_id))

    async def request_cancel(self, run_id: int) -> None:
        await self.r.sadd(CANCEL, str(run_id))

    async def publish_event(self, message: dict[str, Any]) -> None:
        await self.r.publish(EVENTS, json.dumps(message, default=str))

    async def worker_states(self) -> list[dict[str, Any]]:
        out = []
        async for key in self.r.scan_iter(f"{WORKER}:*"):
            raw = await self.r.get(key)
            if raw:
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    pass
        return sorted(out, key=lambda s: s.get("node") or "")

    async def worker_state(self) -> Optional[dict[str, Any]]:
        return merge_worker_states(await self.worker_states())

    async def queue_ids(self) -> list[int]:
        return [int(x) for x in await self.r.lrange(JOBS, 0, -1)]

    def pubsub(self):
        return self.r.pubsub()
