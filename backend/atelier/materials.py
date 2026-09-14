"""Which machine has which materials.

On one machine this is a question about the local disk. With the GPUs on another
machine it is not: the materials are tens of gigabytes, they live on the nodes that
train with them, and the API has none of them. So each worker publishes what it has
and the API reads that.
"""
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import storage
from .config import settings

log = logging.getLogger("atelier.materials")
_cache: dict[str, Any] = {"at": 0.0, "data": {}}
_lock = threading.Lock()
SCAN_TTL = 300.0


def _stat(rel: str) -> Optional[dict[str, Any]]:
    p = storage.material_path(rel)
    if p is None:
        return None
    if p.is_dir():
        st = storage.dir_stats(p)
        return {"files": st.get("files", 0), "bytes": st.get("bytes", 0), "dir": True}
    return {"files": 1, "bytes": p.stat().st_size, "dir": False}


def declared_paths(registry) -> list[str]:
    """Every material path any experiment mentions, deduplicated."""
    seen = []
    for spec in registry.list():
        for m in spec.materials or []:
            rel = (m.get("path") or "").strip()
            if rel and rel not in seen:
                seen.append(rel)
    return seen


def local_inventory(registry, force: bool = False) -> dict[str, Any]:
    """What this machine has. Cached, because walking a materials tree of tens of
    thousands of files every few seconds would be rude to the disk."""
    with _lock:
        if not force and time.time() - _cache["at"] < SCAN_TTL and _cache["data"]:
            return _cache["data"]
    data = {}
    for rel in declared_paths(registry):
        st = _stat(rel)
        if st:
            data[rel] = st
    with _lock:
        _cache.update({"at": time.time(), "data": data})
    return data


def publish(bus, registry, force: bool = False) -> None:
    """Called by the worker. TTL is generous: a node that goes quiet keeps its entry
    briefly rather than making every material blink out of existence."""
    try:
        bus.set_json(f"materials:{settings.node_name}", {"node": settings.node_name, "root": str(settings.materials_dir), "items": local_inventory(registry, force), "at": time.time()}, ttl=int(SCAN_TTL * 2))
    except Exception:
        log.debug("could not publish the materials inventory", exc_info=True)


def node_inventories(bus) -> list[dict[str, Any]]:
    out = []
    try:
        for key in bus.r.scan_iter("atelier:materials:*"):
            snap = bus.get_json(key.split("atelier:", 1)[-1])
            if snap:
                out.append(snap)
    except Exception:
        return []
    return sorted(out, key=lambda s: s.get("node") or "")


def status(rel: str, bus=None, registry=None) -> dict[str, Any]:
    """Where a material can be found, and how big it is.

    local  this machine has it, so it can also be browsed and downloaded
    nodes  the GPU nodes that have it — enough to know a run will work
    """
    local = _stat(rel)
    nodes, stats = [], local
    if bus is not None:
        for snap in node_inventories(bus):
            item = (snap.get("items") or {}).get(rel)
            if item:
                nodes.append(snap.get("node"))
                stats = stats or item
    return {
        "available": bool(local) or bool(nodes),
        "local": bool(local),
        "nodes": nodes,
        "stats": stats,
    }


def missing_for(spec, bus=None) -> list[str]:
    """Material paths no machine reports having. Used to warn a teacher before a
    lesson rather than after a student's run fails."""
    out = []
    for m in spec.materials or []:
        rel = (m.get("path") or "").strip()
        if rel and not status(rel, bus)["available"]:
            out.append(rel)
    return out


def missing_locally(spec) -> list[str]:
    """What this machine lacks. A worker uses it to decide whether it can take a job."""
    return [rel for m in (spec.materials or []) if (rel := (m.get("path") or "").strip()) and storage.material_path(rel) is None]
