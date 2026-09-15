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


# --- the catalog: prepared models and datasets a step can pick from ---------------
#
# A step param of type `material` lists what is under materials/models/ or
# materials/datasets/ so a teacher's own model or data is chosen, not typed. The
# field names below must match experiments/_lib/atelier_world/prepared.py, which
# reads the same files inside the run.
TEXT_FIELDS = ("text", "content", "body", "document")
PROMPT_FIELDS = ("prompt", "question", "instruction", "query", "problem", "input")
ANSWER_FIELDS = ("answer", "answers", "answerKey", "output", "response", "target", "completion", "solution")
DATA_SUFFIXES = (".jsonl", ".json")
TEXT_SUFFIXES = (".txt", ".md")
_catalog_cache: dict[str, Any] = {"at": 0.0, "data": None}


def _first_row(path: Path) -> dict[str, Any]:
    import json

    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                return row if isinstance(row, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


def dataset_schemas(keys: set[str], has_text_files: bool) -> list[str]:
    """What a dataset can be used as, from the fields of its first row.

    documents  plain text to pretrain or build a tokenizer on
    qa         a question with a gradable answer (fine-tuning, RL, evaluation)
    pairs      a prompt with a chosen and a rejected answer (preference optimisation)
    """
    out = []
    if has_text_files or keys & set(TEXT_FIELDS):
        out.append("documents")
    if keys & set(PROMPT_FIELDS) and keys & set(ANSWER_FIELDS):
        out.append("qa")
    if keys & set(PROMPT_FIELDS) and {"chosen", "rejected"} <= keys:
        out.append("pairs")
    return out


def _describe_dataset(p: Path, rel: str) -> Optional[dict[str, Any]]:
    files = [f for f in p.rglob("*") if f.is_file() and not any(part.startswith(".") for part in f.relative_to(p).parts)]
    data_files = [f for f in files if f.suffix.lower() in DATA_SUFFIXES]
    text_files = [f for f in files if f.suffix.lower() in TEXT_SUFFIXES and f.name.lower() != "readme.md"]
    if not data_files and not text_files:
        return None
    keys: set[str] = set()
    for f in data_files[:3]:
        keys |= set(_first_row(f))
    schemas = dataset_schemas(keys, bool(text_files))
    if not schemas:
        return None
    return {
        "path": rel,
        "name": rel.split("/", 1)[-1],
        "schemas": schemas,
        "splits": sorted({f.stem for f in data_files}),
        "fields": sorted(keys),
        "bytes": sum(f.stat().st_size for f in files),
    }


def _describe_model(p: Path, rel: str) -> Optional[dict[str, Any]]:
    import json

    names = {f.name for f in p.iterdir() if f.is_file()}
    info: dict[str, Any] = {"path": rel, "name": p.name}
    if "model.pt" in names:
        # an Atelier checkpoint: what the world experiments train and save
        info["format"] = "atelier"
        if "tokenizer.json" not in names:
            info["problem"] = "no tokenizer.json next to model.pt"
    elif "config.json" in names and any(n.endswith((".safetensors", ".bin")) for n in names):
        info["format"] = "hf"
    else:
        return None
    if "meta.json" in names:
        try:
            meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
            info.update({k: meta[k] for k in ("lang", "description") if k in meta})
        except (OSError, ValueError):
            pass
    info["bytes"] = sum(f.stat().st_size for f in p.iterdir() if f.is_file())
    return info


def local_catalog(force: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Every usable model and dataset on this machine. Only the first rows of each
    file are read, and the result is cached like the inventory."""
    with _lock:
        if not force and _catalog_cache["data"] is not None and time.time() - _catalog_cache["at"] < SCAN_TTL:
            return _catalog_cache["data"]
    root = settings.materials_dir
    models, datasets = [], []
    mroot = root / "models"
    if mroot.is_dir():
        for d in sorted(mroot.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                m = _describe_model(d, f"models/{d.name}")
                if m:
                    models.append(m)
    droot = root / "datasets"
    if droot.is_dir():
        for g in sorted(droot.iterdir()):
            if not g.is_dir() or g.name.startswith("."):
                continue
            # datasets/<group>/<name>/..., or a dataset straight under datasets/<name>/
            subdirs = [s for s in sorted(g.iterdir()) if s.is_dir() and not s.name.startswith(".")]
            candidates = [(s, f"datasets/{g.name}/{s.name}") for s in subdirs] or [(g, f"datasets/{g.name}")]
            for d, rel in candidates:
                ds = _describe_dataset(d, rel)
                if ds:
                    datasets.append(ds)
    data = {"models": models, "datasets": datasets}
    with _lock:
        _catalog_cache.update({"at": time.time(), "data": data})
    return data


def catalog(bus=None, force: bool = False) -> dict[str, list[dict[str, Any]]]:
    """This machine's catalog merged with what the worker nodes publish, each entry
    marked with the nodes that have it."""
    merged: dict[str, dict[str, dict[str, Any]]] = {"models": {}, "datasets": {}}
    sources = [("local", local_catalog(force))]
    if bus is not None:
        sources += [(snap.get("node"), snap.get("catalog") or {}) for snap in node_inventories(bus)]
    for where, cat in sources:
        for kind in ("models", "datasets"):
            for item in cat.get(kind) or []:
                entry = merged[kind].setdefault(item["path"], dict(item, nodes=[]))
                if where and where not in entry["nodes"]:
                    entry["nodes"].append(where)
    return {kind: sorted(v.values(), key=lambda x: x["path"]) for kind, v in merged.items()}


def publish(bus, registry, force: bool = False) -> None:
    """Called by the worker. TTL is generous: a node that goes quiet keeps its entry
    briefly rather than making every material blink out of existence."""
    try:
        bus.set_json(f"materials:{settings.node_name}", {"node": settings.node_name, "root": str(settings.materials_dir), "items": local_inventory(registry, force), "catalog": local_catalog(force), "at": time.time()}, ttl=int(SCAN_TTL * 2))
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


def _clean(rel: Any) -> str:
    return str(rel or "").replace("\\", "/").strip().strip("/")


def choices(spec, param: dict[str, Any]) -> list[dict[str, Any]]:
    """The materials a `material` step param may be set to: the experiment's declared
    materials of the param's kind, narrowed by an entry's `for: [param keys]`. Nothing
    else can be chosen, so the yaml is the one place that decides what a class sees."""
    kind, key = param.get("kind"), param.get("key")
    out = []
    for m in spec.materials or []:
        if m.get("kind") != kind or not _clean(m.get("path")):
            continue
        if m.get("for") and key not in m["for"]:
            continue
        out.append(m)
    return out


def missing_for(spec, bus=None) -> list[str]:
    """Required material paths no machine reports having. Used to warn a teacher
    before a lesson rather than after a student's run fails. Optional materials are
    choices in a step form; a missing one simply shows as unavailable there."""
    out = []
    for m in spec.materials or []:
        rel = _clean(m.get("path"))
        if rel and not m.get("optional") and not status(rel, bus)["available"]:
            out.append(rel)
    return out


def missing_locally(spec, params: Optional[dict[str, Any]] = None) -> list[str]:
    """What this machine lacks for a run. A worker uses it to decide whether it can
    take a job: every required material, and the optional ones the run chose."""
    chosen = {_clean(v) for v in (params or {}).values() if isinstance(v, str)}
    need = [rel for m in (spec.materials or []) if (rel := _clean(m.get("path"))) and (not m.get("optional") or rel in chosen)]
    return [rel for rel in need if storage.material_path(rel) is None]
