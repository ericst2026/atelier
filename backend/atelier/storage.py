"""Filesystem layout and safe file helpers.

    <data>/runs/<run_id>/            params.json inputs.json meta.json run.log progress.json live.json result.json + artifacts
    <data>/workspaces/<user_id>/<experiment>/   the student's project (starts as a copy of the sample)
    <data>/submissions/<submission_id>/         frozen copy of a workspace
"""
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import settings

IGNORED_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "node_modules", ".pytest_cache"}
TEXT_LIMIT = 2 * 1024 * 1024


class StorageError(Exception):
    pass


def safe_join(root: Path, rel: str) -> Path:
    """Join rel onto root, refusing anything that escapes root."""
    rel = (rel or "").replace("\\", "/").lstrip("/")
    target = (root / rel).resolve()
    root_r = root.resolve()
    if target != root_r and root_r not in target.parents:
        raise StorageError("path escapes its root")
    return target


def run_dir(run_id: int) -> Path:
    return settings.runs_dir / str(run_id)


def workspace_dir(user_id: int, experiment: str) -> Path:
    return settings.workspaces_dir / str(user_id) / experiment


def submission_dir(submission_id: int) -> Path:
    return settings.submissions_dir / str(submission_id)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def tree(root: Path, max_depth: int = 6, max_entries: int = 3000) -> list[dict[str, Any]]:
    """Flat list of entries: {path, type, bytes}. Sorted directories-first per level."""
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out

    def walk(d: Path, depth: int) -> None:
        if depth > max_depth or len(out) > max_entries:
            return
        try:
            entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for p in entries:
            if p.name in IGNORED_DIRS or p.name.startswith("."):
                continue
            rel = p.relative_to(root).as_posix()
            if p.is_dir():
                out.append({"path": rel, "type": "dir", "bytes": 0})
                walk(p, depth + 1)
            else:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                out.append({"path": rel, "type": "file", "bytes": size})

    walk(root, 0)
    return out


def dir_stats(root: Path) -> dict[str, int]:
    files = 0
    size = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for f in filenames:
            files += 1
            try:
                size += (Path(dirpath) / f).stat().st_size
            except OSError:
                pass
    return {"files": files, "bytes": size}


def read_text(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StorageError("no such file")
    size = path.stat().st_size
    if size > TEXT_LIMIT:
        return {"path": str(path), "binary": True, "bytes": size, "content": None}
    raw = path.read_bytes()
    if b"\x00" in raw[:4096]:
        return {"binary": True, "bytes": size, "content": None}
    return {"binary": False, "bytes": size, "content": raw.decode("utf-8", errors="replace")}


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*IGNORED_DIRS))


def zip_dir(root: Path, exclude: Iterable[str] = ()) -> bytes:
    exclude = set(exclude) | IGNORED_DIRS
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in exclude]
            for f in filenames:
                p = Path(dirpath) / f
                zf.write(p, p.relative_to(root).as_posix())
    return buf.getvalue()


def unzip_to(data: bytes, root: Path, max_bytes: int) -> int:
    total = 0
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            total += info.file_size
            if total > max_bytes:
                raise StorageError("archive is larger than the allowed size")
            safe_join(root, info.filename)  # traversal check
        root.mkdir(parents=True, exist_ok=True)
        zf.extractall(root)
    return total


def sha256_dir(root: Path) -> str:
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for f in sorted(filenames):
            p = Path(dirpath) / f
            h.update(p.relative_to(root).as_posix().encode())
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()


def tail_log(path: Path, offset: int = 0, max_bytes: int = 256 * 1024) -> dict[str, Any]:
    """Incremental log reader: returns text after `offset` and the next offset."""
    if not path.exists():
        return {"text": "", "next_offset": 0, "eof": True, "size": 0}
    size = path.stat().st_size
    if offset < 0:
        offset = max(0, size - max_bytes)
    with open(path, "rb") as fh:
        fh.seek(offset)
        chunk = fh.read(max_bytes)
    return {"text": chunk.decode("utf-8", errors="replace"), "next_offset": offset + len(chunk), "eof": offset + len(chunk) >= size, "size": size}


def last_lines(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        block = min(size, 128 * 1024)
        fh.seek(size - block)
        text = fh.read(block).decode("utf-8", errors="replace")
    return text.splitlines()[-n:]


def material_path(rel: str) -> Optional[Path]:
    try:
        p = safe_join(settings.materials_dir, rel)
    except StorageError:
        return None
    return p if p.exists() else None
