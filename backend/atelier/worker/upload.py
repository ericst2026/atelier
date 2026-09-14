"""Sending results back to an API on another machine.

Used only when ATELIER_STORAGE_MODE=upload. Written against urllib so the worker
image needs no HTTP client beyond the standard library."""
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

from ..config import settings

log = logging.getLogger("atelier.worker.upload")
SKIP_SUFFIXES = (".pyc",)
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}


def _post(path: str, body: bytes, query: dict) -> Optional[dict]:
    url = f"{settings.api_url}/api{path}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "X-Atelier-Token": settings.worker_token,
        "Content-Type": "application/octet-stream",
    })
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read() or b"{}")


def _get(path: str) -> Optional[dict]:
    url = f"{settings.api_url}/api{path}"
    req = urllib.request.Request(url, headers={"X-Atelier-Token": settings.worker_token})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read() or b"{}")


def check_api() -> bool:
    """Called once at worker startup, so a wrong URL or token fails now rather than
    on the first student's run."""
    if not settings.uploads:
        return True
    try:
        _get("/internal/ping")
    except urllib.error.HTTPError as exc:
        log.error("the API rejected this worker (%s). Check ATELIER_WORKER_TOKEN on both sides.", exc.code)
        return False
    except Exception as exc:
        log.error("cannot reach the API at %s: %s", settings.api_url, exc)
        return False
    try:
        from .. import storage

        mine = storage.sha256_dir(settings.experiments_dir)[:16]
        theirs = (_get("/internal/fingerprint") or {}).get("experiments")
        if theirs and theirs != mine:
            log.warning("the experiments directory here (%s) differs from the API's (%s) — copy the same tree to both, or runs will fail in confusing ways", mine, theirs)
        else:
            log.info("experiments match the API (%s)", mine)
    except Exception as exc:
        log.warning("could not compare experiment directories: %s", exc)
    log.info("uploading results to %s", settings.api_url)
    return True


def send_file(run_id: int, run_dir: Path, rel: str, append_from: int = 0) -> int:
    """Returns how many bytes were sent. append_from > 0 sends only the tail."""
    p = run_dir / rel
    if not p.is_file():
        return 0
    size = p.stat().st_size
    limit = settings.upload_max_mb * 1024 * 1024
    if append_from == 0 and size > limit:
        log.info("not uploading %s for run %s: %.0f MB is over the %d MB limit, it stays on %s", rel, run_id, size / 1e6, settings.upload_max_mb, settings.node_name)
        return 0
    with open(p, "rb") as fh:
        fh.seek(append_from)
        body = fh.read()
    if not body:
        return 0
    _post(f"/internal/runs/{run_id}/file", body, {"path": rel, "append": "true" if append_from else "false"})
    return len(body)


def send_directory(run_id: int, run_dir: Path, skip: Iterable[str] = ()) -> dict:
    """Everything the API needs to render the run: result.json, the log, artifacts."""
    skip = set(skip)
    sent, skipped, total = 0, 0, 0
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file() or p.suffix in SKIP_SUFFIXES or SKIP_DIRS & set(p.parts):
            continue
        rel = p.relative_to(run_dir).as_posix()
        if rel in skip:
            continue
        n = send_file(run_id, run_dir, rel)
        if n:
            sent += 1
            total += n
        else:
            skipped += 1
    log.info("run %s: uploaded %d files (%.1f MB), left %d behind", run_id, sent, total / 1e6, skipped)
    return {"files": sent, "bytes": total, "skipped": skipped}


class LogStreamer:
    """Tails run.log to the API while the job runs, so the student sees output live
    even though the file is on another machine."""

    def __init__(self, run_id: int, run_dir: Path, rel: str = "run.log") -> None:
        self.run_id, self.run_dir, self.rel = run_id, run_dir, rel
        self.offset = 0
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def _pump(self) -> None:
        while not self.stop_event.is_set():
            self.flush()
            self.stop_event.wait(settings.upload_interval_sec)

    def flush(self) -> None:
        try:
            n = send_file(self.run_id, self.run_dir, self.rel, append_from=self.offset)
            self.offset += n
        except Exception as exc:
            log.debug("log upload for run %s failed: %s", self.run_id, exc)

    def __enter__(self) -> "LogStreamer":
        if settings.uploads:
            self.thread = threading.Thread(target=self._pump, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        if settings.uploads:
            self.flush()
