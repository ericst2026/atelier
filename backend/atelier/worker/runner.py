"""Executes one run: builds the command, streams its output, enforces timeout
and cancellation, and records the outcome."""
import json
import logging
import os
import shlex
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .. import storage
from ..bus import SyncBus
from ..config import settings
from . import upload
from ..db import SessionLocal
from ..models import Run, Submission

log = logging.getLogger("atelier.worker.runner")
PROGRESS_PREFIX = "::progress "
MAX_LIVE_POINTS = 4000


def job_env(run: Run, gpu_ids: list[int], run_dir: Path) -> dict[str, str]:
    lib = settings.experiments_dir / "_lib"
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{lib}:{settings.experiments_dir}:{run.inputs.get('workspace_dir', '')}",
        "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids) if gpu_ids else "",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(settings.materials_dir / "hf-cache"),
        "ATELIER_RUN_ID": str(run.id),
        "ATELIER_RUN_DIR": str(run_dir),
        "ATELIER_RUN_KIND": run.kind,
        "ATELIER_MATERIALS": str(settings.materials_dir),
        "ATELIER_EXPERIMENTS": str(settings.experiments_dir),
        "ATELIER_WORKSPACE": run.inputs.get("workspace_dir", ""),
        "ATELIER_GPUS": str(len(gpu_ids)),
        "ATELIER_DEVICE": "cuda" if gpu_ids else "cpu",
        # on a CPU-only node the run is the compute, so it gets the cores
        "OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 8) // settings.cpu_slots)) if settings.gpu_count == 0 else "8",
    }
    for k in ("LD_LIBRARY_PATH", "VIRTUAL_ENV", "CONDA_PREFIX", "NVIDIA_VISIBLE_DEVICES"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def build_command(run: Run, run_dir: Path) -> tuple[str, Path]:
    workspace = Path(run.inputs.get("workspace_dir") or run_dir)
    if run.kind == "workspace":
        return run.command or "", workspace
    cmd = (run.command or "").replace("{run_dir}", shlex.quote(str(run_dir))).replace("{workspace_dir}", shlex.quote(str(workspace)))
    return cmd, run_dir


def docker_command(run: Run, cmd: str, cwd: Path, gpu_ids: list[int], env: dict[str, str]) -> list[str]:
    args = ["docker", "run", "--rm", "--name", f"atelier-run-{run.id}", "--network", "none", "--memory", settings.runner_memory, "--pids-limit", "4096", "--shm-size", "8g", "-w", str(cwd)]
    if gpu_ids:
        args += ["--gpus", f'"device={",".join(str(g) for g in gpu_ids)}"']
    for k, v in env.items():
        if k in ("PATH", "HOME", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
            continue
        args += ["-e", f"{k}={v}"]
    args += ["-v", f"{settings.experiments_dir}:{settings.experiments_dir}:ro", "-v", f"{settings.materials_dir}:{settings.materials_dir}:ro", "-v", f"{storage.run_dir(run.id)}:{storage.run_dir(run.id)}"]
    ws = run.inputs.get("workspace_dir")
    if ws:
        mode = "ro" if run.kind == "grade" else "rw"
        args += ["-v", f"{ws}:{ws}:{mode}"]
    args += [settings.runner_image, "bash", "-lc", cmd]
    return args


class LiveSeries:
    """Accumulates series points from ::progress lines into live.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.series: dict[str, list[dict[str, float]]] = {}
        self.counter = 0
        self.dirty = False

    def add(self, series: dict[str, Any]) -> None:
        if not series:
            return
        x = series.pop("step", series.pop("x", None))
        if x is None:
            x = self.counter
        self.counter += 1
        for k, v in series.items():
            try:
                y = float(v)
            except (TypeError, ValueError):
                continue
            pts = self.series.setdefault(k, [])
            pts.append({"x": float(x), "y": y})
            if len(pts) > MAX_LIVE_POINTS:
                del pts[::2]
        self.dirty = True

    def flush(self) -> None:
        if self.dirty:
            storage.write_json(self.path, {"series": self.series})
            self.dirty = False


def execute(run_id: int, gpu_ids: list[int], bus: SyncBus) -> None:
    run_dir = storage.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        run = db.get(Run, run_id)
        if run is None or run.status != "queued":
            return
        run.status, run.started_at, run.gpu_ids = "running", datetime.utcnow(), gpu_ids
        run.node = settings.node_name
        db.commit()
    bus.publish_run(run_id, {"type": "status", "status": "running", "gpu_ids": gpu_ids})
    bus.publish_event({"type": "run", "id": run_id, "status": "running"})

    cmd, cwd = build_command(run, run_dir)
    env = job_env(run, gpu_ids, run_dir)
    log_path = run_dir / "run.log"
    live = LiveSeries(run_dir / "live.json")
    timeout = max(1, run.timeout_min) * 60
    started = time.time()
    outcome = {"status": "failed", "error": None, "exit_code": None}
    cancelled = threading.Event()

    with upload.LogStreamer(run_id, run_dir), open(log_path, "ab") as logf:
        device = f"gpus {gpu_ids}" if gpu_ids else (f"cpu (asked for {run.gpus} GPU(s); no GPU node is up)" if run.gpus else "cpu")
        header = f"[atelier] run {run_id} · {run.kind} · {device} · on {settings.node_name} · {datetime.utcnow().isoformat()}Z\n[atelier] cwd {cwd}\n[atelier] $ {cmd}\n"
        logf.write(header.encode())
        logf.flush()
        try:
            if settings.runner_backend == "docker":
                argv = docker_command(run, cmd, cwd, gpu_ids, env)
                proc = subprocess.Popen(argv, cwd=str(run_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
            else:
                cwd.mkdir(parents=True, exist_ok=True)
                proc = subprocess.Popen(["bash", "-lc", cmd], cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
        except Exception as exc:
            outcome["error"] = f"could not start: {exc}"
            logf.write(f"[atelier] {outcome['error']}\n".encode())
            proc = None

        if proc is not None:

            def watchdog() -> None:
                while proc.poll() is None:
                    if bus.is_cancelled(run_id):
                        cancelled.set()
                        _terminate(proc, run_id)
                        return
                    if time.time() - started > timeout:
                        outcome["error"] = f"timed out after {run.timeout_min} minutes"
                        _terminate(proc, run_id)
                        return
                    time.sleep(1.0)

            threading.Thread(target=watchdog, daemon=True).start()
            buffer: list[str] = []
            last_flush = time.time()
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line.encode("utf-8", errors="replace"))
                if line.startswith(PROGRESS_PREFIX):
                    try:
                        p = json.loads(line[len(PROGRESS_PREFIX):])
                    except ValueError:
                        p = {}
                    series = p.get("series") or {}
                    live.add(dict(series))
                    with SessionLocal() as db:
                        r = db.get(Run, run_id)
                        if r is not None:
                            if p.get("pct") is not None:
                                r.progress_pct = max(0.0, min(100.0, float(p["pct"])))
                            if p.get("msg") is not None:
                                r.progress_msg = str(p["msg"])[:300]
                            db.commit()
                    bus.publish_run(run_id, {"type": "progress", "pct": p.get("pct"), "msg": p.get("msg"), "series": series})
                    continue
                buffer.append(line.rstrip("\n"))
                if len(buffer) >= 40 or time.time() - last_flush > 0.3:
                    logf.flush()
                    live.flush()
                    bus.publish_run(run_id, {"type": "log", "lines": buffer})
                    buffer, last_flush = [], time.time()
            if buffer:
                bus.publish_run(run_id, {"type": "log", "lines": buffer})
            rc = proc.wait()
            live.flush()
            outcome["exit_code"] = rc
            if cancelled.is_set():
                outcome["status"], outcome["error"] = "cancelled", "Cancelled by request"
            elif rc == 0:
                outcome["status"] = "succeeded"
            else:
                outcome["status"] = "failed"
                outcome["error"] = outcome["error"] or f"exited with code {rc}"
            logf.write(f"[atelier] finished: {outcome['status']} (exit {rc}) after {time.time() - started:.1f}s\n".encode())

    if settings.uploads:
        # the API is on another machine and cannot see this directory, so push the
        # results across before marking the run finished
        try:
            upload.send_directory(run_id, run_dir, skip={"run.log"})
        except Exception as exc:
            log.error("run %s: uploading results failed: %s", run_id, exc)
            outcome["status"] = "failed"
            outcome["error"] = outcome["error"] or f"the run finished but its results could not be sent to the API: {exc}"

    result = storage.read_json(run_dir / "result.json", None)
    metrics_summary = result.get("metrics", []) if isinstance(result, dict) else []
    outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
    with SessionLocal() as db:
        r = db.get(Run, run_id)
        if r is not None:
            r.status = outcome["status"]
            r.error = outcome["error"]
            r.exit_code = outcome["exit_code"]
            r.finished_at = datetime.utcnow()
            r.metrics = metrics_summary if isinstance(metrics_summary, list) else []
            r.outputs = outputs if isinstance(outputs, dict) else {}
            if r.status == "succeeded":
                r.progress_pct = 100.0
            db.commit()
            if r.kind == "grade" and r.submission_id:
                sub = db.get(Submission, r.submission_id)
                if sub is not None:
                    for m in r.metrics:
                        if isinstance(m, dict) and m.get("key") == "score":
                            try:
                                sub.auto_score = float(m.get("value"))
                            except (TypeError, ValueError):
                                pass
                    db.commit()
    bus.clear_cancel(run_id)
    bus.publish_run(run_id, {"type": "status", "status": outcome["status"], "error": outcome["error"], "metrics": metrics_summary})
    bus.publish_event({"type": "run", "id": run_id, "status": outcome["status"]})


def _terminate(proc: subprocess.Popen, run_id: int) -> None:
    if settings.runner_backend == "docker":
        subprocess.run(["docker", "kill", f"atelier-run-{run_id}"], capture_output=True)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        for _ in range(100):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
