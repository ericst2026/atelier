"""Hardware + platform metrics: sampled with NVML, exported for Prometheus and used by the boards."""
import threading
import time
from typing import Any, Optional

from .config import settings

from prometheus_client import CollectorRegistry, Gauge, make_asgi_app

registry = CollectorRegistry()
GPU_UTIL = Gauge("atelier_gpu_utilization_percent", "GPU utilization", ["gpu", "name"], registry=registry)
GPU_MEM_USED = Gauge("atelier_gpu_memory_used_bytes", "GPU memory used", ["gpu", "name"], registry=registry)
GPU_MEM_TOTAL = Gauge("atelier_gpu_memory_total_bytes", "GPU memory total", ["gpu", "name"], registry=registry)
GPU_TEMP = Gauge("atelier_gpu_temperature_celsius", "GPU temperature", ["gpu", "name"], registry=registry)
GPU_POWER = Gauge("atelier_gpu_power_watts", "GPU power draw", ["gpu", "name"], registry=registry)
RUNS = Gauge("atelier_runs", "Runs by status", ["status"], registry=registry)
GPUS_ALLOCATED = Gauge("atelier_gpus_allocated", "GPUs reserved by running jobs", registry=registry)
USERS_ACTIVE = Gauge("atelier_users_active", "Users with a run in the last 15 minutes", registry=registry)

metrics_app = make_asgi_app(registry=registry)

_lock = threading.Lock()
_snapshot: list[dict[str, Any]] = []
_available = False


def _sample_once() -> list[dict[str, Any]]:
    import pynvml  # nvidia-ml-py

    pynvml.nvmlInit()
    out = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temp = 0
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except pynvml.NVMLError:
                power = 0.0
            out.append(
                {"index": i, "name": name, "util": util.gpu, "mem_util": util.memory, "mem_used": mem.used, "mem_total": mem.total, "temp": temp, "power": power}
            )
    finally:
        pynvml.nvmlShutdown()
    return out


def _loop(interval: float) -> None:
    global _snapshot, _available
    while True:
        try:
            snap = _sample_once()
            _available = True
        except Exception:
            snap = []
            _available = False
        with _lock:
            _snapshot = snap
        for g in snap:
            labels = (str(g["index"]), g["name"])
            GPU_UTIL.labels(*labels).set(g["util"])
            GPU_MEM_USED.labels(*labels).set(g["mem_used"])
            GPU_MEM_TOTAL.labels(*labels).set(g["mem_total"])
            GPU_TEMP.labels(*labels).set(g["temp"])
            GPU_POWER.labels(*labels).set(g["power"])
        time.sleep(interval)


def start_sampler(interval: float = 2.0) -> None:
    threading.Thread(target=_loop, args=(interval,), daemon=True, name="gpu-sampler").start()


def start_metrics_server(port: int) -> bool:
    """Expose /metrics over HTTP. The API gets this for free by mounting metrics_app
    on the web server; a worker has no web server, so it needs its own listener for
    Prometheus to scrape."""
    if not port:
        return False
    try:
        from prometheus_client import start_http_server

        start_http_server(port, registry=registry)
        return True
    except Exception:
        return False


def publish_snapshot(bus) -> None:
    """Called by the worker so an API on another machine can draw the GPU strip."""
    try:
        bus.set_json(f"gpu:snapshot:{settings.node_name}", {**gpu_snapshot(), "node": settings.node_name}, ttl=settings.gpu_snapshot_ttl_sec)
    except Exception:  # a failure here must never stop a run
        pass


def remote_snapshots(bus) -> list[dict[str, Any]]:
    """Every GPU node's latest sample. Indices repeat across machines, so each GPU
    also carries the node it belongs to."""
    out = []
    try:
        for key in bus.r.scan_iter("atelier:gpu:snapshot:*"):
            snap = bus.get_json(key.split("atelier:", 1)[-1])
            if snap:
                out.append(snap)
    except Exception:
        return []
    return sorted(out, key=lambda s: s.get("node") or "")


def gpu_snapshot() -> dict[str, Any]:
    with _lock:
        return {"available": _available, "gpus": list(_snapshot)}


def update_run_gauges(counts: dict[str, int], allocated: int, active_users: int) -> None:
    for status in ("queued", "running", "succeeded", "failed", "cancelled"):
        RUNS.labels(status).set(counts.get(status, 0))
    GPUS_ALLOCATED.set(allocated)
    USERS_ACTIVE.set(active_users)
