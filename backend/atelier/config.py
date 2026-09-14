"""Runtime configuration. Everything comes from environment variables so the same
code runs in docker-compose, a systemd unit, or a developer shell."""
import os
from functools import cached_property
from pathlib import Path


def _get(name: str, default, cast=str):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw)


def detect_gpu_count() -> int:
    """How many GPUs the NVIDIA driver shows this process. 0 when there is no driver,
    no nvidia runtime in the container, or no GPU at all."""
    try:
        import pynvml  # nvidia-ml-py

        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount())
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return 0


class Settings:
    def __init__(self) -> None:
        self.root = Path(_get("ATELIER_ROOT", "/srv/atelier"))
        self.experiments_dir = Path(_get("ATELIER_EXPERIMENTS_DIR", self.root / "experiments"))
        self.materials_dir = Path(_get("ATELIER_MATERIALS_DIR", self.root / "materials"))
        self.data_dir = Path(_get("ATELIER_DATA_DIR", self.root / "data"))
        self.runs_dir = self.data_dir / "runs"
        self.workspaces_dir = self.data_dir / "workspaces"
        self.submissions_dir = self.data_dir / "submissions"

        self.database_url = _get("ATELIER_DATABASE_URL", f"sqlite:///{self.data_dir / 'atelier.db'}")
        self.redis_url = _get("ATELIER_REDIS_URL", "redis://localhost:6379/0")

        self.jwt_secret = _get("ATELIER_JWT_SECRET", "change-me-in-.env")
        self.jwt_expire_hours = _get("ATELIER_JWT_EXPIRE_HOURS", 12, int)
        self.admin_username = _get("ATELIER_ADMIN_USERNAME", "teacher")
        self.admin_password = _get("ATELIER_ADMIN_PASSWORD", "teacher")

        # How many GPUs this machine has comes from the driver (see gpu_count below).
        # ATELIER_GPU_COUNT only overrides it, e.g. to hold some GPUs back from jobs.
        self.gpu_count_override = _get("ATELIER_GPU_COUNT", None, int)
        # A node with no GPUs runs jobs on CPU, this many at a time. Each takes all the
        # cores it is given, so more than one mostly makes them all slower.
        self.cpu_slots = max(1, _get("ATELIER_CPU_SLOTS", 1, int))
        self.max_running_per_student = _get("ATELIER_MAX_RUNNING_PER_STUDENT", 2, int)
        self.max_gpus_per_student_run = _get("ATELIER_MAX_GPUS_PER_STUDENT_RUN", 2, int)
        self.default_timeout_min = _get("ATELIER_DEFAULT_TIMEOUT_MIN", 360, int)
        self.runner_backend = _get("ATELIER_RUNNER_BACKEND", "subprocess")  # subprocess | docker
        self.runner_image = _get("ATELIER_RUNNER_IMAGE", "atelier-runtime:latest")
        self.runner_memory = _get("ATELIER_RUNNER_MEMORY", "64g")
        self.python_bin = _get("ATELIER_PYTHON", "python")

        self.display_count = _get("ATELIER_DISPLAY_COUNT", 5, int)
        self.displays_public = _get("ATELIER_DISPLAYS_PUBLIC", True, bool)
        self.board_interval_sec = _get("ATELIER_BOARD_INTERVAL_SEC", 5, float)
        self.grafana_url = _get("ATELIER_GRAFANA_URL", "/grafana/d/atelier-hw/hardware?orgId=1&kiosk&refresh=5s")

        self.submission_max_mb = _get("ATELIER_SUBMISSION_MAX_MB", 2048, int)
        self.workspace_max_mb = _get("ATELIER_WORKSPACE_MAX_MB", 4096, int)
        self.log_tail_lines = _get("ATELIER_LOG_TAIL_LINES", 400, int)
        self.cors_origins = [o for o in _get("ATELIER_CORS_ORIGINS", "http://localhost:5173").split(",") if o]

        # --- where this process sits ---------------------------------------
        # all    one machine runs everything (the default)
        # api    this process serves the UI; the GPUs are somewhere else
        # worker this process runs jobs and has the GPUs
        self.role = _get("ATELIER_ROLE", "all").strip().lower()
        # shared  the API and the worker see the same data directory (one machine,
        #         or a network mount)
        # upload  they do not: the worker posts results back to the API over HTTP
        self.storage_mode = _get("ATELIER_STORAGE_MODE", "shared").strip().lower()
        self.api_url = _get("ATELIER_API_URL", "http://localhost:8000").rstrip("/")
        self.worker_token = _get("ATELIER_WORKER_TOKEN", "")
        self.upload_max_mb = _get("ATELIER_UPLOAD_MAX_MB", 64, int)
        self.upload_interval_sec = _get("ATELIER_UPLOAD_INTERVAL_SEC", 3.0, float)
        self.node_name = _get("ATELIER_NODE_NAME", os.environ.get("HOSTNAME", "node"))
        # the worker publishes a GPU snapshot to redis so an API on another machine
        # can still draw the hardware strip
        self.gpu_snapshot_ttl_sec = _get("ATELIER_GPU_SNAPSHOT_TTL_SEC", 15, int)
        # a worker has no web server of its own, so it listens here for Prometheus.
        # 0 turns it off.
        self.worker_metrics_port = _get("ATELIER_WORKER_METRICS_PORT", 9101, int)

    @cached_property
    def gpu_count(self) -> int:
        """GPUs on *this machine*. Only the worker needs it: the API learns the
        cluster's size from the workers' heartbeats. Read once, on first use."""
        if self.gpu_count_override is not None:
            return self.gpu_count_override
        return detect_gpu_count()

    @property
    def is_worker(self) -> bool:
        return self.role in ("all", "worker")

    @property
    def is_api(self) -> bool:
        return self.role in ("all", "api")

    @property
    def uploads(self) -> bool:
        return self.storage_mode == "upload"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.runs_dir, self.workspaces_dir, self.submissions_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
