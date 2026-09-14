"""The scheduler. One process per GPU node:  python -m atelier.worker"""
import logging
import threading
import time
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from .. import materials, metrics
from ..bus import SyncBus
from ..config import settings
from ..db import SessionLocal, init_db
from ..models import Run
from ..registry import Registry
from . import runner, upload
from .gpus import GpuPool

log = logging.getLogger("atelier.worker")


class Worker:
    def __init__(self) -> None:
        self.bus = SyncBus()
        self.pool = GpuPool(settings.gpu_count)
        self.pending: list[int] = []
        self.running: dict[int, tuple[threading.Thread, list[int]]] = {}
        self.lock = threading.Lock()
        self.registry = Registry(settings.experiments_dir)
        # runs this node cannot serve — usually because the materials are elsewhere.
        # id → the time it may be reconsidered, so the queue is not spun on.
        self.deferred: dict[int, float] = {}
        self.materials_at = 0.0

    def recover(self) -> None:
        """After a restart, clean up after *this* node only.

        With more than one GPU node the queue is shared, so a worker must never fail
        a run another machine is still executing, and must never drain the queue —
        both of which a naive recovery does, and both of which look like the cluster
        eating jobs at random.
        """
        node = settings.node_name
        with SessionLocal() as db:
            mine = db.scalars(select(Run).where(Run.status == "running", Run.node == node)).all()
            for r in mine:
                r.status, r.error, r.finished_at = "failed", f"The worker on {node} restarted while this run was in progress", datetime.utcnow()
                self.bus.publish_event({"type": "run", "id": r.id, "status": "failed"})
            # runs that started before nodes were recorded belong to whoever restarts
            # first, and only when nobody else is alive to claim them
            orphans = db.scalars(select(Run).where(Run.status == "running", Run.node.is_(None))).all()
            others = [st for st in self.bus.worker_states() if st.get("node") != node]
            if orphans and not others:
                for r in orphans:
                    r.status, r.error, r.finished_at = "failed", "The worker restarted while this run was in progress", datetime.utcnow()
                    self.bus.publish_event({"type": "run", "id": r.id, "status": "failed"})
            db.commit()
        log.info("recovered: failed %d run(s) that were mid-flight on %s", len(mine), node)
        self.requeue_orphans()

    def requeue_orphans(self) -> None:
        """A run marked queued in the database but missing from the Redis list — which
        happens if Redis was restarted or flushed. One node does this, under a lock,
        so several starting at once do not enqueue the same job several times."""
        if not self.bus.claim_lock("requeue", settings.node_name, ttl=30):
            return
        try:
            in_queue = set(self.bus.queued_ids())
            with SessionLocal() as db:
                queued = [r.id for r in db.scalars(select(Run).where(Run.status == "queued").order_by(Run.id)).all()]
            missing = [rid for rid in queued if rid not in in_queue]
            for rid in missing:
                self.bus.enqueue(rid)
            if missing:
                log.info("re-queued %d run(s) that were in the database but not in the queue: %s", len(missing), missing)
        except Exception:
            log.exception("could not reconcile the queue")

    @property
    def cpu_only(self) -> bool:
        return settings.gpu_count == 0

    def drain(self) -> None:
        # a run this node just handed back stays in the queue for the others; taking it
        # again at once would keep it away from them
        now, handed_back = time.time(), []
        while True:
            if self.cpu_only and len(self.running) + len(self.pending) >= settings.cpu_slots:
                break  # take only what can start here, and leave the rest to other nodes
            rid = self.bus.pop()
            if rid is None:
                break
            if self.deferred.get(rid, 0) > now:
                handed_back.append(rid)
            elif rid not in self.pending and rid not in self.running:
                self.pending.append(rid)
        for rid in handed_back:
            self.bus.enqueue(rid)

    def gpu_nodes_alive(self) -> bool:
        try:
            return any(int(st.get("gpu_count") or 0) > 0 for st in self.bus.worker_states())
        except Exception:
            return False

    def reap(self) -> None:
        with self.lock:
            done = [rid for rid, (t, _) in self.running.items() if not t.is_alive()]
            for rid in done:
                _, ids = self.running.pop(rid)
                self.pool.release(ids)
                log.info("run %s finished, freed gpus %s", rid, ids)

    def can_run_here(self, run: Run) -> tuple[bool, list[str]]:
        """Whether this node has what the experiment needs. With materials kept local
        to each GPU node, a node without them must let another one take the job."""
        try:
            spec = self.registry.get(run.experiment)
        except Exception:
            return True, []          # unknown experiment: let the run fail with a real error
        return (not (missing := materials.missing_locally(spec))), missing

    def defer(self, rid: int, missing: list[str], seconds: float = 20.0) -> None:
        """Put it back for another node. If nobody has the materials, fail it now
        rather than letting a student watch a queue that will never move."""
        holders = set()
        for rel in missing:
            holders |= set(materials.status(rel, self.bus)["nodes"])
        if holders - {settings.node_name}:
            self.deferred[rid] = time.time() + seconds
            self.bus.enqueue(rid)
            log.info("run %s needs %s, which this node does not have — leaving it for %s", rid, ", ".join(missing), ", ".join(sorted(holders)))
            return
        with SessionLocal() as db:
            r = db.get(Run, rid)
            if r is not None and r.status == "queued":
                r.status = "failed"
                r.error = f"No GPU node has the materials this experiment needs: {', '.join(missing)}. Fetch them with scripts/offline/fetch-materials.py on a node, or run an experiment that generates its own data."
                r.finished_at = datetime.utcnow()
                db.commit()
        self.bus.publish_event({"type": "run", "id": rid, "status": "failed"})
        log.warning("run %s failed: no node has %s", rid, ", ".join(missing))

    def schedule(self) -> None:
        now = time.time()
        gpu_nodes: Optional[bool] = None  # asked of redis at most once a pass
        for rid in list(self.pending):
            if self.deferred.get(rid, 0) > now:
                continue
            with SessionLocal() as db:
                run = db.get(Run, rid)
                if run is None or run.status != "queued":
                    self.pending.remove(rid)
                    continue
                need = max(0, int(run.gpus))
                ok, missing = self.can_run_here(run)
            if not ok:
                self.pending.remove(rid)
                self.defer(rid, missing)
                continue
            if self.cpu_only:
                # a node without GPUs runs everything on CPU, but a run that asked for
                # GPUs goes to a GPU node whenever one is up
                if need and (gpu_nodes if gpu_nodes is not None else (gpu_nodes := self.gpu_nodes_alive())):
                    self.pending.remove(rid)
                    self.deferred[rid] = now + 10
                    self.bus.enqueue(rid)
                    log.info("run %s asks for %d GPU(s) and a GPU node is up — leaving it for that node", rid, need)
                    continue
                with self.lock:
                    if len(self.running) >= settings.cpu_slots:
                        continue
                ids = []
            else:
                ids = self.pool.allocate(need)
                if ids is None:
                    continue  # not enough free GPUs; let smaller jobs pass
            self.pending.remove(rid)
            t = threading.Thread(target=self._safe_execute, args=(rid, ids), name=f"run-{rid}", daemon=True)
            with self.lock:
                self.running[rid] = (t, ids)
            t.start()
            log.info("started run %s on gpus %s", rid, ids)

    def _safe_execute(self, rid: int, ids: list[int]) -> None:
        try:
            runner.execute(rid, ids, self.bus)
        except Exception:
            log.exception("run %s crashed inside the worker", rid)
            with SessionLocal() as db:
                r = db.get(Run, rid)
                if r is not None and r.status in ("queued", "running"):
                    r.status, r.error, r.finished_at = "failed", "worker error (see worker log)", datetime.utcnow()
                    db.commit()
            self.bus.publish_event({"type": "run", "id": rid, "status": "failed"})

    def heartbeat(self) -> None:
        with self.lock:
            running = {str(rid): ids for rid, (_, ids) in self.running.items()}
        self.bus.set_worker_state({"gpu_count": settings.gpu_count, "device": "cpu" if self.cpu_only else "cuda", "cpu_slots": settings.cpu_slots if self.cpu_only else None, "free_gpus": sorted(self.pool.free), "running": running, "pending": list(self.pending), "backend": settings.runner_backend, "node": settings.node_name, "storage_mode": settings.storage_mode, "at": datetime.utcnow().isoformat()})
        # an API on another machine has no GPUs of its own to sample, and no
        # materials either — both are published from here
        metrics.publish_snapshot(self.bus)
        if time.time() - self.materials_at > 60:
            self.registry.reload()
            materials.publish(self.bus, self.registry)
            self.materials_at = time.time()

    def cancel_queued(self) -> None:
        for rid in list(self.pending):
            if self.bus.is_cancelled(rid):
                self.pending.remove(rid)
                self.bus.clear_cancel(rid)

    def loop(self) -> None:
        source = "from ATELIER_GPU_COUNT" if settings.gpu_count_override is not None else "detected"
        log.info("worker up on %s: %d gpus (%s), backend=%s, storage=%s", settings.node_name, settings.gpu_count, source, settings.runner_backend, settings.storage_mode)
        if self.cpu_only:
            log.warning("no GPUs visible on %s: running as a CPU-only node, %d run(s) at a time. Runs that ask for GPUs go to a GPU node when one is up and run here on CPU when none is. "
                        "If this machine does have GPUs, check the NVIDIA driver and the nvidia container runtime", settings.node_name, settings.cpu_slots)
        if settings.uploads and not upload.check_api():
            log.error("refusing to start: ATELIER_STORAGE_MODE=upload but the API is unreachable or rejects this worker")
            raise SystemExit(2)
        beat = 0.0
        while True:
            try:
                self.drain()
                self.cancel_queued()
                self.reap()
                self.schedule()
                if time.time() - beat > 3:
                    self.heartbeat()
                    beat = time.time()
            except Exception:
                log.exception("scheduler loop error")
                time.sleep(2)
            time.sleep(0.5)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings.ensure_dirs()
    init_db()
    # without this the NVML numbers are empty: nothing to publish to redis for the
    # GPU strip, and nothing for Prometheus to scrape
    metrics.start_sampler()
    if metrics.start_metrics_server(settings.worker_metrics_port):
        log.info("metrics on :%d for prometheus to scrape", settings.worker_metrics_port)
    w = Worker()
    materials.publish(w.bus, w.registry, force=True)
    have = len(materials.local_inventory(w.registry))
    log.info("materials on %s: %d of the paths the experiments ask for, under %s", settings.node_name, have, settings.materials_dir)
    w.recover()
    w.loop()


if __name__ == "__main__":
    main()
