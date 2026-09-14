"""GPU allocator for one node. Lowest free indices first; jobs that need more
GPUs than are free wait, smaller jobs may pass them (classroom fairness beats
strict FIFO here — a 1-GPU tokenizer test should not wait behind an 8-GPU pretrain)."""
import threading
from typing import Optional


class GpuPool:
    def __init__(self, count: int) -> None:
        self.free: set[int] = set(range(count))
        self.lock = threading.Lock()

    def allocate(self, n: int) -> Optional[list[int]]:
        with self.lock:
            if n > len(self.free):
                return None
            ids = sorted(self.free)[:n]
            self.free.difference_update(ids)
            return ids

    def release(self, ids: list[int]) -> None:
        with self.lock:
            self.free.update(ids)

    def free_count(self) -> int:
        with self.lock:
            return len(self.free)
