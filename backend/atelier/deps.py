"""Process-wide singletons wired at startup (see main.py)."""
from typing import Optional

from .bus import AsyncBus, SyncBus
from .config import settings
from .events import Hub
from .registry import Registry

registry = Registry(settings.experiments_dir)
sync_bus = SyncBus()
async_bus = AsyncBus()
hub: Optional[Hub] = None


def get_registry() -> Registry:
    return registry


def get_bus() -> SyncBus:
    return sync_bus


def get_hub() -> Hub:
    assert hub is not None, "hub not started"
    return hub
