import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import settings

log = logging.getLogger("atelier.db")

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from . import models  # noqa: F401 (registers tables)

    if settings.database_url.startswith("sqlite"):
        settings.ensure_dirs()
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    models.Base.metadata.create_all(engine)
    _add_missing_columns(engine)


def _add_missing_columns(engine) -> None:
    """create_all() adds tables but never columns. The few columns added after a
    release are put on by hand here, so an existing database keeps working."""
    from sqlalchemy import inspect, text

    wanted = {"submissions": {"step": "INTEGER DEFAULT 0"}}
    insp = inspect(engine)
    for table, cols in wanted.items():
        if table not in insp.get_table_names():
            continue
        have = {c["name"] for c in insp.get_columns(table)}
        for name, decl in cols.items():
            if name in have:
                continue
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {decl}"))
            log.info("added column %s.%s", table, name)
