from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def now() -> datetime:
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), default="")
    role: Mapped[str] = mapped_column(String(16), default="student")  # teacher | student
    password_hash: Mapped[str] = mapped_column(String(128))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Run(Base):
    """One unit of work executed by the worker on the GPU node.

    kind: step       — a guided experiment step (script from the experiment package)
          workspace  — a free-form command inside the student's project
          test       — the experiment grader against the live workspace
          grade      — the experiment grader against a frozen submission
    """

    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    experiment: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16), default="step")
    step: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    parent_run_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    submission_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    label: Mapped[str] = mapped_column(String(200), default="")
    command: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    gpus: Mapped[int] = mapped_column(Integer, default=0)
    gpu_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    timeout_min: Mapped[int] = mapped_column(Integer, default=360)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0)
    progress_msg: Mapped[str] = mapped_column(String(300), default="")
    metrics: Mapped[list[Any]] = mapped_column(JSON, default=list)
    outputs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    node: Mapped[Optional[str]] = mapped_column(String(64), default=None)  # which machine ran it
    exit_code: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Submission(Base):
    __tablename__ = "submissions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    experiment: Mapped[str] = mapped_column(String(64), index=True)
    note: Mapped[str] = mapped_column(Text, default="")
    sha256: Mapped[str] = mapped_column(String(64), default="")
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    bytes: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    auto_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feedback: Mapped[str] = mapped_column(Text, default="")
    graded_by: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    graded_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    last_test_run_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)


class Display(Base):
    """One of the wall screens. mode: grafana | live | progress | leaderboard | run | message"""

    __tablename__ = "displays"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    mode: Mapped[str] = mapped_column(String(16), default="message")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)
