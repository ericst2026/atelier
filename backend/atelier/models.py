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
    role: Mapped[str] = mapped_column(String(16), default="student")  # admin | teacher | student
    password_hash: Mapped[str] = mapped_column(String(128))
    # active is "may sign in". A student who signs up is let in at once; someone
    # asking for a teacher account waits for an admin.
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
    # the step whose code was handed in; 0 is a submission from before steps had code
    step: Mapped[int] = mapped_column(Integer, default=0, index=True)
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
    # the run their own code produced, and the same step run by the standard code
    # with the same settings, so the two can be put side by side
    own_run_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    standard_run_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)


class Display(Base):
    """One of the wall screens. mode: grafana | live | progress | leaderboard | run | message"""

    __tablename__ = "displays"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    mode: Mapped[str] = mapped_column(String(16), default="message")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ClassSession(Base):
    """One experiment the teacher is running with the class. At most one has
    ended_at NULL at any time: that is the class currently in the room."""

    __tablename__ = "class_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment: Mapped[str] = mapped_column(String(64), index=True)
    # what this class is called: the experiment says what kind it is, the name says
    # which one, so two classes on the same experiment stay apart
    name: Mapped[str] = mapped_column(String(120), default="")
    started_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    # a paused class keeps its roster and its settings, and frees the room for
    # another teacher until it is resumed
    paused_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # what the teacher set when starting it: the run or prepared material each step
    # begins from, so students do not have to have done the earlier experiment
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class SessionMember(Base):
    """A student who asked to attend, and whether the teacher let them in."""

    __tablename__ = "session_members"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("class_sessions.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    admitted: Mapped[bool] = mapped_column(Boolean, default=False)
    asked_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class SelfPermission(Base):
    """One experiment one person may run on their own, outside any class. Nobody
    has this by default — not students and not teachers; an admin grants it per
    person and per experiment."""

    __tablename__ = "self_permissions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    experiment: Mapped[str] = mapped_column(String(64), index=True)
    granted_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    granted_at: Mapped[datetime] = mapped_column(DateTime, default=now)
