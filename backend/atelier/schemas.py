from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class UserOut(BaseModel):
    id: int
    username: str
    name: str
    role: str
    active: bool
    # the experiments this person may run on their own, outside a class
    self_experiments: list[str] = []
    created_at: datetime
    model_config = {"from_attributes": True}


class LoginIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    token: str
    user: UserOut


class PasswordIn(BaseModel):
    current: str
    new: str = Field(min_length=6)


class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    name: str = ""
    role: str = "student"
    password: str = Field(min_length=4)


class UserPatch(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    active: Optional[bool] = None
    password: Optional[str] = None


class SelfPermissionsIn(BaseModel):
    experiments: list[str] = []


class SignUpIn(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    name: str = ""
    password: str = Field(min_length=4)
    role: str = "student"  # asking for a teacher account waits for an admin


class RunOut(BaseModel):
    id: int
    user_id: int
    username: str = ""
    experiment: str
    kind: str
    step: Optional[int]
    parent_run_id: Optional[int]
    submission_id: Optional[int]
    label: str
    command: Optional[str]
    params: dict[str, Any]
    inputs: dict[str, Any]
    gpus: int
    gpu_ids: list[int]
    node: Optional[str] = None   # which GPU node ran it, when there is more than one
    status: str
    progress_pct: float
    progress_msg: str
    metrics: list[Any]
    outputs: dict[str, Any]
    error: Optional[str]
    exit_code: Optional[int]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    model_config = {"from_attributes": True}


class StepRunCreate(BaseModel):
    params: dict[str, Any] = {}
    code_source: str = "standard"  # "own" runs the student's file for this step
    parent_run_id: Optional[int] = None
    gpus: Optional[int] = None
    label: str = ""


class WorkspaceRunCreate(BaseModel):
    command: str = Field(min_length=1, max_length=2000)
    gpus: int = 1
    label: str = ""


class FileWrite(BaseModel):
    path: str
    content: str


class StepCodeWrite(BaseModel):
    code: str = Field(max_length=400_000)


class SubmissionCreate(BaseModel):
    note: str = ""
    step: int = 0


class GradeIn(BaseModel):
    score: Optional[float] = Field(default=None, ge=0, le=100)
    feedback: Optional[str] = None
    published: Optional[bool] = None


class SubmissionOut(BaseModel):
    id: int
    user_id: int
    username: str = ""
    name: str = ""
    experiment: str
    step: int = 0
    note: str
    sha256: str
    file_count: int
    bytes: int
    score: Optional[float]
    auto_score: Optional[float]
    feedback: str
    graded_by: Optional[int]
    graded_at: Optional[datetime]
    published: bool
    last_test_run_id: Optional[int]
    created_at: datetime
    model_config = {"from_attributes": True}


class DisplayIn(BaseModel):
    mode: str
    payload: dict[str, Any] = {}
    name: Optional[str] = None


class DisplayOut(BaseModel):
    id: int
    name: str
    mode: str
    payload: dict[str, Any]
    updated_at: datetime
    model_config = {"from_attributes": True}
