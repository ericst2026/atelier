import csv
import io

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import hash_password, require_teacher
from ..db import get_db
from ..models import User
from ..schemas import UserCreate, UserOut, UserPatch

router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(require_teacher)])


@router.get("", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.scalars(select(User).order_by(User.role, User.username)).all()


@router.post("", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    if body.role not in ("teacher", "student"):
        raise HTTPException(400, "role must be teacher or student")
    username = body.username.strip().lower()
    if db.scalar(select(User).where(User.username == username)):
        raise HTTPException(409, f"{username} already exists")
    user = User(username=username, name=body.name, role=body.role, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return user


@router.post("/import")
async def import_users(file: UploadFile, db: Session = Depends(get_db)):
    """CSV with header: username,name,role,password. Existing usernames are skipped."""
    text = (await file.read()).decode("utf-8-sig")
    created, skipped = [], []
    for row in csv.DictReader(io.StringIO(text)):
        username = (row.get("username") or "").strip().lower()
        if not username or not row.get("password"):
            continue
        if db.scalar(select(User).where(User.username == username)):
            skipped.append(username)
            continue
        role = (row.get("role") or "student").strip().lower()
        db.add(User(username=username, name=(row.get("name") or "").strip(), role=role if role in ("teacher", "student") else "student", password_hash=hash_password(row["password"].strip())))
        created.append(username)
    db.commit()
    return {"created": created, "skipped": skipped}


@router.patch("/{user_id}", response_model=UserOut)
def patch_user(user_id: int, body: UserPatch, db: Session = Depends(get_db), me: User = Depends(require_teacher)):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    if body.name is not None:
        user.name = body.name
    if body.role is not None:
        if body.role not in ("teacher", "student"):
            raise HTTPException(400, "role must be teacher or student")
        if user.id == me.id and body.role != "teacher":
            raise HTTPException(400, "You cannot demote yourself")
        user.role = body.role
    if body.active is not None:
        if user.id == me.id and not body.active:
            raise HTTPException(400, "You cannot deactivate yourself")
        user.active = body.active
    if body.password:
        user.password_hash = hash_password(body.password)
    db.commit()
    return user
