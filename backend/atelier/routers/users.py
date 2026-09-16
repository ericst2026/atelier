import csv
import io

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import services
from ..auth import hash_password, require_admin
from ..db import get_db
from ..deps import get_registry
from ..models import SelfPermission, User
from ..registry import Registry
from ..schemas import SelfPermissionsIn, UserCreate, UserOut, UserPatch

# accounts are the admin's: a teacher runs classes and admits students, nothing more
router = APIRouter(prefix="/users", tags=["users"], dependencies=[Depends(require_admin)])


def _out(user: User, db: Session) -> UserOut:
    return services.user_out(db, user)


@router.get("", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db)):
    return [_out(u, db) for u in db.scalars(select(User).order_by(User.role, User.username)).all()]


@router.get("/{user_id}/self-experiments")
def get_self_experiments(user_id: int, db: Session = Depends(get_db)):
    """Which experiments this person may run outside a class."""
    if db.get(User, user_id) is None:
        raise HTTPException(404, "User not found")
    return {"experiments": sorted(db.scalars(select(SelfPermission.experiment).where(SelfPermission.user_id == user_id)).all())}


@router.put("/{user_id}/self-experiments", response_model=UserOut)
def set_self_experiments(user_id: int, body: SelfPermissionsIn, admin: User = Depends(require_admin), db: Session = Depends(get_db), registry: Registry = Depends(get_registry)):
    """Replace the list an admin has granted. Only an admin, and only experiments
    that exist."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    known = {s.slug for s in registry.list()}
    wanted = {e.strip() for e in body.experiments if e.strip()}
    unknown = sorted(wanted - known)
    if unknown:
        raise HTTPException(400, f"No such experiment: {', '.join(unknown)}")
    have = {p.experiment: p for p in db.scalars(select(SelfPermission).where(SelfPermission.user_id == user_id)).all()}
    for slug in wanted - set(have):
        db.add(SelfPermission(user_id=user_id, experiment=slug, granted_by=admin.id))
    for slug in set(have) - wanted:
        db.delete(have[slug])
    db.commit()
    return _out(user, db)


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
def patch_user(user_id: int, body: UserPatch, db: Session = Depends(get_db), me: User = Depends(require_admin)):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    if body.name is not None:
        user.name = body.name
    if body.role is not None:
        if body.role not in ("admin", "teacher", "student"):
            raise HTTPException(400, "role must be admin, teacher or student")
        if me.role != "admin":
            raise HTTPException(403, "Only an admin can change what someone is")
        if user.id == me.id and body.role != "admin":
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
