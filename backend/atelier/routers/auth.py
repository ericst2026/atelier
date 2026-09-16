from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import create_token, current_user, hash_password, verify_password
from ..db import get_db
from ..models import User
from ..schemas import LoginIn, PasswordIn, SignUpIn, TokenOut, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == body.username.strip().lower()))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Wrong username or password")
    if not user.active:
        # the password was right, so say what is actually wrong
        raise HTTPException(403, "Your account is waiting for an admin to approve it.")
    return TokenOut(token=create_token(user), user=UserOut.model_validate(user))


@router.post("/signup", response_model=UserOut, status_code=201)
def signup(body: SignUpIn, db: Session = Depends(get_db)):
    """Students sign themselves up and can sign in at once. A teacher account has to
    be approved by an admin. Running an experiment on your own is a separate
    permission, granted per experiment by an admin."""
    username = body.username.strip().lower()
    if not username.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise HTTPException(400, "A username can hold letters, digits, dot, dash and underscore only")
    if db.scalar(select(User).where(User.username == username)):
        raise HTTPException(409, "That username is taken")
    role = body.role if body.role in ("student", "teacher") else "student"
    # a student is in straight away; a teacher account is a request an admin answers
    user = User(username=username, name=body.name.strip()[:128], role=role, password_hash=hash_password(body.password), active=(role == "student"))
    db.add(user)
    db.commit()
    return user


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.post("/password")
def change_password(body: PasswordIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not verify_password(body.current, user.password_hash):
        raise HTTPException(400, "Current password is wrong")
    user.password_hash = hash_password(body.new)
    db.commit()
    return {"ok": True}
