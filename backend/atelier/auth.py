import time
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import User

_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_token(user: User) -> str:
    payload = {"sub": str(user.id), "role": user.role, "exp": int(time.time()) + settings.jwt_expire_hours * 3600}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def user_from_token(db: Session, token: Optional[str]) -> Optional[User]:
    if not token:
        return None
    data = decode_token(token)
    if not data:
        return None
    user = db.get(User, int(data["sub"]))
    if user is None or not user.active:
        return None
    return user


def current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    token: Optional[str] = Query(default=None, description="Alternative to the Authorization header (downloads, websockets)"),
    db: Session = Depends(get_db),
) -> User:
    raw = creds.credentials if creds else token
    user = user_from_token(db, raw)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue")
    return user


def require_teacher(user: User = Depends(current_user)) -> User:
    if user.role != "teacher":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Teachers only")
    return user


def optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    token: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> Optional[User]:
    return user_from_token(db, creds.credentials if creds else token)
