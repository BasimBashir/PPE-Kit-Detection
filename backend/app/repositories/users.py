"""Local users and opaque bearer sessions (NFR-SEC-002, ADR-012). Passwords: argon2id; tokens stored as sha256."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import select
from sqlalchemy.orm import Session as DB

from app.models import Session, User, utcnow

ROLES = ("viewer", "operator", "admin")
SESSION_TTL = timedelta(hours=12)
_ph = PasswordHasher()
_DUMMY_HASH = _ph.hash("timing-equaliser")


def hash_password(pw: str) -> str:
    if len(pw) < 8:
        raise ValueError("password must be at least 8 characters")
    return _ph.hash(pw)


def create_user(s: DB, username: str, password: str, role: str) -> User:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    u = User(username=username, password_hash=hash_password(password), role=role)
    s.add(u)
    s.flush()
    return u


def authenticate(s: DB, username: str, password: str) -> User | None:
    u = s.scalar(select(User).where(User.username == username))
    try:
        _ph.verify(u.password_hash if u else _DUMMY_HASH, password)  # constant-ish time for unknown users
    except (VerificationError, InvalidHashError):
        return None
    return u if u and u.enabled else None


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(s: DB, user: User) -> tuple[str, Session]:
    token = secrets.token_urlsafe(32)
    sess = Session(token_hash=_h(token), user_id=user.id, expires_at=utcnow() + SESSION_TTL)
    s.add(sess)
    s.flush()
    return token, sess


def user_for_token(s: DB, token: str) -> User | None:
    sess = s.get(Session, _h(token))
    if sess is None or sess.revoked_at is not None or sess.expires_at <= utcnow():
        return None
    u = s.get(User, sess.user_id)
    return u if u and u.enabled else None


def revoke(s: DB, token: str) -> None:
    sess = s.get(Session, _h(token))
    if sess and sess.revoked_at is None:
        sess.revoked_at = utcnow()


def list_users(s: DB) -> list[User]:
    return list(s.scalars(select(User).order_by(User.username)))


def get_user(s: DB, user_id: uuid.UUID) -> User | None:
    return s.get(User, user_id)


def update_user(s: DB, u: User, role: str | None = None, enabled: bool | None = None,
                password: str | None = None) -> User:
    if role is not None:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        u.role = role
    if enabled is not None:
        u.enabled = enabled
    if password is not None:
        u.password_hash = hash_password(password)
    s.flush()
    return u
