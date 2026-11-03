from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import ApiError, AppContext, bearer, check_login_rate, ctx, current_user, db, require
from app.db.base import session_scope
from app.models import User
from app.repositories import users as users_repo
from app.repositories.misc import audit
from app.schemas.requests import LoginIn, UserCreate, UserUpdate
from app.schemas.serializers import iso, user_out

router = APIRouter(prefix="/api/v1", tags=["auth"])


@router.post("/auth/login")
def login(body: LoginIn, request: Request, c: AppContext = Depends(ctx), s: Session = Depends(db)):
    check_login_rate(c, request.client.host if request.client else "unknown")
    u = users_repo.authenticate(s, body.username, body.password)
    if u is None:
        with session_scope(c.sf) as s2:  # own transaction: survives the 401 rollback of the request session
            audit(s2, body.username[:64], "login_failed", "auth")
        raise ApiError(401, "unauthorized", "invalid username or password")
    token, sess = users_repo.create_session(s, u)
    audit(s, u.username, "login", "auth")
    return {"access_token": token, "token_type": "bearer", "expires_at": iso(sess.expires_at), "user": user_out(u)}


@router.post("/auth/logout", status_code=204)
def logout(request: Request, s: Session = Depends(db), _u: User = Depends(current_user)):
    token = bearer(request)
    if token:
        users_repo.revoke(s, token)


@router.get("/auth/me")
def me(u: User = Depends(current_user)):
    return user_out(u)


@router.get("/users")
def list_users(s: Session = Depends(db), _u: User = Depends(require("admin"))):
    return [user_out(u) for u in users_repo.list_users(s)]


@router.post("/users", status_code=201)
def create_user(body: UserCreate, s: Session = Depends(db), u: User = Depends(require("admin"))):
    try:
        new = users_repo.create_user(s, body.username, body.password, body.role)
    except IntegrityError as e:
        raise ApiError(409, "conflict", "username already exists") from e
    audit(s, u.username, "user_create", f"user:{new.username}", {"role": body.role})
    return user_out(new)


@router.patch("/users/{user_id}")
def update_user(user_id: uuid.UUID, body: UserUpdate, s: Session = Depends(db), u: User = Depends(require("admin"))):
    target = users_repo.get_user(s, user_id)
    if target is None:
        raise ApiError(404, "not_found", "user not found")
    users_repo.update_user(s, target, body.role, body.enabled, body.password)
    audit(s, u.username, "user_update", f"user:{target.username}",
          {k: v for k, v in body.model_dump(exclude_none=True).items() if k != "password"})
    return user_out(target)
