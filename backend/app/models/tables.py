"""SQLAlchemy 2 ORM models (implementation-plan 4.10, ADR-007). No identity / biometric fields exist by design."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


TS = DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(16))  # admin | operator | viewer
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class Session(Base):
    __tablename__ = "sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(TS, index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(TS)


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    stream_uri: Mapped[str] = mapped_column(Text)  # validated credential-free
    credential_ref: Mapped[str | None] = mapped_column(String(64))
    transport: Mapped[str] = mapped_column(String(8), default="tcp")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    site: Mapped[str | None] = mapped_column(String(128))
    location: Mapped[str | None] = mapped_column(String(128))
    target_fps: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow, onupdate=utcnow)


class CameraHealthRecord(Base):
    __tablename__ = "camera_health"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    state: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text)
    recorded_at: Mapped[datetime] = mapped_column(TS, default=utcnow, index=True)


class Zone(Base):
    __tablename__ = "zones"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(16), default="monitor")
    polygon: Mapped[list[Any]] = mapped_column(JSONB)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class PPERule(Base):
    __tablename__ = "ppe_rules"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    ppe_class: Mapped[str] = mapped_column(String(32), default="hard_hat")
    camera_id: Mapped[str | None] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    zone_id: Mapped[str | None] = mapped_column(ForeignKey("zones.id", ondelete="SET NULL"))
    severity: Mapped[str] = mapped_column(String(16), default="high")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    thresholds: Mapped[dict[str, Any]] = mapped_column(JSONB)
    temporal_policy: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow, onupdate=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(160), primary_key=True)  # "<name>@<sha12>"
    model_name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(64))
    artifact_hash: Mapped[str] = mapped_column(String(64))
    license: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    first_seen_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[int] = mapped_column(Integer)
    zone_id: Mapped[str | None] = mapped_column(String(64))
    rule_id: Mapped[str] = mapped_column(ForeignKey("ppe_rules.id"), index=True)
    episode_id: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(16), default="OPEN", index=True)
    severity: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text)
    confidence_summary: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    first_seen_at: Mapped[datetime] = mapped_column(TS)
    confirmed_at: Mapped[datetime] = mapped_column(TS, index=True)
    frame_captured_at: Mapped[datetime] = mapped_column(TS)
    persisted_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(TS)
    episode_ended_at: Mapped[datetime | None] = mapped_column(TS)
    reconfirm_count: Mapped[int] = mapped_column(Integer, default=0)
    last_box: Mapped[list[float] | None] = mapped_column(JSONB)
    evidence_path: Mapped[str | None] = mapped_column(Text)
    evidence_error: Mapped[str | None] = mapped_column(String(32))
    model_versions: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    feedback: Mapped[str | None] = mapped_column(String(16))
    feedback_note: Mapped[str | None] = mapped_column(Text)
    acknowledged_by: Mapped[str | None] = mapped_column(String(64))
    acknowledged_at: Mapped[datetime | None] = mapped_column(TS)
    resolved_by: Mapped[str | None] = mapped_column(String(64))
    resolved_at: Mapped[datetime | None] = mapped_column(TS)
    dashboard_rendered_at: Mapped[datetime | None] = mapped_column(TS)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow, onupdate=utcnow)

    __table_args__ = (Index("ix_events_camera_confirmed", "camera_id", "confirmed_at"),)


class AlertOutbox(Base):
    __tablename__ = "alert_outbox"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    channel: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="PENDING", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(TS, default=utcnow, index=True)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    delivered_at: Mapped[datetime | None] = mapped_column(TS)

    __table_args__ = (UniqueConstraint("event_id", "channel", name="uq_outbox_event_channel"),)


class EventAction(Base):
    __tablename__ = "event_actions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True)
    actor_id: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(16))  # ACKNOWLEDGE | RESOLVE | FEEDBACK | REOPEN
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor_id: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    resource: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow, index=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)  # never secrets


class SystemMetricsSnapshot(Base):
    __tablename__ = "system_metrics_snapshots"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(TS, default=utcnow, index=True)
    metric_name: Mapped[str] = mapped_column(String(128))
    labels: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    value: Mapped[float] = mapped_column(Float)


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow, onupdate=utcnow)
