"""Validated request bodies (trust boundary: every write goes through these)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.camera.uri import validate_credential_ref, validate_stream_uri
from app.config.settings import RuleThresholds, TemporalPolicy


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginIn(Strict):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserCreate(Strict):
    username: str = Field(pattern=r"^[a-zA-Z0-9_.-]{3,64}$")
    password: str = Field(min_length=8, max_length=256)
    role: Literal["admin", "operator", "viewer"]


class UserUpdate(Strict):
    role: Literal["admin", "operator", "viewer"] | None = None
    enabled: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=256)


class CameraUpdate(Strict):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    stream_uri: str | None = None
    credential_ref: str | None = None
    transport: Literal["tcp", "udp"] | None = None
    enabled: bool | None = None
    site: str | None = Field(default=None, max_length=128)
    location: str | None = Field(default=None, max_length=128)
    target_fps: float | None = Field(default=None, gt=0, le=30)

    @field_validator("stream_uri")
    @classmethod
    def _uri(cls, v: str | None) -> str | None:
        return None if v is None else validate_stream_uri(v)

    @field_validator("credential_ref")
    @classmethod
    def _ref(cls, v: str | None) -> str | None:
        return validate_credential_ref(v)


class CameraCreate(CameraUpdate):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")
    name: str = Field(min_length=1, max_length=128)  # type: ignore[assignment]
    stream_uri: str  # type: ignore[assignment]


class ZoneIn(Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str = Field(min_length=1, max_length=128)
    kind: Literal["monitor", "ignore", "privacy"] = "monitor"
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=64)
    enabled: bool = True


class ZoneUpdate(Strict):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    kind: Literal["monitor", "ignore", "privacy"] | None = None
    polygon: list[tuple[float, float]] | None = Field(default=None, min_length=3, max_length=64)
    enabled: bool | None = None


class RulePatch(Strict):
    name: str | None = None
    camera_id: str | None = None
    zone_id: str | None = None
    severity: Literal["low", "medium", "high"] | None = None
    enabled: bool | None = None
    thresholds: RuleThresholds | None = None
    temporal_policy: TemporalPolicy | None = None


class NoteIn(Strict):
    note: str | None = Field(default=None, max_length=1000)


class FeedbackIn(Strict):
    label: Literal["TRUE_POSITIVE", "FALSE_POSITIVE"]
    note: str | None = Field(default=None, max_length=1000)


class SettingsPatch(Strict):
    evidence_retention_days: int | None = Field(default=None, ge=1, le=3650)


class FaultIn(Strict):
    kind: Literal["inference_exception", "gpu_oom", "crash", "disk_full", "decoder_error", "stall"]
    camera_id: str | None = None
    duration_s: float | None = Field(default=None, gt=0, le=600)
