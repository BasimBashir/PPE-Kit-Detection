"""Validated request bodies (trust boundary: every write goes through these)."""

from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.camera.uri import validate_credential_ref, validate_stream_uri
from app.config.settings import BEHAVIOURS, PPE_ITEMS, RuleThresholds, Schedule, TemporalPolicy


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
    kind: Literal["monitor", "ignore", "privacy", "door"] = "monitor"
    polygon: list[tuple[float, float]] = Field(min_length=2, max_length=64)
    enabled: bool = True

    @model_validator(mode="after")
    def _door_line(self):
        if self.kind == "door" and len(self.polygon) != 2:
            raise ValueError("door zones need exactly 2 points (the entry/exit line)")
        if self.kind != "door" and len(self.polygon) < 3:
            raise ValueError("non-door zones need >= 3 points")
        return self


class ZoneUpdate(Strict):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    kind: Literal["monitor", "ignore", "privacy", "door"] | None = None
    polygon: list[tuple[float, float]] | None = Field(default=None, min_length=2, max_length=64)
    enabled: bool | None = None

    @model_validator(mode="after")
    def _door_line(self):
        # only a self-consistent body passes here; the route re-validates the merged shape against ZoneSeed
        if self.kind == "door" and self.polygon is not None and len(self.polygon) != 2:
            raise ValueError("door zones need exactly 2 points (the entry/exit line)")
        if self.kind not in (None, "door") and self.polygon is not None and len(self.polygon) < 3:
            raise ValueError("non-door zones need >= 3 points")
        return self


class RulePatch(Strict):
    name: str | None = None
    camera_id: str | None = None
    zone_id: str | None = None
    severity: Literal["low", "medium", "high"] | None = None
    enabled: bool | None = None
    mandatory: bool | None = None
    thresholds: RuleThresholds | None = None
    temporal_policy: TemporalPolicy | None = None
    schedule: Schedule | None = None  # explicit null clears it (exclude_unset in the route)


PRESET_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,23}")


class PresetIn(Strict):
    """Operator preset: item -> "mandatory" | "secondary" (items left out are off). Phase 3, ADR-014."""

    items: dict[str, Literal["mandatory", "secondary"]]

    @field_validator("items")
    @classmethod
    def _known(cls, v: dict[str, str]) -> dict[str, str]:
        known = set(PPE_ITEMS) | set(BEHAVIOURS)
        unknown = set(v) - known
        if unknown or not v:
            raise ValueError(f"items must be non-empty and from {sorted(known)}; unknown: {sorted(unknown)}")
        return v


class PresetApply(Strict):
    camera_id: str | None = None  # None = all cameras
    zone_id: str | None = None  # None = full frame


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


# ---------------------------------------------------------------- attendance (Phase 5)


class PersonIn(Strict):
    name: str = Field(min_length=1, max_length=128)
    allowed_minutes_per_day: int = Field(0, ge=0, le=1440)


class PersonUpdate(Strict):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    allowed_minutes_per_day: int | None = Field(default=None, ge=0, le=1440)


class CrossingAssign(Strict):
    person_id: uuid.UUID | None  # None = back to the Unknown bucket


class EnrolIn(Strict):
    crossing_id: uuid.UUID  # enrols that crossing's stored descriptor into the person's gallery
