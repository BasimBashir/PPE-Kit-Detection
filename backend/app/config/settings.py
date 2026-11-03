"""Configuration: secrets/paths from environment (.env), tunables from YAML (configs/*.yaml).

FR-DET-008: thresholds/class mappings are configuration, not code. Rules (thresholds + temporal policy)
are seeded from rules.yaml into the database and edited through the API afterwards.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    postgres_user: str = "ppe"
    postgres_password: SecretStr = SecretStr("")
    postgres_db: str = "ppe"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url_override: str | None = Field(default=None, alias="PPE_DATABASE_URL")

    ppe_config_dir: Path = Path("configs")
    ppe_admin_password: SecretStr | None = None
    ppe_evidence_dir: Path = Path("evidence")
    ppe_spool_dir: Path = Path("spool")
    ppe_models_dir: Path = Path("models")
    ppe_enable_fault_injection: bool = False  # enables /api/v1/system/faults (tests & demo only)

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    worker_metrics_port: int = 9101

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        pw = quote(self.postgres_password.get_secret_value(), safe="")
        return (
            f"postgresql+psycopg://{self.postgres_user}:{pw}@{self.postgres_host}:"
            f"{self.postgres_port}/{self.postgres_db}"
        )

    def resolve(self, p: Path) -> Path:
        """Relative paths are relative to the repository root."""
        return p if p.is_absolute() else REPO_ROOT / p


# ---------------------------------------------------------------- app.yaml


class PipelineConfig(BaseModel):
    target_fps: float = Field(5.0, gt=0, le=30)  # processed (inference) FPS per camera
    max_batch: int = Field(10, ge=1, le=32)
    max_batch_wait_ms: float = Field(0.0, ge=0, le=200)  # 0 = run with whatever is due
    preview_fps: float = Field(4.0, gt=0, le=15)
    preview_width: int = Field(640, ge=160, le=1920)
    preview_jpeg_quality: int = Field(70, ge=30, le=95)
    convert_fps_factor: float = Field(2.0, ge=1.0)  # ndarray conversion rate = factor * target_fps


class IngestConfig(BaseModel):
    transport: Literal["tcp", "udp"] = "tcp"
    open_timeout_s: float = 5.0
    read_timeout_s: float = 5.0
    stall_timeout_s: float = 5.0  # no frame for this long => OFFLINE (FR-CAM-004)
    degraded_frame_age_s: float = 1.5  # last frame older than this => DEGRADED
    degraded_fps_ratio: float = 0.5  # decoded fps below ratio * nominal => DEGRADED
    backoff_base_s: float = 1.0
    backoff_max_s: float = 10.0
    stable_reset_s: float = 10.0  # ONLINE this long => backoff attempt counter resets
    hwaccel: Literal["none", "cuda"] = "none"


class TrackingConfig(BaseModel):
    tracker: Literal["bytetrack"] = "bytetrack"
    track_activation_threshold: float = 0.35
    lost_track_buffer_s: float = 2.0
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 1
    track_expiry_s: float = 3.0  # FR-TRK-007: PPE state dropped after this long unseen


class AssociationConfig(BaseModel):
    head_roi: Literal["keypoints", "geometry"] = "keypoints"  # keypoints falls back to geometry
    roi_top_frac: float = 0.25  # geometry ROI = top fraction of person box
    roi_side_margin_frac: float = 0.15  # trimmed from each side of the person box
    roi_up_frac: float = 0.06  # ROI extends above the box top (helmets often clipped by person box)
    keypoint_min_conf: float = 0.5
    min_head_keypoints: int = 2
    min_ioa: float = 0.5  # intersection(helmet, roi) / area(helmet)
    max_center_dist: float = 0.6  # helmet-centre to ROI-centre distance / ROI diagonal
    helmet_width_ratio: tuple[float, float] = (0.12, 0.9)  # helmet width / person box width
    conflict_margin: float = 0.15  # |helmet_conf - head_conf| below => ambiguous
    use_classifier: bool = True  # cascade: classifier only for ambiguous cases (FR-DET-006)


class AlertsConfig(BaseModel):
    cooldown_s: float = 60.0  # FR-ALT-007 re-arm: re-confirmation within cooldown reopens the old event
    spatial_dedupe_s: float = 30.0  # merge a new track's episode into a recent event at the same place
    spatial_dedupe_iou: float = 0.3
    spatial_dedupe_head_dist: float = 0.75  # head-ROI centre distance in head widths
    last_seen_update_s: float = 5.0
    outbox_max_attempts: int = 8
    outbox_backoff_base_s: float = 0.5
    outbox_backoff_max_s: float = 60.0
    local_sound_enabled: bool = False  # FR-ALT-005 optional, off by default


class EvidenceConfig(BaseModel):
    retention_days: int = Field(30, ge=1)
    min_free_bytes: int = 1_000_000_000
    jpeg_quality: int = 85
    retention_check_interval_s: float = 3600.0


class DemoConfig(BaseModel):
    """Demo mode only: the API manages the RTSP simulator so a reviewer can pick stream counts / faults in the UI."""

    autostart_simulator: bool = True
    simulator_url: str = "http://127.0.0.1:8890"
    initial_streams: int = Field(10, ge=0, le=10)

    @field_validator("simulator_url")
    @classmethod
    def _http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("simulator_url must be an http(s) URL")
        return v.rstrip("/")


class AppConfig(BaseModel):
    mode: Literal["demo", "live"] = "demo"  # demo => UI shows "SIMULATED SOURCES" banner
    pipeline: PipelineConfig = PipelineConfig()
    ingest: IngestConfig = IngestConfig()
    tracking: TrackingConfig = TrackingConfig()
    association: AssociationConfig = AssociationConfig()
    alerts: AlertsConfig = AlertsConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    demo: DemoConfig = DemoConfig()


# ---------------------------------------------------------------- models.yaml


class ModelSpec(BaseModel):
    weights: str  # path relative to models dir, or an Ultralytics hub name (e.g. yolo26s-pose.pt)
    imgsz: int = 640
    conf: float = 0.25
    half: bool = True
    device: str = "cuda:0"  # "cpu" supported (slow)


class ModelsConfig(BaseModel):
    backend: Literal["ultralytics", "mock"] = "ultralytics"
    person: ModelSpec  # COCO detector or pose model; class 0 = person
    ppe: ModelSpec  # helmet/head detector
    ppe_class_map: dict[str, int] = {"helmet": 0, "head": 1}
    classifier: ModelSpec | None = None  # head-crop classifier (helmet / no_helmet)
    classifier_uncertain_below: float = 0.75
    cascade_skip_ppe_without_person: bool = True


# ---------------------------------------------------------------- cameras.yaml / rules.yaml


class CameraSeed(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")
    name: str
    stream_uri: str  # rtsp://host:port/path or file:///abs/path.mp4 - never contains credentials
    credential_ref: str | None = None  # "env:PREFIX" => PREFIX_USER / PREFIX_PASS env vars
    transport: Literal["tcp", "udp"] = "tcp"
    enabled: bool = True
    site: str | None = None
    location: str | None = None
    target_fps: float | None = None  # overrides pipeline.target_fps

    @field_validator("stream_uri")
    @classmethod
    def _no_userinfo(cls, v: str) -> str:
        from app.camera.uri import validate_stream_uri

        return validate_stream_uri(v)


class TemporalPolicy(BaseModel):
    """FR-TRK-004/006 and spec 5.1. Sample counts are processed-frame observations of one track."""

    mode: Literal["count", "duration", "both"] = "count"
    confirm_n: int = Field(12, ge=1)  # NO_HELMET votes needed ...
    confirm_m: int = Field(15, ge=1)  # ... within the last M decisive (non-UNCERTAIN) observations
    min_duration_s: float = Field(2.0, ge=0)  # duration mode: violation evidence sustained this long
    suspect_n: int = Field(2, ge=1)  # consecutive NO_HELMET to enter SUSPECTED
    compliant_n: int = Field(3, ge=1)  # HELMET votes in last compliant_m to become COMPLIANT
    compliant_m: int = Field(5, ge=1)
    clear_n: int = Field(8, ge=1)  # HELMET votes in last clear_m to clear a CONFIRMED episode
    clear_m: int = Field(10, ge=1)
    unknown_ratio: float = Field(0.6, ge=0, le=1)  # share of UNCERTAIN in window => UNKNOWN (pre-confirm)
    window: int = Field(30, ge=1)  # observations retained per track

    @model_validator(mode="after")
    def _check(self) -> TemporalPolicy:
        if self.confirm_n > self.confirm_m or self.compliant_n > self.compliant_m or self.clear_n > self.clear_m:
            raise ValueError("N must be <= M for confirm/compliant/clear votes")
        if self.window < max(self.confirm_m, self.clear_m, self.compliant_m):
            raise ValueError("window must be >= every M")
        return self


class RuleThresholds(BaseModel):
    person_conf: float = Field(0.4, ge=0, le=1)
    helmet_conf: float = Field(0.35, ge=0, le=1)
    head_conf: float = Field(0.35, ge=0, le=1)
    min_person_height_px: int = Field(60, ge=1)
    min_head_px: int = Field(12, ge=1)


class RuleSeed(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str
    ppe_class: Literal["hard_hat"] = "hard_hat"
    camera_id: str | None = None  # None = all cameras
    zone_id: str | None = None  # None = full frame
    severity: Literal["low", "medium", "high"] = "high"
    enabled: bool = True
    thresholds: RuleThresholds = RuleThresholds()
    temporal_policy: TemporalPolicy = TemporalPolicy()


class ZoneSeed(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    camera_id: str
    name: str
    kind: Literal["monitor", "ignore", "privacy"] = "monitor"
    polygon: list[tuple[float, float]]  # normalized 0..1 image coords, >= 3 points
    enabled: bool = True

    @field_validator("polygon")
    @classmethod
    def _poly(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(v) < 3 or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in v):
            raise ValueError("polygon needs >= 3 points with normalized coords in [0, 1]")
        return v


# ---------------------------------------------------------------- loaders


def _load_yaml(dir_: Path, name: str) -> dict:
    """configs/<name>.yaml, falling back to configs/<name>.example.yaml."""
    for fn in (f"{name}.yaml", f"{name}.example.yaml"):
        p = dir_ / fn
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raise FileNotFoundError(f"{dir_}/{name}.yaml (or .example.yaml) not found")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_app_config(settings: Settings | None = None) -> AppConfig:
    s = settings or get_settings()
    return AppConfig.model_validate(_load_yaml(s.resolve(s.ppe_config_dir), "app"))


def load_models_config(settings: Settings | None = None) -> ModelsConfig:
    s = settings or get_settings()
    return ModelsConfig.model_validate(_load_yaml(s.resolve(s.ppe_config_dir), "models"))


def load_camera_seeds(settings: Settings | None = None) -> list[CameraSeed]:
    s = settings or get_settings()
    return [CameraSeed.model_validate(c) for c in _load_yaml(s.resolve(s.ppe_config_dir), "cameras")["cameras"]]


def load_rule_seeds(settings: Settings | None = None) -> tuple[list[RuleSeed], list[ZoneSeed]]:
    s = settings or get_settings()
    data = _load_yaml(s.resolve(s.ppe_config_dir), "rules")
    return (
        [RuleSeed.model_validate(r) for r in data.get("rules", [])],
        [ZoneSeed.model_validate(z) for z in data.get("zones", [])],
    )
