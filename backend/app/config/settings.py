"""Configuration: secrets/paths from environment (.env), tunables from YAML (configs/*.yaml).

FR-DET-008: thresholds/class mappings are configuration, not code. Rules (thresholds + temporal policy)
are seeded from rules.yaml into the database and edited through the API afterwards.
"""

from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore", populate_by_name=True)

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
    # Network sharding (ADR-014): "i/N" = this deployment handles cameras with sorted-index % N == i. Off when unset.
    worker_shard: str | None = Field(default=None, alias="PPE_WORKER_SHARD")
    ppe_enable_fault_injection: bool = False  # enables /api/v1/system/faults (tests & demo only)
    gemini_api_key: SecretStr | None = None  # teacher review (provider gemini); registered for log redaction

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

    @property
    def shard(self) -> tuple[int, int] | None:
        """Parsed PPE_WORKER_SHARD "i/N"; None = off (this deployment handles every camera)."""
        if not self.worker_shard:
            return None
        i, _, n = self.worker_shard.partition("/")
        if not i.isdigit() or not n.isdigit() or not (0 <= int(i) < int(n) <= 64):
            raise ValueError(f"PPE_WORKER_SHARD must be 'i/N' with 0 <= i < N (got {self.worker_shard!r})")
        return int(i), int(n)


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


class TeacherConfig(BaseModel):
    """Asynchronous teacher review of head crops (app/inference/teacher.py). Off by default; never on the alert
    path. Probabilities are P(no helmet); fused = weight * p_teacher + (1 - weight) * p_model."""

    enabled: bool = False
    provider: Literal["gemini", "systemone", "fake"] = "gemini"
    model: str = "gemini-decision-1"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    weight: float = Field(0.5, ge=0, le=1)
    escalate_threshold: float = Field(0.8, ge=0, le=1)  # compliant sample with fused above => new visible event
    false_alert_threshold: float = Field(0.2, ge=0, le=1)  # alert with fused below => tagged for human review
    compliant_sample_rate: float = Field(0.05, ge=0, le=1)  # share of tracks turning COMPLIANT sent for review
    max_reviews_per_minute: int = Field(30, ge=1)
    timeout_s: float = Field(30.0, gt=0)  # live gemini-decision-1 p95 is ~8 s; 5 s truncated real calls


class BehaviourConfig(BaseModel):
    """Phase 4 behaviour checks (ADR-015). Engineering constants; the operator knobs (hold time = the rule's
    min_duration_s, shift schedule, evidence confidence = the rule's absent_conf) live on each rule (UI)."""

    keypoint_min_conf: float = Field(0.4, ge=0, le=1)
    # fall: torso upright -> lying within fall_window_s while the hips drop >= drop_frac of the standing height
    fall_window_s: float = Field(2.0, gt=0)
    upright_max_deg: float = Field(35.0, ge=0, le=90)  # torso angle from vertical
    lying_min_deg: float = Field(60.0, ge=0, le=90)
    drop_frac: float = Field(0.25, gt=0, le=1)
    # stalling: every keypoint stays within idle_max_range body heights (5th-95th percentile) over idle_window_s
    idle_window_s: float = Field(10.0, gt=0)
    idle_max_range: float = Field(0.08, gt=0)
    # phone: a phone box over a (grown) hand or head ROI; detector dropouts shorter than phone_gap_s are bridged
    phone_roi_scale: float = Field(1.8, ge=1)
    phone_min_ioa: float = Field(0.3, gt=0, le=1)
    phone_gap_s: float = Field(2.0, ge=0)
    # fight: two people closer than fight_pair_dist mean body heights form a pair scored by the fight model
    fight_pair_dist: float = Field(1.2, gt=0)


class AttendanceConfig(BaseModel):
    """Phase 5 door crossings (ADR-016). Thresholds for a deliberately cheap, non-biometric identity: an HSV torso
    histogram matched against the operator-enrolled gallery. Wrong matches are corrected by re-assigning the
    crossing in the UI; unmatched crossings stay in the Unknown bucket."""

    match_threshold: float = Field(0.55, ge=0, le=1)  # cosine similarity; below this the crossing stays Unknown
    gallery_max: int = Field(20, ge=1)  # descriptors kept per person
    line_deadband: float = Field(0.004, gt=0)  # normalised distance from the door line before a side counts
    site_tz: str = "UTC"  # IANA zone used to cut report days; sites with shifts should set this


class AppConfig(BaseModel):
    mode: Literal["demo", "live"] = "demo"  # demo => UI shows "SIMULATED SOURCES" banner
    pipeline: PipelineConfig = PipelineConfig()
    ingest: IngestConfig = IngestConfig()
    tracking: TrackingConfig = TrackingConfig()
    association: AssociationConfig = AssociationConfig()
    alerts: AlertsConfig = AlertsConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    demo: DemoConfig = DemoConfig()
    teacher: TeacherConfig = TeacherConfig()
    behaviour: BehaviourConfig = BehaviourConfig()
    attendance: AttendanceConfig = AttendanceConfig()


# ---------------------------------------------------------------- models.yaml


class PPEItem(BaseModel):
    """One PPE item the system can enforce (Phase 3, ADR-014). Adding a class = an entry in models.yaml `ppe_items`
    + a detector trained with its `present` / `absent` classes (mapped in `items_class_map`) - no code change."""

    present: str  # detector class of the item (e.g. "gloves")
    absent: str  # detector class of the bare body region (e.g. "no_gloves"; "head" for hard hats)
    region: Literal["head", "eyes", "torso", "hands", "feet"]  # where the item must be (pose keypoints)
    width_ratio: tuple[float, float] | None = None  # detection width / person width; None = helmet_width_ratio
    label: str  # operator-facing name, e.g. "Gloves"
    alert: str  # evidence / overlay text, e.g. "NO GLOVES"
    teacher_question: str  # "Is the person NOT wearing <this>?" for the teacher (Phase 2)


DEFAULT_PPE_ITEMS: dict[str, PPEItem] = {
    "hard_hat": PPEItem(present="helmet", absent="head", region="head", label="Hard hat", alert="NO HARD HAT",
                        teacher_question="a hard hat / safety helmet "
                                         "(bare head, cap, hood or hair only counts as NOT)"),
    "goggles": PPEItem(present="goggles", absent="no_goggles", region="eyes", width_ratio=(0.05, 0.7),
                       label="Goggles", alert="NO GOGGLES",
                       teacher_question="safety goggles or safety glasses over the eyes"),
    "safety_vest": PPEItem(present="vest", absent="no_vest", region="torso", width_ratio=(0.3, 1.4),
                           label="Safety vest", alert="NO SAFETY VEST",
                           teacher_question="a high-visibility safety vest"),
    "gloves": PPEItem(present="gloves", absent="no_gloves", region="hands", width_ratio=(0.03, 0.6), label="Gloves",
                      alert="NO GLOVES", teacher_question="a protective work glove"),
    "safety_shoes": PPEItem(present="boots", absent="no_boots", region="feet", width_ratio=(0.04, 0.8),
                            label="Safety shoes", alert="NO SAFETY SHOES",
                            teacher_question="a safety boot / protective work shoe "
                                             "(sandals, bare feet or trainers count as NOT)"),
}
# Process-wide catalogue: defaults, replaced by models.yaml `ppe_items` in load_models_config() (API and worker).
PPE_ITEMS: dict[str, PPEItem] = dict(DEFAULT_PPE_ITEMS)


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
    # Phase 3 second detector: vest / gloves / goggles / boots, each with its "no_*" class (absent = bare region)
    items: ModelSpec | None = None
    items_class_map: dict[str, int] = {"vest": 2, "no_vest": 3, "gloves": 4, "no_gloves": 5, "goggles": 6,
                                       "no_goggles": 7, "boots": 8, "no_boots": 9}
    ppe_items: dict[str, PPEItem] = DEFAULT_PPE_ITEMS  # the enforceable item catalogue (rules, UI, teacher)

    @field_validator("ppe_items")
    @classmethod
    def _merge_items(cls, v: dict[str, PPEItem]) -> dict[str, PPEItem]:
        return {**DEFAULT_PPE_ITEMS, **v}  # models.yaml adds or overrides items; the five built-ins stay available
    classifier: ModelSpec | None = None  # head-crop classifier (helmet / no_helmet)
    classifier_uncertain_below: float = 0.75
    cascade_skip_ppe_without_person: bool = True

    # Phase 4 (ADR-015): COCO detector for the phone check ("cell phone" = COCO class 67); None = phone rules
    # stay UNCERTAIN. fight_weights: models/<file> from scripts/ml/train_fight.py; None = fight rules UNCERTAIN.
    phone: ModelSpec | None = None
    phone_class_map: dict[str, int] = {"phone": 67}
    fight_weights: str | None = None


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
    """Per-rule gates. present/absent = the rule's item detected / its body region detected bare (Phase 3);
    the Phase 1 names (helmet_conf, head_conf, min_head_px) are still accepted on input."""

    person_conf: float = Field(0.4, ge=0, le=1)
    present_conf: float = Field(0.35, ge=0, le=1, validation_alias=AliasChoices("present_conf", "helmet_conf"))
    absent_conf: float = Field(0.35, ge=0, le=1, validation_alias=AliasChoices("absent_conf", "head_conf"))
    min_person_height_px: int = Field(60, ge=1)
    min_roi_px: int = Field(12, ge=1, validation_alias=AliasChoices("min_roi_px", "min_head_px"))


class Behaviour(BaseModel):
    """Phase 4 (ADR-015): a per-person behaviour a rule can enforce, like a PPE item. Its assessor emits ABSENT =
    the behaviour is happening (violation evidence), PRESENT = normal, UNCERTAIN = not assessable, so the PPE state
    machine, events, dedupe and alerts apply unchanged."""

    label: str
    alert: str
    policy: TemporalPolicy  # default temporal policy for new rules of this check (editable per rule)
    thresholds: RuleThresholds = RuleThresholds()


BEHAVIOURS: dict[str, Behaviour] = {
    # "down after a fall" held min_duration_s; the fall itself is the temporal sequence in FallAssessor
    "fall": Behaviour(label="Fall", alert="PERSON DOWN", policy=TemporalPolicy(
        mode="duration", min_duration_s=3.0, suspect_n=1, confirm_n=3, confirm_m=5, compliant_n=3, compliant_m=5,
        clear_n=5, clear_m=8, unknown_ratio=0.8, window=30)),
    "fight": Behaviour(label="Fighting", alert="FIGHT", thresholds=RuleThresholds(absent_conf=0.7),
                       policy=TemporalPolicy(mode="count", confirm_n=3, confirm_m=5, suspect_n=2, compliant_n=3,
                                             compliant_m=5, clear_n=6, clear_m=8, window=30)),
    "phone": Behaviour(label="Phone use", alert="PHONE USE", policy=TemporalPolicy(
        mode="duration", min_duration_s=10.0, suspect_n=2, compliant_n=4, compliant_m=6, clear_n=6, clear_m=8,
        unknown_ratio=0.8, window=30)),
    # measures stillness, not intent: scope it with a work-area zone and a shift schedule
    "stalling": Behaviour(label="Stalling work", alert="STALLING", policy=TemporalPolicy(
        mode="duration", min_duration_s=600.0, suspect_n=3, compliant_n=5, compliant_m=8, clear_n=5, clear_m=8,
        unknown_ratio=0.8, window=30)),
}


def check_meta(key: str) -> PPEItem | Behaviour:
    """A rule's check - PPE item or behaviour; both carry .label and .alert."""
    return PPE_ITEMS[key] if key in PPE_ITEMS else BEHAVIOURS[key]


_WINDOW = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")


def _minutes(window: str) -> tuple[int, int]:
    h1, m1, h2, m2 = map(int, _WINDOW.match(window).groups())  # type: ignore[union-attr]
    return h1 * 60 + m1, h2 * 60 + m2


class Schedule(BaseModel):
    """When a rule is enforced (Phase 4): shift windows in site-local time, breaks left out. A window whose end is
    before its start runs past midnight and belongs to the day it starts. End is exclusive."""

    tz: str
    days: list[int] = Field(min_length=1, max_length=7)  # 0 = Monday ... 6 = Sunday
    windows: list[str] = Field(min_length=1, max_length=12)  # "HH:MM-HH:MM"

    @field_validator("tz")
    @classmethod
    def _tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(f"unknown time zone {v!r} (IANA name, e.g. Asia/Dubai)") from e
        return v

    @field_validator("days")
    @classmethod
    def _days(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v) or any(not 0 <= d <= 6 for d in v):
            raise ValueError("days: unique integers 0 (Monday) .. 6 (Sunday)")
        return v

    @field_validator("windows")
    @classmethod
    def _windows(cls, v: list[str]) -> list[str]:
        for w in v:
            if not _WINDOW.match(w) or _minutes(w)[0] == _minutes(w)[1]:
                raise ValueError(f"window {w!r}: use HH:MM-HH:MM with start != end")
        return v

    def active(self, wall: float) -> bool:
        now = datetime.fromtimestamp(wall, ZoneInfo(self.tz))
        t, day = now.hour * 60 + now.minute, now.weekday()
        for w in self.windows:
            s, e = _minutes(w)
            if s < e and day in self.days and s <= t < e:
                return True
            if s > e and ((day in self.days and t >= s) or ((day - 1) % 7 in self.days and t < e)):
                return True
        return False


class RuleSeed(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str
    ppe_class: str = "hard_hat"  # the check: a PPE item (PPE_ITEMS) or a behaviour (BEHAVIOURS); name kept for API/DB
    camera_id: str | None = None  # None = all cameras
    zone_id: str | None = None  # None = full frame
    severity: Literal["low", "medium", "high"] = "high"
    enabled: bool = True
    mandatory: bool = True  # False = secondary: violations are recorded and listed, but no alert is delivered
    thresholds: RuleThresholds = RuleThresholds()
    temporal_policy: TemporalPolicy = TemporalPolicy()
    schedule: Schedule | None = None  # None = always enforced (Phase 4: shift hours, breaks left out)

    @model_validator(mode="before")
    @classmethod
    def _behaviour_defaults(cls, data: Any) -> Any:
        """New behaviour rules start from that behaviour's tuned policy/thresholds, not the PPE defaults."""
        if isinstance(data, dict) and data.get("ppe_class") in BEHAVIOURS:
            b = BEHAVIOURS[data["ppe_class"]]
            data = {**data}
            if data.get("temporal_policy") is None:
                data["temporal_policy"] = b.policy.model_dump()
            if data.get("thresholds") is None:
                data["thresholds"] = b.thresholds.model_dump()
        return data

    @field_validator("ppe_class")
    @classmethod
    def _known_check(cls, v: str) -> str:
        if v not in PPE_ITEMS and v not in BEHAVIOURS:
            raise ValueError(f"unknown check {v!r}; configured: {', '.join([*PPE_ITEMS, *BEHAVIOURS])}")
        return v


class ZoneSeed(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    camera_id: str
    name: str
    kind: Literal["monitor", "ignore", "privacy", "door"] = "monitor"
    polygon: list[tuple[float, float]]  # normalized 0..1 image coords; door = exactly 2 (line), else >= 3
    enabled: bool = True

    @model_validator(mode="after")
    def _poly(self):
        n = len(self.polygon)
        if (self.kind == "door" and n != 2) or (self.kind != "door" and n < 3):
            raise ValueError("door zones need exactly 2 points; other kinds need >= 3")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.polygon):
            raise ValueError("polygon needs normalized coords in [0, 1]")
        return self


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
    cfg = ModelsConfig.model_validate(_load_yaml(s.resolve(s.ppe_config_dir), "models"))
    PPE_ITEMS.clear()
    PPE_ITEMS.update(cfg.ppe_items)
    return cfg


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
