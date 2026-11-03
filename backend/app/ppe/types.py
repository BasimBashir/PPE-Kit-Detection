"""PPE assessment contracts (FR-DET-003..007, FR-TRK-004..008)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 pixels


class Outcome(StrEnum):
    """Per-frame evidence for one tracked person."""

    HELMET = "HELMET"  # helmet verified on the head ROI
    NO_HELMET = "NO_HELMET"  # head visible, no helmet on it
    UNCERTAIN = "UNCERTAIN"  # head not assessable (occluded, too small, truncated, conflicting)


class PPEState(StrEnum):
    UNKNOWN = "UNKNOWN"
    COMPLIANT = "COMPLIANT"
    SUSPECTED_NON_COMPLIANT = "SUSPECTED_NON_COMPLIANT"
    CONFIRMED_NON_COMPLIANT = "CONFIRMED_NON_COMPLIANT"
    CLEARED = "CLEARED"


@dataclass(slots=True)
class TrackedPerson:
    track_id: int  # tracker-local id; unique only within (camera, worker run)
    box: Box
    conf: float
    keypoints: object | None = None  # np.ndarray (17,3) when a pose model is used


@dataclass(slots=True)
class Observation:
    track_id: int
    outcome: Outcome
    confidence: float  # confidence of the evidence supporting `outcome` (0..1)
    reason: str  # short machine-friendly reason, e.g. "helmet_on_head_roi", "head_too_small"
    person_box: Box
    head_roi: Box | None
    helmet_box: Box | None = None
    head_box: Box | None = None
    roi_source: str = "geometry"  # "keypoints" | "geometry"
    classifier_used: bool = False
    zone_id: str | None = None


@dataclass(slots=True)
class Transition:
    camera_id: str
    track_id: int
    prev: PPEState
    new: PPEState
    ts: float  # wall clock of the observation's frame
    reason: str
    episode_seq: int  # increments each time the track enters SUSPECTED from a non-violation state
