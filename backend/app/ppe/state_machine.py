"""Per-track PPE state machine with N-of-M voting and hysteresis (FR-TRK-004..008, spec 7, ADR-006)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from app.config.settings import TemporalPolicy
from app.observability import metrics as m
from app.ppe.types import Observation, Outcome, PPEState, Transition

S = PPEState
DURATION_MIN_RATIO = 0.8  # share of decisive votes that must be NO_HELMET over the duration window


@dataclass
class TrackPPE:
    track_id: int
    window: int
    state: PPEState = S.UNKNOWN
    episode_seq: int = 0
    first_violation_at: float | None = None  # first NO_HELMET of the current episode
    confirmed_at: float | None = None
    last_seen: float = 0.0
    no_helmet_run: int = 0  # consecutive NO_HELMET (UNCERTAIN does not break the run)
    run_start: float | None = None
    last_observation: Observation | None = None
    history: deque[tuple[float, Outcome, float]] = field(default_factory=deque)

    def decisive(self, n: int) -> list[Outcome]:
        out = [o for _, o, _ in self.history if o is not Outcome.UNCERTAIN]
        return out[-n:]

    def votes(self, n: int, outcome: Outcome) -> int:
        return sum(1 for o in self.decisive(n) if o is outcome)

    def no_helmet_confidence(self, n: int) -> float:
        confs = [c for _, o, c in self.history if o is Outcome.NO_HELMET][-n:]
        return sum(confs) / len(confs) if confs else 0.0


class PPEStateMachine:
    """One per (camera, rule). Not thread-safe: owned by the inference thread."""

    def __init__(self, camera_id: str, policy: TemporalPolicy, track_expiry_s: float):
        self.camera_id = camera_id
        self.policy = policy
        self.track_expiry_s = track_expiry_s
        self.tracks: dict[int, TrackPPE] = {}

    def set_policy(self, policy: TemporalPolicy) -> None:
        self.policy = policy
        for t in self.tracks.values():
            t.window = policy.window
            while len(t.history) > policy.window:
                t.history.popleft()

    def state_of(self, track_id: int) -> PPEState:
        t = self.tracks.get(track_id)
        return t.state if t else S.UNKNOWN

    def _confirm_ok(self, t: TrackPPE, ts: float) -> bool:
        p = self.policy
        count_ok = t.votes(p.confirm_m, Outcome.NO_HELMET) >= p.confirm_n
        duration_ok = False
        if t.first_violation_at is not None and ts - t.first_violation_at >= p.min_duration_s:
            since = [o for when, o, _ in t.history if when >= t.first_violation_at and o is not Outcome.UNCERTAIN]
            duration_ok = bool(since) and since.count(Outcome.NO_HELMET) / len(since) >= DURATION_MIN_RATIO
        return {"count": count_ok, "duration": duration_ok, "both": count_ok and duration_ok}[p.mode]

    def _next_state(self, t: TrackPPE, obs: Observation, ts: float) -> tuple[PPEState, str] | None:
        p, st = self.policy, t.state
        if st is S.CONFIRMED_NON_COMPLIANT:
            if t.votes(p.clear_m, Outcome.HELMET) >= p.clear_n:
                return S.CLEARED, "compliant_evidence_sustained"
            return None  # UNCERTAIN / mixed evidence never ends a confirmed episode (no flapping)
        if st is S.SUSPECTED_NON_COMPLIANT and self._confirm_ok(t, ts):
            return S.CONFIRMED_NON_COMPLIANT, f"temporal_policy_met:{p.mode}"
        last = t.decisive(1)
        if (st is not S.COMPLIANT and last and last[0] is Outcome.HELMET
                and t.votes(p.compliant_m, Outcome.HELMET) >= p.compliant_n):
            return S.COMPLIANT, "helmet_verified"
        if (st in (S.UNKNOWN, S.COMPLIANT, S.CLEARED) and obs.outcome is Outcome.NO_HELMET
                and t.no_helmet_run >= p.suspect_n):
            return S.SUSPECTED_NON_COMPLIANT, "no_helmet_observed"
        if st is S.SUSPECTED_NON_COMPLIANT and obs.outcome is Outcome.UNCERTAIN:
            recent = list(t.history)[-p.confirm_m:]
            unc = sum(1 for _, o, _ in recent if o is Outcome.UNCERTAIN)
            if recent and unc / len(recent) >= p.unknown_ratio:
                return S.UNKNOWN, "insufficient_visibility"
        return None

    def _apply(self, t: TrackPPE, new: PPEState, reason: str, ts: float) -> Transition:
        prev = t.state
        # A visibility dip (SUSPECTED -> UNKNOWN -> SUSPECTED) continues the same episode; anything else starts one.
        if new is S.SUSPECTED_NON_COMPLIANT and (prev is not S.UNKNOWN or t.first_violation_at is None):
            t.episode_seq += 1
            t.first_violation_at = t.run_start if t.run_start is not None else ts
        if new is S.CONFIRMED_NON_COMPLIANT:
            t.confirmed_at = ts
        if new in (S.CLEARED, S.COMPLIANT):
            t.first_violation_at = None
            t.confirmed_at = None
        t.state = new
        m.ppe_state_transitions_total.labels(prev.value, new.value).inc()
        return Transition(self.camera_id, t.track_id, prev, new, ts, reason, t.episode_seq)

    def update(self, observations: list[Observation], ts: float) -> list[Transition]:
        out: list[Transition] = []
        for obs in observations:
            t = self.tracks.get(obs.track_id)
            if t is None:
                t = self.tracks[obs.track_id] = TrackPPE(obs.track_id, self.policy.window)
            t.last_seen = ts
            t.last_observation = obs
            t.history.append((ts, obs.outcome, obs.confidence))
            while len(t.history) > t.window:
                t.history.popleft()
            if obs.outcome is Outcome.NO_HELMET:
                if t.no_helmet_run == 0:
                    t.run_start = ts
                t.no_helmet_run += 1
            elif obs.outcome is Outcome.HELMET:
                t.no_helmet_run, t.run_start = 0, None
            nxt = self._next_state(t, obs, ts)
            if nxt:
                out.append(self._apply(t, nxt[0], nxt[1], ts))
        return out

    def expire(self, ts: float) -> list[Transition]:
        out: list[Transition] = []
        for tid in [k for k, t in self.tracks.items() if ts - t.last_seen > self.track_expiry_s]:
            t = self.tracks.pop(tid)
            if t.state is S.CONFIRMED_NON_COMPLIANT:
                m.ppe_state_transitions_total.labels(t.state.value, S.CLEARED.value).inc()
                out.append(Transition(self.camera_id, tid, t.state, S.CLEARED, ts, "track_lost", t.episode_seq))
        return out
