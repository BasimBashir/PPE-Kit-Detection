"""Fighting (Phase 4, ADR-015): a small GRU scores the keypoint sequences of two nearby tracked people.

pair_features / PairSequencer are used by BOTH this assessor and scripts/ml/extract_fight_sequences.py, so training
and production features cannot drift. Sequences are resampled to the checkpoint's rate (hz), so the model does not
depend on the camera's processed FPS - which must be >= hz (else pairs stay 'warming_up' = UNCERTAIN)."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.behaviour.assessors import BehaviourAssessor, _obs, gate
from app.config.settings import BehaviourConfig, RuleThresholds
from app.inference.types import Detection
from app.inference.ultralytics_detector import weights_version
from app.ppe.types import Box, Observation, Outcome, TrackedPerson
from app.streaming.frame import Frame

SEQ_LEN, HZ = 16, 5.0  # training defaults (3 s of motion at 5 Hz); checkpoints store the values used
N_FEAT = 102  # 2 people x 17 keypoints x (x, y, conf)


def pair_features(ka: np.ndarray | None, kb: np.ndarray | None, box_a: Box, box_b: Box,
                  min_conf: float = 0.3) -> np.ndarray:
    """Keypoints of two people -> (102,): x, y relative to the pair centre in mean body heights, plus confidence.
    Left person first (order-invariant); points below min_conf are zeroed."""
    if box_a[0] + box_a[2] > box_b[0] + box_b[2]:
        ka, kb, box_a, box_b = kb, ka, box_b, box_a
    h = max(((box_a[3] - box_a[1]) + (box_b[3] - box_b[1])) / 2, 1e-6)
    cx = (box_a[0] + box_a[2] + box_b[0] + box_b[2]) / 4
    cy = (box_a[1] + box_a[3] + box_b[1] + box_b[3]) / 4
    out = np.zeros((34, 3), np.float32)
    for i, k in enumerate((ka, kb)):
        if k is None:
            continue
        ok = k[:, 2] >= min_conf
        seg = out[17 * i:17 * i + 17]
        seg[ok, 0] = (k[ok, 0] - cx) / h
        seg[ok, 1] = (k[ok, 1] - cy) / h
        seg[ok, 2] = k[ok, 2]
    return out.reshape(-1)


def _nearest(h: deque, t: float, tol: float):
    best = min(h, key=lambda s: abs(s[0] - t), default=None)
    return best if best is not None and abs(best[0] - t) <= tol else None


class PairSequencer:
    """Per-camera keypoint history -> fixed-rate pair sequences for people closer than pair_dist body heights."""

    def __init__(self, seq_len: int, hz: float, pair_dist: float):
        self.seq_len, self.hz, self.pair_dist = seq_len, hz, pair_dist
        self.hist: dict[int, deque[tuple[float, Box, np.ndarray | None]]] = {}

    @property
    def span(self) -> float:
        return (self.seq_len - 1) / self.hz

    def forget(self, track_id: int) -> None:
        self.hist.pop(track_id, None)

    def _sequence(self, ta: int, tb: int, now: float) -> np.ndarray | None:
        tol, out = 0.6 / self.hz, []
        for i in range(self.seq_len):
            t = now - (self.seq_len - 1 - i) / self.hz
            sa, sb = _nearest(self.hist[ta], t, tol), _nearest(self.hist[tb], t, tol)
            if sa is None or sb is None:
                return None
            out.append(pair_features(sa[2], sb[2], sa[1], sb[1]))
        return np.stack(out)

    def update(self, persons: list[TrackedPerson], now: float,
               eligible: set[int] | None = None) -> tuple[set[int], list[tuple[tuple[int, int], np.ndarray]]]:
        """Record persons; return (track ids with a nearby person, scorable (pair, sequence) items).
        Nearness counts every person; only pairs where both ids are in `eligible` (default: all) are recorded/scored."""
        ok = {p.track_id for p in persons} if eligible is None else eligible
        for p in persons:
            if p.track_id not in ok:
                continue
            h = self.hist.setdefault(p.track_id, deque())
            h.append((now, p.box, p.keypoints))
            while now - h[0][0] > self.span + 1.0:
                h.popleft()
        near: set[int] = set()
        out = []
        for i, a in enumerate(persons):
            for b in persons[i + 1:]:
                mean_h = max(((a.box[3] - a.box[1]) + (b.box[3] - b.box[1])) / 2, 1e-6)
                d = math.hypot((a.box[0] + a.box[2] - b.box[0] - b.box[2]) / 2,
                               (a.box[1] + a.box[3] - b.box[1] - b.box[3]) / 2)
                if d > self.pair_dist * mean_h:
                    continue
                near |= {a.track_id, b.track_id}
                if a.track_id not in ok or b.track_id not in ok:
                    continue
                seq = self._sequence(a.track_id, b.track_id, now)
                if seq is not None:
                    out.append(((a.track_id, b.track_id), seq))
        return near, out


class FightGRU(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(N_FEAT, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, 102) -> (B,) logits
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class FightModel:
    """Checkpoint on CPU: a GRU over 16 x 102 floats costs microseconds; no GPU sync on the inference thread."""

    def __init__(self, path: Path):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.seq_len, self.hz = int(ck["seq_len"]), float(ck["hz"])
        self.net = FightGRU(int(ck["hidden"]))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.version = weights_version(Path(path))

    @torch.no_grad()
    def score(self, x: np.ndarray) -> np.ndarray:
        return torch.sigmoid(self.net(torch.from_numpy(np.ascontiguousarray(x, np.float32)))).numpy()


class FightAssessor(BehaviourAssessor):
    """Per track: max fight probability over its nearby pairs >= the rule's absent_conf -> ABSENT. Alone -> PRESENT
    (one person cannot fight). Partner present but sequence not yet full -> UNCERTAIN. No model -> UNCERTAIN."""

    def __init__(self, cfg: BehaviourConfig, thresholds: RuleThresholds, model: FightModel | None):
        super().__init__(cfg, thresholds)
        self.model = model
        self.seq = PairSequencer(model.seq_len, model.hz, cfg.fight_pair_dist) if model else None

    def _drop(self, track_id: int) -> None:
        if self.seq:
            self.seq.forget(track_id)

    def assess_deferred(self, persons: list[TrackedPerson], dets: list[Detection], frame: Frame) -> list[Observation]:
        now = frame.captured_at
        self._forget(now)
        for p in persons:
            self.seen[p.track_id] = now
        if self.model is None or self.seq is None:
            return [_obs(p, Outcome.UNCERTAIN, 0.0, "fight_model_missing") for p in persons]
        # nearness over ALL persons: a gated-out partner (conf drops while grappling) means "can't tell", not "alone"
        near, items = self.seq.update(persons, now, {p.track_id for p in persons if gate(p, self.t) is None})
        probs = self.model.score(np.stack([s for _, s in items])) if items else []
        best: dict[int, float] = {}
        for ((ta, tb), _), pr in zip(items, probs, strict=True):
            best[ta], best[tb] = max(best.get(ta, 0.0), float(pr)), max(best.get(tb, 0.0), float(pr))
        out = []
        for p in persons:
            reason = gate(p, self.t)
            if reason:
                out.append(_obs(p, Outcome.UNCERTAIN, 0.0, f"not_assessable:{reason}"))
            elif p.track_id in best:
                pr = best[p.track_id]
                out.append(_obs(p, Outcome.ABSENT, pr, "fight_score") if pr >= self.t.absent_conf
                           else _obs(p, Outcome.PRESENT, 1.0 - pr, "pair_normal"))
            elif p.track_id in near:
                out.append(_obs(p, Outcome.UNCERTAIN, 0.0, "warming_up"))
            else:
                out.append(_obs(p, Outcome.PRESENT, 1.0, "no_one_nearby"))
        return out
