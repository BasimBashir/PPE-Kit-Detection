"""Pair keypoint sequences for the fight model (Phase 4, ADR-015), built with the PRODUCTION code:
pose model -> ByteTrack -> person gate -> PairSequencer (app/behaviour/fight.py). One label per clip.

  python scripts/ml/extract_fight_sequences.py --root data/fight/rlvs --pos Violence --neg NonViolence
      --out data/fight/rlvs.npz
  python scripts/ml/extract_fight_sequences.py --root data/fight/scfd --pos fight --neg noFight
      --out data/fight/scfd.npz
--fps is the simulated camera processing rate (>= the model's 5 Hz); sequences are kept every --stride seconds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".mpg"}


def clip_sequences(path: Path, pose, args) -> list[np.ndarray]:
    import cv2
    from app.behaviour.assessors import gate
    from app.behaviour.fight import HZ, SEQ_LEN, PairSequencer
    from app.config.settings import BehaviourConfig, RuleThresholds, TrackingConfig
    from app.inference.types import Detection
    from app.streaming.frame import Frame
    from app.tracking.tracker import make_tracker

    th, tracker = RuleThresholds(), make_tracker(TrackingConfig(), args.fps)
    seq = PairSequencer(SEQ_LEN, HZ, BehaviourConfig().fight_pair_dist)
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, last, i, next_t = [], {}, 0, 0.0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t, i = i / vfps, i + 1
        if t + 1e-6 < next_t:
            continue
        next_t += 1.0 / args.fps
        r = pose.predict(img, imgsz=640, conf=0.3, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes) and r.keypoints is not None:
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.keypoints.data.cpu().numpy(),
                               strict=True):
                dets.append(Detection("person", float(c), tuple(map(float, b)), "x", k))
        persons = [p for p in tracker.update(dets, Frame("x", i, img, None, t, t)) if gate(p, th) is None]
        for pair, s in seq.update(persons, t)[1]:
            if t - last.get(pair, -1e9) >= args.stride:
                last[pair] = t
                out.append(s)
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--stride", type=float, default=0.5)
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo26s-pose.pt"))
    args = ap.parse_args()
    from app.behaviour.fight import HZ, N_FEAT, SEQ_LEN
    from ultralytics import YOLO

    pose = YOLO(args.weights)
    xs, idx, names, labels = [], [], [], []
    for label, sub in ((1, args.pos), (0, args.neg)):
        for p in sorted(q for q in (args.root / sub).rglob("*") if q.suffix.lower() in VIDEO_EXT):
            seqs = clip_sequences(p, pose, args)
            names.append(str(p.relative_to(args.root)))
            labels.append(label)
            xs += seqs
            idx += [len(names) - 1] * len(seqs)
            if len(names) % 100 == 0:
                print(f"{len(names)} clips, {len(xs)} sequences", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    X = np.stack(xs).astype(np.float32) if xs else np.zeros((0, SEQ_LEN, N_FEAT), np.float32)
    np.savez_compressed(args.out, X=X, clip_idx=np.array(idx, np.int32), clip_names=np.array(names),
                        clip_labels=np.array(labels, np.int8), seq_len=SEQ_LEN, hz=HZ)
    print(f"{len(names)} clips ({sum(labels)} fight), {len(xs)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
