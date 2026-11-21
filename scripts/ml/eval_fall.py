"""Event-level fall evaluation (Phase 4, ADR-015): pose model + ByteTrack + FallAssessor + the fall rule's state
machine run offline on labelled videos - the production code path minus RTSP and the database.

  python scripts/ml/eval_fall.py --labels data/fall/of_itw_test.csv --fps 5 --report docs/ml/fall_eval_itw_fps5.json
Labels CSV: video,start,end,label (seconds). Rows labelled 'fall' are fall events; every video in the CSV is
processed. A fall counts as detected when the rule CONFIRMS within [start, end + hold + 5 s]; every other
confirmation is a false alarm (also per hour of processed footage). 'missed_seen_lying_without_event' counts missed
falls where a lying person was seen without a fall event (ID switch mid-fall, or a fall slower than fall_window_s).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def run_video(path: str, fps: float, pose, args):
    import cv2
    from app.behaviour.assessors import FallAssessor
    from app.config.settings import BEHAVIOURS, BehaviourConfig, RuleThresholds, TrackingConfig
    from app.inference.types import Detection
    from app.ppe.state_machine import PPEStateMachine
    from app.ppe.types import PPEState
    from app.streaming.frame import Frame
    from app.tracking.tracker import make_tracker

    tcfg = TrackingConfig()
    tracker, fall = make_tracker(tcfg, fps), FallAssessor(BehaviourConfig(), RuleThresholds())
    fsm = PPEStateMachine("eval", BEHAVIOURS["fall"].policy, tcfg.track_expiry_s)
    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    confirms, lying, i, next_t, t = [], [], 0, 0.0, 0.0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t, i = i / vfps, i + 1
        if t + 1e-6 < next_t:
            continue
        next_t += 1.0 / fps
        r = pose.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes) and r.keypoints is not None:
            for b, c, k in zip(
                r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.keypoints.data.cpu().numpy(), strict=True
            ):
                dets.append(Detection("person", float(c), tuple(map(float, b)), "eval", k))
        frame = Frame("eval", i, img, None, t, t)
        obs = fall.assess_deferred(tracker.update(dets, frame), [], frame)
        lying += [t for o in obs if o.reason == "horizontal_without_fall"]
        confirms += [tr.ts for tr in fsm.update(obs, t) + fsm.expire(t) if tr.new is PPEState.CONFIRMED_NON_COMPLIANT]
    cap.release()
    return confirms, lying, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo26s-pose.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="first N videos only (smoke runs)")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    from app.config.settings import BEHAVIOURS
    from ultralytics import YOLO

    hold = BEHAVIOURS["fall"].policy.min_duration_s
    falls: dict[str, list[tuple[float, float]]] = {}
    with args.labels.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            falls.setdefault(row["video"], [])  # every video is processed, with or without a fall
            if row["label"] == "fall":
                falls[row["video"]].append((float(row["start"]), float(row["end"])))
    videos = sorted(falls)[: args.limit or None]
    pose = YOLO(args.weights)
    tp = fn = fp = seen_lying = 0
    seconds = 0.0
    for n, v in enumerate(videos, 1):
        confirms, lying, dur = run_video(v, args.fps, pose, args)
        used: set[float] = set()
        for s, e in falls[v]:
            hit = next((c for c in confirms if s <= c <= e + hold + 5.0 and c not in used), None)
            if hit is None:
                fn += 1
                seen_lying += any(s <= x <= e + hold + 5.0 for x in lying)
            else:
                used.add(hit)
                tp += 1
        fp += sum(1 for c in confirms if c not in used)
        seconds += dur
        if n % 50 == 0:
            print(f"{n}/{len(videos)} videos  tp={tp} fn={fn} fp={fp}", flush=True)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out = {
        "labels": str(args.labels),
        "videos": len(videos),
        "falls": tp + fn,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / max(p + r, 1e-9), 4),
        "hours": round(seconds / 3600, 3),
        "false_alarms_per_hour": round(fp / max(seconds / 3600, 1e-9), 2),
        "missed_seen_lying_without_event": seen_lying,
        "fps": args.fps,
        "weights": Path(args.weights).name,
        "hold_s": hold,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
