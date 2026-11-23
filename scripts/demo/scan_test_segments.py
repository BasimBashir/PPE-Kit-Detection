"""Scan a Phase 4 real test split with the production pipeline and report which clips confirm an event.

  python scripts/demo/scan_test_segments.py --split fall-syn --min-seconds 3
  python scripts/demo/scan_test_segments.py --split fall-itw --min-seconds 3
  python scripts/demo/scan_test_segments.py --split fight-scfd --max-seconds 20

Splits: fall-syn / fall-itw read the OmniFall test CSVs (labelled fall segments); itw-all reads every in-the-wild
segment of at least --min-seconds (phone use is not fall-specific); fight-scfd / fight-rlvs read the SCFD (test
set) or RLVS (val set) fight folders. Nothing is written - feed the printed clips to behaviour_video.py to render
them. Same pose model, tracker, assessors and state machines as production.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

DATA = ROOT / "data"


def segments(split: str, min_seconds: float, max_seconds: float, limit: int) -> list[tuple[Path, float, float]]:
    out: list[tuple[Path, float, float]] = []
    if split in ("fall-syn", "fall-itw", "itw-all"):
        name = "of_syn_test.csv" if split == "fall-syn" else "of_itw_test.csv"
        with (DATA / "fall" / name).open(newline="") as f:
            for r in csv.DictReader(f):
                start, end = float(r["start"]), float(r["end"])
                if (split == "itw-all" or r["label"] == "fall") and end - start >= min_seconds:
                    out.append((Path(r["video"]), start, end))
    else:
        sub = DATA / "fight" / ("scfd/fight" if split == "fight-scfd" else "rlvs/Violence")
        out = [(p, 0.0, max_seconds) for p in sorted(sub.glob("*.mp4"))]
    return out[:limit] if limit else out


def scan(path: Path, start: float, end: float, fps: float, models) -> list[tuple[str, float, int]]:
    import cv2
    from app.inference.types import Detection
    from app.streaming.frame import Frame
    from app.tracking.tracker import make_tracker

    pose, phone, factory, tcfg = models
    # assessor + FSM state is per (camera, people); a fresh instance per clip stops track-id state leaking across clips
    assessors, fsms = factory()
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    if not vfps or vfps <= 0:  # some OmniFall clips report -1; derive it from duration, else assume 25 fps
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
        dur = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)
        vfps = n / dur if n > 0 and dur > 0 else 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * vfps))
    tracker = make_tracker(tcfg, fps)
    events, next_t, i = [], 0.0, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = start + i / vfps
        i += 1
        if t > end:
            break
        if t + 1e-6 < next_t:
            continue
        next_t += 1.0 / fps
        r = pose.predict(img, imgsz=640, conf=0.3, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes) and r.keypoints is not None:
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                               r.keypoints.data.cpu().numpy(), strict=True):
                dets.append(Detection("person", float(c), tuple(map(float, b)), "scan", k))
        phones = []
        if phone is not None:
            pr = phone.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
            if pr.boxes is not None:
                for cls, c, b in zip(pr.boxes.cls.cpu().numpy(), pr.boxes.conf.cpu().numpy(),
                                     pr.boxes.xyxy.cpu().numpy(), strict=True):
                    if int(cls) == 67:
                        phones.append(Detection("phone", float(c), tuple(map(float, b)), "scan"))
        frame = Frame("scan", i, img, None, t, t)
        persons = tracker.update(dets, frame)
        for rule, as_ in assessors.items():
            for tr in fsms[rule].update(as_.assess_deferred(persons, phones, frame), t):
                if tr.new.name == "CONFIRMED_NON_COMPLIANT":
                    events.append((rule, round(t, 2), tr.track_id))
    cap.release()
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True,
                    choices=["fall-syn", "fall-itw", "itw-all", "fight-scfd", "fight-rlvs"])
    ap.add_argument("--rules", default=None, help="default: fall for fall-*, fight for fight-*")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--min-seconds", type=float, default=3.0, help="skip shorter fall segments")
    ap.add_argument("--max-seconds", type=float, default=20.0, help="cap per fight clip")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from app.behaviour import make_behaviour_assessor
    from app.behaviour.fight import FightModel
    from app.config.settings import BEHAVIOURS, AssociationConfig, BehaviourConfig, TrackingConfig
    from app.ppe.state_machine import PPEStateMachine
    from ultralytics import YOLO

    default_rule = "fall" if a.split.startswith("fall") else "phone" if a.split == "itw-all" else "fight"
    rules = [r for r in (a.rules or default_rule).split(",") if r]
    cfg, assoc, tcfg = BehaviourConfig(), AssociationConfig(), TrackingConfig()
    pose = YOLO(str(ROOT / "models" / "yolo26s-pose.pt"))
    phone = YOLO(str(ROOT / "models" / "yolo26s.pt")) if "phone" in rules else None
    fight = FightModel(ROOT / "models" / "fight_gru.pt") if "fight" in rules else None

    def factory():
        assessors = {r: make_behaviour_assessor(r, cfg, BEHAVIOURS[r].thresholds, assoc, fight) for r in rules}
        fsms = {r: PPEStateMachine("scan", BEHAVIOURS[r].policy, tcfg.track_expiry_s) for r in rules}
        return assessors, fsms

    segs = segments(a.split, a.min_seconds, a.max_seconds, a.limit)
    print(f"{a.split}: {len(segs)} segments, rules {rules}", flush=True)
    fired = 0
    for path, start, end in segs:
        ev = scan(path, start, end, a.fps, (pose, phone, factory, tcfg))
        fired += bool(ev)
        hits = "  ".join(f"FIRED {r} t={t} track={tr}" for r, t, tr in ev) or "-"
        print(f"{path.name:<40} {start:6.2f}-{end:6.2f}  {hits}", flush=True)
    print(f"{fired}/{len(segs)} clips confirmed an event")
    return 0


if __name__ == "__main__":
    sys.exit(main())
