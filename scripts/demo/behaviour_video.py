"""Render a Phase 4 annotated video: the production assessors + state machines drawn on every sampled frame.

  uv run --project backend --no-sync python scripts/demo/behaviour_video.py --video clip.mp4 \
      --rules fall,fight,phone,stalling --out data/demo_out/clip_annotated.mp4

Per person, one label line per enabled rule (its outcome + reason, fight score), box colour = worst outcome
(red ABSENT = behaviour happening, yellow UNCERTAIN = not assessable, green PRESENT = normal). A banner shows each
CONFIRMED event for 3 s - the same confirmation the alert path uses. Frames are assessed at --fps and written at
that rate, so time in the video is source time. This is the production wiring on real footage, not an accuracy run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

RED, YELLOW, GREEN, WHITE, BLACK = (40, 40, 230), (0, 215, 255), (60, 180, 75), (255, 255, 255), (0, 0, 0)
COCO_PHONE = 67


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rules", default="fall,fight,phone,stalling")
    ap.add_argument("--fps", type=float, default=5.0, help="assess + write rate (source seconds)")
    ap.add_argument("--start", type=float, default=0.0, help="source seconds to start at")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop after this many seconds (0 = whole clip)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    a = ap.parse_args()
    import cv2
    from app.behaviour import make_behaviour_assessor
    from app.behaviour.fight import FightModel
    from app.config.settings import BEHAVIOURS, AssociationConfig, BehaviourConfig, TrackingConfig
    from app.inference.types import Detection
    from app.ppe.state_machine import PPEStateMachine
    from app.ppe.types import PPEState
    from app.streaming.frame import Frame
    from app.tracking.tracker import make_tracker
    from ultralytics import YOLO

    rules = [r for r in a.rules.split(",") if r]
    unknown = [r for r in rules if r not in BEHAVIOURS]
    if unknown:
        print(f"unknown rules {unknown}; choose from {sorted(BEHAVIOURS)}")
        return 2
    out = a.out or ROOT / "data" / "demo_out" / f"{a.video.stem}_{'-'.join(rules)}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg, assoc, tcfg = BehaviourConfig(), AssociationConfig(), TrackingConfig()
    pose = YOLO(str(ROOT / "models" / "yolo26s-pose.pt"))
    phone = YOLO(str(ROOT / "models" / "yolo26s.pt")) if "phone" in rules else None
    fight_path = ROOT / "models" / "fight_gru.pt"
    if "fight" in rules and not fight_path.exists():
        rules.remove("fight")  # no weights -> the rule could only ever say UNCERTAIN
    fight = FightModel(fight_path) if "fight" in rules else None
    assessors = {r: make_behaviour_assessor(r, cfg, BEHAVIOURS[r].thresholds, assoc, fight) for r in rules}
    fsms = {r: PPEStateMachine("render", BEHAVIOURS[r].policy, tcfg.track_expiry_s) for r in rules}

    cap = cv2.VideoCapture(str(a.video))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    if not vfps or vfps <= 0:  # some OmniFall clips report -1; derive it from duration, else assume 25 fps
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
        dur = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)
        vfps = n / dur if n > 0 and dur > 0 else 25.0
    i = int(a.start * vfps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    tracker, writer, events, next_t = make_tracker(tcfg, a.fps), None, [], 0.0
    done: set[str] = set()
    print(f"{a.video} @ {vfps:.1f} fps -> assessed at {a.fps:g} fps; rules: {', '.join(rules)}")

    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = i / vfps
        i += 1
        if t + 1e-6 < next_t:
            continue
        next_t += 1.0 / a.fps
        r = pose.predict(img, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes) and r.keypoints is not None:
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                               r.keypoints.data.cpu().numpy(), strict=True):
                dets.append(Detection("person", float(c), tuple(map(float, b)), "render", k))
        phones = []
        if phone is not None:
            pr = phone.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
            if pr.boxes is not None:
                for cls, c, b in zip(pr.boxes.cls.cpu().numpy(), pr.boxes.conf.cpu().numpy(),
                                     pr.boxes.xyxy.cpu().numpy(), strict=True):
                    if int(cls) == COCO_PHONE:
                        phones.append(Detection("phone", float(c), tuple(map(float, b)), "render"))
        frame = Frame("render", i, img, None, t, t)
        persons = tracker.update(dets, frame)
        obs = {rule: {o.track_id: o for o in as_.assess_deferred(persons, phones, frame)}
               for rule, as_ in assessors.items()}
        for rule, fsm in fsms.items():
            for tr in fsm.update(list(obs[rule].values()), t):
                if tr.new is PPEState.CONFIRMED_NON_COMPLIANT:
                    events.append((t + 3.0, f"{BEHAVIOURS[rule].alert}  track {tr.track_id}  t={t:.1f}s"))
                    print(f"  EVENT {BEHAVIOURS[rule].alert:<14} track {tr.track_id} t={t:.1f}s")
                    done.add(rule)
        events = [e for e in events if e[0] > t]

        annotated = img.copy()
        s = max(0.4, annotated.shape[0] / 1440.0)
        colour = {"ABSENT": RED, "UNCERTAIN": YELLOW, "PRESENT": GREEN}
        for p in persons:
            lines = []
            for rule in rules:
                o = obs[rule].get(p.track_id)
                if o is None:
                    continue
                score = f" {o.confidence:.2f}" if rule in ("fight", "phone") else ""
                lines.append((f"{rule}: {o.outcome.name.lower()} {o.reason}{score}", colour[o.outcome.name]))
            if not lines:
                continue
            cols = [c for _, c in lines]
            col = RED if RED in cols else YELLOW if YELLOW in cols else GREEN
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, max(2, int(annotated.shape[0] / 400)))
            ty = y1 - 4
            for line, c in reversed(lines):
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, s, 1)
                cv2.rectangle(annotated, (x1, ty - th - 3), (x1 + tw + 6, ty + 3), BLACK, -1)
                cv2.putText(annotated, line, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)
                ty -= th + 8
        for j, (_, text) in enumerate(events):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s * 1.3, 2)
            top = 10 + j * (th + 14)
            cv2.rectangle(annotated, (10, top), (20 + tw, top + th + 10), RED, -1)
            cv2.putText(annotated, text, (15, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, s * 1.3,
                        WHITE, 2, cv2.LINE_AA)

        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                                     (annotated.shape[1], annotated.shape[0]))
        writer.write(annotated)
        if a.max_seconds and t - a.start >= a.max_seconds:
            break

    cap.release()
    if writer is not None:
        writer.release()
    print(f"wrote {out}  (rule(s) with a confirmed event: {', '.join(sorted(done)) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
