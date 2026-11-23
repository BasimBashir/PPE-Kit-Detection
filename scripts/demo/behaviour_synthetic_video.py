"""Render one annotated video per Phase 4 behaviour from scripted stick figures - the visual twin of
behaviour_demo.py. The production assessors, the real fight GRU and the alert path's state machines run unchanged;
only the input poses are synthetic, so every rule's firing is visible frame by frame. Not an accuracy run.

  uv run --project backend --no-sync python scripts/demo/behaviour_synthetic_video.py [--out data/demo_out]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.behaviour import make_behaviour_assessor  # noqa: E402
from app.behaviour.assessors import torso  # noqa: E402
from app.behaviour.fight import FightModel  # noqa: E402
from app.config.settings import BEHAVIOURS, AssociationConfig, BehaviourConfig  # noqa: E402
from app.inference.types import Detection  # noqa: E402
from app.ppe.state_machine import PPEStateMachine  # noqa: E402
from app.ppe.types import PPEState, TrackedPerson  # noqa: E402
from app.streaming.frame import Frame  # noqa: E402

W, H = 1280, 720
RED, YELLOW, GREEN, WHITE, BLACK, GREY = (40, 40, 230), (0, 215, 255), (60, 180, 75), (255, 255, 255), (0, 0, 0), \
    (200, 200, 200)
BONES = [(0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]
CFG = BehaviourConfig()


def body(cx: float, cy: float, h: float, angle: float = 0.0, flail: float = 0.0, rng=None) -> np.ndarray:
    """17 keypoints: shoulder centre at (cx, cy), direction rotated `angle` deg from vertical."""
    a = np.radians(angle)
    u, p = np.array([np.sin(a), np.cos(a)]), np.array([np.cos(a), -np.sin(a)])
    s = np.array([cx, cy], float)
    hip = s + 0.45 * h * u
    k = np.zeros((17, 3), np.float32)

    def put(i, xy):
        k[i] = (xy[0], xy[1], 0.9)

    put(0, s - 0.18 * h * u)
    put(5, s - 0.15 * h * p)
    put(6, s + 0.15 * h * p)
    put(11, hip - 0.12 * h * p)
    put(12, hip + 0.12 * h * p)
    for sh_i, side, (el_i, wr_i) in ((5, 1, (7, 9)), (6, -1, (8, 10))):
        j = rng.uniform(-flail, flail, 2) if rng is not None and flail else np.zeros(2)
        sh = k[sh_i, :2]
        put(el_i, sh + 0.22 * h * (0.6 * side * p + 0.4 * u) + j)
        put(wr_i, sh + 0.42 * h * (0.2 * side * p + 0.8 * u) + j)
    for hip_i, side, (kn_i, an_i) in ((11, -1, (13, 15)), (12, 1, (14, 16))):
        hp = k[hip_i, :2]
        put(kn_i, hp + 0.25 * h * u + 0.06 * h * side * p)
        put(an_i, hp + 0.5 * h * u + 0.06 * h * side * p)
    return k


def surface() -> np.ndarray:
    img = np.full((H, W, 3), 245, np.uint8)
    img[H - 120:] = GREY  # floor line
    return img


def draw(img: np.ndarray, k: np.ndarray) -> None:
    import cv2
    for a, b in BONES:
        if k[a, 2] > 0 and k[b, 2] > 0:
            cv2.line(img, tuple(map(int, k[a, :2])), tuple(map(int, k[b, :2])), (120, 90, 40), 4)
    for i in (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        if k[i, 2] > 0:
            cv2.circle(img, tuple(map(int, k[i, :2])), 6, (120, 90, 40), -1)


def overlay(img: np.ndarray, rows: list, banner: str | None, caption: str) -> None:
    import cv2
    cv2.putText(img, caption, (16, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 1, cv2.LINE_AA)
    y = 30
    for text, col in rows:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img, (10, y - th - 6), (24 + tw, y + 6), BLACK, -1)
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        y += th + 16
    if banner:
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(img, (W - tw - 40, 20), (W - 20, 20 + th + 20), RED, -1)
        cv2.putText(img, banner, (W - tw - 30, 20 + th + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)


def record(path: Path, frames: list, fps: float) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for img in frames:
        w.write(img)
    w.release()
    print(f"wrote {path}")


def run(rule: str, assessor, frames, caption: str, out: Path, fps: float = 5.0, dets_fn=None) -> None:
    fsm = PPEStateMachine("synthetic", BEHAVIOURS[rule].policy, 5.0)
    imgs, banner = [], None
    for t, persons, extras in frames:
        img = surface()
        for p in persons:
            draw(img, p.keypoints)
        dets = dets_fn(t, persons) if dets_fn else []
        import cv2
        for d in dets:
            x1, y1, x2, y2 = map(int, d.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, 2)
            cv2.putText(img, f"{d.cls} {d.conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1,
                        cv2.LINE_AA)
        obs = {o.track_id: o for o in assessor.assess_deferred(persons, dets, Frame("synthetic", int(t * fps),
                                                                                    img, None, t, t))}
        for tr in fsm.update(list(obs.values()), t):
            if tr.new is PPEState.CONFIRMED_NON_COMPLIANT:
                banner = f"{BEHAVIOURS[rule].alert}  track {tr.track_id}  t={t:.1f}s"
                print(f"  EVENT {BEHAVIOURS[rule].alert} track {tr.track_id} t={t:.1f}s")
        rows = []
        for p in persons:
            o = obs.get(p.track_id)
            if o is not None:
                col = {"ABSENT": RED, "UNCERTAIN": YELLOW, "PRESENT": GREEN}[o.outcome.name]
                extra = f"  conf {o.confidence:.2f}" if rule in ("fight", "phone") else ""
                rows.append((f"{rule}: {o.outcome.name.lower()} {o.reason}{extra}", col))
        for text in extras:
            rows.append((text, WHITE))
        cap = f"{caption}   |   t={t:4.1f}s"
        overlay(img, rows, banner, cap)
        imgs.append(img)
    record(out, imgs, fps)


def fall_frames(upright_s: float = 2.0, total_s: float = 8.0, control: bool = False):
    """Stand, then be lying on the ground from upright_s on (upright box h=400, lying drop 135 px > 0.25*400).

    `control` lies from t=0 with no standing sample, so the fall rule must never confirm."""
    out = []
    for t in np.arange(0.0, total_s + 1e-6, 0.2):
        if t < upright_s:
            k = body(480, 200, 300)
            box = (300.0, 60.0, 660.0, 460.0)  # h=400: the standing height the drop is measured against
            extra = "torso 0 deg (standing)"
        else:
            k = body(430, 470, 300, angle=90)
            box = (280.0, 420.0, 760.0, 560.0)
            extra = "torso 90 deg (lying, no standing sample)" if control else \
                f"torso {torso(TrackedPerson(1, box, 0.9, k), CFG.keypoint_min_conf)[0]:.0f} deg   " \
                "hip drop 135 px (needs >= 100)"
        out.append((t, [TrackedPerson(1, box, 0.9, k)], [extra]))
    return out


def phone_dets(t: float, persons) -> list:
    if 3.0 <= t < 3.6:  # detector dropout, bridged (phone_gap_s = 2 s)
        return []
    wr = persons[0].keypoints[9, :2]
    return [Detection("phone", 0.9, (wr[0] - 14, wr[1] - 20, wr[0] + 14, wr[1] + 20), "synthetic")]


def phone_frames(total_s: float = 12.0):
    out = []
    for t in np.arange(0.0, total_s + 1e-6, 0.2):
        k = body(480, 200, 300)
        out.append((t, [TrackedPerson(1, (300.0, 60.0, 660.0, 460.0), 0.9, k)],
                    ["phone held over the left hand (dropout 3.0-3.6 s)"]))
    return out


def idle_frames(total_s: float = 30.0, move: bool = False):
    rng = np.random.default_rng(0)
    out = []
    for t in np.arange(0.0, total_s + 1e-6, 0.5):
        k = body(480, 200, 300, flail=40 if move else 0, rng=rng)
        if not move:
            k[:, :2] += rng.uniform(-1, 1, (17, 2))
        out.append((t, [TrackedPerson(1, (300.0, 60.0, 660.0, 460.0), 0.9, k)], []))
    return out


def fight_frames(total_s: float = 5.0):
    rng = np.random.default_rng(1)
    out = []
    for t in np.arange(0.0, total_s + 1e-6, 0.2):
        a = TrackedPerson(1, (300.0, 120.0, 560.0, 620.0), 0.9, body(420, 250, 300, flail=60, rng=rng))
        b = TrackedPerson(2, (600.0, 120.0, 860.0, 620.0), 0.9, body(720, 250, 300, flail=60, rng=rng))
        out.append((t, [a, b], ["two people 300 px apart (< 1.2 body heights)"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "demo_out")
    a = ap.parse_args()
    print("Phase 4 synthetic annotated videos - production assessors, scripted poses")
    assoc = AssociationConfig()

    def mk(rule: str):
        model = FightModel(ROOT / "models" / "fight_gru.pt") if rule == "fight" else None
        return make_behaviour_assessor(rule, CFG, BEHAVIOURS[rule].thresholds, assoc, model)

    run("fall", mk("fall"), fall_frames(),
        "SYNTHETIC: fast fall -> held down (rule needs 3 s hold)", a.out / "synthetic_fall.mp4")
    run("fall", mk("fall"), fall_frames(upright_s=0.0, total_s=8.0, control=True),
        "SYNTHETIC control: lying WITHOUT a fall sample -> never an event", a.out / "synthetic_no_fall.mp4")
    run("phone", mk("phone"), phone_frames(),
        "SYNTHETIC: phone at hand, dropout bridged, 10 s hold -> event",
        a.out / "synthetic_phone.mp4", dets_fn=phone_dets)
    run("fight", mk("fight"), fight_frames(),
        "SYNTHETIC: two flailing figures, real GRU, 3-of-5 count -> event",
        a.out / "synthetic_fight.mp4")
    run("stalling", mk("stalling"), idle_frames(),
        "SYNTHETIC: still body - SUSPECTED only (production hold is 600 s)", a.out / "synthetic_stalling.mp4",
        fps=2.0)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
