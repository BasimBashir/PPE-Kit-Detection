"""Phase 4 behaviour checks - annotated run on synthetic people (no camera, no DB, no GPU).

Every verdict printed below comes from the production code (app.behaviour.assessors / app.behaviour.fight and the
same PPEStateMachine the alert path uses). The poses are hand-built so each rule's decision is visible line by
line; this is NOT an accuracy evaluation (that is docs/ml/phase4-log.md). Synthetic motion through the real fight
GRU shows the wiring, not the model's skill.

  uv run --project backend --no-sync python scripts/demo/behaviour_demo.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.behaviour.assessors import FallAssessor, IdleAssessor, PhoneAssessor, torso  # noqa: E402
from app.behaviour.fight import FightAssessor, FightModel  # noqa: E402
from app.config.settings import BEHAVIOURS, AssociationConfig, BehaviourConfig, RuleThresholds, Schedule  # noqa: E402
from app.inference.types import Detection  # noqa: E402
from app.ppe.state_machine import PPEStateMachine  # noqa: E402
from app.ppe.types import PPEState, TrackedPerson  # noqa: E402
from app.streaming.frame import Frame  # noqa: E402

IMG = np.zeros((720, 1280, 3), np.uint8)
CFG, TH = BehaviourConfig(), RuleThresholds()
UPRIGHT_BOX, LYING_BOX = (450.0, 150.0, 550.0, 550.0), (350.0, 480.0, 650.0, 560.0)


def frame(t: float) -> Frame:
    return Frame("cam01", int(t * 10), IMG, None, t, t)


def kps(sh, hip, wrists=None, conf=0.9):
    k = np.zeros((17, 3), np.float32)
    k[5], k[6] = (sh[0] - 20, sh[1], conf), (sh[0] + 20, sh[1], conf)
    k[11], k[12] = (hip[0] - 15, hip[1], conf), (hip[0] + 15, hip[1], conf)
    for i, xy in (wrists or {}).items():
        k[i] = (*xy, conf)
    return k


UPRIGHT = (UPRIGHT_BOX, kps((500, 200), (500, 350), {9: (405, 300), 10: (515, 300)}), "standing")
MID = (UPRIGHT_BOX, kps((440, 380), (520, 450)), "mid angle ~49 deg")
LYING = (LYING_BOX, kps((380, 520), (530, 520)), "lying, hips -170 px")
BENT = (UPRIGHT_BOX, kps((380, 345), (500, 350)), "bending, hips unchanged")


def verdict(o) -> str:
    return f"{o.outcome.name:<9} {o.reason}"


def run_fall(title: str, note: str, poses: list, dt: float = 0.2) -> None:
    print(f"\n[{title}] {note}")
    rule = BEHAVIOURS["fall"]
    a = FallAssessor(CFG, rule.thresholds)
    fsm = PPEStateMachine("cam01", rule.policy, 5.0)
    last = None
    for i, (box, k, tag) in enumerate(poses):
        t = i * dt
        p = TrackedPerson(1, box, 0.9, k)
        (o,) = a.assess_deferred([p], [], frame(t))
        trs = fsm.update([o], t)
        if (o.outcome, o.reason) != last or trs:  # print only when the verdict or the FSM changes
            ang = torso(p, CFG.keypoint_min_conf)
            print(f"  t={t:4.1f}s {tag:<24} torso {ang[0] if ang else 0:5.0f} deg -> {verdict(o)}")
            last = (o.outcome, o.reason)
        for tr in trs:
            print(f"           FSM {tr.prev.name} -> {tr.new.name} ({tr.reason})")
            if tr.new is PPEState.CONFIRMED_NON_COMPLIANT:
                print(f"  >>> EVENT t={t:.1f}s: rule 'fall' track 1 - the event service would write an event now, "
                      f"with this frame as evidence (same path as a missing helmet)")
    if title.startswith("1"):
        print(f"  (hold: the fall itself is instant; 'down after a fall' must last the rule's "
              f"min_duration_s={rule.policy.min_duration_s:g} s before the event - that is the timer seen above)")


def run_phone() -> None:
    print("\n[3] PHONE - phone in hand, detector drops it for 1 s (bridged), then it is gone")
    rule = BEHAVIOURS["phone"]
    a = PhoneAssessor(CFG, rule.thresholds, AssociationConfig())
    person = TrackedPerson(1, (400.0, 100.0, 520.0, 500.0), 0.9,
                           kps((460, 160), (460, 400), {9: (405, 300), 10: (515, 300)}))
    phone = Detection("phone", 0.9, (390.0, 300.0, 420.0, 340.0), "t")
    rows = [(0.0, phone, "phone detected over the left hand"),
            (0.4, None, "phone not detected (0.4 s gap)"),
            (1.0, None, "still missing (<= phone_gap_s = 2 s: bridged)"),
            (2.5, None, "missing > 2 s -> episode over")]
    for t, det, note in rows:
        (o,) = a.assess_deferred([person], [det] if det else [], frame(t))
        print(f"  t={t:4.1f}s {note:<44} -> {verdict(o)}")

    print("\n[4] SCHEDULE - the same rule outside its shift emits nothing (enforced by the worker)")
    sched = Schedule(tz="Asia/Dubai", days=[datetime(2026, 11, 23).weekday()], windows=["06:00-14:00"])
    for hour, label in (("10:00", "inside the shift"), ("22:00", "after hours")):
        t = datetime.fromisoformat(f"2026-11-23T{hour}:00+04:00").timestamp()
        print(f"  {hour} Dubai ({label:<16}) schedule.active={sched.active(t)!s:<5} -> "
              f"{'observations assessed' if sched.active(t) else 'no observations at all'}")


def run_stalling() -> None:
    print("\n[5] STALLING - the body is still for 32 s, then the hands start working")
    rule = BEHAVIOURS["stalling"]
    rng = np.random.default_rng(0)
    base = {0: (460, 130), 5: (420, 160), 6: (500, 160), 7: (410, 240), 8: (510, 240), 9: (405, 300),
            10: (515, 300), 11: (430, 300), 12: (490, 300), 15: (440, 480), 16: (480, 480)}
    for phase, moving in (("still", False), ("hands moving", True)):
        a = IdleAssessor(CFG, rule.thresholds)  # fresh per phase: the second run starts its own history at t=0
        fsm = PPEStateMachine("cam01", rule.policy, 5.0)
        last = None
        for i in range(int(32 / 0.5) + 1):
            t = i * 0.5
            k = np.zeros((17, 3), np.float32)
            for j, (x, y) in base.items():
                jit = rng.uniform(-40, 40, 2) if (moving and j in (9, 10)) else rng.uniform(-1, 1, 2)
                k[j] = (x + jit[0], y + jit[1], 0.9)
            (o,) = a.assess_deferred([TrackedPerson(1, (400.0, 100.0, 520.0, 500.0), 0.9, k)], [], frame(t))
            trs = fsm.update([o], t)
            if (o.outcome, o.reason) != last or trs:
                print(f"  [{phase:<12}] t={t:4.1f}s -> {verdict(o)}")
                last = (o.outcome, o.reason)
            for tr in trs:
                print(f"                 FSM {tr.prev.name} -> {tr.new.name} ({tr.reason})")
    print(f"  (stillness, not intent: the rule's min_duration_s={rule.policy.min_duration_s:g} s is the operator "
          f"knob; scope with a work-area zone + shift schedule - no accuracy claim, there is no ground truth)")


def run_fight() -> None:
    path = ROOT / "models" / "fight_gru.pt"
    print(f"\n[6] FIGHT - real weights ({path.relative_to(ROOT)}), two synthetic people flailing side by side")
    if not path.exists():
        print("  weights missing - run `git lfs pull` (see models/README.md)")
        return
    rule = BEHAVIOURS["fight"]
    model = FightModel(path)
    print(f"  model {model.version}: {model.seq_len} samples @ {model.hz:g} Hz = "
          f"{(model.seq_len - 1) / model.hz:.1f} s window; ABSENT when score >= {rule.thresholds.absent_conf}")
    a = FightAssessor(CFG, rule.thresholds, model)
    rng = np.random.default_rng(1)
    boxes = [(300.0, 150.0, 400.0, 550.0), (500.0, 150.0, 600.0, 550.0)]  # 200 px apart < 1.2 body heights
    last: dict[int, tuple] = {}
    for i in range(31):  # 6 s at 5 Hz: 1.4 s to fill the window, then scored every frame
        t = i / 5.0
        persons = []
        for tid, b in enumerate(boxes, start=1):
            k = np.zeros((17, 3), np.float32)
            for j in (0, 5, 6, 11, 12):
                k[j] = ((b[0] + b[2]) / 2 + rng.uniform(-5, 5), 200 + rng.uniform(-5, 5), 0.9)
            for j in (7, 8, 9, 10):  # arms swinging fast
                k[j] = ((b[0] + b[2]) / 2 + rng.uniform(-60, 60), 300 + rng.uniform(-60, 60), 0.9)
            persons.append(TrackedPerson(tid, b, 0.9, k))
        for o in a.assess_deferred(persons, [], frame(t)):
            key = (o.outcome, o.reason)  # per track; the score drifts in the 3rd decimal every frame
            if last.get(o.track_id) != key:
                print(f"  t={t:4.1f}s track {o.track_id} -> {verdict(o)} (score {o.confidence:.2f})")
                last[o.track_id] = key
    print("  note: the score on synthetic poses shows the pipeline works; it is not an accuracy measurement "
          "(RLVS val F1 0.65, SCFD cross-dataset F1 0.39 - docs/ml/phase4-log.md).")


def main() -> int:
    print("PHASE 4 BEHAVIOUR CHECKS - annotated run (production assessors + PPE state machine, synthetic poses)")
    print(f"config: fall window {CFG.fall_window_s:g} s, upright <= {CFG.upright_max_deg:g} deg, lying >= "
          f"{CFG.lying_min_deg:g} deg, hip drop >= {CFG.drop_frac:g} x standing height, hold "
          f"{BEHAVIOURS['fall'].policy.min_duration_s:g} s | phone gap {CFG.phone_gap_s:g} s | idle window "
          f"{CFG.idle_window_s:g} s | pairs closer than {CFG.fight_pair_dist:g} body heights")
    run_fall("1", "FAST fall: standing -> lying within one 0.2 s step, then held down", [UPRIGHT] * 4 + [LYING] * 20)
    run_fall("2", "SLOW lying down: same end pose reached through 15 mid-angle samples",
             [UPRIGHT] + [MID] * 15 + [LYING] * 30)
    run_fall("2b", "BENDING over: torso horizontal but the hips never drop", [UPRIGHT] * 4 + [BENT] * 6)
    run_phone()
    run_stalling()
    run_fight()
    print("\nDone. These are rule mechanics; field accuracy is NOT measured (docs/ml/phase4-log.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
