"""Unit tests: camera health FSM, backoff, latest-frame slot, fair scheduler (spec 19.1)."""

import random
import threading

import numpy as np
import pytest

from app.camera.health import Backoff, CameraHealth, CameraState
from app.config.settings import IngestConfig
from app.streaming.frame import Frame
from app.streaming.scheduler import InferenceScheduler
from app.streaming.slot import LatestFrameSlot

CS = CameraState


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def tick(self, dt):
        self.t += dt


def health(clock=None, **kw):
    clock = clock or Clock()
    return CameraHealth("camT", IngestConfig(**kw), clock=clock, wall=clock), clock


def frames(h, clock, n, dt):
    for _ in range(n):
        h.on_frame()
        clock.tick(dt)


def test_connect_online_degraded_offline_by_frame_age():
    h, c = health()
    assert h.state is CS.CONNECTING
    h.on_connecting()
    h.on_connected(25.0)
    frames(h, c, 50, 0.04)
    assert h.state is CS.ONLINE and h.evaluate() is CS.ONLINE
    c.tick(2.0)  # last frame 2.04 s ago: > degraded 1.5, < stall 5
    assert h.evaluate() is CS.DEGRADED and "late frames" in h.reason
    c.tick(4.0)
    assert h.evaluate() is CS.OFFLINE and h.reason.startswith("stalled")
    h.on_connecting()
    assert h.state is CS.RECOVERING and h.reconnect_count == 1
    c.tick(1.0)
    h.on_frame()
    assert h.state is CS.ONLINE and h.last_recovery_s == pytest.approx(7.0, abs=0.05)


def test_low_decode_fps_is_degraded_after_grace():
    h, c = health()
    h.on_connecting()
    h.on_connected(30.0)
    frames(h, c, 40, 0.1)  # 10 fps < 0.5 * 30
    assert h.evaluate() is CS.DEGRADED and "low decode rate" in h.reason


def test_trickle_of_frames_is_a_stall_not_degraded():
    h, c = health()
    h.on_connecting()
    h.on_connected(30.0)
    for _ in range(4):  # one straggler frame every ~1.2 s: never 5 s without a frame, but unusable
        h.on_frame()
        c.tick(1.2)
        h.evaluate()
    assert h.state is CS.OFFLINE and h.reason.startswith("stalled: decode rate")


def test_fatal_error_and_retry_keeps_error_reason():
    h, _ = health()
    h.on_connecting()
    h.on_error("HTTPUnauthorizedError: 401 rtsp://***@h/c", fatal=True)
    assert h.state is CS.ERROR
    h.on_connecting()
    assert h.state is CS.ERROR and "401" in h.reason


def test_disabled_and_uptime_accumulation():
    h, c = health()
    h.on_connecting()
    h.on_connected(25.0)
    for _ in range(20):
        h.on_frame()
        c.tick(0.25)
        h.evaluate()
    assert h.online_seconds == pytest.approx(5.0, abs=0.3)
    h.on_disabled()
    assert h.state is CS.DISABLED


def test_backoff_bounded_with_jitter_and_reset():
    b = Backoff(1.0, 30.0, random.Random(1))
    delays = [b.next_delay() for _ in range(10)]
    caps = [min(30.0, 2**i) for i in range(10)]
    assert all(0.5 * cap <= d <= cap for d, cap in zip(delays, caps, strict=True))
    assert max(delays) <= 30.0
    b.reset()
    assert b.attempt == 0 and b.next_delay() <= 1.0


def mkframe(cam, i, clock_t=0.0):
    return Frame(cam, i, np.zeros((4, 4, 3), np.uint8), None, clock_t, clock_t)


def test_slot_drops_and_depth():
    s = LatestFrameSlot("camT", threading.Condition())
    assert s.depth == 0 and s.take() is None
    assert s.put(mkframe("camT", 1)) is False
    assert s.put(mkframe("camT", 2)) is True and s.dropped == 1 and s.depth == 1
    f = s.take()
    assert f.frame_id == 2 and s.depth == 0


def make_sched(n=10, fps=5.0, max_batch=4, clock=None):
    cond = threading.Condition()
    clock = clock or Clock(0.0)
    slots = {f"cam{i:02d}": LatestFrameSlot(f"cam{i:02d}", cond) for i in range(n)}
    sch = InferenceScheduler(slots, dict.fromkeys(slots, fps), max_batch, 0, cond, clock=clock)
    return sch, slots, clock


def test_fairness_every_camera_served_within_three_batches():
    sch, slots, clock = make_sched()
    for c, s in slots.items():
        s.put(mkframe(c, 1))
    served = []
    for _ in range(3):
        b = sch.next_batch(0)
        served += [f.camera_id for f in b]
        for f in b:  # those cameras immediately have a fresh frame again
            slots[f.camera_id].put(mkframe(f.camera_id, 2))
        clock.tick(1.0)
    assert set(served) == set(slots)


def test_fps_pacing_and_no_burst_after_stall():
    sch, slots, clock = make_sched(n=1, fps=5.0)
    (cam,) = slots
    processed = 0
    for i in range(100):  # 10 s of 10 Hz frames
        slots[cam].put(mkframe(cam, i))
        processed += len(sch.next_batch(0))
        clock.tick(0.1)
    assert processed == pytest.approx(50, abs=1)
    clock.tick(10.0)  # stall: nothing arrives for 10 s
    got = 0
    for i in range(5):  # 0.5 s of frames after the stall
        slots[cam].put(mkframe(cam, 200 + i))
        got += len(sch.next_batch(0))
        clock.tick(0.1)
    assert got <= 3  # no catch-up burst of 50 "owed" frames


def test_timeout_returns_empty():
    sch, _, _ = make_sched(n=2)
    assert sch.next_batch(0.01) == []
