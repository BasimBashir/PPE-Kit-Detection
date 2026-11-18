"""Unit tests: CameraSource ingest thread (open -> read -> error -> backoff -> reconnect) on local files, no network."""

import threading
import time

import cv2
import numpy as np
import pytest

from app.camera import source as source_mod
from app.camera.health import Backoff, CameraHealth, CameraState
from app.camera.source import CameraRuntimeConfig, CameraSource
from app.config.settings import IngestConfig
from app.streaming.slot import LatestFrameSlot

CS = CameraState


def wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def good_uri(tmp_path):
    path = tmp_path / "cam.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    for i in range(10):
        w.write(np.full((48, 64, 3), i * 20, np.uint8))
    w.release()
    return path.as_uri()


@pytest.fixture
def bad_uri(tmp_path):
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"not a video stream" * 64)  # exists but undecodable -> transient (non-fatal) error
    return path.as_uri()


@pytest.fixture
def delays(monkeypatch):
    got: list[float] = []

    class RecordingBackoff(Backoff):
        def next_delay(self):
            d = super().next_delay()
            got.append(d)
            return d

    monkeypatch.setattr(source_mod, "Backoff", RecordingBackoff)
    return got


@pytest.fixture
def run_source():
    started: list[CameraSource] = []

    def make(uri):
        cfg = IngestConfig(backoff_base_s=0.02, backoff_max_s=0.1)
        slot = LatestFrameSlot("camT", threading.Condition())
        src = CameraSource(CameraRuntimeConfig("camT", uri), cfg, slot, CameraHealth("camT", cfg), convert_fps=50.0)
        src.start()
        started.append(src)
        return src, slot

    yield make
    for s in started:
        s.stop()
        assert not s.is_alive()


def test_frames_flow_into_slot_and_camera_goes_online(run_source, good_uri):
    src, slot = run_source(good_uri)
    assert wait_until(lambda: src.health.state is CS.ONLINE)
    assert wait_until(lambda: slot.depth == 1)
    first = slot.take()
    assert first.camera_id == "camT" and first.image.shape == (48, 64, 3)
    assert wait_until(lambda: slot.depth == 1)  # keeps flowing (file loops past its 10 frames)
    assert slot.take().frame_id > first.frame_id
    assert (src.width, src.height) == (64, 48) and src.health.nominal_fps == pytest.approx(25.0)


def test_unreadable_source_backs_off_offline_then_recovers(run_source, good_uri, bad_uri, delays):
    src, slot = run_source(bad_uri)
    assert wait_until(lambda: len(delays) >= 3)
    h = src.health
    assert wait_until(lambda: h.state is CS.OFFLINE)  # briefly CONNECTING during each retry
    assert "InvalidDataError" in h.reason
    assert h.reconnect_count >= 2 and slot.depth == 0
    assert all(d <= 0.1 for d in delays) and max(delays) > min(delays)  # exponential growth, capped at max

    src.cam.stream_uri = good_uri  # source becomes readable again
    assert wait_until(lambda: h.state is CS.ONLINE)
    assert wait_until(lambda: slot.depth == 1)

    # mid-stream failure after having been online: OFFLINE -> RECOVERING -> ONLINE with a measured recovery time
    reconnects = h.reconnect_count
    src.hooks.decoder_error_once = True
    assert wait_until(lambda: h.reconnect_count > reconnects and h.state is CS.ONLINE)
    assert h.last_recovery_s is not None and h.last_recovery_s >= 0


def test_watchdog_reconnect_request_reopens_stream(run_source, good_uri):
    src, _ = run_source(good_uri)
    h = src.health
    assert wait_until(lambda: h.state is CS.ONLINE)
    attempts = h.connect_attempts
    src.request_reconnect()
    assert wait_until(lambda: h.connect_attempts > attempts and h.state is CS.ONLINE)
