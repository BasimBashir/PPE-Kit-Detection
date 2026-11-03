"""Camera lifecycle state machine and reconnect backoff (FR-CAM-003/004/005, implementation-plan 4.1)."""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum

from app.config.settings import IngestConfig
from app.observability import metrics as m


class CameraState(StrEnum):
    DISABLED = "DISABLED"
    CONNECTING = "CONNECTING"
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"
    RECOVERING = "RECOVERING"
    ERROR = "ERROR"


LIVE = (CameraState.ONLINE, CameraState.DEGRADED)


class Backoff:
    """Bounded exponential backoff with jitter: min(max, base * 2**attempt) * U(0.5, 1.0)."""

    def __init__(self, base_s: float, max_s: float, rng: random.Random | None = None):
        self.base_s, self.max_s = base_s, max_s
        self.rng = rng or random.Random()
        self.attempt = 0

    def next_delay(self) -> float:
        d = min(self.max_s, self.base_s * (2 ** min(self.attempt, 30))) * self.rng.uniform(0.5, 1.0)
        self.attempt += 1
        return d

    def reset(self) -> None:
        self.attempt = 0


class CameraHealth:
    """Thread-safe: written by the camera's source thread, evaluated by the watchdog thread."""

    FPS_WINDOW_S = 2.0
    FPS_GRACE_S = 3.0  # do not judge decode FPS until this long after connect
    STALLED_FPS_RATIO = 0.1  # below 10 % of nominal the stream is unusable => OFFLINE + reconnect

    def __init__(self, camera_id: str, cfg: IngestConfig, nominal_fps: float | None = None,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time):
        self.camera_id, self.cfg, self.nominal_fps = camera_id, cfg, nominal_fps
        self.clock, self.wall = clock, wall
        self._lock = threading.Lock()
        self.state = CameraState.CONNECTING
        self.reason = "starting"
        self.since_wall = wall()
        self.since_mono = clock()
        self.ever_online = False
        self.connect_attempts = 0
        self.reconnect_count = 0
        self.decode_errors = 0
        self.online_seconds = 0.0
        self.last_recovery_s: float | None = None
        self.last_frame_mono: float | None = None
        self.last_frame_wall: float | None = None
        self.connected_mono: float | None = None
        self._outage_start: float | None = None
        self._frame_times: deque[float] = deque()
        self._last_eval = clock()
        self._publish_state()

    # ------------------------------------------------------------ internals (lock held)
    def _publish_state(self) -> None:
        for s in CameraState:
            m.camera_connection_state.labels(self.camera_id, s.value).set(1 if s is self.state else 0)

    def _set(self, state: CameraState, reason: str) -> None:
        if state is not self.state:
            self.since_wall, self.since_mono = self.wall(), self.clock()
            self.state = state
            self._publish_state()
        self.reason = reason

    def _fps(self, now: float) -> float:
        while self._frame_times and now - self._frame_times[0] > self.FPS_WINDOW_S:
            self._frame_times.popleft()
        return len(self._frame_times) / self.FPS_WINDOW_S

    def _mark_outage(self, now: float) -> None:
        if self._outage_start is None:
            self._outage_start = self.last_frame_mono if self.last_frame_mono is not None else now

    # ------------------------------------------------------------ source-thread events
    def on_connecting(self) -> None:
        with self._lock:
            self.connect_attempts += 1
            if self.connect_attempts > 1:
                self.reconnect_count += 1
                m.camera_reconnect_total.labels(self.camera_id).inc()
            if self.state is CameraState.ERROR:
                return  # keep the fatal reason visible while retrying
            if self.ever_online:
                self._set(CameraState.RECOVERING, "reconnecting")
            else:
                self._set(CameraState.CONNECTING, "connecting")

    def on_connected(self, nominal_fps: float | None) -> None:
        with self._lock:
            if nominal_fps:
                self.nominal_fps = nominal_fps
                m.camera_source_fps.labels(self.camera_id).set(nominal_fps)
            self.connected_mono = self.clock()
            self._frame_times.clear()

    def on_frame(self) -> None:
        with self._lock:
            now = self.clock()
            self.last_frame_mono, self.last_frame_wall = now, self.wall()
            self._frame_times.append(now)
            if self.state not in LIVE:
                if self._outage_start is not None and self.ever_online:
                    self.last_recovery_s = now - self._outage_start
                    m.camera_last_recovery_seconds.labels(self.camera_id).set(self.last_recovery_s)
                self._outage_start = None
                self.ever_online = True
                self._set(CameraState.ONLINE, "receiving frames")

    def on_error(self, reason: str, fatal: bool = False) -> None:
        with self._lock:
            self.decode_errors += 1
            m.camera_decode_errors_total.labels(self.camera_id).inc()
            self._mark_outage(self.clock())
            self._set(CameraState.ERROR if fatal else CameraState.OFFLINE, reason)

    def on_disabled(self) -> None:
        with self._lock:
            self._set(CameraState.DISABLED, "disabled by configuration")

    # ------------------------------------------------------------ watchdog
    def evaluate(self) -> CameraState:
        with self._lock:
            now = self.clock()
            if self.state in LIVE:
                dt = now - self._last_eval
                self.online_seconds += dt
                m.camera_online_seconds_total.labels(self.camera_id).inc(dt)
            self._last_eval = now
            fps = self._fps(now)
            m.camera_decoded_fps.labels(self.camera_id).set(fps)
            if self.last_frame_mono is not None:
                m.camera_last_frame_age_seconds.labels(self.camera_id).set(now - self.last_frame_mono)
            if self.state not in LIVE or self.last_frame_mono is None:
                return self.state
            age = now - self.last_frame_mono
            if age > self.cfg.stall_timeout_s:
                self._mark_outage(now)
                self._set(CameraState.OFFLINE, f"stalled: no frame for {age:.1f} s")
            elif age > self.cfg.degraded_frame_age_s:
                self._set(CameraState.DEGRADED, f"late frames: last frame {age:.1f} s ago")
            elif (self.nominal_fps and self.connected_mono is not None
                  and now - self.connected_mono >= self.FPS_GRACE_S
                  and fps < self.STALLED_FPS_RATIO * self.nominal_fps):
                # A trickle of straggler frames must not keep a dead stream out of OFFLINE (observed in fault tests).
                self._mark_outage(now)
                self._set(CameraState.OFFLINE, f"stalled: decode rate {fps:.1f} fps (nominal {self.nominal_fps:.0f})")
            elif (self.nominal_fps and self.connected_mono is not None
                  and now - self.connected_mono >= self.FPS_GRACE_S
                  and fps < self.cfg.degraded_fps_ratio * self.nominal_fps):
                self._set(CameraState.DEGRADED, f"low decode rate {fps:.1f} fps (nominal {self.nominal_fps:.0f})")
            else:
                self._set(CameraState.ONLINE, "receiving frames")
            return self.state

    # ------------------------------------------------------------ readers
    @property
    def decoded_fps(self) -> float:
        with self._lock:
            return self._fps(self.clock())

    def last_frame_age(self) -> float | None:
        with self._lock:
            return None if self.last_frame_mono is None else self.clock() - self.last_frame_mono

    def stable_for(self) -> float:
        """Seconds continuously ONLINE (0 when not ONLINE)."""
        with self._lock:
            return self.clock() - self.since_mono if self.state is CameraState.ONLINE else 0.0
