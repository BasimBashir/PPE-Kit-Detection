"""Per-camera ingest thread: RTSP (PyAV) or looping file source -> LatestFrameSlot (implementation-plan 4.2).

Blocked RTSP reads are broken by PyAV's read timeout (interrupt callback), so no other thread ever touches the
container - closing it cross-thread risks native crashes. The watchdog only raises a reconnect flag.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import av
import av.error

from app.camera.health import Backoff, CameraHealth, CameraState
from app.camera.uri import CredentialError, resolve_uri
from app.config.settings import IngestConfig
from app.core.redaction import redact
from app.streaming.frame import Frame
from app.streaming.slot import LatestFrameSlot

log = logging.getLogger(__name__)

class StreamConfigError(Exception):
    """Invalid stream configuration (bad URI) - not fixable by retrying."""


# Configuration errors: retried only at max backoff and shown as ERROR. Deliberately explicit: PyAV's
# InvalidDataError subclasses ValueError but is a transient network symptom (observed on RTSP timeouts), and 404 on an
# NVR usually means the camera is offline right now - both must recover quickly.
FATAL_ERRORS: tuple[type[BaseException], ...] = (
    av.error.HTTPUnauthorizedError, av.error.HTTPForbiddenError, CredentialError, StreamConfigError,
    FileNotFoundError,
)


@dataclass(slots=True)
class CameraRuntimeConfig:
    id: str
    stream_uri: str
    credential_ref: str | None = None
    transport: str = "tcp"
    enabled: bool = True
    target_fps: float = 5.0


@dataclass
class FaultHooks:
    """Fault injection (tests / demo only)."""

    decoder_error_once: bool = False
    stall_until_mono: float = 0.0  # frames are discarded until this time -> watchdog sees a stall


class _Reconnect(Exception):
    pass


def file_path_of(uri: str) -> Path:
    p = unquote(urlsplit(uri).path)
    if len(p) >= 3 and p[0] == "/" and p[2] == ":":  # file:///C:/x.mp4 -> C:/x.mp4
        p = p[1:]
    return Path(p)


class CameraSource(threading.Thread):
    def __init__(self, camera: CameraRuntimeConfig, ingest: IngestConfig, slot: LatestFrameSlot,
                 health: CameraHealth, convert_fps: float, fault_hooks: FaultHooks | None = None):
        super().__init__(name=f"src-{camera.id}", daemon=True)
        self.cam, self.cfg, self.slot, self.health = camera, ingest, slot, health
        self.convert_period = 1.0 / max(convert_fps, 0.1)
        self.hooks = fault_hooks or FaultHooks()
        self._stop_evt = threading.Event()
        self._reconnect = threading.Event()
        self._frame_id = 0
        self.width: int | None = None
        self.height: int | None = None
        self.is_file = camera.stream_uri.startswith("file:")

    # ------------------------------------------------------------ control
    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)

    def request_reconnect(self) -> None:
        self._reconnect.set()

    @property
    def stopped(self) -> bool:
        return self._stop_evt.is_set()

    # ------------------------------------------------------------ thread
    def run(self) -> None:
        backoff = Backoff(self.cfg.backoff_base_s, self.cfg.backoff_max_s)
        while not self._stop_evt.is_set():
            self.health.on_connecting()
            fatal = False
            container = None
            try:
                container = self._open()
                self._read(container, backoff)
            except _Reconnect:
                pass  # watchdog already set OFFLINE with the stall reason
            except Exception as e:  # noqa: BLE001 - every failure must lead to a reconnect, never kill the thread
                if self._stop_evt.is_set():
                    break
                fatal = isinstance(e, FATAL_ERRORS)
                msg = redact(f"{type(e).__name__}: {e}")
                self.health.on_error(msg, fatal=fatal)
                log.warning("camera %s %s: %s", self.cam.id, "fatal error" if fatal else "error", msg)
            finally:
                if container is not None:
                    with contextlib.suppress(Exception):  # best-effort release of an already broken stream
                        container.close()
            if self._stop_evt.is_set():
                break
            self._reconnect.clear()
            delay = self.cfg.backoff_max_s if fatal else backoff.next_delay()
            self._stop_evt.wait(delay)
        log.info("camera %s source stopped", self.cam.id)

    def _open(self):  # -> av.container.InputContainer
        if self.is_file:
            path = file_path_of(self.cam.stream_uri)
            if not path.exists():
                raise FileNotFoundError(f"video file not found: {path.name}")
            return av.open(str(path))
        try:
            url = resolve_uri(self.cam.stream_uri, self.cam.credential_ref)
        except CredentialError:
            raise
        except ValueError as e:
            raise StreamConfigError(str(e)) from e
        return av.open(url, options={"rtsp_transport": self.cam.transport},
                       timeout=(self.cfg.open_timeout_s, self.cfg.read_timeout_s))

    def _read(self, container, backoff: Backoff) -> None:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate or stream.guessed_rate
        nominal = float(rate) if rate else None
        self.health.on_connected(nominal)
        last_convert = 0.0
        while True:  # files loop forever; RTSP ends only by error/timeout
            t0_wall = t0_pts = None
            for frame in container.decode(stream):
                if self._stop_evt.is_set():
                    return
                if self._reconnect.is_set():
                    raise _Reconnect
                if self.hooks.decoder_error_once:
                    self.hooks.decoder_error_once = False
                    raise RuntimeError("injected decoder error")
                now = time.monotonic()
                if now < self.hooks.stall_until_mono:
                    continue
                if self.is_file and frame.time is not None:  # real-time pacing for deterministic file sources
                    if t0_wall is None:
                        t0_wall, t0_pts = now, frame.time
                    ahead = (frame.time - t0_pts) - (now - t0_wall)
                    if ahead > 0 and self._stop_evt.wait(ahead):
                        return
                    now = time.monotonic()
                self.health.on_frame()
                if self.health.stable_for() >= self.cfg.stable_reset_s:
                    backoff.reset()
                if now - last_convert >= self.convert_period:
                    last_convert = now
                    img = frame.to_ndarray(format="bgr24")
                    self.height, self.width = img.shape[:2]
                    self._frame_id += 1
                    self.slot.put(Frame(self.cam.id, self._frame_id, img, frame.time, time.time(), now))
            if not self.is_file:
                raise EOFError("stream ended")
            container.seek(0)

    @property
    def state(self) -> CameraState:
        return self.health.state


class CameraWatchdog(threading.Thread):
    """Evaluates every camera's health at ~2 Hz and asks stalled sources to reconnect (FR-CAM-004)."""

    def __init__(self, period_s: float = 0.5):
        super().__init__(name="camera-watchdog", daemon=True)
        self.period_s = period_s
        self._items: dict[str, tuple[CameraHealth, CameraSource]] = {}
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()

    def register(self, health: CameraHealth, source: CameraSource) -> None:
        with self._lock:
            self._items[health.camera_id] = (health, source)

    def unregister(self, camera_id: str) -> None:
        with self._lock:
            self._items.pop(camera_id, None)

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        while not self._stop_evt.wait(self.period_s):
            with self._lock:
                items = list(self._items.values())
            for health, source in items:
                prev = health.state
                if health.evaluate() is CameraState.OFFLINE and prev is not CameraState.OFFLINE:
                    source.request_reconnect()
