"""Supervises the pipeline worker process from the API process (ADR-001/010, spec 11.5 "inference worker crash").

Spawns the worker (spawn start method), consumes telemetry, detects death via process exit or missing heartbeats,
and restarts with bounded exponential backoff. The API and dashboard stay up throughout.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from app.camera.health import Backoff
from app.observability import metrics as m
from app.services.live_state import LiveState

log = logging.getLogger(__name__)
HEARTBEAT_TIMEOUT_S = 10.0
STARTUP_GRACE_S = 120.0  # model load + CUDA init + warmup


class WorkerSupervisor:
    def __init__(self, live: LiveState, on_message: Callable[[dict[str, Any]], None] | None = None,
                 target: Callable[..., None] | None = None):
        self.live = live
        self.on_message = on_message or (lambda _m: None)
        self.ctx = mp.get_context("spawn")
        self.target = target
        self.proc: mp.process.BaseProcess | None = None
        self.telemetry: Any = None
        self.commands: Any = None
        self.restarts = 0
        self._stop = threading.Event()
        self._restart_requested = threading.Event()
        self._backoff = Backoff(1.0, 30.0)
        self._started_mono = 0.0
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        self._spawn()
        for fn, name in ((self._pump, "telemetry-pump"), (self._watch, "worker-watch")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def _spawn(self) -> None:
        from app.workers.pipeline_worker import run_worker

        self.telemetry = self.ctx.Queue(maxsize=512)
        self.commands = self.ctx.Queue()
        self.proc = self.ctx.Process(target=self.target or run_worker, name="ppe-worker",
                                     args=(self.telemetry, self.commands, os.getpid()), daemon=True)
        self.proc.start()
        self._started_mono = time.monotonic()
        log.info("pipeline worker spawned (pid %s)", self.proc.pid)

    def send(self, cmd: dict[str, Any]) -> bool:
        try:
            self.commands.put_nowait(cmd)
            return True
        except Exception:  # noqa: BLE001
            return False

    def request_restart(self) -> None:
        self._restart_requested.set()

    def _pump(self) -> None:
        while not self._stop.is_set():
            q = self.telemetry
            try:
                msg = q.get(timeout=0.5)
            except (queue.Empty, OSError, EOFError, ValueError):
                continue
            try:
                self.live.apply(msg)
                if msg.get("type") == "hello":
                    m.ppe_worker_up.set(1)
                    self._backoff.reset()
                self.on_message(msg)
            except Exception:
                log.exception("telemetry handling failed")

    def _alive(self) -> tuple[bool, str]:
        if self.proc is None or not self.proc.is_alive():
            return False, f"exited with code {self.proc.exitcode if self.proc else None}"
        since_start = time.monotonic() - self._started_mono
        if self.live.worker["up"]:
            if time.monotonic() - self.live.last_heartbeat > HEARTBEAT_TIMEOUT_S:
                return False, f"no heartbeat for {HEARTBEAT_TIMEOUT_S:.0f} s"
        elif since_start > STARTUP_GRACE_S:
            return False, "did not start within grace period"
        return True, ""

    def _watch(self) -> None:
        while not self._stop.wait(1.0):
            alive, why = self._alive()
            if self.live.worker["up"] and self.live.worker.get("started_at"):
                m.ppe_worker_uptime_seconds.set(time.time() - self.live.worker["started_at"])
            if self._restart_requested.is_set():
                self._restart_requested.clear()
                alive, why = False, "restart requested by operator"
            if alive:
                continue
            log.error("pipeline worker down: %s", why)
            m.ppe_worker_up.set(0)
            self.live.worker_down()
            self._kill()
            delay = self._backoff.next_delay()
            if self._stop.wait(delay):
                return
            self.restarts += 1
            self.live.worker["restarts"] = self.restarts
            m.ppe_worker_restarts_total.inc()
            self._spawn()

    def _kill(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(5)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(5)

    def stop(self, timeout: float = 15.0) -> None:
        self._stop.set()
        if self.proc is not None and self.proc.is_alive():
            self.send({"type": "shutdown"})
            self.proc.join(timeout)
        self._kill()
        m.ppe_worker_up.set(0)
