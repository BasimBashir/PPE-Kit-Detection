"""Demo-mode RTSP simulator management (spec 0.1 #11: clearly labelled simulator so the system is testable without
cameras). The API starts scripts/demo/rtsp_simulator.py when it is not already running and proxies its control
API so stream counts and camera/network faults can be driven from the dashboard."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from app.config.settings import REPO_ROOT, DemoConfig

log = logging.getLogger(__name__)
SCRIPT = REPO_ROOT / "scripts" / "demo" / "rtsp_simulator.py"


class SimulatorUnavailable(Exception):
    pass


class SimulatorClient:
    def __init__(self, cfg: DemoConfig):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None

    def call(self, method: str, path: str, body: dict[str, Any] | None = None, timeout: float = 130) -> Any:
        req = urllib.request.Request(self.cfg.simulator_url + path, method=method,  # noqa: S310 - http(s) validated
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - configured localhost URL
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            raise SimulatorUnavailable(f"simulator returned {e.code}") from e
        except (urllib.error.URLError, OSError) as e:
            raise SimulatorUnavailable("RTSP simulator is not running") from e

    def is_up(self) -> bool:
        try:
            self.call("GET", "/cameras", timeout=2)
            return True
        except SimulatorUnavailable:
            return False

    def ensure_running(self) -> bool:
        """Start the simulator (once) if nothing answers on its control port. Returns True when it is up."""
        if self.is_up():
            log.info("RTSP simulator already running at %s", self.cfg.simulator_url)
            return True
        if not SCRIPT.exists():
            log.warning("simulator script missing: %s", SCRIPT)
            return False
        log.info("starting RTSP simulator with %d streams", self.cfg.initial_streams)
        self.proc = subprocess.Popen(  # noqa: S603 - fixed interpreter + repo script
            [sys.executable, str(SCRIPT), "serve", "--count", str(self.cfg.initial_streams)], cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        for _ in range(40):
            time.sleep(0.25)
            if self.is_up():
                return True
            if self.proc.poll() is not None:
                log.error("RTSP simulator exited (code %s) - are the demo clips fetched? "
                          "(python scripts/demo/fetch_demo_videos.py)", self.proc.returncode)
                return False
        return False

    def stop_if_owned(self) -> None:
        """Only stop a simulator this API process started (a manually started one keeps running)."""
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self.call("POST", "/shutdown", {}, timeout=5)
            self.proc.wait(15)
        except Exception:  # noqa: BLE001
            self.proc.kill()
