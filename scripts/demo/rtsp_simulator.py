"""10-camera RTSP simulator with fault injection (demo / test only - clearly simulated sources).

Each clip <STREAMS_DIR>/camNN.mp4 is looped by ffmpeg into MediaMTX (the "NVR"); the app reads it through Toxiproxy
(rtsp://localhost:185NN/camNN) so network faults can be injected per camera.

  python scripts/demo/rtsp_simulator.py serve [--count 10]              # start N publishers + control API :8890
  python scripts/demo/rtsp_simulator.py streams 4                         # change the number of live streams
  python scripts/demo/rtsp_simulator.py status
  python scripts/demo/rtsp_simulator.py stop cam03 | start cam03          # camera unplugged / back
  python scripts/demo/rtsp_simulator.py stall cam03 | resume cam03        # stalled video (TCP open, no frames)
  python scripts/demo/rtsp_simulator.py nvr-restart                       # restart MediaMTX
  python scripts/demo/rtsp_simulator.py net cam03 latency --latency 800 --jitter 400
  python scripts/demo/rtsp_simulator.py net cam03 timeout --timeout 1     # RTSP timeout (data stops)
  python scripts/demo/rtsp_simulator.py net cam03 clear
  python scripts/demo/rtsp_simulator.py proxy cam03 off|on                # connection refused
Every fault is appended to docs/benchmarks/scratch/fault_log.jsonl with a wall-clock timestamp.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "docs" / "benchmarks" / "scratch"
CONTROL = "http://127.0.0.1:8890"
TOXI = "http://127.0.0.1:8474"
USERINFO = re.compile(r"://[^/@\s]+@")


def load_env() -> dict[str, str]:
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            env[k.strip()] = v.strip()
    return env


def log_fault(action: str, camera: str | None, **extra) -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with open(SCRATCH / "fault_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), "action": action, "camera_id": camera, **extra}) + "\n")


def http(method: str, url: str, body: dict | None = None) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 - localhost control endpoints
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


class Publisher:
    def __init__(self, cam: str, clip: Path, url: str):
        self.cam, self.clip, self.url = cam, clip, url
        self.proc: subprocess.Popen | None = None
        self.wanted = True
        self.stalled = False
        self.restarts = 0
        self.log = SCRATCH / "sim_logs" / f"{cam}.log"

    def start(self) -> None:
        self.wanted = True
        if self.proc and self.proc.poll() is None:
            return
        self.log.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-stream_loop", "-1", "-i", str(self.clip),
               "-c", "copy", "-f", "rtsp", "-rtsp_transport", "tcp", self.url]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, text=True,
                                     creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        self.stalled = False

    def _pump(self, proc: subprocess.Popen) -> None:
        with open(self.log, "a", encoding="utf-8") as fh:
            for line in proc.stderr:  # type: ignore[union-attr]
                fh.write(time.strftime("%H:%M:%S ") + USERINFO.sub("://***@", line))
                fh.flush()

    def stop(self) -> None:
        self.wanted = False
        if self.proc and self.proc.poll() is None:
            if self.stalled:
                psutil.Process(self.proc.pid).resume()
            self.proc.kill()
            self.proc.wait(5)
        self.stalled = False

    def stall(self) -> None:
        if self.proc and self.proc.poll() is None and not self.stalled:
            psutil.Process(self.proc.pid).suspend()
            self.stalled = True

    def resume(self) -> None:
        if self.proc and self.stalled:
            psutil.Process(self.proc.pid).resume()
            self.stalled = False

    def status(self) -> dict:
        alive = bool(self.proc and self.proc.poll() is None)
        return {"camera_id": self.cam, "publishing": alive and not self.stalled, "stalled": self.stalled,
                "pid": self.proc.pid if alive else None, "restarts": self.restarts, "wanted": self.wanted}


class Simulator:
    def __init__(self, cams: list[str] | None, count: int | None = None):
        env = load_env()
        streams = Path(env.get("STREAMS_DIR", "data/streams"))
        streams = streams if streams.is_absolute() else ROOT / streams
        clips = sorted(streams.glob("cam*.mp4"))
        if cams:
            clips = [c for c in clips if c.stem in cams]
        if not clips:
            sys.exit(f"no camNN.mp4 clips in {streams} - run scripts/demo/fetch_demo_videos.py")
        auth = f"{env['RTSP_PUBLISH_USER']}:{env['RTSP_PUBLISH_PASS']}"
        self.pubs = {c.stem: Publisher(c.stem, c, f"rtsp://{auth}@localhost:8554/{c.stem}") for c in clips}
        self.count = len(self.pubs) if count is None else max(0, min(count, len(self.pubs)))
        self.lock = threading.Lock()
        self.stop_evt = threading.Event()
        self.on_shutdown = lambda: None

    def set_count(self, n: int) -> list[dict]:
        """First n streams publish, the rest are stopped (e.g. 10 -> 4 cameras)."""
        self.count = max(0, min(n, len(self.pubs)))
        for i, p in enumerate(self.pubs.values()):
            if i < self.count:
                p.start()
            else:
                p.stop()
        log_fault("set_stream_count", None, count=self.count)
        return [p.status() for p in self.pubs.values()]

    def supervise(self) -> None:
        while not self.stop_evt.wait(2.0):
            with self.lock:
                for p in self.pubs.values():
                    if p.wanted and (p.proc is None or p.proc.poll() is not None):
                        p.restarts += 1
                        p.start()

    def action(self, path: str, body: dict) -> tuple[int, object]:
        parts = [x for x in path.strip("/").split("/") if x]
        with self.lock:
            if parts == ["cameras"]:
                return 200, [p.status() for p in self.pubs.values()]
            if parts == ["streams"]:
                if "count" in body:
                    self.set_count(int(body["count"]))
                return 200, {"count": self.count, "available": len(self.pubs),
                             "cameras": [p.status() for p in self.pubs.values()]}
            if parts == ["shutdown"]:
                threading.Thread(target=self.on_shutdown, daemon=True).start()
                return 200, {"ok": True}
            if parts == ["nvr", "restart"]:
                log_fault("nvr_restart", None)
                r = subprocess.run(["docker", "compose", "restart", "mediamtx"], cwd=ROOT, capture_output=True,
                                   text=True, timeout=120)
                return (200 if r.returncode == 0 else 500), {"ok": r.returncode == 0}
            if len(parts) == 3 and parts[0] == "cameras" and parts[1] in self.pubs:
                cam, act, p = parts[1], parts[2], self.pubs[parts[1]]
                if act in ("stop", "start", "stall", "resume"):
                    getattr(p, act)()
                    log_fault(act, cam)
                    return 200, p.status()
                if act in ("disable_proxy", "enable_proxy"):
                    code, _ = http("POST", f"{TOXI}/proxies/{cam}", {"enabled": act == "enable_proxy"})
                    log_fault(act, cam)
                    return code, {"ok": code == 200}
                if act == "network":
                    return self._toxic(cam, body)
        return 404, {"error": "unknown action"}

    def _toxic(self, cam: str, body: dict) -> tuple[int, object]:
        kind = body.get("kind")
        if kind == "clear":
            _, raw = http("GET", f"{TOXI}/proxies/{cam}/toxics")
            for t in json.loads(raw or "[]"):
                http("DELETE", f"{TOXI}/proxies/{cam}/toxics/{t['name']}")
            log_fault("network_clear", cam)
            return 200, {"ok": True}
        attrs = {k: v for k, v in body.items() if k != "kind"}
        toxic = {"latency": "latency", "jitter": "latency", "bandwidth": "bandwidth", "timeout": "timeout",
                 "reset_peer": "reset_peer"}.get(kind or "")
        if toxic is None:
            return 400, {"error": f"unknown kind {kind}"}
        code, raw = http("POST", f"{TOXI}/proxies/{cam}/toxics",
                         {"name": f"{kind}_down", "type": toxic, "stream": "downstream", "toxicity": 1.0,
                          "attributes": attrs})
        log_fault(f"network_{kind}", cam, **attrs)
        return code, json.loads(raw) if raw else {}

    def shutdown(self) -> None:
        self.stop_evt.set()
        with self.lock:
            for cam, p in self.pubs.items():
                p.stop()
                http("POST", f"{TOXI}/proxies/{cam}", {"enabled": True})
                self._toxic(cam, {"kind": "clear"})


def reap_orphans() -> int:
    """Kill ffmpeg publishers left behind by a simulator that was killed hard (they would block the paths)."""
    n = 0
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower().startswith("ffmpeg") and any(
                    "localhost:8554/cam" in a for a in (p.info["cmdline"] or [])):
                p.kill()
                n += 1
        except psutil.Error:
            continue
    return n


def serve(cams: list[str] | None, count: int | None = None) -> None:
    if (n := reap_orphans()):
        print(f"reaped {n} orphaned publishers from a previous run", flush=True)
    sim = Simulator(cams, count)
    for i, p in enumerate(sim.pubs.values()):
        if i < sim.count:
            p.start()
        else:
            p.wanted = False
    threading.Thread(target=sim.supervise, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def _go(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            code, out = sim.action(self.path, body)
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _go

        def log_message(self, *_a) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 8890), H)
    sim.on_shutdown = srv.shutdown  # POST /shutdown: graceful stop (publishers killed, toxics cleared)
    print(f"simulating {len(sim.pubs)} cameras; control API on {CONTROL}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sim.shutdown()
        srv.server_close()
        print("simulator stopped", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--cameras", help="comma-separated subset, e.g. cam01,cam02")
    s.add_argument("--count", type=int, help="number of streams to publish at start (default: all clips)")
    sub.add_parser("status")
    sub.add_parser("streams").add_argument("count", type=int)
    sub.add_parser("nvr-restart")
    for c in ("stop", "start", "stall", "resume"):
        sub.add_parser(c).add_argument("camera")
    n = sub.add_parser("net")
    n.add_argument("camera")
    n.add_argument("kind", choices=["latency", "jitter", "bandwidth", "timeout", "reset_peer", "clear"])
    n.add_argument("--latency", type=int)
    n.add_argument("--jitter", type=int)
    n.add_argument("--rate", type=int)
    n.add_argument("--timeout", type=int)
    pr = sub.add_parser("proxy")
    pr.add_argument("camera")
    pr.add_argument("state", choices=["on", "off"])
    a = ap.parse_args()

    if a.cmd == "serve":
        serve(a.cameras.split(",") if a.cameras else None, a.count)
        return 0
    if a.cmd == "streams":
        code, out = http("POST", f"{CONTROL}/streams", {"count": a.count})
        print(code, out)
        return 0 if code < 400 else 1
    if a.cmd == "status":
        code, out = http("GET", f"{CONTROL}/cameras")
    elif a.cmd == "nvr-restart":
        code, out = http("POST", f"{CONTROL}/nvr/restart")
    elif a.cmd in ("stop", "start", "stall", "resume"):
        code, out = http("POST", f"{CONTROL}/cameras/{a.camera}/{a.cmd}")
    elif a.cmd == "net":
        body = {"kind": a.kind, **{k: getattr(a, k) for k in ("latency", "jitter", "rate", "timeout")
                                   if getattr(a, k) is not None}}
        code, out = http("POST", f"{CONTROL}/cameras/{a.camera}/network", body)
    else:
        code, out = http("POST", f"{CONTROL}/cameras/{a.camera}/{'enable' if a.state == 'on' else 'disable'}_proxy")
    print(code, out)
    return 0 if code < 400 else 1


if __name__ == "__main__":
    sys.exit(main())
