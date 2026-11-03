"""Fault-injection campaign against the RUNNING application (spec 11.5, 19.3). Measures, per fault:
detection time (fault -> dashboard/API shows the camera not ONLINE), recovery time (fault cleared -> ONLINE again),
impact on the unaffected cameras (min processed FPS during the fault) and lost events.

Prereqs: docker compose up -d; rtsp_simulator.py serve; API running with PPE_ENABLE_FAULT_INJECTION=1.
  python scripts/benchmark/fault_injection.py [--only camera_unplugged,worker_crash]
Writes docs/benchmarks/fault_injection.json and .md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8")  # reports contain non-ASCII (Windows console is cp1252)
API = "http://127.0.0.1:8000"
SIM = [sys.executable, str(ROOT / "scripts" / "demo" / "rtsp_simulator.py")]
LIVE = ("ONLINE", "DEGRADED")


def env() -> dict[str, str]:
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            out[k.strip()] = v.strip()
    return out


class Api:
    def __init__(self) -> None:
        self.token = ""
        self.login()

    def login(self) -> None:
        body = {"username": "admin", "password": env()["PPE_ADMIN_PASSWORD"]}
        self.token = self.call("POST", "/api/v1/auth/login", body, auth=False)["access_token"]

    def call(self, method: str, path: str, body: dict | None = None, auth: bool = True, timeout: float = 10):
        req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.token}"} if auth else {})})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - localhost
            raw = r.read()
            return json.loads(raw) if raw else None

    def cams(self) -> dict[str, dict]:
        try:
            return {c["camera_id"]: c for c in self.call("GET", "/api/v1/metrics")["cameras"]}
        except Exception:  # noqa: BLE001 - API/DB briefly unavailable during some faults
            return {}

    def metrics(self) -> dict | None:
        try:
            return self.call("GET", "/api/v1/metrics")
        except Exception:  # noqa: BLE001
            return None


def sim(*args: str) -> None:
    subprocess.run(SIM + list(args), check=True, capture_output=True, timeout=180)


def worker_counter(name: str) -> float:
    try:
        with urllib.request.urlopen("http://127.0.0.1:9101/metrics", timeout=5) as r:  # noqa: S310
            text = r.read().decode()
    except Exception:  # noqa: BLE001
        return float("nan")
    return sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
               if line.startswith(name) and not line.startswith("#"))


def wait(api: Api, cam: str | None, pred, timeout: float, others_fps: list[float], victims: set[str]) -> float | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        cams = api.cams()
        for k, c in cams.items():
            if k not in victims and c["target_fps"] > 0:
                others_fps.append(c["processed_fps"])
        hit = pred(cams) if cam is None else bool(cams.get(cam)) and pred(cams[cam])  # empty = worker gone
        if hit:
            return round(time.time() - t0, 2)
        time.sleep(0.25)
    return None


def scenario(api: Api, name: str, cam: str | None, inject, clear, detect_pred=None, recover_pred=None,
             detect_timeout: float = 30, recover_timeout: float = 90, hold_s: float = 3) -> dict:
    detect_pred = detect_pred or (lambda c: c["state"] != "ONLINE")
    recover_pred = recover_pred or (lambda c: c["state"] == "ONLINE")
    victims = {cam} if cam else set()
    others: list[float] = []
    print(f"--- {name}", flush=True)
    settle(api)
    before = api.cams()
    t_inject = time.time()
    inject()
    detected = wait(api, cam, detect_pred, detect_timeout, others, victims)
    time.sleep(min(hold_s, 6))  # let the state progress (e.g. DEGRADED -> OFFLINE) before sampling it
    state_seen = (api.cams().get(cam) or {}).get("state") if cam else None
    reason = (api.cams().get(cam) or {}).get("state_reason") if cam else None
    time.sleep(hold_s)
    t_clear = time.time()
    clear()
    recovered = wait(api, cam, recover_pred, recover_timeout, others, victims)
    res = {"scenario": name, "camera": cam, "detected_after_s": detected, "state_during_fault": state_seen,
           "reason": reason, "recovered_after_clear_s": recovered,
           "total_outage_s": round(time.time() - t_inject, 1) if recovered is not None else None,
           "other_cameras_min_processed_fps": round(min(others), 2) if others else None,
           "other_cameras_mean_processed_fps": round(sum(others) / len(others), 2) if others else None,
           "reconnects_before": (before.get(cam) or {}).get("reconnect_count") if cam else None,
           "fault_wall_time": t_inject, "clear_wall_time": t_clear}
    print(json.dumps(res), flush=True)
    return res


def settle(api: Api, timeout: float = 120) -> None:
    """Wait until every enabled camera is ONLINE and processing >= 90 % of target (no carry-over between faults)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        cams = api.cams()
        ok = [c for c in cams.values() if c["target_fps"] > 0]
        if ok and all(c["state"] == "ONLINE" and c["processed_fps"] >= 0.9 * c["target_fps"] for c in ok):
            return
        time.sleep(1)


def all_live(cams: dict) -> bool:
    enabled = [c for c in cams.values() if c["target_fps"] > 0]
    return bool(enabled) and all(c["state"] == "ONLINE" for c in enabled)


def run(only: set[str] | None) -> list[dict]:
    api = Api()
    out: list[dict] = []
    S = scenario  # noqa: N806

    def want(n: str) -> bool:
        return not only or n in only

    if want("camera_unplugged"):
        out.append(S(api, "camera_unplugged", "cam03", lambda: sim("stop", "cam03"), lambda: sim("start", "cam03")))
    if want("stalled_video"):
        out.append(S(api, "stalled_video", "cam04", lambda: sim("stall", "cam04"), lambda: sim("resume", "cam04")))
    if want("rtsp_timeout"):
        out.append(S(api, "rtsp_timeout", "cam06", lambda: sim("net", "cam06", "timeout", "--timeout", "1"),
                     lambda: sim("net", "cam06", "clear")))
    if want("connection_refused"):
        out.append(S(api, "connection_refused", "cam07", lambda: sim("proxy", "cam07", "off"),
                     lambda: sim("proxy", "cam07", "on")))
    if want("network_jitter"):
        out.append(S(api, "network_latency_jitter_800ms", "cam05",
                     lambda: sim("net", "cam05", "latency", "--latency", "800", "--jitter", "400"),
                     lambda: sim("net", "cam05", "clear"),
                     detect_pred=lambda c: c["state"] != "ONLINE" or (c["frame_age_at_inference_p95_s"] or 0) > 0.5,
                     detect_timeout=40))
    if want("decoder_error"):
        e0 = (api.cams().get("cam09") or {}).get("decode_errors", 0)
        out.append(S(api, "decoder_exception", "cam09",
                     lambda: api.call("POST", "/api/v1/system/faults", {"kind": "decoder_error", "camera_id": "cam09"}),
                     lambda: None, detect_pred=lambda c: c["state"] != "ONLINE" or c["decode_errors"] > e0,
                     hold_s=0))
    if want("invalid_credentials"):
        out.append(S(api, "invalid_credentials_config", "cam08",
                     lambda: api.call("PATCH", "/api/v1/cameras/cam08", {"credential_ref": "env:WRONG_CREDS"}),
                     lambda: api.call("PATCH", "/api/v1/cameras/cam08", {"credential_ref": "env:RTSP_READ"}),
                     detect_pred=lambda c: c["state"] == "ERROR"))
    if want("inference_exception"):
        e0 = worker_counter("inference_errors_total")
        r = S(api, "inference_exception", None,
              lambda: api.call("POST", "/api/v1/system/faults", {"kind": "inference_exception"}), lambda: None,
              detect_pred=lambda cams: worker_counter("inference_errors_total") > e0, recover_pred=all_live, hold_s=0)
        r["inference_errors_delta"] = worker_counter("inference_errors_total") - e0
        out.append(r)
    if want("gpu_oom"):
        e0 = worker_counter("inference_errors_total")
        r = S(api, "gpu_oom_simulated", None, lambda: api.call("POST", "/api/v1/system/faults", {"kind": "gpu_oom"}),
              lambda: None, detect_pred=lambda cams: worker_counter("inference_errors_total") > e0,
              recover_pred=all_live, hold_s=0)
        out.append(r)
    if want("nvr_restart"):
        out.append(S(api, "nvr_restart_all_cameras", None, lambda: sim("nvr-restart"), lambda: None,
                     detect_pred=lambda cams: any(c["state"] not in LIVE for c in cams.values()),
                     recover_pred=all_live, hold_s=0, recover_timeout=120))
    if want("worker_crash"):
        out.append(S(api, "worker_process_crash", None,
                     lambda: api.call("POST", "/api/v1/system/faults", {"kind": "crash"}), lambda: None,
                     detect_pred=lambda cams: not (api.metrics() or {"worker": {"up": True}})["worker"]["up"],
                     recover_pred=all_live, hold_s=0, recover_timeout=180))
    if want("disk_full"):
        f0 = worker_counter("ppe_evidence_failures_total")
        r = S(api, "disk_full_60s", None,
              lambda: api.call("POST", "/api/v1/system/faults", {"kind": "disk_full", "duration_s": 60}),
              lambda: None, detect_pred=lambda cams: True, recover_pred=all_live, hold_s=60)
        r["evidence_failures_during_fault"] = worker_counter("ppe_evidence_failures_total") - f0
        r["pipeline_kept_running"] = r["other_cameras_min_processed_fps"] is not None
        out.append(r)
    if want("db_outage"):
        sp0 = worker_counter("ppe_events_spooled_total")
        ev0 = (api.metrics() or {}).get("aggregate", {}).get("events_last_hour")
        r = S(api, "database_outage_60s", None,
              lambda: subprocess.run(["docker", "compose", "stop", "postgres"], cwd=ROOT, capture_output=True),
              lambda: subprocess.run(["docker", "compose", "start", "postgres"], cwd=ROOT, capture_output=True),
              detect_pred=lambda cams: not (api.metrics() or {"database": {"up": True}})["database"]["up"],
              recover_pred=lambda cams: (api.metrics() or {"database": {"up": False}})["database"]["up"]
              and (api.metrics() or {}).get("spool", {}).get("pending", 1) == 0, hold_s=60, recover_timeout=120)
        time.sleep(5)
        r["events_spooled_during_outage"] = worker_counter("ppe_events_spooled_total") - sp0
        r["spool_pending_after"] = (api.metrics() or {}).get("spool", {}).get("pending")
        r["events_last_hour_before"] = ev0
        r["events_last_hour_after"] = (api.metrics() or {}).get("aggregate", {}).get("events_last_hour")
        out.append(r)
    return out


def write(new: list[dict]) -> None:
    d = ROOT / "docs" / "benchmarks"
    old = json.loads((d / "fault_injection.json").read_text()) if (d / "fault_injection.json").exists() else []
    names = {r["scenario"] for r in new}  # --only reruns replace their scenarios, keep the others
    results = [r for r in old if r["scenario"] not in names] + new
    (d / "fault_injection.json").write_text(json.dumps(results, indent=1))
    lines = ["# Fault-injection results (generated by scripts/benchmark/fault_injection.py)", "",
             "Detection = fault injected → API reports the camera not ONLINE (or the stated signal). Recovery = fault "
             "cleared → ONLINE again. Other cameras = processed FPS of every unaffected camera sampled during the "
             "whole scenario (target 5 fps).", "",
             "| scenario | camera | detected after | state / reason during fault | recovered after clear | "
             "other cams min / mean fps | notes |", "|---|---|---|---|---|---|---|"]
    for r in results:
        notes = {k: v for k, v in r.items() if k in ("inference_errors_delta", "evidence_failures_during_fault",
                                                     "events_spooled_during_outage", "spool_pending_after")}
        lines.append(f"| {r['scenario']} | {r['camera'] or 'all'} | {r['detected_after_s']} s | "
                     f"{r['state_during_fault'] or ''} {('— ' + r['reason']) if r['reason'] else ''} | "
                     f"{r['recovered_after_clear_s']} s | {r['other_cameras_min_processed_fps']} / "
                     f"{r['other_cameras_mean_processed_fps']} | {json.dumps(notes) if notes else ''} |")
    (d / "fault_injection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    a = ap.parse_args()
    write(run(set(a.only.split(",")) if a.only else None))
