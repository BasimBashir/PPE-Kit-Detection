"""Soak test against the RUNNING application (spec 19.4, NFR-REL-003): samples /api/v1/metrics every --interval s
for --hours and reports drift (RSS / VRAM growth), throughput stability, camera uptime, reconnects and events.

  python scripts/benchmark/soak.py --hours 1
Writes docs/benchmarks/soak_<hours>h.csv and soak_<hours>h.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8")  # reports contain non-ASCII (Windows console is cp1252)
API = "http://127.0.0.1:8000"


def env() -> dict[str, str]:
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            out[k.strip()] = v.strip()
    return out


def call(path: str, token: str | None = None, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body else None,
                                 method="POST" if body else "GET",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def slope_per_hour(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den * 3600


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--interval", type=float, default=10.0)
    a = ap.parse_args()
    token = call("/api/v1/auth/login", body={"username": "admin", "password": env()["PPE_ADMIN_PASSWORD"]})[
        "access_token"]
    tag = f"{a.hours:g}h"
    out_csv = ROOT / "docs" / "benchmarks" / f"soak_{tag}.csv"
    rows: list[dict] = []
    t0 = time.time()
    first = call("/api/v1/metrics", token)
    while time.time() - t0 < a.hours * 3600:
        try:
            m = call("/api/v1/metrics", token)
        except Exception as e:  # noqa: BLE001 - record gaps, keep soaking
            rows.append({"t_s": round(time.time() - t0), "error": type(e).__name__})
            time.sleep(a.interval)
            continue
        r, ag = m["resources"], m["aggregate"]
        rows.append({"t_s": round(time.time() - t0), "processed_fps": ag["processed_fps"], "online": ag["online"],
                     "worker_rss_mb": round((r["worker_rss_bytes"] or 0) / 2**20, 1),
                     "api_rss_mb": round((r["api_rss_bytes"] or 0) / 2**20, 1),
                     "vram_mb": round((r["gpu_memory_used_bytes"] or 0) / 2**20, 1),
                     "torch_alloc_mb": round((r["torch_cuda_allocated_bytes"] or 0) / 2**20, 1),
                     "gpu_util": r["gpu_util_percent"], "cpu": r["system_cpu_percent"],
                     "gpu_temp_c": r["gpu_temperature_c"],
                     "frame_age_p95_s": m["latency"]["frame_age_at_inference_p95_s"],
                     "events_last_hour": ag["events_last_hour"], "worker_restarts": m["worker"]["restarts"],
                     "reconnects": sum(c["reconnect_count"] for c in m["cameras"]), "error": ""})
        if len(rows) % 30 == 0:
            print(json.dumps(rows[-1]), flush=True)
        time.sleep(a.interval)
    ok = [x for x in rows if not x.get("error")]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ok[0]) if ok else ["t_s", "error"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    last = call("/api/v1/metrics", token)
    warm = [x for x in ok if x["t_s"] >= 300] or ok  # steady state: exclude first 5 min
    ts = [x["t_s"] for x in warm]

    def s(key: str) -> dict:
        v = [x[key] for x in warm if x[key] is not None]
        return {"min": min(v), "mean": round(statistics.mean(v), 2), "max": max(v)} if v else {}

    summary = {
        "duration_h": a.hours, "samples": len(rows), "sample_errors": len(rows) - len(ok),
        "processed_fps": s("processed_fps"), "cameras_online": s("online"), "frame_age_p95_s": s("frame_age_p95_s"),
        "worker_rss_mb": s("worker_rss_mb"), "worker_rss_slope_mb_per_h": round(slope_per_hour(
            ts, [x["worker_rss_mb"] for x in warm]), 1),
        "api_rss_slope_mb_per_h": round(slope_per_hour(ts, [x["api_rss_mb"] for x in warm]), 1),
        "vram_mb": s("vram_mb"), "vram_slope_mb_per_h": round(slope_per_hour(ts, [x["vram_mb"] for x in warm]), 1),
        "gpu_util": s("gpu_util"), "cpu": s("cpu"), "gpu_temp_c": s("gpu_temp_c"),
        "worker_restarts": last["worker"]["restarts"] - first["worker"]["restarts"],
        "reconnects": sum(c["reconnect_count"] for c in last["cameras"]) - sum(
            c["reconnect_count"] for c in first["cameras"]),
        "per_camera_uptime_ratio_this_worker_run": {c["camera_id"]: c["uptime_ratio"] for c in last["cameras"]},
    }
    md = [f"# Soak test ({a.hours:g} h, generated by scripts/benchmark/soak.py)", "",
          "Full application (API + worker + 10 simulated RTSP cameras at 5 processed FPS each), sampled every "
          f"{a.interval:g} s. Steady-state statistics exclude the first 5 minutes. Raw samples: `{out_csv.name}`.", "",
          "```json", json.dumps(summary, indent=1), "```"]
    (ROOT / "docs" / "benchmarks" / f"soak_{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
