"""Reproducible pipeline benchmarks (spec 11, 19.4). Never edit results by hand - rerun.

Each profile runs the real PipelineWorker in a fresh subprocess against the RTSP simulator (start it first:
`python scripts/demo/rtsp_simulator.py serve`), warms up, then measures for --seconds. Uses a dedicated database
`ppe_bench` (recreated per profile) so benchmark events never mix with demo data. The API must NOT be running
(GPU contention would invalidate the numbers).

  python scripts/benchmark/run_benchmark.py --all                 # every profile below
  python scripts/benchmark/run_benchmark.py --profile cams10_fps5  # one profile
  python scripts/benchmark/run_benchmark.py --summarize            # rebuild docs/benchmarks/README tables
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8")  # reports contain non-ASCII (Windows console is cp1252)
OUT = ROOT / "docs" / "benchmarks" / "results"
sys.path.insert(0, str(ROOT / "backend"))

# name -> (cameras, target_fps, model overrides, association overrides)
PROFILES: dict[str, dict] = {
    "cams1_fps5": {"cams": 1, "fps": 5},
    "cams4_fps5": {"cams": 4, "fps": 5},
    "cams10_fps2": {"cams": 10, "fps": 2},
    "cams10_fps5": {"cams": 10, "fps": 5},
    "cams10_fps10": {"cams": 10, "fps": 10},
    "cams10_fps30_saturation": {"cams": 10, "fps": 30},
    # Architecture ablation at the baseline profile (interview: "justify every extra model by measurement")
    "ablation_A_coco_geometry": {"cams": 10, "fps": 5, "person": "yolo26s.pt", "head_roi": "geometry",
                                 "classifier": False},
    "ablation_B_pose_geometry": {"cams": 10, "fps": 5, "head_roi": "geometry", "classifier": False},
    "ablation_C_pose_keypoints": {"cams": 10, "fps": 5, "head_roi": "keypoints", "classifier": False},
    "ablation_D_full_cascade": {"cams": 10, "fps": 5, "head_roi": "keypoints", "classifier": True},
}


def load_env() -> None:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            os.environ.setdefault(k.strip(), v.strip())


def hw_info() -> dict:
    import psutil
    import torch

    info = {"os": platform.platform(), "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    import ultralytics

    info["ultralytics"] = ultralytics.__version__
    return info


def run_profile(name: str, seconds: float, warmup: float) -> dict:
    import multiprocessing as mp
    import queue

    from app.config.settings import Settings, load_app_config, load_camera_seeds, load_models_config, load_rule_seeds
    from app.core.logging import setup_logging
    from app.db.base import make_engine, make_session_factory, session_scope
    from app.db.migrate import upgrade_to_head
    from app.models import Camera, Event
    from app.repositories.misc import seed_if_empty
    from app.workers.pipeline_worker import PipelineWorker
    from prometheus_client import REGISTRY
    from sqlalchemy import text

    setup_logging("bench")
    p = PROFILES[name]
    base = Settings()
    url = base.database_url.rsplit("/", 1)[0] + "/ppe_bench"
    admin = make_engine(base.database_url.rsplit("/", 1)[0] + "/postgres", pool_size=1)
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as c:
        c.execute(text("DROP DATABASE IF EXISTS ppe_bench WITH (FORCE)"))
        c.execute(text("CREATE DATABASE ppe_bench"))
    admin.dispose()
    upgrade_to_head(url)
    s = Settings(PPE_DATABASE_URL=url, ppe_evidence_dir=Path(f"docs/benchmarks/scratch/evidence_{name}"),
                 ppe_spool_dir=Path(f"docs/benchmarks/scratch/spool_{name}"))
    sf = make_session_factory(make_engine(url))
    rules, zones = load_rule_seeds(s)
    cams = load_camera_seeds(s)
    with session_scope(sf) as ss:
        seed_if_empty(ss, cams, rules, zones, None)
        for i, c in enumerate(ss.query(Camera).order_by(Camera.id)):
            c.enabled = i < p["cams"]
    app_cfg = load_app_config(s)
    app_cfg.pipeline.target_fps = p["fps"]
    app_cfg.pipeline.max_batch = max(10, p["cams"])
    if "head_roi" in p:
        app_cfg.association.head_roi = p["head_roi"]
    if "classifier" in p:
        app_cfg.association.use_classifier = p["classifier"]
    models = load_models_config(s)
    if "person" in p:
        models.person.weights = p["person"]
    if p.get("classifier") is False:
        models.classifier = None

    tq, cq = mp.Queue(1024), mp.Queue()
    w = PipelineWorker(s, app_cfg, models, tq, cq)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()

    def counter(metric: str) -> float:
        return sum(sm.value for m in REGISTRY.collect() if m.name == metric for sm in m.samples
                   if sm.name.endswith("_total"))

    samples, last_status, last_stats = [], None, None
    t_start = time.time()
    measuring = False
    c0 = {}
    while time.time() - t_start < warmup + seconds:
        try:
            msg = tq.get(timeout=1)
        except queue.Empty:
            continue
        if msg["type"] == "camera_status":
            last_status = msg
        elif msg["type"] == "stats":
            last_stats = msg
            if not measuring and time.time() - t_start >= warmup:
                measuring = True
                c0 = {k: counter(k) for k in ("ppe_uncertain_observations", "ppe_classifier_invocations",
                                              "camera_frames_dropped", "inference_errors", "camera_reconnect")}
                with session_scope(sf) as ss:
                    c0["events"] = ss.query(Event).count()
            if measuring:
                r = msg["resources"]
                samples.append({"t": round(time.time() - t_start, 1), "agg_fps": msg["aggregate_processed_fps"],
                                "rss": r.get("rss_bytes"), "cpu": r.get("cpu_percent"),
                                "sys_cpu": r.get("system_cpu_percent"), "gpu": r.get("gpu_util_percent"),
                                "vram": r.get("gpu_memory_used_bytes")})
    cq.put({"type": "shutdown"})
    t.join(20)
    with session_scope(sf) as ss:
        events = ss.query(Event).count() - c0.get("events", 0)
    deltas = {k: counter(k) - v for k, v in c0.items() if k != "events"}

    def stat(key: str) -> dict:
        vals = [x[key] for x in samples if x[key] is not None]
        return {"mean": round(sum(vals) / len(vals), 2), "max": round(max(vals), 2)} if vals else {}

    enabled = [c for c in (last_status or {}).get("cameras", []) if c["target_fps"] > 0]
    result = {
        "profile": name, "config": p, "measured_seconds": seconds, "warmup_seconds": warmup,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hardware": hw_info(),
        "model_versions": w.model_versions,
        "aggregate_processed_fps": stat("agg_fps"), "target_aggregate_fps": p["cams"] * p["fps"],
        "per_camera": [{k: c[k] for k in ("camera_id", "state", "source_fps", "decoded_fps", "processed_fps",
                                          "frame_age_at_inference_p95_s", "frames_dropped", "reconnect_count")}
                       for c in enabled],
        "inference_ms": (last_stats or {}).get("inference", {}),
        "batch_size_median": (last_stats or {}).get("batch_size_mean"),
        "worker_rss_bytes": stat("rss"), "worker_rss_growth_bytes": (samples[-1]["rss"] - samples[0]["rss"])
        if len(samples) > 1 else None,
        "worker_cpu_percent": stat("cpu"), "system_cpu_percent": stat("sys_cpu"), "gpu_util_percent": stat("gpu"),
        "gpu_memory_used_bytes": stat("vram"),
        "events_created": events, "events_per_camera_hour": round(events / (p["cams"] * seconds / 3600), 1),
        "counters_during_measurement": deltas,
        "uncertain_per_processed_frame": round(deltas.get("ppe_uncertain_observations", 0)
                                               / max(1, stat("agg_fps").get("mean", 0) * seconds), 3),
        "samples": samples,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("profile", "aggregate_processed_fps", "target_aggregate_fps",
                                             "gpu_util_percent", "events_created")}))
    return result


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    if not rows:
        print("no results")
        return
    hw = rows[0]["hardware"]
    lines = ["# Benchmark results (generated by scripts/benchmark/run_benchmark.py --summarize)", "",
             f"Hardware: {hw.get('gpu')} · {hw['cpu']} ({hw['logical_cpus']} threads) · {hw['ram_gb']} GB RAM · "
             f"{hw['os']} · torch {hw['torch']} (CUDA {hw['cuda']}) · ultralytics {hw['ultralytics']}", "",
             "Sources: 720p H.264 demo clips looped through MediaMTX + Toxiproxy (RTSP over TCP, simulated cameras). "
             "Measurement boundaries: *processed FPS* = frames that completed detection+tracking+association per "
             "second (5 s window); *frame age* = decode → inference start; stage latencies are per frame "
             "(batch time / batch size), p95 over the last 60 s.", "",
             "| profile | cams × target fps | processed fps (mean) | frame age p95 (worst cam) | person ms p95 | "
             "PPE ms p95 | assoc ms p95 | GPU % (mean/max) | VRAM GB (max) | worker RSS GB | RSS growth MB | "
             "CPU % host | events / cam-h | uncertain / frame |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        inf = r["inference_ms"]
        age = max((c["frame_age_at_inference_p95_s"] or 0) for c in r["per_camera"]) if r["per_camera"] else None
        lines.append(
            f"| {r['profile']} | {r['config']['cams']} × {r['config']['fps']} | "
            f"{r['aggregate_processed_fps'].get('mean')} / {r['target_aggregate_fps']} | {age} s | "
            f"{inf.get('person', {}).get('p95_ms', 0):.1f} | {inf.get('ppe', {}).get('p95_ms', 0):.1f} | "
            f"{inf.get('association', {}).get('p95_ms', 0):.2f} | {r['gpu_util_percent'].get('mean')}/"
            f"{r['gpu_util_percent'].get('max')} | {r['gpu_memory_used_bytes'].get('max', 0) / 1e9:.2f} | "
            f"{r['worker_rss_bytes'].get('max', 0) / 1e9:.2f} | "
            f"{(r['worker_rss_growth_bytes'] or 0) / 1e6:.0f} | {r['system_cpu_percent'].get('mean')} | "
            f"{r['events_per_camera_hour']} | {r['uncertain_per_processed_frame']} |")
    (ROOT / "docs" / "benchmarks" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=list(PROFILES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=20)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    load_env()
    if a.summarize:
        summarize()
        return 0
    if a.all:
        for name in PROFILES:
            subprocess.run([sys.executable, __file__, "--profile", name, "--seconds", str(a.seconds),
                            "--warmup", str(a.warmup)], check=False, cwd=ROOT)
        summarize()
        return 0
    if a.profile:
        os.chdir(ROOT)
        run_profile(a.profile, a.seconds, a.warmup)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
