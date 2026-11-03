"""One-command start for reviewers: python scripts/start_demo.py [--streams 10] [--weights-url URL] [--cpu]

Idempotent - every step is skipped when already done:
  1. prerequisite check (uv, docker, ffmpeg, node/npm, NVIDIA GPU optional)
  2. .env with random local secrets            (scripts/bootstrap/init_env.py)
  3. docker compose up -d (PostgreSQL, MediaMTX, Toxiproxy, Prometheus, Grafana) and wait for PostgreSQL
  4. backend dependencies                       (uv sync)
  5. demo clips -> STREAMS_DIR                  (scripts/demo/fetch_demo_videos.py, Pexels)
  6. model weights in models/: committed via Git LFS; else download (--weights-url, a .zip) or train on the GPU
  7. dashboard build                            (npm ci && npm run build)
  8. start the app on http://127.0.0.1:8000 - in demo mode it starts the RTSP simulator itself
Stop with Ctrl+C (the app stops the simulator it started); `docker compose down` stops the services.
Stdlib only, so it runs before any dependency is installed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
WEIGHTS = ("yolo26s-pose.pt", "ppe_yolo26s.pt", "head_cls_yolo26n.pt")


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def run(cmd: list[str], cwd: Path = ROOT, check: bool = True, **kw) -> subprocess.CompletedProcess:
    print("   $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check, **kw)


def have(exe: str) -> bool:
    return shutil.which(exe) is not None


def env() -> dict[str, str]:
    out = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                out[k.strip()] = v.strip()
    return out


def uv_py(*args: str, **kw) -> subprocess.CompletedProcess:
    return run(["uv", "run", "--project", str(BACKEND), "python", *args], **kw)


def gpu_available() -> bool:
    if not have("nvidia-smi"):
        return False
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0


def write_override(name: str, text: str) -> None:
    """configs/<name>.yaml overrides the committed *.example.yaml (git-ignored local choice)."""
    p = ROOT / "configs" / f"{name}.yaml"
    if not p.exists():
        p.write_text(text, encoding="utf-8")
        print(f"   wrote {p.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streams", type=int, default=None, help="initial number of simulated cameras (1-10)")
    ap.add_argument("--weights-url", help="zip with yolo26s-pose.pt, ppe_yolo26s.pt, head_cls_yolo26n.pt")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference (slow: defaults to 2 streams @ 2 fps)")
    ap.add_argument("--work-dir", default=str(Path.home() / "ppe-work"), help="dataset / training scratch space")
    ap.add_argument("--no-start", action="store_true", help="prepare everything but do not start the app")
    a = ap.parse_args()

    step("1/8 prerequisites")
    missing = [x for x in ("uv", "docker", "ffmpeg", "npm") if not have(x)]
    if missing:
        print(f"   missing on PATH: {', '.join(missing)} - see README 'Prerequisites'")
        return 1
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        print("   Docker is installed but not running - start Docker Desktop / dockerd and retry")
        return 1
    cpu = a.cpu or not gpu_available()
    print(f"   inference device: {'CPU (no NVIDIA GPU detected or --cpu)' if cpu else 'NVIDIA GPU'}")

    step("2/8 .env")
    if not (ROOT / ".env").exists():
        run([sys.executable, str(ROOT / "scripts" / "bootstrap" / "init_env.py")])
    else:
        print("   .env exists")
    if cpu:
        write_override("models", (ROOT / "configs" / "models.example.yaml").read_text(encoding="utf-8")
                       .replace("device: cuda:0", "device: cpu").replace("half: true", "half: false"))
    streams = a.streams if a.streams is not None else (2 if cpu else 10)
    app_example = (ROOT / "configs" / "app.example.yaml").read_text(encoding="utf-8")
    if a.streams is not None or cpu:
        txt = app_example.replace("initial_streams: 10", f"initial_streams: {max(0, min(streams, 10))}")
        if cpu:
            txt = txt.replace("target_fps: 5 ", "target_fps: 2 ")
        p = ROOT / "configs" / "app.yaml"
        p.write_text(txt, encoding="utf-8")
        print(f"   configs/app.yaml: initial_streams={streams}{' target_fps=2 (CPU)' if cpu else ''}")

    step("3/8 docker services")
    run(["docker", "compose", "up", "-d"])
    for _ in range(60):
        r = subprocess.run(["docker", "compose", "ps", "postgres", "--format", "{{.Health}}"], cwd=ROOT,
                           capture_output=True, text=True)
        if "healthy" in r.stdout:
            break
        time.sleep(2)
    else:
        print("   PostgreSQL did not become healthy - check `docker compose logs postgres`")
        return 1

    step("4/8 backend dependencies (uv sync)")
    run(["uv", "sync", "--project", str(BACKEND)])

    step("5/8 demo clips")
    streams_dir = Path(env().get("STREAMS_DIR", "data/streams"))
    streams_dir = streams_dir if streams_dir.is_absolute() else ROOT / streams_dir
    if len(list(streams_dir.glob("cam*.mp4"))) >= 10:
        print(f"   10 clips present in {streams_dir}")
    else:
        uv_py(str(ROOT / "scripts" / "demo" / "fetch_demo_videos.py"), "--streams-dir", str(streams_dir))

    step("6/8 model weights")
    models = ROOT / "models"
    pointers = [w for w in WEIGHTS if (models / w).exists() and (models / w).stat().st_size < 1024]
    if pointers:  # cloned without Git LFS: the files are ~130-byte LFS pointers, not weights
        print(f"   {', '.join(pointers)}: Git LFS pointer files - run `git lfs install && git lfs pull`, then retry")
        return 1
    if all((models / w).exists() for w in WEIGHTS[1:]):
        print("   ppe_yolo26s.pt and head_cls_yolo26n.pt present")
    elif a.weights_url:
        with urllib.request.urlopen(a.weights_url, timeout=300) as r:  # noqa: S310 - user-supplied release URL
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(models)
        print(f"   extracted weights into {models}")
    elif not cpu:
        work = Path(a.work_dir)
        print("   no weights found: preparing the CC0 dataset and training (~30-40 min on an RTX 3090)")
        uv_py(str(ROOT / "scripts" / "dataset" / "prepare_hardhat_dataset.py"), "--out", str(work / "hardhat"))
        uv_py(str(ROOT / "scripts" / "dataset" / "train_ppe_detector.py"), "--data",
              str(work / "hardhat" / "yolo" / "data.yaml"), "--project", str(work / "runs"), "--install")
        uv_py(str(ROOT / "scripts" / "dataset" / "train_head_classifier.py"), "--data", str(work / "hardhat" / "cls"),
              "--project", str(work / "runs"), "--install")
    else:
        print("   no weights and no GPU: pass --weights-url <release zip> (training on CPU is impractical)")
        return 1

    step("7/8 dashboard build")
    if (ROOT / "frontend" / "dist" / "index.html").exists():
        print("   frontend/dist exists")
    else:
        npm = "npm.cmd" if os.name == "nt" else "npm"
        run([npm, "ci"], cwd=ROOT / "frontend")
        run([npm, "run", "build"], cwd=ROOT / "frontend")

    pw = env().get("PPE_ADMIN_PASSWORD", "(see PPE_ADMIN_PASSWORD in .env)")
    print("\n" + "=" * 72)
    print("  PPE Intelligence  ->  http://127.0.0.1:8000")
    print(f"  login: admin / {pw}        (local secret generated into .env)")
    print("  'Demo control' page: number of streams, camera/network/worker faults")
    print("  Grafana http://127.0.0.1:3000 (admin / GRAFANA_ADMIN_PASSWORD) · Prometheus http://127.0.0.1:9090")
    print("=" * 72 + "\n")
    if a.no_start:
        return 0
    step("8/8 starting the application (Ctrl+C to stop)")
    with contextlib.suppress(KeyboardInterrupt):
        uv_py("-m", "uvicorn", "app.main:factory", "--factory", "--host", "127.0.0.1", "--port", "8000",
              cwd=BACKEND, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
