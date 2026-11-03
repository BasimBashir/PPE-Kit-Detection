"""Download the Pexels demo clips listed in demo_videos.yaml and transcode them for RTSP looping.

Output: <streams_dir>/<camera_id>.mp4, H.264, no B-frames, 1 s GOP, no audio, so MediaMTX can
re-publish them with `-c copy` and every loop restart begins on a keyframe.

Usage: python scripts/demo/fetch_demo_videos.py [--streams-dir DIR] [--height 720] [--max-fps 30]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def probe_fps(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(Fraction(out))


def download(pexels_id: int, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    req = urllib.request.Request(
        f"https://www.pexels.com/download/video/{pexels_id}/", headers={"User-Agent": "Mozilla/5.0"}
    )
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:  # noqa: S310 - fixed https host
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)


def transcode(src: Path, dst: Path, height: int, max_fps: int) -> None:
    fps = min(round(probe_fps(src)), max_fps)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an",
         "-vf", f"scale=-2:{height},fps={fps}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-bf", "0", "-g", str(fps), "-keyint_min", str(fps), "-sc_threshold", "0",
         "-movflags", "+faststart", str(dst)],
        check=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams-dir", default=os.environ.get("STREAMS_DIR", str(ROOT / "data" / "streams")))
    ap.add_argument("--raw-dir", default=None, help="cache for original downloads (default: <streams-dir>/raw)")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-fps", type=int, default=30)
    args = ap.parse_args()

    streams = Path(args.streams_dir)
    raw = Path(args.raw_dir) if args.raw_dir else streams / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    streams.mkdir(parents=True, exist_ok=True)
    manifest = yaml.safe_load((HERE / "demo_videos.yaml").read_text(encoding="utf-8"))

    for cam in manifest["cameras"]:
        src = raw / f"{cam['pexels_id']}.mp4"
        dst = streams / f"{cam['camera_id']}.mp4"
        print(f"{cam['camera_id']}: pexels {cam['pexels_id']} ({cam['expected']})", flush=True)
        download(cam["pexels_id"], src)
        if not dst.exists():
            transcode(src, dst, args.height, args.max_fps)
    print(f"done -> {streams}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
