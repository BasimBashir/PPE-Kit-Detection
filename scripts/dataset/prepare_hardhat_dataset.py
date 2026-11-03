"""Download the CC0 "Hard Hat Workers" dataset (Voxel51/hard-hat-detection on Hugging Face) and convert
it to Ultralytics YOLO format, plus a head-crop classification dataset (helmet / no_helmet).

Split strategy (spec 12.2): the source has no video/site metadata, so near-duplicate leakage is limited by
grouping images with an identical 8x8 average-hash into the same split before a seeded 80/10/10 split.

Usage: python scripts/dataset/prepare_hardhat_dataset.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

BASE = "https://huggingface.co/datasets/Voxel51/hard-hat-detection/resolve/main/"
CLASSES = ["helmet", "head", "person"]  # YOLO class ids 0,1,2


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:  # noqa: S310 - fixed host
                fh.write(r.read())
            tmp.replace(dest)
            return dest
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    return dest


def ahash(path: Path) -> int:
    img = np.asarray(Image.open(path).convert("L").resize((8, 8)), dtype=np.float32)
    bits = (img > img.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    raw = out / "raw"
    samples = json.loads(fetch(BASE + "samples.json", raw / "samples.json").read_text())["samples"]
    print(f"{len(samples)} samples; downloading images...", flush=True)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda s: fetch(BASE + s["filepath"], raw / s["filepath"]), samples))

    groups: dict[int, list[dict]] = defaultdict(list)
    for s in samples:
        groups[ahash(raw / s["filepath"])].append(s)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    n = len(keys)
    split_of = {k: ("train" if i < 0.8 * n else "val" if i < 0.9 * n else "test") for i, k in enumerate(keys)}

    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for k, members in groups.items():
        split = split_of[k]
        for s in members:
            src = raw / s["filepath"]
            stem = Path(s["filepath"]).stem
            img_dst = out / "yolo" / "images" / split / f"{stem}.png"
            lbl_dst = out / "yolo" / "labels" / split / f"{stem}.txt"
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            if not img_dst.exists():
                img_dst.write_bytes(src.read_bytes())
            lines = []
            img = None
            for d in (s.get("ground_truth") or {}).get("detections", []):
                label = d["label"].lower()
                if label not in CLASSES:
                    continue
                x, y, w, h = d["bounding_box"]
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{CLASSES.index(label)} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
                stats[split][label] += 1
                # Head-crop classifier data: helmet => helmet, head => no_helmet (crop expanded 25%).
                if label in ("helmet", "head") and w * s["metadata"]["width"] >= 12:
                    img = img or Image.open(src).convert("RGB")
                    W, H = img.size
                    cx, cy, ex, ey = (x + w / 2) * W, (y + h / 2) * H, w * W * 0.625, h * H * 0.625
                    crop = img.crop((max(0, cx - ex), max(0, cy - ey), min(W, cx + ex), min(H, cy + ey)))
                    cls = "helmet" if label == "helmet" else "no_helmet"
                    cdir = out / "cls" / split / cls
                    cdir.mkdir(parents=True, exist_ok=True)
                    crop.save(cdir / f"{stem}_{d['_id']['$oid']}.jpg", quality=92)
            lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""))
            stats[split]["images"] += 1

    (out / "yolo" / "data.yaml").write_text(
        f"path: {(out / 'yolo').as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    report = {"source": "huggingface.co/datasets/Voxel51/hard-hat-detection", "license": "CC0-1.0",
              "seed": args.seed, "hash_groups": n, "splits": {k: dict(v) for k, v in stats.items()}}
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
