"""Build per-head feature records for the fusion benchmark (scripts/ml/fusion_benchmark.py).

For every labelled helmet(0)/head(1) box in the val and test splits (not train: the classifier trained on train
crops) it records the label (1 = no_helmet violation), the head classifier's P(no_helmet) on the same 25%-expanded
crop used to build its training data, the PPE detector's best-IoU (>= 0.3) matched helmet and head confidences on the
full image (production imgsz/conf), and box geometry. Boxes narrower than 12 px are skipped, as in the classifier data.

Usage (project venv, GPU):
  cd backend && UV_PROJECT_ENVIRONMENT=<venv> uv run python ../scripts/ml/fusion_features.py [--force]
Writes <out>/features_{val,test}.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home() / "ppe-work"


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box `a` against an (N, 4) array `b`."""
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def best_conf(gt: np.ndarray, xyxy: np.ndarray, conf: np.ndarray, min_iou: float = 0.3) -> float:
    if not len(xyxy):
        return 0.0
    o = iou(gt, xyxy)
    j = int(o.argmax())
    return float(conf[j]) if o[j] >= min_iou else 0.0


def split_records(split: str, data: Path, det, cls, device: str) -> list[dict]:
    no_helmet = next(k for k, v in cls.names.items() if v == "no_helmet")
    rows = []
    for img_path in sorted((data / "images" / split).glob("*.png")):
        lbl = data / "labels" / split / f"{img_path.stem}.txt"
        boxes = [line.split() for line in lbl.read_text().splitlines() if line.strip()] if lbl.exists() else []
        boxes = [(int(c), *map(float, r)) for c, *r in boxes if c in ("0", "1")]
        if not boxes:
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        boxes = [b for b in boxes if b[3] * W >= 12]
        if not boxes:
            continue
        # Production settings (configs/models.example.yaml): ppe imgsz 640 conf 0.25, classifier imgsz 96, fp16.
        d = det.predict(str(img_path), imgsz=640, conf=0.25, quantize=16, device=device, verbose=False)[0].boxes
        dx, dc, dk = d.xyxy.cpu().numpy(), d.conf.cpu().numpy(), d.cls.cpu().numpy().astype(int)
        crops, recs = [], []
        for c, x, y, w, h in boxes:
            cx, cy, ex, ey = x * W, y * H, w * W * 0.625, h * H * 0.625  # same crop as prepare_hardhat_dataset.py
            box = (max(0, cx - ex), max(0, cy - ey), min(W, cx + ex), min(H, cy + ey))
            crop = img.crop(box)
            crops.append(np.asarray(crop)[:, :, ::-1].copy())  # ultralytics expects BGR arrays (production frames)
            gt = np.array([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H])
            recs.append(
                {
                    "image": img_path.stem,
                    "label": int(c == 1),
                    "det_helmet_conf": round(best_conf(gt, dx[dk == 0], dc[dk == 0]), 4),
                    "det_head_conf": round(best_conf(gt, dx[dk == 1], dc[dk == 1]), 4),
                    "box_w_px": round(w * W, 1),
                    "box_h_px": round(h * H, 1),
                    "area_rel": round(w * h, 6),
                    "aspect": round(w * W / (h * H), 3),
                    "crop_px": crop.size[0] * crop.size[1],
                }
            )
        for r, res in zip(recs, cls.predict(crops, imgsz=96, quantize=16, device=device, verbose=False), strict=True):
            r["cls_p_no_helmet"] = round(float(res.probs.data[no_helmet]), 4)
        rows += recs
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(WORK / "hardhat" / "yolo"))
    ap.add_argument("--out", default=str(WORK / "fusion"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det, cls = YOLO(str(ROOT / "models" / "ppe_yolo26s.pt")), YOLO(str(ROOT / "models" / "head_cls_yolo26n.pt"))
    for split in ("val", "test"):
        dst = out / f"features_{split}.jsonl"
        if dst.exists() and not a.force:
            print(f"{dst} exists (use --force to recompute)")
            continue
        rows = split_records(split, Path(a.data), det, cls, a.device)
        dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{split}: {len(rows)} records, {sum(r['label'] for r in rows)} violations -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
