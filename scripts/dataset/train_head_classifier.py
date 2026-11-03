"""Train the optional head-crop classifier (helmet / no_helmet) with YOLO26n-cls and evaluate on the TEST split.

Usage: python scripts/dataset/train_head_classifier.py --data <work>/hardhat/cls [--epochs 30] [--install]
Writes docs/ml/head_classifier_eval.json (top-1 accuracy, confusion counts, accuracy vs confidence threshold).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default=str(Path.home() / "ppe-work" / "runs"))
    ap.add_argument("--name", default="head_cls_yolo26n")
    ap.add_argument("--install", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO

    t0 = time.time()
    YOLO("yolo26n-cls.pt").train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=256, device=a.device, seed=42,
                                 project=a.project, name=a.name, exist_ok=True, patience=10, workers=8)
    best = Path(a.project) / a.name / "weights" / "best.pt"
    model = YOLO(str(best))
    names = model.names
    # Per-image evaluation on the held-out test split, incl. abstention behaviour for the UNCERTAIN threshold.
    rows = []
    test = Path(a.data) / "test"
    for cls_dir in sorted(p for p in test.iterdir() if p.is_dir()):
        files = sorted(cls_dir.glob("*.jpg"))
        for i in range(0, len(files), 512):
            for _f, r in zip(files[i:i + 512], model.predict([str(x) for x in files[i:i + 512]], imgsz=a.imgsz,
                                                             device=a.device, verbose=False), strict=True):
                rows.append((cls_dir.name, names[int(r.probs.top1)], float(r.probs.top1conf)))
    confusion: dict[str, dict[str, int]] = {}
    for truth, pred, _ in rows:
        confusion.setdefault(truth, {}).setdefault(pred, 0)
        confusion[truth][pred] += 1
    by_threshold = {}
    for th in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
        kept = [(t, p) for t, p, c in rows if c >= th]
        by_threshold[str(th)] = {"coverage": round(len(kept) / len(rows), 4),
                                 "accuracy_on_kept": round(sum(t == p for t, p in kept) / max(len(kept), 1), 4)}
    out = {
        "model": a.name, "weights_sha256_12": hashlib.sha256(best.read_bytes()).hexdigest()[:12],
        "split": "test", "n": len(rows), "top1_accuracy": round(sum(t == p for t, p, _ in rows) / len(rows), 4),
        "confusion": confusion, "uncertain_threshold_tradeoff": by_threshold,
        "train_minutes": round((time.time() - t0) / 60, 1), "epochs": a.epochs, "imgsz": a.imgsz,
        "note": "Crops from labelled boxes (oracle localisation); runtime crops come from the head ROI and are "
                "noisier.",
    }
    dst = ROOT / "docs" / "ml" / "head_classifier_eval.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if a.install:
        shutil.copy2(best, ROOT / "models" / "head_cls_yolo26n.pt")
        print("installed models/head_cls_yolo26n.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
