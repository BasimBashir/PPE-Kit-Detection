"""Fine-tune YOLO26 on Hard Hat Workers (helmet / head / person) and evaluate on the held-out TEST split.

Examples:
  python scripts/dataset/train_ppe_detector.py --data <work>/hardhat/yolo/data.yaml               # full fine-tune
  python scripts/dataset/train_ppe_detector.py --data ... --freeze 10 --name ppe_yolo26s_freeze10   # frozen backbone
  python scripts/dataset/train_ppe_detector.py --data ... --eval-only <weights.pt> --name ppe_yolo26s
Writes docs/ml/<name>_eval.json; with --install copies best.pt to models/ppe_yolo26s.pt.
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


def sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def evaluate(weights: Path, data: str, name: str, device: str, extra: dict) -> dict:
    from ultralytics import YOLO

    r = YOLO(str(weights)).val(data=data, split="test", imgsz=640, batch=32, device=device, plots=False,
                               verbose=False, project=str(Path(weights).parents[2]), name=f"{name}_test", exist_ok=True)
    per_class = {}
    for i, c in enumerate(r.box.ap_class_index):
        p, rc, ap50, ap = r.box.class_result(i)
        per_class[r.names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(rc), 4),
                                      "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    mp, mr, map50, mapall = (round(float(v), 4) for v in r.box.mean_results())
    out = {
        "model": name, "weights_sha256_12": sha12(weights), "split": "test", "dataset": data,
        "dataset_license": "CC0-1.0 (Hard Hat Workers via Voxel51/hard-hat-detection)",
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "per_class": per_class, "mean": {"precision": mp, "recall": mr, "mAP50": map50, "mAP50_95": mapall},
        "speed_ms_per_image": {k: round(v, 3) for k, v in r.speed.items()},
        "note": "Image-level detector metrics on 416 px construction images; NOT system-level event recall on "
                "site video.",
        **extra,
    }
    dst = ROOT / "docs" / "ml" / f"{name}_eval.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--freeze", type=int, default=0, help="freeze first N layers (0 = full fine-tune)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default=str(Path.home() / "ppe-work" / "runs"))
    ap.add_argument("--name", default="ppe_yolo26s")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--eval-only", type=Path, help="skip training, evaluate these weights")
    ap.add_argument("--install", action="store_true", help="copy best.pt to models/ppe_yolo26s.pt")
    a = ap.parse_args()

    extra = {"base_model": a.model, "epochs": a.epochs, "imgsz": a.imgsz, "freeze": a.freeze, "seed": 42}
    if a.eval_only:
        best = a.eval_only
    else:
        from ultralytics import YOLO

        t0 = time.time()
        YOLO(a.model).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                            freeze=a.freeze or None, seed=42, project=a.project, name=a.name, exist_ok=True,
                            fraction=a.fraction, patience=20, workers=8, plots=True)
        extra["train_minutes"] = round((time.time() - t0) / 60, 1)
        best = Path(a.project) / a.name / "weights" / "best.pt"
    evaluate(best, a.data, a.name, a.device, extra)
    if a.install:
        dst = ROOT / "models" / "ppe_yolo26s.pt"
        shutil.copy2(best, dst)
        print(f"installed {dst} sha256[:12]={sha12(dst)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
