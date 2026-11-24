"""COCO val2017 'cell phone' (class 67) quality of the phone-stage weights at the deployed input sizes.

  python scripts/ml/eval_phone_coco.py --weights models/yolo26s.pt --imgsz 640 1280
      --report docs/ml/phone_coco_eval.json
Downloads val2017 images (~1 GB) + labels into data/coco on first run. A PROXY only: COCO phones are mostly
close-up; CCTV-distance phones are smaller - the field number must come from site footage (docs/ml/phase4-log.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHONE = 67


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "yolo26s.pt"))
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280])
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "coco")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils.downloads import download

    root = a.data_root
    if not (root / "images" / "val2017").exists():
        download(["https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"], dir=root.parent)
        download(["http://images.cocodataset.org/zips/val2017.zip"], dir=root / "images")
    model = YOLO(a.weights)
    cfg = root / "val_only.yaml"
    cfg.write_text(yaml.safe_dump({"path": str(root), "train": "images/val2017", "val": "images/val2017",
                                   "names": model.names}), encoding="utf-8")
    out = []
    for imgsz in a.imgsz:
        r = model.val(data=str(cfg), imgsz=imgsz, batch=16, half=True, plots=False, verbose=False)
        i = list(r.box.ap_class_index).index(PHONE)
        out.append({"imgsz": imgsz, "precision": round(float(r.box.p[i]), 4), "recall": round(float(r.box.r[i]), 4),
                    "ap50": round(float(r.box.ap50[i]), 4), "ap50_95": round(float(r.box.ap[i]), 4)})
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps({"weights": Path(a.weights).name, "class": "cell phone", "results": out},
                                   indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
