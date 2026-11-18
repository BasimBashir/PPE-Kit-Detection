"""Distillation flywheel demo: fine-tune the head classifier on teacher-labelled crops (no ground truth for those
crops) and measure the effect on crops it never trained on, against a control without teacher crops.

1. export confident teacher labels (scripts/ml/export_teacher_labels.py --source cache, --margin)
2. split them by file-name hash: 70% fine-tune, 30% holdout (never trained on)
3. per seed, fine-tune models/head_cls_yolo26n.pt with the scripts/dataset/train_head_classifier.py recipe on
   - distilled: the original ground-truth train crops + the 70% teacher-labelled crops (under their TEACHER label)
   - control:   the original ground-truth train crops only (isolates the effect of continued training)
   (val = original val split, for best.pt selection)
4. evaluate shipped / control / distilled against ground truth on the holdout and on the rest of the public test
   split (every test crop except the fine-tune crops); write docs/ml/distill_eval.json under key margin_<m>

models/ is never modified. Run: cd backend && UV_PROJECT_ENVIRONMENT=... uv run python ../scripts/ml/distill_finetune.py
  --margin 0.2 --seeds 42 43 44
Self-check: ... distill_finetune.py --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from export_teacher_labels import label
from teacher_benchmark import auroc, ece

ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home() / "ppe-work"
MAX_FALSE_ALARM = 0.02


def split(rel: str) -> str:
    """Deterministic 70/30 split by file-name hash (same hashing as teacher_benchmark.py's half split)."""
    return "finetune" if int(hashlib.sha256(rel.encode()).hexdigest()[:8], 16) % 10 < 7 else "holdout"


def threshold_at_fa(p_neg: list[float], budget: float = MAX_FALSE_ALARM) -> float:
    """Lowest alarm threshold (alarm = p >= t) with at most `budget` of the negatives alarming."""
    neg = sorted(p_neg, reverse=True)
    k = math.floor(budget * len(neg))  # allowed false alarms
    return math.nextafter(neg[k], math.inf) if k < len(neg) else 0.0


def p_no_helmet(weights: Path, files: list[Path]) -> list[float]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    no_idx = next(k for k, v in model.names.items() if v == "no_helmet")
    out = []
    for i in range(0, len(files), 512):
        rs = model.predict([str(f) for f in files[i:i + 512]], imgsz=96, device="0", verbose=False)
        out += [float(r.probs.data[no_idx]) for r in rs]
    return out


def metrics(y: list[int], p: list[float], t: float) -> dict:
    pos = [s for v, s in zip(y, p, strict=True) if v]
    neg = [s for v, s in zip(y, p, strict=True) if not v]
    return {
        "n": len(y), "n_violations": len(pos),
        "accuracy": round(sum((s >= 0.5) == bool(v) for v, s in zip(y, p, strict=True)) / len(y), 4),
        "auroc": round(auroc(y, p), 4) if pos and neg else None,  # undefined with one class
        "missed_violations_at_0.5": sum(s < 0.5 for s in pos),
        "false_alarms_at_0.5": sum(s >= 0.5 for s in neg),
        "miss_rate_at_fa_threshold": round(sum(s < t for s in pos) / len(pos), 4) if pos else None,
        "false_alarm_rate_at_fa_threshold": round(sum(s >= t for s in neg) / len(neg), 4) if neg else None,
        "ece_10bin": round(ece(y, p), 4),
    }


def link_tree(src: Path, dst: Path) -> int:
    n = 0
    for f in src.rglob("*.jpg"):
        (dst / f.parent.name).mkdir(parents=True, exist_ok=True)
        os.link(f, dst / f.parent.name / f.name)  # hardlink: no copy of ~20k crops, originals untouched
        n += 1
    return n


def train(data: Path, weights: Path, seed: int, epochs: int) -> float | None:
    """Fine-tune the shipped model with the train_head_classifier.py recipe; returns minutes (None if cached)."""
    if weights.exists():  # shortcut: cached by name, delete the .pt to retrain
        return None
    import numpy as np
    import torch
    from ultralytics import YOLO

    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    YOLO(str(ROOT / "models" / "head_cls_yolo26n.pt")).train(
        data=str(data), epochs=epochs, imgsz=96, batch=256, device="0", seed=seed, deterministic=True,
        project=str(weights.parent / "runs"), name=weights.stem, exist_ok=True, patience=10, workers=8)
    shutil.copy2(weights.parent / "runs" / weights.stem / "weights" / "best.pt", weights)
    return round((time.time() - t0) / 60, 1)


def spread(xs: list[float]) -> dict:
    return {"mean": round(statistics.mean(xs), 4), "min": min(xs), "max": max(xs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--work", default=str(WORK / "distill"))
    ap.add_argument("--cls-data", default=str(WORK / "hardhat" / "cls"))
    a = ap.parse_args()
    work, cls_data = Path(a.work), Path(a.cls_data)
    labels, data, data_ctl = work / f"labels_m{a.margin}", work / f"data_m{a.margin}", work / "data_control"
    for d in (labels, data):
        shutil.rmtree(d, ignore_errors=True)

    subprocess.run([sys.executable, str(Path(__file__).with_name("export_teacher_labels.py")), "--source", "cache",
                    "--out", str(labels), "--margin", str(a.margin), "--data", str(cls_data / "test")], check=True)
    manifest = json.loads((labels / "manifest.json").read_text())
    n_cache = len((WORK / "teacher" / "cache.jsonl").read_text().splitlines())
    rows = [{"path": labels / k, "teacher": v["label"], "truth": v["crop"].split("/")[0], "split": split(v["crop"])}
            for k, v in sorted(manifest.items())]
    ft = [r for r in rows if r["split"] == "finetune"]
    ho = [r for r in rows if r["split"] == "holdout"]

    # distilled training set = original ground-truth train crops + fine-tune teacher crops under their TEACHER label
    n_train = link_tree(cls_data / "train", data / "train")
    for r in ft:
        os.link(r["path"], data / "train" / r["teacher"] / f"teacher_{r['path'].name}")
    link_tree(cls_data / "val", data / "val")
    if not data_ctl.exists():
        link_tree(cls_data / "train", data_ctl / "train")
        link_tree(cls_data / "val", data_ctl / "val")

    ft_crops = {f"{r['truth']}/{r['path'].name}" for r in ft}
    rest = [f for c in ("helmet", "no_helmet") for f in sorted((cls_data / "test" / c).glob("*.jpg"))
            if f"{c}/{f.name}" not in ft_crops]

    def evaluate(w: Path) -> dict:
        p_ft = p_no_helmet(w, [r["path"] for r in ft])
        # threshold from the fine-tune set's TEACHER labels only (no ground truth in the flywheel)
        t = threshold_at_fa([p for r, p in zip(ft, p_ft, strict=True) if r["teacher"] == "helmet"])
        return {"sha256_12": hashlib.sha256(w.read_bytes()).hexdigest()[:12], "fa_threshold": round(t, 6),
                "holdout": metrics([int(r["truth"] == "no_helmet") for r in ho],
                                   p_no_helmet(w, [r["path"] for r in ho]), t),
                "rest_of_test": metrics([int(f.parent.name == "no_helmet") for f in rest], p_no_helmet(w, rest), t)}

    runs: dict = {"shipped": evaluate(ROOT / "models" / "head_cls_yolo26n.pt")}
    for seed in a.seeds:
        for variant, d, w in (("control", data_ctl, work / f"head_cls_control_s{seed}.pt"),
                              ("distilled", data, work / f"head_cls_distilled_m{a.margin}_s{seed}.pt")):
            minutes = train(d, w, seed, a.epochs)
            runs[f"{variant}_s{seed}"] = {"seed": seed, "train_minutes": minutes, "weights": w.name, **evaluate(w)}

    def by_class(rs: list[dict]) -> dict:
        return {c: sum(r["teacher"] == c for r in rs) for c in ("helmet", "no_helmet")}

    summary = {
        v: {"rest_missed_violations": spread([r["rest_of_test"]["missed_violations_at_0.5"] for r in rs]),
            "rest_ece": spread([r["rest_of_test"]["ece_10bin"] for r in rs]),
            "holdout_accuracy": spread([r["holdout"]["accuracy"] for r in rs]),
            "holdout_ece": spread([r["holdout"]["ece_10bin"] for r in rs])}
        for v in ("control", "distilled") if (rs := [r for k, r in runs.items() if k.startswith(v)])
    }
    result = {
        "margin": a.margin, "seeds": a.seeds, "teacher_cache_rows": n_cache,
        "teacher_labels_used": len(rows), "teacher_labels_skipped": n_cache - len(rows),
        "labels_by_class": by_class(rows), "finetune_by_class": by_class(ft), "holdout_by_class": by_class(ho),
        "teacher_label_precision_vs_truth": round(sum(r["teacher"] == r["truth"] for r in rows) / max(len(rows), 1), 4),
        "teacher_label_errors_in_finetune_set": sum(r["teacher"] != r["truth"] for r in ft),
        "original_train_crops": n_train, "epochs": a.epochs, "imgsz": 96,
        "holdout_files": [r["path"].relative_to(labels).as_posix() for r in ho],
        "summary": summary, "runs": runs,
        "note": "Scored against ground-truth folders. Holdout = 30% of the teacher-labelled crops; rest_of_test = "
                "every public test crop except the fine-tune crops (includes the holdout). Same public split the "
                "teacher saw: flywheel mechanics, not site performance. Metrics at 0.5 unless named fa_threshold.",
    }
    dst = ROOT / "docs" / "ml" / "distill_eval.json"
    doc = json.loads(dst.read_text()) if dst.exists() else {}
    doc = {k: v for k, v in doc.items() if k.startswith("margin_")}  # drop the pre-multi-seed flat layout
    doc[f"margin_{a.margin}"] = result
    dst.write_text(json.dumps(doc, indent=2))
    print(json.dumps({k: result[k] for k in ("labels_by_class", "finetune_by_class", "holdout_by_class",
                                             "teacher_label_precision_vs_truth", "summary")}, indent=2))
    return 0


def selftest() -> None:
    names = [f"helmet/crop{i}.jpg" for i in range(2000)]
    s1, s2 = [split(n) for n in names], [split(n) for n in names]
    assert s1 == s2  # deterministic
    ft = {n for n, s in zip(names, s1, strict=True) if s == "finetune"}
    ho = {n for n, s in zip(names, s1, strict=True) if s == "holdout"}
    assert not ft & ho and len(ft) + len(ho) == len(names)  # disjoint and complete
    assert 0.66 < len(ft) / len(names) < 0.74, len(ft)  # ~70/30
    assert label(0.95, 0.4) == "no_helmet" and label(0.05, 0.4) == "helmet"
    assert label(0.85, 0.4) is None and label(0.12, 0.4) is None and label(None, 0.4) is None
    assert label(0.75, 0.2) == "no_helmet" and label(0.31, 0.2) is None
    assert threshold_at_fa([0.1] * 99 + [0.9]) > 0.1 and threshold_at_fa([0.1] * 99 + [0.9]) <= 0.9
    assert threshold_at_fa([0.1, 0.2]) > 0.2  # 2% of 2 negatives = 0 false alarms allowed
    assert spread([1, 3]) == {"mean": 2, "min": 1, "max": 3}
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    sys.exit(main())
