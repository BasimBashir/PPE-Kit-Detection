"""Phase 3: build one 10-class present/absent PPE dataset (YOLO format) from public sources.

Sources (downloaded beforehand, see docs/ml/phase3-log.md):
  cppe   Ultralytics Construction-PPE (CC BY 4.0), 1,000 images, 800 train / 200 val, all five items + "no_*" classes
  hb1204 huggingface.co/datasets/HB1204/PPE_Detection (Roboflow export, CC BY 4.0), 3,086 images, no "no boots" class

Unified classes: helmet no_helmet vest no_vest gloves no_gloves goggles no_goggles boots no_boots.
Test split = the official Construction-PPE val (never trained on); our val = a seeded 12.5 % of cppe train.

  python scripts/dataset/prepare_multippe_dataset.py --cppe <dir> [--hb1204 <dir>] --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import sys
from pathlib import Path

NAMES = ["helmet", "no_helmet", "vest", "no_vest", "gloves", "no_gloves", "goggles", "no_goggles", "boots", "no_boots"]
# source class id -> unified id; ids not listed (person, forklift, "None") are dropped
CPPE = {0: 0, 6: 1, 2: 2, 11: 3, 1: 4, 8: 5, 4: 6, 7: 7, 3: 8, 9: 9}
HB1204 = {3: 0, 6: 1, 10: 2, 7: 3, 1: 4, 4: 5, 2: 6, 5: 7, 0: 8}


def remap(src: Path, mapping: dict[int, int]) -> list[str]:
    out = []
    for line in src.read_text().splitlines():
        p = line.split()
        if len(p) == 5 and int(p[0]) in mapping:
            out.append(" ".join([str(mapping[int(p[0])]), *p[1:]]))
    return out


def place(img: Path, lines: list[str], out: Path, split: str, prefix: str, counts: dict) -> None:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    dst = out / "images" / split / f"{prefix}_{img.name}"
    if not dst.exists():
        try:
            os.link(img, dst)  # same volume: no copy
        except OSError:
            shutil.copy2(img, dst)
    (out / "labels" / split / f"{prefix}_{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    c = counts.setdefault(split, collections.Counter())
    c["_images"] += 1
    for line in lines:
        c[NAMES[int(line.split()[0])]] += 1


def pairs(img_dir: Path, lbl_dir: Path) -> list[tuple[Path, Path]]:
    return sorted((i, lbl_dir / f"{i.stem}.txt") for i in img_dir.iterdir()
                  if i.suffix.lower() in (".jpg", ".jpeg", ".png") and (lbl_dir / f"{i.stem}.txt").exists())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cppe", type=Path, required=True)
    ap.add_argument("--hb1204", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.125)
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    counts: dict[str, collections.Counter] = {}

    train = pairs(a.cppe / "images" / "train", a.cppe / "labels" / "train")
    random.Random(42).shuffle(train)
    n_val = int(len(train) * a.val_frac)
    for i, (img, lbl) in enumerate(train):
        place(img, remap(lbl, CPPE), a.out, "val" if i < n_val else "train", "cppe", counts)
    for img, lbl in pairs(a.cppe / "images" / "val", a.cppe / "labels" / "val"):
        place(img, remap(lbl, CPPE), a.out, "test", "cppe", counts)
    if a.hb1204:
        for split, dst in (("train", "train"), ("valid", "val"), ("test", "test_hb1204")):
            for img, lbl in pairs(a.hb1204 / split / "images", a.hb1204 / split / "labels"):
                place(img, remap(lbl, HB1204), a.out, dst, "hb", counts)

    yaml = [f"path: {a.out.as_posix()}", "train: images/train", "val: images/val", "test: images/test"]
    if (a.out / "images" / "test_hb1204").is_dir():  # extra holdout: absolute path - ultralytics resolves custom
        yaml.insert(-1, f"test_hb1204: {(a.out / 'images' / 'test_hb1204').as_posix()}")  # split keys against CWD
    yaml.append("names:")
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (a.out / "data.yaml").write_text("\n".join(yaml) + "\n")
    report = {"sources": {"cppe": str(a.cppe), "hb1204": str(a.hb1204) if a.hb1204 else None},
              "names": NAMES, "counts": {k: dict(v) for k, v in counts.items()}}
    (a.out / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert counts["test"]["_images"] > 0 and counts["train"]["_images"] > 0, "empty split"
    return 0


if __name__ == "__main__":
    sys.exit(main())
