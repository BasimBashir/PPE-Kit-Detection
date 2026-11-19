"""caps fix, step 3-5: identity split, three-model comparison, docs/ml/caps_eval.json.

Variants (identical recipe, seed 42, 30 epochs, imgsz 96 - scripts/dataset/train_head_classifier.py):
  shipped   models/head_cls_yolo26n.pt (no training)
  control   CC0 crops + CC0 teacher labels (margin 0.2) WITHOUT caps crops - the distill study's control_s42
            (reused when its sha256 matches docs/ml/distill_eval.json, else retrained here)
  distilled control + the caps train crops (80% of identities) under their TEACHER label

Holdout honesty: split by worker identity (greedy-IoU track from caps_extract.py), so held-out cap crops were
never trained on. CC0 regression check on the distill study's 30% hash-holdout (truth = original folder).
models/ is never modified; weights go to C:/Users/mianx/ppe-work/caps/.

Run: cd backend && UV_PROJECT_ENVIRONMENT=... uv run python ../scripts/ml/caps_finetune.py
Self-check: ... caps_finetune.py --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "backend"))

from distill_finetune import link_tree, metrics, p_no_helmet, split, threshold_at_fa, train  # noqa: E402
from export_teacher_labels import label  # noqa: E402

WORK = Path.home() / "ppe-work"
CAPS = WORK / "caps"
DISTILL = WORK / "distill"
CLS = WORK / "hardhat" / "cls"
MARGIN = 0.2
TRAIN_FRAC = 0.8


def split_ids(ids: list[int], frac: float = TRAIN_FRAC, salt: str = "caps") -> tuple[list[int], list[int]]:
    """Deterministic identity split: ids sorted by sha256 hash order, first round(frac*n) are train."""
    order = sorted(ids, key=lambda i: hashlib.sha256(f"{salt}:{i}".encode()).hexdigest())
    k = round(frac * len(order))
    return sorted(order[:k]), sorted(order[k:])


def split_blocks(frames: list[int], fps: float, frac: float = TRAIN_FRAC) -> tuple[list[int], list[int]]:
    """Fallback when identities are unreliable: deterministic split of 2-second frame-time blocks."""
    blocks = sorted({f // max(1, round(2 * fps)) for f in frames})
    return split_ids(blocks, frac, salt="caps-block")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(CAPS))
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    caps, distill = Path(a.work), DISTILL

    man = json.loads((caps / "manifest.json").read_text())
    rows = man["crops"]
    fps = man.get("sample_fps", 3.0)
    for r in rows:  # teacher label by the export rule (margin), skipped when unconfident
        r["teacher"] = label(r.get("p_teacher"), a.margin)

    tracks = sorted({r["track"] for r in rows})
    if len(tracks) >= 5:
        tr_ids, ho_ids = split_ids(tracks)
        split_by = "identity_track"
    else:
        tr_ids, ho_ids = split_blocks([r["frame"] for r in rows], fps)
        split_by = "frame_time_block_2s"
    train_rows = [r for r in rows if r["track"] in tr_ids]
    holdout_rows = [r for r in rows if r["track"] in ho_ids]

    dist_labels = distill / "labels_m0.2"
    dman = json.loads((dist_labels / "manifest.json").read_text())
    ft_items = [(k, v) for k, v in sorted(dman.items()) if split(v["crop"]) == "finetune"]
    ho_items = [(k, v) for k, v in sorted(dman.items()) if split(v["crop"]) == "holdout"]
    ft_files = [dist_labels / k for k, _ in ft_items]
    ho_files = [dist_labels / k for k, _ in ho_items]
    # truth = the crops' ORIGINAL test-split folder (v["crop"]), not the teacher-label folder of the key
    cc0_truth = [int(v["crop"].split("/")[0] == "no_helmet") for _, v in ho_items]

    # training data: hardlink CC0 train+val; caps train crops under their teacher label (distilled only)
    data_ctl, data_dist = caps / "data_ctl", caps / "data_dist"
    if not data_ctl.exists():
        link_tree(CLS / "train", data_ctl / "train")
        link_tree(CLS / "val", data_ctl / "val")
    for d in (data_dist,):
        shutil.rmtree(d, ignore_errors=True)
    link_tree(CLS / "train", data_dist / "train")
    link_tree(CLS / "val", data_dist / "val")
    n_caps_train = 0
    for r in train_rows:
        if r["teacher"]:
            os.link(caps / "raw" / r["file"], data_dist / "train" / r["teacher"] / f"caps_{r['file']}")
            n_caps_train += 1

    # control weights: reuse the distill study's control_s42 (identical recipe/data) when its hash matches
    dstudy = json.loads((ROOT / "docs" / "ml" / "distill_eval.json").read_text())["margin_0.2"]
    want = dstudy["runs"]["control_s42"]["sha256_12"]
    control_w = distill / "head_cls_control_s42.pt"
    control_reused = hashlib.sha256(control_w.read_bytes()).hexdigest()[:12] == want
    if not control_reused:
        control_w = caps / f"head_cls_control_s{a.seed}.pt"
        train(data_ctl, control_w, a.seed, a.epochs)

    distilled_w = caps / f"head_cls_caps_s{a.seed}.pt"
    train(data_dist, distilled_w, a.seed, a.epochs)

    models = {
        "shipped": ROOT / "models" / "head_cls_yolo26n.pt",
        "control": control_w,
        "distilled": distilled_w,
    }

    def cc0_eval(w: Path) -> dict:
        """Study method: per-model FA threshold from the CC0 finetune set's TEACHER-helmet probs, then the
        study's 30% hash-holdout scored against ground truth."""
        ft_p = p_no_helmet(w, ft_files)
        t = threshold_at_fa([p for (_, v), p in zip(ft_items, ft_p, strict=True) if v["label"] == "helmet"])
        return {"threshold_at_2pct_far": round(t, 4),
                **metrics(cc0_truth, p_no_helmet(w, ho_files), t)}

    ho_conf = [r for r in holdout_rows if r["teacher"]]  # holdout crops the teacher was confident about
    raw = caps / "raw"

    def caps_eval(name: str, w: Path, subset_rows: list[dict]) -> dict:
        p = p_no_helmet(w, [raw / r["file"] for r in subset_rows])
        per_id = {}
        for tid in sorted({r["track"] for r in subset_rows}):
            pid = [q for q, r in zip(p, subset_rows, strict=True) if r["track"] == tid]
            per_id[str(tid)] = {"mean_p": round(sum(pid) / len(pid), 4),
                                "share_no_helmet_at_0.5": round(sum(q >= 0.5 for q in pid) / len(pid), 4)}
        return {"n": len(p), "mean_p_no_helmet": round(sum(p) / len(p), 4),
                "share_no_helmet_at_0.5": round(sum(q >= 0.5 for q in p) / len(p), 4),
                "per_identity": per_id}

    out = {
        "clip": man["clip"],
        "clip_note": "Pexels stock clip (metal workshop, workers in caps) - not site footage",
        "split_by": split_by,
        "n_crops_distinct": len(rows),
        "n_identities": len(tracks),
        "teacher": {
            "model": "gemini-decision-1", "question": "noul (same as the app)",
            "mean_p_no_helmet": round(sum(r["p_teacher"] for r in rows if r["p_teacher"] is not None)
                                      / max(sum(r["p_teacher"] is not None for r in rows), 1), 4),
            "refused_or_blocked": sum(r["p_teacher"] is None for r in rows),
            "labels_margin_0.2": {c: sum(r["teacher"] == c for r in rows) for c in ("helmet", "no_helmet")},
            "skipped_unconfident": sum(r["teacher"] is None for r in rows),
        },
        "split": {"train_identities": tr_ids, "holdout_identities": ho_ids,
                  "train_crops": len(train_rows), "holdout_crops": len(holdout_rows),
                  "holdout_crops_with_teacher_label": len(ho_conf),
                  "holdout_teacher_labels": {c: sum(r["teacher"] == c for r in ho_conf)
                                             for c in ("helmet", "no_helmet")}},
        "control_weights": {"reused_from_distill_study": control_reused,
                            "sha256_12": hashlib.sha256(control_w.read_bytes()).hexdigest()[:12]},
        "cc0_hash_holdout": {name: cc0_eval(w) for name, w in models.items()},
        "caps_holdout": {
            "note": "held-out worker identities, never trained on",
            **{name: caps_eval(name, w, holdout_rows) for name, w in models.items()},
            "teacher_confident_subset": {
                "note": "holdout crops the teacher labels confidently (the cap crops) - the fix metric",
                **{name: caps_eval(name, w, ho_conf) for name, w in models.items()},
            },
        },
    }

    dst = ROOT / "docs" / "ml" / "caps_eval.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


def selftest() -> None:
    ids = list(range(20))
    tr, ho = split_ids(ids)
    assert not set(tr) & set(ho) and sorted(tr + ho) == ids  # disjoint and complete
    assert tr == split_ids(ids)[0] and ho == split_ids(ids)[1]  # deterministic
    assert len(tr) == 16 and len(ho) == 4  # 80/20
    assert split_ids([7, 3, 11, 1]) == split_ids([1, 3, 7, 11])  # order-independent
    tr1, ho1 = split_blocks(list(range(0, 200, 3)), 3.0)
    assert not set(tr1) & set(ho1) and tr1 == split_blocks(list(range(0, 200, 3)), 3.0)[0]
    assert split_blocks([0, 3, 6], 3.0)[1] == []  # single block => everything trains, nothing held out
    assert label(0.95, 0.2) == "no_helmet" and label(0.05, 0.2) == "helmet" and label(0.6, 0.2) is None
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    sys.exit(main())
