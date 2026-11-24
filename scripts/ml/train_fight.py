"""Train / evaluate the Phase 4 fight model (ADR-015) on sequences from extract_fight_sequences.py.

  python scripts/ml/train_fight.py train --data data/fight/rlvs.npz --out models/fight_gru.pt
      --report docs/ml/fight_eval.json
  python scripts/ml/train_fight.py eval --data data/fight/scfd.npz --weights models/fight_gru.pt
      --report docs/ml/fight_eval.json
Clip-level metrics: a clip is a fight if any of its pair sequences scores >= threshold; clips without any pair
sequence count as 'no fight'. Train/val split is by clip (no leakage). Reports are merged into one JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def clip_scores(probs: np.ndarray, clip_idx: np.ndarray, n_clips: int) -> np.ndarray:
    s = np.zeros(n_clips, np.float32)
    np.maximum.at(s, clip_idx, probs)
    return s


def metrics(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred, y = scores >= thr, labels == 1
    tp, fp, fn, tn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()), int((~pred & ~y).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"threshold": round(float(thr), 3), "clips": len(labels),
            "accuracy": round((tp + tn) / max(len(labels), 1), 4), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-9), 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def predict(net, X: np.ndarray, device) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(torch.sigmoid(net(torch.from_numpy(X[i:i + 1024]).to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def merge_report(path: Path, key: str, value: dict) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({key: value}, indent=2))


def train(a) -> None:
    import torch
    from app.behaviour.fight import FightGRU

    d = np.load(a.data)
    X, ci, labels = d["X"], d["clip_idx"], d["clip_labels"].astype(np.int64)
    rng = np.random.default_rng(0)
    val = np.zeros(len(labels), bool)
    for c in (0, 1):  # stratified 80/20 split by clip
        ids = rng.permutation(np.flatnonzero(labels == c))
        val[ids[: len(ids) // 5]] = True
    tr_m, va_m = ~val[ci], val[ci]
    if not tr_m.any():
        raise SystemExit(f"no training sequences in {a.data} - clips too short for the window, or no nearby pairs")
    y = labels[ci].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FightGRU(a.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = max(float(y[tr_m].sum()), 1.0)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((tr_m.sum() - pos) / pos, device=device))
    Xt, yt = torch.from_numpy(X[tr_m]), torch.from_numpy(y[tr_m])
    best, best_state = -1.0, None
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), a.batch):
            b = perm[i:i + a.batch]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b].to(device)), yt[b].to(device))
            loss.backward()
            opt.step()
        net.eval()
        vs = clip_scores(predict(net, X[va_m], device), np.searchsorted(np.flatnonzero(val), ci[va_m]), int(val.sum()))
        f1 = metrics(vs, labels[val], 0.5)["f1"]
        if f1 > best:
            best, best_state = f1, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        print(f"epoch {epoch + 1}: loss {loss.item():.4f}  val clip F1@0.5 {f1:.4f}", flush=True)
    net.load_state_dict(best_state)
    vs = clip_scores(predict(net, X[va_m], device), np.searchsorted(np.flatnonzero(val), ci[va_m]), int(val.sum()))
    sweep = [metrics(vs, labels[val], t) for t in np.arange(0.3, 0.95, 0.05)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "hidden": a.hidden, "seq_len": int(d["seq_len"]), "hz": float(d["hz"])},
               a.out)
    merge_report(a.report, f"val:{Path(a.data).stem}", {"at_0.5": metrics(vs, labels[val], 0.5),
                                                       "best_f1": max(sweep, key=lambda m: m["f1"]),
                                                       "sequences_train": int(tr_m.sum()), "epochs": a.epochs})


def evaluate(a) -> None:
    import torch
    from app.behaviour.fight import FightModel

    d, m = np.load(a.data), FightModel(a.weights)
    scores = clip_scores(m.score(d["X"]) if len(d["X"]) else np.zeros(0, np.float32), d["clip_idx"],
                         len(d["clip_labels"]))
    labels = d["clip_labels"].astype(np.int64)
    merge_report(a.report, f"test:{Path(a.data).stem}", {"at_0.5": metrics(scores, labels, 0.5),
                                                        "at_0.7_rule_default": metrics(scores, labels, 0.7),
                                                        "weights": m.version, "torch": torch.__version__})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=256)
    e = sub.add_parser("eval")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--weights", type=Path, required=True)
    for p in (t, e):
        p.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
