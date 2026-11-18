"""Benchmark last-layer fusion of the hard-hat signals: head classifier alone vs logistic regression vs Laya.

Laya (github.com/NandhaKishorM/laya, PyPI `laya`, Apache-2.0) is a text-only typed-decision model: it reads the
numeric signals serialised as text and answers a yes/no (`noul`) question; it never sees pixels. Every method is
fitted / tuned on the val features and evaluated once on test (features from scripts/ml/fusion_features.py).

Usage (isolated env; nothing is added to the project; CUDA torch for the Laya forward pass):
  uv run --no-project --python 3.12 --with laya --with scikit-learn --with numpy \
    --with "torch==2.14.1+cu130" --index https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match python scripts/ml/fusion_benchmark.py [--no-laya] [--selftest]
Writes docs/ml/fusion_eval.json and docs/ml/fusion_eval.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATS = Path.home() / "ppe-work" / "fusion"
MAX_FAR = 0.02
LAYA_MODEL = ("convaiinnovations/laya", "typed-decisions")  # ModernBERT-large 421M, English typed-decisions
QUESTIONS = {
    "q_violation": "Is this person's head NOT protected by a hard hat (a safety violation)?",
    "q_missing": "Is the worker missing a hard hat?",
}


# ---------- metric helpers ----------
def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(
        sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    )


def threshold_at_far(y: np.ndarray, s: np.ndarray, max_far: float = MAX_FAR) -> float:
    """Lowest threshold t (predict violation if s >= t) whose false-alarm rate on (y, s) is <= max_far."""
    neg = np.sort(s[y == 0])[::-1]
    k = int(np.floor(max_far * len(neg)))  # at most k negatives may score >= t
    return float(np.nextafter(neg[k], np.inf)) if k < len(neg) else float(neg.min())


def rates(y: np.ndarray, s: np.ndarray, t: float) -> tuple[float, float]:
    """(miss rate on violations, false-alarm rate on helmets) for 'violation if s >= t'."""
    pred = s >= t
    return float((~pred[y == 1]).mean()), float(pred[y == 0].mean())


def best_acc_threshold(y: np.ndarray, s: np.ndarray) -> float:
    cands = np.unique(s)
    return float(cands[int(np.argmax([((s >= t) == y).mean() for t in cands]))])


def evaluate(yv, sv, yt, st, pt, boot_idx) -> dict:
    """sv/st: val/test scores (ranking); pt: test probabilities (calibration). Thresholds come from val only."""
    from sklearn.metrics import roc_auc_score

    t_acc, t_far = best_acc_threshold(yv, sv), threshold_at_far(yv, sv)
    miss, far = rates(yt, st, t_far)
    aucs = np.array([roc_auc_score(yt[i], st[i]) for i in boot_idx])
    return {
        "accuracy": round(float(((st >= t_acc) == yt).mean()), 4),
        "auroc": round(roc_auc_score(yt, st), 4),
        "auroc_ci95": [round(float(x), 4) for x in np.percentile(aucs, [2.5, 97.5])],
        "miss_rate_at_val_far2pct": round(miss, 4),
        "test_far_at_that_threshold": round(far, 4),
        "ece_10bin": round(ece(yt, pt), 4),
        "_boot": aucs,
    }


# ---------- methods ----------
def load(split: str) -> list[dict]:
    return [json.loads(line) for line in (FEATS / f"features_{split}.jsonl").read_text().splitlines()]


def logit(p) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def numeric(rows: list[dict]) -> np.ndarray:
    return np.column_stack(
        [
            logit([r["cls_p_no_helmet"] for r in rows]),
            [r["det_helmet_conf"] for r in rows],
            [r["det_head_conf"] for r in rows],
            np.log([r["box_w_px"] for r in rows]),
            np.log([r["box_h_px"] for r in rows]),
            np.log([r["area_rel"] for r in rows]),
            [r["aspect"] for r in rows],
            np.log([r["crop_px"] for r in rows]),
        ]
    )


def state_json(r: dict) -> str:
    return json.dumps(
        {
            "head_classifier_p_no_helmet": r["cls_p_no_helmet"],
            "detector_helmet_conf": r["det_helmet_conf"],
            "detector_bare_head_conf": r["det_head_conf"],
            "head_box_px": [r["box_w_px"], r["box_h_px"]],
            "head_area_frac_of_image": r["area_rel"],
            "aspect_w_over_h": r["aspect"],
        },
        separators=(",", ":"),
    )


def state_text(r: dict) -> str:
    return (
        f"Head-crop classifier: {100 * r['cls_p_no_helmet']:.0f}% probability that the head has no hard hat. "
        f"Object detector: hard hat seen with confidence {r['det_helmet_conf']:.2f}; bare head seen with "
        f"confidence {r['det_head_conf']:.2f}. Head box {r['box_w_px']:.0f}x{r['box_h_px']:.0f} px."
    )


def laya_scores(agent, rows: list[dict], fmt, question: str) -> np.ndarray:
    q = {"v": {"type": "noul", "instructions": question}}
    out = agent.predict_batch([fmt(r) for r in rows], q, batch_size=128)
    return np.array([o["answers"]["v"]["noul"] for o in out])


def per_call_ms(fn, n: int) -> float:
    t0 = time.perf_counter()
    for i in range(n):
        fn(i)
    return round(1000 * (time.perf_counter() - t0) / n, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-laya", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    val, test = load("val"), load("test")
    yv, yt = np.array([r["label"] for r in val]), np.array([r["label"] for r in test])
    boot = [np.random.default_rng(i).integers(0, len(yt), len(yt)) for i in range(a.bootstrap)]
    res, notes = {}, []

    cv, ct = np.array([r["cls_p_no_helmet"] for r in val]), np.array([r["cls_p_no_helmet"] for r in test])
    res["classifier_alone"] = evaluate(yv, cv, yt, ct, ct, boot)
    res["classifier_alone"]["latency_ms"] = {"cpu": per_call_ms(lambda i: ct[i] >= 0.5, 1000)}

    xv, xt = numeric(val), numeric(test)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(xv, yv)
    lv, lt = lr.predict_proba(xv)[:, 1], lr.predict_proba(xt)[:, 1]
    res["logistic_fusion"] = evaluate(yv, lv, yt, lt, lt, boot)
    res["logistic_fusion"]["latency_ms"] = {"cpu": per_call_ms(lambda i: lr.predict_proba(xt[i : i + 1]), 500)}
    res["logistic_fusion"]["coefficients_standardized"] = dict(
        zip(
            [
                "cls_logit",
                "det_helmet_conf",
                "det_head_conf",
                "log_w",
                "log_h",
                "log_area_rel",
                "aspect",
                "log_crop_px",
            ],
            np.round(lr[-1].coef_[0], 3).tolist(),
            strict=True,
        )
    )

    laya_info: dict = {"model": "/".join(LAYA_MODEL), "variants_val_auroc": {}}
    if a.no_laya:
        notes.append("Laya skipped (--no-laya).")
    else:
        try:
            import laya
            import torch
            from sklearn.metrics import roc_auc_score

            agent = laya.load(LAYA_MODEL[0], subfolder=LAYA_MODEL[1], device=a.device)
            laya_info.update(
                version=laya.__version__,
                device=str(agent.device),
                dtype=str(agent.dtype),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            )
            # Prompt tuning on val only: 2 serialisations x 2 questions; the variant whose val AUROC is furthest
            # from chance (most informative, either direction) goes to test; Platt below fixes the direction.
            best = None
            for fname, fmt in (("json", state_json), ("text", state_text)):
                for qname, question in QUESTIONS.items():
                    sv = laya_scores(agent, val, fmt, question)
                    auc = roc_auc_score(yv, sv)
                    laya_info["variants_val_auroc"][f"{fname}/{qname}"] = round(auc, 4)
                    if best is None or abs(auc - 0.5) > abs(best[0] - 0.5):
                        best = (auc, fname, fmt, question, sv)
            _, fname, fmt, question, sv = best
            laya_info.update(selected_state=fname, selected_question=question, example_state=fmt(test[0]))
            st = laya_scores(agent, test, fmt, question)
            res["laya_zero_shot"] = evaluate(yv, sv, yt, st, st, boot)
            # Platt scaling on the logit of Laya's probability (may flip the sign if Laya ranks backwards).
            platt = LogisticRegression().fit(logit(sv)[:, None], yv)
            pv, pt = platt.predict_proba(logit(sv)[:, None])[:, 1], platt.predict_proba(logit(st)[:, None])[:, 1]
            res["laya_platt_recalibrated"] = evaluate(yv, pv, yt, pt, pt, boot)
            q = {"v": {"type": "noul", "instructions": question}}
            agent.predict(fmt(test[0]), q)  # warm-up
            single = per_call_ms(lambda i: agent.predict(fmt(test[i]), q), 200)
            t0 = time.perf_counter()
            laya_scores(agent, test, fmt, question)
            batched = round(1000 * (time.perf_counter() - t0) / len(test), 4)
            dev = "gpu" if str(agent.device).startswith("cuda") else "cpu"
            for k in ("laya_zero_shot", "laya_platt_recalibrated"):
                res[k]["latency_ms"] = {f"{dev}_batch1": single, f"{dev}_batch128": batched}
        except Exception as e:  # record exactly why Laya could not be measured; never fabricate numbers
            notes.append(f"Laya could not be measured: {type(e).__name__}: {e}")
    notes.append(
        "Laya fine-tuning not run: the package exposes no fine-tuning API for the ModernBERT checkpoints; "
        "its documented route is a separate RLCD training notebook (Kaggle 2xT4 / Apple MPS script). "
        "`Agent.fit` exists only for the different vegaml reader (Qwen 0.8B), not this checkpoint."
    )

    ref = res["logistic_fusion"]["_boot"]
    for k, v in res.items():
        b = v.pop("_boot")
        if k != "logistic_fusion":
            v["auroc_minus_logistic_ci95"] = [round(float(x), 4) for x in np.percentile(b - ref, [2.5, 97.5])]
    out = {
        "split_fit": "val",
        "split_eval": "test",
        "n_val": len(val),
        "n_test": len(test),
        "violations_val": int(yv.sum()),
        "violations_test": int(yt.sum()),
        "max_far_val": MAX_FAR,
        "bootstrap_resamples": a.bootstrap,
        "methods": res,
        "laya": laya_info,
        "notes": notes,
    }
    (ROOT / "docs" / "ml" / "fusion_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


def selftest() -> int:
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    assert ece((rng.uniform(size=p.size) < p).astype(int), p) < 0.01  # calibrated by construction
    assert abs(ece(np.zeros(1000, int), np.full(1000, 0.9)) - 0.9) < 1e-9  # always says 0.9, never right
    y = np.array([0] * 100 + [1] * 4)
    s = np.r_[np.arange(100) / 100, [0.5, 0.985, 0.995, 2.0]]
    t = threshold_at_far(y, s)  # 2 of 100 negatives (0.98, 0.99) may alarm -> t just above 0.97
    assert 0.97 < t <= 0.98, t
    assert rates(y, s, t) == (0.25, 0.02)  # the 0.5 violation is missed
    assert best_acc_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.9])) == 0.7
    print("selftest ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
