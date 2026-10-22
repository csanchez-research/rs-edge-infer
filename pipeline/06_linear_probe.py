#!/usr/bin/env python3
# =============================================================================
#  06_linear_probe.py  -  frozen linear probe on reBEN, PER PRECISION
# -----------------------------------------------------------------------------
#  Trains a linear classifier (384 -> 19) on the cached frozen features and
#  measures micro-mAP (primary) + macro-mAP on the reBEN test split, for EACH
#  precision (fp32/fp16/int8). Reports mean +/- std over seeds and the delta vs
#  fp32 (= the accuracy cost of quantization). This is the paper's accuracy axis.
#
#  `train_linear_probe` is a FAITHFUL copy of the harness
#  (strata-s12-eval-harness/linear_probe.py): multi-label,
#  19 sigmoid outputs, BCE loss, AdamW lr=1e-3, 100 full-batch epochs, micro/
#  macro average precision. Same recipe, so numbers are comparable across
#  precisions and with Paper 1. The ONLY change is the outer loop: the harness's
#  "which" (pretrained vs random) becomes the precision. (Variable casing is
#  lowercased to satisfy the repo linter; the method is unchanged.)
#
#  Runs in the DEV image (needs torch + scikit-learn). Features come from
#  05_extract_features.
# =============================================================================
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jsonio
from logsetup import get_logger

import config

log = get_logger("06_linear_probe")


def _load_split(feat_dir, split, precision):
    d = np.load(os.path.join(feat_dir, f"{split}_{precision}.npz"))
    return d["features"].astype(np.float32), d["labels"].astype(np.float32)


def train_linear_probe(xtr, ytr, xte, yte, *, seed, epochs=100, lr=1e-3,
                       weight_decay=0.0, device="cpu"):
    """Faithful copy of the harness probe. Returns micro/macro-mAP on test."""
    import torch
    from sklearn.metrics import average_precision_score
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)  # intentional global seeding, as in the harness

    in_dim, n_cls = xtr.shape[1], ytr.shape[1]
    clf = nn.Linear(in_dim, n_cls).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(xtr).to(device)
    yt = torch.from_numpy(ytr).to(device)

    clf.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(clf(xt), yt).backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        probs = torch.sigmoid(clf(torch.from_numpy(xte).to(device))).cpu().numpy()

    micro = average_precision_score(yte, probs, average="micro")
    # macro only over classes with >=1 positive in test (BigEarthNet standard).
    valid = yte.sum(axis=0) > 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        macro = float("nan")
    else:
        ap = average_precision_score(yte[:, valid], probs[:, valid], average=None)
        macro = float(np.mean(ap))

    return {
        "micro_mAP": float(micro),
        "macro_mAP": float(macro),
        "n_classes_valid": n_valid,
        "n_classes_total": int(yte.shape[1]),
    }


def _mean_std(rows, key):
    v = np.array([r[key] for r in rows], dtype=np.float64)
    return float(v.mean()), float(v.std())


def main():
    ap = argparse.ArgumentParser(description="Frozen linear probe on reBEN, per precision.")
    ap.add_argument("--precisions", nargs="+", default=config.PRECISIONS)
    ap.add_argument("--feat-dir", default=config.FEATURES_DIR)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--require-classes", type=int, default=0,
                    help="abort if test has fewer than N classes with positives "
                         "(set 19 for the final run; 0 for smoke tests)")
    ap.add_argument(
        "--out",
        default=os.path.join(config.MODEL_OUTPUT_DIR, "probe_accuracy.json"),
    )
    args = ap.parse_args()

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    per_precision = {}
    summary = {}

    for p in args.precisions:
        xtr, ytr = _load_split(args.feat_dir, args.train_split, p)
        xte, yte = _load_split(args.feat_dir, args.test_split, p)
        n_valid = int((yte.sum(axis=0) > 0).sum())
        if args.require_classes and n_valid < args.require_classes:
            raise SystemExit(
                f"[{p}] test has only {n_valid} classes with positives; "
                f"{args.require_classes} required. macro-mAP over a subset is NOT "
                "comparable with the 19-class benchmark. Use the full test split."
            )
        log.info("[%s] train %s  test %s  (%d/19 classes present)",
                 p, xtr.shape, xte.shape, n_valid)

        rows = []
        for seed in range(args.seeds):
            r = train_linear_probe(xtr, ytr, xte, yte, seed=seed,
                                   epochs=args.epochs, lr=args.lr, device=dev)
            rows.append(r)
            log.info("  seed %d: micro-mAP %.4f  macro-mAP %.4f",
                     seed, r["micro_mAP"], r["macro_mAP"])
        per_precision[p] = rows
        mi_m, mi_s = _mean_std(rows, "micro_mAP")
        ma_m, ma_s = _mean_std(rows, "macro_mAP")
        summary[p] = {"micro_mAP": [mi_m, mi_s], "macro_mAP": [ma_m, ma_s],
                      "n_classes_valid": rows[0]["n_classes_valid"]}

    # Delta vs fp32 (the full-precision reference) = cost of quantization.
    ref = "fp32" if "fp32" in summary else args.precisions[0]
    ref_micro = summary[ref]["micro_mAP"][0]
    for _p, s in summary.items():
        s["d_micro_vs_ref"] = round(s["micro_mAP"][0] - ref_micro, 5)

    out = {
        "kind": "probe_accuracy",
        "model_id": config.MODEL_ID,
        "reference_precision": ref,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "lr": args.lr,
        "train_split": args.train_split,
        "test_split": args.test_split,
        "summary": summary,
        "per_seed": per_precision,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        jsonio.dump(out, f)
    log.info("results -> %s", args.out)

    log.info("%-6s %18s %18s %14s", "prec", "micro-mAP", "macro-mAP", f"d_micro_vs_{ref}")
    for p in args.precisions:
        s = summary[p]
        log.info("%-6s   %.4f±%.4f    %.4f±%.4f   %+.4f",
                 p, s["micro_mAP"][0], s["micro_mAP"][1],
                 s["macro_mAP"][0], s["macro_mAP"][1], s["d_micro_vs_ref"])


if __name__ == "__main__":
    main()
