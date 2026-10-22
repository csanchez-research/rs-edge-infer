#!/usr/bin/env python3
# =============================================================================
#  05_extract_features.py  -  cache 384-d features per precision (accuracy axis)
# -----------------------------------------------------------------------------
#  Runs each precision's OpenVINO IR as a FROZEN feature extractor over the reBEN
#  train/test subset (built by prepare_data.py) and caches, per split and
#  precision, the CLS-token features (384-d) + 19-hot labels to
#  features/<split>_<precision>.npz, so 06_linear_probe can just load matrices.
#
#  Why per PRECISION (and not one torch encoder, as the harness does): the paper
#  measures how quantization affects the representation. So the feature extractor
#  is the IR itself - fp32 is the full-precision reference, fp16/int8 show the
#  accuracy cost of quantizing. Accuracy is a property of the IR, measured ONCE on
#  the dev box; the targets only measure latency/energy.
#
#  Faithful to the harness probe (strata-s12-eval-harness/extract_features.py +
#  reben_loader.py): same 6-channel transform, same 19-class
#  label order. Runs in the DEV image (needs rasterio + openvino). Each patch is
#  read ONCE and pushed through all precisions, so the .tif I/O is not repeated.
# =============================================================================
import argparse
import os
import sys
import time

import numpy as np
import openvino as ov

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reben
from logsetup import get_logger

import config

log = get_logger("05_extract_features")


def _compile(core, precision):
    xml = config.ir_xml(precision)
    if not os.path.exists(xml):
        raise SystemExit(
            f"[extract] IR not found for {precision} at {xml}. Run 02_quantize first."
        )
    return core.compile_model(core.read_model(xml), "CPU")


def extract_split(core, split, args, means, stds):
    import pandas as pd

    tree = os.path.join(args.tree, split)
    meta = os.path.join(tree, "metadata.parquet")
    if not os.path.exists(meta):
        raise SystemExit(f"[extract] {meta} missing. Run prepare_data (extract) first.")
    df = pd.read_parquet(meta)

    cap = None
    if split == "train" and args.max_train:
        cap = args.max_train
    elif split == "test" and args.max_test:
        cap = args.max_test
    if cap and cap < len(df):
        # SAME rule as the harness: fixed-seed subset, reproducible + representative.
        df = df.sample(n=cap, random_state=42).reset_index(drop=True)

    compiled = {p: _compile(core, p) for p in args.precisions}
    feats = {p: [] for p in args.precisions}
    labels = []

    log.info("[%s] %d patches, precisions=%s", split, len(df), args.precisions)
    t0 = time.time()
    for i, (_, row) in enumerate(df.iterrows()):
        x = reben.preprocess(tree, row, means, stds, args.img_size)[None]  # (1,6,224,224)
        for p, cm in compiled.items():
            out = cm({cm.input(0): x})[cm.output(0)]  # (1, 384)
            feats[p].append(np.asarray(out[0], dtype=np.float32))
        labels.append(reben.labels_19hot(row))
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            log.info("  %s: %d/%d (%.1f patch/s)", split, i + 1, len(df), rate)

    lab_arr = np.stack(labels, axis=0)  # (N, 19)
    os.makedirs(args.out_dir, exist_ok=True)
    for p in args.precisions:
        feat_arr = np.stack(feats[p], axis=0)  # (N, 384)
        path = os.path.join(args.out_dir, f"{split}_{p}.npz")
        np.savez_compressed(path, features=feat_arr, labels=lab_arr)
        log.info("[%s] %s: features %s labels %s -> %s",
                 p, split, feat_arr.shape, lab_arr.shape, path)


def main():
    ap = argparse.ArgumentParser(description="Cache per-precision reBEN features for the probe.")
    ap.add_argument("--precisions", nargs="+", default=config.PRECISIONS)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--tree", default=config.REBEN_SUBSET_DIR)
    ap.add_argument("--stats", default=os.path.join(config.INPUT_DIR, "norm_stats.json"))
    ap.add_argument("--out-dir", default=config.FEATURES_DIR)
    ap.add_argument("--img-size", type=int, default=config.INPUT_H)
    ap.add_argument("--max-train", type=int, default=None, help="cap train patches (quick tests)")
    ap.add_argument("--max-test", type=int, default=None, help="cap test patches (quick tests)")
    args = ap.parse_args()

    if not os.path.exists(args.stats):
        raise SystemExit(f"[extract] norm_stats.json not found at {args.stats}.")
    means, stds = reben.load_norm_stats(args.stats)

    core = ov.Core()
    log.info("model_id=%s  out=%s", config.MODEL_ID, args.out_dir)
    for split in args.splits:
        extract_split(core, split, args, means, stds)
    log.info("=== features cached for %s ===", ", ".join(args.splits))


if __name__ == "__main__":
    main()
