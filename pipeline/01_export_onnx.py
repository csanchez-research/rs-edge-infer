#!/usr/bin/env python3
# =============================================================================
#  01_export_onnx.py  -  PyTorch -> ONNX + graph validation
# -----------------------------------------------------------------------------
#  Exports the paper-1 MAE ENCODER (inference mode, NO masking) to ONNX, ready
#  to be converted/quantised by 02_quantize.py.
#
#  Runs on a dev machine (your Windows PC or the training env) - it needs
#  torch + timm (docker/requirements-dev.txt), not the lean benchmark image.
#  Only the benchmark (03) needs to run on the N5105 itself.
#
#  Input shape and channels come from config.py (paper 1: 6-ch early fusion,
#  224x224). The only thing you may need to point at is your encoder.pt path.
# =============================================================================
import argparse
import os
import sys

import torch

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifest
from encoder_infer import load_encoder
from logsetup import get_logger

import config


def dummy_inputs():
    return [torch.randn(*s, dtype=torch.float32) for s in config.input_shapes()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=config.CHECKPOINT)
    ap.add_argument("--out", default=config.ONNX_PATH)
    ap.add_argument("--opset", type=int, default=config.ONNX_OPSET)
    args = ap.parse_args()

    log = get_logger("01_export")
    log.info("model_id=%s  ->  %s", config.MODEL_ID, args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if config.INPUT_MODE != "stacked":
        raise SystemExit(
            "01_export: paper-1 encoder is early-fusion 'stacked'. "
            "Dual-input export is not wired for this encoder."
        )

    model = load_encoder(
        args.checkpoint,
        in_chans=config.TOTAL_CHANNELS,
        img_size=config.INPUT_H,
        patch=config.ARCH["patch_size"],
    )
    x = dummy_inputs()[0]

    with torch.no_grad():
        emb = model(x)
    log.info("forward OK, embedding shape = %s (expect [1, 384])", tuple(emb.shape))

    torch.onnx.export(
        model,
        x,
        args.out,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
    )
    log.info(
        "ONNX exported -> %s (channels=%d, %dx%d)",
        args.out,
        config.TOTAL_CHANNELS,
        config.INPUT_H,
        config.INPUT_W,
    )

    import onnx

    onnx.checker.check_model(onnx.load(args.out))
    log.info("onnx.checker OK")

    # Record SHA256 of the input checkpoint and the produced ONNX in the model
    # manifest, for an end-to-end traceability chain (encoder.pt -> onnx -> IR).
    manifest.record("checkpoint", args.checkpoint)
    mpath = manifest.record("onnx", args.out)
    log.info("manifest updated -> %s", mpath)


if __name__ == "__main__":
    main()
