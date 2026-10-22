#!/usr/bin/env python3
# =============================================================================
#  02_quantize.py  -  build the precision variants to compare
# -----------------------------------------------------------------------------
#  From the fp32 ONNX, produce OpenVINO IR in:
#     - fp32 : direct conversion
#     - fp16 : half precision
#     - int8 : post-training quantisation with NNCF (needs calibration data)
#
#  Hardware reminder:
#   - On the N5105 (no AVX2/VNNI) INT8 on CPU may NOT speed up, or may be worse.
#     That is a RESULT of the paper, not a bug. We still build INT8 to measure
#     it; the iGPU is where INT8 makes most sense on that chip.
#
#  Calibration: INT8 PTQ uses REAL paired SAR+optical samples from the reBEN
#  'validation' split (built by prepare_data.py into config.CALIB_PACK), with the
#  same preprocessing as the probe - never random noise.
#
#  Method reference: post-training INT8 quantization via NNCF
#  [Kozlov et al. 2020, arXiv:2002.08679] on the OpenVINO runtime.
# =============================================================================
import argparse
import os
import sys

import openvino as ov

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifest
from logsetup import get_logger

import config

#  NOTE: nncf.quantize() emits a DeprecationWarning because NNCF still imports
#  the deprecated `openvino.runtime` module internally. We do NOT silence it: it
#  is a true statement about a dependency, and hiding a warning hides a truth.
#  It is not ours to fix (our code uses the new flat `openvino` API); it goes
#  away when NNCF migrates. Left visible on purpose.

log = get_logger("02_quantize")


def to_ir(onnx_path, out_xml, compress_to_fp16):
    model = ov.convert_model(onnx_path)
    ov.save_model(model, out_xml, compress_to_fp16=compress_to_fp16)
    log.info("saved %s -> %s", "fp16" if compress_to_fp16 else "fp32", out_xml)


def calibration_dataset():
    """Real calibration samples for INT8 PTQ, loaded from the calib pack.

    The pack (config.CALIB_PACK) is built by prepare_data.py from reBEN with the
    SAME preprocessing as the probe (reben_loader): 6x224x224, resized and
    per-band normalised. NNCF runs these forward to set activation ranges.
    """
    import nncf
    import numpy as np

    pack = config.CALIB_PACK
    if not os.path.exists(pack):
        raise SystemExit(
            f"INT8 needs the calibration pack at {pack}. Build it with "
            "prepare_data.py (select -> extract -> pack) and copy the .npz here. "
            "fp32/fp16 do not need it."
        )
    data = np.load(pack)
    x = data["x"].astype(np.float32)  # (N, 6, 224, 224), already normalised
    n = int(x.shape[0])
    log.info("calibration: %d samples from %s", n, pack)
    manifest.record("calib_pack", pack)  # provenance: calib bytes -> int8 IR
    samples = [x[i : i + 1] for i in range(n)]  # each (1, 6, 224, 224)
    # transform_func (2nd positional arg) maps a dataset item -> model input. Our
    # items are already model-ready arrays, so it is the identity. Passed
    # positionally on purpose: the parameter is named transform_func, not _fn.
    return nncf.Dataset(samples, lambda s: s), n


def to_int8(onnx_path, out_xml):
    import nncf

    model = ov.convert_model(onnx_path)
    # subset_size = the whole calib pack (explicit, documented): calibrate on ALL
    # the samples we prepared, not NNCF's default cap of 300.
    dataset, n = calibration_dataset()
    quantized = nncf.quantize(model, dataset, subset_size=n)
    ov.save_model(quantized, out_xml, compress_to_fp16=False)
    log.info("saved int8 -> %s (PTQ, subset_size=%d)", out_xml, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=config.ONNX_PATH)
    ap.add_argument("--precisions", nargs="+", default=config.PRECISIONS)
    args = ap.parse_args()

    log.info("model_id=%s  onnx=%s", config.MODEL_ID, args.onnx)
    for p in args.precisions:
        # Each precision gets its own identified folder: input/<model_id>/<p>/model.xml
        os.makedirs(config.ir_dir(p), exist_ok=True)
        xml = config.ir_xml(p)
        if p == "fp32":
            to_ir(args.onnx, xml, compress_to_fp16=False)
        elif p == "fp16":
            to_ir(args.onnx, xml, compress_to_fp16=True)
        elif p == "int8":
            to_int8(args.onnx, xml)
        else:
            log.warning("unknown precision, skipped: %s", p)
            continue
        # Record both IR files (xml + bin) in the model manifest.
        bin_path = os.path.splitext(xml)[0] + ".bin"
        manifest.record(f"ir_{p}_xml", xml)
        manifest.record(f"ir_{p}_bin", bin_path)
    log.info("manifest -> %s", manifest.manifest_path())


if __name__ == "__main__":
    main()
