#!/usr/bin/env python3
# =============================================================================
#  03_infer_benchmark.py  -  run inference and TIME it (inside the container)
# -----------------------------------------------------------------------------
#  For one precision (fp32/fp16/int8), run inference on the requested devices
#  (CPU, GPU) and write one JSON per precision containing results for ALL
#  devices, plus a full metadata + config snapshot for reproducibility.
#
#  Measurement protocol:
#   - batch = 1 (edge deployment, one image at a time)
#   - warmup iterations discarded (first inference loads kernels/cache)
#   - N measured iterations -> mean, std, p50, p95, min, max
#   - UTC start/end timestamps. The container shares the host kernel clock, so
#     these timestamps cross-reference directly with the energy files
#     (RAPL + Tasmota) the host logs: a single shared clock.
#   - thermal reading before/after (Tiny PCs throttle; record the drift).
#   - load_eval_sample() uses a real preprocessed test patch when present, and
#     falls back to synthetic noise (flagged synthetic_input=true) otherwise.
# =============================================================================
import argparse
import os
import statistics
import sys
import time
from datetime import UTC, datetime

import numpy as np
import openvino as ov

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jsonio
import metadata
import stats
from logsetup import get_logger

import config


def now_iso():
    return datetime.now(UTC).isoformat()


def _synthetic_feed(model):
    """Correctly-shaped NOISE for every model input (fallback only)."""
    feed = {}
    for port in model.inputs:
        shape = [d if d > 0 else 1 for d in port.get_partial_shape().get_min_shape()]
        if not shape or any(s == 0 for s in shape):
            shape = [config.BATCH, config.TOTAL_CHANNELS, config.INPUT_H or 1, config.INPUT_W or 1]
        feed[port] = np.random.rand(*shape).astype(np.float32)
    return feed


def load_eval_sample(model):
    """Return (feed_dict, synthetic_flag) matching the model's inputs.

    Prefers a REAL preprocessed sample: config.EVAL_SAMPLE is a tiny .npz packed
    on the dev box by prepare_data.py (one test-split patch, 1x6x224x224,
    normalised exactly as in training) and shipped beside the IR. The lean target
    only needs numpy to read it - no rasterio/torch. If the pack is absent the
    harness still runs on NOISE and flags synthetic_input=True (latency/energy
    stay valid; only task accuracy would not).

    Note: inference latency on a dense ViT is input-INDEPENDENT (same shapes ->
    same compute), so the real sample does not change the timing; it removes the
    synthetic asterisk and uses the same preprocessing as the probe.
    """
    pack = config.EVAL_SAMPLE
    if not os.path.exists(pack):
        return _synthetic_feed(model), True
    x = np.load(pack)["x"].astype(np.float32)  # (1, 6, 224, 224), already normalised
    ports = list(model.inputs)
    if len(ports) != 1:
        # Real sample is a single stacked tensor; multi-input models aren't covered.
        return _synthetic_feed(model), True
    return {ports[0]: x}, False


def _measure_once(core, model, device, n_iter, warmup, feed):
    """One independent measurement block: fresh compile, warmup, timed loop.

    Re-compiling per repeat resets device/runtime state, so the spread across
    repeats reflects real run-to-run variance, not just within-loop jitter.
    """
    compiled = core.compile_model(model, device)
    cfeed = {compiled.input(i): list(feed.values())[i] for i in range(len(feed))}

    for _ in range(warmup):
        compiled(cfeed)

    lat_ms = []
    t_start = now_iso()
    t0 = time.time()
    for _ in range(n_iter):
        t = time.perf_counter()
        compiled(cfeed)
        lat_ms.append((time.perf_counter() - t) * 1000.0)
    t1 = time.time()
    t_end = now_iso()

    s = sorted(lat_ms)
    return {
        "ts_start_utc": t_start,
        "ts_end_utc": t_end,
        "wall_seconds": round(t1 - t0, 4),
        "latency_ms_mean": round(statistics.mean(lat_ms), 4),
        "latency_ms_std": round(statistics.pstdev(lat_ms), 4),
        "latency_ms_p50": round(s[int(0.50 * (len(s) - 1))], 4),
        "latency_ms_p95": round(s[int(0.95 * (len(s) - 1))], 4),
        "latency_ms_min": round(min(lat_ms), 4),
        "latency_ms_max": round(max(lat_ms), 4),
        "throughput_ips": round(n_iter / (t1 - t0), 3),
        "thermal_c_after": metadata.thermal_c(),
    }


def bench_device(core, model, device, n_iter, warmup, feed, repeats, log):
    """Run `repeats` independent measurement blocks and aggregate with CI95."""
    per_repeat = []
    for r in range(repeats):
        log.info("  repeat %d/%d on %s ...", r + 1, repeats, device)
        per_repeat.append(_measure_once(core, model, device, n_iter, warmup, feed))

    lat_means = [pr["latency_ms_mean"] for pr in per_repeat]
    thr_means = [pr["throughput_ips"] for pr in per_repeat]
    return {
        "device": device,
        "n_iter": n_iter,
        "warmup": warmup,
        "repeats": repeats,
        "ts_start_utc": per_repeat[0]["ts_start_utc"],
        "ts_end_utc": per_repeat[-1]["ts_end_utc"],
        # Aggregate ACROSS repeats (this is what the paper reports):
        "latency_ms_agg": stats.summarize(lat_means),
        "throughput_ips_agg": stats.summarize(thr_means),
        # Full per-repeat detail, for auditability:
        "per_repeat": per_repeat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8"])
    ap.add_argument(
        "--model",
        default=None,
        help="path to the OpenVINO IR .xml (default: input/<model_id>/<precision>/model.xml)",
    )
    ap.add_argument("--devices", nargs="+", default=config.DEVICES)
    ap.add_argument("--n-iter", type=int, default=config.N_ITER)
    ap.add_argument("--warmup", type=int, default=config.WARMUP)
    ap.add_argument("--repeats", type=int, default=config.REPEATS)
    ap.add_argument("--outdir", default=config.RUN_DIR)
    args = ap.parse_args()

    # Default the model path from the identified per-precision folder.
    if args.model is None:
        args.model = config.ir_xml(args.precision)

    rid = config.RUN_ID
    log = get_logger("03_benchmark", run_id=f"{rid}__{args.precision}")
    os.makedirs(args.outdir, exist_ok=True)
    log.info("model_id=%s  model=%s", config.MODEL_ID, args.model)

    core = ov.Core()
    model = core.read_model(args.model)
    feed, synthetic = load_eval_sample(model)

    # SHA256 of the exact model bytes measured (IR = .xml + .bin), for traceability.
    bin_path = os.path.splitext(args.model)[0] + ".bin"
    result = {
        "kind": "benchmark",
        "run_id": rid,
        "precision": args.precision,
        "model_path": args.model,
        "model_sha256": {
            "xml": metadata.sha256_file(args.model),
            "bin": metadata.sha256_file(bin_path),
        },
        "synthetic_input": synthetic,  # True = noise (harness), False = real
        "available_devices": core.available_devices,
        "config": config.as_dict(),  # full config snapshot (metadata)
        "metadata": metadata.collect(),  # system metadata (metadata)
        "devices": {},
    }

    for dev in args.devices:
        present = dev in core.available_devices or any(
            d.startswith(dev) for d in core.available_devices
        )
        if not present:
            result["devices"][dev] = {"skipped": f"{dev} not available"}
            log.warning("%s not available, skipped", dev)
            continue
        log.info("benchmark %s on %s (%d repeats) ...", args.precision, dev, args.repeats)
        try:
            result["devices"][dev] = bench_device(
                core, model, dev, args.n_iter, args.warmup, feed, args.repeats, log
            )
        except Exception as exc:
            # A failing device must not lose the others: e.g. a GPU OpenCL kernel that
            # will not compile on this driver. Record the error for this device and keep
            # the results already gathered, so the JSON is still written (one bad device
            # never voids the whole run).
            result["devices"][dev] = {"error": str(exc).strip()[:2000]}
            log.warning("%s benchmark failed (%s); recorded, continuing", dev, type(exc).__name__)

    out_path = os.path.join(args.outdir, f"bench_{args.precision}.json")
    with open(out_path, "w") as f:
        jsonio.dump(result, f)
    log.info("results -> %s (json backend: %s)", out_path, jsonio.BACKEND)
    if synthetic:
        log.warning(
            "SYNTHETIC input (noise). Latency/energy valid, task accuracy NOT. "
            "Fill load_eval_sample() with real data."
        )


if __name__ == "__main__":
    main()
