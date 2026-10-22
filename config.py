#!/usr/bin/env python3
# =============================================================================
#  config.py  -  single source of truth for the whole experiment
# -----------------------------------------------------------------------------
#  Both the in-container pipeline (pipeline/*.py) and the host scripts import
#  this. Values can be overridden via environment variables so the orchestrator
#  (host/run_experiment.sh) can change behaviour without editing code.
#
#  Import contract:
#    - In the container the file is copied to /work/config.py and scripts add
#      /work to sys.path.
#    - On the host, scripts run from the repo root, so `import config` works.
#
#  Values below are the real paper-1 encoder settings (STRATA):
#  MAE, ViT-S/16, 6-channel early fusion, 224x224, mask 0.70, norm_pix_loss.
# =============================================================================
import os
from pathlib import Path

# --- Optional .env (local, gitignored) --------------------------------------
#  Machine/network-specific values (e.g. the Tasmota IP) live in a .env at the
#  repo root, NOT hardcoded here, so published code carries no private config.
#  Stdlib only (no python-dotenv). Real exported env vars and
#  `docker --env-file` WIN: we only setdefault, never overwrite. In the
#  container there is no .env (gitignored, not copied) -> the guard skips it.
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.is_file():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


def _env(name, default):
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def _env_int(name, default):
    try:
        return int(_env(name, default))
    except (TypeError, ValueError):
        return default


def _env_list(name, default):
    v = _env(name, None)
    return v.split() if v else list(default)


# --- Model input: paper-1 encoder (paired SAR + optical, early fusion) -------
#  MAE ViT-S/16 trained on 6-channel EARLY-FUSION inputs:
#      optical (Sentinel-2): B4, B3, B2        -> 3 channels
#      SAR     (Sentinel-1): VV, VH, VV-VH     -> 3 channels
#  => single stacked input of 6 channels (NOT two separate inputs).
#
#  CHANNEL ORDER MATTERS at inference: it must match training. The paper-1
#  dataloader stacks OPTICAL FIRST, then SAR (see CHANNEL_ORDER). If your
#  encoder.pt was trained with a different order, change it here, or the
#  embeddings are meaningless.
SAR_CHANNELS = _env_int("RSEI_SAR_CH", 3)
OPTICAL_CHANNELS = _env_int("RSEI_OPT_CH", 3)
INPUT_MODE = _env("RSEI_INPUT_MODE", "stacked")  # "stacked" (early fusion) | "dual"
TOTAL_CHANNELS = SAR_CHANNELS + OPTICAL_CHANNELS  # 6

#  Order used to build the 6-channel tensor (optical first, then SAR).
CHANNEL_ORDER = ["S2_B4", "S2_B3", "S2_B2", "S1_VV", "S1_VH", "S1_VV_minus_VH"]

#  Patch size fed to the encoder: 224x224 crops (paper 1).
INPUT_H = _env_int("RSEI_INPUT_H", 224)
INPUT_W = _env_int("RSEI_INPUT_W", 224)

# --- Encoder architecture (recorded as metadata; informational) --------------
ARCH = {
    "family": "MAE",
    "backbone": "ViT-S/16",
    "patch_size": 16,
    "img_size": 224,
    "in_chans": TOTAL_CHANNELS,
    "mask_ratio": 0.70,
    "norm_pix_loss": True,
    "decoder_embed_dim": 512,
    "note": "deploy uses the ENCODER only; the MAE decoder is dropped.",
}

# --- Precisions / devices to sweep ------------------------------------------
PRECISIONS = _env_list("RSEI_PRECISIONS", ["fp32", "fp16", "int8"])
DEVICES = _env_list("RSEI_DEVICES", ["CPU", "GPU"])

#  Hardware profile label for this run: "avx2" | "noavx2" | "dev".
#  The measurement images bake it (/opt/rsei_target) and the entrypoint exports
#  RSEI_TARGET before Python starts, so on the N5105/Lenovo it is the real
#  profile. Running by hand on the dev box leaves it unset -> "dev", which is a
#  deliberate, self-documenting label (not a measurement target), not a gap.
TARGET = _env("RSEI_TARGET", "dev")

# --- Benchmark protocol ------------------------------------------------------
N_ITER = _env_int("RSEI_N_ITER", 200)  # measured iterations per repeat
WARMUP = _env_int("RSEI_WARMUP", 20)  # discarded warmup iterations
REPEATS = _env_int("RSEI_REPEATS", 5)  # independent repeats -> mean +/- CI95
BATCH = 1  # edge deployment: one image at a time

# --- Energy measurement ------------------------------------------------------
TASMOTA_IP = _env("RSEI_TASMOTA_IP", "")  # e.g. "192.168.1.50"; "" = RAPL only
POWER_INTERVAL_S = float(_env("RSEI_POWER_INTERVAL", "0.5"))
IDLE_SECS = _env_int("RSEI_IDLE_SECS", 60)

# --- Paths -------------------------------------------------------------------
#  Everything lives under the mounted volume so it is reachable from the host
#  (and from outside the container). In Docker the volume is /work/{input,output};
#  running natively (e.g. export/quantize on your Windows dev box, no container)
#  it falls back to repo-relative ./input and ./output. Override with
#  RSEI_INPUT_DIR / RSEI_OUTPUT_DIR to point anywhere else.
_IN_CONTAINER = os.path.isdir("/work")
_BASE = "/work" if _IN_CONTAINER else os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = _env("RSEI_INPUT_DIR", os.path.join(_BASE, "input"))
OUTPUT_DIR = _env("RSEI_OUTPUT_DIR", os.path.join(_BASE, "output"))
LOGS_DIR = _env("RSEI_LOGS_DIR", os.path.join(_BASE, "logs"))
#  Provenance / chain-of-custody record. Treated as DATA, like input/output/logs:
#  gitignored, NOT versioned. But unlike those it is the local source of truth of
#  the data - the acta of what was produced and with which SHA256 - and is meant
#  to persist across the whole project, not be wiped between runs.
STATE_DIR = _env("RSEI_STATE_DIR", os.path.join(_BASE, "state"))
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")


# --- Run identity ------------------------------------------------------------
#  One orchestrated measurement = one RUN_ID = one self-contained folder
#  output/runs/<RUN_ID>/ holding host_env.json, envtest.json, bench_<prec>.json,
#  power_<label>.csv and summary.csv. The orchestrator (run_experiment.sh)
#  computes RUN_ID on the HOST and injects it as RSEI_RUN_ID, so every container
#  writes under the SAME id. Without that, each container would mint its own
#  hostname-based id and scatter the files across output/. Falls back to a
#  locally generated id when run standalone (manual `docker run`, no orchestrator)
#  so nothing breaks outside the harness.
def _default_run_id():
    import socket
    from datetime import UTC, datetime

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{socket.gethostname()}_{TARGET}_{stamp}"


RUN_ID = _env("RSEI_RUN_ID", "") or _default_run_id()
RUN_DIR = os.path.join(RUNS_DIR, RUN_ID)
CHECKPOINT = _env("RSEI_CHECKPOINT", os.path.join(INPUT_DIR, "encoder.pt"))
ONNX_OPSET = _env_int("RSEI_ONNX_OPSET", 17)
#  INT8 calibration pack (normalised 6x224x224 reBEN tensors), built once by
#  prepare_data.py (harness) and copied here. Only INT8 needs it; fp32/fp16 do
#  not. Override with RSEI_CALIB_PACK.
CALIB_PACK = _env("RSEI_CALIB_PACK", os.path.join(INPUT_DIR, "calib", "calib_int8.npz"))

#  Benchmark input: one real, preprocessed test-split patch (1x6x224x224),
#  packed once by prepare_data.py so the LEAN target (no rasterio/torch) can feed
#  03 a REAL sample instead of synthetic noise. Tiny; ships beside the IR.
#  Override with RSEI_EVAL_SAMPLE.
EVAL_SAMPLE = _env("RSEI_EVAL_SAMPLE", os.path.join(INPUT_DIR, "bench_sample.npz"))
#  Accuracy axis: the reBEN subset (train/test trees) is the SAME data for every
#  model, so it stays shared under input/. The cached features are per-model and
#  live under MODEL_DIR (defined below, FEATURES_DIR). Only needed on the dev box.
REBEN_SUBSET_DIR = _env("RSEI_REBEN_SUBSET", os.path.join(INPUT_DIR, "reben_subset"))

# --- Model identity & per-step, per-precision folder layout ------------------
#  Every transformation lands in an identified folder so each step is traceable:
#     input/<MODEL_ID>/onnx/model.onnx                      (ONNX)
#     input/<MODEL_ID>/<precision>/model.xml + model.bin    (OpenVINO IR)
#  Transformed models live under input/ (what the benchmark consumes); results
#  live under output/. The SHA256 of each artifact is recorded separately in the
#  provenance manifest at state/<MODEL_ID>/manifest.json (see MODEL_STATE_DIR).
#  MODEL_ID defaults to the checkpoint filename stem, overridable with
#  RSEI_MODEL_ID.
MODEL_ID = _env(
    "RSEI_MODEL_ID",
    os.path.splitext(os.path.basename(CHECKPOINT))[0] or "encoder",
)
MODEL_DIR = os.path.join(INPUT_DIR, MODEL_ID)
ONNX_PATH = os.path.join(MODEL_DIR, "onnx", "model.onnx")

#  Per-model provenance folder (local source of truth; gitignored). Holds
#  manifest.json.
MODEL_STATE_DIR = os.path.join(STATE_DIR, MODEL_ID)

#  Cached 384-d features per (split, precision) for the accuracy axis. These are
#  PRODUCED (vary by model and precision, not shared across models like input/),
#  so they live under output/<MODEL_ID>/ - namespaced, otherwise a second encoder
#  (e.g. the comparative paper) would overwrite the first and silently corrupt the
#  comparison. Override with RSEI_FEATURES_DIR.
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, MODEL_ID)
FEATURES_DIR = _env("RSEI_FEATURES_DIR", os.path.join(MODEL_OUTPUT_DIR, "features"))


def ir_dir(precision):
    """Folder holding the OpenVINO IR for a given precision."""
    return os.path.join(MODEL_DIR, precision)


def ir_xml(precision):
    """Path to the IR .xml for a given precision (model.bin sits beside it)."""
    return os.path.join(ir_dir(precision), "model.xml")


def input_shapes():
    """Return the list of input shapes [B,C,H,W] the model expects.

    "stacked" -> one shape with TOTAL_CHANNELS (early fusion, paper 1).
    "dual"    -> two shapes, SAR then optical.
    """
    if INPUT_H <= 0 or INPUT_W <= 0:
        raise ValueError("INPUT_H/INPUT_W must be positive.")
    if INPUT_MODE == "stacked":
        return [[BATCH, TOTAL_CHANNELS, INPUT_H, INPUT_W]]
    if INPUT_MODE == "dual":
        return [
            [BATCH, SAR_CHANNELS, INPUT_H, INPUT_W],
            [BATCH, OPTICAL_CHANNELS, INPUT_H, INPUT_W],
        ]
    raise ValueError(f"Unknown INPUT_MODE: {INPUT_MODE!r} (use 'stacked' or 'dual')")


def as_dict():
    """Config snapshot, embedded into every result file as metadata."""
    return {
        "model_id": MODEL_ID,
        "target": TARGET,
        "sar_channels": SAR_CHANNELS,
        "optical_channels": OPTICAL_CHANNELS,
        "total_channels": TOTAL_CHANNELS,
        "channel_order": CHANNEL_ORDER,
        "input_mode": INPUT_MODE,
        "input_h": INPUT_H,
        "input_w": INPUT_W,
        "arch": ARCH,
        "precisions": PRECISIONS,
        "devices": DEVICES,
        "n_iter": N_ITER,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "batch": BATCH,
        "tasmota_ip_set": bool(TASMOTA_IP),
        "power_interval_s": POWER_INTERVAL_S,
        "idle_secs": IDLE_SECS,
        "onnx_opset": ONNX_OPSET,
    }
