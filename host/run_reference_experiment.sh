#!/usr/bin/env bash
# =============================================================================
#  run_reference_experiment.sh  -  LATENCY-ONLY reference run (CPU, no energy)
# -----------------------------------------------------------------------------
#  For a box that is NOT a cheap edge target but a reference point - e.g. the dev
#  workstation (AVX2 + VNNI) as the top of the SIMD gradient. It runs NATIVELY
#  (python + openvino, no docker, no /dev/dri), which is what you want when the
#  dev itself already runs inside a container: no docker-in-docker.
#
#  It does NOT touch run_experiment.sh (the validated energy-measuring orchestrator
#  for the real edge targets). This one is deliberately stripped:
#     - CPU only (RSEI_DEVICES=CPU), no iGPU
#     - no energy at all (no RAPL / no wall plug) -> latency-only summary
#     - self-labelled: writes straight to output/<label>/runs/<run>/ (no collect)
#
#  Run it from inside the dev container (where you run 05/06/build), from repo root:
#     RSEI_LABEL=dev bash host/run_reference_experiment.sh
#
#  Needs: python with openvino + numpy, the IRs in input/<model>/<prec>/, and
#  input/bench_sample.npz - all of which the dev already has.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."                               # repo root
LABEL="${RSEI_LABEL:-dev}"
HOST="$(hostname)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN="${HOST}_${LABEL}_${STAMP}"

# One labelled folder; config.py derives RUN_DIR = <OUTPUT_DIR>/runs/<RUN_ID>, so
# pointing OUTPUT_DIR at output/<label> lands everything under output/<label>/runs/.
export RSEI_OUTPUT_DIR="$(pwd)/output/${LABEL}"
export RSEI_RUN_ID="$RUN"
export RSEI_DEVICES="CPU"                              # reference = CPU only
PRECS="${RSEI_PRECISIONS:-fp32 fp16 int8}"
RUN_DIR="${RSEI_OUTPUT_DIR}/runs/${RUN}"
mkdir -p "$RUN_DIR" logs

echo "=== reference run: label=$LABEL  run=$RUN  (latency-only, CPU, native) ==="

# Host provenance (OS, python, openvino) so the assets have the version/flags.
python - "$RUN" > "$RUN_DIR/host_env.json" <<'PY'
import json, platform, sys
try:
    import openvino
    ov = openvino.__version__
except Exception:
    ov = None
print(json.dumps({
    "run_id": sys.argv[1],
    "os_release": platform.platform(),
    "python": sys.version.split()[0],
    "openvino": ov,
    "note": "latency-only reference, native (no docker, no energy)",
}, indent=2))
PY

# Environment gate: records cpu flags (AVX2/VNNI) + openvino version in envtest.json.
echo "--- 00_test (records cpu flags + openvino version) ---"
python pipeline/00_test.py || echo "!! 00_test returned non-zero; continuing (latency still valid)"

# Benchmarks per precision - CPU only, no power sampling.
for P in $PRECS; do
    echo "--- benchmark precision=$P (CPU) ---"
    python pipeline/03_infer_benchmark.py --precision "$P"
done

# Merge -> latency-only summary (no power CSVs, so energy columns stay blank).
echo "--- merge ---"
python host/04_merge_stats.py --run-dir "$RUN_DIR"

echo "=== done. results in $RUN_DIR (latency-only). ==="
echo "    the paper-assets generator reads output/${LABEL}/ and shows it as a"
echo "    latency-only reference node (no energy rows/figures)."
