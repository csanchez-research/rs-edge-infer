#!/usr/bin/env bash
# =============================================================================
#  run_experiment.sh  -  ORCHESTRATOR (runs on the HOST)
# -----------------------------------------------------------------------------
#  Does, in order:
#    1. Detect AVX2 on THIS machine -> choose TARGET (avx2 / noavx2).
#    2. Build the Docker image with --build-arg TARGET (correct variant).
#    3. Run the 00_test gate. If a critical check fails, STOP.
#    4. Measure IDLE with the energy sampler (baseline to subtract).
#    5. For each precision: start the sampler, run the benchmark in Docker,
#       stop the sampler.
#
#  ENERGY + ROOT: energy is sampled on the HOST by measure_power.py. Reading the
#  RAPL package counter (/sys/class/powercap/.../energy_uj) requires ROOT (file
#  mode 0400 since the side-channel lockdown); Tasmota (HTTP) does not. Run this
#  orchestrator as root to get BOTH sources:
#       sudo -E ./host/run_experiment.sh
#  Without root the run still completes, but RAPL energy is EMPTY (Tasmota only).
#
#  Config is centralised in config.py, overridable via RSEI_* env vars. The few
#  knobs below mirror the most common ones for convenience.
#    TASMOTA_IP   plug IP (e.g. 192.168.1.50). Empty -> RAPL only.
#    N_ITER       measured iterations per benchmark (default from config)
#    IDLE_SECS    idle measurement duration (default from config)
#    PRECISIONS   "fp32 fp16 int8"
#    SKIP_BUILD   =1 to reuse an already-loaded image instead of building here
#                 (build on the dev box, `docker save`, ship it, `docker load` on
#                  the target -> the target needs no build toolchain or package index).
#    RSEI_IMAGE   image name[:tag] to run (default rs-edge-infer); with SKIP_BUILD
#                 set this to the loaded tag, e.g. rs-edge-infer:noavx2.
#    RSEI_TEST_ONLY =1 to stop after the 00_test environment gate (no measurement).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."     # repo root

# Mirror shell vars into the RSEI_* names config.py reads.
# Export the Tasmota IP ONLY when set, so an empty value does not clobber
# config.py's own default (os.environ.get returns "" for an empty-but-set var).
RSEI_TASMOTA_IP="${TASMOTA_IP:-${RSEI_TASMOTA_IP:-}}"
[ -n "$RSEI_TASMOTA_IP" ] && export RSEI_TASMOTA_IP || true
export RSEI_N_ITER="${N_ITER:-${RSEI_N_ITER:-200}}"
export RSEI_REPEATS="${REPEATS:-${RSEI_REPEATS:-5}}"
export RSEI_IDLE_SECS="${IDLE_SECS:-${RSEI_IDLE_SECS:-60}}"
export RSEI_PRECISIONS="${PRECISIONS:-${RSEI_PRECISIONS:-fp32 fp16 int8}}"
IMAGE="${RSEI_IMAGE:-rs-edge-infer}"   # override with RSEI_IMAGE for a loaded/tagged image
HOST="$(hostname)"
mkdir -p output logs

# --- 0. RAPL needs root ------------------------------------------------------
# energy_uj is 0400 (root-only) since the side-channel lockdown. measure_power.py
# reads it directly; if we are not root it silently logs empty RAPL rows (the
# wall plug still works). Warn loudly rather than discovering it in 04_merge.
if [ "$(id -u)" -ne 0 ]; then
    echo "!! WARNING: not running as root -> RAPL package energy will be UNREADABLE."
    echo "!!          Only the Tasmota wall-plug will be logged (J_per_infer_rapl = empty)."
    echo "!!          For BOTH energy sources, relaunch:  sudo -E $0 $*"
fi

# --- 1. Build (or reuse) the single image ------------------------------------
# SKIP_BUILD=1 reuses an already-loaded image (built on the dev PC and shipped as
# a tarball) -> the target downloads nothing. Otherwise build here; the image
# self-detects avx2/noavx2 from THIS machine's CPU at build time.
if [ -n "${SKIP_BUILD:-}" ]; then
    echo "--- SKIP_BUILD set: reusing existing image '${IMAGE}' (no build) ---"
    if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
        echo "!! SKIP_BUILD set but image '${IMAGE}' is not loaded. Load it first, e.g.:"
        echo "!!   zstd -d -c rsei-noavx2.tar.zst | docker load"
        echo "!! then run with its tag:  RSEI_IMAGE=rs-edge-infer:noavx2 SKIP_BUILD=1 ..."
        exit 1
    fi
else
    # Load Nexus pip proxy settings if nexus.env is present (keeps the IP out of git).
    if [ -f nexus.env ]; then
        set -a; . ./nexus.env; set +a
        echo "--- nexus.env loaded (pip index: ${PIP_INDEX_URL:-unset}) ---"
    fi
    echo "--- build ---"
    docker build -f docker/Dockerfile \
        ${PIP_INDEX_URL:+--build-arg PIP_INDEX_URL="$PIP_INDEX_URL"} \
        ${PIP_EXTRA_INDEX_URL:+--build-arg PIP_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL"} \
        ${PIP_TRUSTED_HOST:+--build-arg PIP_TRUSTED_HOST="$PIP_TRUSTED_HOST"} \
        -t "${IMAGE}" .
fi

# Read the baked target back from the image (single source of truth) just to
# label this run's files; the container itself uses its own immutable value.
TARGET="$(docker run --rm --entrypoint cat "${IMAGE}" /opt/rsei_target)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN="${HOST}_${TARGET}_${STAMP}"
# One run = one self-contained folder. Inject RSEI_RUN_ID so the containers
# (00_test, 03) write under this SAME id (config.RUN_DIR) instead of minting
# their own hostname-based id and scattering files across output/.
RUN_DIR="output/runs/${RUN}"
mkdir -p "$RUN_DIR"
export RSEI_RUN_ID="$RUN"
# Mirror everything this script prints to a per-run log (console + file). In
# test-only mode skip it, so the gate's output streams straight to the caller
# (tee does not forward cleanly over a one-shot SSH session).
if [ -z "${RSEI_TEST_ONLY:-}" ]; then
    exec > >(tee -a "logs/${RUN}__orchestrator.log") 2>&1
fi
echo "=== run_experiment: host=$HOST target=$TARGET run=$RUN ==="

# --- 2b. HOST environment provenance ----------------------------------------
# The in-container metadata records the kernel + CPU (shared with the host), but
# its /etc/os-release is the CONTAINER's (Ubuntu). The HOST distro (e.g. Debian
# 13 trixie), its exact version, the kernel and the Docker version are recorded
# HERE, on the host, where they are real - one sidecar per run, for pinning.
python3 - "$RUN" "$IMAGE" > "${RUN_DIR}/host_env.json" <<'PY'
import json, platform, subprocess, sys
from datetime import UTC, datetime

def _read(p):
    try:
        return open(p).read().strip()
    except OSError:
        return None

def _os_release():
    d = {}
    try:
        for line in open("/etc/os-release"):
            if "=" in line:
                k, v = line.rstrip().split("=", 1)
                d[k] = v.strip('"')
    except OSError:
        pass
    return d.get("PRETTY_NAME")

def _docker():
    try:
        return subprocess.check_output(["docker", "--version"], text=True).strip()
    except Exception:
        return None

run, image = sys.argv[1], sys.argv[2]
json.dump({
    "collected_utc": datetime.now(UTC).isoformat(),
    "run_id": run,
    "os_release": _os_release(),         # host distro, e.g. "Debian GNU/Linux 13 (trixie)"
    "debian_version": _read("/etc/debian_version"),
    "kernel": platform.release(),
    "docker": _docker(),
    "image": image,
}, sys.stdout, indent=2)
PY
echo "--- host env recorded -> ${RUN_DIR}/host_env.json ---"

# Pass RSEI_* env into the container so config.py sees the same settings.
# NOTE: RSEI_TARGET is NOT passed - the image's baked value wins.
ENV_ARGS=(-e RSEI_RUN_ID -e RSEI_TASMOTA_IP -e RSEI_N_ITER -e RSEI_REPEATS -e RSEI_IDLE_SECS -e RSEI_PRECISIONS)
[ -n "${RSEI_INPUT_H:-}" ] && ENV_ARGS+=(-e RSEI_INPUT_H)
[ -n "${RSEI_INPUT_W:-}" ] && ENV_ARGS+=(-e RSEI_INPUT_W)
[ -n "${RSEI_INPUT_MODE:-}" ] && ENV_ARGS+=(-e RSEI_INPUT_MODE)

DOCKER_COMMON=(--rm --device /dev/dri
    "${ENV_ARGS[@]}"
    -v "$(pwd)/config.py:/work/config.py:ro"   # live code
    -v "$(pwd)/pipeline:/work/pipeline:ro"      # live code
    -v "$(pwd)/input:/work/input:ro"
    -v "$(pwd)/output:/work/output"
    -v "$(pwd)/logs:/work/logs")

# --- 3. 00_test gate ---------------------------------------------------------
echo "--- 00_test ---"
if ! docker run "${DOCKER_COMMON[@]}" "${IMAGE}" test; then
    echo "!! 00_test failed on critical checks. See ${RUN_DIR}/envtest.json. Stopping."
    exit 1
fi

# --- 3b. Energy-sources check (part of the readiness gate; runs in test too) -
# Verify the measurement sources with the sampler's own reader functions: RAPL
# package counter (needs root) and/or the Tasmota wall plug (HTTP). Abort only
# if NEITHER is available. Part of 'is everything ready', so it runs for the
# gate (test) as well as the full run.
echo "--- energy sources check ---"
python3 host/measure_power.py --check \
    ${RSEI_TASMOTA_IP:+--tasmota-ip "$RSEI_TASMOTA_IP"} \
    || { echo "!! energy check failed: no measurement source. Stopping."; exit 1; }

# Stop here if only the environment gate was requested.
if [ -n "${RSEI_TEST_ONLY:-}" ]; then
    echo "=== RSEI_TEST_ONLY: environment gate passed; stopping before measurement ==="
    exit 0
fi

# --- sampler helpers ---------------------------------------------------------
# NOTE: this script should be run as root (see header) so measure_power.py can
# read RAPL. The sampler inherits that privilege here - no per-call sudo (which
# would prompt for a password inside a backgrounded process).
start_power() {   # $1=label  $2=csv
    # Redirect stdout/stderr to /dev/null: the sampler runs until killed, so if it
    # kept the command-substitution pipe open, PID=$(start_power ...) would block
    # forever waiting for EOF instead of returning the PID. Its own file log
    # (logs/, via get_logger) is unaffected, and the CSV is flushed per row.
    python3 host/measure_power.py --out "$2" --label "$1" \
        ${RSEI_TASMOTA_IP:+--tasmota-ip "$RSEI_TASMOTA_IP"} >/dev/null 2>&1 &
    echo $!
}
stop_power() { kill -TERM "$1" 2>/dev/null || true; wait "$1" 2>/dev/null || true; }

# --- 4. IDLE baseline --------------------------------------------------------
echo "--- idle baseline (${RSEI_IDLE_SECS}s) ---"
PID=$(start_power "idle" "${RUN_DIR}/power_idle.csv")
sleep "$RSEI_IDLE_SECS"
stop_power "$PID"

# --- 5. Benchmarks per precision --------------------------------------------
# No --model: 03 defaults it to input/<model_id>/<precision>/model.xml.
for P in $RSEI_PRECISIONS; do
    echo "--- benchmark precision=$P ---"
    PID=$(start_power "bench_${P}" "${RUN_DIR}/power_${P}.csv")
    docker run "${DOCKER_COMMON[@]}" "${IMAGE}" \
        benchmark --precision "$P" || true
    stop_power "$PID"
done

# --- 6. Merge latency + energy into the per-run summary ----------------------
echo "--- merge stats ---"
python3 host/04_merge_stats.py --run-dir "$RUN_DIR" || echo "!! 04_merge_stats failed; raw files remain in ${RUN_DIR}/"

echo "=== done. Results, energy CSVs and summary in ${RUN_DIR}/ ==="
