#!/usr/bin/env bash
# =============================================================================
#  entrypoint.sh  -  dispatches to each pipeline step INSIDE the container.
#  Usage:  docker run ... <image> <command> [args]
#  Commands:
#     test        -> 00_test.py   (environment gate: AVX2, OpenVINO, iGPU)
#     benchmark   -> 03_infer_benchmark.py  (the timing step)
#     shell       -> interactive bash (debug)
#  (export/quantize are DEV-box steps run in the dev image; the lean image has
#   no torch/nncf, so they are intentionally not dispatched here.)
# =============================================================================
set -euo pipefail

# The target was decided and baked at build time; export it for the pipeline.
# A caller may still override by passing -e RSEI_TARGET, but the default is the
# immutable value chosen inside the image.
if [ -z "${RSEI_TARGET:-}" ] && [ -f /opt/rsei_target ]; then
    RSEI_TARGET="$(cat /opt/rsei_target)"
fi
export RSEI_TARGET="${RSEI_TARGET:-unknown}"
echo "[entrypoint] RSEI_TARGET=${RSEI_TARGET}"

cmd="${1:-test}"
shift || true

case "$cmd" in
  test)      exec python3 /work/pipeline/00_test.py "$@" ;;
  benchmark) exec python3 /work/pipeline/03_infer_benchmark.py "$@" ;;
  shell)     exec bash ;;
  *) echo "[entrypoint] unknown command: $cmd"; exit 2 ;;
esac
