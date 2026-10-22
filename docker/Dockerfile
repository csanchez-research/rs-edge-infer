# =============================================================================
#  Self-detecting Dockerfile  -  rs-edge-infer
# -----------------------------------------------------------------------------
#  ONE image. At BUILD time it tests the CPU and installs the matching profile,
#  baking an immutable target that the whole pipeline reads:
#    AVX2 present  -> avx2   profile  (Lenovo M720q/M920q, Coffee Lake)
#    AVX2 absent   -> noavx2 profile  (N5105, Jasper Lake / Tremont) -> OpenVINO 2024.6
#
#  The test runs INSIDE the build (reads /proc/cpuinfo), installs
#  requirements-<target>.txt, and writes the chosen target to /opt/rsei_target.
#  The entrypoint exports it as RSEI_TARGET, so it is fixed (immutable) in the
#  image - no external script or --build-arg decides it.
#
#  IMPORTANT: detection reflects the BUILD machine's CPU, so build the image ON
#  the machine you will run it on (docker build on the N5105 -> noavx2; on a
#  Lenovo -> avx2). 00_test re-checks at run time and fails loudly on a mismatch.
#  To force a profile (e.g. cross-building), pass --build-arg FORCE_TARGET=noavx2.
#
#  NOTE: the container is Ubuntu (OpenVINO's officially supported base) even
#  though the HOST is Debian Trixie. The host provides the kernel + the i915
#  iGPU driver; the container provides OpenVINO. Different layers - that is why
#  Debian on the host is fine.
#
#  PYTHON: base is ubuntu:24.04 (Python 3.12). The project targets Python 3.11+
#  (ruff target-version = py311; the dev image is python:3.11-slim), so the
#  pipeline uses the 3.11+ idioms (e.g. datetime.UTC). 3.12 >= 3.11, so the same
#  code runs here unchanged - the whole project speaks one Python dialect.
# =============================================================================

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# pip network tolerance as CONFIG (applies to every pip call in the build):
# long read timeout + many retries so a slow/flaky link does not abort.
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=20

# Empty = auto-detect from the build host's CPU. Set to "avx2"/"noavx2" to force.
ARG FORCE_TARGET=

# pip source config. PIP_INDEX_URL = primary (your Nexus); PIP_EXTRA_INDEX_URL =
# fallback(s) for anything the primary does not serve (public PyPI, etc.). All
# empty by default -> plain public PyPI. Set them (nexus.env) for Nexus-first.
ARG PIP_INDEX_URL=
ARG PIP_EXTRA_INDEX_URL=
ARG PIP_TRUSTED_HOST=

# --- System dependencies -----------------------------------------------------
#  ocl-icd-libopencl1 + the Intel compute runtime (installed below) are needed
#  for OpenVINO to see the iGPU via /dev/dri (passed with --device at run time).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        ca-certificates \
        ocl-icd-libopencl1 clinfo \
    && rm -rf /var/lib/apt/lists/*

# --- Intel GPU runtime (NEO / compute-runtime) -------------------------------
#  Required for OpenVINO's GPU plugin on the Jasper Lake iGPU (UHD, Gen11).
#  The stock intel-opencl-icd from the Ubuntu repos is enough (no "legacy"
#  runtime needed). 00_test re-checks iGPU enumeration at run time on the real
#  target, so a base change is caught there rather than slipping through.
RUN apt-get update && apt-get install -y --no-install-recommends \
        intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/* || echo "WARN: intel-opencl-icd not installed, GPU may be unavailable"

# --- Self-detect target + install matching OpenVINO stack --------------------
#  The test: if the build CPU advertises avx2 -> avx2 profile, else noavx2.
#  Both requirements files are copied; only the selected one is installed. The
#  decision is written to /opt/rsei_target (immutable in the image).
#
#  Hypothesis (to be confirmed empirically with 00_test on the real N5105):
#   - OpenVINO is pinned to 2024.6 (see requirements-*.txt): it supports pre-AVX2
#     CPUs (2026.0+ requires AVX2) AND compiles the Gen11 iGPU kernel, which the
#     2025.x series regressed. Bisected by the gpu-probe sweep.
#   - pip wheels may still be AVX2-compiled; if so, OpenVINO crashes with SIGILL
#     on the N5105. 00_test catches that, not this build step.
# Network resilience: timeout/retries come from the ENV above, and a BuildKit
# pip CACHE MOUNT lets a retried build reuse already-downloaded wheels.
COPY docker/requirements-noavx2.txt docker/requirements-avx2.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    if [ -n "${FORCE_TARGET}" ]; then \
        TARGET="${FORCE_TARGET}"; \
    elif grep -qw avx2 /proc/cpuinfo; then \
        TARGET="avx2"; \
    else \
        TARGET="noavx2"; \
    fi; \
    echo "[build] detected target: ${TARGET}"; \
    echo "${TARGET}" > /opt/rsei_target; \
    pip3 install --break-system-packages \
        ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"} \
        ${PIP_EXTRA_INDEX_URL:+--extra-index-url "$PIP_EXTRA_INDEX_URL"} \
        ${PIP_TRUSTED_HOST:+--trusted-host "$PIP_TRUSTED_HOST"} \
        -r "/tmp/requirements-${TARGET}.txt"

WORKDIR /work
COPY config.py /work/config.py
COPY pipeline/ /work/pipeline/
COPY docker/entrypoint.sh /work/entrypoint.sh
RUN chmod +x /work/entrypoint.sh

# /work/input  -> encoder + data (mounted read-only volume)
# /work/output -> results        (mounted read-write volume)
ENTRYPOINT ["/work/entrypoint.sh"]