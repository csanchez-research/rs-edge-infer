#!/usr/bin/env python3
# =============================================================================
#  00_test.py  -  ENVIRONMENT GATE (runs INSIDE the container)
# -----------------------------------------------------------------------------
#  This is not a routine check: it decides whether the rest of the experiment
#  is viable on THIS machine. It verifies, in order and WITHOUT aborting on the
#  first failure (so it produces a complete report):
#
#    1. Real CPU flags (avx2 / avx_vnni / f16c / sse4_2).
#    2. That the RSEI_TARGET baked into the image matches the hardware (if you
#       built 'avx2' and run it on the N5105, raise a hard warning).
#    3. That OpenVINO IMPORTS and, critically, RUNS a minimal inference on CPU
#       without SIGILL (illegal instruction). This is the real test of the
#       pre-AVX2 problem: wheels may ship AVX2 and crash on the N5105.
#    4. The iGPU: that OpenVINO enumerates it, what OpenCL runtime/driver version
#       it exposes (recorded for provenance), and -- decisively -- that the model
#       actually COMPILES on it (enumeration alone can pass while kernel build
#       fails). Non-critical: the CPU axis stands alone.
#    5. Read access to RAPL (/sys/class/powercap) -- although energy is measured
#       from the HOST, checking here flags permission issues early.
#
#  Exit code: 0 if the CRITICAL checks pass, 1 if any critical check fails.
#  Writes a JSON report to the output volume and a log to the logs volume.
# =============================================================================
import os
import platform
import shutil
import socket
import subprocess
import sys
import traceback
from datetime import UTC, datetime

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jsonio
from logsetup import get_logger

import config

log = get_logger("00_test")

report = {
    "kind": "env_test",
    "hostname": socket.gethostname(),
    "utc": datetime.now(UTC).isoformat(),
    "rsei_target": config.TARGET,
    "checks": {},
    "critical_ok": True,
}


def record(name, ok, critical, detail=""):
    report["checks"][name] = {"ok": bool(ok), "critical": critical, "detail": str(detail)}
    if critical and not ok:
        report["critical_ok"] = False
    if ok:
        log.info("[OK]   %s: %s", name, detail)
    elif critical:
        log.error("[FAIL] %s: %s", name, detail)
    else:
        log.warning("[warn] %s: %s", name, detail)


def clinfo_summary():
    """Key OpenCL lines (device + driver + OpenCL C version) via clinfo.

    This is what pins down kernel-compile issues like the work_group one: the
    failing piece is the Intel GPU OpenCL runtime (NEO / intel-opencl-icd), not a
    pinned pip package, so its version belongs in the report for reproducibility.
    """
    exe = shutil.which("clinfo")
    if not exe:
        return "clinfo not installed"
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"clinfo failed: {type(e).__name__}"
    keys = ("Device Name", "Device Version", "Driver Version", "Device OpenCL C Version")
    lines = [ln.strip() for ln in out.splitlines() if any(k in ln for k in keys)]
    return " | ".join(dict.fromkeys(lines)) or "no OpenCL device lines"


log.info("00_test - environment gate")

# --- 1. CPU flags ------------------------------------------------------------
cpu_flags = set()
try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith(("flags", "Features")):
                cpu_flags = set(line.split(":", 1)[1].split())
                break
except OSError as e:
    record("read_cpuinfo", False, critical=True, detail=e)

has_avx2 = "avx2" in cpu_flags
has_vnni = "avx_vnni" in cpu_flags or "avx512_vnni" in cpu_flags
report["cpu"] = {
    "avx2": has_avx2,
    "avx_vnni": has_vnni,
    "f16c": "f16c" in cpu_flags,
    "sse4_2": "sse4_2" in cpu_flags,
    "model": platform.processor(),
}
record(
    "cpu_flags",
    True,
    critical=False,
    detail=f"avx2={has_avx2} vnni={has_vnni} f16c={'f16c' in cpu_flags}",
)

# --- 2. TARGET vs hardware coherence ----------------------------------------
target = report["rsei_target"]
coherent = not ((target == "avx2" and not has_avx2) or (target == "noavx2" and has_avx2))
record(
    "target_matches_hw",
    coherent,
    critical=True,
    detail=f"image={target}, cpu_avx2={has_avx2}"
    + ("" if coherent else "  <-- WRONG IMAGE FOR THIS CPU"),
)

# --- 3. OpenVINO import + minimal CPU inference (catches SIGILL) -------------
#  If the wheel carries instructions this CPU lacks, the process dies with
#  'Illegal instruction' HERE. Reaching the logged result proves OpenVINO
#  actually runs on this CPU.
ov_devices = []
model = None  # the synthetic smoke model; used as a GPU fallback if no IR is shipped
try:
    import numpy as np
    import openvino as ov

    report["openvino_version"] = ov.__version__
    core = ov.Core()
    ov_devices = core.available_devices
    record("openvino_import", True, critical=True, detail=f"v{ov.__version__}")
    record("openvino_devices", bool(ov_devices), critical=False, detail=str(ov_devices))

    # Minimal synthetic model: an element-wise add. Enough to force the exec
    # path and trigger a SIGILL if the wheel is incompatible with this CPU.
    # Prefer the flat namespace (OpenVINO 2024+); fall back to the deprecated
    # openvino.runtime only if this build does not expose it. If the first
    # import succeeds we never touch the deprecated path (no warning).
    try:
        import openvino.opset13 as ops
    except ImportError:
        import openvino.runtime.opset13 as ops

    p0 = ops.parameter([1, 8], ov.Type.f32, name="a")
    p1 = ops.parameter([1, 8], ov.Type.f32, name="b")
    model = ov.Model([ops.add(p0, p1)], [p0, p1], "smoke")
    compiled = core.compile_model(model, "CPU")
    a = np.ones((1, 8), dtype=np.float32)
    b = np.ones((1, 8), dtype=np.float32) * 2
    out = compiled([a, b])[compiled.output(0)]
    ok = bool(np.allclose(out, 3.0))
    record(
        "openvino_cpu_infer",
        ok,
        critical=True,
        detail=f"add 1+2 -> {out.ravel()[:3]} (no SIGILL = OpenVINO runs here)",
    )
except Exception as e:
    record("openvino_import_or_infer", False, critical=True, detail=f"{type(e).__name__}: {e}")
    report["openvino_traceback"] = traceback.format_exc()

# --- 4. iGPU: enumerate, record OpenCL version, and COMPILE-TEST -------------
#  Enumeration alone is NOT enough: OpenVINO can list the GPU while its OpenCL
#  runtime still fails to COMPILE the model's kernels (e.g. the LayerNorm kernel
#  needing OpenCL work_group functions this driver does not declare). So we also
#  record the driver/OpenCL version and actually compile on the GPU -- with the
#  real IR when present (it carries the kernels that break), else the synthetic
#  add. GPU checks are non-critical: the CPU axis stands on its own, and a broken
#  GPU is reported honestly here instead of discovered mid-run.
has_gpu = any(d.startswith("GPU") for d in ov_devices)
record(
    "igpu_available",
    has_gpu,
    critical=False,
    detail="iGPU seen by OpenVINO"
    if has_gpu
    else "iGPU NOT available (check intel-opencl-icd / --device /dev/dri)",
)

if has_gpu:
    gpu = {"opencl": clinfo_summary()}
    try:
        gpu["full_name"] = core.get_property("GPU", "FULL_DEVICE_NAME")
    except Exception as e:  # property set varies by plugin version
        gpu["full_name"] = f"(unavailable: {type(e).__name__})"
    report["gpu"] = gpu
    record("igpu_opencl", True, critical=False, detail=f"{gpu['full_name']} | {gpu['opencl']}")

    # Decisive smoke: compile ON the GPU. Prefer the real fp32 IR (has the kernels
    # that actually fail); fall back to the synthetic add model if no IR is shipped.
    try:
        ir = config.ir_xml("fp32")
        if os.path.exists(ir):
            gpu_model = core.read_model(ir)
            what = f"model IR ({os.path.basename(ir)})"
        else:
            gpu_model = model  # the synthetic add from the CPU smoke
            what = "synthetic add (no model IR present to compile-test)"
        core.compile_model(gpu_model, "GPU")
        record("igpu_compile", True, critical=False, detail=f"compiled {what} on GPU")
    except Exception as e:
        record(
            "igpu_compile",
            False,
            critical=False,
            detail=f"GPU compile FAILED: {type(e).__name__}: {str(e).strip()[:400]}",
        )

# --- 5. RAPL readable --------------------------------------------------------
rapl = "/sys/class/powercap/intel-rapl"
rapl_ok = os.path.isdir(rapl) and os.access(rapl, os.R_OK)
record(
    "rapl_readable",
    rapl_ok,
    critical=False,
    detail=f"{rapl} readable"
    if rapl_ok
    else f"{rapl} not accessible (wall-plug meter still works)",
)

# --- Report ------------------------------------------------------------------
os.makedirs(config.RUN_DIR, exist_ok=True)
out_path = os.path.join(config.RUN_DIR, "envtest.json")
with open(out_path, "w") as f:
    jsonio.dump(report, f)

log.info("report written to %s", out_path)
if report["critical_ok"]:
    log.info("CRITICAL: ALL OK")
else:
    log.error("CRITICAL: FAILURES PRESENT")
sys.exit(0 if report["critical_ok"] else 1)
