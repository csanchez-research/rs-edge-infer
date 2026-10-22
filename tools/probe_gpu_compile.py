#!/usr/bin/env python3
# =============================================================================
#  probe_gpu_compile.py  -  minimal, faithful iGPU compile reproducer
# -----------------------------------------------------------------------------
#  Runs INSIDE a probe image (tools/Dockerfile.gpu-probe) that has a specific
#  Intel GPU OpenCL runtime (NEO / intel-opencl-icd) installed. It:
#    1. enumerates the OpenVINO devices,
#    2. records the OpenCL device/driver/OpenCL-C version (via clinfo),
#    3. COMPILES the real IR on the GPU for EVERY precision we actually benchmark
#       (fp32, fp16, int8) -- not just fp32. int8 carries extra quant/dequant
#       kernels, so a runtime can compile fp32 yet fail int8; "the iGPU works"
#       only holds if all the precisions we report compile.
#
#  Mount the model dir at /ir (with fp32/, fp16/, int8/ subdirs, each a model.xml
#  + model.bin), or pass its path as argv[1]. A flat /ir/model.xml is accepted as
#  a single-precision fallback.
#
#  Output is line-oriented and clean so the sweep script can parse it:
#    devices: [...]
#    installed: intel-opencl-icd <ver>
#    opencl: device=... | ocl_ver=... | driver=... | oclc=...
#    driver: <driver version>
#    oclc: <OpenCL C version>
#    gpu: <full device name>
#    precisions: fp32=OK fp16=OK int8=FAIL
#    RESULT: PASS | FAIL | NO_GPU | NO_IR
#  Exit code mirrors RESULT (0 PASS = all compiled, 1 FAIL, 2 NO_GPU, 3 NO_IR).
# =============================================================================
import os
import shutil
import subprocess
import sys

import openvino as ov

_FIELDS = {
    "Device Name": "device",
    "Device Version": "ocl_ver",
    "Driver Version": "driver",
    "Device OpenCL C Version": "oclc",
}


def clinfo_fields():
    exe = shutil.which("clinfo")
    if not exe:
        return {}
    try:
        out = subprocess.run(
            [exe], capture_output=True, text=True, timeout=20, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    found = {}
    for raw in out.splitlines():
        line = raw.strip()
        for key, label in _FIELDS.items():
            if line.startswith(key):
                value = line[len(key):].strip()
                if value:
                    found[label] = value
    return found


def installed_icd():
    try:
        out = subprocess.run(
            ["dpkg-query", "-W", "-f=${Package} ${Version}\n",
             "intel-opencl-icd", "intel-opencl-icd-legacy1"],
            capture_output=True, text=True, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1]:
            return line.strip()
    return ""


def compile_one(core, ir):
    """Compile one IR on the GPU; return (OK|FAIL, first_error_line)."""
    try:
        model = core.read_model(ir)
        core.compile_model(model, "GPU")
    except Exception as e:
        msg = str(e).strip().splitlines()
        return "FAIL", (msg[0] if msg else type(e).__name__)
    return "OK", ""


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/ir"
    core = ov.Core()
    print("devices:", core.available_devices)

    f = clinfo_fields()
    icd = installed_icd()
    print("installed:", icd or "none")
    print("opencl:", " | ".join(f"{k}={v}" for k, v in f.items()) or "no OpenCL device")
    print("driver:", f.get("driver") or icd or "n/a")
    print("oclc:", f.get("oclc") or "n/a")

    if not any(d.startswith("GPU") for d in core.available_devices):
        print("RESULT: NO_GPU")
        return 2

    try:
        print("gpu:", core.get_property("GPU", "FULL_DEVICE_NAME"))
    except Exception as e:  # property set varies by plugin version
        print("gpu: (unavailable:", type(e).__name__, ")")

    # Compile each precision we benchmark. Prefer per-precision subdirs; fall back
    # to a flat model.xml as a single precision.
    precs, details = [], []
    for p in ("fp32", "fp16", "int8"):
        ir = os.path.join(root, p, "model.xml")
        if os.path.exists(ir):
            status, err = compile_one(core, ir)
            precs.append(f"{p}={status}")
            if err:
                details.append(f"  {p}: {err}")
    if not precs:
        ir = os.path.join(root, "model.xml")
        if not os.path.exists(ir):
            print(f"RESULT: NO_IR (no model.xml under {root})")
            return 3
        status, err = compile_one(core, ir)
        precs.append(f"model={status}")
        if err:
            details.append(f"  model: {err}")

    print("precisions:", " ".join(precs))
    for d in details[:8]:
        print(d)

    if all(x.endswith("=OK") for x in precs):
        print(f"RESULT: PASS (GPU compiled: {' '.join(precs)})")
        return 0
    print(f"RESULT: FAIL ({' '.join(precs)})")
    return 1


if __name__ == "__main__":
    sys.exit(main())