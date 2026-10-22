#!/usr/bin/env python3
# =============================================================================
#  metadata.py  -  collect as much reproducibility metadata as cheaply possible
# -----------------------------------------------------------------------------
#  Embedded into every benchmark result. The richer the metadata, the easier it
#  is for a reviewer (or future-you) to trust and reproduce the numbers. Every
#  probe is best-effort: if something is unreadable it is recorded as None, the
#  collection never fails.
#
#  Note: inside the container /proc and /sys reflect the HOST kernel, so CPU
#  model, flags, governor and thermals are the real machine's, which is exactly
#  what we want to record.
# =============================================================================
import glob
import hashlib
import platform
import socket
from datetime import UTC, datetime

import config


def sha256_file(path, chunk=1 << 20):
    """SHA256 of a file, or None if it cannot be read. For artifact traceability."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _first_line_field(path, prefix):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def cpu_info():
    flags = set()
    model = None
    cores = 0
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and model is None:
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("processor"):
                    cores += 1
                elif line.startswith(("flags", "Features")) and not flags:
                    flags = set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return {
        "model": model,
        "logical_cores": cores or None,
        "avx2": "avx2" in flags,
        "avx_vnni": ("avx_vnni" in flags or "avx512_vnni" in flags),
        "f16c": "f16c" in flags,
        "sse4_2": "sse4_2" in flags,
    }


def cpu_governor():
    govs = set()
    for p in glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"):
        g = _read(p)
        if g:
            govs.add(g)
    return sorted(govs) or None


def mem_total_kb():
    v = _first_line_field("/proc/meminfo", "MemTotal")
    if v:
        try:
            return int(v.split()[0])
        except (ValueError, IndexError):
            return None
    return None


def thermal_c():
    """Max temperature across thermal zones, in Celsius (best effort)."""
    temps = []
    for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        v = _read(p)
        if v and v.lstrip("-").isdigit():
            temps.append(int(v) / 1000.0)
    return max(temps) if temps else None


def openvino_version():
    try:
        import openvino as ov

        return ov.__version__
    except Exception:
        return None


def collect():
    """Full metadata snapshot (safe to call anywhere)."""
    return {
        "collected_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "rsei_target": config.TARGET,
        "kernel": platform.release(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_info(),
        "cpu_governor": cpu_governor(),
        "mem_total_kb": mem_total_kb(),
        "thermal_c_at_start": thermal_c(),
        "openvino_version": openvino_version(),
    }
