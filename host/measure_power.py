#!/usr/bin/env python3
# =============================================================================
#  measure_power.py  -  energy sampler (runs on the HOST, NOT in Docker)
# -----------------------------------------------------------------------------
#  Runs in the background WHILE the container does inference. Each interval it
#  logs one row with:
#     - HOST UTC timestamp   (THE single clock -> cross-refs the benchmark JSON)
#     - RAPL package energy  (energy_uj counter, microjoules)
#     - Tasmota wall-plug power/energy (via local HTTP)
#
#  Why direct RAPL and not CodeCarbon: CodeCarbon reads this same counter under
#  the hood, but is designed to WRAP code in the same process. Here the measured
#  code lives in another process (the container), so sampling RAPL + Tasmota
#  from the host and cross-referencing by timestamp is cleaner and measures the
#  same thing. Both sources carry the host clock.
#
#  RAPL is cumulative and WRAPS at its max: window energy is the difference
#  between two readings (04_merge_stats handles the wrap via max_energy_range_uj).
#
#  Usage:
#     python3 measure_power.py --out output/power_<run>.csv --label benchmark \
#         [--tasmota-ip 192.168.1.50] [--interval 0.5]
#     python3 measure_power.py --check        # self-test the energy sources
#  Stop with Ctrl-C or SIGTERM (the orchestrator does this).
# =============================================================================
import argparse
import csv
import glob
import json
import os
import signal
import sys
import time
import urllib.request
from datetime import UTC, datetime

# Make repo-root config and pipeline logsetup importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
try:
    import config
except Exception:  # config is optional for this standalone tool
    config = None
from logsetup import get_logger  # noqa: E402  (import after sys.path setup)

_RUN = True


def _stop(signum, frame):
    global _RUN
    _RUN = False


signal.signal(signal.SIGINT, _stop)
signal.signal(signal.SIGTERM, _stop)


def find_rapl_package():
    """Return (energy_uj_path, max_range_path) for domain 'package-0', or None."""
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        try:
            with open(os.path.join(d, "name")) as f:
                if f.read().strip().startswith("package"):
                    return (os.path.join(d, "energy_uj"), os.path.join(d, "max_energy_range_uj"))
        except OSError:
            continue
    return None


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except OSError:
        return None


def poll_tasmota(ip, timeout=1.0):
    """Status 8 -> instantaneous power/energy. Returns dict, or {} on failure."""
    url = f"http://{ip}/cm?cmnd=Status%208"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        e = data.get("StatusSNS", {}).get("ENERGY", {})
        return {
            "plug_power_w": e.get("Power"),
            "plug_voltage_v": e.get("Voltage"),
            "plug_current_a": e.get("Current"),
            "plug_today_kwh": e.get("Today"),
            "plug_total_kwh": e.get("Total"),
            "plug_apparent_va": e.get("ApparentPower"),
            "plug_factor": e.get("Factor"),
        }
    except Exception:  # network/timeout: row without plug data, don't abort
        return {}


def run_check(tasmota_ip, log):
    """Self-test the energy sources with the SAME reader functions the sampler
    uses. Prints the status of each and returns an exit code:
      0 -> at least one source is live (RAPL and/or Tasmota)
      1 -> NO source available (a run would have no energy data -> fail fast)
    """
    log.info("energy-sources self-check")

    rapl = find_rapl_package()
    rapl_ok = False
    if rapl is None:
        log.warning("[warn] RAPL: no 'package' domain under /sys/class/powercap")
    else:
        val = read_int(rapl[0])
        if val is None:
            log.warning("[warn] RAPL: energy_uj present but NOT readable "
                        "(root needed; relaunch with sudo -E)")
        else:
            rapl_ok = True
            log.info("[OK]   RAPL package: energy_uj readable (%d uJ)", val)

    plug_ok = False
    if not tasmota_ip:
        log.warning("[warn] Tasmota: no IP configured (RSEI_TASMOTA_IP / config.TASMOTA_IP empty)")
    else:
        d = poll_tasmota(tasmota_ip, timeout=5.0)
        power = d.get("plug_power_w")
        if power is None:
            log.warning("[warn] Tasmota %s: no response / no ENERGY.Power "
                        "(check IP, network, or the Status endpoint)", tasmota_ip)
        else:
            plug_ok = True
            log.info("[OK]   Tasmota %s: ENERGY.Power = %s W", tasmota_ip, power)

    if rapl_ok or plug_ok:
        log.info("ENERGY SOURCES OK (rapl=%s plug=%s)", rapl_ok, plug_ok)
        return 0
    log.error("NO ENERGY SOURCE AVAILABLE (neither RAPL nor Tasmota). A measurement "
              "run would record no energy. Aborting.")
    return 1


def main():
    cfg_ip = config.TASMOTA_IP if config else ""
    cfg_interval = config.POWER_INTERVAL_S if config else 0.5

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasmota-ip", default=cfg_ip or None)
    ap.add_argument("--interval", type=float, default=cfg_interval)
    ap.add_argument("--out", help="CSV output path (required unless --check)")
    ap.add_argument("--label", default="", help="idle | benchmark | ...")
    ap.add_argument("--check", action="store_true",
                    help="self-test the energy sources (RAPL + Tasmota) and exit")
    args = ap.parse_args()

    # --- self-check mode: verify the measurement sources, then exit ----------
    if args.check:
        log = get_logger("measure_power_check")
        sys.exit(run_check(args.tasmota_ip, log))

    if not args.out:
        ap.error("--out is required unless --check is given")

    log = get_logger(f"measure_power_{args.label or 'run'}")

    rapl = find_rapl_package()
    if rapl is None:
        log.warning("RAPL not found; only the plug will be logged.")
    energy_path, maxrange_path = rapl if rapl else (None, None)
    max_range = read_int(maxrange_path) if maxrange_path else None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = [
        "ts_utc",
        "label",
        "rapl_energy_uj",
        "rapl_max_range_uj",
        "plug_power_w",
        "plug_voltage_v",
        "plug_current_a",
        "plug_today_kwh",
        "plug_total_kwh",
        "plug_apparent_va",
        "plug_factor",
    ]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        log.info("sampling every %ss -> %s (label=%s)", args.interval, args.out, args.label)
        while _RUN:
            row = {
                "ts_utc": datetime.now(UTC).isoformat(),
                "label": args.label,
                "rapl_energy_uj": read_int(energy_path) if energy_path else None,
                "rapl_max_range_uj": max_range,
            }
            if args.tasmota_ip:
                row.update(poll_tasmota(args.tasmota_ip))
            w.writerow(row)
            f.flush()
            time.sleep(args.interval)
    log.info("stopped. Rows written to %s", args.out)


if __name__ == "__main__":
    main()
