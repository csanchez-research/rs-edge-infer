#!/usr/bin/env python3
# =============================================================================
#  04_merge_stats.py  -  cross-reference latency (JSON) + energy (CSV) -> stats
# -----------------------------------------------------------------------------
#  Reads one run folder, output/runs/<RUN_ID>/ (--run-dir):
#     - bench_<precision>.json          (benchmark: latency + timestamps + meta)
#     - power_<precision>.csv           (energy during the benchmark)
#     - power_idle.csv                  (idle baseline to subtract)
#  Everything for a run shares one RUN_ID folder (the orchestrator injects
#  RSEI_RUN_ID so the container writes under the same id), so no cross-run
#  matching is needed - the folder IS the run.
#
#  For each (precision, device) it computes:
#     - latency (already in the JSON)
#     - mean wall-plug power during the device window (Tasmota)
#     - mean idle power (baseline)
#     - NET power = load - idle
#     - energy per inference (J/infer) = net_power * window_time / n_iter
#     - RAPL energy over the window (counter difference, wrap-aware) / n_iter
#
#  Cross-reference is by timestamp (device ts_start/ts_end in the JSON vs ts_utc
#  rows in the CSV); both come from the host clock, so the match is exact.
#
#  Output: <run-dir>/summary.csv  +  a printed table. The summary carries
#  the key metadata columns (target, channels, input_mode) for traceability.
# =============================================================================
import argparse
import csv
import glob
import itertools
import os
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
import jsonio  # noqa: E402  (import after sys.path setup)
from logsetup import get_logger  # noqa: E402  (import after sys.path setup)


def parse_ts(s):
    return datetime.fromisoformat(s)


def load_power_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                r["_ts"] = parse_ts(r["ts_utc"])
            except Exception:
                continue
            rows.append(r)
    return rows


def _floats(rows, key, lo=None, hi=None):
    out = []
    for r in rows:
        v = r.get(key)
        if v in (None, "", "None"):
            continue
        try:
            v = float(v)
        except ValueError:
            continue
        if lo is not None and r["_ts"] < lo:
            continue
        if hi is not None and r["_ts"] > hi:
            continue
        out.append(v)
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def rapl_energy_j(rows, lo, hi):
    """RAPL energy (J) over [lo,hi] by counter difference, wrap-aware."""
    win = [
        r
        for r in rows
        if lo <= r["_ts"] <= hi and r.get("rapl_energy_uj") not in (None, "", "None")
    ]
    if len(win) < 2:
        return None
    total_uj = 0.0
    for a, b in itertools.pairwise(win):
        ea, eb = int(float(a["rapl_energy_uj"])), int(float(b["rapl_energy_uj"]))
        d = eb - ea
        if d < 0:  # counter wrap
            mx = b.get("rapl_max_range_uj")
            if mx not in (None, "", "None"):
                d += int(float(mx))
            else:
                continue
        total_uj += d
    return total_uj / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        required=True,
        help="run folder output/runs/<RUN_ID>/ with power_<label>.csv + bench_<prec>.json",
    )
    args = ap.parse_args()

    run_dir = os.path.normpath(args.run_dir)
    run_id = os.path.basename(run_dir)
    log = get_logger(f"04_merge_{run_id}")

    idle_rows = load_power_csv(os.path.join(run_dir, "power_idle.csv"))
    idle_w = mean(_floats(idle_rows, "plug_power_w"))

    summary = []
    for jpath in sorted(glob.glob(os.path.join(run_dir, "bench_*.json"))):
        with open(jpath) as f:
            bench = jsonio.load(f)
        if bench.get("kind") != "benchmark":
            continue

        precision = bench["precision"]
        cfg = bench.get("config", {})
        meta = bench.get("metadata", {})
        power_rows = load_power_csv(os.path.join(run_dir, f"power_{precision}.csv"))

        for dev, d in bench.get("devices", {}).items():
            lat_agg = d.get("latency_ms_agg")
            if lat_agg is None:
                continue
            lo, hi = parse_ts(d["ts_start_utc"]), parse_ts(d["ts_end_utc"])
            load_w = mean(_floats(power_rows, "plug_power_w", lo, hi))
            net_w = (load_w - idle_w) if (load_w is not None and idle_w is not None) else None
            window_s = (hi - lo).total_seconds()
            # Energy is over the WHOLE device window = n_iter * repeats inferences.
            total_infers = d["n_iter"] * d.get("repeats", 1)
            j_plug = (net_w * window_s / total_infers) if net_w is not None else None
            rapl_j = rapl_energy_j(power_rows, lo, hi)
            j_rapl = (rapl_j / total_infers) if rapl_j is not None else None

            summary.append(
                {
                    "run_id": bench["run_id"],
                    "hostname": meta.get("hostname"),
                    "target": meta.get("rsei_target"),
                    "input_mode": cfg.get("input_mode"),
                    "total_channels": cfg.get("total_channels"),
                    "precision": precision,
                    "device": dev,
                    "repeats": d.get("repeats"),
                    "synthetic_input": bench.get("synthetic_input"),
                    "lat_mean_ms": lat_agg["mean"],
                    "lat_ci95_ms": lat_agg["ci95_half"],
                    "lat_std_ms": lat_agg["std"],
                    "throughput_ips": d["throughput_ips_agg"]["mean"],
                    "idle_w": round(idle_w, 3) if idle_w is not None else None,
                    "load_w": round(load_w, 3) if load_w is not None else None,
                    "net_w": round(net_w, 3) if net_w is not None else None,
                    "J_per_infer_plug": round(j_plug, 5) if j_plug is not None else None,
                    "J_per_infer_rapl": round(j_rapl, 5) if j_rapl is not None else None,
                }
            )

    if not summary:
        log.warning("No benchmarks found in %s (expected bench_*.json).", run_dir)
        return

    cols = list(summary[0].keys())
    out_csv = os.path.join(run_dir, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary)

    log.info("Summary -> %s", out_csv)
    hdr = (
        f"{'prec':<5} {'dev':<4} {'lat_ms(mean±ci95)':>20} "
        f"{'net_W':>7} {'J/inf(plug)':>12} {'J/inf(rapl)':>12}"
    )
    log.info(hdr)
    log.info("-" * len(hdr))
    for s in summary:
        ci = s["lat_ci95_ms"]
        lat = f"{s['lat_mean_ms']}±{ci}" if ci is not None else f"{s['lat_mean_ms']}"
        log.info(
            "%-5s %-4s %20s %7s %12s %12s",
            s["precision"],
            s["device"],
            lat,
            s["net_w"],
            s["J_per_infer_plug"],
            s["J_per_infer_rapl"],
        )
    if any(s["synthetic_input"] for s in summary):
        log.warning("some runs used SYNTHETIC input -> latency/energy valid, task accuracy NOT.")


if __name__ == "__main__":
    main()
