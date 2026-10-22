#!/usr/bin/env python3
# =============================================================================
#  00_verify.py  -  PROVENANCE GATE (integrity + chain of custody)
# -----------------------------------------------------------------------------
#  Verifies the artifacts against the provenance manifest
#  (state/<model_id>/manifest.json). Two independent checks, neither aborts on
#  the first failure (so you get a complete report):
#
#    A. INTEGRITY (always): for every artifact recorded in the manifest,
#       recompute its SHA256 on disk and compare to the recorded value.
#         - present + hash MATCH    -> OK
#         - present + hash MISMATCH -> CRITICAL  (bytes changed: corruption /
#                                                  bad copy / tampering)
#         - absent                  -> warn      (expected on the target machine,
#                                                  which only receives the IR,
#                                                  not encoder.pt / onnx)
#
#    B. CHAIN OF CUSTODY (only if benchmark results exist in output/runs/):
#       each result records model_sha256 of the exact IR it measured. Compare
#       it to the manifest's ir_<precision>_xml / _bin entries.
#         - match    -> OK
#         - mismatch -> CRITICAL  (the result measured different bytes than the
#                                  ones the pipeline recorded)
#       With no results yet (e.g. running this first) B is simply skipped.
#
#  Runs anywhere; dev or target. On the target, copy state/<model_id>/ next to
#  input/<model_id>/ so this can verify the IR you received matches the dev build.
#
#  Exit code: 0 if the CRITICAL checks pass, 1 if any critical check fails.
#  Writes a JSON report to the output volume and a log to the logs volume.
# =============================================================================
import argparse
import glob
import os
import socket
import sys
from datetime import UTC, datetime

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jsonio
import manifest
import metadata
from logsetup import get_logger

import config

log = get_logger("00_verify")

report = {
    "kind": "provenance_verify",
    "hostname": socket.gethostname(),
    "utc": datetime.now(UTC).isoformat(),
    "model_id": config.MODEL_ID,
    "manifest": manifest.manifest_path(),
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


def load_manifest():
    mpath = manifest.manifest_path()
    try:
        with open(mpath) as f:
            return jsonio.load(f)
    except (OSError, ValueError) as e:
        record("manifest_present", False, critical=True, detail=f"cannot read {mpath}: {e}")
        return None


def abs_artifact(entry):
    """Manifest stores paths relative to INPUT_DIR; resolve to an absolute path."""
    return os.path.join(config.INPUT_DIR, entry["file"])


# --- A. Integrity: on-disk bytes vs recorded SHA256 --------------------------
def check_integrity(man):
    artifacts = man.get("artifacts", {})
    if not artifacts:
        record("integrity", False, critical=True, detail="manifest has no artifacts")
        return {}
    ir_hashes = {}  # key -> recorded sha256, for the chain check later
    for key, entry in sorted(artifacts.items()):
        recorded = entry.get("sha256")
        ir_hashes[key] = recorded
        path = abs_artifact(entry)
        if not os.path.exists(path):
            record(
                f"integrity:{key}",
                True,  # absence is not a failure by itself
                critical=False,
                detail=f"recorded but NOT present here ({entry['file']}) "
                "- expected if this is the target machine",
            )
            continue
        actual = metadata.sha256_file(path)
        match = actual == recorded
        record(
            f"integrity:{key}",
            match,
            critical=True,
            detail=f"{entry['file']}  recorded={_short(recorded)} disk={_short(actual)}"
            + ("" if match else "  <-- BYTES CHANGED"),
        )
    return ir_hashes


# --- B. Chain of custody: benchmark results vs manifest IR hashes ------------
def check_chain(ir_hashes):
    # New layout: output/runs/<run_id>/bench_<prec>.json. Also accept the old flat
    # output/runs/<id>__<prec>.json so verification still works on pre-refactor data.
    result_files = sorted(
        glob.glob(os.path.join(config.RUNS_DIR, "*", "bench_*.json"))
        + glob.glob(os.path.join(config.RUNS_DIR, "*.json"))
    )
    results = []
    for rf in result_files:
        try:
            with open(rf) as f:
                r = jsonio.load(f)
            if r.get("kind") == "benchmark":
                results.append((rf, r))
        except (OSError, ValueError):
            continue
    if not results:
        record(
            "chain_of_custody",
            True,
            critical=False,
            detail="no benchmark results yet - skipped (run 03 first to close the chain)",
        )
        return
    for rf, r in results:
        p = r.get("precision", "?")
        got = r.get("model_sha256", {})
        for part in ("xml", "bin"):
            key = f"ir_{p}_{part}"
            recorded = ir_hashes.get(key)
            measured = got.get(part)
            if recorded is None:
                record(
                    f"chain:{os.path.basename(rf)}:{part}",
                    True,
                    critical=False,
                    detail=f"result precision '{p}' has no matching manifest key '{key}'",
                )
                continue
            match = recorded == measured
            record(
                f"chain:{os.path.basename(rf)}:{part}",
                match,
                critical=True,
                detail=f"{key}  manifest={_short(recorded)} result={_short(measured)}"
                + ("" if match else "  <-- RESULT MEASURED OTHER BYTES"),
            )


def _short(h):
    return (h[:12] + "...") if isinstance(h, str) and len(h) > 15 else h


def main():
    ap = argparse.ArgumentParser(description="Verify artifacts against the provenance manifest.")
    ap.add_argument("--no-chain", action="store_true", help="skip the result/chain check")
    args = ap.parse_args()

    log.info("00_verify - provenance gate (model_id=%s)", config.MODEL_ID)
    man = load_manifest()
    if man is None:
        _write_report()
        sys.exit(1)
    record("manifest_present", True, critical=True, detail=report["manifest"])

    ir_hashes = check_integrity(man)
    if not args.no_chain:
        check_chain(ir_hashes)

    _write_report()
    if report["critical_ok"]:
        log.info("CRITICAL: ALL OK")
    else:
        log.error("CRITICAL: FAILURES PRESENT")
    sys.exit(0 if report["critical_ok"] else 1)


def _write_report():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(
        config.OUTPUT_DIR, f"verify_{report['hostname']}_{config.MODEL_ID}_{stamp}.json"
    )
    with open(out_path, "w") as f:
        jsonio.dump(report, f)
    log.info("report written to %s", out_path)


if __name__ == "__main__":
    main()
