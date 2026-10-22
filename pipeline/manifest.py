#!/usr/bin/env python3
# =============================================================================
#  manifest.py  -  per-model provenance manifest (chain of custody)
# -----------------------------------------------------------------------------
#  A single human-readable JSON per model, state/<model_id>/manifest.json, that
#  records the SHA256 of every artifact as each pipeline step produces it:
#     checkpoint -> onnx -> IR (xml+bin) per precision.
#  Each step calls record(); the manifest is read, the entry added/updated, and
#  written back. This is the chain of custody: given a benchmark result, you can
#  trace the exact bytes of every upstream artifact that produced it.
#
#  Lives under state/, the local source of truth: gitignored like input/ (it is
#  data), but persisted across the whole project rather than wiped between runs.
#  Artifact paths are stored relative to INPUT_DIR so they stay portable.
# =============================================================================
import os
from datetime import UTC, datetime

import jsonio
import metadata

import config


def manifest_path():
    return os.path.join(config.MODEL_STATE_DIR, "manifest.json")


def _rel(path):
    """Path relative to INPUT_DIR (portable), falling back to the given path."""
    try:
        return os.path.relpath(path, config.INPUT_DIR)
    except ValueError:
        return path


def record(key, path, model_id=None):
    """Add/update one artifact's SHA256 under `key` in the model manifest.

    Returns the manifest path. Safe to call repeatedly; re-recording a key
    overwrites it (e.g. re-exporting the ONNX updates its hash).
    """
    model_id = model_id or config.MODEL_ID
    os.makedirs(config.MODEL_STATE_DIR, exist_ok=True)
    mpath = manifest_path()

    try:
        with open(mpath) as f:
            man = jsonio.load(f)
    except (OSError, ValueError):
        man = {
            "model_id": model_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "artifacts": {},
        }

    man.setdefault("artifacts", {})
    man["artifacts"][key] = {
        "file": _rel(path),
        "sha256": metadata.sha256_file(path),
        "bytes": os.path.getsize(path) if os.path.exists(path) else None,
        "recorded_utc": datetime.now(UTC).isoformat(),
    }
    man["updated_utc"] = datetime.now(UTC).isoformat()

    with open(mpath, "w") as f:
        jsonio.dump(man, f)
    return mpath
