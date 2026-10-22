#!/usr/bin/env python3
# =============================================================================
#  prepare_data.py  -  one command: download reBEN, carve the slice we use, pack
# -----------------------------------------------------------------------------
#  SELF-CONTAINED data-prep for THIS project. It does not import the eval harness
#  (separate project); it only CONSUMES the harness's products through input/:
#  the encoder (01_export) and norm_stats.json (here). Everything it makes lands
#  under input/ too, which is the boundary.
#
#  one download, everything in one pass: `--stage all` (default), in order:
#    1. download - fetch pinned reBEN v2.0 (Zenodo 10891137) with aria2c
#                  (segmented, resumable, md5-verified). Idempotent.
#    2. select   - read metadata.parquet, per role pick a SEEDED subset
#                  (df[df.split==s].sample(n, random_state=seed)); 'calib' from
#                  'validation', 'train'/'test' from theirs (disjoint).
#    3. extract  - stream-filter the .tar.zst in one pass, routing each selected
#                  patch into tree/<role>/ (calib/train/test physically split),
#                  each with its own metadata.parquet. Never materialises ~300 GB.
#    4. pack     - apply the preprocessing (faithful copy of the harness
#                  reben_loader transform) and write calib_int8.npz (normalised
#                  6x224x224 float32) to config.CALIB_PACK for INT8 PTQ.
#    5. cleanup  - delete the 117 GB archives and the calib tree (now in the npz);
#                  keep train/ + test/ (for the probe) and the calib npz. Skip
#                  with --keep-archives.
#
#  Provenance: state/data/data_manifest.json records the Zenodo record, source
#  MD5s (expected + seen), parquet sha256, seed, per-role counts, and the sha256
#  of the calib pack. Reproducible by {record + parquet + seed + N}.
#
#  Runs INSIDE the dev container (deps baked from docker/requirements-dev.txt),
#  like 01_export and 02_quantize - not with a loose pip on the host. Downloads
#  use aria2c (segmented + resumable + md5-verified), baked into the dev image.
# =============================================================================
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logsetup import get_logger

import config

log = get_logger("prepare_data")

# Role -> reBEN split it is drawn from, and default N. calib feeds INT8 PTQ.
ROLE_SPLIT = {"calib": "validation", "train": "train", "test": "test"}
DEFAULT_N = {"calib": 512, "train": 20000, "test": 10000}


# Dataset: reBEN / BigEarthNet v2.0 [Clasen et al. 2024, arXiv:2407.03653],
# distributed on Zenodo record 10891137 (v2.0.0).
ZENODO_RECORD = "10891137"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
ARCHIVE_MD5 = {  # pinned from the Zenodo record, for integrity
    "BigEarthNet-S2.tar.zst": "2245ed2d1a93f6ce637d839bc856396e",
    "BigEarthNet-S1.tar.zst": "a55eaa2cdf6a917e296bd6601ec1e348",
    "metadata.parquet": "55687065e77b6d0b0f1ff604a6e7b49c",
}
# S2 bands the encoder uses (B04,B03,B02) and S1 (VV,VH). VV-VH is derived.
S2_BANDS = ("B04", "B03", "B02")
S1_BANDS = ("VV", "VH")
#  Channel order of the packed tensor, == pre-training / probe order.
BAND_ORDER = ["S2_B4", "S2_B3", "S2_B2", "S1_VV", "S1_VH", "S1_VV_VH_DIFF"]


def _hash(path, algo, chunk=1 << 20):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _human_bytes(n):
    """Binary units (GiB = 1024^3), to report the actual size reclaimed."""
    unit = float(n)
    for u in ("B", "KiB", "MiB", "GiB"):
        if unit < 1024:
            return f"{unit:.1f} {u}"
        unit /= 1024
    return f"{unit:.1f} TiB"


def _dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(root, name))
    return total


def _data_dir():
    return os.path.join(config.STATE_DIR, "data")


def _manifest_path():
    return os.path.join(_data_dir(), "data_manifest.json")


def _load_manifest():
    try:
        with open(_manifest_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {
            "zenodo_record": ZENODO_RECORD,
            "archive_md5_expected": ARCHIVE_MD5,
            "created_utc": datetime.now(UTC).isoformat(),
        }


def _save_manifest(man):
    man["updated_utc"] = datetime.now(UTC).isoformat()
    os.makedirs(_data_dir(), exist_ok=True)
    with open(_manifest_path(), "w") as f:
        json.dump(man, f, indent=2)


def _subset_parquet():
    return os.path.join(_data_dir(), "metadata_subset.parquet")


# --- 1. DOWNLOAD -------------------------------------------------------------
def _ensure(name, archives):
    """Fetch one Zenodo file into archives/, segmented + resumable + verified.

    Uses aria2c: multiple connections (faster on the big archives), -c resume,
    auto-retry, and --checksum so a corrupt download FAILS instead of silently
    completing. Idempotent: on a complete, matching file aria2c just re-verifies
    and exits. Each stage fetches only what it needs, so `select` pulls just the
    3.6 MB parquet - never the 117 GB archives.
    """
    md5 = ARCHIVE_MD5[name]
    os.makedirs(archives, exist_ok=True)
    url = f"{ZENODO_BASE}/{name}?download=1"
    log.info("[download] %s <- %s", name, url)
    ret = subprocess.run(
        [
            "aria2c",
            "-c",
            "-x",
            "8",
            "-s",
            "8",
            "--max-tries=20",
            "--retry-wait=10",
            f"--checksum=md5={md5}",
            "-d",
            archives,
            "-o",
            name,
            url,
        ],
        check=False,
    ).returncode
    if ret != 0:
        raise SystemExit(f"[download] aria2c failed (code {ret}) for {name} (md5/network)")
    log.info("[download] %s OK (md5 verified)", name)
    man = _load_manifest()
    man.setdefault("archive_md5_seen", {})[name] = md5
    _save_manifest(man)
    return os.path.join(archives, name)


def stage_download(args):
    for name in ARCHIVE_MD5:
        _ensure(name, args.archives)


# --- 2. SELECT ---------------------------------------------------------------
def stage_select(args):
    import pandas as pd

    pq = args.parquet or os.path.join(args.archives, "metadata.parquet")
    if not args.parquet:
        _ensure("metadata.parquet", args.archives)  # only the 3.6 MB index
    df = pd.read_parquet(pq)
    man = _load_manifest()
    man["parquet_sha256"] = _hash(pq, "sha256")
    man["seed"] = args.seed
    man.setdefault("roles", {})

    pieces = []
    for role in args.roles:
        split = ROLE_SPLIT[role]
        n = getattr(args, f"n_{role}")
        sub = df[df["split"] == split]
        if n and n < len(sub):
            # SAME rule as the harness: fixed seed -> reproducible, representative.
            sub = sub.sample(n=n, random_state=args.seed)
        sub = sub.reset_index(drop=True).copy()
        sub["role"] = role
        pieces.append(sub)
        man["roles"][role] = {"split": split, "n_requested": n, "n_selected": len(sub)}
        log.info("[select] %-5s <- %-10s : %d patches", role, split, len(sub))

    subset = pd.concat(pieces, ignore_index=True)
    os.makedirs(_data_dir(), exist_ok=True)
    subset.to_parquet(_subset_parquet(), index=False)
    man["subset_parquet_sha256"] = _hash(_subset_parquet(), "sha256")
    _save_manifest(man)
    log.info("[select] wrote %s (%d rows)", _subset_parquet(), len(subset))


# --- 3. EXTRACT (stream-filter) ---------------------------------------------
def _s2_tile(patch_id):
    return patch_id.rsplit("_", 2)[0]


def _s1_tile(s1_name):
    return s1_name.rsplit("_", 3)[0]


def _role_map(subset):
    """basename -> role, so each band file is routed to its role's folder."""
    m = {}
    for _, r in subset.iterrows():
        for b in S2_BANDS:
            m[f"{r['patch_id']}_{b}.tif"] = r["role"]
        for b in S1_BANDS:
            m[f"{r['s1_name']}_{b}.tif"] = r["role"]
    return m


def _role_dir(tree, role):
    return os.path.join(tree, role)


def _stream_extract(archive, role_map, tree):
    import zstandard as zstd

    got = 0
    dctx = zstd.ZstdDecompressor()
    # one streaming pass over the whole tar (same cost whatever we keep); each
    # wanted file lands under tree/<role>/, so calib/train/test end up physically
    # separated. Streaming ('r|'): members in order, no seeking.
    with (
        open(archive, "rb") as fh,
        dctx.stream_reader(fh) as reader,
        tarfile.open(fileobj=reader, mode="r|") as tar,
    ):
        for m in tar:
            if not m.isfile():
                continue
            role = role_map.get(os.path.basename(m.name))
            if role:
                tar.extract(m, path=_role_dir(tree, role))
                got += 1
    return got


def stage_extract(args):
    import pandas as pd

    subset = pd.read_parquet(_subset_parquet())
    role_map = _role_map(subset)
    log.info(
        "[extract] %d band files for %d patches across %s",
        len(role_map),
        len(subset),
        sorted(subset["role"].unique()),
    )

    total = 0
    for arc in ("BigEarthNet-S2.tar.zst", "BigEarthNet-S1.tar.zst"):
        ap = os.path.join(args.archives, arc)
        if not os.path.exists(ap):
            _ensure(arc, args.archives)  # fetch+verify only if missing (the 117 GB)
        n = _stream_extract(ap, role_map, args.tree)
        log.info("[extract] %s: wrote %d files", arc, n)
        total += n

    # Per role: hard-gate completeness, and write that role's own parquet so the
    # probe can point at tree/<role>/ with tree/<role>/metadata.parquet.
    missing = []
    for role in sorted(subset["role"].unique()):
        rtree = _role_dir(args.tree, role)
        rows = subset[subset["role"] == role].reset_index(drop=True)
        for _, r in rows.iterrows():
            s2d = os.path.join(rtree, "BigEarthNet-S2", _s2_tile(r["patch_id"]), r["patch_id"])
            s1d = os.path.join(rtree, "BigEarthNet-S1", _s1_tile(r["s1_name"]), r["s1_name"])
            for b in S2_BANDS:
                if not os.path.exists(os.path.join(s2d, f"{r['patch_id']}_{b}.tif")):
                    missing.append(f"{role}:{r['patch_id']}_{b}")
            for b in S1_BANDS:
                if not os.path.exists(os.path.join(s1d, f"{r['s1_name']}_{b}.tif")):
                    missing.append(f"{role}:{r['s1_name']}_{b}")
        os.makedirs(rtree, exist_ok=True)
        rows.to_parquet(os.path.join(rtree, "metadata.parquet"), index=False)
    if missing:
        raise SystemExit(
            f"[extract] {len(missing)} band files missing after extraction "
            f"(e.g. {missing[:5]}). Archive/parquet mismatch - refusing to continue."
        )
    log.info("[extract] OK: %d files; roles separated under %s", total, args.tree)


# --- 4. PACK -----------------------------------------------------------------
#  Faithful copy of the published harness transform (strata-s12-eval-harness/
#  reben_loader.py, from Paper 1): read 6 bands -> derive
#  VV-VH -> stack -> bilinear resize to img_size -> per-band (x-mean)/std from
#  norm_stats. Kept here so this project is self-contained; it must stay in sync
#  with that file.
def _read_band(path):
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _load_6ch(tree, row):
    import numpy as np

    pid, s1 = row["patch_id"], row["s1_name"]
    s2d = os.path.join(tree, "BigEarthNet-S2", _s2_tile(pid), pid)
    s1d = os.path.join(tree, "BigEarthNet-S1", _s1_tile(s1), s1)
    b4 = _read_band(os.path.join(s2d, f"{pid}_B04.tif"))
    b3 = _read_band(os.path.join(s2d, f"{pid}_B03.tif"))
    b2 = _read_band(os.path.join(s2d, f"{pid}_B02.tif"))
    vv = _read_band(os.path.join(s1d, f"{s1}_VV.tif"))
    vh = _read_band(os.path.join(s1d, f"{s1}_VH.tif"))
    return np.stack([b4, b3, b2, vv, vh, vv - vh], axis=0)  # (6, H, W)


def _preprocess(tree, row, means, stds, img_size):
    """6-channel patch -> (6, img_size, img_size) float32, normalised.

    Exactly the probe/pack transform: bilinear resize to img_size, then per-band
    (x-mean)/std. Shared by the calib pack and the single benchmark sample.
    """
    import torch

    arr = _load_6ch(tree, row)  # (6, 120, 120)
    x = torch.from_numpy(arr).unsqueeze(0)
    x = torch.nn.functional.interpolate(
        x, size=(img_size, img_size), mode="bilinear", align_corners=False
    ).squeeze(0)
    return ((x - means) / stds).to(torch.float32).numpy()


def stage_pack(args):
    import numpy as np
    import pandas as pd
    import torch

    with open(args.stats) as f:
        s = json.load(f)
    means = torch.tensor(s["band_means"], dtype=torch.float32).view(-1, 1, 1)
    stds = torch.clamp(torch.tensor(s["band_stds"], dtype=torch.float32).view(-1, 1, 1), min=1e-6)

    subset = pd.read_parquet(_subset_parquet())
    calib = subset[subset["role"] == "calib"].reset_index(drop=True)
    if len(calib) == 0:
        raise SystemExit("[pack] no calib patches; run select+extract first.")

    calib_tree = _role_dir(args.tree, "calib")
    xs, pids = [], []
    for _, r in calib.iterrows():
        xs.append(_preprocess(calib_tree, r, means, stds, args.img_size))
        pids.append(str(r["patch_id"]))
    out_arr = np.stack(xs, axis=0)  # (N, 6, 224, 224)

    out = args.out or config.CALIB_PACK
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = {
        "role": "calib",
        "split": ROLE_SPLIT["calib"],
        "n": int(out_arr.shape[0]),
        "seed": args.seed,
        "img_size": args.img_size,
        "zenodo_record": ZENODO_RECORD,
        "band_order": BAND_ORDER,
    }
    np.savez_compressed(out, x=out_arr, patch_ids=np.array(pids), meta=json.dumps(meta))
    man = _load_manifest()
    man.setdefault("packs", {})
    man["packs"]["calib_int8"] = {
        "path": out,
        "sha256": _hash(out, "sha256"),
        "shape": list(out_arr.shape),
        **meta,
    }

    # Benchmark sample: one real test-split patch, preprocessed identically, so
    # the LEAN target (no rasterio/torch) can feed 03 a REAL sample instead of
    # noise. Tiny and portable; ships beside the IR. Best-effort: if 'test' was
    # not selected (e.g. --roles calib), skip it and 03 falls back to synthetic.
    test = subset[subset["role"] == "test"].reset_index(drop=True)
    if len(test):
        r0 = test.iloc[0]
        x0 = _preprocess(_role_dir(args.tree, "test"), r0, means, stds, args.img_size)[None]
        bench = config.EVAL_SAMPLE
        os.makedirs(os.path.dirname(bench), exist_ok=True)
        bmeta = {
            "role": "test",
            "split": ROLE_SPLIT["test"],
            "patch_id": str(r0["patch_id"]),
            "img_size": args.img_size,
            "seed": args.seed,
            "zenodo_record": ZENODO_RECORD,
            "band_order": BAND_ORDER,
        }
        np.savez_compressed(bench, x=x0, meta=json.dumps(bmeta))
        man["packs"]["bench_sample"] = {
            "path": bench,
            "sha256": _hash(bench, "sha256"),
            "shape": list(x0.shape),
            **bmeta,
        }
        log.info("[pack] wrote %s  shape=%s (1 real test patch)", bench, x0.shape)
    else:
        log.warning("[pack] no 'test' role in subset; skipped bench_sample (03 -> synthetic)")

    _save_manifest(man)
    log.info("[pack] wrote %s  shape=%s  sha256=%s", out, out_arr.shape, _hash(out, "sha256")[:12])


# --- 5. CLEANUP --------------------------------------------------------------
def stage_cleanup(args):
    """Delete what is no longer needed once the subset is on disk (one pass).

    The 117 GB .tar.zst archives are not needed after extraction; calib is now
    inside the npz, so its extracted tree is redundant too. train/ and test/
    (+ their parquets) and the calib npz are kept. --keep-archives skips this.
    """
    if args.keep_archives:
        log.info("[cleanup] --keep-archives set: nothing removed")
        return
    freed = 0
    for arc in ("BigEarthNet-S2.tar.zst", "BigEarthNet-S1.tar.zst"):
        p = os.path.join(args.archives, arc)
        if os.path.exists(p):
            n = os.path.getsize(p)  # measure BEFORE deleting - report the truth
            os.remove(p)
            freed += n
            log.info("[cleanup] removed %s (%s reclaimed)", p, _human_bytes(n))
    calib_tree = _role_dir(args.tree, "calib")
    if os.path.isdir(calib_tree):
        n = _dir_bytes(calib_tree)
        shutil.rmtree(calib_tree)
        freed += n
        log.info("[cleanup] removed %s (%s, calib is in the npz)", calib_tree, _human_bytes(n))
    log.info("[cleanup] total reclaimed: %s", _human_bytes(freed))


STAGES = {
    "download": stage_download,
    "select": stage_select,
    "extract": stage_extract,
    "pack": stage_pack,
    "cleanup": stage_cleanup,
}


# --- PREFLIGHT ---------------------------------------------------------------
def _preflight(args, order):
    """Fail LOUD, UP FRONT - before any expensive work - on inputs this script
    cannot produce itself.

    norm_stats.json (a harness product placed in input/) is only READ by 'pack',
    but on `--stage all` pack runs AFTER the ~1 h download+extract. Checking it
    here means a missing file costs a second, not an hour. Same principle as the
    rest of the pipeline: if a required input is absent, stop at the start with a
    clear message, don't discover it deep in the run.
    """
    problems = []
    if "pack" in order and not os.path.exists(args.stats):
        problems.append(
            f"norm_stats.json not found at '{args.stats}'. It is a harness "
            f"product (the pre-training band mean/std); place it in input/ before "
            f"running 'pack' or '--stage all'. Override the path with --stats."
        )
    if problems:
        raise SystemExit(
            "[preflight] cannot start - missing required input(s):\n  - " + "\n  - ".join(problems)
        )


def main():
    ap = argparse.ArgumentParser(
        description="Download reBEN and build the calib subset (self-contained).",
    )
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--roles", nargs="+", default=list(ROLE_SPLIT), choices=list(ROLE_SPLIT))
    ap.add_argument("--archives", default=os.path.join(config.INPUT_DIR, "reben"))
    ap.add_argument("--parquet", default=None, help="default: <archives>/metadata.parquet")
    ap.add_argument("--tree", default=os.path.join(config.INPUT_DIR, "reben_subset"))
    ap.add_argument("--stats", default=os.path.join(config.INPUT_DIR, "norm_stats.json"))
    ap.add_argument("--out", default=None, help="calib pack (default: config.CALIB_PACK)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-size", type=int, default=config.INPUT_H)
    ap.add_argument("--n-calib", type=int, default=DEFAULT_N["calib"])
    ap.add_argument("--n-train", type=int, default=DEFAULT_N["train"])
    ap.add_argument("--n-test", type=int, default=DEFAULT_N["test"])
    ap.add_argument(
        "--keep-archives", action="store_true", help="do NOT delete the .tar.zst after extraction"
    )
    args = ap.parse_args()

    order = list(STAGES) if args.stage == "all" else [args.stage]
    _preflight(args, order)  # check unproducible inputs BEFORE the expensive work
    for st in order:
        STAGES[st](args)
    log.info("=== DONE (%s) ===", ", ".join(order))


if __name__ == "__main__":
    main()
