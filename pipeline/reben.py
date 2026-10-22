#!/usr/bin/env python3
# =============================================================================
#  reben.py  -  reBEN loader bits, a FAITHFUL copy of the eval harness
# -----------------------------------------------------------------------------
#  Single source of truth (in THIS project) for how a reBEN patch becomes the
#  6-channel tensor the encoder was pre-trained on, plus the 19-class label
#  vector. It is a documented, faithful copy of the published harness
#  (strata-s12-eval-harness/reben_loader.py, from Paper 1) so
#  the accuracy axis is evaluated with the EXACT input distribution and label
#  order of the probe. It must stay in sync with that file.
#
#  Used by the accuracy branch (05_extract_features, 06_linear_probe). The data
#  prep (prepare_data.py) carries its own equivalent transform for the calib/
#  bench packs; both mirror the same harness source.
# =============================================================================
from __future__ import annotations

import json
import os

import numpy as np

# reBEN's 19 classes in FIXED order (alphabetical, as in the parquet). Fixed so
# the 19-hot vector is reproducible. Verbatim from the harness reben_loader.py.
REBEN_19_CLASSES = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(REBEN_19_CLASSES)}

# S2 bands the encoder uses (B04,B03,B02) and S1 (VV,VH). VV-VH is derived.
S2_BANDS = ("B04", "B03", "B02")
S1_BANDS = ("VV", "VH")


def s2_tile_of(patch_id: str) -> str:
    """S2 tile = patch_id minus its last 2 components (row, col)."""
    return patch_id.rsplit("_", 2)[0]


def s1_tile_of(s1_name: str) -> str:
    """S1 tile = s1_name minus its last 3 components (zone, row, col)."""
    return s1_name.rsplit("_", 3)[0]


def read_band(path):
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)  # (H, W)


def load_6ch(tree, row):
    """Read the 6-channel stack for one patch: [B4,B3,B2,VV,VH,VV-VH] (6,120,120)."""
    pid, s1 = row["patch_id"], row["s1_name"]
    s2d = os.path.join(tree, "BigEarthNet-S2", s2_tile_of(pid), pid)
    s1d = os.path.join(tree, "BigEarthNet-S1", s1_tile_of(s1), s1)
    b4 = read_band(os.path.join(s2d, f"{pid}_B04.tif"))
    b3 = read_band(os.path.join(s2d, f"{pid}_B03.tif"))
    b2 = read_band(os.path.join(s2d, f"{pid}_B02.tif"))
    vv = read_band(os.path.join(s1d, f"{s1}_VV.tif"))
    vh = read_band(os.path.join(s1d, f"{s1}_VH.tif"))
    return np.stack([b4, b3, b2, vv, vh, vv - vh], axis=0)


def load_norm_stats(path):
    """Return (means, stds) as torch tensors shaped (6,1,1); std clamped at 1e-6."""
    import torch

    with open(path) as f:
        s = json.load(f)
    means = torch.tensor(s["band_means"], dtype=torch.float32).view(-1, 1, 1)
    stds = torch.clamp(torch.tensor(s["band_stds"], dtype=torch.float32).view(-1, 1, 1), min=1e-6)
    return means, stds


def preprocess(tree, row, means, stds, img_size=224):
    """6-ch patch -> (6, img_size, img_size) float32, normalised. Harness transform:
    resize to img_size (bilinear, align_corners=False) then per-band (x-mean)/std."""
    import torch

    arr = load_6ch(tree, row)  # (6, 120, 120)
    x = torch.from_numpy(arr).unsqueeze(0)
    x = torch.nn.functional.interpolate(
        x, size=(img_size, img_size), mode="bilinear", align_corners=False
    ).squeeze(0)
    return ((x - means) / stds).to(torch.float32).numpy()


def labels_19hot(row):
    """19-hot label vector (float32) from the parquet 'labels' list for one patch."""
    vec = np.zeros(len(REBEN_19_CLASSES), dtype=np.float32)
    for lab in row["labels"]:
        idx = CLASS_TO_IDX.get(lab)
        if idx is not None:
            vec[idx] = 1.0
    return vec
