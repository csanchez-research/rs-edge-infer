#!/usr/bin/env python3
# =============================================================================
#  encoder_infer.py  -  inference-only encoder (NO masking) for deployment
# -----------------------------------------------------------------------------
#  Rebuilds the paper-1 MAE ENCODER with the SAME layer names as the training
#  model.py, so the trained weights load straight from encoder.pt. The decoder
#  is not built (deployment does not need it).
#
#  The checkpoint IS the source of truth. This module ADAPTS to it:
#    - It accepts the paper-1 layout {"encoder": <state_dict>, "arch", "in_chans"}
#      as well as a bare state_dict or the usual "model"/"state_dict" wrappers.
#    - It INFERS embed_dim, depth, in_chans and patch from the weights themselves
#      (and num_heads from the `arch` string, falling back to head_dim=64), so a
#      different ViT size loads without code changes.
#    - It loads with strict=True and refuses (raises) on any missing/unexpected
#      key or shape mismatch. A partial load would silently inject random weights
#      and produce meaningless embeddings, so if it is wrong it must fail here.
#
#  The only difference vs training is NO random masking: all patches go through
#  the encoder and we return the cls-token embedding. Masking is a pretraining
#  trick; at inference it would drop patches and be non-deterministic.
#
#  Requires torch + timm (docker/requirements-dev.txt). Runs on a dev machine,
#  NOT necessarily on the N5105.
#
#  References: ViT backbone [Dosovitskiy et al. 2021,
#  arXiv:2010.11929]; MAE self-supervised pre-training [He et al. 2022,
#  arXiv:2111.06377]; instantiated with timm [Wightman 2019].
# =============================================================================
from __future__ import annotations

import re

import numpy as np
import torch
from timm.models.vision_transformer import Block
from torch import nn


def _posenc_2d_sincos(embed_dim: int, grid_size: int) -> torch.Tensor:
    """(1, grid*grid, embed_dim) fixed 2D sin/cos position (no cls here).

    Placeholder of the right shape: the trained pos_embed in the checkpoint
    overwrites this on load. It only has to match the shape so strict load fits.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"
    g = np.arange(grid_size, dtype=np.float32)
    gy, gx = np.meshgrid(g, g, indexing="ij")
    dim_q = embed_dim // 4
    omega = 1.0 / (10000 ** (np.arange(dim_q, dtype=np.float32) / dim_q))

    def _emb(pos):
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    pos = np.concatenate([_emb(gy), _emb(gx)], axis=1)
    return torch.from_numpy(pos).float().unsqueeze(0)


class PatchEmbed6(nn.Module):
    """in_chans-channel patches -> tokens, via conv with stride = patch."""

    def __init__(self, img_size: int, patch: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.grid = img_size // patch
        self.n_patches = self.grid * self.grid
        self.patch = patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):  # x: (B, C, H, W)
        x = self.proj(x)  # (B, E, grid, grid)
        return x.flatten(2).transpose(1, 2)  # (B, n_patches, E)


class EncoderInfer(nn.Module):
    """ViT encoder, inference forward (no masking) -> cls embedding.

    Dimensions are passed in (inferred from the checkpoint), not hard-coded, so
    the same class loads a ViT-S / ViT-B / ... without edits.
    """

    def __init__(
        self,
        in_chans: int,
        img_size: int,
        patch: int,
        embed_dim: int,
        depth: int,
        heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed6(img_size, patch, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer(
            "pos_embed",
            torch.cat(
                [
                    torch.zeros(1, 1, embed_dim),
                    _posenc_2d_sincos(embed_dim, self.patch_embed.grid),
                ],
                dim=1,
            ),
        )
        self.blocks = nn.ModuleList(
            [Block(embed_dim, heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)  # all patches, no masking
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # embedding = cls token


#  num_heads is not recoverable from the weight shapes, so it comes from the
#  checkpoint's arch string; otherwise we assume the standard head_dim = 64.
_HEADS_BY_ARCH = {"vit_tiny": 3, "vit_small": 6, "vit_base": 12, "vit_large": 16}


def _find_state_dict(ckpt):
    """Locate the tensor state_dict in any of the known checkpoint layouts."""
    if not isinstance(ckpt, dict):
        return ckpt
    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt  # already a bare state_dict
    for key in ("encoder", "model", "state_dict"):  # paper-1 uses "encoder"
        sub = ckpt.get(key)
        if isinstance(sub, dict) and sub and all(torch.is_tensor(v) for v in sub.values()):
            return sub
    raise RuntimeError(
        f"No weights state_dict found in checkpoint. Top-level keys: {list(ckpt.keys())[:12]}"
    )


def _infer_dims(state, arch_str):
    """Infer (embed_dim, depth, heads, in_chans, patch) from the weights."""
    if "cls_token" not in state or "patch_embed.proj.weight" not in state:
        raise RuntimeError(
            "Checkpoint lacks cls_token / patch_embed.proj.weight; it does not "
            "look like the paper-1 ViT encoder."
        )
    embed_dim = int(state["cls_token"].shape[-1])
    w = state["patch_embed.proj.weight"]  # (embed, in_chans, patch, patch)
    in_chans, patch = int(w.shape[1]), int(w.shape[2])
    idxs = {int(m.group(1)) for k in state if (m := re.match(r"blocks\.(\d+)\.", k))}
    depth = (max(idxs) + 1) if idxs else 0
    heads = next(
        (h for pre, h in _HEADS_BY_ARCH.items() if str(arch_str or "").startswith(pre)),
        max(embed_dim // 64, 1),
    )
    return embed_dim, depth, heads, in_chans, patch


def load_encoder(checkpoint_path: str, in_chans: int, img_size: int, patch: int):
    """Build the inference encoder and load the trained weights (strict).

    Adapts to the checkpoint (layout + dimensions) and fails loudly on any
    mismatch: a silent partial load would inject random weights.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _find_state_dict(ckpt)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

    arch_str = ckpt.get("arch") if isinstance(ckpt, dict) else None
    embed_dim, depth, heads, ck_in_chans, ck_patch = _infer_dims(state, arch_str)

    # Adapt-or-fail: the checkpoint is the source of truth; refuse on any clash
    # with the configured values (so a wrong config can never pass silently).
    if ck_in_chans != in_chans:
        raise RuntimeError(f"checkpoint in_chans={ck_in_chans} != configured {in_chans}")
    if ck_patch != patch:
        raise RuntimeError(f"checkpoint patch={ck_patch} != configured {patch}")
    meta_in = ckpt.get("in_chans") if isinstance(ckpt, dict) else None
    if meta_in is not None and int(meta_in) != ck_in_chans:
        raise RuntimeError(
            f"checkpoint metadata in_chans={meta_in} != weights in_chans={ck_in_chans}"
        )

    model = EncoderInfer(
        in_chans=in_chans,
        img_size=img_size,
        patch=patch,
        embed_dim=embed_dim,
        depth=depth,
        heads=heads,
    )

    # Hard gate: key sets must match exactly before we even attempt the load.
    own, have = set(model.state_dict().keys()), set(state.keys())
    missing, extra = own - have, have - own
    if missing or extra:
        raise RuntimeError(
            "Checkpoint does not match the inference encoder - refusing to load. "
            "A partial load would silently inject random weights.\n"
            f"  arch={arch_str!r} embed_dim={embed_dim} depth={depth} heads={heads}\n"
            f"  missing in checkpoint ({len(missing)}): {sorted(missing)[:8]}\n"
            f"  unexpected in checkpoint ({len(extra)}): {sorted(extra)[:8]}"
        )

    # strict=True also checks every tensor SHAPE and raises on any mismatch.
    model.load_state_dict(state, strict=True)

    print(
        f"[encoder] loaded {len(have)}/{len(own)} tensors (strict) | arch={arch_str} "
        f"embed_dim={embed_dim} depth={depth} heads={heads} in_chans={in_chans} patch={patch}"
    )
    return model.eval()
