#!/usr/bin/env python
"""Numerical parity of :class:`tsm.dino.DinoVolViT` against villa's ``dinovol_2``.

Both nets get the *same* random weights (ours is loaded from the reference net's
state dict) and the same random 48^3 input on CPU; the script prints max|delta|
of the normalised patch tokens, the cls token and the per-block RoPE embedding.

The reference implementation lives in the villa monorepo and needs ``timm`` and
``einops`` on top of torch/numpy (nothing else in ``dinovol_2/model``), so no
throwaway venv is needed -- point ``PYTHONPATH`` at a shallow clone and a
``--target`` install of the two packages::

    git clone --depth 1 https://github.com/ScrollPrize/villa ~/tsm-output/tmp_villa_upstream
    uv pip install --target ~/tsm-output/tmp_dino_deps 'timm==1.0.20' einops --no-deps
    uv pip install --target ~/tsm-output/tmp_dino_deps huggingface_hub safetensors pyyaml
    .venv/bin/python dev/parity_dino.py

Options: ``--size`` (input cube, default 48 = 6^3 tokens; keep it small, this is
a 216M-param ViT on CPU), ``--depth`` (truncate the reference net for speed),
``--real`` (load the released checkpoint into both instead of random weights).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

VILLA = os.path.expanduser("~/tsm-output/tmp_villa_upstream/dinovol")
DEPS = os.path.expanduser("~/tsm-output/tmp_dino_deps")


def _import_reference():
    for p in (VILLA, DEPS):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    from dinovol_2.model.dinov2_eva import Eva  # noqa: E402
    from dinovol_2.model.rope import MixedRopePositionEmbedding  # noqa: E402

    return Eva, MixedRopePositionEmbedding


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", action="store_true", help="load ~/.cache/tsm-models/dinovol.pt into both")
    args = ap.parse_args()

    from tsm.dino import DinoVolViT, ViTSpec, rope_coords

    try:
        Eva, MixedRope = _import_reference()
    except Exception as exc:
        print(f"reference import failed ({type(exc).__name__}: {exc})")
        print(f"expected the villa clone at {VILLA} and timm/einops at {DEPS}; see the module docstring")
        return 2

    torch.manual_seed(args.seed)
    depth = args.depth
    ref = Eva(
        input_channels=1, global_crops_size=(128, 128, 128), local_crops_size=(64, 64, 64),
        embed_dim=864, patch_size=(8, 8, 8), depth=depth, num_heads=16, qkv_bias=True, qkv_fused=True,
        mlp_ratio=2.6666666666666665, swiglu_mlp=True, scale_mlp=True, scale_attn_inner=False,
        class_token=True, use_abs_pos_emb=False, use_rot_pos_emb=True, num_reg_tokens=4,
        rope_impl=MixedRope, rope_kwargs={"base": 100.0, "normalize_coords": "separate"},
    ).eval()
    spec = ViTSpec(in_channels=1, embed_dim=864, depth=depth, num_heads=16, patch=8,
                   mlp_hidden=2304, num_reg_tokens=4)
    ours = DinoVolViT(spec).eval()

    sd = ref.state_dict()
    if args.real:
        raw = torch.load(os.path.expanduser("~/.cache/tsm-models/dinovol.pt"), map_location="cpu",
                         mmap=True, weights_only=False)["teacher"]
        raw = {k[len("backbone."):]: v for k, v in raw.items() if k.startswith("backbone.")}
        sd = {k: raw[k] for k in sd if k in raw}
        ref.load_state_dict(sd, strict=False)
    missing, unexpected = ours.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("k_bias")]
    print(f"load into ours: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing or unexpected:
        print("  missing:", missing[:8])
        print("  unexpected:", unexpected[:8])

    n = args.size
    x = torch.randn(1, 1, n, n, n)

    # --- RoPE only ---------------------------------------------------------- #
    grid = (n // 8,) * 3
    r_ref = ref.blocks[0].rope_embed
    c_ref = r_ref.get_coords(grid)
    c_ours = rope_coords(grid)
    print(f"coords          max|d| = {float((c_ref - c_ours).abs().max()):.3e}")
    sin_r, cos_r = r_ref.get_embed_from_coords(c_ref)
    sin_o, cos_o = ours.blocks[0].rope_embed.embed(c_ours)
    print(f"rope sin/cos    max|d| = {max(float((sin_r - sin_o).abs().max()), float((cos_r - cos_o).abs().max())):.3e}")

    # --- full forward ------------------------------------------------------- #
    with torch.inference_mode():
        out_ref = ref.forward_features(x, masks=None, view_kind="global")
        out_ours = ours.forward_features(x)
    pt_r = torch.nn.functional.normalize(out_ref["x_norm_patchtokens"], dim=-1)
    pt_o = torch.nn.functional.normalize(out_ours["patch_tokens"], dim=-1)
    print(f"patch tokens    shape {tuple(pt_o.shape)} grid {out_ours['grid']}")
    print(f"patch tokens    max|d| = {float((pt_r - pt_o).abs().max()):.3e}  "
          f"(rel rms {float(((pt_r - pt_o) ** 2).mean().sqrt() / pt_r.std()):.3e})")
    print(f"cls token       max|d| = {float((out_ref['x_norm_clstoken'] - out_ours['cls']).abs().max()):.3e}")
    print(f"reg tokens      max|d| = {float((out_ref['x_norm_regtokens'] - out_ours['reg']).abs().max()):.3e}")
    ok = float((pt_r - pt_o).abs().max()) < 1e-4
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
