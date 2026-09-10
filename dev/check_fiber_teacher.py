"""Load the 4-class fibre teacher strictly and run one small forward (CPU, no GPU needed).

    uv run python dev/check_fiber_teacher.py [--patch 128]

Prints the inferred architecture, confirms the state dict loads with exact key parity and
that the head produces a 4-channel softmax that sums to 1 over the class axis.  Also dumps
the checkpoint's embedded training config so the normalisation can be re-checked against
villa's ``scripts/fiber_5class/dataset.py`` (percentile_minmax, 1st-99th percentile).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from tsm.teachers import DEFAULT_MODELS_DIR, TEACHERS, infer_vesuvius_arch, load_teacher, shapes_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    # 7 encoder stages (total stride 64) + InstanceNorm: the bottleneck needs > 1 voxel
    # per axis, so the smallest forward this net accepts is 128^3.
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--dump-inventory", default=None, help="write the shape-only tensor inventory here")
    a = ap.parse_args(argv)

    spec = TEACHERS["fiber"]
    path = os.path.join(a.models_dir, spec.file)
    print(f"checkpoint {path} ({os.path.getsize(path) / 1e9:.2f} GB)")
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    print("top-level keys:", list(ck))
    print("step:", ck.get("step"))
    cfg = ck.get("config") or {}
    for k in ("target_name", "out_channels", "in_channels", "patch_size", "activation", "dataset"):
        print(f"  config.{k} = {cfg.get(k)}")
    sd = ck["ema"]["model_state"]
    shapes = shapes_of(sd)
    if a.dump_inventory:
        json.dump({"tensors": {k: {"shape": list(v)} for k, v in shapes.items()}}, open(a.dump_inventory, "w"))
        print(f"wrote {a.dump_inventory} ({len(shapes)} tensors)")
    arch = infer_vesuvius_arch(shapes)
    print(f"inferred arch: features={arch.features_per_stage} blocks={arch.n_blocks_per_stage} "
          f"strides={arch.strides} se={arch.squeeze_excitation} norm={arch.norm} "
          f"task_heads={arch.task_heads} shared_decoder={arch.shared_decoder is not None} "
          f"task_decoders={list(arch.task_decoders)}")
    del ck, sd

    net, spec = load_teacher("fiber", a.models_dir)  # raises unless the load is exact
    print(f"strict load OK; activation={spec.activation} normalizer={spec.normalizer} target={spec.target!r}")

    p = int(a.patch)
    rng = np.random.default_rng(0)
    x = spec.normalizer(rng.integers(0, 256, (p, p, p), dtype=np.uint8))[None, None]
    with torch.inference_mode():
        logits = spec.select(net(x))
    prob = torch.softmax(logits, dim=1)
    print(f"forward {p}^3 -> logits {tuple(logits.shape)}; finite={bool(torch.isfinite(logits).all())}")
    print(f"softmax sums to 1: {bool(torch.allclose(prob.sum(1), torch.ones(1, p, p, p), atol=1e-5))}")
    print("mean prob per class [bg, vt, hz, ink]:", [round(float(v), 4) for v in prob.mean((0, 2, 3, 4))])
    assert tuple(logits.shape) == (1, 4, p, p, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
