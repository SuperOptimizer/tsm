#!/usr/bin/env python
"""Numerical parity of tsm.teachers against the reference implementations.

Dev-only. Never imported by src/tsm. Run it inside a *separate* venv that has the
reference code, e.g.::

    uv venv -p 3.12 /tmp/villa-venv
    uv pip install -p /tmp/villa-venv --index-url https://download.pytorch.org/whl/cpu torch numpy
    uv pip install -p /tmp/villa-venv --no-deps dynamic-network-architectures     # nnU-Net blocks (m7)
    uv pip install -p /tmp/villa-venv einops timm huggingface_hub pyyaml          # imported by villa's builder
    /tmp/villa-venv/bin/python dev/parity_villa.py recto m7 ink lasagna

villa itself is *not* installed: the script puts ``--villa-src`` on ``sys.path``
and registers a stub ``vesuvius`` package so that ``vesuvius.models.build.*`` can
be imported without ``vesuvius/__init__.py`` pulling in the volume/S3 stack.
(``uv pip install -e /home/forrest/villa/vesuvius`` works too but drags in
volume-cartographer & co.)

For each teacher it:
  1. loads the checkpoint once (mmap),
  2. builds the reference net -- villa ``NetworkFromConfig`` from the checkpoint's
     ``model_config`` (recto), the training ``config`` (ink) or the lasagna
     trainer's recipe (lasagna); ``ResidualEncoderUNet`` from nnU-Net ``init_args``
     plans (m7) -- and loads the weights strictly,
  3. builds our net via ``tsm.teachers.load_teacher``,
  4. runs both on the same seeded random fp32 input and prints max|delta| of the
     final logits plus per-encoder-stage deltas.

The patch (default 128^3) must be >= 2^(n_stages-1) * 2: with 7 stages a 64^3
input reaches a 1^3 bottleneck and InstanceNorm rejects it (in both nets).
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from types import SimpleNamespace

import torch
import torch.nn as nn

MODELS_DIR = os.path.expanduser("~/.cache/tsm-models")


def _register_stub_package(name: str, path: str) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # namespace-like: submodules resolve, __init__ is skipped
    sys.modules[name] = mod


def _setup_paths(villa_src: str, tsm_src: str) -> None:
    sys.path.insert(0, tsm_src)
    sys.path.insert(0, villa_src)
    _register_stub_package("vesuvius", os.path.join(villa_src, "vesuvius"))


# --------------------------------------------------------------------------- #
# Reference builders
# --------------------------------------------------------------------------- #
def _mgr(model_config: dict, targets: dict, patch: int, in_channels: int = 1, autoconfigure: bool = False):
    return SimpleNamespace(
        model_config=model_config,
        targets=targets,
        train_patch_size=(patch, patch, patch),
        train_batch_size=1,
        in_channels=in_channels,
        autoconfigure=autoconfigure,
        spacing=[1, 1, 1],
        enable_deep_supervision=False,
        model_name="parity",
    )


def build_villa_recto(ckpt: dict) -> nn.Module:
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig

    mc = dict(ckpt["model_config"])
    targets = {k: dict(v) for k, v in mc["targets"].items()}
    patch = int(mc.get("train_patch_size", mc.get("patch_size", [256]))[0])
    return NetworkFromConfig(_mgr(mc, targets, patch, int(mc.get("in_channels", 1)), autoconfigure=False))


def build_villa_ink(ckpt: dict) -> nn.Module:
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig

    cfg = ckpt["config"]
    mc = dict(cfg["model_config"])
    targets = {k: dict(v) for k, v in cfg["targets"].items()}
    patch = int(cfg["patch_size"][0])
    return NetworkFromConfig(_mgr(mc, targets, patch, int(cfg.get("in_channels", 1)),
                                  autoconfigure=bool(mc.get("autoconfigure", True))))


def build_villa_lasagna(ckpt: dict, n_stages: int) -> nn.Module:
    # mirrors villa/lasagna/train_unet_3d.py:build_model
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig

    norm_type = ckpt.get("norm_type", "instance")
    mc: dict = {"autoconfigure": True, "architecture_type": "unet"}
    if norm_type == "group":
        mc["norm_op"] = "nn.GroupNorm"
        mc["norm_op_kwargs"] = {"num_groups": 32, "affine": True, "eps": 1e-5}
    elif norm_type == "none":
        mc["norm_op"] = None
    mc["upsample_mode"] = ckpt.get("upsample_mode", "transpconv")
    targets = {"output": {"out_channels": int(ckpt.get("out_channels", 8)), "activation": "none"}}
    # autoconfigure derives the stage count from the patch (min feature map 4)
    arch_patch = int(ckpt.get("model_patch_size") or 4 * 2 ** (n_stages - 1))
    return NetworkFromConfig(_mgr(mc, targets, arch_patch, int(ckpt.get("in_channels", 1)), autoconfigure=True))


def build_nnunet_m7(ckpt: dict) -> nn.Module:
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

    init = ckpt["init_args"]
    cfg = init["plans"]["configurations"][init["configuration"]]
    arch = cfg["architecture"]
    kw = dict(arch["arch_kwargs"])
    for k in arch.get("_kw_requires_import", []):
        if kw.get(k) is not None:
            mod, _, attr = kw[k].rpartition(".")
            kw[k] = getattr(__import__(mod, fromlist=[attr]), attr)
    # nnU-Net's LabelManager drops the 'ignore' label, so read the class count off the seg head.
    num_classes = int(ckpt["network_weights"]["decoder.seg_layers.0.weight"].shape[0])
    net = ResidualEncoderUNet(input_channels=1, num_classes=num_classes, deep_supervision=True, **kw)
    net.decoder.deep_supervision = False
    return net


# --------------------------------------------------------------------------- #
def _stage_hooks(module_list, store: dict, prefix: str):
    handles = []
    for i, m in enumerate(module_list):
        handles.append(m.register_forward_hook(lambda _m, _i, out, i=i: store.__setitem__(f"{prefix}{i}", out.detach())))
    return handles


def run(name: str, patch: int, seed: int) -> float:
    from tsm.teachers import TEACHERS, load_teacher

    spec = TEACHERS[name]
    path = os.path.join(MODELS_DIR, spec.file)
    ours, spec = load_teacher(name, MODELS_DIR)
    ours = ours.float().eval()

    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    sd = ckpt[spec.state_key]
    if name == "recto":
        ref = build_villa_recto(ckpt)
    elif name == "ink":
        ref = build_villa_ink(ckpt)
    elif name == "lasagna":
        ref = build_villa_lasagna(ckpt, ours.spec.n_stages)
    elif name == "m7":
        ref = build_nnunet_m7(ckpt)
    else:
        raise SystemExit(f"unknown teacher {name}")
    missing, unexpected = ref.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[{name}] reference strict load FAILED: missing={list(missing)[:10]} unexpected={list(unexpected)[:10]}")
        raise SystemExit(1)
    del ckpt, sd
    ref = ref.float().eval()

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, patch, patch, patch, generator=g)

    ref_acts, our_acts = {}, {}
    if spec.kind == "vesuvius":
        h = _stage_hooks(ref.shared_encoder.stages, ref_acts, "enc") + _stage_hooks(ours.shared_encoder.stages, our_acts, "enc")
    else:
        h = _stage_hooks(ref.encoder.stages, ref_acts, "enc") + _stage_hooks(ours.encoder.stages, our_acts, "enc")

    with torch.inference_mode():
        y_ref = ref(x)
        y_ours = ours(x)
    for hh in h:
        hh.remove()
    if isinstance(y_ref, dict):
        y_ref = y_ref[spec.target]
    if isinstance(y_ref, (list, tuple)):
        y_ref = y_ref[0]
    y_ours = spec.select(y_ours)

    print(f"[{name}] input {tuple(x.shape)}  ref out {tuple(y_ref.shape)}  ours {tuple(y_ours.shape)}")
    for k in sorted(ref_acts, key=lambda s: int(s[3:])):
        d = (ref_acts[k] - our_acts[k]).abs().max().item()
        print(f"[{name}]   {k}: max|delta| = {d:.3e}   (|ref| max {ref_acts[k].abs().max().item():.3e})")
    d = (y_ref - y_ours).abs().max().item()
    print(f"[{name}] final logits max|delta| = {d:.3e}   (|ref| max {y_ref.abs().max().item():.3e})  "
          f"{'PASS' if d < 1e-4 else 'FAIL'}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("teachers", nargs="*", default=["recto", "m7", "ink", "lasagna"])
    ap.add_argument("--villa-src", default=os.path.expanduser("~/villa/vesuvius/src"))
    ap.add_argument("--tsm-src", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    _setup_paths(args.villa_src, args.tsm_src)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    worst = 0.0
    for name in args.teachers:
        worst = max(worst, run(name, args.patch, args.seed))
    print(f"worst max|delta| over {args.teachers}: {worst:.3e}")
    sys.exit(0 if worst < 1e-4 else 1)


if __name__ == "__main__":
    main()
