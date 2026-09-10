"""Exact state-dict key/shape parity against the published tensor inventories.

For every ``~/.cache/tsm-production/models/<name>/<rev>/<sha>.tsm-model/model.json``
we build the network from the shape-only inventory and assert that
``{k: shape for k in net.state_dict()}`` equals the inventory exactly.  No
checkpoint is loaded; constructing the nets takes a few seconds on CPU.
"""

from __future__ import annotations

import glob
import json
import os

import pytest
import torch

from tsm.teachers import build_teacher, infer_nnunet_arch, infer_vesuvius_arch

INVENTORY_ROOT = os.path.expanduser("~/.cache/tsm-production/models")
SKIP = ("dinovol", "winding")  # not U-Nets


def _inventories() -> list[str]:
    files = sorted(glob.glob(os.path.join(INVENTORY_ROOT, "*", "*", "*.tsm-model", "model.json")))
    return [f for f in files if not any(s in f for s in SKIP)]


def load_inventory(path: str) -> dict[str, tuple[int, ...]]:
    d = json.load(open(path))
    t = d["tensors"]
    items = t.items() if isinstance(t, dict) else ((x["name"], x) for x in t)
    return {k: tuple(int(i) for i in v["shape"]) for k, v in items}


def kind_of(shapes: dict) -> str:
    if any(k.startswith("shared_encoder.") for k in shapes):
        return "vesuvius"
    if any(k.startswith("encoder.") for k in shapes):
        return "nnunet"
    raise AssertionError("unknown key layout")


def diff_report(expected: dict, got: dict) -> str:
    missing = sorted(set(expected) - set(got))
    unexpected = sorted(set(got) - set(expected))
    wrong = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
    lines = [f"{len(missing)} missing, {len(unexpected)} unexpected, {len(wrong)} wrong shape"]
    lines += [f"  missing:    {k} {expected[k]}" for k in missing[:15]]
    lines += [f"  unexpected: {k} {got[k]}" for k in unexpected[:15]]
    lines += [f"  shape:      {k} expected {expected[k]} got {got[k]}" for k in wrong[:15]]
    return "\n".join(lines)


@pytest.mark.parametrize("path", _inventories() or [pytest.param(None, marks=pytest.mark.skip("no inventories"))],
                         ids=lambda p: p.split(os.sep)[-4] if p else "none")
def test_inventory_key_parity(path):
    expected = load_inventory(path)
    kind = kind_of(expected)
    net = build_teacher(kind, expected)
    got = {k: tuple(v.shape) for k, v in net.state_dict().items()}
    assert got == expected, diff_report(expected, got)


def test_infer_recto_spec_matches_published_config():
    files = [f for f in _inventories() if "surface_recto_3dunet" in f]
    if not files:
        pytest.skip("recto inventory not present")
    spec = infer_vesuvius_arch(load_inventory(files[0]))
    assert spec.features_per_stage == [32, 64, 128, 256, 320, 320, 320]
    assert spec.n_blocks_per_stage == [1, 3, 4, 6, 6, 6, 6]
    assert spec.strides == [1, 2, 2, 2, 2, 2, 2]
    assert spec.squeeze_excitation == "scse"
    assert spec.norm == "instance" and spec.conv_bias
    assert list(spec.task_decoders) == ["surface"]
    assert spec.task_decoders["surface"].out_channels == 2
    assert spec.task_decoders["surface"].n_conv_per_stage == [1] * 6
    assert spec.task_decoders["surface"].upsample_mode == "transpconv"
    assert spec.shared_decoder is None and spec.task_heads == {}


def test_infer_ink_spec_uses_shared_decoder_head():
    files = [f for f in _inventories() if "ink_3d_dino_guided" in f]
    if not files:
        pytest.skip("ink inventory not present")
    spec = infer_vesuvius_arch(load_inventory(files[0]))
    assert spec.features_per_stage == [32, 64, 128, 256, 320, 320, 320]
    assert spec.squeeze_excitation is None
    assert spec.task_decoders == {}
    assert spec.task_heads == {"ink": 1}
    assert spec.shared_decoder is not None and spec.shared_decoder.out_channels is None


def test_infer_m7_spec_matches_nnunet_plans():
    files = [f for f in _inventories() if "surface_m7_nnunet" in f]
    if not files:
        pytest.skip("m7 inventory not present")
    spec = infer_nnunet_arch(load_inventory(files[0]))
    assert spec.features_per_stage == [32, 64, 128, 256, 320, 320]
    assert spec.n_blocks_per_stage == [1, 3, 4, 6, 6, 6]
    assert spec.strides == [1, 2, 2, 2, 2, 2]
    assert spec.decoder.out_channels == 2
    assert spec.decoder.n_conv_per_stage == [1] * 5
    assert spec.squeeze_excitation is None


def test_shape_dict_accepts_tensors_and_inventory_entries():
    files = [f for f in _inventories() if "surface_m7_nnunet" in f]
    if not files:
        pytest.skip("m7 inventory not present")
    inv = load_inventory(files[0])
    as_meta = {k: torch.empty(s, device="meta") for k, s in inv.items()}
    a = infer_nnunet_arch(inv)
    b = infer_nnunet_arch(as_meta)
    c = infer_nnunet_arch({"module." + k: v for k, v in inv.items()})
    assert a == b == c
