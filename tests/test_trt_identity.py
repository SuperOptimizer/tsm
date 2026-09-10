"""R01: ONNX / TensorRT engine cache names are bound to the checkpoint, not just its shape.

CPU-only: nothing here builds an engine, it only exercises the naming/fingerprint layer
(and one real ONNX export, which runs on the CPU).
"""

from __future__ import annotations

import os

import pytest
import torch

from tsm.student import TSMNet
from tsm.teachers import TEACHERS
from tsm.trt import (
    TRTStudent,
    TRTTeacher,
    arch_fingerprint,
    engine_cache_key,
    net_fingerprint,
    student_engine_name,
    student_head_channels,
    teacher_cache_name,
    weight_fingerprint,
)

SMALL = dict(in_ch=2, widths=(8, 16, 32), blocks=1, dec_blocks=1, bottleneck_blocks=1, ds_levels=1)
HEADS = {"surface": 2, "ink": 1, "winding": 8}


def _net(seed: int, **kw) -> TSMNet:
    torch.manual_seed(seed)
    return TSMNet(**{**SMALL, "heads": HEADS, "fullres_width": 8, **kw}).eval()


def _student_name(net) -> str:
    heads = student_head_channels(net)
    return student_engine_name(int(net.in_ch), sum(heads.values()), net.widths,
                               net.body_stride, fp=net_fingerprint(net))


def test_weight_fingerprint_tracks_the_weights():
    a, b = _net(0), _net(1)
    assert weight_fingerprint(a) == weight_fingerprint(_net(0))
    assert weight_fingerprint(a) != weight_fingerprint(b)
    assert arch_fingerprint(a) == arch_fingerprint(b)  # same architecture, different weights
    # a single perturbed parameter moves the fingerprint
    before = weight_fingerprint(a)
    with torch.no_grad():
        next(iter(a.parameters())).add_(1e-3)
    assert weight_fingerprint(a) != before


@pytest.mark.parametrize("kw", [
    {"norm": "batch"}, {"blocks": 2}, {"dec_blocks": 2}, {"bottleneck_blocks": 2},
    {"groups": 4}, {"ds_levels": 2}, {"fullres_width": 16}, {"body_stride": 2},
    {"aux_ch": 3}, {"widths": (8, 16, 64)}, {"heads": {"surface": 3, "ink": 1, "winding": 8}},
])
def test_arch_fingerprint_tracks_every_architecture_knob(kw):
    """Changes that used to be invisible to the cache name must change the fingerprint."""
    assert arch_fingerprint(_net(0, **kw)) != arch_fingerprint(_net(0))


def test_same_shape_different_weights_get_different_onnx_and_plan_paths(tmp_path, monkeypatch):
    pytest.importorskip("tensorrt")
    monkeypatch.setattr("tsm.trt._gpu_tag", lambda device=0: "FAKE_GPU_sm89")  # no CUDA on CI
    a, b, again = _net(0), _net(1), _net(0)
    na, nb, nc = _student_name(a), _student_name(b), _student_name(again)
    assert na == nc and na != nb
    assert na.startswith("student_i2o11_w8x16x32_bs1_f")

    d = str(tmp_path)
    pa, pb, pc = (TRTStudent.onnx_path(n, 1, "fp16", d) for n in (na, nb, nc))
    assert pa == pc and pa != pb
    ka, kb, kc = (engine_cache_key(n, 128, 1, "fp16", torch.device("cpu")) for n in (na, nb, nc))
    assert ka == kc and ka != kb

    # a fresh net with the *same* listed dimensions but a different architecture also differs
    assert _student_name(_net(0, norm="batch", fullres_width=16)) not in (na, nb)


def test_missing_plan_with_an_existing_onnx_does_not_reuse_the_wrong_export(tmp_path):
    """Deleting only the plan used to leave checkpoint A's ONNX in place for B."""
    onnx = pytest.importorskip("onnx")
    from tsm.trt import export_student_onnx

    a, b = _net(0), _net(1)
    d = str(tmp_path)
    pa = TRTStudent.onnx_path(_student_name(a), 1, "fp16", d)
    export_student_onnx(a, pa, in_ch=int(a.in_ch), batch=1, precision="fp16", patch=32)
    assert os.path.exists(pa)
    # B's build_or_load would look here; the path is different, so it must export its own
    pb = TRTStudent.onnx_path(_student_name(b), 1, "fp16", d)
    assert pb != pa and not os.path.exists(pb)
    export_student_onnx(b, pb, in_ch=int(b.in_ch), batch=1, precision="fp16", patch=32)
    from tsm.trt import onnx_fingerprint

    assert onnx_fingerprint(pa) != onnx_fingerprint(pb)
    onnx.checker.check_model(onnx.load(pb))
    # and the two nets really do compute different things
    with torch.no_grad():
        x = torch.randn(1, int(a.in_ch), 32, 32, 32)
        assert (a(x)["ink"] - b(x)["ink"]).abs().max() > 0


def test_teacher_cache_name_carries_the_model_and_calibration(tmp_path):
    tspec = TEACHERS["recto"]
    a, b = _net(0), _net(1)
    d = str(tmp_path)
    fa, fb = net_fingerprint(a), net_fingerprint(b)
    assert teacher_cache_name(tspec, net=a) != teacher_cache_name(tspec, net=b)
    assert teacher_cache_name(tspec, fp=fa).startswith(tspec.name + "_f")

    pa = TRTTeacher.onnx_path(tspec, 1, "fp16", d, 128, net=a)
    pb = TRTTeacher.onnx_path(tspec, 1, "fp16", d, 128, net=b)
    assert pa != pb
    assert TRTTeacher.onnx_path(tspec, 1, "fp16", d, 128, fp=fa) == pa
    # int8 calibration provenance is part of the name
    assert (TRTTeacher.onnx_path(tspec, 1, "int8", d, 128, fp=fa, calib_algorithm="max")
            != TRTTeacher.onnx_path(tspec, 1, "int8", d, 128, fp=fa, calib_algorithm="entropy"))
    # a legacy per-patch export is only reused for the *same* fingerprint
    legacy = os.path.join(d, f"{teacher_cache_name(tspec, fp=fa)}_p128_b1_fp16.onnx")
    open(legacy, "wb").close()
    assert TRTTeacher.onnx_path(tspec, 1, "fp16", d, 128, fp=fa) == legacy
    assert TRTTeacher.onnx_path(tspec, 1, "fp16", d, 128, fp=fb) != legacy
