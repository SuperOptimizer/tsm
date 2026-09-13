"""Student speed work (docs/student_v2_plan.md A1 / A5 / A6), CPU only.

A1  half-resolution body (``TSMNet(body_stride=2)``) + full-resolution block + norm option
A5  receptive-field halo tiling (``TSMNet.receptive_field_radius``, ``extra.infer.halo = "rf"``,
    uniform core blending in :mod:`tsm.sliding`)
A6  TensorRT export path for the student (ONNX export only here; the engine build is GPU-only,
    see ``dev/bench_student.py``)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from tsm.infer import rf_tiling, student_build_kw, student_rf_radius
from tsm.sliding import WindowSpec, predict_box, uniform_core_weight, window_weight
from tsm.student import HEADS, TSMNet, build_model, conv_macs

SMALL = dict(in_ch=2, widths=(8, 16, 32), blocks=1, dec_blocks=1, bottleneck_blocks=1, ds_levels=1)


# --------------------------------------------------------------------------- #
# A1: body_stride, full-resolution block, norm option
# --------------------------------------------------------------------------- #
@torch.no_grad()
def test_body_stride2_shapes_and_deep_supervision():
    net = TSMNet(in_ch=5, body_stride=2, fullres_width=32)
    assert net.divisor == 32  # 4 encoder halvings + the stride-2 stem (16 at body_stride 1)
    x = torch.randn(1, 5, 64, 64, 64)
    net.train()
    out = net(x)
    for name, c in HEADS.items():
        assert len(out[name]) == 2
        assert tuple(out[name][0].shape) == (1, c, 64, 64, 64)  # full res, from the full-res block
        assert tuple(out[name][1].shape) == (1, c, 32, 32, 32)  # decoder level 0 = half res
    net.eval()
    assert tuple(net(x)["surface"].shape) == (1, 2, 64, 64, 64)
    # the full-res block feeds level 0, the decoder levels 1..; heads are 1x1x1 convs
    assert net.level_width(0) == 32 and net.level_width(1) == net.widths[0]
    assert net.head["surface"][0].in_channels == 32
    assert all(h.kernel_size == (1, 1, 1) for hs in net.head.values() for h in hs)
    # the full-res block sees the network input concatenated with the upsampled decoder output
    assert net.fullres.conv1.in_channels == 5 + net.widths[0]
    with pytest.raises(ValueError):
        net(torch.randn(1, 5, 48, 48, 48))  # 48 is not a multiple of the divisor


@torch.no_grad()
def test_body_stride2_ds_levels_and_fullres_width():
    net = TSMNet(**{**SMALL, "body_stride": 2, "fullres_width": 16, "ds_levels": 3})
    assert net.n_levels == 3  # 2 decoder levels + the full-res level
    net.train()
    out = net(torch.randn(1, 2, 32, 32, 32))
    assert [tuple(t.shape[2:]) for t in out["ink"]] == [(32,) * 3, (16,) * 3, (8,) * 3]
    assert net.level_width(0) == 16
    with pytest.raises(ValueError):
        TSMNet(**{**SMALL, "body_stride": 2, "ds_levels": 4})
    with pytest.raises(ValueError):
        TSMNet(**{**SMALL, "body_stride": 3})


def test_body_stride2_macs_and_params():
    """~3.3x fewer MACs at 128^3 for ~the same parameter count (the body is unchanged)."""
    p, x = 128, torch.zeros(1, 5, 128, 128, 128)
    m1 = conv_macs(TSMNet(in_ch=5, body_stride=1).eval(), x)
    m2 = conv_macs(TSMNet(in_ch=5, body_stride=2).eval(), x)
    assert m1 / m2 > 3.0, (m1, m2)
    n1 = TSMNet(in_ch=5, body_stride=1).num_params()
    n2 = TSMNet(in_ch=5, body_stride=2).num_params()
    assert abs(n2 - n1) / n1 < 0.02  # only the full-res block and the level-0 head differ
    # sanity of the counter itself: one conv, known MACs
    conv = nn.Conv3d(2, 4, 3, padding=1, bias=False)
    assert conv_macs(conv, torch.zeros(1, 2, 8, 8, 8)) == 8 ** 3 * 4 * 2 * 27


def test_norm_option_group_or_batch():
    g = TSMNet(**SMALL)
    b = TSMNet(**{**SMALL, "norm": "batch"})
    assert any(isinstance(m, nn.GroupNorm) for m in g.modules())
    assert not any(isinstance(m, nn.BatchNorm3d) for m in g.modules())
    assert any(isinstance(m, nn.BatchNorm3d) for m in b.modules())
    assert not any(isinstance(m, nn.GroupNorm) for m in b.modules())
    b.eval()
    with torch.no_grad():
        assert tuple(b(torch.randn(1, 2, 32, 32, 32))["surface"].shape) == (1, 2, 32, 32, 32)
    with pytest.raises(ValueError):
        TSMNet(**{**SMALL, "norm": "layer"})
    # build_model passes the new options through (what train.py hands over)
    net = build_model(widths=(8, 16), in_ch=2, body_stride=2, fullres_width=16, norm="batch",
                      blocks=1, dec_blocks=1, bottleneck_blocks=1, ds_levels=1)
    assert net.body_stride == 2 and net.fullres_width == 16 and net.norm_kind == "batch"


# --------------------------------------------------------------------------- #
# A5: receptive field
# --------------------------------------------------------------------------- #
def test_receptive_field_analytic_matches_empirical():
    """The gradient support of one output voxel is exactly the analytic radius (norms bypassed)."""
    for kw in (SMALL, {**SMALL, "blocks": 2, "dec_blocks": 2}):
        net = TSMNet(**kw)
        assert net.receptive_field_radius("empirical", size=96) == net.receptive_field_radius(), kw
    # with the stride-2 body the trilinear upsample makes the analytic bound conservative by 1
    net = TSMNet(**{**SMALL, "body_stride": 2, "fullres_width": 8})
    assert net.receptive_field_radius() - net.receptive_field_radius("empirical", size=96) in (0, 1)
    with pytest.raises(ValueError):
        TSMNet(**SMALL).receptive_field_radius("guess")
    with pytest.raises(ValueError, match="reaches the border"):  # too small a measurement volume
        TSMNet(**SMALL).receptive_field_radius("empirical", size=32)


def test_receptive_field_sanity():
    r1 = TSMNet().receptive_field_radius()
    r2 = TSMNet(body_stride=2).receptive_field_radius()
    assert r1 == 122 and r2 == 248  # canonical widths (32, 64, 128, 256, 256), blocks 2 / dec 1
    assert 1.9 < r2 / r1 < 2.1  # the stride-2 stem doubles every downstream jump
    assert TSMNet(widths=(8, 16), ds_levels=1).receptive_field_radius() < r1  # shallower -> smaller
    # deeper / wider kernels grow it; the input channel count does not
    assert TSMNet(in_ch=5).receptive_field_radius() == r1
    assert TSMNet(blocks=3).receptive_field_radius() > r1


def test_rf_tiling_geometry():
    halo, step = rf_tiling(384, 122)
    assert halo == 124 and step == 384 - 2 * 124
    # rf tiling beats 50 % overlap (8 windows per voxel) only once step > patch/2, i.e.
    # patch > 4*halo: with the canonical student's radius that is patch >= 512, not 384.
    assert (384 / step) ** 3 > 8
    assert (512 / rf_tiling(512, 122)[1]) ** 3 < 8
    assert (768 / rf_tiling(768, 122)[1]) ** 3 < 3.5
    with pytest.raises(ValueError) as e:  # the canonical student does not fit a 128 window
        rf_tiling(128, 122)
    assert "patch > 248" in str(e.value)


def test_uniform_core_weight():
    w = uniform_core_weight(64, 8)
    assert w.shape == (1, 1, 64, 64, 64) and float(w.sum()) == 48 ** 3
    assert w[0, 0, 8:56, 8:56, 8:56].min() == 1.0 and w[0, 0, 7].max() == 0.0
    assert uniform_core_weight(64, 8) is w  # cached
    spec = WindowSpec(patch=64, step=48, halo=8, weight="uniform")
    assert torch.equal(window_weight(spec), w)
    assert window_weight(WindowSpec(patch=64, halo=8)).min() > 0  # gaussian default
    with pytest.raises(ValueError):
        WindowSpec(patch=64, halo=32, weight="uniform")  # 2*halo >= patch
    with pytest.raises(ValueError):
        WindowSpec(patch=64, halo=8, weight="cosine")
    with pytest.raises(ValueError, match="uncovered"):  # cores do not touch: step > patch - 2*halo
        WindowSpec(patch=8, step=8, halo=1, weight="uniform")
    with pytest.raises(ValueError, match="uncovered"):
        WindowSpec(patch=64, step=49, halo=8, weight="uniform")
    WindowSpec(patch=64, step=48, halo=8, weight="uniform")  # exactly touching cores is fine


def test_uniform_weight_never_leaves_a_hole_in_the_output_core():
    """R13: patch 8 / step 8 / halo 1 used to construct and then leave 1016 of 2744 core voxels 0."""
    import numpy as np

    with pytest.raises(ValueError, match="uncovered"):
        WindowSpec(patch=8, step=8, halo=1, weight="uniform")

    # and the accumulate path refuses a hole even if a spec sneaks past construction
    spec = WindowSpec(patch=8, step=6, halo=1, weight="uniform", batch=1, out_tile=16, dtype=torch.float32)
    spec.step = 8  # bypass the constructor check
    box = np.full((16, 16, 16), 200, np.uint8)
    net = lambda t: torch.ones(t.shape[0], 1, *t.shape[2:])  # noqa: E731
    with pytest.raises(ValueError, match="uncovered|zero accumulated weight"):
        predict_box(box, net, lambda t: t.float() / 255.0, spec, 1, "none", None, torch.device("cpu"),
                    core_offset=(1, 1, 1), core_shape=(14, 14, 14))

    # the legal tiling covers everything: an all-ones net gives an all-255 core
    good = WindowSpec(patch=8, step=6, halo=1, weight="uniform", batch=1, out_tile=16, dtype=torch.float32)
    out, _ = predict_box(box, net, lambda t: t.float() / 255.0, good, 1, "none", None, torch.device("cpu"),
                         core_offset=(1, 1, 1), core_shape=(14, 14, 14))
    assert out.shape == (1, 14, 14, 14) and int(out.min()) == 255


# --------------------------------------------------------------------------- #
# A5: rf-halo tiling is exact for an exactly-local net
# --------------------------------------------------------------------------- #
class LocalNet(nn.Module):
    """One fixed conv of radius ``r``: output voxel v depends exactly on the input ball of radius r."""

    def __init__(self, r: int = 3, cout: int = 1, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv3d(1, cout, 2 * r + 1, padding=r, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g) * 0.3)
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_rf_halo_tiling_matches_the_whole_volume_exactly():
    """With halo >= rf radius and uniform core weights, tiled windows reproduce one big pass."""
    r, core = 3, 64
    halo = r + 2
    box = np.random.default_rng(0).integers(1, 256, size=(core + 2 * halo,) * 3, dtype=np.uint8)
    net = LocalNet(r)
    ident = lambda t: t.float()  # noqa: E731  (locality: no window/box normalisation)
    common = dict(net=net, normalizer=ident, channels_out=1, activation="sigmoid", fg_channel=None,
                  device=torch.device("cpu"), core_offset=(halo,) * 3, core_shape=(core,) * 3)

    p = box.shape[0]  # one window over the whole box: the reference
    ref, nref = predict_box(box, spec=WindowSpec(patch=p, step=p - 2 * halo, halo=halo, weight="uniform",
                                                 dtype=torch.float32, norm_scope="box"), **common)
    assert nref == 1

    for patch in (32, 48):
        halo_, step = rf_tiling(patch, r)
        assert halo_ == halo
        spec = WindowSpec(patch=patch, step=step, halo=halo, weight="uniform", dtype=torch.float32,
                          norm_scope="box")
        got, n = predict_box(box, spec=spec, **common)
        assert n >= 8  # really tiled (2+ windows per axis)
        np.testing.assert_array_equal(got, ref)

    # ... and the halo is what makes it exact: too small a halo breaks the seams
    spec = WindowSpec(patch=32, step=32 - 2, halo=1, weight="uniform", dtype=torch.float32, norm_scope="box")
    bad, _ = predict_box(box, spec=spec, **{**common, "core_offset": (1,) * 3,
                                            "core_shape": (box.shape[0] - 2,) * 3})
    inner = bad[:, halo - 1:halo - 1 + core, halo - 1:halo - 1 + core, halo - 1:halo - 1 + core]
    assert int(np.abs(inner.astype(int) - ref.astype(int)).max()) > 1


# --------------------------------------------------------------------------- #
# A5: extra.infer.halo = "rf"
# --------------------------------------------------------------------------- #
def test_student_rf_radius_and_build_kw(tmp_path):
    from tsm.train import EMA, save_checkpoint

    widths = (8, 16, 16)
    net = TSMNet(widths=widths)
    ckpt = str(tmp_path / "latest.pt")
    save_checkpoint(ckpt, net, EMA(net, 0.5), torch.optim.AdamW(net.parameters(), lr=1e-3), None, 3,
                    {"widths": list(widths), "train": {"widths": list(widths)}})
    assert student_rf_radius(ckpt) == net.receptive_field_radius()
    kw = student_build_kw({"widths": list(widths), "body_stride": 2, "fullres_width": 16, "norm": "batch"})
    assert kw == {"widths": widths, "surface_mode": "medial", "in_ch": 2, "body_stride": 2,
                  "fullres_width": 16, "norm": "batch", "fiber": False, "fiber_mode": "class",
                  "gap_class": False}
    # a checkpoint from before these options existed builds the full-resolution GroupNorm student
    old = student_build_kw({"train": {"widths": list(widths), "input_radial": True}})
    assert old["body_stride"] == 1 and old["norm"] == "group" and old["in_ch"] == 5


def test_run_infer_rf_halo(tmp_path):
    from synth import make_synthetic
    from test_infer import _cfg, _checkpoint
    from tsm.infer import run_infer

    root = str(tmp_path / "out")
    s = make_synthetic(root)
    ckpt = os.path.join(root, "train", "latest.pt")
    _checkpoint(ckpt, widths=(8, 16, 16))
    rf = student_rf_radius(ckpt)
    assert rf == 26
    cfg = _cfg(root, s["ct"], {"patch": 64, "halo": "rf", "out_tile": 32, "chunk": 32, "device": "cpu",
                               "stats_box": [32, 32, 32], "prefetch": False, "previews": False})
    cfg.region.size_zyx = (32, 32, 32)
    d = run_infer(cfg, dry_run=True)
    assert d["tiles"] == 1
    summary = run_infer(cfg)
    assert summary["done"] == 1
    sj = json.load(open(os.path.join(root, "student", "pred.summary.json")))
    assert sj["student"]["rf_radius"] == rf
    a = np.asarray(__import__("zarr").open_array(store=summary["pred"], mode="r")[:])
    assert (a[0] != 0).mean() > 0.8  # every core voxel covered
    # the halo has to fit twice into the patch
    with pytest.raises(ValueError):
        run_infer(_cfg(root, s["ct"], {"patch": 32, "halo": "rf", "device": "cpu"}), dry_run=True)


# --------------------------------------------------------------------------- #
# A6: TensorRT export path (ONNX only on the CPU; the engine build is GPU-only)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("body_stride,in_ch,surface_ch", [(1, 5, 2), (2, 5, 3), (1, 2, 2)])
def test_student_onnx_export_loads_and_validates(tmp_path, body_stride, in_ch, surface_ch):
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper

    from tsm.trt import export_student_onnx, student_engine_name, student_head_channels

    heads = {"surface": surface_ch, "ink": 1, "winding": 8}
    net = TSMNet(**{**SMALL, "in_ch": in_ch, "heads": heads, "body_stride": body_stride,
                    "fullres_width": 8}).eval()
    assert student_head_channels(net) == heads
    path = export_student_onnx(net, str(tmp_path / "student.onnx"), batch=1, precision="fp16", patch=64)
    model = onnx.load(path)
    onnx.checker.check_model(model)
    dims = lambda vi: [d.dim_value for d in vi.type.tensor_type.shape.dim]  # noqa: E731
    # the spatial dims are symbolic (dim_value 0): one export serves every patch
    assert dims(model.graph.input[0]) == [1, in_ch, 0, 0, 0]
    assert dims(model.graph.output[0]) == [1, surface_ch + 9, 0, 0, 0]
    # the graph is shape agnostic: any upsampling is by scales, so it re-dims for the build
    for node in model.graph.node:
        assert node.op_type != "Resize" or node.input[2], "Resize must carry scales, not sizes"
    # ... and no Reshape target carries baked spatial dims: the GroupNorm decomposition
    # (reshape to groups -> InstanceNorm -> reshape back) used to freeze the export window
    # into a [1, C, p, p, p] constant, which TensorRT rejected at any other patch.
    consts = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    baked = [n.name for n in model.graph.node
             if n.op_type == "Reshape" and len(consts.get(n.input[1], ())) == 5]
    assert not baked, f"Reshape with a baked 5-d target shape: {baked}"
    from tsm.trt import _redim_onnx

    redim = onnx.load_model_from_string(_redim_onnx(path, 192, 2))
    assert dims(redim.graph.input[0]) == [2, in_ch, 192, 192, 192]
    assert dims(redim.graph.output[0]) == [2, surface_ch + 9, 192, 192, 192]
    onnx.checker.check_model(redim, full_check=True)  # shape inference at the build patch
    assert student_engine_name(in_ch, surface_ch + 9, net.widths, body_stride).startswith(
        f"student_i{in_ch}o{surface_ch + 9}_w8x16x32_bs{body_stride}")


def test_trt_student_wrap_splits_heads():
    """``TRTStudent._wrap`` turns the engine's one output back into the TSMNet head dict."""
    from tsm.trt import TRTStudent

    obj = object.__new__(TRTStudent)
    obj.heads = {"surface": 2, "ink": 1, "winding": 8}
    y = torch.arange(11 * 8, dtype=torch.float32).reshape(1, 11, 2, 2, 2)
    out = TRTStudent._wrap(obj, y)
    assert [tuple(v.shape) for v in out.values()] == [(1, 2, 2, 2, 2), (1, 1, 2, 2, 2), (1, 8, 2, 2, 2)]
    assert torch.equal(torch.cat(list(out.values()), dim=1), y)
