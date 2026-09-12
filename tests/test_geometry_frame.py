"""The exact geometry frame: local umbilicus tangent + perpendicular radial (plan section 1).

``labels.geometry_frame`` replaces the two hard-coded fields the student used to be fed -- a
constant ``(1, 0, 0)`` scroll axis and a radial with ``r_z == 0`` -- with the real pair, and
the dataset, the augmentation and ``infer.StudentNet`` all have to agree on it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from synth import AXIS_YX, make_synthetic, synthetic_axis, synthetic_axis_tilted
from tsm import equivariance as eq
from tsm.data import Augment, CropDataset, LabelStore, SpatialParams, uniform_rotation_matrix
from tsm.infer import CLIP, StudentNet, checkpoint_axis_tangent, student_build_kw
from tsm.labels import axis_tangent_at, axis_tangent_field, geometry_frame, radial_field
from tsm.student import build_model, in_channels, input_vec_slices, load_student_state, make_input
from tsm.train import TRAIN_DEFAULTS
from tsm.volume import VolumeReader

from tests.test_fiber import cube_rotations

TILT = (0.3, -0.2)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
# 1. the tangent and the frame
# --------------------------------------------------------------------------- #
def test_tangent_of_a_straight_tilted_axis_is_the_analytic_direction():
    ax = synthetic_axis_tilted(*TILT)
    want = _unit((1.0, TILT[0], TILT[1]))
    for z in (-500.0, 0.0, 1234.0):
        assert np.abs(axis_tangent_at(ax, z) - want).max() < 1e-6
    t = axis_tangent_at(ax, np.array([0.0, 10.0, 20.0]))
    assert t.shape == (3, 3) and np.abs(t - want[:, None]).max() < 1e-6
    # the field is that tangent at every voxel of the box (it depends on z only)
    f = axis_tangent_field(ax, (7, 30, 40), (4, 5, 6))
    assert f.shape == (3, 4, 5, 6) and np.abs(f - want[:, None, None, None]).max() < 1e-6
    # a straight z axis is exactly (1, 0, 0)
    assert np.abs(axis_tangent_field(synthetic_axis(), (0, 0, 0), (2, 2, 2))
                  - np.array([1.0, 0.0, 0.0])[:, None, None, None]).max() == 0.0


def test_frame_is_orthonormal_and_the_radial_is_perpendicular_to_the_axis():
    ax = synthetic_axis_tilted(*TILT)
    fr = geometry_frame(ax, (-20, 60, 70), (16, 24, 20))
    r, a = fr["radial"], fr["axis_dir"]
    assert r.shape == a.shape == (3, 16, 24, 20) and r.dtype == a.dtype == np.float32
    assert np.abs(np.linalg.norm(r, axis=0) - 1.0).max() < 1e-5
    assert np.abs(np.linalg.norm(a, axis=0) - 1.0).max() < 1e-5
    assert np.abs((r * a).sum(0)).max() < 1e-5          # r . a == 0
    assert np.abs(r[0]).max() > 0.01                    # ... so the radial has a z component now
    # the radial still points away from the axis: its in-plane part agrees with (y, x) - axis
    z, y, x = np.meshgrid(*[np.arange(o, o + s) for o, s in zip((-20, 60, 70), (16, 24, 20))], indexing="ij")
    ay = AXIS_YX[0] + TILT[0] * z
    axx = AXIS_YX[1] + TILT[1] * z
    assert (r[1] * (y - ay) + r[2] * (x - axx)).min() > 0.0


def test_the_straight_z_axis_path_is_bit_identical_to_the_legacy_radial():
    ax = synthetic_axis()
    origin, shape = (16, 32, 32), (8, 12, 10)
    legacy = radial_field(ax, origin, shape, scale=1)
    assert np.array_equal(radial_field(ax, origin, shape, scale=1, tangent=None), legacy)
    # the tangent of a straight z axis is exactly (1, 0, 0), so the frame reproduces it bit for bit
    assert np.array_equal(geometry_frame(ax, origin, shape, scale=1)["radial"], legacy)
    assert np.abs(legacy[0]).max() == 0.0
    for k in (2, 4):  # and on a coarse grid
        assert np.array_equal(geometry_frame(ax, origin, shape, scale=k)["radial"],
                              radial_field(ax, origin, shape, scale=k))


def test_a_tangent_makes_the_radial_differ_from_the_in_plane_one():
    ax = synthetic_axis_tilted(*TILT)
    origin, shape = (-30, 40, 50), (8, 12, 10)
    r_new = geometry_frame(ax, origin, shape)["radial"]
    r_old = radial_field(ax, origin, shape)
    assert np.abs(r_new - r_old).max() > 0.05  # the whole point of the change
    assert np.abs(r_old[0]).max() == 0.0


# --------------------------------------------------------------------------- #
# 2. equivariance of the frame channels
# --------------------------------------------------------------------------- #
def _frame_input(origin=(5, 88, 96), P=12, tilt=TILT):
    fr = geometry_frame(synthetic_axis_tilted(*tilt), origin, (P,) * 3)
    r = torch.from_numpy(fr["radial"])[None]
    a = torch.from_numpy(fr["axis_dir"])[None]
    return fr, torch.cat([torch.zeros(1, 2, P, P, P), r, a], dim=1)


def _aug():
    return Augment("none", seed=0, input_vec=input_vec_slices(True, True))


def test_frame_channels_are_equivariant_under_all_24_cube_rotations():
    fr, inp = _frame_input()
    aug = _aug()
    assert input_vec_slices(True, True) == [(2, 5), (5, 8)]
    for m in cube_rotations():
        out = aug.apply_spatial({"input": inp.clone()}, [SpatialParams(rot90=m)])["input"]
        g = np.round(m.numpy()).astype(np.int64)
        for lo, hi, key in ((2, 5, "radial"), (5, 8, "axis_dir")):
            want = torch.from_numpy(np.asarray(eq.apply_vector(fr[key], g))).float()[None]
            torch.testing.assert_close(out[:, lo:hi], want, atol=1e-5, rtol=0)
        # and the pair is still orthonormal after the transport
        r, a = out[:, 2:5], out[:, 5:8]
        assert float((r * a).sum(1).abs().max()) < 1e-5
        assert float((r.norm(dim=1) - 1).abs().max()) < 1e-5


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_frame_channels_are_equivariant_under_random_so3(seed):
    fr, inp = _frame_input(P=16)
    R = uniform_rotation_matrix(torch.Generator().manual_seed(seed))
    out = _aug().apply_spatial({"input": inp.clone()}, [SpatialParams(rotation=R)])["input"]
    Rn = R.numpy().astype(np.float64)
    for lo, hi, key in ((2, 5, "radial"), (5, 8, "axis_dir")):
        want = torch.from_numpy(np.asarray(eq.apply_vector(fr[key], Rn))).float()[None]
        want = torch.nn.functional.normalize(want, dim=1)
        cos = (out[:, lo:hi] * want).sum(1)[:, 4:-4, 4:-4, 4:-4]  # drop the rim (padding differs)
        assert float(cos.min()) > 0.999, f"{key}: worst angle {np.degrees(np.arccos(float(cos.min()))):.2f} deg"
    r, a = out[:, 2:5], out[:, 5:8]
    assert float((r * a).sum(1).abs().max()) < 1e-5  # re-orthonormalised after the resampling
    assert float((a.norm(dim=1) - 1).abs().max()) < 1e-5


def test_elastic_and_scale_leave_the_frame_orthonormal():
    """Trilinear resampling and an anisotropic scale both break ``r . a == 0``;
    ``data._orthonormalise_frame`` is what puts it back."""
    _, inp = _frame_input(P=16)
    gen = torch.Generator().manual_seed(0)
    p = SpatialParams(rotation=uniform_rotation_matrix(gen), scale=torch.tensor([1.3, 0.8, 1.1]),
                      elastic=torch.randn(3, 4, 4, 4, generator=gen) * 2.0)
    out = _aug().apply_spatial({"input": inp.clone()}, [p])["input"]
    r, a = out[:, 2:5], out[:, 5:8]
    assert float((r * a).sum(1).abs().max()) < 1e-5
    assert float((r.norm(dim=1) - 1).abs().max()) < 1e-5
    assert float((a.norm(dim=1) - 1).abs().max()) < 1e-5


# --------------------------------------------------------------------------- #
# 3. the dataset and the inference wrapper build the same frame
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic(str(tmp_path_factory.mktemp("frame")))


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = x.clone()
        b, _, *sp = x.shape
        return {"surface": torch.zeros(b, 2, *sp), "ink": torch.zeros(b, 1, *sp),
                "winding": torch.zeros(b, 8, *sp)}


def test_student_net_frame_equals_the_dataset_channels_at_the_same_global_origin(synth):
    ax = synthetic_axis_tilted(*TILT)
    fine = LabelStore(synth["fine"])
    ds = CropDataset(VolumeReader(synth["ct"], 0, 2.4), fine, LabelStore(synth["coarse"]), patch=16,
                     length=2, stride=16, augment=False, input_radial=True, input_axis=True, axis=ax)
    assert ds.axis_tangent is True
    o = ds.origins[0]
    item = ds.to_tensors(ds.load(o))
    assert item["input"].shape[0] == in_channels(True, True) == 8
    g = tuple(int(a) + int(b) for a, b in zip(fine.origin_zyx, o))

    rec = _Recorder()
    net = StudentNet(rec, 2.4, CLIP, tta="none", input_radial=True, input_axis=True, axis=ax)
    assert net.needs_box_origin is True
    x = torch.rand(1, 1, 16, 16, 16)
    net(x, box_origin_zyx=[g])
    torch.testing.assert_close(rec.seen[0, 2:8], item["input"][2:8], atol=1e-6, rtol=0)
    fr = net.frame(x, [g])
    torch.testing.assert_close(fr["radial"][0], item["input"][2:5], atol=1e-6, rtol=0)
    torch.testing.assert_close(fr["axis_dir"][0], item["input"][5:8], atol=1e-6, rtol=0)
    torch.testing.assert_close(net.radial(x, [g]), fr["radial"])  # the alias still works
    # make_input builds the same 8-channel stack
    mi = make_input(x[:, 0], 2.4, fr["radial"], fr["axis_dir"])
    torch.testing.assert_close(mi[:, 2:8], rec.seen[:, 2:8], atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        make_input(x[:, 0], 2.4, None, fr["axis_dir"])


def test_axis_tangent_false_reproduces_the_legacy_constant_fields(synth):
    ax = synthetic_axis_tilted(*TILT)
    fine = LabelStore(synth["fine"])
    kw = dict(patch=16, length=2, stride=16, augment=False, input_radial=True, input_axis=True, axis=ax)
    old = CropDataset(VolumeReader(synth["ct"], 0, 2.4), fine, None, axis_tangent=False, **kw)
    new = CropDataset(VolumeReader(synth["ct"], 0, 2.4), fine, None, **kw)
    o = old.origins[0]
    a_old = old.load(o)["axis_dir"]
    r_old = old.load(o)["radial"]
    g = tuple(int(p) + int(q) for p, q in zip(fine.origin_zyx, o))
    assert np.array_equal(a_old, np.broadcast_to(np.array([1.0, 0.0, 0.0], np.float32)[:, None, None, None],
                                                 a_old.shape))
    assert np.array_equal(r_old, radial_field(ax, g, (16,) * 3, scale=1))
    assert np.abs(new.load(o)["radial"] - r_old).max() > 0.05
    # ... and the inference wrapper has the same switch
    rec = _Recorder()
    net = StudentNet(rec, 2.4, CLIP, tta="none", input_radial=True, input_axis=True, axis=ax,
                     axis_tangent=False)
    fr = net.frame(torch.rand(1, 1, 16, 16, 16), [g])
    assert np.abs(fr["radial"][0].numpy() - r_old).max() < 1e-6
    assert np.abs(fr["axis_dir"][0].numpy() - a_old).max() < 1e-6


def test_multi_store_datasets_must_agree_on_axis_tangent(synth):
    from tsm.data import MultiStoreDataset

    fine = LabelStore(synth["fine"])
    kw = dict(patch=16, length=2, stride=16, augment=False, input_radial=True, axis=synthetic_axis())
    a = CropDataset(VolumeReader(synth["ct"], 0, 2.4), fine, None, **kw)
    b = CropDataset(VolumeReader(synth["ct"], 0, 2.4), fine, None, axis_tangent=False, **kw)
    MultiStoreDataset([a, a], names=["p", "q"], length=2)  # compatible
    with pytest.raises(ValueError, match="axis_tangent"):
        MultiStoreDataset([a, b], names=["p", "q"], length=2)


# --------------------------------------------------------------------------- #
# 4. defaults and old checkpoints
# --------------------------------------------------------------------------- #
def test_geometry_defaults_and_old_checkpoint_blobs():
    assert TRAIN_DEFAULTS["input_radial"] is True
    assert TRAIN_DEFAULTS["input_axis"] is True      # was False before 2026-09-12
    assert TRAIN_DEFAULTS["axis_tangent"] is True
    # a checkpoint from before either key existed: 2 channels, constant axis
    old = {"surface_mode": "faces", "widths": [8, 16, 32]}
    assert student_build_kw(old)["in_ch"] == 2
    assert checkpoint_axis_tangent(old) is False
    assert checkpoint_axis_tangent({"train": {"axis_tangent": True}}) is True
    assert checkpoint_axis_tangent({"axis_tangent": True}) is True
    net2 = build_model(**student_build_kw(old))
    load_student_state(net2, build_model(widths=(8, 16, 32), in_ch=2, surface_mode="faces").state_dict(),
                       what="old checkpoint")
    # ... and one from the input_radial era: 5 channels, still no axis input
    five = {"surface_mode": "faces", "widths": [8, 16, 32], "train": {"input_radial": True}}
    assert student_build_kw(five)["in_ch"] == 5 and checkpoint_axis_tangent(five) is False
    net5 = build_model(**student_build_kw(five))
    load_student_state(net5, build_model(widths=(8, 16, 32), in_ch=5, surface_mode="faces").state_dict(),
                       what="old checkpoint")
    # today's blob is 8 channels
    now = {"surface_mode": "faces", "widths": [8, 16, 32],
           "train": {"input_radial": True, "input_axis": True, "axis_tangent": True}}
    assert student_build_kw(now)["in_ch"] == 8 and checkpoint_axis_tangent(now) is True


def _cfg(tmp_path, train):
    from tsm.config import parse_config

    return parse_config({
        "volume": {"url": "x", "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [0, 0, 0], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": str(tmp_path), "extra": {"train": train},
    })


def test_train_opts_carry_the_geometry_flags(tmp_path):
    from tsm.train import train_opts

    o = train_opts(_cfg(tmp_path, {"steps": 1}))
    assert o["input_axis"] is True and o["axis_tangent"] is True and o["fiber_mode"] == "direction"
    o2 = train_opts(_cfg(tmp_path, {"steps": 1, "axis_tangent": False, "input_axis": False}))
    assert o2["axis_tangent"] is False and o2["input_axis"] is False
