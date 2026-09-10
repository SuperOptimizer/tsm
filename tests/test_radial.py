"""Radial input channels: dataset field, augmentation equivariance, infer/sliding plumbing."""

import itertools

import numpy as np
import pytest
import torch

from synth import AXIS_YX, make_synthetic, synthetic_axis
from tsm.data import Augment, CropDataset, LabelStore, SpatialParams, augment, rot90_matrix_inplane, spatial_transform
from tsm.infer import CLIP, StudentNet, tta_forward, tta_transforms
from tsm.labels import radial_field
from tsm.student import in_channels, input_channels, make_input
from tsm.volume import VolumeReader


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic(str(tmp_path_factory.mktemp("radial")))


def _analytic(origin, shape):
    """Outward unit radial (3, Z, Y, X) about AXIS_YX, computed independently of labels.py."""
    z, y, x = (np.arange(o, o + s) for o, s in zip(origin, shape))
    yy, xx = np.meshgrid(y, x, indexing="ij")
    dy, dx = yy - AXIS_YX[0], xx - AXIS_YX[1]
    r = np.hypot(dy, dx)
    r[r == 0] = 1.0
    f = np.stack([np.zeros_like(dy), dy / r, dx / r]).astype(np.float32)
    return np.repeat(f[:, None], len(z), axis=1)


def test_radial_field_matches_analytic_and_is_unit():
    axis = synthetic_axis()
    origin, shape = (16, 32, 32), (8, 12, 10)
    r = radial_field(axis, origin, shape, scale=1)
    assert r.shape == (3,) + shape
    assert np.abs(r - _analytic(origin, shape)).max() < 1e-5
    assert np.abs(np.linalg.norm(r, axis=0) - 1.0).max() < 1e-5
    assert np.abs(r[0]).max() == 0.0  # the axis is along z: no z component


# --------------------------------------------------------------------------- dataset
def _dataset(synth, **kw):
    fine = LabelStore(synth["fine"])
    return CropDataset(
        VolumeReader(synth["ct"], 0, 2.4), fine, LabelStore(synth["coarse"]), patch=32, length=4,
        stride=16, augment=False, input_radial=True, axis=synthetic_axis(), **kw)


def test_dataset_input_is_five_channels_with_the_outward_radial(synth):
    ds = _dataset(synth)
    o = ds.origins[0]
    it = ds.to_tensors(ds.load(o))
    assert it["input"].shape[0] == in_channels(True) == 5
    assert input_channels(True) == ["ct", "scale", "r_z", "r_y", "r_x"]
    g = tuple(int(a) + int(b) for a, b in zip(ds.fine.origin_zyx, o))
    ref = _analytic(g, (ds.patch,) * 3)
    assert np.abs(it["input"][2:5].numpy() - ref).max() < 1e-5
    # outward: r points away from the axis
    yy = np.arange(g[1], g[1] + ds.patch)[:, None] - AXIS_YX[0]
    assert np.sign(it["input"][3, 0].numpy()[yy[:, 0] != 0][:, 0]).tolist() == np.sign(yy[yy[:, 0] != 0][:, 0]).tolist()


def test_dataset_without_input_radial_is_two_channels(synth):
    fine = LabelStore(synth["fine"])
    ds = CropDataset(VolumeReader(synth["ct"], 0, 2.4), fine, None, patch=32, length=2, stride=16, augment=False)
    assert ds.to_tensors(ds.load(ds.origins[0]))["input"].shape[0] == 2


# --------------------------------------------------------------------------- augmentation v1
@pytest.mark.parametrize("fz,fy,fx,k", list(itertools.product([0, 1], [0, 1], [0, 1], [0, 1, 2, 3]))[::3])
def test_v1_augment_radial_round_trips_through_the_inverse(fz, fy, fx, k):
    """flip/rot90 then the inverse flip/rot90 recovers the analytic field exactly."""
    origin, shape = (16, 32, 32), (8, 16, 16)
    r = radial_field(synthetic_axis(), origin, shape, scale=1)
    f = {"radial": r[::-1].copy()}  # (x, y, z) order, as the vector rule wants
    t = spatial_transform(f, [bool(fz), bool(fy), bool(fx)], k, normal_key=["radial"])
    # inverse: undo the rot90 first, then the flips
    t = spatial_transform(t, [False] * 3, (4 - k) % 4, normal_key=["radial"])
    t = spatial_transform(t, [bool(fz), bool(fy), bool(fx)], 0, normal_key=["radial"])
    assert np.abs(t["radial"][::-1] - r).max() < 1e-6


def test_v1_augment_sample_moves_radial_like_the_normal():
    """The radial channels obey exactly the winding-normal rule (same field, same transform)."""
    origin, shape = (16, 32, 32), (8, 16, 16)
    r = radial_field(synthetic_axis(), origin, shape, scale=1)
    n = np.zeros((6,) + shape, np.float32)
    n[3:6] = r[::-1]  # winding normal channels are (nx, ny, nz)
    sample = {"ct": np.zeros((1,) + shape, np.float32), "winding": n, "radial": r.copy()}
    out = augment(sample, np.random.default_rng(3), spatial=True, intensity=False)
    assert np.abs(out["radial"] - out["winding"][3:6][::-1]).max() < 1e-6


# --------------------------------------------------------------------------- augmentation v2
def _params(**kw):
    return SpatialParams(**kw)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_v2_apply_spatial_radial_inverse_recovers_the_analytic_field(k):
    origin, shape = (16, 32, 32), (8, 16, 16)
    r = radial_field(synthetic_axis(), origin, shape, scale=1)
    ct = torch.rand(1, 1, *shape)
    inp = torch.cat([ct, torch.zeros_like(ct), torch.from_numpy(r)[None]], dim=1)
    aug = Augment("light", seed=0)
    p = _params(flips=(True, False, True), rot90=rot90_matrix_inplane(k))
    fwd = aug.apply_spatial({"input": inp}, [p])["input"]
    assert fwd.shape[1] == 5
    inv = _params(rotation=torch.linalg.inv(p.matrix()).float())
    back = aug.apply_spatial({"input": fwd}, [inv])["input"]
    assert torch.abs(back[0, 2:5] - torch.from_numpy(r)).max() < 1e-4
    assert torch.abs(back[0, 0:1] - ct[0]).max() < 1e-4


def test_v2_radial_stays_unit_length_under_a_random_rotation():
    origin, shape = (16, 32, 32), (8, 16, 16)
    r = torch.from_numpy(radial_field(synthetic_axis(), origin, shape, scale=1))[None]
    inp = torch.cat([torch.rand(1, 1, *shape), torch.zeros(1, 1, *shape), r], dim=1)
    aug = Augment("light", seed=1)
    p = _params(rotation=torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]))
    out = aug.apply_spatial({"input": inp}, [p])["input"]
    assert torch.abs(out[:, 2:5].norm(dim=1) - 1.0).max() < 1e-4


def test_v2_pipeline_keeps_five_channels():
    inp = torch.cat([torch.rand(2, 1, 16, 16, 16), torch.zeros(2, 1, 16, 16, 16),
                     torch.rand(2, 3, 16, 16, 16)], dim=1)
    out, _ = Augment("strong", seed=2)({"input": inp})
    assert out["input"].shape[1] == 5


# --------------------------------------------------------------------------- student / infer
def test_make_input_five_channels():
    ct = torch.rand(2, 8, 8, 8)
    rad = torch.rand(2, 3, 8, 8, 8)
    x = make_input(ct, 2.4, rad)
    assert x.shape[1] == 5 and torch.equal(x[:, 2:5], rad)
    with pytest.raises(ValueError):
        make_input(ct, 2.4, rad[:, :2])


class _Recorder(torch.nn.Module):
    """Records the input it was handed and returns zero heads."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = x.clone()
        b, _, *sp = x.shape
        return {"surface": torch.zeros(b, 2, *sp), "ink": torch.zeros(b, 1, *sp), "winding": torch.zeros(b, 8, *sp)}


def test_student_net_builds_a_five_channel_input_from_the_box_origin():
    rec = _Recorder()
    net = StudentNet(rec, 2.4, CLIP, tta="none", input_radial=True, axis=synthetic_axis())
    assert net.needs_box_origin is True
    origin, shape = (16, 32, 32), (8, 16, 16)
    x = torch.rand(1, 1, *shape)
    out = net(x, box_origin_zyx=[origin])
    assert out.shape[1] == 11
    assert rec.seen.shape[1] == 5
    assert torch.abs(rec.seen[0, 2:5] - torch.from_numpy(radial_field(synthetic_axis(), origin, shape, scale=1))).max() < 1e-5
    with pytest.raises(ValueError):
        net(x)


def test_student_net_without_radial_is_two_channels():
    rec = _Recorder()
    StudentNet(rec, 2.4, CLIP)(torch.rand(1, 1, 8, 8, 8))
    assert rec.seen.shape[1] == 2


@pytest.mark.parametrize("mode", ["flip8", "flip8_rot4"])
def test_tta_forward_moves_the_radial_channels_as_a_vector(mode):
    origin, shape = (16, 32, 32), (8, 16, 16)
    r = torch.from_numpy(radial_field(synthetic_axis(), origin, shape, scale=1))[None]
    inp = torch.cat([torch.rand(1, 2, *shape), r], dim=1)
    transforms = tta_transforms(mode)
    assert len(transforms) == (16 if mode == "flip8_rot4" else 8)
    for axes, tr in transforms:
        t = tta_forward(inp, axes, tr, vec=(2, 5))
        assert torch.abs(t[:, 2:5].norm(dim=1) - 1.0).max() < 1e-5
        # independent oracle: the grid moves (flips, then the y<->x transpose) and the components are
        # mapped by the matching signed permutation matrix M = P @ F (flip first, then swap y/x).
        ref = torch.flip(r, [a + 2 for a in axes]) if axes else r
        if tr:
            ref = ref.transpose(3, 4)
        F = torch.eye(3)
        for a in axes:
            F[a, a] = -1.0
        P = torch.eye(3)
        if tr:
            P = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        M = P @ F
        want = torch.einsum("ij,bjzyx->bizyx", M, ref)
        assert torch.abs(t[:, 2:5] - want).max() < 1e-5
        # the untouched channels only move with the grid
        other = torch.flip(inp[:, :2], [a + 2 for a in axes]) if axes else inp[:, :2]
        if tr:
            other = other.transpose(3, 4)
        assert torch.equal(t[:, :2], other)


# --------------------------------------------------------------------------- sliding
def test_predict_box_passes_absolute_window_origins():
    from tsm.sliding import WindowSpec, predict_box

    seen = []

    class _Net(torch.nn.Module):
        needs_box_origin = True

        def forward(self, x, box_origin_zyx=None):
            seen.extend(box_origin_zyx)
            return torch.zeros(x.shape[0], 1, *x.shape[2:])

    box = np.full((16, 16, 16), 200, np.uint8)
    spec = WindowSpec(patch=8, step=8, out_tile=16, halo=0, batch=1, tta=False, dtype=torch.float32)
    out, n = predict_box(box, _Net(), lambda t: t.float(), spec, 1, "sigmoid", None,
                         torch.device("cpu"), box_origin_zyx=(100, 200, 300))
    assert n == 8 and out.shape == (1, 16, 16, 16)
    assert (100, 200, 300) in seen and (108, 208, 308) in seen and len(seen) == 8


def test_tta_forward_inverse_round_trips_a_vector_field():
    """forward(vec) then inverse(vec-as-normal) is the identity for all 16 transforms."""
    import torch as _t

    from tsm.infer import tta_inverse

    r = _t.randn(1, 3, 4, 6, 6)
    for axes, tr in tta_transforms("flip8_rot4"):
        # a "winding" output carries the normal in channels 3:6 as (nx, ny, nz) = reversed (z, y, x)
        fwd = tta_forward(_t.cat([_t.zeros(1, 2, 4, 6, 6), r], dim=1), axes, tr, vec=(2, 5))
        n = fwd[:, 2:5].flip(1)  # (z,y,x) -> (nx,ny,nz)
        back = tta_inverse({"winding": _t.cat([_t.zeros(1, 3, *n.shape[2:]), n], dim=1)}, axes, tr)["winding"]
        assert _t.abs(back[:, 3:6].flip(1) - r).max() < 1e-5, (axes, tr)
