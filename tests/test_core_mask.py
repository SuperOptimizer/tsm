"""Umbilicus core mask: ``extra.labels.core_radius_vox`` (build time) and
``extra.train.core_radius_vox`` (on the fly in the dataset).

Near the core the in/out face convention degenerates (``recto_is_in`` ~ 0.47 on the Paris 4
slab), so both paths turn the surface labels there into "ignore" (``faces_valid`` /
``surface_valid`` = 2) and the winding validity into "no data" (0), leaving the CT untouched.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
import zarr

from tsm.config import load_config
from synth import AXIS_YX, make_synthetic, synthetic_axis
from tsm.config import parse_config
from tsm.data import CropDataset
from tsm.labels import core_mask, load_axis, run_labels
from tsm.volume import VolumeReader


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_core_mask_is_an_in_plane_disc_around_the_interpolated_axis():
    axis = np.array([[0.0, 10.0, 10.0], [100.0, 30.0, 10.0]])  # y drifts 10 -> 30 with z
    m = core_mask(axis, (0, 0, 0), (101, 64, 64), 5.0)
    assert m.dtype == bool and m.shape == (101, 64, 64)
    for z, ay in ((0, 10.0), (50, 20.0), (100, 30.0)):
        yy, xx = np.nonzero(m[z])
        assert yy.mean() == pytest.approx(ay, abs=0.5) and xx.mean() == pytest.approx(10.0, abs=0.5)
        d = np.hypot(yy - ay, xx - 10.0)
        assert d.max() <= 5.0 + 1e-9
        assert len(yy) == pytest.approx(np.pi * 25.0, rel=0.12)  # a disc, not a square


def test_core_mask_off_and_scaled_grids():
    axis = synthetic_axis()
    assert not core_mask(axis, (0, 0, 0), (4, 8, 8), 0.0).any()
    # the coarse grid: voxel i covers level-0 k*i + (k-1)/2, so the same physical disc
    fine = core_mask(axis, (0, 0, 0), (4, 128, 128), 24.0, scale=1)
    coarse = core_mask(axis, (0, 0, 0), (1, 32, 32), 24.0, scale=4)
    assert coarse.sum() * 16 == pytest.approx(fine[:1].sum(), rel=0.1)


def test_core_mask_offset_origin_matches_a_shifted_crop():
    axis = synthetic_axis()
    big = core_mask(axis, (0, 60, 60), (2, 64, 64), 20.0)
    sub = core_mask(axis, (0, 76, 84), (2, 16, 16), 20.0)
    np.testing.assert_array_equal(sub, big[:, 16:32, 24:40])


# --------------------------------------------------------------------------- #
# train-time path (data.CropDataset)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic(str(tmp_path_factory.mktemp("core")), faces=True, winding_fine=True, rv=True)


def _ds(synth, radius, **kw):
    return CropDataset(lambda: VolumeReader(synth["ct"], 0, 2.4), synth["fine"], synth["coarse"],
                       patch=32, seed=0, length=8, stride=16, augment=False, surface_mode="faces",
                       axis=synthetic_axis(), core_radius_vox=radius, **kw)


RADIUS = 30.0  # the synthetic axis sits at (100, 100), just outside the (32..96)^2 crop grid


def test_dataset_core_radius_masks_surface_and_winding(synth):
    off, on = _ds(synth, 0.0), _ds(synth, RADIUS)
    assert off.core_radius_vox == 0.0 and on.core_radius_vox == RADIUS
    o = on.origins[len(on.origins) - 1]  # the crop closest to the axis (highest y, x)
    a, b = off.load(o), on.load(o)
    gz, gy, gx = (int(v) for v in b["origin_zyx"])
    cm = core_mask(synthetic_axis(), (gz, gy, gx), (32, 32, 32), RADIUS)
    assert 0.01 < cm.mean() < 0.99, "the test crop must straddle the core boundary"

    assert (b["surface_valid"][0][cm] == 2).all()
    assert (b["winding_valid"][0][cm] == 0).all()
    # ...and nothing outside the disc moved, in any target or in the CT
    np.testing.assert_array_equal(b["surface_valid"][0][~cm], a["surface_valid"][0][~cm])
    np.testing.assert_array_equal(b["winding_valid"][0][~cm], a["winding_valid"][0][~cm])
    np.testing.assert_array_equal(b["ct"], a["ct"])
    for k in ("surface_sdf", "ink_prob", "ink_valid", "winding", "winding_conf"):
        np.testing.assert_array_equal(b[k], a[k])
    # the mask really changed something (the crop is not already all-ignore)
    assert (a["surface_valid"][0][cm] != 2).any()


def test_dataset_core_radius_zero_is_byte_identical(synth):
    off, on = _ds(synth, 0.0), _ds(synth, 0.0)
    for o in on.origins[:3]:
        a, b = off.load(o), on.load(o)
        for k, v in a.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(b[k], v, err_msg=k)


def test_dataset_core_radius_loads_the_axis_without_input_radial(synth):
    ds = _ds(synth, RADIUS, input_radial=False)
    assert ds.axis is not None and not ds.input_radial
    it = ds.to_tensors(ds.load(ds.origins[len(ds.origins) - 1]))
    assert tuple(it["input"].shape) == (2, 32, 32, 32)  # no radial channels
    assert torch.any(it["surface_valid"] == 2)


def test_dataset_rejects_a_negative_core_radius(synth):
    with pytest.raises(ValueError, match="core_radius_vox"):
        _ds(synth, -1.0)


def _train_cfg(synth, root: str, train: dict):
    return parse_config({
        "volume": {"url": synth["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root, "extra": {"train": train},
    })


def test_train_opts_core_radius_reaches_the_dataset(synth, tmp_path):
    from tsm.train import build_dataset, train_opts

    root = str(tmp_path)
    cfg = _train_cfg(synth, root, {"patch": 32, "stride": 16, "steps": 1, "surface_mode": "faces",
                                   "core_radius_vox": RADIUS, "input_radial": False,
                                   "axis_path": _axis_json(root),
                                   "fine_store": synth["fine"], "coarse_store": synth["coarse"],
                                   "eval_holdout": False})
    opts = train_opts(cfg)
    assert opts["core_radius_vox"] == RADIUS
    ds = build_dataset(cfg, opts, augment=False)
    assert ds.core_radius_vox == RADIUS
    assert train_opts(_train_cfg(synth, root, {}))["core_radius_vox"] == 0.0
    with pytest.raises(ValueError, match="core_radius_vox"):
        train_opts(_train_cfg(synth, root, {"core_radius_vox": -5}))


def _axis_json(root: str) -> str:
    path = os.path.join(root, "axis.json")
    with open(path, "w") as fh:
        json.dump({"coordinate_order": "xyz", "coordinate_space": "full_resolution",
                   "control_points": [{"z": -100000, "y": AXIS_YX[0], "x": AXIS_YX[1]},
                                      {"z": 100000, "y": AXIS_YX[0], "x": AXIS_YX[1]}]}, fh)
    return path


# --------------------------------------------------------------------------- #
# build-time path (labels.run_labels)
# --------------------------------------------------------------------------- #
CORE_R = 12.0


def _label_config(root: str, out_dir: str, radius: float) -> str:
    from test_labels_workers import ORIGIN, SHAPE

    cfg = {
        "volume": {"url": os.path.join(root, "ct.zarr"), "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": list(ORIGIN), "size_zyx": list(SHAPE)},
        "budget": {"ram_bytes": 4 << 30, "array_bytes": 1 << 30, "min_avail_mb": 256},
        "out_dir": out_dir,
        "extra": {"labels": {
            "clip": 6, "halo": 8, "brick": [16, 32, 32], "coarse_brick": [8, 16, 16],
            "coarse_halo": 2, "probe_brick": 8, "min_component": 10,
            "teachers_dir": os.path.join(root, "teachers"),
            "axis_path": os.path.join(root, "axis.json"),
            "faces": {"enabled": True, "body_threshold": 60.0, "min_thickness": 2.0,
                      "close_radius": 0.0, "min_component": 10},
            "winding_fine": True,
            "core_radius_vox": radius,
        }},
    }
    path = os.path.join(root, f"cfg_core{radius:g}.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return path


def _centre_axis(path: str) -> None:
    with open(path, "w") as fh:
        json.dump({"coordinate_order": "xyz", "coordinate_space": "full_resolution",
                   "control_points": [{"z": -1000, "y": 32.0, "x": 32.0},
                                      {"z": 100000, "y": 32.0, "x": 32.0}]}, fh)


def _store(path: str) -> tuple[list[str], np.ndarray]:
    a = zarr.open_array(store=path, mode="r")
    return list(a.attrs["channels"]), np.asarray(a[:])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One label build with the core mask off and one with it on (same inputs)."""
    from test_labels_workers import _teachers

    root = str(tmp_path_factory.mktemp("corelabels"))
    _teachers(os.path.join(root, "teachers"))
    _centre_axis(os.path.join(root, "axis.json"))
    out = {}
    for name, r in (("off", 0.0), ("on", CORE_R)):
        d = os.path.join(root, name)
        run_labels(load_config(_label_config(root, d, r)))
        out[name] = d
    out["root"] = root
    return out


def test_label_build_core_mask_ignores_faces_and_kills_winding_validity(built):
    from test_labels_workers import ORIGIN, SHAPE

    axis = load_axis(os.path.join(built["root"], "axis.json"))
    cm = core_mask(axis, ORIGIN, SHAPE, CORE_R)
    assert 0.01 < cm.mean() < 0.5

    ch_off, off = _store(os.path.join(built["off"], "labels", "fine.zarr"))
    ch_on, on = _store(os.path.join(built["on"], "labels", "fine.zarr"))
    assert ch_off == ch_on and "faces_valid" in ch_on and "wf_valid" in ch_on
    fv, wv = ch_on.index("faces_valid"), ch_on.index("wf_valid")

    assert (on[fv][cm] == 2).all()
    assert (on[wv][cm] == 0).all()
    assert (off[fv][cm] != 2).any(), "the unmasked build must have real labels in the core"
    # Everything else is essentially untouched away from the disc -- "essentially" because the
    # fine build reads the *coarse* normal field, whose validity the same mask clears, so inside
    # (and one coarse voxel around) the disc the sheet normal falls back to the radial direction.
    # That changes the sdf sign there, and through the component-level thickness / side gates it
    # can flip the odd voxel elsewhere in the same brick.  Bound the ripple instead of forbidding
    # it: outside a generously dilated disc every channel is >= 99.9 % identical.
    wide = core_mask(axis, ORIGIN, SHAPE, CORE_R + 2 * 4)
    for c, name in enumerate(ch_on):
        diff = float((on[c][~wide] != off[c][~wide]).mean())
        assert diff < 1e-3, f"{name}: {diff:.4%} of the voxels outside the core changed"
    # the two validity channels the mask owns are exact outside the ring
    for c in (fv, wv):
        np.testing.assert_array_equal(on[c][~wide], off[c][~wide], err_msg=ch_on[c])


def test_label_build_core_mask_clears_the_coarse_validity(built):
    from test_labels_workers import ORIGIN, SHAPE

    axis = load_axis(os.path.join(built["root"], "axis.json"))
    k = 4
    cmc = core_mask(axis, [o // k for o in ORIGIN], [s // k for s in SHAPE], CORE_R, scale=k)
    ch_off, off = _store(os.path.join(built["off"], "labels", "coarse.zarr"))
    ch_on, on = _store(os.path.join(built["on"], "labels", "coarse.zarr"))
    v = ch_on.index("valid")
    assert cmc.any() and (on[v][cmc] == 0).all()
    assert (off[v][cmc] != 0).any()
    # the coarse stage only rewrites "valid", and only inside the disc
    for c, name in enumerate(ch_on):
        if name == "valid":
            np.testing.assert_array_equal(on[c][~cmc], off[c][~cmc])
        else:
            np.testing.assert_array_equal(on[c], off[c], err_msg=name)


def test_label_build_summary_reports_the_core(built):
    with open(os.path.join(built["on"], "labels", "labels.summary.json")) as fh:
        s = json.load(fh)
    assert s["opts"]["core_radius_vox"] == CORE_R
    assert s["fine"]["faces"]["core_radius_vox"] == CORE_R
    assert s["fine"]["faces"]["core_masked_voxels"] > 0
    with open(os.path.join(built["off"], "labels", "labels.summary.json")) as fh:
        s0 = json.load(fh)
    assert s0["fine"]["faces"]["core_masked_voxels"] == 0


def test_label_opts_rejects_a_negative_core_radius(tmp_path):
    from tsm.config import parse_config
    from tsm.labels import _label_opts

    cfg = parse_config({
        "volume": {"url": "x", "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [0, 0, 0], "size_zyx": [32, 32, 32]},
        "budget": {"ram_bytes": 4 << 30, "array_bytes": 1 << 30},
        "out_dir": str(tmp_path), "extra": {"labels": {"core_radius_vox": -1}},
    })
    with pytest.raises(ValueError, match="core_radius_vox"):
        _label_opts(cfg)
