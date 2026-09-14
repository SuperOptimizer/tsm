"""``extra.labels.require_teachers``: build a slab from ``lasagna.zarr`` alone.

A slab whose faces come from the upstream recto/verso bands needs neither the recto nor the
ink teacher.  With ``require_teachers: []`` the build tolerates both being absent: the ink
channels are written all-zero (``ink_valid = 0``, so the ink head is masked out on that store)
and the recto-derived ``sdf``/``sdf_valid`` channels are all-ignore.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import zarr

from tsm.config import load_config
from tsm.labels import run_labels

SHAPE = (32, 64, 64)
ORIGIN = (0, 0, 0)
K = 4
#: the umbilicus sits far on the -y side, so the outward radial direction is ~ +y and the
#: sheets below (stacked along y) are crossed by it, as in a real scroll
AXIS_YX = (-4000.0, 32.0)


def _fields():
    """(rectoverso class volume, CT) for sheets stacked along +y (period 24, half width 4)."""
    z, y, x = np.mgrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    ph = y - 3.0 * np.sin(x / 9.0) - 2.0 * np.cos(z / 7.0)
    d = (ph + 12.0) % 24.0 - 12.0
    rv = np.zeros(SHAPE, np.uint8)
    rv[(d >= -4.5) & (d <= -3.0)] = 1   # recto = the in face (toward the axis, -y)
    rv[(d >= 3.0) & (d <= 4.5)] = 2     # verso = the out face
    ct = np.where(np.abs(d) <= 4.5, 220, 10).astype(np.uint8)
    return rv, ct


def _lasagna_only(tdir: str, *, ink: bool = False, recto: bool = False) -> None:
    """``teachers_dir`` with the coarse winding store and (optionally) recto / ink."""
    os.makedirs(tdir, exist_ok=True)
    rng = np.random.default_rng(5)
    cs = tuple(s // K for s in SHAPE)
    las = zarr.create_array(store=os.path.join(tdir, "lasagna.zarr"), shape=(8,) + cs,
                            chunks=(1, 8, 8, 8), dtype="uint8", fill_value=0,
                            attributes={"origin_zyx": [v // K for v in ORIGIN], "scale": float(K)})
    for c in range(8):
        las[c] = rng.integers(0, 256, cs, dtype=np.uint8)
    for name, on in (("recto", recto), ("ink", ink)):
        if not on:
            continue
        a = zarr.create_array(store=os.path.join(tdir, f"{name}.zarr"), shape=(1,) + SHAPE,
                              chunks=(1, 16, 16, 16), dtype="uint8", fill_value=0,
                              attributes={"origin_zyx": list(ORIGIN)})
        a[0] = rng.integers(0, 256, SHAPE, dtype=np.uint8)


def _rectoverso(path: str, rv: np.ndarray) -> str:
    a = zarr.create_array(store=path, shape=(2,) + SHAPE, chunks=(1, 16, 16, 16), dtype="uint8",
                          fill_value=0, attributes={"origin_zyx": list(ORIGIN)})
    a[0] = rv
    a[1] = np.zeros(SHAPE, np.uint8)
    return path


def _ct(path: str, ct: np.ndarray) -> None:
    g = zarr.open_group(store=path, mode="w")
    a0 = g.create_array("0", shape=SHAPE, chunks=(16, 16, 16), dtype="uint8", fill_value=0)
    a0[:] = ct
    cs = tuple(s // K for s in SHAPE)
    a2 = g.create_array("2", shape=cs, chunks=(8, 8, 8), dtype="uint8", fill_value=0)
    a2[:] = ct[::K, ::K, ::K]


def _axis(path: str) -> str:
    with open(path, "w") as fh:
        json.dump({"coordinate_order": "xyz", "coordinate_space": "full_resolution",
                   "control_points": [{"z": -1000, "y": AXIS_YX[0], "x": AXIS_YX[1]},
                                      {"z": 100000, "y": AXIS_YX[0], "x": AXIS_YX[1]}]}, fh)
    return path


def _config(root: str, out_dir: str, *, source: str = "rectoverso", require=None) -> str:
    labels = {
        "clip": 6, "halo": 8, "brick": [16, 32, 32], "coarse_brick": [8, 16, 16],
        "coarse_halo": 2, "probe_brick": 8, "min_component": 10,
        "teachers_dir": os.path.join(root, "teachers"),
        "axis_path": os.path.join(root, "axis.json"),
        "faces": {"enabled": True, "source": source, "min_component": 10,
                  "rv_min_component": 8, "rv_min_normal_count": 4},
    }
    if source == "ct":
        labels["faces"] = {"enabled": True, "min_component": 10, "body_threshold": 60.0,
                           "min_thickness": 2.0, "close_radius": 0.0}
    else:
        labels["faces"]["rectoverso_store"] = os.path.join(root, "rectoverso.zarr")
    if require is not None:
        labels["require_teachers"] = require
    cfg = {
        "volume": {"url": os.path.join(root, "ct.zarr"), "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": list(ORIGIN), "size_zyx": list(SHAPE)},
        "budget": {"ram_bytes": 4 << 30, "array_bytes": 1 << 30, "min_avail_mb": 256},
        "out_dir": out_dir,
        "extra": {"labels": labels},
    }
    path = os.path.join(root, f"cfg_{os.path.basename(out_dir)}.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return path


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("optional_teachers"))
    rv, ct = _fields()
    _lasagna_only(os.path.join(root, "teachers"))
    _rectoverso(os.path.join(root, "rectoverso.zarr"), rv)
    _ct(os.path.join(root, "ct.zarr"), ct)
    _axis(os.path.join(root, "axis.json"))
    return root


def test_lasagna_only_build_with_rectoverso_faces(slab):
    out = os.path.join(slab, "out")
    summary = run_labels(load_config(_config(slab, out, require=[])))
    assert summary["teachers"] == {"recto": "absent", "ink": "absent", "fiber": "absent",
                                   "lasagna": "present"}
    a = zarr.open_array(store=os.path.join(out, "labels", "fine.zarr"), mode="r")
    ch = list(a.attrs["channels"])
    assert ch[:4] == ["sdf", "sdf_valid", "ink", "ink_valid"]
    assert "faces_valid" in ch and "faces_source" in ch and "fiber_valid" not in ch
    data = np.asarray(a[:])
    # ink is fully absent: no probability, no supervision
    assert (data[ch.index("ink")] == 0).all()
    assert (data[ch.index("ink_valid")] == 0).all()
    # the recto-derived sdf carries no supervision either (0 no-data / 2 ignore, never 1)
    sv = data[ch.index("sdf_valid")]
    assert set(np.unique(sv).tolist()) <= {0, 2} and (sv == 2).any()
    # the faces, however, were built from the rectoverso bands
    fv = data[ch.index("faces_valid")]
    assert (fv == 1).any()
    assert summary["fine"]["faces"]["recto_is_in_mean"] is None
    assert summary["fine"]["faces"]["rectoverso"]["source"] == "rectoverso"
    assert summary["fine"]["ink_valid_fraction"] == 0.0


def test_fiber_valid_still_follows_the_ct_mask_without_ink(tmp_path):
    """`fiber_valid` is the CT data mask, not `ink_valid`: a missing ink teacher must not take
    the fibre supervision down with it."""
    root = str(tmp_path)
    rv, ct = _fields()
    tdir = os.path.join(root, "teachers")
    _lasagna_only(tdir)
    rng = np.random.default_rng(7)
    fib = zarr.create_array(store=os.path.join(tdir, "fiber.zarr"), shape=(4,) + SHAPE,
                            chunks=(1, 16, 16, 16), dtype="uint8", fill_value=0,
                            attributes={"origin_zyx": list(ORIGIN)})
    for c in range(4):
        fib[c] = rng.integers(0, 256, SHAPE, dtype=np.uint8)
    _rectoverso(os.path.join(root, "rectoverso.zarr"), rv)
    _ct(os.path.join(root, "ct.zarr"), ct)
    _axis(os.path.join(root, "axis.json"))
    out = os.path.join(root, "out")
    summary = run_labels(load_config(_config(root, out, require=[])))
    assert summary["teachers"]["fiber"] == "present" and summary["teachers"]["ink"] == "absent"
    a = zarr.open_array(store=os.path.join(out, "labels", "fine.zarr"), mode="r")
    ch = list(a.attrs["channels"])
    data = np.asarray(a[:])
    assert (data[ch.index("ink_valid")] == 0).all()
    assert (data[ch.index("fiber_valid")] == 1).any()
    assert summary["fine"]["fiber"]["valid_fraction"] > 0.0


def test_ct_face_source_still_requires_recto(slab):
    with pytest.raises(FileNotFoundError, match="recto"):
        run_labels(load_config(_config(slab, os.path.join(slab, "out_ct"), source="ct", require=[])))


def test_default_require_teachers_still_fails_on_a_missing_recto(slab):
    with pytest.raises(FileNotFoundError, match=r"\['ink', 'recto'\]"):
        run_labels(load_config(_config(slab, os.path.join(slab, "out_default"))))


def test_require_teachers_validation(slab):
    from tsm.labels import _label_opts

    cfg = load_config(_config(slab, os.path.join(slab, "out_val"), require=["ink"]))
    assert _label_opts(cfg)["require_teachers"] == ["ink"]
    with pytest.raises(ValueError, match="require_teachers"):
        _label_opts(load_config(_config(slab, os.path.join(slab, "out_val2"), require=["lasagna"])))
    with pytest.raises(ValueError, match="require_teachers"):
        _label_opts(load_config(_config(slab, os.path.join(slab, "out_val3"), require="recto")))


def test_dry_run_reports_which_teachers_are_present(slab, capsys):
    run_labels(load_config(_config(slab, os.path.join(slab, "out_dry"), require=[])), dry_run=True)
    line = [l for l in capsys.readouterr().out.splitlines() if "teachers_dir" in l]
    assert line and "recto=ABSENT" in line[0] and "ink=ABSENT" in line[0]
    assert "lasagna=present (required)" in line[0]
    assert not os.path.exists(os.path.join(slab, "out_dry", "labels"))
