"""``extra.labels.workers``: a fork pool must produce exactly the single-process build."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import zarr

from tsm.config import load_config
from tsm.labels import merge_stats, run_labels

SHAPE = (32, 64, 64)
ORIGIN = (0, 0, 0)
K = 4


def _teachers(tdir: str) -> None:
    os.makedirs(tdir, exist_ok=True)
    rng = np.random.default_rng(3)
    z, y, x = np.mgrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    # three wavy sheets stacked along z (the same geometry tests/synth.py uses)
    phase = z - 3.0 * np.sin(x / 9.0) - 2.0 * np.cos(y / 7.0)
    d = (phase + 6.0) % 12.0 - 6.0
    recto = np.clip(255.0 * np.exp(-(d ** 2) / 2.0), 0, 255).astype(np.uint8)
    ink = np.clip(recto.astype(np.int32) - 60, 0, 255).astype(np.uint8)
    for name, vol in (("recto", recto), ("ink", ink)):
        a = zarr.create_array(store=os.path.join(tdir, f"{name}.zarr"), shape=(1,) + SHAPE,
                              chunks=(1, 16, 16, 16), dtype="uint8", fill_value=0,
                              attributes={"origin_zyx": list(ORIGIN)})
        a[0] = vol
    cs = tuple(s // K for s in SHAPE)
    las = zarr.create_array(store=os.path.join(tdir, "lasagna.zarr"), shape=(8,) + cs,
                            chunks=(1, 8, 8, 8), dtype="uint8", fill_value=0,
                            attributes={"origin_zyx": [v // K for v in ORIGIN], "scale": float(K)})
    for c in range(8):
        las[c] = rng.integers(0, 256, cs, dtype=np.uint8)
    fib = zarr.create_array(store=os.path.join(tdir, "fiber.zarr"), shape=(4,) + SHAPE,
                            chunks=(1, 16, 16, 16), dtype="uint8", fill_value=0,
                            attributes={"origin_zyx": list(ORIGIN)})
    for c in range(4):
        fib[c] = rng.integers(0, 256, SHAPE, dtype=np.uint8)
    # CT: an OME-Zarr-ish group with the level 0 and level 2 the label build reads
    ct = np.clip(40.0 + 200.0 * np.exp(-(d ** 2) / 4.0) + rng.normal(0, 4, SHAPE), 1, 255).astype(np.uint8)
    g = zarr.open_group(store=os.path.join(os.path.dirname(tdir), "ct.zarr"), mode="w")
    a0 = g.create_array("0", shape=SHAPE, chunks=(16, 16, 16), dtype="uint8", fill_value=0)
    a0[:] = ct
    a2 = g.create_array("2", shape=cs, chunks=(8, 8, 8), dtype="uint8", fill_value=0)
    a2[:] = ct[::K, ::K, ::K]


def _config(root: str, out_dir: str, workers: int, mp_start: str = "fork") -> str:
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
            "workers": workers, "mp_start": mp_start,
        }},
    }
    path = os.path.join(root, f"cfg_w{workers}_{mp_start}.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return path


def _axis(path: str) -> None:
    with open(path, "w") as fh:
        json.dump({"coordinate_order": "xyz", "coordinate_space": "full_resolution",
                   "control_points": [{"z": -1000, "y": 200.0, "x": 200.0},
                                      {"z": 100000, "y": 200.0, "x": 200.0}]}, fh)


def _store_bytes(path: str) -> dict[str, np.ndarray]:
    a = zarr.open_array(store=path, mode="r")
    return {"channels": list(a.attrs["channels"]), "data": np.asarray(a[:])}


def _clean(summary: dict) -> dict:
    """The summary minus what legitimately differs between two runs."""
    out = json.loads(json.dumps(summary))
    out.pop("seconds", None)
    out.pop("peak_rss_mb", None)
    out.pop("stores", None)
    out.pop("png", None)
    out.get("opts", {}).pop("workers", None)
    out.get("opts", {}).pop("mp_start", None)
    out.get("opts", {}).pop("teachers_dir", None)
    for k in ("fine", "coarse"):
        out.get(k, {}).pop("seconds", None)
    return out


@pytest.mark.parametrize("mp_start", ["fork", "forkserver", "spawn"])
def test_worker_pool_reproduces_the_single_process_build(tmp_path, mp_start):
    workers = 3
    root = str(tmp_path)
    _teachers(os.path.join(root, "teachers"))
    _axis(os.path.join(root, "axis.json"))
    s1 = run_labels(load_config(_config(root, os.path.join(root, "out1"), 1)))
    sN = run_labels(load_config(_config(root, os.path.join(root, "outN"), workers, mp_start)))

    for name in ("fine.zarr", "coarse.zarr"):
        a = _store_bytes(os.path.join(root, "out1", "labels", name))
        b = _store_bytes(os.path.join(root, "outN", "labels", name))
        assert a["channels"] == b["channels"], name
        np.testing.assert_array_equal(a["data"], b["data"], err_msg=name)
        with open(os.path.join(root, "out1", "labels", name, "done.json")) as fh:
            d1 = fh.read()
        with open(os.path.join(root, "outN", "labels", name, "done.json")) as fh:
            dN = fh.read()
        assert d1 == dN, f"{name}/done.json differs"
    assert _clean(s1) == _clean(sN)
    # the build really did something to compare
    assert s1["fine"]["computed"] == 8 and "fiber" in s1["fine"] and "faces" in s1["fine"]


def test_merge_stats_adds_counters_arrays_and_lists():
    dst: dict = {}
    merge_stats(dst, {"n": 2, "h": np.array([1, 0, 0]), "comp": [{"a": 1}], "f": 0.5})
    merge_stats(dst, {"n": 3, "h": np.array([0, 4, 0]), "comp": [{"a": 2}], "f": 0.25, "new": 7})
    assert dst["n"] == 5 and dst["f"] == 0.75 and dst["new"] == 7
    np.testing.assert_array_equal(dst["h"], [1, 4, 0])
    assert dst["comp"] == [{"a": 1}, {"a": 2}]


def test_workers_are_refused_when_they_would_not_fit(tmp_path):
    from tsm.labels import _fine_pool_limit
    from tsm.limits import Budget, BudgetError

    b = Budget(ram_bytes=1 << 30, array_bytes=1 << 30, min_avail_mb=256)
    opts = {"workers_ram_bytes": 4 << 30}
    assert _fine_pool_limit(4, 1 << 30, opts, b) == (4 << 30)
    with pytest.raises(BudgetError, match="labels.workers"):
        _fine_pool_limit(5, 1 << 30, opts, b)
