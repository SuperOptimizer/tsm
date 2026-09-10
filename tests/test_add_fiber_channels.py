"""dev/add_fiber_channels.py: inserting the fibre block into an existing fine store."""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest
import zarr

from tsm.labels import FIBER_CHANNELS, fine_channels
from tsm.volume import BrickWriter

_SPEC = importlib.util.spec_from_file_location(
    "add_fiber_channels",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dev", "add_fiber_channels.py"),
)
afc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(afc)  # type: ignore[union-attr]

SHAPE = (16, 24, 24)
ORIGIN = (8, 16, 16)
BRICK = [8, 12, 12]
CHUNK = 4


def _fields(channels: list[str], fiber_vt: np.ndarray, fiber_hz: np.ndarray,
            rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One uint8 volume per channel; the fibre block matches the teacher + ink_valid."""
    out: dict[str, np.ndarray] = {}
    for name in channels:
        if name == "fiber_vt":
            out[name] = fiber_vt
        elif name == "fiber_hz":
            out[name] = fiber_hz
        elif name.endswith("_valid"):
            out[name] = rng.integers(0, 3, SHAPE, dtype=np.uint8)
        else:
            out[name] = rng.integers(0, 256, SHAPE, dtype=np.uint8)
    if "fiber_valid" in out:
        out["fiber_valid"] = out["ink_valid"]
    return out


def _write_store(path: str, channels: list[str], f: dict[str, np.ndarray]) -> str:
    w = BrickWriter(path, channels, SHAPE, chunk=CHUNK, origin_zyx=ORIGIN, voxel_um=2.4, scale=1.0)
    for ci, name in enumerate(channels):
        for z0 in range(0, SHAPE[0], BRICK[0]):
            for y0 in range(0, SHAPE[1], BRICK[1]):
                for x0 in range(0, SHAPE[2], BRICK[2]):
                    blk = f[name][z0:z0 + BRICK[0], y0:y0 + BRICK[1], x0:x0 + BRICK[2]]
                    w.write(ci, ORIGIN[0] + z0, ORIGIN[1] + y0, ORIGIN[2] + x0, np.ascontiguousarray(blk))
    w.array.attrs.update({"note": "synthetic", "summary_seconds": 1.5})
    return path


def _config(tmp_path, source: str) -> str:
    """A labels config whose region/brick match the synthetic stores."""
    faces: dict = {"enabled": True, "body_threshold": 60.0}
    if source != "ct":
        faces["source"] = source
        faces["rectoverso_store"] = str(tmp_path / "rectoverso.zarr")
    cfg = {
        "volume": {"url": str(tmp_path / "ct.zarr"), "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": list(ORIGIN), "size_zyx": list(SHAPE)},
        "budget": {"ram_bytes": 1 << 30, "array_bytes": 1 << 28},
        "out_dir": str(tmp_path / "out"),
        "extra": {"labels": {"clip": 20, "halo": 24, "brick": BRICK,
                             "teachers_dir": str(tmp_path / "teachers")}},
    }
    cfg["extra"]["labels"]["faces"] = faces
    path = str(tmp_path / f"cfg_{source}.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return path


def _setup(tmp_path, source: str) -> tuple[str, dict[str, np.ndarray], list[str], list[str]]:
    """Old store (no fibre), reference store (fibre from the start) and the fibre teacher."""
    rng = np.random.default_rng(7)
    do_rv = source != "ct"
    want_old = fine_channels(True, False, do_rv)
    want_new = fine_channels(True, True, do_rv)
    vt = rng.integers(0, 256, SHAPE, dtype=np.uint8)
    hz = rng.integers(0, 256, SHAPE, dtype=np.uint8)
    f = _fields(want_new, vt, hz, rng)
    _write_store(str(tmp_path / "out" / "labels" / "fine.zarr"), want_old, f)
    _write_store(str(tmp_path / "ref.zarr"), want_new, f)
    tdir = tmp_path / "teachers"
    tdir.mkdir(parents=True, exist_ok=True)
    fib = zarr.create_array(store=str(tdir / "fiber.zarr"), shape=(4,) + SHAPE, chunks=(1, 4, 4, 4),
                            dtype="uint8", fill_value=0, attributes={"origin_zyx": list(ORIGIN)})
    fib[0] = rng.integers(0, 256, SHAPE, dtype=np.uint8)
    fib[1] = vt
    fib[2] = hz
    fib[3] = rng.integers(0, 256, SHAPE, dtype=np.uint8)
    return _config(tmp_path, source), f, want_old, want_new


@pytest.mark.parametrize("source", ["ct", "merge"])
def test_append_matches_a_store_built_with_fiber_from_the_start(tmp_path, source):
    cfg_path, f, want_old, want_new = _setup(tmp_path, source)
    fine = str(tmp_path / "out" / "labels" / "fine.zarr")
    assert [str(c) for c in zarr.open_array(store=fine, mode="r").attrs["channels"]] == want_old

    assert afc.main(["add_fiber_channels", cfg_path, "--force"]) == 0

    got = zarr.open_array(store=fine, mode="r")
    ref = zarr.open_array(store=str(tmp_path / "ref.zarr"), mode="r")
    # channels identical by name (and so is the whole array, channel for channel)
    assert [str(c) for c in got.attrs["channels"]] == want_new == [str(c) for c in ref.attrs["channels"]]
    assert got.shape == ref.shape and got.chunks == ref.chunks
    for i, name in enumerate(want_new):
        np.testing.assert_array_equal(np.asarray(got[i]), np.asarray(ref[i]), err_msg=name)
        np.testing.assert_array_equal(np.asarray(got[i]), f[name], err_msg=name)
    # the fibre block really is the teacher's channels 1/2 plus ink_valid
    for name in FIBER_CHANNELS:
        assert name in want_new
    np.testing.assert_array_equal(np.asarray(got[want_new.index("fiber_valid")]),
                                  np.asarray(got[want_new.index("ink_valid")]))
    if source == "merge":  # the rv block was inserted in front of, not overwritten by, the fibre block
        assert want_new.index("faces_source") == want_new.index("fiber_valid") + 1


def test_swapped_store_carries_old_attrs_and_a_valid_done_json(tmp_path):
    cfg_path, _, _, want_new = _setup(tmp_path, "merge")
    assert afc.main(["add_fiber_channels", cfg_path, "--force"]) == 0
    fine = str(tmp_path / "out" / "labels" / "fine.zarr")
    attrs = dict(zarr.open_array(store=fine, mode="r").attrs)
    assert attrs["note"] == "synthetic" and attrs["summary_seconds"] == 1.5
    assert attrs["origin_zyx"] == list(ORIGIN) and attrs["voxel_um"] == 2.4
    assert not os.path.exists(os.path.join(fine, afc.MARKER))
    assert os.path.exists(str(tmp_path / "out" / "labels" / "fine.zarr.prev"))
    # a reader/resume reopening the store with the new channel list keeps the done keys
    w = BrickWriter(fine, want_new, SHAPE, chunk=CHUNK, origin_zyx=ORIGIN, voxel_um=2.4, scale=1.0)
    with open(os.path.join(fine, "done.json")) as fh:
        d = json.load(fh)
    assert d["writer_fingerprint"] == w.fingerprint
    assert all(w.has_brick(ORIGIN[0] + z, ORIGIN[1] + y, ORIGIN[2] + x)
               for z in range(0, SHAPE[0], BRICK[0])
               for y in range(0, SHAPE[1], BRICK[1])
               for x in range(0, SHAPE[2], BRICK[2]))


def test_resumes_after_a_kill_mid_copy(tmp_path):
    cfg_path, f, want_old, want_new = _setup(tmp_path, "merge")
    fine = str(tmp_path / "out" / "labels" / "fine.zarr")
    new = fine + ".new"
    # simulate a killed run: a partial .new store with one brick finished
    w = BrickWriter(new, want_new, SHAPE, chunk=CHUNK, origin_zyx=ORIGIN, voxel_um=2.4, scale=1.0)
    for ci in range(len(want_new)):
        w.write(ci, *ORIGIN, np.zeros(tuple(BRICK), np.uint8))
    del w
    assert afc.main(["add_fiber_channels", cfg_path, "--force"]) == 0
    got = zarr.open_array(store=fine, mode="r")
    # the already-"done" brick is kept as written (zeros), the rest copied properly
    b = tuple(slice(0, v) for v in BRICK)
    for i, name in enumerate(want_new):
        assert np.asarray(got[i])[b].max() == 0, name
        np.testing.assert_array_equal(np.asarray(got[i])[BRICK[0]:], f[name][BRICK[0]:], err_msg=name)


def test_refuses_when_the_channel_list_is_not_the_expected_one(tmp_path):
    cfg_path, _, _, want_new = _setup(tmp_path, "merge")
    assert afc.main(["add_fiber_channels", cfg_path, "--force"]) == 0
    with pytest.raises(SystemExit, match="channels"):  # already has the fibre block
        afc.main(["add_fiber_channels", cfg_path, "--force"])


def test_refuses_while_a_build_looks_active(tmp_path, monkeypatch):
    cfg_path, _, _, _ = _setup(tmp_path, "merge")
    fine = str(tmp_path / "out" / "labels" / "fine.zarr")
    monkeypatch.setattr(afc, "labels_processes", lambda p: ["4242 uv run tsm labels " + p])
    with pytest.raises(SystemExit, match="label build looks active"):
        afc.main(["add_fiber_channels", cfg_path])
    # no process, but a fresh done.json (the synthetic store was just written)
    monkeypatch.setattr(afc, "labels_processes", lambda p: [])
    with pytest.raises(SystemExit, match="modified"):
        afc.main(["add_fiber_channels", cfg_path])
    assert not os.path.exists(fine + ".new")
    # stale enough -> the completeness check passes and the run goes through
    assert afc.main(["add_fiber_channels", cfg_path, "--stale-minutes", "0"]) == 0


def test_refuses_when_the_old_store_is_incomplete(tmp_path):
    cfg_path, _, want_old, _ = _setup(tmp_path, "merge")
    fine = str(tmp_path / "out" / "labels" / "fine.zarr")
    with open(os.path.join(fine, "done.json")) as fh:
        d = json.load(fh)
    d["bricks"].pop(next(iter(d["bricks"])))
    with open(os.path.join(fine, "done.json"), "w") as fh:
        json.dump(d, fh)
    with pytest.raises(SystemExit, match="incomplete"):
        afc.main(["add_fiber_channels", cfg_path, "--stale-minutes", "0"])


def test_dry_run_writes_nothing(tmp_path):
    cfg_path, _, want_old, _ = _setup(tmp_path, "merge")
    fine = str(tmp_path / "out" / "labels" / "fine.zarr")
    assert afc.main(["add_fiber_channels", cfg_path, "--force", "--dry-run"]) == 0
    assert not os.path.exists(fine + ".new") and not os.path.exists(fine + ".prev")
    assert [str(c) for c in zarr.open_array(store=fine, mode="r").attrs["channels"]] == want_old
