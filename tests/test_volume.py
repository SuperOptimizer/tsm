import os

import numpy as np
import pytest
import zarr

from tsm.config import RegionCfg
from tsm.limits import Budget, BudgetError
from tsm.volume import BrickWriter, VolumeReader, iter_bricks, open_array

SHAPE = (16, 24, 32)


def _data(dtype=np.uint8):
    n = int(np.prod(SHAPE))
    return (np.arange(n, dtype=np.int64) % 251).astype(dtype).reshape(SHAPE)


def _ome_v2(tmp_path):
    path = str(tmp_path / "ome_v2.zarr")
    g = zarr.open_group(store=path, mode="w", zarr_format=2)
    g.attrs["multiscales"] = [{"datasets": [{"path": "0"}, {"path": "1"}]}]
    a0 = g.create_array("0", shape=SHAPE, chunks=(8, 8, 8), dtype="uint8")
    a0[:] = _data()
    a1 = g.create_array("1", shape=(8, 12, 16), chunks=(4, 4, 4), dtype="uint8")
    a1[:] = _data()[::2, ::2, ::2]
    return path


def _v3_array(tmp_path, dtype="uint8"):
    path = str(tmp_path / f"v3_{dtype}.zarr")
    a = zarr.create_array(store=path, shape=SHAPE, chunks=(8, 8, 8), dtype=dtype)
    a[:] = _data(np.dtype(dtype))
    return path


def test_open_ome_v2_levels(tmp_path):
    path = _ome_v2(tmp_path)
    assert open_array(path, 0).shape == SHAPE
    assert open_array(path, 1).shape == (8, 12, 16)
    with pytest.raises(KeyError):
        open_array(path, 7)


def test_open_missing_path():
    with pytest.raises(FileNotFoundError):
        open_array("/definitely/not/here.zarr", 0)


def test_reader_v3_and_metadata(tmp_path):
    r = VolumeReader(_v3_array(tmp_path), 0)
    assert r.shape == SHAPE and r.chunks == (8, 8, 8) and r.dtype == np.uint8
    np.testing.assert_array_equal(r.read(0, 4, 0, 4, 0, 4), _data()[:4, :4, :4])


def test_reader_uint16_downshift(tmp_path):
    path = str(tmp_path / "u16.zarr")
    a = zarr.create_array(store=path, shape=SHAPE, chunks=(8, 8, 8), dtype="uint16")
    a[:] = (_data(np.uint16) * 257)
    r = VolumeReader(path, 0)
    got = r.read(0, 2, 0, 2, 0, 2)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, ((_data(np.uint16)[:2, :2, :2] * 257) >> 8).astype(np.uint8))


def test_read_edge_padding(tmp_path):
    r = VolumeReader(_ome_v2(tmp_path), 0)
    out = r.read(-2, 4, -3, 3, 30, 36)
    assert out.shape == (6, 6, 6)
    assert out[:2].max() == 0 and out[:, :3].max() == 0 and out[:, :, 2:].max() == 0
    np.testing.assert_array_equal(out[2:, 3:, :2], _data()[0:4, 0:3, 30:32])


def test_read_fully_outside_is_zero(tmp_path):
    r = VolumeReader(_v3_array(tmp_path), 0)
    assert r.read(100, 104, 0, 4, 0, 4).max() == 0


def test_read_budget_enforced(tmp_path):
    r = VolumeReader(_v3_array(tmp_path), 0, budget=Budget(ram_bytes=1 << 30, array_bytes=64))
    with pytest.raises(BudgetError):
        r.read(0, 8, 0, 8, 0, 8)


def test_iter_bricks_covers_region_exactly_once(tmp_path):
    r = VolumeReader(_v3_array(tmp_path), 0)
    region = RegionCfg(start_zyx=(0, 4, 4), size_zyx=(16, 20, 20))
    cover = np.zeros(SHAPE, np.int32)
    seen = set()
    for b in iter_bricks(r, region, brick=8, halo=0):
        assert b.index_zyx not in seen
        seen.add(b.index_zyx)
        assert b.data.shape == b.core_shape
        cz, cy, cx = b.core_shape
        cover[b.z0 : b.z0 + cz, b.y0 : b.y0 + cy, b.x0 : b.x0 + cx] += 1
        np.testing.assert_array_equal(
            b.data, _data()[b.z0 : b.z0 + cz, b.y0 : b.y0 + cy, b.x0 : b.x0 + cx]
        )
    assert len(seen) == 2 * 3 * 3
    inside = cover[0:16, 4:24, 4:24]
    assert inside.min() == 1 and inside.max() == 1
    assert cover.sum() == inside.size


def test_iter_bricks_halo_and_padding(tmp_path):
    r = VolumeReader(_v3_array(tmp_path), 0)
    region = RegionCfg(start_zyx=(0, 0, 0), size_zyx=(8, 8, 8))
    bricks = list(iter_bricks(r, region, brick=8, halo=2))
    assert len(bricks) == 1
    b = bricks[0]
    assert b.data.shape == (12, 12, 12) and b.halo == 2 and b.core_shape == (8, 8, 8)
    assert b.data[:2].max() == 0
    np.testing.assert_array_equal(b.data[2:, 2:, 2:], _data()[0:10, 0:10, 0:10])


def test_iter_bricks_bad_args(tmp_path):
    r = VolumeReader(_v3_array(tmp_path), 0)
    with pytest.raises(ValueError):
        list(iter_bricks(r, RegionCfg((0, 0, 0), (4, 4, 4)), brick=0))


def test_brick_writer_roundtrip(tmp_path):
    path = str(tmp_path / "out.zarr")
    w = BrickWriter(path, ["recto", "ink"], (16, 16, 16), chunk=8, origin_zyx=(100, 0, 0))
    assert w.array.shape == (2, 16, 16, 16) and w.array.chunks == (1, 8, 8, 8)
    a = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)
    assert not w.has_brick(100, 0, 0)
    w.write(0, 100, 0, 0, a)
    assert not w.has_brick(100, 0, 0)
    w.write(1, 100, 0, 0, a + 1)
    assert w.has_brick(100, 0, 0)
    assert not w.has_brick(108, 0, 0)

    w2 = BrickWriter.open_existing(path)
    assert w2.channels == ["recto", "ink"]
    assert w2.origin_zyx == (100, 0, 0)
    assert w2.has_brick(100, 0, 0) and not w2.has_brick(108, 0, 0)
    np.testing.assert_array_equal(np.asarray(w2.array[0, 0:8, 0:8, 0:8]), a)
    np.testing.assert_array_equal(np.asarray(w2.array[1, 0:8, 0:8, 0:8]), a + 1)
    assert np.asarray(w2.array[0, 8:, :, :]).max() == 0
    assert dict(w2.array.attrs)["voxel_um"] == 2.4


def test_brick_writer_rejects_out_of_range(tmp_path):
    w = BrickWriter(str(tmp_path / "o2.zarr"), ["a"], (8, 8, 8), chunk=8)
    with pytest.raises(ValueError):
        w.write(0, 4, 0, 0, np.zeros((8, 8, 8), np.uint8))
    with pytest.raises(ValueError):
        w.write(0, 0, 0, 0, np.zeros((8, 8), np.uint8))


# --------------------------------------------------------------- R12: per-axis bounds
def test_brick_writer_rejects_each_axis_independently(tmp_path):
    """A small Z extent used to hide a Y or X overflow (tuple comparison is lexicographic)."""
    w = BrickWriter(str(tmp_path / "axes.zarr"), ["a"], (8, 8, 8), chunk=8)
    for z0, y0, x0, shape in [(0, 7, 0, (1, 2, 1)),   # Y only
                              (0, 0, 7, (1, 1, 2)),   # X only
                              (7, 0, 0, (2, 1, 1)),   # Z only
                              (0, 0, 0, (9, 9, 9))]:  # every axis
        with pytest.raises(ValueError, match="exceeds"):
            w.write(0, z0, y0, x0, np.ones(shape, np.uint8))
        assert not w.has_brick(z0, y0, x0)  # a rejected write never marks progress
    # wholly out of range, and before the origin
    with pytest.raises(ValueError, match="exceeds"):
        w.write(0, 0, 0, 9, np.ones((1, 1, 1), np.uint8))
    with pytest.raises(ValueError, match="before origin"):
        w.write(0, -1, 0, 0, np.ones((1, 1, 1), np.uint8))
    assert np.asarray(w.array[:]).sum() == 0  # nothing was persisted


def test_brick_writer_rejects_bad_channel_index(tmp_path):
    w = BrickWriter(str(tmp_path / "chan.zarr"), ["a", "b"], (8, 8, 8), chunk=8)
    for bad in (2, 5, -1):
        with pytest.raises(ValueError, match="channel index"):
            w.write(bad, 0, 0, 0, np.ones((1, 1, 1), np.uint8))
    assert not w.has_brick(0, 0, 0) and np.asarray(w.array[:]).sum() == 0


# --------------------------------------------------------------- R04: writer identity
def test_brick_writer_rejects_incompatible_reopen(tmp_path):
    from tsm.volume import WriterMismatch

    path = str(tmp_path / "reopen.zarr")
    BrickWriter(path, ["sdf"], (8, 8, 8), chunk=8, origin_zyx=(0, 0, 0))
    ok = lambda **kw: dict(dict(channels=["sdf"], shape_zyx=(8, 8, 8), chunk=8,  # noqa: E731
                               origin_zyx=(0, 0, 0)), **kw)
    for field, kw in [("shape", ok(shape_zyx=(16, 16, 16))),
                      ("channels", ok(channels=["ink"])),
                      ("channels", ok(channels=["sdf", "ink"])),
                      ("origin_zyx", ok(origin_zyx=(100, 0, 0))),
                      ("dtype", ok(dtype=np.float32)),
                      ("chunk", ok(chunk=4)),
                      ("voxel_um", ok(voxel_um=7.0))]:
        with pytest.raises(WriterMismatch) as e:
            BrickWriter(path, **kw)
        assert field in str(e.value) and "new out_dir" in str(e.value)
    # channel *order* counts too
    BrickWriter(str(tmp_path / "order.zarr"), ["a", "b"], (8, 8, 8), chunk=8)
    with pytest.raises(WriterMismatch, match="order"):
        BrickWriter(str(tmp_path / "order.zarr"), ["b", "a"], (8, 8, 8), chunk=8)
    # an identical request reopens fine and keeps its completion keys
    w = BrickWriter(path, **ok())
    w.write(0, 0, 0, 0, np.ones((8, 8, 8), np.uint8))
    assert BrickWriter(path, **ok()).has_brick(0, 0, 0)


def test_stale_done_json_is_ignored(tmp_path):
    """R04: completion keys from a different run must not skip this run's bricks."""
    import json

    path = str(tmp_path / "stale.zarr")
    w1 = BrickWriter(path, ["sdf"], (16, 16, 16), chunk=8, origin_zyx=(0, 0, 0), brick=8)
    w1.write(0, 0, 0, 0, np.full((8, 8, 8), 5, np.uint8))
    assert w1.has_brick(0, 0, 0)
    saved = json.load(open(os.path.join(path, "done.json")))
    assert saved["writer_fingerprint"] == w1.fingerprint and saved["bricks"] == {"0,0,0": [0]}

    # same store, different brick pitch -> a different writer identity
    w2 = BrickWriter(path, ["sdf"], (16, 16, 16), chunk=8, origin_zyx=(0, 0, 0), brick=16)
    assert w2.fingerprint != w1.fingerprint and not w2.has_brick(0, 0, 0)

    # a *present but different* fingerprint is what gets ignored
    d = json.load(open(os.path.join(path, "done.json")))
    with open(os.path.join(path, "done.json"), "w") as fh:
        json.dump({**d, "writer_fingerprint": "deadbeef", "bricks": {"0,0,0": [0]}}, fh)
    assert not BrickWriter(path, ["sdf"], (16, 16, 16), chunk=8, brick=8).has_brick(0, 0, 0)


def test_legacy_flat_done_json_resumes_and_is_upgraded(tmp_path, capsys):
    """R04: a mid-build production store with the old flat sidecar must resume, not restart."""
    import json

    from tsm.volume import WriterMismatch

    path = str(tmp_path / "legacy.zarr")
    BrickWriter(path, ["a", "b"], (32, 32, 32), chunk=8, origin_zyx=(64, 0, 0))
    # hand-write the pre-fingerprint format: two bricks fully done, one half done
    legacy = {"64,0,0": [0, 1], "72,0,0": [0, 1], "80,0,0": [0]}
    with open(os.path.join(path, "done.json"), "w") as fh:
        json.dump(legacy, fh)

    w = BrickWriter(path, ["a", "b"], (32, 32, 32), chunk=8, origin_zyx=(64, 0, 0))
    assert w.has_brick(64, 0, 0) and w.has_brick(72, 0, 0)
    assert not w.has_brick(80, 0, 0) and not w.has_brick(88, 0, 0)
    assert "legacy sidecar (3 bricks) accepted" in capsys.readouterr().out

    # upgraded in place, and the upgrade is stable across another reopen
    saved = json.load(open(os.path.join(path, "done.json")))
    assert saved["writer_fingerprint"] == w.fingerprint and saved["bricks"] == legacy
    w2 = BrickWriter(path, ["a", "b"], (32, 32, 32), chunk=8, origin_zyx=(64, 0, 0))
    assert w2.has_brick(72, 0, 0) and "legacy sidecar" not in capsys.readouterr().out
    w2.write(1, 80, 0, 0, np.ones((8, 8, 8), np.uint8))
    assert w2.has_brick(80, 0, 0)

    # a genuinely different run is refused loudly rather than resumed
    with pytest.raises(WriterMismatch, match="origin_zyx"):
        BrickWriter(path, ["a", "b"], (32, 32, 32), chunk=8, origin_zyx=(0, 0, 0))
    with pytest.raises(WriterMismatch, match="channels"):
        BrickWriter(path, ["a", "c"], (32, 32, 32), chunk=8, origin_zyx=(64, 0, 0))


def test_legacy_sidecar_next_to_a_new_array_is_ignored(tmp_path):
    """Nothing validates a sidecar when the store is created fresh, so it proves nothing."""
    import json

    path = str(tmp_path / "orphan.zarr")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "done.json"), "w") as fh:
        json.dump({"0,0,0": [0]}, fh)
    assert not BrickWriter(path, ["a"], (8, 8, 8), chunk=8).has_brick(0, 0, 0)
