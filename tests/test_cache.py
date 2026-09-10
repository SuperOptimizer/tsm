"""Retry behaviour, the local CT cache (``tsm cache``) and local-first reads."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import zarr

from tsm.cache import padded_region, plan_levels, run_cache
from tsm.config import Budget, RegionCfg, RunCfg, VolumeCfg, parse_config
from tsm.volume import (
    LocalRegionArray,
    VolumeReader,
    VolumeReadError,
    cache_array_path,
    cache_volume_name,
    is_transient,
    open_cached_reader,
    open_local_region,
    retry_fetch,
)

SHAPE = (256, 256, 256)
START = (64, 64, 64)
SIZE = (128, 128, 128)


def _remote(tmp_path, name="remote.zarr", shape=SHAPE):
    path = str(tmp_path / name)
    a = zarr.create_array(store=path, shape=shape, chunks=(64, 64, 64), dtype="uint8", fill_value=0)
    z, y, x = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    a[:] = ((z * 7 + y * 3 + x) % 251 + 1).astype(np.uint8)  # never 0
    return path


def _cfg(tmp_path, url, cache_root=None, margin=0, extra=None, region=None):
    return RunCfg(
        volume=VolumeCfg(url=url, level=0, voxel_um=2.4,
                         cache_root=cache_root or str(tmp_path / "cache"), cache_margin=margin),
        region=region or RegionCfg(START, SIZE),
        budget=Budget(),
        out_dir=str(tmp_path / "out"),
        extra=extra if extra is not None else {"cache": {"levels": 1, "brick": 64, "probe_brick": 16}},
    )


# ----------------------------------------------------------------------------- retries
class Boom(OSError):
    pass


class _FlakyArray:
    """zarr-Array-like proxy that raises ``fails`` times before each read succeeds."""

    def __init__(self, array, fails, exc=None):
        self.array = array
        self.fails = int(fails)
        self.calls = 0
        self.exc = exc or Boom("Could not connect to the endpoint URL")
        self.shape = array.shape
        self.ndim = array.ndim
        self.chunks = array.chunks
        self.dtype = array.dtype

    def __getitem__(self, sel):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc
        return self.array[sel]


def test_is_transient_classification():
    import botocore.exceptions as be

    assert is_transient(be.EndpointConnectionError(endpoint_url="http://x"))
    assert is_transient(ConnectionResetError("reset by peer"))
    assert is_transient(RuntimeError("read timeout while fetching chunk"))
    assert is_transient(Exception("Could not connect to the Endpoint URL"))
    assert not is_transient(KeyError("0/1/2"))
    assert not is_transient(ValueError("bad dtype"))


def test_retry_fetch_succeeds_after_failures():
    slept = []
    state = {"n": 0}

    def fn():
        state["n"] += 1
        if state["n"] < 4:
            raise Boom("connection reset by peer")
        return "ok"

    assert retry_fetch(fn, "x", attempts=6, base=1.0, sleep=slept.append) == "ok"
    assert slept == [1.0, 2.0, 4.0]


def test_retry_fetch_gives_up_with_volume_read_error():
    slept = []

    def fn():
        raise Boom("EndpointConnectionError: could not connect")

    with pytest.raises(VolumeReadError) as e:
        retry_fetch(fn, "chunk 1/2/3", attempts=4, base=1.0, sleep=slept.append)
    assert "chunk 1/2/3" in str(e.value) and slept == [1.0, 2.0, 4.0]


def test_retry_fetch_does_not_retry_real_errors():
    calls = []

    def fn():
        calls.append(1)
        raise KeyError("missing chunk")

    with pytest.raises(KeyError):
        retry_fetch(fn, "x", attempts=6, base=0.0, sleep=lambda s: None)
    assert len(calls) == 1


def test_reader_read_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("tsm.volume.time.sleep", lambda s: None)
    path = _remote(tmp_path, shape=(64, 64, 64))
    src = zarr.open_array(store=path, mode="r")
    flaky = _FlakyArray(src, fails=2)
    r = VolumeReader(path, 0, 2.4, array=flaky)
    got = r.read(0, 8, 0, 8, 0, 8)
    np.testing.assert_array_equal(got, np.asarray(src[0:8, 0:8, 0:8]))
    assert flaky.calls == 3


def test_reader_read_raises_volume_read_error(tmp_path, monkeypatch):
    monkeypatch.setattr("tsm.volume.time.sleep", lambda s: None)
    path = _remote(tmp_path, shape=(64, 64, 64))
    flaky = _FlakyArray(zarr.open_array(store=path, mode="r"), fails=99)
    r = VolumeReader(path, 0, 2.4, array=flaky)
    with pytest.raises(VolumeReadError):
        r.read(0, 8, 0, 8, 0, 8)


# ----------------------------------------------------------------------------- LocalRegionArray
def test_local_region_array_maps_and_pads(tmp_path):
    path = _remote(tmp_path)
    src = np.asarray(zarr.open_array(store=path, mode="r")[:])
    local_path = str(tmp_path / "local.zarr")
    a = zarr.create_array(store=local_path, shape=SIZE, chunks=(64,) * 3, dtype="uint8")
    a[:] = src[64:192, 64:192, 64:192]
    loc = LocalRegionArray(a, START, SHAPE, path=local_path)

    assert loc.shape == SHAPE and loc.local_shape == SIZE and loc.chunks == (64, 64, 64)
    assert loc.covers((64, 64, 64), (192, 192, 192))
    assert not loc.covers((0, 64, 64), (192, 192, 192))
    # absolute coordinates inside the footprint
    np.testing.assert_array_equal(loc[70:80, 90:100, 64:70], src[70:80, 90:100, 64:70])
    # straddling the footprint: outside is zero, inside is real
    out = loc[60:70, 64:70, 64:70]
    assert out.shape == (10, 6, 6)
    assert out[:4].max() == 0
    np.testing.assert_array_equal(out[4:], src[64:70, 64:70, 64:70])
    # entirely outside
    assert loc[0:8, 0:8, 0:8].max() == 0

    strict = LocalRegionArray(a, START, SHAPE, strict=True, path=local_path)
    with pytest.raises(VolumeReadError):
        strict[0:8, 0:8, 0:8]


def test_volume_reader_over_local_region_matches_remote(tmp_path):
    path = _remote(tmp_path)
    remote = VolumeReader(path, 0, 2.4)
    local_path = str(tmp_path / "local2.zarr")
    a = zarr.create_array(store=local_path, shape=SIZE, chunks=(64,) * 3, dtype="uint8")
    a[:] = np.asarray(zarr.open_array(store=path, mode="r")[64:192, 64:192, 64:192])
    r = VolumeReader(path, 0, 2.4, array=LocalRegionArray(a, START, SHAPE))
    assert r.shape == SHAPE and r.chunks == (64, 64, 64)
    for box in ((64, 96, 64, 96, 64, 96), (100, 132, 70, 102, 180, 190)):
        np.testing.assert_array_equal(r.read(*box), remote.read(*box))


def test_cache_volume_name_matches_for_primary_and_alt():
    a = "https://dl.ash2txt.org/community-uploads/x/volumes/v-masked.zarr"
    b = "s3://vesuvius-challenge-open-data/PHercParis4/volumes/v-masked.zarr"
    assert cache_volume_name(a) == cache_volume_name(b) == "v-masked"
    assert cache_array_path("/root", b, 2).endswith("/root/v-masked/L2.zarr")


# ----------------------------------------------------------------------------- plan
def test_plan_levels_and_margin(tmp_path):
    cfg = _cfg(tmp_path, "s3://bucket/vol.zarr", margin=64,
               extra={"cache": {"levels": 3, "brick": 128}})
    pad = padded_region(cfg)
    assert pad.start_zyx == (0, 0, 0)  # start 64 - margin 64, clipped at 0
    assert pad.size_zyx == (256, 256, 256)
    plans = plan_levels(cfg)
    assert [p.level for p in plans] == [0, 1, 2]
    assert [p.size_zyx for p in plans] == [(256, 256, 256), (128, 128, 128), (64, 64, 64)]
    assert [p.origin_zyx for p in plans] == [(0, 0, 0)] * 3
    assert plans[0].path.endswith("/vol/L0.zarr") and plans[2].path.endswith("/vol/L2.zarr")
    assert plans[0].bytes == 256 ** 3


def test_config_accepts_cache_keys():
    raw = {
        "volume": {"url": "s3://b/v.zarr", "cache_root": "/home/forrest/tsm-cache", "cache_margin": 32},
        "region": {"start_zyx": [0, 0, 0], "size_zyx": [8, 8, 8]},
        "out_dir": "/tmp/x",
    }
    cfg = parse_config(raw)
    assert cfg.volume.cache_root == "/home/forrest/tsm-cache" and cfg.volume.cache_margin == 32
    assert parse_config({**raw, "volume": {"url": "s3://b/v.zarr"}}).volume.cache_root is None
    with pytest.raises(ValueError):
        parse_config({**raw, "volume": {"url": "s3://b/v.zarr", "cache_root": 3}})
    with pytest.raises(ValueError):
        parse_config({**raw, "volume": {"url": "s3://b/v.zarr", "cache_margin": -1}})


# ----------------------------------------------------------------------------- cache stage
def test_cache_dry_run_writes_nothing(tmp_path, capsys):
    cfg = _cfg(tmp_path, _remote(tmp_path))
    out = run_cache(cfg, dry_run=True)
    assert out["bytes"] == 128 ** 3
    assert "128" in capsys.readouterr().out
    assert not os.path.exists(cfg.volume.cache_root)


def test_cache_copies_region_and_resumes(tmp_path, capsys):
    url = _remote(tmp_path)
    cfg = _cfg(tmp_path, url)
    summary = run_cache(cfg)
    assert summary["levels"][0]["bricks"] == 8  # 128^3 region in 64^3 bricks

    path = cache_array_path(cfg.volume.cache_root, url, 0)
    arr = zarr.open_array(store=path, mode="r")
    src = np.asarray(zarr.open_array(store=url, mode="r")[:])
    assert arr.shape == SIZE and arr.chunks == (128, 128, 128) and arr.dtype == np.uint8
    np.testing.assert_array_equal(np.asarray(arr[:]), src[64:192, 64:192, 64:192])
    a = dict(arr.attrs)
    assert a["source_url"] == url and a["level"] == 0 and a["origin_zyx"] == list(START)
    assert a["voxel_um"] == 2.4 and a["full_shape_zyx"] == list(SHAPE)

    # resume: nothing left to do
    summary2 = run_cache(cfg)
    assert summary2["levels"][0]["bricks"] == 0
    assert "8/8 bricks already cached" in capsys.readouterr().out

    # done.json lost with a *sub-chunk* brick (64 in a 128 chunk): the chunk files cannot
    # prove completion (several bricks share one chunk file), so everything runs again (R03)
    import json
    import shutil

    dp = os.path.join(path, "done.json")
    keys = json.load(open(dp))
    assert len(keys) == 8
    os.remove(dp)
    assert run_cache(cfg)["levels"][0]["bricks"] == 8

    # bricks that are in neither done.json nor on disk run again
    json.dump(keys[:4], open(dp, "w"))
    shutil.rmtree(os.path.join(path, "c"))
    assert run_cache(cfg)["levels"][0]["bricks"] == 4

    # --force redoes everything and restores the full box
    assert run_cache(cfg, force=True)["levels"][0]["bricks"] == 8
    np.testing.assert_array_equal(np.asarray(zarr.open_array(store=path, mode="r")[:]),
                                  src[64:192, 64:192, 64:192])


def test_cache_reader_matches_remote_and_open_ct_uses_it(tmp_path):
    from tsm.cli import open_ct

    url = _remote(tmp_path)
    cfg = _cfg(tmp_path, url)
    run_cache(cfg)

    r = open_ct(cfg, 0, probe=16)
    assert isinstance(r.array, LocalRegionArray)
    remote = VolumeReader(url, 0, 2.4)
    box = (70, 102, 80, 112, 90, 122)
    np.testing.assert_array_equal(r.read(*box), remote.read(*box))
    # reads outside the cached box fall back to zeros (still inside the volume)
    assert r.read(0, 8, 0, 8, 0, 8).max() == 0

    # a region the cache does not cover falls back to the remote volume
    wide = _cfg(tmp_path, url, cache_root=cfg.volume.cache_root,
                region=RegionCfg((0, 0, 0), (256, 256, 256)))
    assert open_cached_reader(wide.volume.cache_root, url, 0, 2.4, wide.region) is None
    assert not isinstance(open_ct(wide, 0, probe=16).array, LocalRegionArray)


def test_cache_second_disjoint_region_gets_its_own_array(tmp_path):
    url = _remote(tmp_path)
    root = str(tmp_path / "cache")
    run_cache(_cfg(tmp_path, url, cache_root=root))
    other = _cfg(tmp_path, url, cache_root=root, region=RegionCfg((192, 0, 0), (64, 64, 64)))
    run_cache(other)
    base = cache_array_path(root, url, 0)
    assert open_local_region(base).origin == START
    second = cache_array_path(root, url, 0, (192, 0, 0))
    assert second != base and open_local_region(second).origin == (192, 0, 0)
    # each region finds its own cached box
    assert open_cached_reader(root, url, 0, 2.4, other.region).array.origin == (192, 0, 0)
    assert open_cached_reader(root, url, 0, 2.4, RegionCfg(START, SIZE)).array.origin == START


# ----------------------------------------------------------------------------- dataset resilience
def _synth(tmp_path):
    from tests.synth import make_synthetic

    return make_synthetic(str(tmp_path / "synth"))


class _FlakyReader:
    """VolumeReader stand-in that fails the first ``fails`` reads."""

    def __init__(self, inner, fails):
        self.inner = inner
        self.fails = int(fails)
        self.calls = 0

    def read(self, *box):
        self.calls += 1
        if self.calls <= self.fails:
            raise VolumeReadError("read failed after 6 attempts: EndpointConnectionError")
        return self.inner.read(*box)


def test_dataset_substitutes_origin_on_read_failure(tmp_path, capsys):
    from tsm.data import CropDataset

    s = _synth(tmp_path)
    inner = VolumeReader(s["ct"], 0, 2.4)
    ds = CropDataset(_FlakyReader(inner, fails=2), s["fine"], None, patch=32, stride=16,
                     length=4, augment=False)
    item = ds[0]
    assert tuple(item["input"].shape) == (2, 32, 32, 32)
    assert ds.read_failures == 2
    assert "retrying at another random origin" in capsys.readouterr().out


def test_dataset_gives_up_after_bounded_substitutions(tmp_path):
    from tsm.data import CropDataset

    s = _synth(tmp_path)
    inner = VolumeReader(s["ct"], 0, 2.4)
    ds = CropDataset(_FlakyReader(inner, fails=999), s["fine"], None, patch=32, stride=16,
                     length=4, augment=False)
    with pytest.raises(VolumeReadError):
        ds[0]
    assert ds.read_failures == CropDataset.MAX_READ_SUBSTITUTIONS + 1


# --------------------------------------------------------------- R02/R03/R04: completion
def _partial_cache(tmp_path, url, size=(128, 128, 128), brick_value=7, keys=("0,0,0",),
                   root=None, box=(64, 64, 64)):
    """A cache array at its final shape with only its first ``box`` brick written."""
    root = root or str(tmp_path / "cache")
    path = cache_array_path(root, url, 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = zarr.create_array(
        store=path, shape=size, chunks=(128,) * 3, dtype=np.uint8, fill_value=0,
        attributes={"origin_zyx": [0, 0, 0], "size_zyx": list(size), "full_shape_zyx": list(size),
                    "level": 0, "voxel_um": 2.4, "source_url": url})
    arr[: box[0], : box[1], : box[2]] = brick_value
    with open(os.path.join(path, "done.json"), "w") as fh:
        json.dump(list(keys), fh)
    return root, path, arr


def test_partial_cache_is_a_miss_and_complete_manifest_is_a_hit(tmp_path, capsys):
    """R02: an interrupted build (array at full shape, one brick written) must not be a HIT."""
    url = "https://example.org/vol/TESTVOL.zarr"
    root, path, arr = _partial_cache(tmp_path, url)
    region = RegionCfg((0, 0, 0), (128, 128, 128))

    assert open_cached_reader(root, url, 0, 2.4, region) is None
    out = capsys.readouterr().out
    assert "not proven complete" in out and "does not pin down a brick pitch" in out
    assert not os.path.exists(os.path.join(path, "complete.json"))

    # a partially listed multi-brick grid is explicitly reported as incomplete
    with open(os.path.join(path, "done.json"), "w") as fh:
        json.dump(["0,0,0", "0,0,64"], fh)
    assert open_cached_reader(root, url, 0, 2.4, region) is None
    assert "incomplete" in capsys.readouterr().out

    # the same array once every brick is listed: complete.json is derived and the cache hits
    with open(os.path.join(path, "done.json"), "w") as fh:
        json.dump([f"{z},{y},{x}" for z in (0, 64) for y in (0, 64) for x in (0, 64)], fh)
    r = open_cached_reader(root, url, 0, 2.4, region)
    assert r is not None
    info = json.load(open(os.path.join(path, "complete.json")))
    assert info["brick"] == 64 and info["size_zyx"] == [128, 128, 128] and info["levels"] == [0]
    assert info["source_url"] == url and "written" in info
    # ... and it stays a hit on the cheap file check
    assert open_cached_reader(root, url, 0, 2.4, region) is not None


def test_complete_manifest_footprint_must_cover_the_request(tmp_path):
    url = "https://example.org/vol/COVER.zarr"
    root, path, _ = _partial_cache(tmp_path, url, size=(128, 128, 128))
    from tsm.volume import write_complete

    write_complete(path, 0, (0, 0, 0), (64, 128, 128), 64, url)  # only half committed
    assert open_cached_reader(root, url, 0, 2.4, RegionCfg((0, 0, 0), (128, 128, 128))) is None
    assert open_cached_reader(root, url, 0, 2.4, RegionCfg((0, 0, 0), (64, 128, 128))) is not None


def test_sub_chunk_brick_interruption_does_not_mark_neighbours_done(tmp_path, monkeypatch):
    """R03: one 64^3 brick written into a 128^3 chunk must not complete the other seven."""
    import tsm.cache as C

    url = "https://example.org/vol/TESTVOL.zarr"
    root, path, arr = _partial_cache(tmp_path, url)

    class FakeReader:
        shape = (128, 128, 128)
        url = "fake"

        def read(self, z0, z1, y0, y1, x0, x1):
            return np.full((z1 - z0, y1 - y0, x1 - x0), 7, np.uint8)

    monkeypatch.setattr(C, "_open_source", lambda cfg, plan, probe: FakeReader())
    cfg = _cfg(tmp_path, url, cache_root=root, region=RegionCfg((0, 0, 0), (128, 128, 128)),
               extra={"cache": {"levels": 1, "brick": 64}})
    summary = run_cache(cfg)
    assert summary["levels"][0]["bricks"] == 7  # the seven unwritten sub-bricks
    assert int(np.asarray(arr[100, 100, 100])) == 7
    assert os.path.exists(os.path.join(path, "complete.json"))


def test_chunk_aligned_brick_still_resumes_from_chunk_files(tmp_path):
    """The fallback survives where it is sound: brick 128 == the storage chunk."""
    url = _remote(tmp_path)
    cfg = _cfg(tmp_path, url, extra={"cache": {"levels": 1, "brick": 128, "probe_brick": 16}})
    assert run_cache(cfg)["levels"][0]["bricks"] == 1
    path = cache_array_path(cfg.volume.cache_root, url, 0)
    os.remove(os.path.join(path, "done.json"))
    assert run_cache(cfg)["levels"][0]["bricks"] == 0
    assert os.path.exists(os.path.join(path, "complete.json"))


def test_cache_refuses_to_grow_an_existing_array(tmp_path, monkeypatch):
    """R04: a larger region must not report success while the store stays small."""
    import tsm.cache as C

    url = "https://example.org/vol/GROW.zarr"
    root, path, arr = _partial_cache(tmp_path, url, box=(128, 128, 128), brick_value=3)

    class FakeReader:
        shape = (1024, 1024, 1024)
        url = "fake"

        def read(self, z0, z1, y0, y1, x0, x1):
            return np.full((z1 - z0, y1 - y0, x1 - x0), 9, np.uint8)

    monkeypatch.setattr(C, "_open_source", lambda cfg, plan, probe: FakeReader())
    cfg = _cfg(tmp_path, url, cache_root=root, region=RegionCfg((0, 0, 0), (256, 128, 128)),
               extra={"cache": {"levels": 1, "brick": 128}})
    with pytest.raises(RuntimeError, match="shape"):
        run_cache(cfg)
    assert zarr.open_array(store=path, mode="r").shape == (128, 128, 128)
