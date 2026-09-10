"""Sliding-window engine on a synthetic local zarr with an identity net (CPU, fp32)."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import zarr

from tsm.config import RegionCfg
from tsm.limits import Budget
from tsm.sliding import (
    WindowSpec,
    gaussian_weight,
    plan_bytes,
    plan_tiles,
    run_sliding,
    window_starts,
)
from tsm.teachers import Normalizer
from tsm.volume import BrickWriter, VolumeReader

SHAPE = (320, 256, 256)


class IdentityNet:
    """Returns the input repeated to C channels (a 'net' with activation none)."""

    def __init__(self, c: int = 1):
        self.c = c
        self.calls = 0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x.repeat(1, self.c, 1, 1, 1)


def _volume(tmp_path, data: np.ndarray) -> str:
    path = str(tmp_path / "vol.zarr")
    a = zarr.create_array(store=path, shape=data.shape, chunks=(64, 64, 64), dtype="uint8")
    a[:] = data
    return path


def _random(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, 256, size=SHAPE, dtype=np.uint8)


def _run(tmp_path, data, out_tile, c=1, halo=None, step=None, name="out"):
    reader = VolumeReader(_volume(tmp_path, data), 0)
    region = RegionCfg(start_zyx=(0, 0, 0), size_zyx=SHAPE)
    writer = BrickWriter(str(tmp_path / f"{name}.zarr"), [f"c{i}" for i in range(c)], SHAPE, chunk=64)
    spec = WindowSpec(patch=64, step=step, out_tile=out_tile, halo=halo, dtype=torch.float32)
    net = IdentityNet(c)
    summary = run_sliding(
        reader, net, Normalizer("div255"), spec, region, writer, channels_out=c, activation="none",
        budget=Budget(), device="cpu", log=lambda m: None,
    )
    return writer, summary, net


def test_gaussian_weight_shape_and_clamp():
    w = gaussian_weight(64, device="cpu")
    assert w.shape == (1, 1, 64, 64, 64)
    assert float(w.max()) == pytest.approx(1.0)
    assert float(w.min()) >= 1e-3
    assert gaussian_weight(64, device="cpu") is w  # cached


def test_window_starts_cover_exactly():
    assert window_starts(64, 64, 32) == [0]
    s = window_starts(160, 64, 32)
    assert s[0] == 0 and s[-1] == 96 and all(b - a <= 32 for a, b in zip(s, s[1:]))
    with pytest.raises(ValueError):
        window_starts(32, 64, 32)


def test_plan_tiles_defaults():
    spec = WindowSpec(patch=64, out_tile=96)
    assert spec.step == 32 and spec.halo == 32
    tiles = plan_tiles(RegionCfg((0, 0, 0), SHAPE), spec)
    assert len(tiles) == 4 * 3 * 3
    for tp in tiles:
        assert min(tp.box_shape) >= 64
        assert all(o >= 32 for o in tp.core_offset)
    rows, _ = plan_bytes(spec, RegionCfg((0, 0, 0), SHAPE), 2, "softmax", 1)
    names = [r[0] for r in rows]
    assert any(n.startswith("acc") for n in names)


@pytest.mark.parametrize("out_tile", [64, 96])
def test_identity_partition_of_unity(tmp_path, out_tile):
    data = _random()
    writer, summary, net = _run(tmp_path, data, out_tile)
    got = np.asarray(writer.array[0])
    diff = np.abs(got.astype(np.int16) - data.astype(np.int16))
    assert diff.max() <= 1, f"max abs diff {diff.max()}"
    # seamless across tile boundaries: difference along boundary planes is no
    # worse than elsewhere
    for b in range(out_tile, SHAPE[1], out_tile):
        assert diff[:, b - 1 : b + 1, :].max() <= 1
    assert summary["done"] == summary["tiles"] and summary["skipped"] == 0
    assert net.calls > 0


def test_multichannel_and_resume(tmp_path):
    data = _random(1)
    writer, s1, net1 = _run(tmp_path, data, 96, c=3)
    for c in range(3):
        d = np.abs(np.asarray(writer.array[c]).astype(np.int16) - data.astype(np.int16))
        assert d.max() <= 1
    # resume: reopen the writer from disk; every tile is done -> no inference
    reader = VolumeReader(str(tmp_path / "vol.zarr"), 0)
    w2 = BrickWriter.open_existing(str(tmp_path / "out.zarr"))
    net2 = IdentityNet(3)
    s2 = run_sliding(
        reader, net2, Normalizer("div255"), WindowSpec(64, out_tile=96, dtype=torch.float32),
        RegionCfg((0, 0, 0), SHAPE), w2, 3, "none", Budget(), device="cpu", log=lambda m: None,
    )
    assert s2["skipped"] == s1["tiles"] and s2["done"] == 0 and net2.calls == 0


def test_empty_tile_shortcut(tmp_path):
    data = _random(2)
    data[:, :, 128:] = 0  # right half is air/masked
    writer, summary, net = _run(tmp_path, data, 64, halo=0)  # box == core -> exact empties
    got = np.asarray(writer.array[0])
    assert got[:, :, 128:].max() == 0
    assert summary["empty"] == 5 * 4 * 2
    assert np.abs(got[:, :, :128].astype(np.int16) - data[:, :, :128].astype(np.int16)).max() <= 1


def test_softmax_two_class_difference_trick(tmp_path):
    """Softmax fg with 2 logit channels == sigmoid(l1 - l0); acc uses one channel."""

    class TwoLogit:
        def __call__(self, x):
            return torch.cat([torch.zeros_like(x), (x - 0.5) * 8.0], dim=1)

    data = _random(3)
    reader = VolumeReader(_volume(tmp_path, data), 0)
    region = RegionCfg((0, 0, 0), (64, 64, 128))
    writer = BrickWriter(str(tmp_path / "sm.zarr"), ["fg"], (64, 64, 128), chunk=64)
    run_sliding(
        reader, TwoLogit(), Normalizer("div255"), WindowSpec(64, out_tile=64, halo=0, dtype=torch.float32),
        region, writer, 2, "softmax", Budget(), device="cpu", fg_channel=1, log=lambda m: None,
    )
    got = np.asarray(writer.array[0]).astype(np.float32) / 255.0
    exp = torch.sigmoid((torch.from_numpy(data[:64, :64, :128]).float() / 255.0 - 0.5) * 8.0).numpy()
    assert np.abs(got - exp).max() <= 1.5 / 255


def test_window_skipping_zero_region_no_nan(tmp_path):
    """Windows whose input is all zero are skipped; voxels covered only by skipped
    windows come out 0 (no NaN from wsum == 0), the rest is unchanged."""
    data = _random(4)
    data[:, :, 128:] = 0  # right half is air / masked
    reader = VolumeReader(_volume(tmp_path, data), 0)
    region = RegionCfg((0, 0, 0), SHAPE)
    outs = []
    for ewm in (-1, 0):
        writer = BrickWriter(str(tmp_path / f"skip{ewm}.zarr"), ["c0"], SHAPE, chunk=64)
        # empty_max=-1: the whole-tile shortcut is off, so only window skipping acts
        spec = WindowSpec(patch=64, out_tile=128, halo=32, dtype=torch.float32, empty_max=-1,
                          empty_window_max=ewm)
        net = IdentityNet(1)
        summary = run_sliding(reader, net, Normalizer("div255"), spec, region, writer, 1, "none",
                              Budget(), device="cpu", log=lambda m: None)
        outs.append((np.asarray(writer.array[0]), summary, net.calls))
    (full, s_full, calls_full), (skip, s_skip, calls_skip) = outs
    assert s_skip["windows"] < s_full["windows"] and calls_skip < calls_full
    assert skip[:, :, 128:].max() == 0
    assert np.array_equal(skip, full)  # identity net: zero windows contribute nothing anyway
    assert np.abs(skip[:, :, :128].astype(np.int16) - data[:, :, :128].astype(np.int16)).max() <= 1


def test_window_skipping_sigmoid_uncovered_is_zero(tmp_path):
    """With a softmax teacher, uncovered voxels must be 0 rather than sigmoid(0) = 0.5."""

    class TwoLogit:
        def __call__(self, x):
            return torch.cat([torch.zeros_like(x), (x - 0.5) * 8.0], dim=1)

    data = _random(5)
    data[:, :, 64:] = 0
    reader = VolumeReader(_volume(tmp_path, data), 0)
    region = RegionCfg((0, 0, 0), (64, 64, 128))
    writer = BrickWriter(str(tmp_path / "sm.zarr"), ["fg"], (64, 64, 128), chunk=64)
    run_sliding(
        reader, TwoLogit(), Normalizer("div255"),
        WindowSpec(64, out_tile=128, halo=0, dtype=torch.float32, empty_max=-1, empty_window_max=0),
        region, writer, 2, "softmax", Budget(), device="cpu", fg_channel=1, log=lambda m: None,
    )
    got = np.asarray(writer.array[0])
    # windows start at x = 0, 32, 64: the one at 64 is all zero and skipped, so
    # x >= 96 is covered by nothing -> exactly 0 (not sigmoid(0) = 0.5)
    assert got[:, :, 96:].max() == 0
    assert got[:, :, 64:96].max() <= round(255 * float(torch.sigmoid(torch.tensor(-4.0))))
    assert got[:, :, :64].max() > 0


def test_prefetch_matches_sequential(tmp_path):
    data = _random(6)
    reader = VolumeReader(_volume(tmp_path, data), 0)
    region = RegionCfg((0, 0, 0), SHAPE)
    results = []
    for prefetch in (False, True):
        writer = BrickWriter(str(tmp_path / f"pf{int(prefetch)}.zarr"), ["c0"], SHAPE, chunk=64)
        spec = WindowSpec(patch=64, out_tile=96, dtype=torch.float32, prefetch=prefetch)
        summary = run_sliding(reader, IdentityNet(1), Normalizer("div255"), spec, region, writer, 1, "none",
                              Budget(), device="cpu", log=lambda m: None)
        assert summary["done"] == summary["tiles"] and summary["prefetch"] is prefetch
        results.append(np.asarray(writer.array[0]))
    assert np.array_equal(results[0], results[1])


def test_pad_batch_partial_chunk(tmp_path):
    """compile/TRT paths pad the last chunk to the fixed batch; output must be unchanged."""
    data = _random(7)
    reader = VolumeReader(_volume(tmp_path, data), 0)
    region = RegionCfg((0, 0, 0), (64, 64, 192))  # 5 windows along x -> batch 2 leaves a partial chunk
    outs = []
    for backend in ("torch", "trt"):
        writer = BrickWriter(str(tmp_path / f"pb_{backend}.zarr"), ["c0"], (64, 64, 192), chunk=64)
        spec = WindowSpec(patch=64, out_tile=192, halo=0, batch=2, dtype=torch.float32, backend=backend)
        assert spec.pad_batch is (backend == "trt")
        run_sliding(reader, IdentityNet(1), Normalizer("div255"), spec, region, writer, 1, "none",
                    Budget(), device="cpu", log=lambda m: None)
        outs.append(np.asarray(writer.array[0]))
    assert np.array_equal(outs[0], outs[1])


# --------------------------------------------------------------------------- #
# Test-time augmentation
# --------------------------------------------------------------------------- #
from itertools import permutations, product  # noqa: E402

from tsm.labels import _enc_dir  # noqa: E402
from tsm.sliding import Xform, transform_lasagna_dirs, tta_transforms  # noqa: E402


def _encode_lasagna_np(n_xyz: np.ndarray) -> np.ndarray:
    """(3, N) normals in (x, y, z) -> (8, N) lasagna channels (cos = 1, grad_mag = |n|)."""
    nx, ny, nz = n_xyz
    z0, z1 = _enc_dir(nx, ny, 1e-12)
    y0, y1 = _enc_dir(nx, nz, 1e-12)
    x0, x1 = _enc_dir(ny, nz, 1e-12)
    return np.stack([np.ones_like(nx), np.sqrt(nx ** 2 + ny ** 2 + nz ** 2), z0, z1, y0, y1, x0, x1])


def test_tta_transform_sets():
    assert len(tta_transforms("none")) == 1 and len(tta_transforms("flip8")) == 8
    assert len(tta_transforms("flip4rot2")) == 8
    assert len(tta_transforms("flip8_rot4")) == 16  # 32 candidates, each group element twice
    assert tta_transforms(False) == tta_transforms("none") and tta_transforms(True) == tta_transforms("flip8")
    x = torch.randn(2, 3, 4, 6, 6)
    for xf in tta_transforms("flip8_rot4"):
        assert torch.equal(xf.invert(xf.apply(x)), x)
        m = xf.matrix()
        assert np.array_equal(m @ m.T, np.eye(3, dtype=np.int64))
    assert WindowSpec(64, tta="flip8_rot4").n_passes == 16
    with pytest.raises(ValueError):
        WindowSpec(64, tta="rot17")


def test_xform_matrix_matches_geometry():
    """The signed permutation of ``Xform.matrix`` is the actual action of apply() on vectors:
    a plane wave with normal n becomes one with normal M n."""
    rng = np.random.default_rng(0)
    p = 24
    zz, yy, xx = np.meshgrid(*(np.arange(p, dtype=np.float64) - (p - 1) / 2,) * 3, indexing="ij")
    for xf in tta_transforms("flip8_rot4"):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        vol = np.sin(0.7 * (n[0] * xx + n[1] * yy + n[2] * zz))
        moved = xf.apply(torch.from_numpy(vol)[None, None])[0, 0].numpy()
        n2 = xf.matrix() @ n
        expect = np.sin(0.7 * (n2[0] * xx + n2[1] * yy + n2[2] * zz))
        assert np.allclose(moved, expect, atol=1e-9), xf


def test_transform_lasagna_dirs_all_signed_permutations():
    rng = np.random.default_rng(1)
    n = rng.normal(size=(3, 500))
    n /= np.linalg.norm(n, axis=0)
    ch = _encode_lasagna_np(n)
    y = torch.from_numpy(ch)[None]  # [1, 8, N]
    for perm in permutations(range(3)):
        for signs in product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=np.int64)
            for a in range(3):
                m[a, perm[a]] = signs[a]
            got = transform_lasagna_dirs(y, m)[0].numpy()
            expect = _encode_lasagna_np(m @ n)
            assert np.allclose(got, expect, atol=1e-9), (perm, signs)
    with pytest.raises(ValueError):
        transform_lasagna_dirs(y, np.eye(3) * 2)


class GradientLasagnaNet:
    """Exactly equivariant 'lasagna': encodes the local CT gradient direction as the 8 channels."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gz, gy, gx = torch.gradient(x, dim=(2, 3, 4), edge_order=1)
        eps = 1e-12

        def enc(a, b):
            r2 = a * a + b * b + eps
            c2 = (a * a - b * b) / r2
            s2 = 2 * a * b / r2
            return 0.5 + 0.5 * c2, 0.5 + 0.5 * (c2 - s2) / math.sqrt(2.0)

        z0, z1 = enc(gx, gy)
        y0, y1 = enc(gx, gz)
        x0, x1 = enc(gy, gz)
        mag = torch.sqrt(gx * gx + gy * gy + gz * gz)
        return torch.cat([torch.ones_like(x), mag, z0, z1, y0, y1, x0, x1], dim=1)


def _plane_wave(shape, n, freq=0.35, seed=0):
    zz, yy, xx = np.meshgrid(*(np.arange(s, dtype=np.float64) for s in shape), indexing="ij")
    v = 0.5 + 0.45 * np.sin(freq * (n[0] * xx + n[1] * yy + n[2] * zz))
    return np.round(v * 255).astype(np.uint8)


def _run_tta(tmp_path, data, net, c, tta, field, name):
    reader = VolumeReader(_volume(tmp_path / name, data), 0)
    region = RegionCfg(start_zyx=(0, 0, 0), size_zyx=data.shape)
    writer = BrickWriter(str(tmp_path / f"{name}.zarr"), [f"c{i}" for i in range(c)], data.shape, chunk=64)
    spec = WindowSpec(patch=64, out_tile=64, halo=32, dtype=torch.float32, tta=tta, tta_field=field,
                      empty_window_max=-1)
    summary = run_sliding(reader, net, Normalizer("div255"), spec, region, writer, channels_out=c,
                          activation="none", budget=Budget(), device="cpu", log=lambda m: None)
    return np.stack([np.asarray(writer.array[i]) for i in range(c)]), summary


def test_lasagna_tta_equivariance(tmp_path):
    """A constant-normal plane wave through an exactly equivariant net: every TTA pass, once
    transformed back (spatially + dir channels), equals the plain pass, so the average is
    identical.  Ignoring the dir-channel transform (field='scalar') is measurably wrong."""
    n = np.array([0.3, -0.8, 0.52])
    n /= np.linalg.norm(n)
    data = _plane_wave((64, 96, 96), n)
    ref, s0 = _run_tta(tmp_path, data, GradientLasagnaNet(), 8, "none", "lasagna", "ref")
    assert s0["tta_passes"] == 1
    # the plain pass encodes n (median: at the wave extrema the uint8 gradient vanishes)
    expect = _encode_lasagna_np(n[:, None])[:, 0]
    mid = ref[:, 16:-16, 16:-16, 16:-16].astype(np.float64) / 255.0
    med = np.median(mid[2:].reshape(6, -1), axis=1)
    assert np.abs(med - expect[2:]).max() <= 0.03
    for mode in ("flip8", "flip4rot2", "flip8_rot4"):
        got, s = _run_tta(tmp_path, data, GradientLasagnaNet(), 8, mode, "lasagna", mode)
        assert s["tta_passes"] == len(tta_transforms(mode))
        d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
        assert d.max() <= 1, (mode, d.max())
    wrong, _ = _run_tta(tmp_path, data, GradientLasagnaNet(), 8, "flip8_rot4", "scalar", "wrong")
    d = np.abs(wrong[2:].astype(np.int16) - ref[2:].astype(np.int16))
    assert d.max() > 20


def test_flip8_identity_net_matches_no_tta(tmp_path):
    """Identity net is trivially equivariant: flip8 / flip8_rot4 averages equal the plain pass
    on a mirror-symmetric volume (and on a random one)."""
    rng = np.random.default_rng(8)
    half = rng.integers(1, 256, size=(64, 64, 32), dtype=np.uint8)
    sym = np.concatenate([half, half[:, :, ::-1]], axis=2)
    for di, data in enumerate((sym, _random(9)[:64, :64, :64])):
        ref, _ = _run_tta(tmp_path, data, IdentityNet(1), 1, "none", "scalar", f"id_ref{di}")
        for mode in ("flip8", "flip8_rot4"):
            got, _ = _run_tta(tmp_path, data, IdentityNet(1), 1, mode, "scalar", f"id_{mode}{di}")
            assert np.array_equal(got, ref)
        assert np.abs(ref[0].astype(np.int16) - data.astype(np.int16)).max() <= 1


class AsyncIdentity(IdentityNet):
    """submit/wait protocol (the TRT wrapper) on the CPU path."""

    def submit(self, x):
        return ("h", self(x))

    def wait(self, h):
        assert h[0] == "h"
        return h[1]


def test_async_submit_wait_protocol(tmp_path):
    data = _random(10)[:64, :96, :96]
    ref, _ = _run_tta(tmp_path, data, IdentityNet(2), 2, "flip8", "scalar", "sync")
    got, _ = _run_tta(tmp_path, data, AsyncIdentity(2), 2, "flip8", "scalar", "async")
    assert np.array_equal(got, ref)


def test_plan_tiles_even_partition_no_slivers():
    """Cores partition the region exactly, in near-equal tiles (no clipped slivers)."""
    size = (256, 1024, 1024)
    spec = WindowSpec(patch=64, out_tile=192)
    start = (7, 13, 5)
    tiles = plan_tiles(RegionCfg(start, size), spec)
    for a in range(3):
        edges = sorted({(tp.core0[a], tp.core0[a] + tp.core_shape[a]) for tp in tiles})
        # the per-axis cores tile [start, start + size) with no gap and no overlap
        assert edges[0][0] == start[a]
        assert edges[-1][1] == start[a] + size[a]
        assert all(b[0] == a0[1] for a0, b in zip(edges, edges[1:]))
        lens = [e1 - e0 for e0, e1 in edges]
        assert min(lens) >= spec.out_tile // 2, (a, lens)
        assert max(lens) - min(lens) <= 1, (a, lens)
        assert max(lens) <= spec.out_tile, (a, lens)
    # the full 3D core set covers the region volume exactly once
    assert sum(int(np.prod(tp.core_shape)) for tp in tiles) == int(np.prod(size))
    assert len(tiles) == math.prod((s + spec.out_tile - 1) // spec.out_tile for s in size)
