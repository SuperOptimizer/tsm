import itertools
import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from synth import AXIS_YX, PERIOD_FINE, coarse_fields, fine_fields, make_synthetic
from tsm.data import (
    CLIP,
    TARGET_KEYS,
    CropDataset,
    LabelStore,
    augment,
    build_origins,
    decode_density,
    decode_prob,
    decode_sdf,
    decode_signed,
    encode_density,
    encode_prob,
    encode_sdf,
    encode_signed,
    intensity_augment,
    spatial_transform,
    store_factor,
    upsample_coarse_targets,
)
from tsm.volume import VolumeReader


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic(str(tmp_path_factory.mktemp("synth")))


# ----------------------------------------------------------------------------- encodings
def test_sdf_round_trip():
    v = np.linspace(-CLIP, CLIP, 255, dtype=np.float32)
    u = encode_sdf(v)
    assert u.dtype == np.uint8 and u.min() >= 1  # 0 reserved for no data
    assert np.abs(decode_sdf(u) - v).max() <= CLIP / 127 / 2 + 1e-5
    assert decode_sdf(np.array([128], np.uint8))[0] == 0.0
    assert encode_sdf(np.array([100.0]))[0] == 255 and encode_sdf(np.array([-100.0]))[0] == 1


def test_signed_density_prob_round_trips():
    v = np.linspace(-1, 1, 101, dtype=np.float32)
    assert np.abs(decode_signed(encode_signed(v)) - v).max() <= 0.5 / 127.5 + 1e-6
    assert encode_signed(np.array([0.0]))[0] in (127, 128)
    d = np.array([0.0, 0.05, 0.1234, 0.255, 0.3], np.float32)
    assert np.abs(decode_density(encode_density(d))[:4] - d[:4]).max() <= 0.0005 + 1e-6
    assert encode_density(d)[4] == 255
    p = np.linspace(0, 1, 50, dtype=np.float32)
    assert np.abs(decode_prob(encode_prob(p)) - p).max() <= 0.5 / 255 + 1e-6


# ----------------------------------------------------------------------------- store + origins
def test_label_store_reads_with_padding(synth):
    st = LabelStore(synth["fine"])
    assert st.channels == ["sdf", "sdf_valid", "ink", "ink_valid"]
    assert st.shape_zyx == (64, 64, 64) and st.origin_zyx == (16, 32, 32) and st.voxel_um == 2.4
    full = np.asarray(st.array[1])
    got = st.read("sdf_valid", 0, 0, 0, 16)
    np.testing.assert_array_equal(got, full[:16, :16, :16])
    edge = st.read("sdf_valid", -4, 60, 60, 8)
    assert edge.shape == (8, 8, 8) and edge[:4].max() == 0 and edge[:, 4:].max() == 0
    np.testing.assert_array_equal(edge[4:, :4, :4], full[0:4, 60:64, 60:64])
    with pytest.raises(KeyError):
        st.read("nope", 0, 0, 0, 8)


def test_build_origins_threshold_and_cache(synth, tmp_path):
    st = LabelStore(synth["fine"])
    valid = np.asarray(st.array[1]) == 1
    patch, stride = 32, 16
    o = build_origins(st, "sdf_valid", patch, stride, min_frac=0.05, cache_dir=str(tmp_path))
    assert o.dtype == np.int32 and o.ndim == 2 and o.shape[1] == 3 and len(o) > 0
    # brute force reference
    ref = []
    for z in range(0, 64 - patch + 1, stride):
        for y in range(0, 64 - patch + 1, stride):
            for x in range(0, 64 - patch + 1, stride):
                if valid[z:z + patch, y:y + patch, x:x + patch].mean() >= 0.05:
                    ref.append((z, y, x))
    assert sorted(map(tuple, o.tolist())) == sorted(ref)
    cache = os.path.join(str(tmp_path), f"origins_fine_p{patch}_s{stride}_sdf_valid_f0.05.npy")
    assert os.path.exists(cache) and os.path.exists(cache[:-4] + ".json")
    o2 = build_origins(st, "sdf_valid", patch, stride, min_frac=0.05, cache_dir=str(tmp_path))
    np.testing.assert_array_equal(o, o2)
    # a strict threshold excludes everything
    o3 = build_origins(st, "sdf_valid", patch, stride, min_frac=1.01, cache_dir=str(tmp_path), name="strict")
    assert len(o3) == 0
    with pytest.raises(ValueError):
        build_origins(st, "sdf_valid", 32, 24, cache_dir=str(tmp_path))


def test_build_origins_thin_store_is_centred(tmp_path):
    from synth import COARSE_CHANNELS, coarse_fields, encode_coarse, _write_store

    cf = coarse_fields((12, 64, 64))
    path = _write_store(str(tmp_path / "thin.zarr"), COARSE_CHANNELS, encode_coarse(cf), (0, 0, 0), 9.6, 4.0, 32)
    st = LabelStore(path)
    o = build_origins(st, "valid", 32, 16, min_frac=0.05, cache_dir=str(tmp_path))
    assert len(o) > 0 and set(o[:, 0].tolist()) == {(12 - 32) // 2}
    assert o[:, 1].min() >= 0 and o[:, 1].max() <= 32
    crop = st.read("valid", int(o[0, 0]), int(o[0, 1]), int(o[0, 2]), 32)
    assert crop[:10].max() == 0 and crop[22:].max() == 0 and crop[10:22].max() > 0


def test_build_origins_small_blocks_match(synth, tmp_path):
    st = LabelStore(synth["fine"])
    a = build_origins(st, "sdf_valid", 32, 16, cache_dir=str(tmp_path), name="a", block=16)
    b = build_origins(st, "sdf_valid", 32, 16, cache_dir=str(tmp_path), name="b", block=4096)
    np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------------- augmentation
ALL_TRANSFORMS = list(itertools.product([False, True], [False, True], [False, True], range(4)))


def _np_transform(a: np.ndarray, flips, k) -> np.ndarray:
    """Reference: np.flip on spatial axes then np.rot90 in the (y, x) plane, for (C, Z, Y, X)."""
    for ax in range(3):
        if flips[ax]:
            a = np.flip(a, axis=ax + 1)
    return np.ascontiguousarray(np.rot90(a, k, axes=(2, 3)))


@pytest.mark.parametrize("fz,fy,fx,k", ALL_TRANSFORMS)
def test_spatial_transform_matches_numpy_and_normal_is_equivariant(fz, fy, fx, k):
    rng = np.random.default_rng(1)
    Z, Y, X = 6, 10, 12
    # smooth scalar potential; normal = its gradient (nx, ny, nz) = (df/dx, df/dy, df/dz)
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    f = np.sin(x / 3.0) + np.cos(y / 4.0 + 0.3) + 0.5 * np.sin(z / 2.0 + 1.0) + 0.1 * x * y / 20.0
    gz, gy, gx = np.gradient(f)
    normal = np.stack([gx, gy, gz]).astype(np.float32)
    scalar = rng.random((2, Z, Y, X)).astype(np.float32)
    out = spatial_transform({"f": f[None], "s": scalar, "normal": normal}, (fz, fy, fx), k)
    flips = (fz, fy, fx)
    np.testing.assert_array_equal(out["s"], _np_transform(scalar, flips, k))
    np.testing.assert_allclose(out["f"], _np_transform(f[None], flips, k))
    # gradient of the transformed potential == transformed normal
    tgz, tgy, tgx = np.gradient(out["f"][0])
    np.testing.assert_allclose(out["normal"], np.stack([tgx, tgy, tgz]), atol=1e-5)


@pytest.mark.parametrize("fz,fy,fx,k", ALL_TRANSFORMS)
def test_encode_then_augment_equals_augment_then_encode(fz, fy, fx, k):
    rng = np.random.default_rng(2)
    n = rng.normal(size=(3, 4, 6, 8)).astype(np.float32)
    n /= np.linalg.norm(n, axis=0, keepdims=True)
    a = spatial_transform({"normal": decode_signed(encode_signed(n))}, (fz, fy, fx), k)["normal"]
    b = decode_signed(encode_signed(spatial_transform({"normal": n}, (fz, fy, fx), k)["normal"]))
    assert np.abs(a - b).max() <= 1.0 / 127.5 + 1e-6  # at most one u8 rounding step


def test_augment_sample_keeps_scalars_and_moves_normal(synth):
    cf = coarse_fields((8, 8, 8), origin=(0, 40, 40))
    w = np.stack([cf["phase_sin"], cf["phase_cos"], cf["density"], cf["nx"], cf["ny"], cf["nz"]])
    sample = {"ct": cf["ct"][None].astype(np.float32) / 255.0, "winding": w, "winding_conf": cf["conf"][None],
              "winding_valid": cf["valid"][None].astype(np.float32)}
    rng = np.random.default_rng(0)
    out = augment(sample, rng, intensity=False)
    flips, k = out["aug"]["flips"], out["aug"]["rot_k"]
    ref = spatial_transform({"w": w}, flips, k, normal_key=None)["w"]
    np.testing.assert_array_equal(out["winding"][:3], ref[:3])  # sin, cos, density are plain scalars
    np.testing.assert_array_equal(out["winding_valid"], spatial_transform({"v": sample["winding_valid"]}, flips, k, None)["v"])
    # the normal stays radial (outward from the transformed axis position) after the transform
    n = out["winding"][3:6]
    assert np.allclose(np.linalg.norm(n, axis=0), 1.0, atol=1e-5)


def test_intensity_augment_bounds():
    rng = np.random.default_rng(3)
    ct = rng.random((1, 8, 8, 8)).astype(np.float32)
    x = intensity_augment(ct, rng, p=1.0)
    assert x.shape == ct.shape and x.dtype == np.float32
    assert np.abs(x - ct).mean() < 0.5


# ----------------------------------------------------------------------------- coarse -> fine upsampling
def _box(w, conf, valid):
    return {"winding": w.astype(np.float32), "conf": conf.astype(np.float32), "valid": valid.astype(np.float32)}


def test_upsample_constant_field_round_trips():
    f, n = 4, 6  # coarse box (n, n, n) incl. 1 margin -> fine (16, 16, 16)
    vec = np.array([0.6, 0.8, 0.05, 0.0, 0.6, 0.8], np.float32)
    w = np.broadcast_to(vec[:, None, None, None], (6, n, n, n)).copy()
    out = upsample_coarse_targets(_box(w, np.full((1, n, n, n), 0.7), np.ones((1, n, n, n))), f, (16, 16, 16))
    assert out["winding"].shape == (6, 16, 16, 16) and out["winding_conf"].shape == (1, 16, 16, 16)
    exp = vec.copy()
    exp[2] /= f
    np.testing.assert_allclose(out["winding"], np.broadcast_to(exp[:, None, None, None], out["winding"].shape), atol=1e-5)
    assert np.allclose(out["winding_conf"], 0.7) and np.all(out["winding_valid"] == 1)


def test_upsample_linear_phase_ramp_keeps_slope_over_factor():
    f, n = 4, 6
    # phase w = x_c * k (wraps per coarse voxel) along coarse x; slope in fine voxels must be k / f
    k = 0.03
    xc = np.arange(n, dtype=np.float32)[None, None, :]
    ph = 2 * np.pi * k * np.broadcast_to(xc, (n, n, n))
    w = np.stack([np.sin(ph), np.cos(ph), np.full((n, n, n), k), np.ones((n, n, n)), np.zeros((n, n, n)), np.zeros((n, n, n))])
    out = upsample_coarse_targets(_box(w, np.ones((1, n, n, n)), np.ones((1, n, n, n))), f, (16, 16, 16))
    s_, c_ = out["winding"][0], out["winding"][1]
    ang = np.unwrap(np.arctan2(s_, c_), axis=-1)
    slope = np.diff(ang, axis=-1).mean() / (2 * np.pi)
    assert slope == pytest.approx(k / f, rel=0.05)
    assert np.allclose(np.hypot(s_, c_), 1.0, atol=1e-5)
    np.testing.assert_allclose(out["winding"][2], k / f, atol=1e-6)
    # fine voxel j sits at coarse coordinate (j - 1.5)/4 + 1: check one exact centre (j=6 -> coarse 2.125)
    assert ang[8, 8, 6] / (2 * np.pi) == pytest.approx(k * 2.125, abs=2e-3)


def test_upsample_masks_invalid_coarse_voxels_and_nearest_masks():
    f, n = 2, 4
    w = np.ones((6, n, n, n), np.float32)
    w[:, :, :, 2:] = -5.0  # garbage in the right half
    valid = np.ones((1, n, n, n))
    valid[:, :, :, 2:] = 0
    conf = np.where(valid == 1, 0.9, 0.1).astype(np.float32)
    out = upsample_coarse_targets(_box(w, conf, valid), f, (4, 4, 4))
    # left fine half (coarse x=1 only, after the margin) is influenced by valid voxels only
    assert np.allclose(out["winding"][2][:, :, :2], 1.0 / f, atol=1e-5)
    assert np.all(out["winding_valid"][0][:, :, :2] == 1) and np.all(out["winding_valid"][0][:, :, 2:] == 0)
    assert np.all(out["winding_conf"][0][:, :, :2] == pytest.approx(0.9))


def test_store_factor(synth):
    assert store_factor(LabelStore(synth["fine"]), LabelStore(synth["coarse"])) == 4


# ----------------------------------------------------------------------------- dataset
def _dataset(synth, augment=True, **kw):
    return CropDataset(lambda: VolumeReader(synth["ct"], 0, 2.4), synth["fine"], synth["coarse"],
                       patch=32, seed=0, length=64, stride=16, augment=augment, **kw)


def test_dataset_item_shapes_and_values(synth):
    ds = _dataset(synth, augment=False)
    assert ds.factor == 4
    origin = ds.origins[0]
    it = ds.to_tensors(ds.load(origin))
    assert tuple(it["input"].shape) == (2, 32, 32, 32) and it["input"].dtype == torch.float32
    for k, c in TARGET_KEYS.items():
        assert tuple(it[k].shape) == (c, 32, 32, 32), k
    assert it["voxel_um"] == pytest.approx(2.4) and torch.all(it["input"][1] == 0.0)
    assert 0.0 <= it["input"][0].min() and it["input"][0].max() <= 1.0
    assert (it["surface_valid"] == 1).float().mean() >= 0.05
    assert it["surface_sdf"].abs().max() <= CLIP + 1e-4
    ff = fine_fields((64, 64, 64))
    z0, y0, x0 = (int(v) for v in origin)
    m = it["surface_valid"][0].numpy() == 1
    ref = ff["sdf"][z0:z0 + 32, y0:y0 + 32, x0:x0 + 32]
    assert np.abs(it["surface_sdf"][0].numpy() - ref)[m].max() <= CLIP / 127 / 2 + 1e-4
    np.testing.assert_allclose(it["input"][0].numpy(), ff["ct"][z0:z0 + 32, y0:y0 + 32, x0:x0 + 32] / 255.0, atol=1e-6)
    # winding targets come from the coarse store, upsampled x4: compare with the analytic fine-grid field
    wv = it["winding_valid"][0].numpy() == 1
    assert wv.mean() > 0.3
    gz, gy, gx = (int(v) for v in it["origin_zyx"])
    yy, xx = np.meshgrid(np.arange(gy, gy + 32), np.arange(gx, gx + 32), indexing="ij")
    dy, dx = yy - AXIS_YX[0], xx - AXIS_YX[1]
    r = np.hypot(dy, dx)
    n_ref = np.stack([dx / r, dy / r])[:, None].repeat(32, axis=1)  # (2, z, y, x)
    n_got = it["winding"][3:5].numpy()
    cosang = (n_ref * n_got).sum(0)
    assert np.percentile(cosang[wv], 5) > 0.98
    np.testing.assert_allclose(it["winding"][2].numpy()[wv], 1.0 / PERIOD_FINE, atol=2e-3)
    ang_ref = 2 * np.pi * r / PERIOD_FINE
    ang_got = np.arctan2(it["winding"][0].numpy(), it["winding"][1].numpy())
    d = np.angle(np.exp(1j * (ang_got - ang_ref[None].repeat(32, axis=0))))
    assert np.percentile(np.abs(d[wv]), 50) < 0.35  # coarse phase is sampled every 4 fine voxels (1/6 wrap)


def test_dataset_getitem_deterministic_and_worker_processes(synth):
    ds = _dataset(synth)
    a, b = ds[3], ds[3]
    torch.testing.assert_close(a["input"], b["input"])
    torch.testing.assert_close(a["winding"], b["winding"])
    dl = DataLoader(ds, batch_size=2, num_workers=1, prefetch_factor=2)
    batch = next(iter(dl))
    assert tuple(batch["input"].shape) == (2, 2, 32, 32, 32) and tuple(batch["winding"].shape) == (2, 6, 32, 32, 32)


def test_dataset_without_coarse_store(synth):
    ds = CropDataset(lambda: VolumeReader(synth["ct"], 0, 2.4), synth["fine"], None, patch=32, stride=16, length=4)
    it = ds[0]
    assert ds.factor == 1 and it["winding_valid"].max() == 0 and it["surface_valid"].max() > 0


def _slab_origins(patch=128, stride=64, size=(256, 6144, 6144), yx_stride=None):
    """The origin grid of the paris4 slab: 256 deep with 128-cubes gives z in {0, 64, 128}.

    ``yx_stride`` subsamples the (huge) in-plane grid; the z spacing is the real one."""
    Z, Y, X = size
    s = yx_stride or stride * 16
    return np.array(
        [[z, y, x]
         for z in range(0, Z - patch + 1, stride)
         for y in range(0, Y - patch + 1, s)
         for x in range(0, X - patch + 1, s)],
        np.int32,
    )


def _no_overlap(train, hold, patch):
    if not len(train) or not len(hold):
        return True
    d = np.abs(train[:, None, :].astype(np.int64) - hold[None, :, :].astype(np.int64))
    return not (d < patch).all(-1).any()


def test_split_holdout_thin_region_falls_back_to_y_band():
    from tsm.data import split_holdout

    o = _slab_origins()
    assert len(np.unique(o[:, 0])) == 3  # the 256-deep region has only 3 distinct z origins
    tr, ho = split_holdout(o, 128, {"z_frac": 0.15})
    assert len(ho) > 0 and len(tr) > 0
    assert set(ho[:, 0].tolist()) == {0, 64, 128}  # the band is on y now, every z kept
    ymin = ho[:, 1].min()
    assert tr[:, 1].max() + 128 <= ymin
    assert _no_overlap(tr, ho, 128)
    assert 0.05 < len(ho) / len(o) < 0.3


def test_split_holdout_yx_frac_corner():
    from tsm.data import split_holdout

    o = _slab_origins()
    tr, ho = split_holdout(o, 128, {"yx_frac": 0.1})
    assert len(ho) > 0 and len(tr) > 0
    assert ho[:, 1].min() > tr[:, 1].min() and ho[:, 2].min() > tr[:, 2].min()
    assert _no_overlap(tr, ho, 128)
    assert 0.03 < len(ho) / len(o) < 0.25
    # a region with no deep axis at all (a 256^3 block: 3 origins per axis) uses the corner too
    small = _slab_origins(size=(256, 256, 256), yx_stride=64)
    tr2, ho2 = split_holdout(small, 128, {"z_frac": 0.25})
    assert len(ho2) > 0 and len(tr2) > 0 and _no_overlap(tr2, ho2, 128)


def test_split_holdout_axis_modes_and_errors():
    from tsm.data import split_holdout

    o = _slab_origins()
    tr, ho = split_holdout(o, 128, {"y_frac": 0.2})
    assert len(ho) > 0 and tr[:, 1].max() + 128 <= ho[:, 1].min() and _no_overlap(tr, ho, 128)
    tr, ho = split_holdout(o, 128, {"x_range": [4096, 6016]})
    assert len(ho) > 0 and ho[:, 2].min() >= 4096 and _no_overlap(tr, ho, 128)
    deep = np.array([[z, 0, 0] for z in range(0, 1024, 64)], np.int32)
    tr, ho = split_holdout(deep, 128, {"z_frac": 0.15})  # deep z: unchanged behaviour
    assert len(ho) > 0 and ho[:, 0].min() >= 896 - 128 and tr[:, 0].max() + 128 <= ho[:, 0].min()
    with pytest.raises(ValueError):
        split_holdout(o, 128, {"z_frac": 0.15, "y_frac": 0.1})
    with pytest.raises(ValueError):
        split_holdout(o, 128, {"z_frac": 1.5})
    with pytest.raises(ValueError):
        split_holdout(o, 128, {"bogus": 1})


# ----------------------------------------------------------------------------- origins cache key (R14)
def test_build_origins_cache_tracks_min_frac_channel_and_store(synth, tmp_path):
    """R14: the default cache must not serve origins selected with other criteria / older labels."""
    import json as _json
    import time as _time

    st = LabelStore(synth["fine"])
    lo = build_origins(st, "sdf_valid", 32, 16, min_frac=0.05, cache_dir=str(tmp_path))
    hi = build_origins(st, "sdf_valid", 32, 16, min_frac=0.95, cache_dir=str(tmp_path))
    assert len(lo) != len(hi), "a changed min_frac must change the sampled origins"

    # switching the validity channel re-scans rather than reusing the previous selection
    other = next(c for c in st.channels if c != "sdf_valid" and "valid" in c)
    build_origins(st, other, 32, 16, min_frac=0.05, cache_dir=str(tmp_path), name="shared_ch")
    k_other = _json.load(open(os.path.join(str(tmp_path), "origins_shared_ch.json")))
    build_origins(st, "sdf_valid", 32, 16, min_frac=0.05, cache_dir=str(tmp_path), name="shared_ch")
    k_sdf = _json.load(open(os.path.join(str(tmp_path), "origins_shared_ch.json")))
    assert k_other["valid_channel"] == other and k_sdf["valid_channel"] == "sdf_valid"

    # an explicit name that collides across criteria is still invalidated by the manifest
    a = build_origins(st, "sdf_valid", 32, 16, min_frac=0.05, cache_dir=str(tmp_path), name="shared")
    b = build_origins(st, "sdf_valid", 32, 16, min_frac=0.95, cache_dir=str(tmp_path), name="shared")
    assert len(a) != len(b)

    # rebuilding the labels in place (zarr.json touched) invalidates the cache too
    man = os.path.join(str(tmp_path), "origins_shared.json")
    key = _json.load(open(man))
    _time.sleep(0.01)
    os.utime(os.path.join(st.path, "zarr.json"))
    build_origins(st, "sdf_valid", 32, 16, min_frac=0.95, cache_dir=str(tmp_path), name="shared")
    assert _json.load(open(man)) != key


def test_upsample_offset_shifts_the_target_by_the_coarse_remainder():
    """R18: a fine origin not on the coarse grid must shift the interpolated block, not repeat it."""
    f, n, P = 4, 6, 8  # coarse box (n,n,n) with 1 voxel margin -> up to 16 fine voxels
    for axis in range(3):
        ramp = np.arange(n, dtype=np.float32)
        sh = [1, 1, 1]
        sh[axis] = n
        r = np.broadcast_to(ramp.reshape(sh), (n, n, n)).astype(np.float32)
        w = np.stack([np.zeros_like(r), np.ones_like(r), r, np.ones_like(r), np.zeros_like(r), np.zeros_like(r)])
        box = _box(w, np.ones((1, n, n, n)), np.ones((1, n, n, n)))
        prev = None
        for rem in range(f):
            off = [0, 0, 0]
            off[axis] = rem
            out = upsample_coarse_targets(box, f, (P, P, P), offset=off)["winding"][2]
            line = np.moveaxis(out, axis, 0)[:, 0, 0]
            if prev is not None:
                assert not np.allclose(line, prev, atol=1e-6), (axis, rem)
                # the ramp is linear in the coarse coordinate: one fine voxel shift == 1/f per step
                assert np.allclose(line - prev, (1.0 / f) / f, atol=1e-5), (axis, rem, line - prev)
            prev = line
    with pytest.raises(ValueError):
        upsample_coarse_targets(_box(np.zeros((6, 6, 6, 6)), np.ones((1, 6, 6, 6)), np.ones((1, 6, 6, 6))),
                                4, (8, 8, 8), offset=(4, 0, 0))


def test_dataset_unaligned_origin_uses_the_coarse_remainder(synth, tmp_path):
    """The dataset path must pass the remainder through (fine X=8 vs X=9 differ)."""
    from tsm.data import CropDataset
    from tsm.volume import VolumeReader

    fine, coarse = LabelStore(synth["fine"]), LabelStore(synth["coarse"])
    ds = CropDataset(VolumeReader(synth["ct"]), fine, coarse, patch=16, origins=np.zeros((1, 3), np.int32))
    f = ds.factor
    a = ds.load((0, 0, 8))["winding"]
    b = ds.load((0, 0, 8 + 1))["winding"]
    assert f > 1
    assert not np.allclose(a, b, atol=1e-6), "unaligned fine origins gave identical winding targets"
