"""CPU tests for the cached teacher-encoder feature stage (:mod:`tsm.feats`).

Everything runs on a tiny random ResEnc encoder standing in for the recto / ink
teachers (same monkeypatched-loader trick as ``tests/test_feat_distill.py``);
no checkpoint and no GPU is needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
import pytest
import torch

from tsm.config import parse_config
from tsm.feats import (FEATS_DEFAULTS, CachedFeatSource, FeatCache, core_intervals, feats_opts,
                       feats_path, pca_path, run_feats, window_features, window_plan)
from tsm.teachers import TEACHERS, ArchSpec, Normalizer, ResidualEncoder, encoder_channels, encoder_features, encoder_strides

W = (8, 16, 16, 16, 16)          # tiny student widths
T_FEATS = [4, 8, 12, 16, 20]     # tiny "teacher" encoder widths (stage k at /2**k)


class TinyTeacher(torch.nn.Module):
    """Stands in for VesuviusUNet: only ``shared_encoder`` is used by the stage."""

    def __init__(self, seed: int = 0, norm: str = "instance"):
        super().__init__()
        torch.manual_seed(seed)
        spec = ArchSpec(in_channels=1, features_per_stage=list(T_FEATS), n_blocks_per_stage=[1] * 5,
                        strides=[1, 2, 2, 2, 2], norm=norm)
        self.shared_encoder = ResidualEncoder(spec)


def tiny_loader(name, models_dir=None, device="cpu", dtype=torch.float32):
    """(model, spec) like ``teachers.load_teacher``, with the published InstanceNorm."""
    m = TinyTeacher(seed={"recto": 1, "ink": 2}.get(name, 0)).to(device=device, dtype=torch.float32).eval()
    m.requires_grad_(False)
    return m, TEACHERS[name]


def conv_loader(name, models_dir=None, device="cpu", dtype=torch.float32):
    """A purely convolutional stand-in (no norm layer, no intensity normalisation), so the
    encoder is translation-equivariant and the window/border logic can be tested in isolation."""
    m = TinyTeacher(seed={"recto": 1, "ink": 2}.get(name, 0), norm="none")
    m = m.to(device=device, dtype=torch.float32).eval()
    m.requires_grad_(False)
    return m, replace(TEACHERS[name], normalizer=Normalizer("none"))


def smooth_volume(shape, seed=0) -> np.ndarray:
    z, y, x = np.meshgrid(*[np.linspace(0, 3, n) for n in shape], indexing="ij")
    v = np.sin(z + 0.3 * seed) * np.cos(1.7 * y) + np.sin(0.9 * x) + 0.5 * np.cos(0.4 * (z + y + x))
    v = (v - v.min()) / (v.max() - v.min())
    return (v * 255).astype(np.uint8)


def make_ct(path, vol):
    import zarr

    zarr.create_array(store=path, shape=vol.shape, chunks=(32, 32, 32), dtype="uint8", fill_value=0)[:] = vol
    return path


def cfg_for(tmp_path, ct, size, start=(0, 0, 0), **feats):
    return parse_config({
        "volume": {"url": ct, "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": list(start), "size_zyx": list(size)},
        "budget": {"ram_bytes": 4 << 30, "array_bytes": 1 << 30},
        "out_dir": str(tmp_path / "out"),
        "extra": {"feats": feats},
    })


# --------------------------------------------------------------------------- #
# options + tiling
# --------------------------------------------------------------------------- #
def test_defaults_and_opts_validation(tmp_path):
    assert FEATS_DEFAULTS["teachers"] == ["recto", "ink"] and FEATS_DEFAULTS["stages"] == [3, 4]
    assert (FEATS_DEFAULTS["window"], FEATS_DEFAULTS["border"], FEATS_DEFAULTS["pca_dim"]) == (256, 32, 64)
    assert FEATS_DEFAULTS["block_windows"] == 4 and FEATS_DEFAULTS["backend"] == "torch"
    o = feats_opts(cfg_for(tmp_path, "x", (64, 64, 64)))
    assert o["window"] == 256 and o["pca_sample_windows"] == 16 and o["dtype"] == "float16"
    with pytest.raises(ValueError):
        feats_opts(cfg_for(tmp_path, "x", (64,) * 3, nope=1))
    with pytest.raises(ValueError):
        feats_opts(cfg_for(tmp_path, "x", (64,) * 3, window=40))       # not a multiple of 16
    with pytest.raises(ValueError):
        feats_opts(cfg_for(tmp_path, "x", (64,) * 3, border=128))      # no core left
    with pytest.raises(ValueError):
        feats_opts(cfg_for(tmp_path, "x", (64,) * 3, backend="trt"))
    with pytest.raises(ValueError):
        feats_opts(cfg_for(tmp_path, "x", (64,) * 3, teachers=[]))


def test_core_intervals_partition_the_axis():
    # window bigger than the axis: one window, nothing discarded
    assert core_intervals(256, 256, 32) == [(0, 0, 256)]
    assert core_intervals(128, 256, 32) == [(0, 0, 128)]
    iv = core_intervals(6144, 256, 32)
    assert len(iv) == 32 and iv[0] == (0, 0, 224) and iv[1] == (192, 224, 416)
    assert iv[-1][0] == 6144 - 256 and iv[-1][2] == 6144
    for extent in (192, 256, 512, 640, 6144):
        for win, bor in ((256, 32), (128, 16), (64, 16)):
            iv = core_intervals(extent, win, bor)
            lo_hi = [(lo, hi) for _, lo, hi in iv]
            assert lo_hi[0][0] == 0 and lo_hi[-1][1] == extent
            assert all(a[1] == b[0] for a, b in zip(lo_hi, lo_hi[1:])), (extent, win, bor)
            for s, lo, hi in iv:                       # every core lies inside its window
                assert s <= lo < hi <= s + min(win, extent)


def test_window_plan_counts():
    plan = window_plan((256, 6144, 6144), 256, 32)
    assert [len(a) for a in plan] == [1, 32, 32]


# --------------------------------------------------------------------------- #
# dry-run accounting
# --------------------------------------------------------------------------- #
def test_dry_run_accounting(tmp_path, capsys):
    cfg = cfg_for(tmp_path, "http://example.invalid/x.zarr", (256, 6144, 6144))
    r = run_feats(cfg, dry_run=True)
    assert r["windows"] == 1 * 32 * 32 and r["windows_per_axis"] == [1, 32, 32]
    assert r["grids"] == {3: (32, 768, 768), 4: (16, 384, 384)}
    assert r["bytes"]["recto_s3"] == 64 * 32 * 768 * 768 * 2      # 2.42 GB
    assert r["bytes"]["ink_s4"] == 64 * 16 * 384 * 384 * 2
    assert r["total_bytes"] == 2 * (r["bytes"]["recto_s3"] + r["bytes"]["recto_s4"])
    assert "dry run" in capsys.readouterr().out
    # no PCA -> the raw ResEnc widths (256 at /8, 320 at /16)
    r2 = run_feats(cfg_for(tmp_path, "x", (256, 6144, 6144), pca_dim=None), dry_run=True)
    assert r2["bytes"]["recto_s3"] == 256 * 32 * 768 * 768 * 2
    assert r2["bytes"]["recto_s4"] == 320 * 16 * 384 * 384 * 2


def test_cli_registers_feats_stage():
    from tsm.cli import STAGES

    assert "feats" in STAGES and STAGES["feats"].__name__ == "stage_feats"


# --------------------------------------------------------------------------- #
# window stitching: the discarded border makes adjacent windows agree
# --------------------------------------------------------------------------- #
def test_window_cores_agree_with_a_single_pass():
    """On a smooth input, the core of every window matches a single whole-region
    forward, while the discarded border does not -- which is why the border exists."""
    enc, spec = conv_loader("recto")
    size, win, bor = 128, 64, 16
    vol = smooth_volume((size,) * 3, seed=1)
    full = window_features(enc.shared_encoder, spec.normalizer, vol, [3])[3]
    plan = window_plan((size,) * 3, win, bor)
    stage = 3
    core_err, border_err = 0.0, 0.0
    for wz in plan[0]:
        for wy in plan[1]:
            for wx in plan[2]:
                sub = vol[wz[0]:wz[0] + win, wy[0]:wy[0] + win, wx[0]:wx[0] + win]
                f = window_features(enc.shared_encoder, spec.normalizer, sub, [stage])[stage]
                cut = tuple(slice((w[1] - w[0]) >> stage, (w[2] - w[0]) >> stage) for w in (wz, wy, wx))
                whole = tuple(slice(w[0] >> stage, (w[0] + win) >> stage) for w in (wz, wy, wx))
                d = (f - full[(slice(None), *whole)]).abs()
                core_err = max(core_err, float(d[(slice(None), *cut)].max()))
                d[(slice(None), *cut)] = 0.0  # what is left is the discarded border ring
                border_err = max(border_err, float(d.max()))
    scale = float(full.abs().max())
    assert core_err < 0.05 * scale, (core_err, scale)      # measured ~2 % of the feature scale
    assert border_err > 10 * core_err, (core_err, border_err)  # measured ~25x worse in the ring


def test_instance_norm_is_the_residual_seam_not_the_halo():
    """The published encoders use InstanceNorm3d, whose statistics are per window, so a
    window's features carry a per-channel affine offset that no border can remove.  The
    border only fixes the *padding halo*: with the norm removed the ring is >10x worse
    than the core (test above), with InstanceNorm the two are the same size."""
    enc, spec = tiny_loader("recto")
    size, win, bor, stage = 128, 64, 16, 3
    vol = smooth_volume((size,) * 3, seed=1)
    full = window_features(enc.shared_encoder, spec.normalizer, vol, [stage])[stage]
    plan = window_plan((size,) * 3, win, bor)
    core_err, border_err = 0.0, 0.0
    for wz in plan[0]:
        for wy in plan[1]:
            for wx in plan[2]:
                sub = vol[wz[0]:wz[0] + win, wy[0]:wy[0] + win, wx[0]:wx[0] + win]
                f = window_features(enc.shared_encoder, spec.normalizer, sub, [stage])[stage]
                cut = tuple(slice((w[1] - w[0]) >> stage, (w[2] - w[0]) >> stage) for w in (wz, wy, wx))
                whole = tuple(slice(w[0] >> stage, (w[0] + win) >> stage) for w in (wz, wy, wx))
                d = (f - full[(slice(None), *whole)]).abs()
                core_err = max(core_err, float(d[(slice(None), *cut)].max()))
                d[(slice(None), *cut)] = 0.0
                border_err = max(border_err, float(d.max()))
    assert border_err < 3 * core_err, (core_err, border_err)


def test_stage_output_matches_a_single_pass(tmp_path):
    """The tiled stage (many windows, cores stitched) reproduces one whole-region forward."""
    import zarr

    size = 128
    vol = smooth_volume((size,) * 3, seed=2)
    ct = make_ct(str(tmp_path / "ct.zarr"), vol)
    cfg = cfg_for(tmp_path, ct, (size,) * 3, window=64, border=16, stages=[3], pca_dim=None,
                  block_windows=2, device="cpu")
    r = run_feats(cfg, loader=conv_loader)
    got = np.asarray(zarr.open_array(store=r["arrays"]["recto_s3"], mode="r")[:], dtype=np.float32)
    enc, spec = conv_loader("recto")
    ref = window_features(enc.shared_encoder, spec.normalizer, vol, [3])[3].numpy()
    assert got.shape == ref.shape
    # 2 % of the feature scale: the padding halo left inside the core, plus the fp16 store
    assert np.abs(got - ref).max() < 0.05 * np.abs(ref).max()


# --------------------------------------------------------------------------- #
# the stage end to end
# --------------------------------------------------------------------------- #
def test_run_feats_writes_arrays_attrs_and_pca(tmp_path):
    import zarr

    size = (64, 128, 128)
    vol = smooth_volume(size, seed=3)
    ct = make_ct(str(tmp_path / "ct.zarr"), vol)
    cfg = cfg_for(tmp_path, ct, size, start=(16, 32, 32), window=64, border=16, stages=[3, 4],
                  pca_dim=6, pca_sample_windows=4, pca_vectors_per_window=64, block_windows=2, device="cpu")
    r = run_feats(cfg, loader=tiny_loader)
    assert r["windows"] == 1 * 3 * 3 and r["grids"] == {3: (8, 16, 16), 4: (4, 8, 8)}
    assert sorted(r["arrays"]) == ["ink_s3", "ink_s4", "recto_s3", "recto_s4"]

    a = zarr.open_array(store=feats_path(r["out_dir"], "recto", 3), mode="r")
    assert a.shape == (6, 8, 16, 16) and a.dtype == np.float16
    assert tuple(a.chunks) == (6, 8, 32, 32)
    at = dict(a.attrs)
    assert at["origin_zyx"] == [16, 32, 32] and at["stride"] == 8 and at["scale"] == 8
    assert at["channels"] == 6 and at["pca"] == 6 and at["raw_channels"] == T_FEATS[3]
    assert at["teacher"] == "recto" and at["stage"] == 3 and at["window"] == 64 and at["border"] == 16
    assert np.isfinite(np.asarray(a[:], dtype=np.float32)).all()
    assert dict(zarr.open_array(store=feats_path(r["out_dir"], "ink", 4), mode="r").attrs)["stride"] == 16

    p = np.load(pca_path(r["out_dir"], "recto", 3))
    assert p["components"].shape == (6, T_FEATS[3]) and p["mean"].shape == (T_FEATS[3],)
    assert 0.0 < float(p["explained"].sum()) <= 1.0 + 1e-5

    prog = json.load(open(os.path.join(r["out_dir"], "progress.json")))
    # 4 blocks x 2 teachers x 2 stages (3x3 windows, block_windows=2); completion is keyed
    # per (teacher, stage, block) so a later run that adds a stage still builds it (R10)
    assert len(prog["blocks"]) == 16 and prog["window"] == 64
    assert sorted(prog["blocks"])[0].count("/") == 2 and "/s3/" in sorted(prog["blocks"])[0]
    assert prog["manifest"]["window"] == 64 and prog["manifest"]["pca_dim"] == 6

    # resumable: a second run skips every completed block
    r2 = run_feats(cfg, loader=tiny_loader)
    assert r2["blocks"] == 0
    # the cache reader agrees with the array
    c = FeatCache(feats_path(r["out_dir"], "recto", 3))
    assert (c.channels, c.scale, c.stride, c.origin_zyx, c.stage, c.pca) == (6, 8, 8, (16, 32, 32), 3, 6)
    crop = c.read_crop((16 + 8, 32 + 16, 32 + 16), 32)
    assert crop.shape == (6, 4, 4, 4)
    assert np.allclose(crop, np.asarray(a[:, 1:5, 2:6, 2:6], dtype=np.float32))
    assert c.read_crop((0, 0, 0), 32) is None and c.read_crop((16 + 4, 32, 32), 32) is None


def test_encoder_stage_index_is_the_downsampling_power():
    enc, _ = conv_loader("recto")
    assert encoder_strides(enc) == [1, 2, 4, 8, 16] and encoder_channels(enc) == T_FEATS
    x = torch.randn(1, 1, 32, 32, 32)
    f = encoder_features(enc, x)
    assert tuple(f[3].shape) == (1, T_FEATS[3], 4, 4, 4) and tuple(f[4].shape) == (1, T_FEATS[4], 2, 2, 2)


def test_run_feats_rejects_a_stage_whose_stride_is_not_the_power(tmp_path):
    vol = smooth_volume((32, 32, 32))
    ct = make_ct(str(tmp_path / "ct.zarr"), vol)
    cfg = cfg_for(tmp_path, ct, (32, 32, 32), window=32, border=0, stages=[5], pca_dim=None, device="cpu")
    with pytest.raises(ValueError, match="outside the encoder"):
        run_feats(cfg, loader=conv_loader)


# --------------------------------------------------------------------------- #
# cached feat_distill source
# --------------------------------------------------------------------------- #
def _built(tmp_path, size=32, pca_dim=None, dtype="float16", loader=conv_loader):
    """A one-window cache of the tiny recto teacher over a `size`^3 region."""
    vol = smooth_volume((size,) * 3, seed=5)
    ct = make_ct(str(tmp_path / "ct.zarr"), vol)
    cfg = cfg_for(tmp_path, ct, (size,) * 3, window=size, border=0, stages=[3, 4], pca_dim=pca_dim,
                  pca_sample_windows=1, pca_vectors_per_window=32, teachers=["recto"], dtype=dtype, device="cpu")
    r = run_feats(cfg, loader=loader)
    return vol, r


@pytest.mark.parametrize("loader", [conv_loader, tiny_loader], ids=["no_norm", "instance_norm"])
def test_cached_source_matches_the_live_teacher_under_identity(tmp_path, loader):
    """Same tiny teacher, same projector weights, identity augmentation: the cached term
    reproduces the live term.  Exact for raw (un-reduced) channels when the cache window
    covers the crop -- with InstanceNorm that condition is what makes the window statistics
    of the cached pass and the live pass identical."""
    from tsm.student import TSMNet, feature_distill_loss
    from tsm.train import FeatDistiller, feat_distill_opts

    torch.manual_seed(0)
    vol, r = _built(tmp_path, size=32, pca_dim=None, loader=loader)
    pairs = [[3, 3], [4, 4]]
    fd = feat_distill_opts({"enabled": True, "weight": 1.0, "pairs": pairs, "teachers": ["recto"],
                            "source": "cached", "feats_dir": r["out_dir"]})
    cached = FeatDistiller(fd, W, "cpu")
    live = FeatDistiller(feat_distill_opts({"enabled": True, "weight": 1.0, "pairs": pairs,
                                            "teachers": ["recto"]}), W, "cpu", loader=loader)
    live.proj["recto"].load_state_dict(cached.proj["recto"].proj.state_dict())  # identical projectors

    net = TSMNet(widths=W).eval()
    ct01 = torch.from_numpy(vol.astype(np.float32) / 255.0)[None, None]
    batch = {"input": torch.cat([ct01, torch.zeros_like(ct01)], 1),
             "origin_zyx": torch.zeros(1, 3, dtype=torch.int64)}
    with torch.no_grad():
        _, feats = net(batch["input"], return_features=True)
    lc, lp = cached(feats, ct01[:, 0], batch=batch)
    ll, lpp = live(feats, ct01[:, 0], batch=batch)
    assert lp["feat_recto_frac"] == 1.0
    lcf, llf = float(lc.detach()), float(ll.detach())
    assert abs(lcf - llf) < 1e-3, (lcf, llf)
    assert abs(lp["feat_recto"] - lpp["feat_recto"]) < 1e-3
    # and the cached loss really is the plain distillation loss against the cached target
    src = cached.cached["recto"]
    tgt = src.targets((0, 0, 0), 32)
    ref = feature_distill_loss(src.proj(feats), [None, None, None, tgt[3][None], tgt[4][None]], "cosine")
    assert abs(float(ref.detach()) - lp["feat_recto"]) < 1e-6


def test_cached_source_skips_non_signed_permutations(tmp_path):
    from tsm.data import SpatialParams, rot90_matrix_inplane, rotation_matrix

    torch.manual_seed(0)
    _, r = _built(tmp_path, size=32, pca_dim=4)
    caches = {k: FeatCache(feats_path(r["out_dir"], "recto", k)) for k in (3, 4)}
    src = CachedFeatSource("recto", caches, [[3, 3], [4, 4]], W)
    feats = {"enc3": torch.randn(2, 16, 4, 4, 4), "enc4": torch.randn(2, 16, 2, 2, 2)}
    batch = {"input": torch.zeros(2, 2, 32, 32, 32), "origin_zyx": torch.zeros(2, 3, dtype=torch.int64)}
    flip = SpatialParams(flips=(True, False, False), rot90=rot90_matrix_inplane(2))
    rot = SpatialParams(rotation=rotation_matrix(torch.tensor([1.0, 0, 0]), 0.3))
    loss, parts = src(feats, batch, [flip, rot])
    assert parts["feat_recto_frac"] == 0.5 and torch.isfinite(loss) and loss.requires_grad
    assert src(feats, batch, [rot, rot])[1]["feat_recto_frac"] == 0.0
    assert src(feats, batch, [flip, flip])[1]["feat_recto_frac"] == 1.0
    # out-of-cache origins skip too
    b2 = dict(batch, origin_zyx=torch.tensor([[0, 0, 0], [512, 0, 0]]))
    assert src(feats, b2)[1]["feat_recto_frac"] == 0.5
    # the target is transformed exactly like the crop
    t = src.targets((0, 0, 0), 32, SpatialParams(flips=(False, True, False)))
    assert torch.allclose(t[3], torch.from_numpy(caches[3].read_crop((0, 0, 0), 32)).flip(2))


def test_mixed_sources_and_describe(tmp_path):
    """recto cached + ink live in one distiller (per-teacher ``sources`` override)."""
    from tsm.train import FeatDistiller, feat_distill_opts

    torch.manual_seed(0)
    _, r = _built(tmp_path, size=32, pca_dim=4)
    fd = feat_distill_opts({"enabled": True, "weight": 1.0, "pairs": [[3, 3]], "teachers": ["recto", "ink"],
                            "sources": {"recto": "cached"}, "feats_dir": r["out_dir"]})
    d = FeatDistiller(fd, W, "cpu", loader=conv_loader)
    assert set(d.cached) == {"recto"} and set(d.encoders) == {"ink"}
    lines = "\n".join(d.describe(W))
    assert "recto: student enc3 (16 ch, /8) -> CACHED stage 3 (4 ch, /8" in lines
    assert "ink: student enc3 (16 ch, /8) -> teacher stage 3" in lines
    assert any("cached recto features" in n for n, _ in d.bytes_estimate(32, 1))

    ct01 = torch.rand(1, 32, 32, 32)
    batch = {"input": torch.stack([ct01, torch.zeros_like(ct01)], 1),
             "origin_zyx": torch.zeros(1, 3, dtype=torch.int64)}
    feats = {f"enc{k}": torch.randn(1, c, 32 >> k, 32 >> k, 32 >> k, requires_grad=True) for k, c in enumerate(W)}
    loss, parts = d(feats, ct01, batch=batch)
    assert set(parts) >= {"feat_recto", "feat_recto_frac", "feat_ink", "feat"}
    assert float(loss.detach()) == pytest.approx((parts["feat_recto"] + parts["feat_ink"]) / 2, rel=1e-5)
    loss.backward()
    assert any(p.grad is not None for p in d.parameters())
    sd = d.state_dict()
    d.load_state_dict(sd)
    assert any(k.startswith("recto.proj.") for k in sd)


def test_feat_distill_opts_source_validation():
    from tsm.train import feat_distill_opts

    assert feat_distill_opts({"enabled": True})["source"] == "live"
    assert feat_distill_opts({"enabled": True, "sources": {"ink": "cached"}})["sources"] == {"ink": "cached"}
    with pytest.raises(ValueError):
        feat_distill_opts({"enabled": True, "source": "zarr"})
    with pytest.raises(ValueError):
        feat_distill_opts({"enabled": True, "sources": {"ink": "zarr"}})
    with pytest.raises(ValueError):
        feat_distill_opts({"enabled": True, "sources": {"nosuch": "cached"}})


def test_cached_source_missing_cache_message(tmp_path):
    from tsm.train import FeatDistiller, feat_distill_opts

    fd = feat_distill_opts({"enabled": True, "teachers": ["recto"], "pairs": [[3, 3]], "source": "cached",
                            "feats_dir": str(tmp_path / "nope")})
    with pytest.raises(FileNotFoundError, match="tsm feats"):
        FeatDistiller(fd, W, "cpu", loader=conv_loader)


def test_ablate_feats_config_is_loadable():
    from tsm.ablate import ablate_opts, variant_config
    from tsm.config import load_config

    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "ablate_feats.json"))
    o = ablate_opts(cfg)
    assert o["steps"] == 3000 and o["seed"] == 0 and "y_frac" in o["holdout_origins"]
    assert set(o["variants"]) == {"live_every1", "live_every2", "cached_every1", "cached_plus_dino", "none"}
    for name, ov in o["variants"].items():
        vc = variant_config(cfg, name, ov, o, "/tmp/tsm-ablate-feats")
        fd = vc.extra["train"]["feat_distill"]
        assert fd["enabled"] == (name != "none")
    assert o["variants"]["live_every2"]["feat_distill.every"] == 2


# --------------------------------------------------------------------------- #
# R10: resume is keyed per (teacher, stage, block) and guarded by a run manifest
# --------------------------------------------------------------------------- #
def _frac_nonzero(path):
    import zarr

    return float((np.asarray(zarr.open_array(store=path, mode="r")[:]) != 0).mean())


def test_adding_a_stage_to_a_complete_cache_builds_it(tmp_path):
    """A cache completed for stage [1] must populate stage 2 when [1, 2] is requested."""
    size = (64, 64, 64)
    ct = make_ct(str(tmp_path / "ct.zarr"), smooth_volume(size, seed=5))
    common = dict(window=32, border=8, pca_dim=0, block_windows=4, device="cpu")
    r1 = run_feats(cfg_for(tmp_path, ct, size, stages=[1], **common), loader=tiny_loader)
    assert sorted(r1["arrays"]) == ["ink_s1", "recto_s1"]
    assert _frac_nonzero(feats_path(r1["out_dir"], "recto", 1)) > 0.5
    prog = json.load(open(os.path.join(r1["out_dir"], "progress.json")))
    assert all(k.split("/")[1] == "s1" for k in prog["blocks"])

    r2 = run_feats(cfg_for(tmp_path, ct, size, stages=[1, 2], **common), loader=tiny_loader)
    assert r2["blocks"] > 0  # the stage-1-only progress must not skip the new stage
    for name in ("recto", "ink"):
        assert _frac_nonzero(feats_path(r2["out_dir"], name, 2)) > 0.5, f"{name} stage 2 is empty"
        assert _frac_nonzero(feats_path(r2["out_dir"], name, 1)) > 0.5
    # now that both stages are recorded, a third run is a no-op
    assert run_feats(cfg_for(tmp_path, ct, size, stages=[1, 2], **common),
                     loader=tiny_loader)["blocks"] == 0


def test_changed_manifest_rebuilds_the_stage_arrays(tmp_path, capsys):
    size = (64, 64, 64)
    ct = make_ct(str(tmp_path / "ct2.zarr"), smooth_volume(size, seed=6))
    mk = lambda **kw: cfg_for(tmp_path, ct, size, stages=[1], pca_dim=0, block_windows=4,  # noqa: E731
                              device="cpu", **kw)
    r1 = run_feats(mk(window=32, border=8), loader=tiny_loader)
    assert r1["blocks"] > 0
    capsys.readouterr()
    r2 = run_feats(mk(window=32, border=4), loader=tiny_loader)  # border changed
    out = capsys.readouterr().out
    assert "run manifest changed" in out and "border" in out
    assert r2["blocks"] > 0
    assert json.load(open(os.path.join(r2["out_dir"], "progress.json")))["manifest"]["border"] == 4


def test_changed_pca_dim_rebuilds_the_array_shape(tmp_path):
    import zarr

    size = (64, 64, 64)
    ct = make_ct(str(tmp_path / "ct3.zarr"), smooth_volume(size, seed=7))
    mk = lambda pca: cfg_for(tmp_path, ct, size, stages=[1], window=32, border=8,  # noqa: E731
                             pca_dim=pca, pca_sample_windows=2, pca_vectors_per_window=32,
                             block_windows=4, device="cpu")
    r1 = run_feats(mk(0), loader=tiny_loader)
    p = feats_path(r1["out_dir"], "recto", 1)
    assert zarr.open_array(store=p, mode="r").shape[0] == T_FEATS[1]
    r2 = run_feats(mk(3), loader=tiny_loader)
    assert zarr.open_array(store=p, mode="r").shape[0] == 3 and r2["blocks"] > 0
    assert _frac_nonzero(p) > 0.5
