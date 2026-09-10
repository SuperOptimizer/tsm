"""CPU tests for the dinovol reimplementation, the token cache and the DINO
``feat_distill`` source (:mod:`tsm.dino`).

Everything here runs on a tiny random ViT except the key-parity test (shapes
only, no checkpoint) and the optional real-checkpoint forward, which is skipped
when ``~/.cache/tsm-models/dinovol.pt`` is absent.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tsm.dino import (DinoFeatSource, DinoTokenCache, DinoVolViT, ViTSpec, apply_pca, apply_rope,
                      apply_signed_permutation, build_dinovol, fit_pca, infer_vit_spec, load_dinovol,
                      normalize_robust, patch_token_grid, rope_coords, signed_permutation, window_starts,
                      window_weights)

INVENTORY = os.path.expanduser("~/.cache/tsm-production/models/dinovol_v2_*/*/*.tsm-model/model.json")
CKPT = os.path.expanduser("~/.cache/tsm-models/dinovol.pt")


def tiny_spec(**kw) -> ViTSpec:
    s = dict(in_channels=1, embed_dim=24, depth=2, num_heads=4, patch=8, mlp_hidden=16,
             num_reg_tokens=2, rope_periods=1)
    s.update(kw)
    return ViTSpec(**s)


def tiny_model(seed: int = 0, **kw) -> DinoVolViT:
    torch.manual_seed(seed)
    m = DinoVolViT(tiny_spec(**kw)).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.05)
        for b in m.blocks:
            b.rope_embed.mix_frequencies.copy_(torch.randn_like(b.rope_embed.mix_frequencies) * 0.5)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def token_only_model(seed: int = 0) -> DinoVolViT:
    """A tiny ViT with the residual branches zeroed: every patch token is a pure
    function of its own 8^3 voxels, so the network is translation-equivariant and
    overlapping windows must agree exactly."""
    m = tiny_model(seed)
    with torch.no_grad():
        for b in m.blocks:
            b.attn.proj.weight.zero_()
            b.attn.proj.bias.zero_()
            b.mlp.fc2.weight.zero_()
            b.mlp.fc2.bias.zero_()
    return m


# --------------------------------------------------------------------------- #
# key parity
# --------------------------------------------------------------------------- #
def _inventory() -> dict[str, tuple[int, ...]] | None:
    hits = sorted(glob.glob(INVENTORY))
    if not hits:
        return None
    d = json.load(open(hits[0]))["tensors"]
    items = d.items() if isinstance(d, dict) else ((x["name"], x) for x in d)
    inv = {k: tuple(int(i) for i in v["shape"]) for k, v in items}
    return {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in inv.items()}


def test_inventory_key_parity():
    inv = _inventory()
    if inv is None:
        pytest.skip("no dinovol inventory in ~/.cache/tsm-production/models")
    spec = infer_vit_spec(inv)
    assert (spec.embed_dim, spec.depth, spec.num_heads, spec.patch, spec.mlp_hidden, spec.num_reg_tokens) == \
        (864, 24, 16, 8, 2304, 4)
    net = build_dinovol(inv)
    got = {k: tuple(v.shape) for k, v in net.state_dict().items()}
    missing = sorted(set(inv) - set(got))
    unexpected = sorted(set(got) - set(inv))
    wrong = sorted(k for k in set(inv) & set(got) if inv[k] != got[k])
    assert not (missing or unexpected or wrong), \
        f"missing {missing[:5]} unexpected {unexpected[:5]} wrong {[(k, inv[k], got[k]) for k in wrong[:5]]}"
    assert net.num_params() == 215_859_168


def test_infer_spec_rejects_non_dinovol():
    with pytest.raises(ValueError):
        infer_vit_spec({"encoder.stem.convs.0.conv.weight": (32, 1, 3, 3, 3)})


# --------------------------------------------------------------------------- #
# forward shapes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [32, 48])
def test_forward_shapes_tiny(n):
    m = tiny_model()
    out = m.forward_features(torch.randn(2, 1, n, n, n))
    g = n // 8
    assert out["grid"] == (g, g, g)
    assert out["patch_tokens"].shape == (2, g ** 3, 24)
    assert out["cls"].shape == (2, 24)
    assert out["reg"].shape == (2, 2, 24)
    assert torch.isfinite(out["patch_tokens"]).all()


def test_forward_rejects_unaligned():
    m = tiny_model()
    with pytest.raises(ValueError):
        m.forward_features(torch.randn(1, 1, 20, 32, 32))


@pytest.mark.skipif(not os.path.exists(CKPT), reason="dinovol.pt not cached")
def test_real_checkpoint_forward_32():
    m = load_dinovol(CKPT)
    assert m.spec.embed_dim == 864 and m.spec.depth == 24
    with torch.inference_mode():
        out = m.forward_features(torch.zeros(1, 1, 32, 32, 32))
    assert out["grid"] == (4, 4, 4)
    assert out["patch_tokens"].shape == (1, 64, 864)
    assert torch.isfinite(out["patch_tokens"]).all()
    # tokens are not degenerate: the normalised tokens are not all identical
    t = F.normalize(out["patch_tokens"][0], dim=-1)
    assert float((t @ t.T).min()) < 0.999


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #
def reference_rope(coords: np.ndarray, mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direct port of ``MixedRopePositionEmbedding.get_embed_from_coords`` (rope.py)."""
    angles = 2 * math.pi * np.einsum("td,hpd->htp", coords, mix)
    angles = np.tile(angles, 2)
    return np.sin(angles), np.cos(angles)


def reference_coords(shape) -> np.ndarray:
    """Direct port of ``_BaseRopePositionEmbedding._get_coords`` with normalize_coords='separate'."""
    axes = [np.arange(0.5, n, dtype=np.float64) / n for n in shape]
    c = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(shape))
    return 2.0 * c - 1.0


def test_rope_matches_direct_port():
    grid = (2, 3, 4)
    m = tiny_model()
    c_ours = rope_coords(grid, dtype=torch.float64)
    c_ref = reference_coords(grid)
    assert np.abs(c_ours.numpy() - c_ref).max() < 1e-12
    mix = m.blocks[0].rope_embed.mix_frequencies.double()
    sin_o, cos_o = m.blocks[0].rope_embed.embed(c_ours)
    sin_r, cos_r = reference_rope(c_ref, mix.numpy())
    assert np.abs(sin_o.numpy() - sin_r).max() < 1e-10
    assert np.abs(cos_o.numpy() - cos_r).max() < 1e-10


def test_rope_periods_are_the_geometric_ladder():
    m = tiny_model(embed_dim=48, num_heads=2, rope_periods=4)  # head_dim 24, axis_dim 8
    per = m.blocks[0].rope_embed.periods
    assert torch.allclose(per, 100.0 ** (2 * torch.arange(4, dtype=torch.float32) / 8))


def test_rope_preserves_norms():
    m = tiny_model()
    c = rope_coords((2, 2, 2))
    sin, cos = m.blocks[0].rope_embed.embed(c)
    q = torch.randn(2, 4, 8 + 3, 6)  # (B, heads, prefix + tokens, head_dim)
    r = apply_rope(q, sin, cos, prefix_tokens=3)
    assert torch.allclose(q.norm(dim=-1), r.norm(dim=-1), atol=1e-5)
    # prefix tokens untouched
    assert torch.equal(q[:, :, :3], r[:, :, :3])


def test_rope_zero_frequencies_is_identity():
    m = tiny_model()
    with torch.no_grad():
        m.blocks[0].rope_embed.mix_frequencies.zero_()
    sin, cos = m.blocks[0].rope_embed.embed(rope_coords((2, 2, 2)))
    assert torch.allclose(sin, torch.zeros_like(sin))
    assert torch.allclose(cos, torch.ones_like(cos))
    q = torch.randn(1, 4, 8, 6)
    assert torch.allclose(apply_rope(q, sin, cos), q, atol=1e-6)


def test_rope_relative_inner_product():
    """A shared shift of all coordinates leaves q.k invariant (the defining RoPE property)."""
    m = tiny_model()
    c = rope_coords((2, 2, 2), dtype=torch.float64)
    mix = m.blocks[0].rope_embed
    q = torch.randn(1, 4, 8, 6, dtype=torch.float64)
    k = torch.randn(1, 4, 8, 6, dtype=torch.float64)
    def qk(coords):
        sin, cos = mix.embed(coords)
        return apply_rope(q, sin, cos) @ apply_rope(k, sin, cos).transpose(-2, -1)
    shift = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    assert torch.allclose(qk(c), qk(c + shift), atol=1e-9)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def test_normalize_robust_median_mad():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(8, 8, 8)).astype(np.uint8)
    y = normalize_robust(x)
    lo, hi = np.percentile(x.astype(np.float32), [1.0, 99.0])
    c = np.clip(x.astype(np.float32), lo, hi)
    med = np.median(c)
    mad = 1.4826 * np.median(np.abs(c - med))
    assert np.abs(y - (c - med) / mad).max() < 1e-4
    assert abs(float(np.median(y))) < 1e-5


def test_normalize_robust_constant_volume():
    y = normalize_robust(np.full((8, 8, 8), 42, np.uint8))
    assert np.isfinite(y).all() and np.abs(y).max() == 0.0


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
def test_pca_round_trip():
    rng = np.random.default_rng(0)
    basis = np.linalg.qr(rng.normal(size=(32, 32)))[0][:6]
    x = (rng.normal(size=(500, 6)) * np.array([5, 4, 3, 2, 1, 0.5])) @ basis + 3.0
    pca = fit_pca(x, dim=6)
    y = apply_pca(x, pca, normalize=False)
    rec = y @ pca["components"] + pca["mean"]
    assert np.abs(rec - x).max() < 1e-3
    assert pca["explained"].sum() > 0.999
    assert np.allclose(pca["components"] @ pca["components"].T, np.eye(6), atol=1e-4)
    yn = apply_pca(x, pca)
    assert np.abs(np.linalg.norm(yn, axis=-1) - 1).max() < 1e-5


def test_pca_commutes_with_convex_stitching():
    """The stitch is a convex combination and the PCA is affine, so reducing per
    window and blending equals blending then reducing (what patch_token_grid does)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 12)).astype(np.float32)
    pca = fit_pca(x, dim=4)
    a, b = x[:5], x[5:10]
    w = np.float32(0.3)
    mix = w * a + (1 - w) * b
    assert np.abs(apply_pca(mix, pca, normalize=False)
                  - (w * apply_pca(a, pca, normalize=False) + (1 - w) * apply_pca(b, pca, normalize=False))).max() < 1e-4


def test_fit_pca_needs_enough_tokens():
    with pytest.raises(ValueError):
        fit_pca(np.zeros((3, 8), np.float32), dim=4)


# --------------------------------------------------------------------------- #
# windowing / stitching
# --------------------------------------------------------------------------- #
def test_window_starts_and_weights():
    assert window_starts(16, 16, 2) == [0]
    assert window_starts(40, 16, 2) == [0, 14, 24]  # last start snapped to the end
    w = window_weights((6, 6, 6), (2, 2, 2))
    assert w.shape == (6, 6, 6)
    assert w.max() == pytest.approx(1.0)
    assert w[0, 3, 3] == pytest.approx(1 / 3)  # (0 + 1) / (overlap + 1)


def smooth_volume(shape=(32, 48, 48), seed=0) -> np.ndarray:
    z, y, x = np.meshgrid(*[np.linspace(0, 3, n) for n in shape], indexing="ij")
    v = np.sin(z) * np.cos(1.7 * y) + np.sin(0.9 * x)
    v = (v - v.min()) / (v.max() - v.min())
    return (v * 255).astype(np.uint8)


def _tokens(m, vol, normalize="none"):
    x = normalize_robust(vol) if normalize == "robust" else vol.astype(np.float32)
    with torch.inference_mode():
        t = m.forward_features(torch.from_numpy(np.ascontiguousarray(x))[None, None])
    g = t["grid"]
    return F.normalize(t["patch_tokens"][0], dim=-1).reshape(*g, -1)


def test_overlapping_windows_agree_in_the_overlap():
    """Two windows overlapping by one token produce identical tokens there, for a
    per-patch (translation-equivariant) tiny ViT on a smooth volume.  This is the
    property the token-level stitch relies on; the real ViT only satisfies it
    approximately (attention context and the per-window robust normalisation both
    change), which is exactly what the linear ramp blending is for."""
    m = token_only_model()
    vol = smooth_volume((16, 16, 32))
    a = _tokens(m, vol[:, :, :16])        # x tokens 0..1
    b = _tokens(m, vol[:, :, 8:24])       # x tokens 1..2
    assert torch.allclose(a[:, :, 1], b[:, :, 0], atol=1e-6)


def test_patch_token_grid_single_window_matches_forward():
    m = token_only_model()
    vol = smooth_volume((16, 16, 16))
    g = patch_token_grid(vol, m, window=16, overlap_tokens=0)
    with torch.inference_mode():
        t = F.normalize(m.forward_features(torch.from_numpy(normalize_robust(vol))[None, None])["patch_tokens"][0], dim=-1)
    ref = np.moveaxis(t.reshape(2, 2, 2, -1).numpy(), -1, 0)
    assert g.shape == (24, 2, 2, 2)
    assert np.abs(g - ref).max() < 1e-5


def test_patch_token_grid_stitch_is_equivariant_for_a_per_patch_model():
    """A per-patch model (no per-window normalisation): the stitched grid equals
    the single-shot grid exactly, so the blending itself is loss-free."""
    m = token_only_model()
    vol = smooth_volume((16, 16, 48))
    stitched = patch_token_grid(vol, m, window=16, overlap_tokens=1, normalize="none")
    # the reference: every token's own 8^3 patch, in one window
    single = patch_token_grid(vol, m, window=48, overlap_tokens=0, normalize="none")
    cos = (stitched * single).sum(0)
    assert cos.min() > 1 - 1e-5, f"worst token cosine {cos.min():.6f}"


def test_patch_token_grid_weights_cover_every_token():
    m = tiny_model()
    vol = smooth_volume((16, 16, 40))
    g = patch_token_grid(vol, m, window=16, overlap_tokens=1)
    assert g.shape == (24, 2, 2, 5)
    assert np.abs(np.linalg.norm(g, axis=0) - 1).max() < 1e-4  # every token got weight


def test_patch_token_grid_external_accumulator_matches():
    """The block-tiled path (external accumulator + a subset of the global window
    starts) reproduces the monolithic stitch exactly."""
    m = tiny_model()
    vol = smooth_volume((16, 16, 48))
    ref = patch_token_grid(vol, m, window=16, overlap_tokens=1)
    grid = (2, 2, 6)
    starts = [window_starts(g, min(2, g), 1) for g in grid]
    acc = np.zeros((24, *grid), np.float32)
    wsum = np.zeros(grid, np.float32)
    for i in range(0, len(starts[2]), 2):
        sel = [starts[0], starts[1], starts[2][i:i + 2]]
        patch_token_grid(vol, m, window=16, overlap_tokens=1, out=acc, weight_out=wsum, starts=sel)
    acc /= np.maximum(wsum, 1e-30)[None]
    acc /= np.maximum(np.linalg.norm(acc, axis=0, keepdims=True), 1e-12)
    assert np.abs(acc - ref).max() < 1e-5


def test_patch_token_grid_with_pca():
    m = tiny_model()
    vol = smooth_volume((16, 16, 32))
    full = patch_token_grid(vol, m, window=16, overlap_tokens=1)
    pca = fit_pca(np.moveaxis(full, 0, -1).reshape(-1, 24), dim=8)
    red = patch_token_grid(vol, m, window=16, overlap_tokens=1, pca=pca)
    assert red.shape == (8, 2, 2, 4)
    assert np.abs(np.linalg.norm(red, axis=0) - 1).max() < 1e-4


# --------------------------------------------------------------------------- #
# signed permutations (augmentation subset)
# --------------------------------------------------------------------------- #
def test_signed_permutation_detection():
    from tsm.data import SpatialParams, rot90_matrix_inplane, rotation_matrix

    assert signed_permutation(torch.eye(3)) == ([0, 1, 2], [False, False, False])
    p = SpatialParams(flips=(True, False, True), rot90=rot90_matrix_inplane(1))
    assert signed_permutation(p.matrix()) is not None
    assert signed_permutation(rotation_matrix(torch.tensor([1.0, 0, 0]), 0.3)) is None
    assert signed_permutation(torch.diag(torch.tensor([1.5, 1.0, 1.0]))) is None


def test_signed_permutation_matches_augment_grid_sample():
    """``apply_signed_permutation`` reproduces Augment's grid_sample resampling for
    every flip/rot90 transform (this is why the cached tokens may be reused there)."""
    from tsm.data import Augment, SpatialParams, rot90_matrix_inplane

    torch.manual_seed(0)
    x = torch.randn(1, 2, 6, 6, 6)  # apply_spatial resamples channel 0 (ct) and passes 1 (scale) through
    aug = Augment("none")
    for flips in [(False, False, False), (True, False, False), (True, True, True), (False, True, False)]:
        for k in range(4):
            p = SpatialParams(flips=flips, rot90=rot90_matrix_inplane(k) if k else None)
            batch = {"input": x}
            got = aug.apply_spatial(batch, [p])["input"][:, 0:1]
            perm, fl = signed_permutation(p.matrix())
            mine = apply_signed_permutation(x[:, 0:1], perm, fl)
            assert torch.allclose(got, mine, atol=1e-5), f"flips={flips} k={k}"


# --------------------------------------------------------------------------- #
# token cache + feat_distill source
# --------------------------------------------------------------------------- #
class FakeCache:
    """In-memory stand-in for :class:`DinoTokenCache`."""

    def __init__(self, grid=(8, 8, 8), channels=6, scale=8, origin=(0, 0, 0)):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(channels, *grid)).astype(np.float32)
        self.data = a / np.linalg.norm(a, axis=0, keepdims=True)
        self.channels, self.scale, self.origin_zyx = channels, scale, tuple(origin)
        self.shape_zyx, self.path = tuple(grid), "<fake>"

    def read_crop(self, origin_zyx, patch):
        n = patch // self.scale
        lo = []
        for i in range(3):
            v = int(origin_zyx[i]) - self.origin_zyx[i]
            if v % self.scale:
                return None
            v //= self.scale
            if v < 0 or v + n > self.shape_zyx[i]:
                return None
            lo.append(v)
        z, y, x = lo
        return self.data[:, z:z + n, y:y + n, x:x + n].copy()


def test_dino_token_cache_zarr(tmp_path):
    import zarr

    grid = (4, 6, 6)
    a = np.random.default_rng(0).normal(size=(5, *grid)).astype(np.float16)
    p = str(tmp_path / "tokens.zarr")
    zarr.create_array(store=p, shape=a.shape, chunks=(5, 2, 3, 3), dtype=np.float16, fill_value=0,
                      attributes={"origin_zyx": [16, 32, 0], "scale": 8, "voxel_um": 2.4, "l2_normalised": True},
                      overwrite=True)[:] = a
    c = DinoTokenCache(p)
    assert (c.channels, c.scale, c.origin_zyx, c.shape_zyx) == (5, 8, (16, 32, 0), grid)
    got = c.read_crop((16 + 8, 32 + 8, 8), 16)
    assert got.shape == (5, 2, 2, 2)
    assert np.allclose(got, a[:, 1:3, 1:3, 1:3].astype(np.float32))
    assert c.read_crop((0, 0, 0), 16) is None          # before the origin
    assert c.read_crop((16 + 4, 32, 0), 16) is None    # unaligned
    assert c.read_crop((16, 32, 40), 16) is None       # past the end


def test_dino_feat_source_loss_and_projection():
    torch.manual_seed(0)
    cache = FakeCache()
    src = DinoFeatSource(cache, student_ch=12, stage=3)
    feats = {"enc3": torch.randn(2, 12, 4, 4, 4)}
    batch = {"input": torch.zeros(2, 2, 32, 32, 32), "origin_zyx": torch.tensor([[0, 0, 0], [8, 8, 8]])}
    loss, parts = src(feats, batch)
    assert loss.requires_grad and torch.isfinite(loss)
    assert 0.0 <= float(loss.detach()) <= 2.0
    assert parts["feat_dino_frac"] == 1.0
    loss.backward()
    assert src.proj[0].weight.grad is not None
    # a perfect student reaches ~zero loss
    tgt = torch.stack([torch.from_numpy(cache.read_crop((0, 0, 0), 32)), torch.from_numpy(cache.read_crop((8, 8, 8), 32))])
    assert float((1 - F.cosine_similarity(tgt, tgt, dim=1)).mean()) < 1e-6


def test_dino_feat_source_skips_non_signed_permutations():
    from tsm.data import SpatialParams, rot90_matrix_inplane, rotation_matrix

    cache = FakeCache()
    src = DinoFeatSource(cache, student_ch=12, stage=3)
    feats = {"enc3": torch.randn(2, 12, 4, 4, 4)}
    batch = {"input": torch.zeros(2, 2, 32, 32, 32), "origin_zyx": torch.tensor([[0, 0, 0], [8, 8, 8]])}
    flip = SpatialParams(flips=(True, False, False), rot90=rot90_matrix_inplane(2))
    rot = SpatialParams(rotation=rotation_matrix(torch.tensor([1.0, 0, 0]), 0.3))
    _, parts = src(feats, batch, [flip, rot])
    assert parts["feat_dino_frac"] == 0.5
    _, parts = src(feats, batch, [rot, rot])
    assert parts["feat_dino_frac"] == 0.0
    _, parts = src(feats, batch, [flip, flip])
    assert parts["feat_dino_frac"] == 1.0
    # out-of-cache origins are skipped too
    batch2 = dict(batch, origin_zyx=torch.tensor([[0, 0, 0], [900, 0, 0]]))
    _, parts = src(feats, batch2)
    assert parts["feat_dino_frac"] == 0.5


def test_dino_feat_source_target_is_transformed_like_the_crop():
    from tsm.data import SpatialParams

    cache = FakeCache()
    src = DinoFeatSource(cache, student_ch=12, stage=3)
    p = SpatialParams(flips=(False, True, False))
    t = src.target((0, 0, 0), 32, p)
    ref = torch.from_numpy(cache.read_crop((0, 0, 0), 32)).flip(2)
    assert torch.allclose(t, ref)


def test_feat_distiller_with_dino_only():
    from tsm.train import FeatDistiller, feat_distill_opts

    fd = feat_distill_opts({"enabled": True, "teachers": ["dino"], "pairs": [], "weight": 0.5, "dino_stage": 3})
    d = FeatDistiller(fd, widths=(8, 16, 32, 12, 12), device="cpu", dino_cache=FakeCache())
    assert d.dino is not None and not d.encoders
    assert "dino" in "\n".join(d.describe((8, 16, 32, 12, 12)))
    feats = {"enc3": torch.randn(2, 12, 4, 4, 4)}
    batch = {"input": torch.zeros(2, 2, 32, 32, 32), "origin_zyx": torch.tensor([[0, 0, 0], [8, 8, 8]])}
    loss, parts = d(feats, batch["input"][:, 0], batch=batch)
    assert set(parts) >= {"feat", "feat_dino", "feat_dino_frac"}
    assert float(loss.detach()) == pytest.approx(0.5 * parts["feat_dino"], rel=1e-5)
    assert list(d.parameters())
    sd = d.state_dict()
    d.load_state_dict(sd)
    assert any(k.startswith("dino.") for k in sd)
    with pytest.raises(ValueError):
        d(feats, batch["input"][:, 0])  # needs the batch


def test_feat_distill_opts_defaults():
    from tsm.train import feat_distill_opts

    fd = feat_distill_opts({"enabled": True, "teachers": ["recto", "dino"]})
    assert fd["dino_stage"] == 3 and fd["dino_cache"] is None
    with pytest.raises(ValueError):
        feat_distill_opts({"enabled": True, "teachers": []})


# --------------------------------------------------------------------------- #
# stage plumbing
# --------------------------------------------------------------------------- #
def test_dino_stage_dry_run(capsys):
    from tsm.config import parse_config
    from tsm.dino import dino_opts, run_dino

    cfg = parse_config({
        "volume": {"url": "http://example.invalid/x.zarr", "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [0, 0, 0], "size_zyx": [128, 256, 256]},
        "out_dir": "/tmp/tsm-dino-test",
        "extra": {"dino": {"window": 128, "overlap_tokens": 2, "pca_dim": 64}},
    })
    assert dino_opts(cfg)["window"] == 128
    r = run_dino(cfg, dry_run=True)
    assert r["grid"] == (16, 32, 32)
    # 16 tokens deep = 1 window in z; 32 tokens with 16-token windows and 2 overlap -> 3 starts
    assert r["windows"] == 1 * 3 * 3
    assert r["tokens_bytes"] == 64 * 16 * 32 * 32 * 2
    assert "dry run" in capsys.readouterr().out


def test_dino_opts_rejects_unknown():
    from tsm.config import parse_config
    from tsm.dino import dino_opts

    cfg = parse_config({"volume": {"url": "x"}, "region": {"start_zyx": [0, 0, 0], "size_zyx": [8, 8, 8]},
                        "out_dir": "/tmp/x", "extra": {"dino": {"nope": 1}}})
    with pytest.raises(ValueError):
        dino_opts(cfg)


def test_cli_registers_dino_stage():
    from tsm.cli import STAGES

    assert "dino" in STAGES


def test_run_dino_end_to_end_toy(tmp_path):
    """The full ``tsm dino`` path on a 256^3 (32^3 token) synthetic volume with the
    tiny ViT: PCA fit, windowed stitch, fp16 token zarr and uint8 ink-likeness."""
    import zarr

    from tsm.config import parse_config
    from tsm.dino import run_dino

    vol = smooth_volume((256, 256, 256), seed=3)
    ct = str(tmp_path / "ct.zarr")
    zarr.create_array(store=ct, shape=vol.shape, chunks=(64, 64, 64), dtype="uint8", fill_value=0)[:] = vol
    out = str(tmp_path / "out")
    cfg = parse_config({
        "volume": {"url": ct, "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [0, 0, 0], "size_zyx": [256, 256, 256]},
        "out_dir": out,
        "extra": {"dino": {"window": 128, "overlap_tokens": 2, "pca_dim": 8, "block_windows": 2,
                           "pca_sample_windows": 3, "pca_tokens_per_window": 512, "device": "cpu",
                           "ink_ref": None}},
    })
    ref = tmp_path / "avg_ref_embedding.npy"
    np.save(ref, np.random.default_rng(0).normal(size=24).astype(np.float32))
    cfg.extra["dino"]["ink_ref"] = str(ref)
    r = run_dino(cfg, model=tiny_model())
    assert r["grid"] == (32, 32, 32) and r["windows"] == 3 ** 3 and r["blocks"] == 2 ** 3

    tok = zarr.open_array(store=r["tokens"], mode="r")
    assert tok.shape == (8, 32, 32, 32) and tok.dtype == np.float16
    assert tuple(tok.chunks) == (8, 16, 64, 64)  # as requested; zarr keeps chunks larger than the array
    a = np.asarray(tok[:], dtype=np.float32)
    assert np.isfinite(a).all()
    assert np.abs(np.linalg.norm(a, axis=0) - 1).max() < 2e-2  # fp16 round trip of unit vectors
    assert dict(tok.attrs)["scale"] == 8 and dict(tok.attrs)["origin_zyx"] == [0, 0, 0]

    ink = zarr.open_array(store=r["ink"], mode="r")
    assert ink.shape == (1, 32, 32, 32) and ink.dtype == np.uint8
    cos = np.asarray(ink[0], np.float32) / 127.5 - 1.0
    assert -1.0 <= cos.min() and cos.max() <= 1.0 and cos.std() > 0

    pca = np.load(r["pca"])
    assert pca["components"].shape == (8, 24) and pca["mean"].shape == (24,)


# --------------------------------------------------------------------------- block stitching (R06)
class _ContextModel:
    """Deterministic tokens that depend on the whole window (as attention does)."""

    class spec:  # noqa: N801
        patch = 8
        embed_dim = 4

    def forward_features(self, t):
        v = t[0, 0].float()
        wt = v.shape[0] // 8
        ctx = float(v.mean())
        flat = v.reshape(wt, 8, wt, 8, wt, 8).mean(dim=(1, 3, 5)).reshape(-1)
        tok = torch.zeros(flat.shape[0], 4)
        for c in range(4):
            tok[:, c] = flat * (c + 1) + ctx * (4 - c) * 0.7 + c * c
        return {"patch_tokens": tok[None]}


def _block_stitch(vol, model, starts, grid, per, wt, p, ov):
    """run_dino's block loop, reduced to the accumulate/normalise/write core."""
    from tsm.dino import _blocks, patch_token_grid

    out = np.zeros((4, *grid), np.float32)
    for sel, org, core in _blocks(starts, per, grid):
        span = tuple(sel[i][-1] - sel[i][0] + wt for i in range(3))
        sub = vol[org[0] * p:(org[0] + span[0]) * p, org[1] * p:(org[1] + span[1]) * p,
                  org[2] * p:(org[2] + span[2]) * p]
        acc = np.zeros((4, *span), np.float32)
        wsum = np.zeros(span, np.float32)
        rel = [[s - org[i] for s in sel[i]] for i in range(3)]
        patch_token_grid(sub, model, window=wt * p, overlap_tokens=ov, normalize="raw",
                         out=acc, weight_out=wsum, starts=rel)
        acc /= np.maximum(wsum, np.finfo(np.float32).eps)[None]
        g = torch.from_numpy(acc)
        g = g / g.norm(dim=0, keepdim=True).clamp_min(1e-12)
        loc = tuple(slice(core[i][0] - org[i], core[i][1] - org[i]) for i in range(3))
        sl = tuple(slice(core[i][0], core[i][1]) for i in range(3))
        out[(slice(None), *sl)] = g.numpy()[(slice(None), *loc)]
    return out


def test_dino_block_stitching_is_independent_of_block_windows():
    """R06: blocks used to overwrite the overlap instead of combining every contributing window."""
    from tsm.dino import patch_token_grid, window_starts

    vol = np.random.default_rng(0).integers(0, 255, (32, 32, 32), dtype=np.uint8)
    model, p, wt, ov = _ContextModel(), 8, 2, 1
    grid = (4, 4, 4)
    starts = [window_starts(g, wt, ov) for g in grid]
    ref = patch_token_grid(vol, model, window=wt * p, overlap_tokens=ov, normalize="raw")
    for per in (1, 2, 3):
        got = _block_stitch(vol, model, starts, grid, per, wt, p, ov)
        assert np.abs(got - ref).max() < 1e-5, (per, float(np.abs(got - ref).max()))


def test_dino_axis_blocks_cover_the_grid_exactly():
    from tsm.dino import _axis_blocks, window_starts

    for g, wt, ov in ((4, 2, 1), (32, 16, 2), (10, 4, 1), (7, 3, 2)):
        s = window_starts(g, wt, ov)
        for per in (1, 2, 3, 5):
            blocks = _axis_blocks(s, per, g)
            cores = [(lo, hi) for _, lo, hi in blocks]
            assert cores[0][0] == 0 and cores[-1][1] == g
            assert all(a[1] == b[0] for a, b in zip(cores, cores[1:])), cores  # disjoint and contiguous
            for contrib, lo, hi in blocks:
                covering = [x for x in s if x < hi and x + wt > lo]
                assert contrib == covering
                assert min(contrib) <= lo and max(contrib) + wt >= hi
