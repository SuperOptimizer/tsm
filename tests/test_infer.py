"""Student inference + lasagna / spiral export on synthetic data (CPU)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch
import zarr
from scipy import ndimage as ndi

from synth import AXIS_YX, PERIOD_FINE, coarse_fields, fine_fields, make_synthetic
from tsm.config import parse_config
from tsm.data import CLIP, decode_density, decode_prob, decode_sdf, decode_signed, encode_density, encode_prob, encode_sdf, encode_signed
from tsm.infer import (
    N_HEAD_CH,
    PRED_CHANNELS,
    SPIRAL_CHANNELS,
    StudentNet,
    activate_heads,
    decode_pred,
    encode_pred_dt,
    export_lasagna,
    export_spiral,
    extract_surface,
    hemisphere_nxny,
    load_student,
    omezarr_level_shape,
    pool_mean,
    run_infer,
    to_unit,
    tta_forward,
    tta_inverse,
    tta_transforms,
)
from tsm.labels import neighbor_count6
from tsm.student import TSMNet, normalize_ct
from tsm.train import EMA, save_checkpoint
from tsm.volume import BrickWriter

STRUCT6 = ndi.generate_binary_structure(3, 1)


# --------------------------------------------------------------------------- #
# encodings
# --------------------------------------------------------------------------- #
def test_to_unit_matches_label_store_bytes():
    rng = np.random.default_rng(0)
    B, P = 2, 4
    phys = np.zeros((B, N_HEAD_CH, P, P, P), np.float32)
    phys[:, 0] = rng.uniform(-30, 30, (B, P, P, P))  # sdf (beyond clip on purpose)
    for c in (1, 2, 9, 10):
        phys[:, c] = rng.uniform(0, 1, (B, P, P, P))
    ang = rng.uniform(0, 2 * np.pi, (B, P, P, P))
    phys[:, 3], phys[:, 4] = np.sin(ang), np.cos(ang)
    phys[:, 5] = rng.uniform(0, 0.25, (B, P, P, P))
    n = rng.normal(size=(B, 3, P, P, P))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    phys[:, 6:9] = n
    u = to_unit(torch.from_numpy(phys), CLIP).numpy()
    assert u.min() >= 0 and u.max() <= 1
    q = np.round(u * 255).astype(np.uint8)
    np.testing.assert_array_equal(q[:, 0], encode_sdf(phys[:, 0]))
    assert q[:, 0].min() >= 1  # 0 stays reserved for no data
    for c in (1, 2, 9, 10):
        np.testing.assert_array_equal(q[:, c], encode_prob(phys[:, c]))
    for c in (3, 4, 6, 7, 8):
        np.testing.assert_array_equal(q[:, c], encode_signed(phys[:, c]))
    np.testing.assert_array_equal(q[:, 5], encode_density(phys[:, 5]))
    dec = decode_pred(np.concatenate([q[0], np.zeros((1, P, P, P), np.uint8)]), CLIP)
    assert np.abs(dec["sdf"] - np.clip(phys[0, 0], -CLIP, CLIP)).max() < CLIP / 127 + 1e-4
    assert np.abs(dec["sin"] - phys[0, 3]).max() < 1 / 127
    assert np.abs(dec["density"] - phys[0, 5]).max() < 1e-3 + 1e-6
    assert dec["data"].all() and not dec["surface1"].any()


def test_activate_heads_shapes_and_ranges():
    net = TSMNet(widths=(8, 16, 16)).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 2, 16, 16, 16))
        a = activate_heads(out)
    assert a.shape == (1, N_HEAD_CH, 16, 16, 16) and torch.isfinite(a).all()
    assert torch.allclose((a[:, 3] ** 2 + a[:, 4] ** 2), torch.ones_like(a[:, 3]), atol=1e-4)
    assert torch.allclose((a[:, 6:9] ** 2).sum(1), torch.ones_like(a[:, 3]), atol=1e-4)
    assert a[:, 5].min() >= 0 and a[:, [1, 2, 9, 10]].min() >= 0 and a[:, [1, 2, 9, 10]].max() <= 1
    w = StudentNet(net, 2.4, CLIP)
    with torch.no_grad():
        u = w(normalize_ct(torch.randint(0, 255, (1, 16, 16, 16), dtype=torch.uint8)).unsqueeze(1))
    assert u.shape == (1, N_HEAD_CH, 16, 16, 16) and u.min() >= 0 and u.max() <= 1


# --------------------------------------------------------------------------- #
# student TTA
# --------------------------------------------------------------------------- #
class _EquivariantNet(torch.nn.Module):
    """Exactly flip/transpose-equivariant synthetic student: scalar heads = functions of the CT, the
    normal = the (replicate-padded) central-difference gradient of the CT (a true vector field)."""

    def forward(self, x):
        ct = x[:, 0:1]
        p = torch.nn.functional.pad(ct, (1, 1, 1, 1, 1, 1), mode="replicate")
        gz = 0.5 * (p[:, :, 2:, 1:-1, 1:-1] - p[:, :, :-2, 1:-1, 1:-1])
        gy = 0.5 * (p[:, :, 1:-1, 2:, 1:-1] - p[:, :, 1:-1, :-2, 1:-1])
        gx = 0.5 * (p[:, :, 1:-1, 1:-1, 2:] - p[:, :, 1:-1, 1:-1, :-2])
        return {
            "surface": torch.cat([3.0 * ct, -ct], 1),
            "ink": 0.5 * ct,
            "winding": torch.cat([torch.sin(ct), torch.cos(ct), ct.abs(), gx, gy, gz, ct * 2, ct * 0.1], 1),
        }


class _ConstantNormalNet(_EquivariantNet):
    def forward(self, x):
        o = super().forward(x)
        n = torch.zeros_like(o["winding"][:, 3:6])
        n[:, 2] = 1.0  # nz = +1 everywhere: NOT equivariant (a z-flip must negate it)
        o["winding"] = torch.cat([o["winding"][:, :3], n, o["winding"][:, 6:]], 1)
        return o


@pytest.mark.parametrize("mode", ["flip8", "flip8_rot4"])
def test_tta_equals_no_tta_on_an_equivariant_net(mode):
    torch.manual_seed(0)
    x = normalize_ct(torch.rand(2, 12, 12, 12)).unsqueeze(1)
    base = StudentNet(_EquivariantNet(), 2.4, CLIP, tta="none")
    tta = StudentNet(_EquivariantNet(), 2.4, CLIP, tta=mode)
    assert len(tta.tta) == {"flip8": 8, "flip8_rot4": 16}[mode] and len(set(tta.tta)) == len(tta.tta)
    with torch.no_grad():
        r0, r1 = base.raw(x), tta.raw(x)
        u0, u1 = base(x), tta(x)
    for k in r0:
        torch.testing.assert_close(r1[k], r0[k], atol=1e-5, rtol=0)
    torch.testing.assert_close(u1, u0, atol=1e-5, rtol=0)
    # a non-equivariant normal is averaged in the pre-activation space: nz = +1 and -1 cancel
    with torch.no_grad():
        rc = StudentNet(_ConstantNormalNet(), 2.4, CLIP, tta=mode).raw(x)
    assert rc["winding"][:, 5].abs().max() < 1e-6 and rc["winding"][:, 3:5].abs().max() < 1e-6
    with pytest.raises(ValueError):
        tta_transforms("flip2")
    assert tta_transforms(False) == tta_transforms("none") and len(tta_transforms(True)) == 8


def test_tta_inverse_undoes_forward_per_channel():
    torch.manual_seed(1)
    out = {"surface": torch.randn(1, 2, 4, 6, 6), "ink": torch.randn(1, 1, 4, 6, 6), "winding": torch.randn(1, 8, 4, 6, 6)}
    for axes, tr in tta_transforms("flip8_rot4"):
        fwd = {k: tta_forward(v, axes, tr) for k, v in out.items()}
        # forward-transformed *vector* field: rotate the normal like the grid, then invert
        n = fwd["winding"][:, 3:6].clone()
        for a in axes:  # flips first (negate the flipped component) ...
            n[:, 2 - a] *= -1
        if tr:  # ... then the y<->x transpose swaps nx and ny
            n = n[:, [1, 0, 2]]
        fwd["winding"] = torch.cat([fwd["winding"][:, :3], n, fwd["winding"][:, 6:]], 1)
        back = tta_inverse(fwd, axes, tr)
        for k in out:
            torch.testing.assert_close(back[k], out[k])


# --------------------------------------------------------------------------- #
# surface extraction
# --------------------------------------------------------------------------- #
def test_extract_surface_plane_is_one_voxel_thick_and_6_connected():
    Z, Y, X = 24, 20, 20
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    sdf = (z - 11.3 - 0.3 * x + 0.2 * y).astype(np.float32)  # tilted plane, |slope| < 1 along z
    sdf_u8 = encode_sdf(sdf)
    valid = np.full(sdf.shape, 255, np.uint8)
    surf = extract_surface(sdf_u8, valid, CLIP)
    assert surf.dtype == bool and surf.sum() == Y * X
    assert (surf.sum(axis=0) == 1).all()  # exactly one voxel per column
    assert (sdf[surf] <= 0).all() and (sdf[surf] > -1.0).all()  # the negative side of the crossing
    # a tilted digital plane's inner boundary is a staircase: one 26-component (its 6-components
    # are what surface_stats reports as components_6 vs components_26)
    _, n26 = ndi.label(surf, structure=np.ones((3, 3, 3), bool))
    assert n26 == 1
    flat = extract_surface(encode_sdf((z - 11.3).astype(np.float32)), valid, CLIP)
    assert ndi.label(flat, structure=STRUCT6)[1] == 1 and flat.sum() == Y * X and flat[11].all()
    # restricted to valid > 0.5 and to data voxels
    valid[:, :5] = 100
    sdf_u8[:, :, :3] = 0
    s2 = extract_surface(sdf_u8, valid, CLIP)
    assert not s2[:, :5].any() and not s2[:, :, :3].any() and s2[:, 5:, 3:].sum() == (Y - 5) * (X - 3)
    # an exact-zero medial voxel (byte 128) between +-1 neighbours is the surface itself
    sdf3 = np.zeros((5, 3, 3), np.float32)
    sdf3[0], sdf3[1], sdf3[2], sdf3[3], sdf3[4] = -2.0, -1.0, 0.0, 1.0, 2.0
    s3 = extract_surface(encode_sdf(sdf3), np.full(sdf3.shape, 255, np.uint8), CLIP)
    assert s3[2].all() and not s3[1].any() and not s3[3].any()
    assert neighbor_count6(s3)[s3].min() >= 1


# --------------------------------------------------------------------------- #
# run_infer end to end (tiny random net)
# --------------------------------------------------------------------------- #
def _checkpoint(path: str, widths=(8, 16, 16, 16, 16)) -> None:
    net = TSMNet(widths=widths)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    ema = EMA(net, 0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    ema.update(net)
    save_checkpoint(path, net, ema, opt, None, 7, {"widths": list(widths), "train": {"widths": list(widths)}})


def _cfg(root, ct_url, infer):
    return parse_config({
        "volume": {"url": ct_url, "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root,
        "extra": {"infer": infer},
    })


def test_run_infer_synthetic(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root)
    ckpt = os.path.join(root, "train", "latest.pt")
    _checkpoint(ckpt)
    model, info = load_student(ckpt)
    assert info["ema"] and info["step"] == 7 and info["widths"] == [8, 16, 16, 16, 16]
    cfg = _cfg(root, s["ct"], {"patch": 32, "out_tile": 64, "chunk": 32, "stats_box": [32, 64, 64], "prefetch": False, "device": "cpu"})
    d = run_infer(cfg, dry_run=True)
    assert d["tiles"] == 1 and not os.path.exists(os.path.join(root, "student"))
    summary = run_infer(cfg)
    pred = os.path.join(root, "student", "pred.zarr")
    arr = zarr.open_array(store=pred, mode="r")
    assert arr.shape == (len(PRED_CHANNELS), 64, 64, 64) and arr.dtype == np.uint8
    assert list(arr.attrs["channels"]) == PRED_CHANNELS and list(arr.attrs["origin_zyx"]) == [16, 32, 32]
    a = np.asarray(arr[:])
    dec = decode_pred(a, CLIP)
    ct = zarr.open_array(store=s["ct"], mode="r")[16:80, 32:96, 32:96]
    covered = a[0] != 0
    assert covered.mean() > 0.8  # the air strip is narrower than a window, so every voxel is covered
    assert np.abs(dec["sdf"]).max() <= CLIP + 1e-6 and np.isfinite(dec["sdf"]).all()
    for k in ("valid", "ink", "conf"):
        assert dec[k].min() >= 0 and dec[k].max() <= 1
    sc = np.sqrt(dec["sin"] ** 2 + dec["cos"] ** 2)[covered]
    assert sc.max() < 1.02  # unit circle, up to blending / quantisation
    nn = np.sqrt(dec["nx"] ** 2 + dec["ny"] ** 2 + dec["nz"] ** 2)[covered]
    assert nn.max() < 1.02
    # surface1 is a subset of the negative-side zero crossings inside valid > 0.5
    ref = extract_surface(a[0], a[1], CLIP)
    np.testing.assert_array_equal(dec["surface1"], ref)
    sj = json.load(open(os.path.join(root, "student", "pred.summary.json")))
    assert sj["channels"] == PRED_CHANNELS and sj["clip"] == CLIP and sj["student"]["step"] == 7
    assert sj["volume_shape_zyx"] == list(ct.shape) or len(sj["volume_shape_zyx"]) == 3
    assert "surface1" in sj and sj["surface1"]["n_voxels"] == int(ref.sum())
    assert os.path.exists(os.path.join(root, "student", "preview_sdf.png"))
    assert os.path.exists(os.path.join(root, "student", "preview_normal.png"))
    assert summary["done"] == 1 and summary["tta"] == "none"
    with pytest.raises(ValueError):
        run_infer(_cfg(root, s["ct"], {"bogus": 1}), dry_run=True)
    with pytest.raises(ValueError):
        run_infer(_cfg(root, s["ct"], {"tta": "flip2"}), dry_run=True)
    d = run_infer(_cfg(root, s["ct"], {"patch": 32, "out_tile": 64, "tta": "flip8_rot4", "device": "cpu"}), dry_run=True)
    assert d["tiles"] == 1


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def _pred_store(path: str, origin=(16, 32, 32), shape=(64, 128, 128), chunk=32) -> dict[str, np.ndarray]:
    """Synthetic pred.zarr: sheets along z (fine_fields) + concentric winding field about AXIS_YX."""
    ff = fine_fields(shape, seed=1)
    cf = coarse_fields(shape, seed=1, origin=origin, factor=1)  # fine-resolution winding field
    valid = (ff["sdf_valid"] > 0) & (cf["valid"] > 0)
    fields = {
        "sdf": ff["sdf"], "valid": valid.astype(np.float32), "ink": ff["ink"], "sin": cf["phase_sin"], "cos": cf["phase_cos"],
        "density": cf["density"], "nx": cf["nx"], "ny": cf["ny"], "nz": cf["nz"], "conf": cf["conf"],
        "spare": np.zeros(shape, np.float32), "surface1": np.zeros(shape, np.float32),
    }
    enc = {
        "sdf": encode_sdf(fields["sdf"]), "valid": encode_prob(fields["valid"]), "ink": encode_prob(fields["ink"]),
        "sin": encode_signed(fields["sin"]), "cos": encode_signed(fields["cos"]), "density": encode_density(fields["density"]),
        "nx": encode_signed(fields["nx"]), "ny": encode_signed(fields["ny"]), "nz": encode_signed(fields["nz"]),
        "conf": encode_prob(fields["conf"]), "spare": np.zeros(shape, np.uint8), "surface1": np.zeros(shape, np.uint8),
    }
    enc["sdf"][ff["sdf_valid"] == 0] = 0  # no data in the air strip
    w = BrickWriter(path, PRED_CHANNELS, shape, chunk=chunk, origin_zyx=origin, voxel_um=2.4, scale=1.0)
    for ci, name in enumerate(PRED_CHANNELS):
        for z0 in range(0, shape[0], chunk):
            for y0 in range(0, shape[1], chunk):
                for x0 in range(0, shape[2], chunk):
                    blk = enc[name][z0:z0 + chunk, y0:y0 + chunk, x0:x0 + chunk]
                    w.write(ci, origin[0] + z0, origin[1] + y0, origin[2] + x0, np.ascontiguousarray(blk))
    fields["mask"] = valid & (enc["sdf"] != 0)
    return fields


def test_pool_mean_and_pred_dt_and_hemisphere():
    v = np.arange(2 * 4 * 4 * 4, dtype=np.float32).reshape(2, 4, 4, 4)
    m = np.ones((4, 4, 4), bool)
    m[0, 0, 0] = False
    mean, cover = pool_mean(v, m, 2)
    assert mean.shape == (2, 2, 2, 2) and cover.shape == (2, 2, 2)
    assert cover[0, 0, 0] == 7 / 8 and cover[1, 1, 1] == 1.0
    blk = v[0, :2, :2, :2]
    assert np.isclose(mean[0, 0, 0, 0], blk[m[:2, :2, :2]].mean())
    sdf = np.array([0.0, 1.0, 2.0, 2.4, 3.0, 10.0, 60.0, -5.0], np.float32)
    dt = encode_pred_dt(sdf, np.ones(8, bool), half=2.0)
    np.testing.assert_array_equal(dt, [130, 129, 128, 127, 127, 120, 80, 125])
    assert encode_pred_dt(sdf, np.zeros(8, bool), 2.0).max() == 0
    nx, ny = hemisphere_nxny(np.array([0.6]), np.array([0.0]), np.array([-0.8]))
    assert nx[0] == np.uint8(np.rint(-0.6 * 127 + 128)) and ny[0] == 128
    d = (nx.astype(np.float32) - 128) / 127
    assert np.isclose(np.sqrt(1 - d * d), 0.8, atol=0.01)
    assert omezarr_level_shape((100, 7, 1), 3) == (13, 1, 1)


def _load_3d_like(manifest: str, crop_xyzwhd):
    """Minimal re-implementation of villa fit_data.load_3d's reading/decoding (no villa import)."""
    d = json.load(open(manifest))
    base = os.path.dirname(manifest)
    gm_enc = d["grad_mag_encode_scale"] / d["grad_mag_factor"]
    s2b = d["source_to_base"]
    out = {}
    for name in ["cos", "grad_mag", "nx", "ny", "pred_dt"]:
        grp = next(g for g in d["groups"].values() if name in g["channels"])
        z = zarr.open(os.path.join(base, grp["zarr"]), mode="r")
        assert isinstance(z, zarr.Array) and z.ndim == 3
        exp = omezarr_level_shape(d["base_shape_zyx"], grp["scaledown"])
        assert tuple(int(v) for v in z.shape) == exp, (name, z.shape, exp)
        ds = int(round((1 << grp["scaledown"]) * s2b))
        x0, y0, z0, cw, chh, cd = crop_xyzwhd
        a = np.asarray(z[z0 // ds:(z0 + cd + ds - 1) // ds, y0 // ds:(y0 + chh + ds - 1) // ds, x0 // ds:(x0 + cw + ds - 1) // ds])
        out[name + "_u8"] = a
        out[name + "_ds"] = ds
    out["cos"] = out["cos_u8"] / 255.0
    out["grad_mag"] = out["grad_mag_u8"] / gm_enc
    out["nx"] = (out["nx_u8"] - 128.0) / 127.0
    out["ny"] = (out["ny_u8"] - 128.0) / 127.0
    out["nz"] = np.sqrt(np.clip(1 - out["nx"] ** 2 - out["ny"] ** 2, 0, None))
    return d, out


def test_export_lasagna_roundtrip(tmp_path):
    pred = str(tmp_path / "pred.zarr")
    origin, shape = (16, 32, 32), (64, 128, 128)
    f = _pred_store(pred, origin, shape)
    out = str(tmp_path / "lasagna")
    r = export_lasagna(pred, out, name="toy", level_shift=2, cos_scaledown=2, scaledown=4, base_shape_zyx=(96, 192, 192),
                       brick=(32, 64, 64), log=lambda m: None)
    manifest = os.path.join(out, "toy.lasagna.json")
    assert r["manifest"] == manifest and os.path.exists(manifest)
    d, got = _load_3d_like(manifest, r["crop_xyzwhd"])
    assert set(d) >= {"version", "source_to_base", "grad_mag_encode_scale", "grad_mag_factor", "groups", "crops", "base_shape_zyx", "preprocess_params"}
    assert d["groups"]["cos"]["scaledown"] == 3 and d["groups"]["grad_mag"]["scaledown"] == 4 and d["groups"]["pred_dt"]["scaledown"] == 3
    assert d["groups"]["cos"]["zarr"] == "toy_cos.ome.zarr/3" and d["grad_mag_factor"] == 0.25
    pp = d["preprocess_params"]
    assert pp["source"] == "tsm" and pp["scaledown"] == 4 and pp["cos_scaledown"] == 2 and pp["grad_mag_encode_scale"] == 1000
    assert pp["channels"] == ["cos", "grad_mag", "nx", "ny", "pred_dt"] and pp["crop_xyzwhd"] == [32, 32, 16, 128, 128, 64]
    assert d["crops"] == [[32, 32, 16, 128, 128, 64]]
    # OME metadata as villa writes it
    za = json.load(open(os.path.join(out, "toy_cos.ome.zarr", ".zattrs")))
    assert za["multiscales"][0]["datasets"][0]["path"] == "3" and za["lasagna_pyramid_downsample"] == "mean_pool2x"
    assert json.load(open(os.path.join(out, "toy_cos.ome.zarr", "3", ".zarray")))["dimension_separator"] == "/"
    assert os.path.exists(os.path.join(out, "toy_nx.ome.zarr", "5"))  # n_levels >= other_level + 2

    # --- decode vs the mask-aware pooled ground truth ---
    mask = f["mask"]
    fc, fo = 8, 16
    # crop reads level indices [origin//ds, ceil((origin+size)/ds)); our pooled bricks are aligned the same way
    def pooled(vals, fac):
        lo = [(o // fac) * fac for o in origin]
        hi = [int(np.ceil((o + s) / fac)) * fac for o, s in zip(origin, shape)]
        big = np.zeros((vals.shape[0],) + tuple(h - l for l, h in zip(lo, hi)), np.float32)
        mbig = np.zeros(big.shape[1:], bool)
        sl = tuple(slice(o - l, o - l + s) for o, l, s in zip(origin, lo, shape))
        big[(slice(None),) + sl] = vals
        mbig[sl] = mask
        return pool_mean(big, mbig, fac)

    m, cover = pooled(np.stack([f["sin"], f["cos"]]), fc)
    ok = cover >= 0.5
    sc = m / np.maximum(np.sqrt((m ** 2).sum(0, keepdims=True)), 1e-6)
    assert got["cos"].shape == ok.shape
    assert np.abs(got["cos"][ok] - (0.5 + 0.5 * sc[1][ok])).max() < 2.5 / 255
    assert (got["cos_u8"][~ok] == 0).all()
    mo, cover_o = pooled(np.stack([f["density"], f["nx"], f["ny"], f["nz"]]), fo)
    oko = cover_o >= 0.5
    # grad_mag decodes to wraps per BASE (2.4 um) voxel = 1/PERIOD_FINE
    assert np.abs(got["grad_mag"][oko] - mo[0][oko]).max() < 1.5e-3 and np.isclose(mo[0][oko].mean(), 1 / PERIOD_FINE, atol=1e-4)
    assert (got["grad_mag_u8"][~oko] == 0).all() and (got["nx_u8"][~oko] == 128).all()
    n = mo[1:4] / np.maximum(np.sqrt((mo[1:4] ** 2).sum(0, keepdims=True)), 1e-6)
    flip = np.where(n[2] < 0, -1.0, 1.0)
    dots = (got["nx"] * n[0] * flip + got["ny"] * n[1] * flip + got["nz"] * np.abs(n[2]))[oko]
    assert dots.min() > 0.98  # hemisphere-encoded normal recovers the pooled normal
    dt = got["pred_dt_u8"]
    assert set(np.unique(dt)) <= ({0} | set(range(80, 176)))
    assert (dt[ok] > 0).all() and (dt[~ok] == 0).all()
    # sheets every PERIOD_FINE along z: the mean of the encoded band over an 8-cell is well inside [80, 175]
    assert 100 < dt[ok].mean() < 140
    with pytest.raises(FileExistsError):
        export_lasagna(pred, out, name="toy", base_shape_zyx=(96, 192, 192), log=lambda m: None)
    r2 = export_lasagna(pred, out, name="toy", base_shape_zyx=(96, 192, 192), brick=(32, 64, 64), force=True, log=lambda m: None)
    assert r2["bricks"] == r["bricks"]


def test_export_spiral_rays_monotone(tmp_path):
    pred = str(tmp_path / "pred.zarr")
    origin, shape = (16, 32, 32), (64, 128, 128)
    _pred_store(pred, origin, shape)
    axis = str(tmp_path / "axis.json")
    zs = [0, 200]
    json.dump({"coordinate_order": "xyz", "coordinate_space": "full_resolution",
               "control_points": [{"z": z, "y": AXIS_YX[0], "x": AXIS_YX[1]} for z in zs]}, open(axis, "w"))
    out = str(tmp_path / "spiral")
    r = export_spiral(pred, out, level_shift=2, axis=axis, ray_spacing=4, ray_angle_deg=45.0, ray_step=1.0, brick=(32, 64, 64),
                      log=lambda m: None)
    w = zarr.open_array(store=r["winding_zarr"], mode="r")
    assert list(w.attrs["channels"]) == SPIRAL_CHANNELS and w.shape == (8, 16, 32, 32)
    assert list(w.attrs["origin_zyx"]) == [4, 8, 8] and w.attrs["voxel_um"] == 9.6
    a = np.asarray(w[:])
    valid = a[7] > 0
    assert valid.mean() > 0.5
    sc = np.sqrt(decode_signed(a[0]) ** 2 + decode_signed(a[1]) ** 2)[valid]
    assert sc.min() > 0.97 and sc.max() < 1.02
    assert np.abs(decode_density(a[2])[valid] - 4 / PERIOD_FINE).max() < 2e-3  # wraps per level-2 voxel
    assert (decode_prob(a[6])[valid] > 0.5).all() and (a[:7][:, ~valid] == 0).all()
    j = json.load(open(r["json"]))
    assert j["channels"] == SPIRAL_CHANNELS and j["factor"] == 4 and j["origin_zyx"] == [4, 8, 8] and j["rays"]["file"] == "rays.npz"
    rays = np.load(r["rays"])
    n_z = len(range(0, 16, 4))
    assert rays["centers_zyx"].shape == (n_z * 8, 3) and rays["directions_zyx"].shape == (n_z * 8, 3)
    assert np.allclose(np.linalg.norm(rays["directions_zyx"], axis=1), 1.0, atol=1e-6)
    assert (rays["directions_zyx"][:, 0] == 0).all()  # in-plane radial rays
    # centre = midpoint of the sample run along the direction
    n = rays["phase"].shape[1]
    np.testing.assert_allclose(rays["centers_zyx"], rays["starts_zyx"] + 0.5 * (n - 1) * float(rays["spacing"]) * rays["directions_zyx"], atol=1e-4)
    assert rays["points_zyx"].shape == (n_z * 8, n, 3)
    # starts sit on the axis in level-2 coordinates: (100 - 1.5) / 4
    np.testing.assert_allclose(rays["starts_zyx"][:, 1:], np.broadcast_to((np.array(AXIS_YX) - 1.5) / 4, (n_z * 8, 2)), atol=1e-5)
    ph, ok = rays["phase"], rays["valid"]
    assert ok.sum() > 0 and np.isnan(ph[~ok]).all() and np.isfinite(ph[ok]).all()
    n_checked = 0
    for r_ in range(ph.shape[0]):
        idx = np.flatnonzero(ok[r_])
        if idx.size < 4:
            continue
        idx = idx[idx >= 2]  # the pooled cells on the axis itself straddle every direction (phase undefined there)
        p = ph[r_, idx]
        d = np.diff(p)
        assert (d > 0).all(), (r_, d.min())  # phase increases outward along every ray
        # w = r / PERIOD_FINE -> 4 / PERIOD_FINE wraps per level-2 voxel step
        assert np.isclose(np.median(d), 4 / PERIOD_FINE, atol=0.02)
        n_checked += 1
    assert n_checked >= n_z * 4
    with pytest.raises(FileExistsError):
        export_spiral(pred, out, level_shift=2, log=lambda m: None)


def test_export_stage_config_validation(tmp_path):
    from tsm.infer import EXPORT_DEFAULTS, _opts

    cfg = parse_config({"volume": {"url": "x"}, "region": {"start_zyx": [0, 0, 0], "size_zyx": [1, 1, 1]}, "out_dir": str(tmp_path),
                        "extra": {"export": {"lasagna": {"name": "a"}, "spiral": False}}})
    o = _opts(cfg, "export", EXPORT_DEFAULTS)
    assert o["lasagna"]["name"] == "a" and o["lasagna"]["scaledown"] == 4 and o["spiral"] is False
    cfg.extra["export"]["lasagna"]["nope"] = 1
    with pytest.raises(ValueError):
        _opts(cfg, "export", EXPORT_DEFAULTS)


@pytest.mark.slow
def test_villa_load_3d_reads_export(tmp_path):
    """Point villa's fit_data.load_3d at the export (needs villa's lasagna deps importable)."""
    lasagna_dir = os.path.expanduser("~/villa/lasagna")
    if not os.path.isdir(lasagna_dir):
        pytest.skip("villa lasagna checkout not found")
    sys.path.insert(0, lasagna_dir)
    try:
        import fit_data  # type: ignore
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"villa fit_data not importable: {exc}")
    finally:
        sys.path.pop(0)
    pred = str(tmp_path / "pred.zarr")
    _pred_store(pred)
    out = str(tmp_path / "lasagna")
    r = export_lasagna(pred, out, name="toy", base_shape_zyx=(96, 192, 192), brick=(32, 64, 64), log=lambda m: None)
    data = fit_data.load_3d(path=r["manifest"], device=torch.device("cpu"), crop=tuple(r["crop_xyzwhd"]))
    assert data.cos is not None and data.grad_mag is not None and data.pred_dt is not None
    assert data.grad_mag_scale == 4000.0


def test_run_infer_direction_fibre_store(tmp_path):
    """A direction-mode checkpoint writes the 4 head channels plus the derived vt/hz classes."""
    from tsm.infer import FIBER_DIR_PRED_CHANNELS, n_head_ch, pred_channels
    from tsm.student import build_model

    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True, fiber=True)
    widths = (8, 16, 16, 16, 16)
    net = build_model(widths=widths, in_ch=2, surface_mode="faces", fiber=True, fiber_mode="direction")
    ckpt = os.path.join(root, "train", "latest.pt")
    ema = EMA(net, 0.5)
    ema.update(net)
    save_checkpoint(ckpt, net, ema, torch.optim.AdamW(net.parameters(), lr=1e-3), None, 5,
                    {"widths": list(widths), "surface_mode": "faces", "heads": {"fiber": True},
                     "fiber_mode": "direction", "train": {"widths": list(widths)}})
    cfg = _cfg(root, s["ct"], {"patch": 32, "out_tile": 64, "chunk": 32, "stats_box": [32, 64, 64],
                               "prefetch": False, "device": "cpu"})
    summary = run_infer(cfg)
    ch = pred_channels("faces", True, "direction")
    assert summary["channels"] == ch and summary["fiber_mode"] == "direction"
    assert summary["fiber_derived"]["basis_defined_frac"] > 0.0
    arr = zarr.open_array(store=os.path.join(root, "student", "pred.zarr"), mode="r")
    a = np.asarray(arr[:])
    assert a.shape[0] == len(ch)
    dec = decode_pred(a, CLIP, ch)
    d = np.stack([dec["fiber_dz"], dec["fiber_dy"], dec["fiber_dx"]])
    cov = a[0] != 0
    assert np.linalg.norm(d, axis=0)[cov].max() < 1.02  # a unit direction, up to blending
    for k in ("fiber_strength", "fiber_vt", "fiber_hz"):
        assert dec[k].min() >= 0 and dec[k].max() <= 1
    # the derived classes are bounded by the strength (they are s * |cos|)
    assert (dec["fiber_vt"] <= dec["fiber_strength"] + 2 / 255).all()
    assert n_head_ch("faces", True, "direction") == 16 and ch[12:16] == FIBER_DIR_PRED_CHANNELS
