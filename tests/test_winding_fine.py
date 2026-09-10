"""Native-2.4 um winding field from the two face SDFs (tsm.winding_fine) + the data.py merge."""

from __future__ import annotations

import numpy as np
import pytest

from tsm.data import WF_CHANNELS, decode_density, decode_signed
from tsm.labels import encode_sdf_u8, fine_channels
from tsm.winding_fine import (
    WF_DEFAULTS,
    decode_density_u8,
    sheet_period,
    upsample_coarse_prior,
    wf_opts,
    winding_fine_box,
)

CLIP = 20.0


# --------------------------------------------------------------------------- #
# synthetic sheet stacks
# --------------------------------------------------------------------------- #
def _sawtooth(d: np.ndarray, period: float) -> np.ndarray:
    """Signed distance to the nearest plane of a stack spaced ``period`` apart, i.e. the
    value in [-P/2, P/2) of ``d`` modulo ``P`` -- exactly what an EDT to that plane set gives."""
    return (np.asarray(d, dtype=np.float64) + period / 2.0) % period - period / 2.0


def _stack(shape=(32, 40, 40), period=24.0, thickness=10.0, kind="planar", tilt=0.0):
    """A stack of parallel/tilted/cylindrical sheets: ``(d, sdf_in, sdf_out, normal)``.

    ``d`` is the outward coordinate with the IN face at ``d = k*period`` and the OUT face at
    ``d = k*period + thickness``, so the true winding is ``w = d / period`` and its fractional
    part is the value the module must reproduce.  Both SDFs are positive on the outward side
    of their own face (the store convention).
    """
    zz, yy, xx = np.mgrid[: shape[0], : shape[1], : shape[2]].astype(np.float64)
    if kind == "planar":
        d = zz
        n = np.zeros((3,) + shape)
        n[0] = 1.0
    elif kind == "tilted":
        v = np.array([1.0, tilt, -0.5 * tilt])
        v /= np.linalg.norm(v)
        d = v[0] * zz + v[1] * yy + v[2] * xx
        n = np.broadcast_to(v[:, None, None, None], (3,) + shape).copy()
    elif kind == "cylinder":  # curved: concentric cylinders about a z-axis outside the box
        cy, cx = -60.0, -70.0
        ry, rx = yy - cy, xx - cx
        r = np.sqrt(ry * ry + rx * rx)
        d = r
        n = np.zeros((3,) + shape)
        n[1], n[2] = ry / r, rx / r
    else:
        raise ValueError(kind)
    return (d, _sawtooth(d, period).astype(np.float32),
            _sawtooth(d - thickness, period).astype(np.float32), n.astype(np.float32))


def _inputs(shape, period, thickness, kind="planar", tilt=0.0, prior_err=0.0, clip=CLIP):
    d, si, so, n = _stack(shape, period, thickness, kind, tilt)
    sdf_in_u8 = encode_sdf_u8(si, clip)
    sdf_out_u8 = encode_sdf_u8(so, clip)
    faces_valid = np.ones(shape, np.uint8)
    err = prior_err if np.ndim(prior_err) else np.full(shape, float(prior_err))
    prior = {"phase": np.mod(d / period + err, 1.0).astype(np.float32),
             "valid": np.ones(shape, np.uint8)}
    return sdf_in_u8, sdf_out_u8, faces_valid, prior, n, d


def _phase(sin_u8, cos_u8):
    return np.mod(np.arctan2(decode_signed(sin_u8), decode_signed(cos_u8)) / (2 * np.pi), 1.0)


def _turn_err(a, b):
    """Circular |a - b| in turns."""
    d = np.abs(a - b) % 1.0
    return np.minimum(d, 1.0 - d)


# --------------------------------------------------------------------------- #
# the geometry
# --------------------------------------------------------------------------- #
def test_sheet_period_recovers_thickness_and_gap():
    shape, P, t = (48, 8, 8), 24.0, 10.0
    _, si, so, _ = _stack(shape, P, t)
    usable = np.ones(shape, bool)
    tf, gf = sheet_period(si, so, usable)
    assert np.allclose(tf, t, atol=1e-3)
    assert np.allclose(gf, P - t, atol=1e-3)


def test_no_gap_in_the_box_is_refused():
    """A box that never shows one of (thickness, gap) cannot know the period -> all invalid."""
    shape = (6, 8, 8)
    si = np.full(shape, 2.0, np.float32)
    so = np.full(shape, -3.0, np.float32)  # D = +5 everywhere: thickness only, no gap
    prior = {"phase": np.zeros(shape, np.float32), "valid": np.ones(shape, np.uint8)}
    out = winding_fine_box(encode_sdf_u8(si, CLIP), encode_sdf_u8(so, CLIP),
                           np.ones(shape, np.uint8), prior, np.zeros((3,) + shape, np.float32))
    assert out[-1].max() == 0


@pytest.mark.parametrize("period,thickness", [(24.0, 10.0), (30.0, 18.0), (16.0, 5.0)])
def test_parallel_sheets_exact_phase_density_and_normal(period, thickness):
    shape = (int(3 * period), 16, 16)
    args = _inputs(shape, period, thickness)
    sdf_in, sdf_out, fv, prior, n, d = args
    radial = np.zeros((3,) + shape, np.float32)
    radial[0] = 1.0
    out = winding_fine_box(sdf_in, sdf_out, fv, prior, radial, CLIP)
    sin_u8, cos_u8, dens, nx, ny, nz, conf, valid = out
    m = valid == 1
    assert m.mean() > 0.9, m.mean()
    # phase: exact both inside the sheets and inside the gaps
    err = _turn_err(_phase(sin_u8, cos_u8)[m], np.mod(d / period, 1.0)[m])
    assert err.max() < 0.02, (err.max(), err.mean())
    # density = 1 / (thickness + gap) = 1 / period, in wraps per fine voxel
    # (the accuracy floor is the SDF byte itself: clip/127 = 0.16 voxels on each of t and g)
    dens_f = decode_density_u8(dens[m])
    assert np.abs(dens_f - 1.0 / period).max() <= 0.03 / period
    # normal: outward (+z), unit
    nn = np.stack([decode_signed(nz), decode_signed(ny), decode_signed(nx)])
    assert np.abs(nn[0][m] - 1.0).max() < 0.02
    assert np.abs(nn[1][m]).max() < 0.02 and np.abs(nn[2][m]).max() < 0.02
    assert conf[m].min() > 200  # the prior agrees exactly


@pytest.mark.parametrize("kind,tilt", [("tilted", 0.5), ("cylinder", 0.0)])
def test_tilted_and_curved_sheets(kind, tilt):
    shape, period, thickness = (40, 40, 40), 24.0, 9.0
    sdf_in, sdf_out, fv, prior, n, d = _inputs(shape, period, thickness, kind, tilt)
    out = winding_fine_box(sdf_in, sdf_out, fv, prior, n, CLIP)
    sin_u8, cos_u8, dens, nx, ny, nz, conf, valid = out
    m = valid == 1
    assert m.mean() > 0.85, m.mean()
    err = _turn_err(_phase(sin_u8, cos_u8)[m], np.mod(d / period, 1.0)[m])
    assert np.percentile(err, 99) < 0.02, (err.max(), np.percentile(err, 99))
    assert np.abs(decode_density_u8(dens[m]) - 1.0 / period).max() <= 0.05 / period
    nn = np.stack([decode_signed(nz), decode_signed(ny), decode_signed(nx)])
    cos_ang = (nn * n).sum(0)[m]
    assert cos_ang.min() > 0.98, cos_ang.min()  # outward and aligned with the true normal


# --------------------------------------------------------------------------- #
# the coarse prior: snapping and the consistency gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("err", [0.0, 0.3, -0.3])
def test_prior_error_under_a_third_of_a_turn_leaves_the_phase_exact(err):
    """With the gate on (opt-in) the phase is still the geometry's; only conf tracks the prior."""
    shape, period, thickness = (48, 16, 16), 24.0, 10.0
    sdf_in, sdf_out, fv, prior, n, d = _inputs(shape, period, thickness, prior_err=err)
    out = winding_fine_box(sdf_in, sdf_out, fv, prior, n, CLIP, snap_max=0.5)
    m = out[-1] == 1
    assert m.mean() > 0.9
    e = _turn_err(_phase(out[0], out[1])[m], np.mod(d / period, 1.0)[m])
    assert e.max() < 0.02, e.max()
    # conf = geometric coherence * (1 - |dw| / snap_max), and the coherence is ~1 here
    assert np.abs(out[6][m].astype(np.float32) / 255.0 - (1.0 - abs(err) / 0.5)).max() < 0.03


# --------------------------------------------------------------------------- #
# the default: the coarse prior does not gate anything
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("err", [0.0, 0.3, 0.5, 0.45])
def test_by_default_the_prior_never_changes_the_result(err):
    """snap_max=None: the phase, the density and the validity come from the faces alone."""
    shape, period, thickness = (48, 16, 16), 24.0, 10.0
    args = _inputs(shape, period, thickness, prior_err=err)
    ref = winding_fine_box(*args[:4], args[4], CLIP)
    for a, b in zip(ref, winding_fine_box(*args[:3], None, args[4], CLIP)):
        np.testing.assert_array_equal(a, b)  # a prior at all is optional
    m = ref[-1] == 1
    assert m.mean() > 0.9
    e = _turn_err(_phase(ref[0], ref[1])[m], np.mod(args[5] / period, 1.0)[m])
    assert e.max() < 0.02


def test_an_aliased_prior_does_not_reduce_the_supervised_fraction():
    """A coarse prior whose period is 6x too long (the crumpled-crop case) must be ignored."""
    shape, period, thickness = (48, 16, 16), 24.0, 10.0
    d, si, so, n = _stack(shape, period, thickness)
    sdf_in, sdf_out = encode_sdf_u8(si, CLIP), encode_sdf_u8(so, CLIP)
    fv = np.ones(shape, np.uint8)
    aliased = {"phase": np.mod(d / (6 * period), 1.0).astype(np.float32),
               "valid": np.ones(shape, np.uint8)}
    exact = {"phase": np.mod(d / period, 1.0).astype(np.float32), "valid": np.ones(shape, np.uint8)}
    ung = [winding_fine_box(sdf_in, sdf_out, fv, p, n, CLIP) for p in (aliased, exact)]
    np.testing.assert_array_equal(ung[0][-1], ung[1][-1])       # same supervised set
    np.testing.assert_array_equal(ung[0][0], ung[1][0])         # same phase
    # ... while the opt-in gate would have thrown most of it away
    gated = winding_fine_box(sdf_in, sdf_out, fv, aliased, n, CLIP, snap_max=0.25)
    assert (gated[-1] == 1).mean() < 0.6 * (ung[0][-1] == 1).mean()


def test_prior_off_by_half_a_turn_is_refused():
    """|wrap(phase_coarse - f)| >= snap_max -> wf_valid = 0 (the spec's consistency check).

    A prior off by exactly half a turn sits on the snap threshold; with the default
    ``snap_max`` = 0.5 it is refused only up to floating point, so the check is exercised
    with ``snap_max`` = 0.4, which refuses every error in [0.4, 0.6]."""
    shape, period, thickness = (48, 16, 16), 24.0, 10.0
    sdf_in, sdf_out, fv, prior, n, _ = _inputs(shape, period, thickness, prior_err=0.5)
    out = winding_fine_box(sdf_in, sdf_out, fv, prior, n, CLIP, snap_max=0.4)
    assert out[-1].max() == 0
    # ... and a 0.3 error is still accepted at the same threshold
    sdf_in, sdf_out, fv, prior, n, _ = _inputs(shape, period, thickness, prior_err=0.3)
    out = winding_fine_box(sdf_in, sdf_out, fv, prior, n, CLIP, snap_max=0.4)
    assert (out[-1] == 1).mean() > 0.9


def test_ignored_faces_are_not_supervised_but_an_ignored_prior_is_irrelevant():
    shape, period, thickness = (48, 16, 16), 24.0, 10.0
    sdf_in, sdf_out, fv, prior, n, _ = _inputs(shape, period, thickness)
    prior["valid"][:, :8] = 2          # coarse ignore
    fv = fv.copy()
    fv[:, 8:12] = 2                    # faces ignore
    fv[:, 12:14] = 0                   # no data
    out = winding_fine_box(sdf_in, sdf_out, fv, prior, n, CLIP)
    v = out[-1]
    assert v[:, 8:14].max() == 0                 # the faces decide
    assert (v[:, :8] == 1).mean() > 0.9          # the coarse ignore does not
    assert (v[:, 14:] == 1).mean() > 0.9
    # with the gate on, the coarse ignore does remove its strip
    g = winding_fine_box(sdf_in, sdf_out, fv, prior, n, CLIP, snap_max=0.5)[-1]
    assert g[:, :14].max() == 0
    # faces_valid == 2 is accepted when the caller asks for it
    out2 = winding_fine_box(sdf_in, sdf_out, fv, prior, n, CLIP, require_faces_valid=False)
    assert (out2[-1][:, 8:12] == 1).mean() > 0.8


def test_phase_offset_reanchors_the_phase():
    shape, period, thickness = (48, 16, 16), 24.0, 10.0
    sdf_in, sdf_out, fv, prior, n, d = _inputs(shape, period, thickness)
    out = winding_fine_box(sdf_in, sdf_out, fv, prior, n, CLIP, phase_offset=0.25)
    m = out[-1] == 1
    e = _turn_err(_phase(out[0], out[1])[m], np.mod(d / period + 0.25, 1.0)[m])
    assert e.max() < 0.02


# --------------------------------------------------------------------------- #
# the coarse prior sampler
# --------------------------------------------------------------------------- #
def test_upsample_coarse_prior_matches_the_coarse_phase():
    from tsm.labels import encode_signed_u8

    cs = (6, 8, 8)
    zz, yy, xx = np.mgrid[: cs[0], : cs[1], : cs[2]]
    w = (yy + 0.25 * xx) / 5.0
    coarse = np.zeros((8,) + cs, np.uint8)
    coarse[0] = encode_signed_u8(np.sin(2 * np.pi * w))
    coarse[1] = encode_signed_u8(np.cos(2 * np.pi * w))
    coarse[7] = 1
    coarse[7, :, 0] = 0  # a strip the prior does not cover
    pr = upsample_coarse_prior(coarse, (0, 0, 0), (8, 16, 16), scale=4)
    assert pr["phase"].shape == (8, 16, 16) and pr["valid"].shape == (8, 16, 16)
    assert (pr["valid"][:, :2] == 0).all()          # the invalid coarse strip
    ok = pr["valid"] == 1
    # fine voxel i sits at coarse coordinate (i + 0.5)/4 - 0.5, and the phase there is the
    # trilinear blend of the coarse sin/cos -- so compare against w at that coordinate
    yc, xc = 9, 13
    cy, cx = (yc + 0.5) / 4 - 0.5, (xc + 0.5) / 4 - 0.5
    assert _turn_err(pr["phase"][0, yc, xc], np.mod((cy + 0.25 * cx) / 5.0, 1.0)) < 0.02
    assert ok.mean() > 0.7


# --------------------------------------------------------------------------- #
# options / channel list
# --------------------------------------------------------------------------- #
def test_wf_opts_and_channels():
    assert wf_opts(None)["enabled"] is False and wf_opts(True)["snap_max"] is None
    assert wf_opts(True)["enabled"] is True
    o = wf_opts({"snap_max": 0.3})
    assert o["enabled"] is True and o["snap_max"] == 0.3 and o["min_period"] == WF_DEFAULTS["min_period"]
    with pytest.raises(ValueError):
        wf_opts({"nope": 1})
    base = fine_channels(faces=True, fiber=True, rv=True)
    full = fine_channels(faces=True, fiber=True, rv=True, winding_fine=True)
    assert full[: len(base)] == base and full[len(base):] == WF_CHANNELS


# --------------------------------------------------------------------------- #
# tsm.data: the fine / coarse / merge winding source
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wf_synth(tmp_path_factory):
    from tests.synth import make_synthetic

    return make_synthetic(str(tmp_path_factory.mktemp("wf")), faces=True, winding_fine=True)


def _dataset(paths, source):
    from tsm.data import CropDataset, LabelStore
    from tsm.volume import VolumeReader

    ct = VolumeReader(paths["ct"], 0, 2.4)
    return CropDataset(ct, LabelStore(paths["fine"]), LabelStore(paths["coarse"]), patch=32, stride=32,
                       augment=False, surface_mode="faces", winding_source=source, min_frac=0.0)


def test_dataset_winding_source_merge_and_fine(wf_synth):
    from tsm.data import LabelStore

    store = LabelStore(wf_synth["fine"])
    origin = (16, 16, 16)
    ds = {s: _dataset(wf_synth, s) for s in ("coarse", "merge", "fine")}
    out = {s: d.load(origin) for s, d in ds.items()}
    wfv = store.read("wf_valid", *origin, 32)[None]
    m = wfv == 1
    assert 0.05 < m.mean() < 0.95, m.mean()  # the fixture supervises only part of the crop

    # default is unchanged behaviour
    assert ds["coarse"].winding_source == "coarse"
    for k in ("winding", "winding_conf", "winding_valid"):
        assert out["coarse"][k].shape[1:] == (32, 32, 32)

    # merge: the fine block where wf_valid == 1, the upsampled coarse targets elsewhere
    for k in ("winding", "winding_conf", "winding_valid"):
        assert np.array_equal(out["merge"][k][np.broadcast_to(~m, out["merge"][k].shape)],
                              out["coarse"][k][np.broadcast_to(~m, out["coarse"][k].shape)])
    assert (out["merge"]["winding_valid"][m] == 1).all()
    sin_ref = decode_signed(store.read("wf_sin", *origin, 32))
    assert np.abs(out["merge"]["winding"][0][m[0]] - sin_ref[m[0]]).max() < 1e-6
    dens_ref = decode_density(store.read("wf_density", *origin, 32))
    assert np.abs(out["merge"]["winding"][2][m[0]] - dens_ref[m[0]]).max() < 1e-6
    # the fine block really is different from the coarse one (the test would be vacuous otherwise)
    assert np.abs(out["merge"]["winding"][0] - out["coarse"]["winding"][0]).max() > 0.1

    # fine: nothing is supervised outside wf_valid == 1
    assert np.array_equal(out["fine"]["winding_valid"], m.astype(np.float32))
    assert (out["fine"]["winding"][np.broadcast_to(~m, out["fine"]["winding"].shape)] == 0).all()
    assert np.abs(out["fine"]["winding"][0][m[0]] - sin_ref[m[0]]).max() < 1e-6


def test_dataset_without_wf_channels_falls_back_to_coarse(tmp_path):
    from tests.synth import make_synthetic

    paths = make_synthetic(str(tmp_path), faces=True, winding_fine=False)
    ds = _dataset(paths, "merge")
    assert ds.winding_source == "coarse" and not ds.has_wf_channels
    ref = _dataset(paths, "coarse").load((16, 16, 16))
    got = ds.load((16, 16, 16))
    for k in ("winding", "winding_conf", "winding_valid"):
        assert np.array_equal(got[k], ref[k])


# --------------------------------------------------------------------------- #
# tsm labels: extra.labels.winding_fine appends the block
# --------------------------------------------------------------------------- #
def test_labels_build_appends_the_winding_fine_block(tmp_path):
    """End-to-end ``tsm labels`` with ``extra.labels.winding_fine`` on the synthetic teachers.

    The block must be appended *after* every other channel (so an older store is a prefix),
    every byte must be produced brick-wise, and the summary must carry the residual stats."""
    import json
    import os

    import zarr

    from tests.test_labels_workers import _axis, _config, _teachers
    from tsm.config import load_config
    from tsm.labels import fine_channels, run_labels

    root = str(tmp_path)
    _teachers(os.path.join(root, "teachers"))
    _axis(os.path.join(root, "axis.json"))
    path = _config(root, os.path.join(root, "out"), 1)
    with open(path) as fh:
        cfg = json.load(fh)
    cfg["extra"]["labels"]["winding_fine"] = {"min_period": 4.0}
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    summary = run_labels(load_config(path))

    arr = zarr.open_array(store=os.path.join(root, "out", "labels", "fine.zarr"), mode="r")
    channels = list(arr.attrs["channels"])
    assert channels == fine_channels(faces=True, fiber=True, rv=False, winding_fine=True)
    assert channels[-8:] == WF_CHANNELS
    wf = summary["fine"]["winding_fine"]
    assert wf["channels"] == WF_CHANNELS and 0.0 <= wf["valid_fraction"] <= 1.0
    assert wf["abs_resid_turns"]["p50"] is None or 0.0 <= wf["abs_resid_turns"]["p50"] <= 0.5
    v = np.asarray(arr[channels.index("wf_valid")])
    assert set(np.unique(v)) <= {0, 1}


def test_labels_rejects_winding_fine_without_faces(tmp_path):
    import json
    import os

    from tests.test_labels_workers import _axis, _config, _teachers
    from tsm.config import load_config
    from tsm.labels import _label_opts

    root = str(tmp_path)
    _teachers(os.path.join(root, "teachers"))
    _axis(os.path.join(root, "axis.json"))
    path = _config(root, os.path.join(root, "out"), 1)
    with open(path) as fh:
        cfg = json.load(fh)
    cfg["extra"]["labels"]["faces"] = {"enabled": False}
    cfg["extra"]["labels"]["winding_fine"] = True
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    with pytest.raises(ValueError, match="faces.enabled"):
        _label_opts(load_config(path))


# --------------------------------------------------------------------------- #
# dev/add_winding_fine.py: appending the block to an existing store
# --------------------------------------------------------------------------- #
def _labels_case(root, winding_fine):
    """A synthetic `tsm labels` build (the tests/test_labels_workers harness)."""
    import json
    import os

    from tests.test_labels_workers import _axis, _config, _teachers
    from tsm.config import load_config
    from tsm.labels import run_labels

    os.makedirs(root, exist_ok=True)
    _teachers(os.path.join(root, "teachers"))
    _axis(os.path.join(root, "axis.json"))
    path = _config(root, os.path.join(root, "out"), 1)
    with open(path) as fh:
        cfg = json.load(fh)
    cfg["extra"]["labels"]["winding_fine"] = {"min_period": 4.0} if winding_fine else None
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    run_labels(load_config(path))
    return path, os.path.join(root, "out", "labels", "fine.zarr")


@pytest.mark.parametrize("workers", [1, 3])
def test_add_winding_fine_matches_a_full_rebuild(tmp_path, workers):
    """Appending the block to an existing store must reproduce `tsm labels` byte for byte."""
    import json
    import os
    import sys

    import zarr

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))
    add_winding_fine = pytest.importorskip("add_winding_fine")

    ref_cfg, ref_store = _labels_case(str(tmp_path / "ref"), True)
    # the realistic flow: a store built by an older config, then the config gains the option
    cfg_path, store = _labels_case(str(tmp_path / "add"), False)
    with open(cfg_path) as fh:
        c = json.load(fh)
    c["extra"]["labels"]["winding_fine"] = {"min_period": 4.0}
    with open(cfg_path, "w") as fh:
        json.dump(c, fh)
    before = zarr.open_array(store=store, mode="r")
    assert "wf_sin" not in list(before.attrs["channels"])
    old = {c: np.asarray(before[i]) for i, c in enumerate(before.attrs["channels"])}
    del before

    rc = add_winding_fine.main([cfg_path, "--stale-minutes", "0", "--workers", str(workers)])
    assert rc == 0
    new = zarr.open_array(store=store, mode="r")
    channels = list(new.attrs["channels"])
    ref = zarr.open_array(store=ref_store, mode="r")
    assert channels == list(ref.attrs["channels"])
    got = {c: np.asarray(new[i]) for i, c in enumerate(channels)}
    want = {c: np.asarray(ref[i]) for i, c in enumerate(ref.attrs["channels"])}
    for c in old:  # every pre-existing channel is copied through untouched
        np.testing.assert_array_equal(got[c], old[c], err_msg=c)
    # The wf block is byte-identical to the inline build everywhere except a thin brick
    # fringe: `tsm labels` computes it from the brick's own haloed faces, the append reads the
    # halo back from the finished store, and only there do the two see different data (see the
    # script's docstring).  Assert exactly that -- identical away from the brick boundaries.
    shape = got["wf_valid"].shape
    brick = (16, 32, 32)  # tests.test_labels_workers._config
    zz, yy, xx = np.mgrid[: shape[0], : shape[1], : shape[2]]
    edge = np.minimum.reduce([np.minimum(a % b, b - 1 - a % b) for a, b in zip((zz, yy, xx), brick)])
    interior = edge >= 4  # < halo (8); the measured fringe is 3 voxels deep
    for c in WF_CHANNELS:
        np.testing.assert_array_equal(got[c][interior], want[c][interior], err_msg=f"{c} (interior)")
        assert float((got[c] != want[c]).mean()) < 0.02, (c, float((got[c] != want[c]).mean()))
    vg, vw = got["wf_valid"] == 1, want["wf_valid"] == 1
    assert abs(vg.mean() - vw.mean()) < 0.005, (vg.mean(), vw.mean())
    assert os.path.exists(store + ".prev")
    # a second run is a no-op
    with pytest.raises(SystemExit, match="already carries"):
        add_winding_fine.main([cfg_path, "--stale-minutes", "0"])


# --------------------------------------------------------------------------- #
# dev/eval_region.py: the winding_fine section
# --------------------------------------------------------------------------- #
def test_eval_region_reports_winding_fine(tmp_path):
    """With wf_* in the fine store the report gains a fine-referenced winding block."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))
    eval_region = pytest.importorskip("eval_region")
    from tests.test_eval_region import _make_faces_case

    case = _make_faces_case(str(tmp_path / "out"), shift=0, winding_fine=True)
    m = eval_region.main([case["cfg"], "--no-gallery", "--brick", "32,32,32"])
    assert "winding_fine" in m, sorted(m)
    wf = m["winding_fine"]
    assert wf["n_wf_valid"] > 0
    for k in ("phase_err_deg_median", "normal_err_deg_median", "density_mae"):
        assert wf[k] is not None, k
    # the section is a *different* reference from the coarse one
    assert wf["phase_err_deg_median"] != m["winding"].get("phase_err_deg_median")
    assert "winding_fine" in eval_region.to_markdown(m)
