"""Robust side extraction for ``surface_mode="sides"`` (:func:`tsm.infer.extract_sides`).

A trained student's unsigned face distance is *compressed*: it saturates one or two voxels
above zero and never reaches the faces, so the old pointwise rule ``d_face <= 1`` selects
nothing at all while the body head is clean.  These tests use exactly that failure mode --
``d = 0.2 * d_true + 1.5`` -- on a two-sheet volume whose sheets touch along a contact plane,
and check that the body skin plus the in-body distance valleys recover both outer faces AND
the contact plane, that the store round trip and the offline re-derivation agree, and that the
old rule recovers nothing.

All CPU, all synthetic: no network, no GPU.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import zarr
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))

from synth import make_synthetic  # noqa: E402
from tsm.data import CLIP, body_mask_np, decode_prob, decode_sdf, encode_prob, encode_sdf, encode_signed, face_dist  # noqa: E402
from tsm.infer import SIDES_PRED_CHANNELS, decode_pred, extract_sides, rederive_sides, write_surface_channel  # noqa: E402
from tsm.volume import BrickWriter  # noqa: E402

SHAPE = (32, 16, 16)
#: two 4-voxel sheets that TOUCH: the body is one slab z in [10, 18), and the four faces are
#: the two outer skins (z = 10, 17) plus the contact plane in the middle (z = 13, 14).
BODY_Z = (10, 18)
FACE_Z = (10, 13, 14, 17)
COMPRESS = (0.2, 1.5)   # d_pred = 0.2 * d_true + 1.5: never reaches 0, saturates at ~1.5-4


def _fields(shape=SHAPE):
    """(compressed d_face, body prob, valid prob, true face mask) float arrays."""
    z = np.arange(shape[0], dtype=np.float32)[:, None, None] * np.ones(shape, np.float32)
    d_true = np.min(np.stack([np.abs(z - f) for f in FACE_Z]), axis=0)
    d = COMPRESS[0] * d_true + COMPRESS[1]
    body = ((z >= BODY_Z[0]) & (z < BODY_Z[1])).astype(np.float32)
    valid = np.ones(shape, np.float32)
    faces = np.zeros(shape, bool)
    for f in FACE_Z:
        faces[f] = True
    return d.astype(np.float32), body, valid, faces


def _dice_at(a: np.ndarray, b: np.ndarray, tol: int = 1) -> float:
    """Tolerant Dice, the ``dev/eval_region.py`` convention (dilate the other side by ``tol``)."""
    st = ndi.generate_binary_structure(3, 3)
    hit = int((a & ndi.binary_dilation(b, st, iterations=tol)).sum()) \
        + int((b & ndi.binary_dilation(a, st, iterations=tol)).sum())
    n = int(a.sum()) + int(b.sum())
    return (hit / n) if n else 0.0


# --------------------------------------------------------------------------- #
# the pure function
# --------------------------------------------------------------------------- #
def test_compressed_distance_breaks_the_old_rule_and_extract_sides_recovers_the_faces():
    d, body, valid, faces = _fields()
    assert float(d.min()) > 1.0                        # never reaches 0 at the faces
    old = (d <= 1.0) & (valid >= 0.5)
    assert int(old.sum()) == 0                         # the old `d_face <= 1` finds NOTHING

    m = extract_sides(d, body, valid)
    assert int(m.sum()) > 0
    assert _dice_at(m, faces, 1) > 0.9
    # every face plane is hit, the contact plane included
    for f in FACE_Z:
        assert m[f].all()


def test_the_skin_is_taken_on_the_body_side():
    d, body, valid, _ = _fields()
    m = extract_sides(d, body, valid)
    # no air voxel is ever a side: the skin lies on the papyrus, like `extract_surface` keeps
    # the non-positive (inside) side of the zero crossing
    assert not m[(body < 0.5)].any()
    assert not m[BODY_Z[0] - 1].any() and not m[BODY_Z[1]].any()
    assert m[BODY_Z[0]].all() and m[BODY_Z[1] - 1].all()


def test_the_valley_term_alone_finds_the_contact_plane():
    d, body, valid, _ = _fields()
    # valley_max below every predicted distance disables the valley term -> the skin alone
    skin = extract_sides(d, body, valid, valley_max=-1.0)
    assert set(np.flatnonzero(skin.any(axis=(1, 2)))) == {10, 17}   # outer faces only
    assert not skin[13].any() and not skin[14].any()                # blind to the contact plane

    m = extract_sides(d, body, valid)
    valley_only = m & ~skin
    assert valley_only[13].all() and valley_only[14].all()          # the valley finds it


def test_valleys_in_air_are_ignored():
    # one sheet only: the air voxels next to the face sit in the distance valley of that face,
    # but they are not body, so they are not sides
    z = np.arange(SHAPE[0], dtype=np.float32)[:, None, None] * np.ones(SHAPE, np.float32)
    d_true = np.minimum(np.abs(z - 10), np.abs(z - 13))
    d = (COMPRESS[0] * d_true + COMPRESS[1]).astype(np.float32)
    body = ((z >= 10) & (z <= 13)).astype(np.float32)
    m = extract_sides(d, body, np.ones(SHAPE, np.float32))
    assert not m[9].any() and not m[14].any()
    assert m[10].all() and m[13].all()


def test_valid_gates_the_sides():
    d, body, valid, _ = _fields()
    valid = valid.copy()
    valid[:, :8] = 0.0
    m = extract_sides(d, body, valid)
    assert not m[:, :8].any()
    assert m[:, 8:].any()
    assert np.array_equal(m[:, 8:], extract_sides(d, body, np.ones_like(valid))[:, 8:])


def test_min_component_drops_specks():
    d, body, valid, _ = _fields()
    body = body.copy()
    body[2, 2, 2] = 1.0                                  # an isolated one-voxel body speck
    m = extract_sides(d, body, valid)
    assert m[2, 2, 2]
    assert not extract_sides(d, body, valid, min_component=8)[2, 2, 2]
    assert extract_sides(d, body, valid, min_component=8)[FACE_Z[0]].all()


def test_body_thr_moves_the_boundary():
    d, _, valid, _ = _fields()
    z = np.arange(SHAPE[0], dtype=np.float32)[:, None, None] * np.ones(SHAPE, np.float32)
    soft = np.where((z >= BODY_Z[0]) & (z < BODY_Z[1]), 0.9, 0.0).astype(np.float32)
    soft[BODY_Z[0] - 1] = 0.7                            # a half-believed extra layer
    assert not extract_sides(d, soft, valid, body_thr=0.95).any()
    assert extract_sides(d, soft, valid, body_thr=0.8)[BODY_Z[0]].all()
    assert extract_sides(d, soft, valid, body_thr=0.6)[BODY_Z[0] - 1].all()


# --------------------------------------------------------------------------- #
# the store round trip
# --------------------------------------------------------------------------- #
def _write_sides_store(path: str, d, body, valid, shape=SHAPE, chunk=8, origin=(0, 0, 0)):
    zeros = np.zeros(shape, np.uint8)
    fields = {"d_face": encode_sdf(d, CLIP), "body": encode_prob(body),
              "valid": encode_prob(valid), "surface_side1": zeros}
    w = BrickWriter(path, list(SIDES_PRED_CHANNELS), shape, chunk=chunk, origin_zyx=origin,
                    voxel_um=2.4, scale=1.0)
    for ci, name in enumerate(SIDES_PRED_CHANNELS):
        w.write(ci, origin[0], origin[1], origin[2],
                np.ascontiguousarray(fields.get(name, zeros)))
    return path


def test_write_surface_channel_round_trip(tmp_path):
    d, body, valid, faces = _fields()
    p = _write_sides_store(str(tmp_path / "pred.zarr"), d, body, valid)
    out = write_surface_channel(p, CLIP, brick=8, stats_box=(8, 8, 8), min_component=0)

    arr = zarr.open_array(store=p, mode="r")
    a = np.asarray(arr[:])
    got = a[SIDES_PRED_CHANNELS.index("surface_side1")] > 127
    dec = decode_pred(a, CLIP, SIDES_PRED_CHANNELS)
    # the brick pass (1-voxel halo) reproduces the whole-volume pure function exactly
    want = extract_sides(dec["d_face"], dec["body"], dec["valid"]) & dec["data"]
    np.testing.assert_array_equal(got, want)
    assert got.sum() == out["n_voxels"] > 0
    assert _dice_at(got, faces, 1) > 0.9
    # the old rule is still empty on the store bytes
    assert int(((dec["d_face"] <= 1.0) & dec["data"]).sum()) == 0
    # the parameters are recorded, in the summary and in the store attrs
    assert out["surface_side1_params"]["valley_tol"] == 0.5
    assert dict(arr.attrs["surface_side1_params"])["body_thr"] == 0.5


def test_write_surface_channel_honours_sides_params(tmp_path):
    d, body, valid, _ = _fields()
    p = _write_sides_store(str(tmp_path / "pred.zarr"), d, body, valid)
    write_surface_channel(p, CLIP, brick=8, stats_box=(8, 8, 8), min_component=0,
                          sides_params={"valley_max": -1.0})
    arr = zarr.open_array(store=p, mode="r")
    got = np.asarray(arr[SIDES_PRED_CHANNELS.index("surface_side1")]) > 127
    assert set(np.flatnonzero(got.any(axis=(1, 2)))) == {10, 17}      # skin only
    assert dict(arr.attrs["surface_side1_params"])["valley_max"] == -1.0


def test_rederive_sides_refreshes_an_existing_store(tmp_path):
    d, body, valid, _ = _fields()
    p = _write_sides_store(str(tmp_path / "pred.zarr"), d, body, valid)
    arr = zarr.open_array(store=p, mode="r")
    assert int(np.asarray(arr[SIDES_PRED_CHANNELS.index("surface_side1")]).sum()) == 0
    out = rederive_sides(p, CLIP, brick=8, stats_box=(8, 8, 8), min_component=0)
    got = np.asarray(zarr.open_array(store=p, mode="r")[SIDES_PRED_CHANNELS.index("surface_side1")]) > 127
    assert got.sum() == out["n_voxels"] > 0


def test_rederive_sides_rejects_a_non_sides_store(tmp_path):
    from tsm.infer import PRED_CHANNELS

    shape = (8, 8, 8)
    w = BrickWriter(str(tmp_path / "med.zarr"), list(PRED_CHANNELS), shape, chunk=8)
    for ci in range(len(PRED_CHANNELS)):
        w.write(ci, 0, 0, 0, np.zeros(shape, np.uint8))
    with pytest.raises(ValueError, match="not a sides store"):
        rederive_sides(str(tmp_path / "med.zarr"), CLIP)


def test_dev_rederive_sides_cli(tmp_path):
    import rederive_sides as cli

    d, body, valid, _ = _fields()
    p = _write_sides_store(str(tmp_path / "pred.zarr"), d, body, valid)
    out = cli.main([p, "--brick", "8", "--min-component", "0"])
    got = np.asarray(zarr.open_array(store=p, mode="r")[SIDES_PRED_CHANNELS.index("surface_side1")]) > 127
    assert got.sum() == out["n_voxels"] > 0


# --------------------------------------------------------------------------- #
# dev/eval_region.py --rederive-sides
# --------------------------------------------------------------------------- #
eval_region = pytest.importorskip("eval_region")

E2E_SHAPE = (64, 64, 64)
E2E_ORIGIN = (16, 32, 32)


def _make_compressed_sides_case(root: str) -> tuple[str, str]:
    """(config path, pred.zarr path): a sides run whose d_face is compressed and whose stored
    ``surface_side1`` is empty -- exactly the state ``--rederive-sides`` exists to repair."""
    s = make_synthetic(root, fine_shape=E2E_SHAPE, fine_origin=E2E_ORIGIN, faces=True)
    arr = zarr.open_array(store=s["fine"], mode="r")
    ch = list(arr.attrs["channels"])
    g = lambda n: np.asarray(arr[ch.index(n)])  # noqa: E731
    si, so = decode_sdf(g("sdf_in")), decode_sdf(g("sdf_out"))
    d = COMPRESS[0] * face_dist(si, so) + COMPRESS[1]
    bod = body_mask_np(si, so).astype(np.float32)
    zer = np.zeros(E2E_SHAPE, np.float32)
    fields = {
        "d_face": encode_sdf(d, CLIP), "body": encode_prob(bod),
        "valid": np.full(E2E_SHAPE, 255, np.uint8), "ink": g("ink"),
        "sin": encode_signed(zer), "cos": encode_signed(zer + 1),
        "density": np.full(E2E_SHAPE, 40, np.uint8),
        "nx": encode_signed(zer), "ny": encode_signed(zer), "nz": encode_signed(zer + 1),
        "conf": encode_prob(np.full(E2E_SHAPE, 0.7, np.float32)),
        "spare": np.zeros(E2E_SHAPE, np.uint8),
        "surface_side1": np.zeros(E2E_SHAPE, np.uint8),   # the old rule selected nothing
    }
    pred = os.path.join(root, "student", "pred.zarr")
    w = BrickWriter(pred, list(SIDES_PRED_CHANNELS), E2E_SHAPE, chunk=32, origin_zyx=E2E_ORIGIN,
                    voxel_um=2.4, scale=1.0)
    for ci, name in enumerate(SIDES_PRED_CHANNELS):
        w.write(ci, *E2E_ORIGIN, np.ascontiguousarray(fields[name]))
    tw = BrickWriter(os.path.join(root, "teachers", "ink.zarr"), ["ink"], E2E_SHAPE, chunk=32,
                     origin_zyx=E2E_ORIGIN, voxel_um=2.4, scale=1.0)
    tw.write(0, *E2E_ORIGIN, np.ascontiguousarray(g("ink")))
    cfg = {"volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
           "region": {"start_zyx": list(E2E_ORIGIN), "size_zyx": list(E2E_SHAPE)},
           "budget": {"ram_bytes": 2 << 30, "array_bytes": 1 << 30},
           "out_dir": root, "extra": {}}
    path = os.path.join(root, "eval.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return path, pred


def test_eval_region_rederive_sides(tmp_path):
    root = str(tmp_path / "out")
    cfg, pred = _make_compressed_sides_case(root)
    i_s = SIDES_PRED_CHANNELS.index("surface_side1")

    m = eval_region.main([cfg, "--no-gallery", "--brick", "32,32,32", "--axis", "none"])
    assert m["surface"]["student_surface"] == "d_face"
    assert int(np.asarray(zarr.open_array(store=pred, mode="r")[i_s]).sum()) == 0
    # the scored mask is the (empty) stored channel
    assert not eval_region.body_surface_mask(eval_region.open_store(pred), E2E_ORIGIN,
                                             [o + n for o, n in zip(E2E_ORIGIN, E2E_SHAPE)]).any()

    eval_region.main([cfg, "--no-gallery", "--brick", "32,32,32", "--axis", "none",
                      "--rederive-sides"])
    a = np.asarray(zarr.open_array(store=pred, mode="r")[i_s])
    assert int(a.sum()) > 0
    got = eval_region.body_surface_mask(eval_region.open_store(pred), E2E_ORIGIN,
                                        [o + n for o, n in zip(E2E_ORIGIN, E2E_SHAPE)])
    assert got.any() and np.array_equal(got, a > 127)
