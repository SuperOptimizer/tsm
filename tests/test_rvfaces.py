"""Face labels from the upstream recto/verso bands (``tsm.rvfaces``) + the merge and the
teacher-independent ``upstream_faces`` metric of ``dev/eval_region.py``.

The synthetic volume is three parallel bands stacked along +z (which is the outward/radial
direction here): a recto band, a *contact* band (class 3 -- the out face of the inner sheet and
the in face of the outer one at once) and a verso band, i.e. two sheets sharing one face.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import zarr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))

from tsm.labels import LABEL_DEFAULTS, decode_sdf_u8, encode_sdf_u8, fine_channels, _faces_opts
from tsm.rvfaces import (
    RV_CHANNELS,
    SOURCE_CT,
    SOURCE_NONE,
    SOURCE_RECTOVERSO,
    SubView,
    build_rv_face_labels,
    disagree_summary,
    local_band_normals,
    merge_face_labels,
    read_rv_box,
    thin_band,
)

CLIP = 20.0
Z, Y, X = 72, 32, 32
Z_IN, Z_MID, Z_OUT = 12, 32, 52   # medial planes of the three bands


def three_bands(band: int = 5, shape=(Z, Y, X)):
    """(rv_class, ct, radial) -- recto at z=10-14, contact at 30-34, verso at 50-54, outward +z."""
    rv = np.zeros(shape, np.uint8)
    h = band // 2
    rv[Z_IN - h:Z_IN + h + 1] = 1        # recto  -> IN face of sheet A
    rv[Z_MID - h:Z_MID + h + 1] = 3      # contact -> OUT face of A *and* IN face of B
    rv[Z_OUT - h:Z_OUT + h + 1] = 2      # verso  -> OUT face of sheet B
    ct = np.full(shape, 200, np.uint8)   # everything is data (the CT never names a face)
    radial = np.zeros((3,) + shape, np.float32)
    radial[0] = 1.0                      # outward = +z
    return rv, ct, radial


def _build(**kw):
    rv, ct, radial = three_bands()
    return rv, build_rv_face_labels(rv, ct, radial, CLIP, min_component=8, **kw)


# --------------------------------------------------------------------------- #
def test_thin_band_collapses_a_thick_band_to_one_voxel():
    rv, _, _ = three_bands(band=7)
    thin = thin_band((rv == 1) | (rv == 3), min_component=8)
    per_column = thin[:, 16, 16]
    assert per_column.sum() == 2                       # one voxel per band, not seven
    assert list(np.nonzero(per_column)[0]) == [Z_IN, Z_MID]
    assert thin.sum() < ((rv == 1) | (rv == 3)).sum() / 3


def test_sdf_signs_zero_sets_and_thickness():
    _, (si_u8, so_u8, valid, thick) = _build()
    si = decode_sdf_u8(si_u8, CLIP)[:, 16, 16]
    so = decode_sdf_u8(so_u8, CLIP)[:, 16, 16]
    # zero sets: in face on the recto and contact planes, out face on the contact and verso ones
    assert list(np.nonzero(si == 0)[0]) == [Z_IN, Z_MID]
    assert list(np.nonzero(so == 0)[0]) == [Z_MID, Z_OUT]
    # each SDF is positive on the OUTWARD (+z) side of its own face
    assert si[Z_IN + 3] > 0 and si[Z_IN - 3] < 0
    assert so[Z_OUT + 3] > 0 and so[Z_OUT - 3] < 0
    # inside a sheet: sdf_in > 0 > sdf_out, and the gap is the sheet thickness.  Both SDFs are
    # distances to the NEAREST face voxel (the store's convention, as in the CT builder), so the
    # sheet interior is resolved out to the point equidistant from its own and the next sheet's
    # face -- here the contact plane is shared, so the inner half of each sheet.
    for mid, lo, hi in ((Z_IN + 5, Z_IN, Z_MID),):
        assert si[mid] > 0 > so[mid]
        assert si[mid] - so[mid] == pytest.approx(hi - lo, abs=0.5)
        assert thick[mid, 16, 16] == hi - lo
    assert thick[Z_IN - 3, 16, 16] == 0                # outside the sheet: no thickness


def test_contact_band_is_both_faces():
    """A class-3 voxel is the out face of the inner sheet and the in face of the outer one, so it
    is a zero of *both* SDFs.  Consequence of the nearest-face rule (documented in the module):
    just outside a contact plane both SDFs share its sign, and the sheet interior is only resolved
    as far as the point equidistant from its own face and the next one's."""
    _, (si_u8, so_u8, _, _) = _build()
    assert np.all(decode_sdf_u8(si_u8, CLIP)[Z_MID] == 0)
    assert np.all(decode_sdf_u8(so_u8, CLIP)[Z_MID] == 0)
    si = decode_sdf_u8(si_u8, CLIP)[:, 16, 16]
    so = decode_sdf_u8(so_u8, CLIP)[:, 16, 16]
    assert si[Z_MID + 5] > 0 and so[Z_MID + 5] > 0


def test_faces_valid_needs_both_faces_within_clip():
    _, (si_u8, so_u8, valid, _) = _build()
    col = valid[:, 16, 16]
    assert set(np.unique(col)) <= {1, 2}
    sup = np.nonzero(col == 1)[0]
    # supervised = within CLIP of an in-face AND of an out-face voxel
    assert sup.min() == Z_IN and sup.max() == Z_OUT
    assert col[Z_IN - 1] == 2 and col[Z_OUT + 1] == 2
    assert 0.3 < float((valid == 1).mean()) < 0.7


def test_no_band_is_all_ignore():
    ct = np.full((Z, Y, X), 200, np.uint8)
    radial = np.zeros((3, Z, Y, X), np.float32)
    radial[0] = 1.0
    si, so, valid, thick = build_rv_face_labels(np.zeros((Z, Y, X), np.uint8), ct, radial, CLIP)
    assert np.all(valid == 2)
    assert np.all(thick == 0)
    assert np.all(si == encode_sdf_u8(np.float32(CLIP), CLIP))


def test_one_sided_band_is_all_ignore():
    """Only a recto band: the out face is unknown, so nothing is determined."""
    rv, ct, radial = three_bands()
    rv = np.where(rv == 1, np.uint8(1), np.uint8(0))
    _, _, valid, _ = build_rv_face_labels(rv, ct, radial, CLIP, min_component=8)
    assert np.all(valid == 2)


def test_no_data_outside_the_ct_mask():
    rv, ct, radial = three_bands()
    ct = ct.copy()
    ct[:, :8] = 0
    si, so, valid, thick = build_rv_face_labels(rv, ct, radial, CLIP, min_component=8)
    assert np.all(valid[:, :4] == 0) and np.all(si[:, :4] == 0) and np.all(thick[:, :4] == 0)


def test_local_band_normals_follow_a_tilted_sheet():
    """PCA of the band voxels recovers the sheet normal where the radial guess is 30 deg off."""
    shape = (48, 48, 48)
    zz, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    band = np.zeros(shape, bool)
    # plane z = 24 + 0.5*y  -> unit normal ~ (1, -0.5, 0)/|.|
    for x in range(shape[2]):
        zc = np.rint(24 + 0.5 * np.arange(shape[1])).astype(int)
        for y, z0 in enumerate(zc):
            band[max(z0 - 2, 0):z0 + 3, y, x] = True
    radial = np.zeros((3,) + shape, np.float32)
    radial[0] = 1.0
    pts = np.nonzero(thin_band(band, 8))
    n, ok = local_band_normals(band, radial, pts, window=9)
    truth = np.array([1.0, -0.5, 0.0], np.float32)
    truth /= np.linalg.norm(truth)
    inner = np.all([(p > 8) & (p < s - 8) for p, s in zip(pts, shape)], axis=0)
    assert ok[inner].mean() > 0.9
    cos = (n[:, inner] * truth[:, None]).sum(0)
    assert float(np.median(cos)) > 0.98                       # much better than radial (0.894)
    assert float(np.median((n[:, inner] * radial[:, pts[0], pts[1], pts[2]][:, inner]).sum(0))) > 0.8
    del zz, yy


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def _tuple(sdf_in, sdf_out, valid, thick):
    return (encode_sdf_u8(sdf_in, CLIP), encode_sdf_u8(sdf_out, CLIP),
            valid.astype(np.uint8), thick.astype(np.uint8))


def test_merge_precedence_and_faces_source():
    shape = (4, 4, 4)
    ones = np.ones(shape, np.float32)
    rv_valid = np.full(shape, 2, np.uint8)
    rv_valid[0] = 1                       # rectoverso supervises the first z plane
    ct_valid = np.full(shape, 2, np.uint8)
    ct_valid[:2] = 1                      # the CT builder the first two
    ct_valid[3] = 0                       # ... and reports no data on the last
    rv_valid[3] = 0
    rv = _tuple(3 * ones, -3 * ones, rv_valid, 6 * ones)
    ct = _tuple(7 * ones, -7 * ones, ct_valid, 14 * ones)
    (mi, mo, mv, mt), src = merge_face_labels(rv, ct, CLIP)
    assert np.all(src[0] == SOURCE_RECTOVERSO) and np.all(mv[0] == 1)
    assert np.all(src[1] == SOURCE_CT) and np.all(mv[1] == 1)      # rectoverso ignored -> CT fills in
    assert np.all(src[2] == SOURCE_NONE) and np.all(mv[2] == 2)    # neither -> ignore
    assert np.all(src[3] == SOURCE_NONE) and np.all(mv[3] == 0)    # both no data -> no data
    assert decode_sdf_u8(mi[0], CLIP) == pytest.approx(3.0, abs=0.2)
    assert decode_sdf_u8(mi[1], CLIP) == pytest.approx(7.0, abs=0.2)
    assert np.all(mt[0] == 6) and np.all(mt[1] == 14)
    assert np.all(mi[3] == 0) and np.all(mo[3] == 0) and np.all(mt[3] == 0)


def test_merge_disagreement_stats():
    shape = (4, 4, 4)
    ones = np.ones(shape, np.float32)
    both = np.ones(shape, np.uint8)
    rv = _tuple(3 * ones, -3 * ones, both, 6 * ones)
    ct = _tuple(7 * ones, -1 * ones, both, 8 * ones)
    stats: dict = {}
    band = np.zeros(shape, bool)
    band[0] = True
    merge_face_labels(rv, ct, CLIP, band=band, stats=stats)
    s = disagree_summary(stats)
    assert s["both_valid_voxels"] == 64
    assert s["source_fraction"]["rectoverso"] == 1.0
    assert s["abs_delta_sdf_in_all"]["median"] == pytest.approx(4.0, abs=0.2)
    assert s["abs_delta_sdf_out_all"]["median"] == pytest.approx(2.0, abs=0.2)
    assert s["abs_delta_sdf_in_band"]["n"] == 16
    # per-source thickness: the check that the CT builder is not labelling welded sheet pairs
    th = s["thickness_voxels_by_source"]
    assert th["rv"]["median"] == pytest.approx(6.0, abs=0.5)
    assert th["ct"]["median"] == pytest.approx(8.0, abs=0.5)
    assert th["rv_both"]["n"] == 64 and th["ct_both"]["n"] == 64


def test_rv_channels_and_config_validation():
    assert fine_channels(True, False, True)[-3:] == RV_CHANNELS
    assert fine_channels(True, True, True)[-3:] == RV_CHANNELS
    assert fine_channels(True, True, False)[-1] == "fiber_valid"
    assert _faces_opts({"enabled": True})["source"] == "ct"
    assert LABEL_DEFAULTS["faces"] is None
    with pytest.raises(ValueError, match="faces.source"):
        _faces_opts({"enabled": True, "source": "upstream", "rectoverso_store": "/x"})
    with pytest.raises(ValueError, match="rectoverso_store"):
        _faces_opts({"enabled": True, "source": "rectoverso"})


# --------------------------------------------------------------------------- #
# store helpers
# --------------------------------------------------------------------------- #
def test_read_rv_box_zero_pads_outside_the_store(tmp_path):
    path = str(tmp_path / "rv.zarr")
    data = np.zeros((2, 8, 8, 8), np.uint8)
    data[0, 4] = 1
    zarr.create_array(store=path, shape=data.shape, chunks=(1, 4, 4, 4), dtype=np.uint8,
                      attributes={"origin_zyx": [100, 200, 300], "channels": ["rectoverso", "hzvt"]},
                      overwrite=True)[:] = data
    arr = zarr.open_array(store=path, mode="r")
    blk = read_rv_box(arr, (100, 200, 300), (98, 198, 298), (106, 206, 306))
    assert blk.shape == (2, 8, 8, 8)
    assert np.all(blk[0, :2] == 0)                                 # below the store -> 0
    assert np.all(blk[0, 6, 2:, 2:] == 1)                          # global z 100+4 -> local 6
    assert np.all(blk[:, :, :2] == 0) and np.all(blk[:, :, :, :2] == 0)


def test_subview_reads_the_sub_box(tmp_path):
    a = np.arange(2 * 8 * 8 * 8, dtype=np.uint8).reshape(2, 8, 8, 8)
    v = SubView(a, (2, 1, 0), (4, 4, 8))
    assert v.shape == (2, 4, 4, 8)
    assert np.array_equal(np.asarray(v[:, 0:2, 0:2, 0:8]), a[:, 2:4, 1:3, 0:8])


# --------------------------------------------------------------------------- #
# the teacher-independent eval metric
# --------------------------------------------------------------------------- #
def _store(path, channels, arrs, origin):
    from tsm.volume import BrickWriter

    w = BrickWriter(path, list(channels), arrs[0].shape, chunk=32, origin_zyx=origin,
                    voxel_um=2.4, scale=1.0)
    for ci, a in enumerate(arrs):
        w.write(ci, origin[0], origin[1], origin[2], np.ascontiguousarray(a))
    return path


@pytest.mark.parametrize("shift", [0, 6])
def test_upstream_faces_pass(tmp_path, shift):
    """A student whose faces sit exactly on the thinned bands scores ~perfectly; a shifted one
    does not.  Nothing in this metric comes from a teacher."""
    eval_region = pytest.importorskip("eval_region")
    origin = (0, 0, 0)
    rv, _, _ = three_bands()
    rv_path = _store(str(tmp_path / "rv.zarr"), ["rectoverso", "hzvt"],
                     [rv, np.zeros_like(rv)], origin)
    s_in = np.zeros((Z, Y, X), np.uint8)
    s_out = np.zeros((Z, Y, X), np.uint8)
    s_in[[Z_IN + shift, Z_MID + shift]] = 255
    s_out[[Z_MID + shift, min(Z_OUT + shift, Z - 1)]] = 255
    ch = ["sdf_in", "sdf_out", "surface_in1", "surface_out1"]
    pred_path = _store(str(tmp_path / "pred.zarr"), ch,
                       [np.full((Z, Y, X), 128, np.uint8), np.full((Z, Y, X), 128, np.uint8),
                        s_in, s_out], origin)
    from tsm.config import parse_config

    budget = parse_config({"volume": {"url": "x"}, "region": {"start_zyx": [0, 0, 0],
                                                              "size_zyx": [Z, Y, X]},
                           "out_dir": str(tmp_path)}).budget
    m = eval_region.upstream_faces_pass(
        eval_region.open_store(pred_path), eval_region.open_store(rv_path),
        [0, 0, 0], [Z, Y, X], budget, brick=(32, 32, 32), min_component=8)
    assert m["teacher_independent"] is True
    assert m["n_near_band"] > 0
    for face in ("in", "out"):
        r = m[face]
        assert r["n_band"] > 0 and r["n_pred"] > 0
        if shift == 0:
            assert r["dice_within2"] == pytest.approx(1.0)
            assert r["pred_to_band_median"] == pytest.approx(0.0)
            assert r["band_to_pred_median"] == pytest.approx(0.0)
        else:
            assert r["dice_within2"] < 0.2
            assert r["pred_to_band_median"] >= 4.0


def test_merge_source_retunes_the_ct_fallback_defaults():
    """A non-CT face source switches the CT builder to its fallback tuning (and narrows the
    recto ignore band) unless the config says otherwise."""
    from tsm.config import parse_config
    from tsm.labels import MERGE_CT_DEFAULTS, MERGE_IGNORE_BAND, _label_opts

    def opts(labels):
        return _label_opts(parse_config({"volume": {"url": "x"},
                                         "region": {"start_zyx": [0, 0, 0], "size_zyx": [8, 8, 8]},
                                         "out_dir": "/tmp/x", "extra": {"labels": labels}}))

    ct = opts({"faces": {"enabled": True}})
    assert ct["faces"]["body_threshold"] == 60.0 and ct["faces"]["side_reach"] == 6.0
    assert ct["ignore_band"] == [0.2, 0.8]
    mg = opts({"faces": {"enabled": True, "source": "merge", "rectoverso_store": "/x"}})
    for k, v in MERGE_CT_DEFAULTS.items():
        assert mg["faces"][k] == v
    assert mg["ignore_band"] == MERGE_IGNORE_BAND
    # explicit config values still win
    ex = opts({"ignore_band": [0.1, 0.9],
               "faces": {"enabled": True, "source": "merge", "rectoverso_store": "/x",
                         "side_reach": 6.0}})
    assert ex["faces"]["side_reach"] == 6.0 and ex["ignore_band"] == [0.1, 0.9]


def test_ambiguity_ignore_is_capped_at_side_reach():
    """The ambiguous-march ignore reaches at most `side_reach` voxels from an ambiguous boundary
    voxel, not `clip` (it used to be `clip`: 20-voxel ignore teardrops around every one of them,
    99.96 % of all ignores on the crumpled crop)."""
    from scipy import ndimage as ndi

    from tsm.labels import _local_faces, _normal_field, _shift_any, build_face_labels

    shape = (80, 24, 24)
    reach = 8.0
    ct = np.full(shape, 40, np.uint8)
    ct[10:30] = 200                      # two sheets with a 6-voxel gap: every gap-facing
    ct[36:56] = 200                      # boundary voxel sees body on both sides within `reach`
    radial = np.zeros((3,) + shape, np.float32)
    radial[0] = 1.0
    _, _, valid, _ = build_face_labels(np.zeros(shape, np.uint8), ct, None, radial, CLIP,
                                       body_threshold=120.0, close_radius=0.0, min_component=8,
                                       min_thickness=5.0, side_reach=reach)
    body = ndi.gaussian_filter(ct.astype(np.float32), 2.0, mode="nearest") > 120.0
    n_field, _, _ = _normal_field(None, radial, 0.5)
    _, _, amb = _local_faces(body & _shift_any(~body, False), body, n_field, int(reach))
    assert amb.any()
    d = ndi.distance_transform_edt(~amb)
    assert (valid == 2).any()
    assert d[valid == 2].max() <= reach + 1e-6
    assert d[valid == 2].max() > CLIP / 4          # ... and it does reach out that far
