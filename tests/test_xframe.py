"""Cross-frame transform + label resampling (tsm.xframe)."""

import json

import numpy as np
import pytest
import zarr

from tsm import xframe


def _write_transform(tmp_path, m34, fixed="fixed-volume"):
    path = tmp_path / "transform.json"
    path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "fixed_volume": fixed,
        "transformation_matrix": [list(map(float, r)) for r in m34],
        "fixed_landmarks": [[1.0, 2.0, 3.0]],
        "moving_landmarks": [[4.0, 5.0, 6.0]],
    }))
    return path


def _scale_translate_xyz(scale, tx, ty, tz):
    return [[scale, 0.0, 0.0, tx], [0.0, scale, 0.0, ty], [0.0, 0.0, scale, tz]]


def _rot_z_xyz(deg, scale, t):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return [[scale * c, -scale * s, 0.0, t[0]],
            [scale * s, scale * c, 0.0, t[1]],
            [0.0, 0.0, scale, t[2]]]


def test_read_transform_shape_and_checksum(tmp_path):
    p = _write_transform(tmp_path, _scale_translate_xyz(0.3, 10.0, -5.0, 2.0))
    doc = xframe.read_transform(str(p))
    assert doc.matrix_xyz.shape == (4, 4)
    assert doc.fixed_volume == "fixed-volume"
    np.testing.assert_allclose(doc.matrix_xyz[3], [0, 0, 0, 1])
    assert doc.checksum == xframe.matrix_checksum(doc.matrix_xyz)
    assert len(doc.checksum) == 16


def test_read_transform_rejects_bad_schema(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"schema_version": "2.0.0", "fixed_volume": "x",
                             "transformation_matrix": _scale_translate_xyz(1, 0, 0, 0)}))
    with pytest.raises(ValueError, match="schema_version"):
        xframe.read_transform(str(p))


def test_read_transform_rejects_bad_matrix_shape(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"schema_version": "1.0.0", "fixed_volume": "x",
                             "transformation_matrix": [[1, 0, 0, 0], [0, 1, 0, 0]]}))
    with pytest.raises(ValueError, match="3x4"):
        xframe.read_transform(str(p))


def test_swap_is_self_inverse():
    rng = np.random.default_rng(0)
    m = np.eye(4)
    m[:3] = rng.normal(size=(3, 4))
    np.testing.assert_allclose(xframe.swap_xyz_zyx(xframe.swap_xyz_zyx(m)), m, atol=1e-12)


@pytest.mark.parametrize("m34", [
    _scale_translate_xyz(0.30657, 11024.0, 2861.8, -9057.0),
    _rot_z_xyz(39.0, 0.307, (11024.0, 2861.8, -9057.0)),
])
def test_zyx_matrices_match_xyz_application(m34):
    m = np.vstack([np.asarray(m34, float), [0, 0, 0, 1.0]])
    fwd = xframe.image_to_label_matrix_zyx(m)
    inv = xframe.label_to_image_matrix_zyx(m)

    rng = np.random.default_rng(1)
    pts_zyx = rng.uniform(0, 20000, size=(50, 3))
    # reference: apply the XYZ matrix to XYZ-ordered points
    pts_xyz = pts_zyx[:, ::-1]
    ref_xyz = pts_xyz @ m[:3, :3].T + m[:3, 3]
    got = xframe.apply_affine_zyx(fwd, pts_zyx)
    np.testing.assert_allclose(got, ref_xyz[:, ::-1], atol=1e-8)

    # round trip
    back = xframe.apply_affine_zyx(inv, got)
    np.testing.assert_allclose(back, pts_zyx, atol=1e-6)


def test_known_point_round_trip_pure_scale():
    # label = 0.5 * image + (100, 200, 300) in XYZ
    m = np.vstack([np.asarray(_scale_translate_xyz(0.5, 100.0, 200.0, 300.0), float),
                   [0, 0, 0, 1.0]])
    fwd = xframe.image_to_label_matrix_zyx(m)
    # image ZYX (40, 20, 10) -> XYZ (10, 20, 40) -> label XYZ (105, 210, 320) -> ZYX
    np.testing.assert_allclose(xframe.apply_affine_zyx(fwd, [[40.0, 20.0, 10.0]]),
                               [[320.0, 210.0, 105.0]])
    inv = xframe.label_to_image_matrix_zyx(m)
    np.testing.assert_allclose(xframe.apply_affine_zyx(inv, [[320.0, 210.0, 105.0]]),
                               [[40.0, 20.0, 10.0]], atol=1e-9)


def test_level_scale_matrix_pixel_centres():
    m = xframe.level_scale_matrix(1)
    # level-0 voxels 0 and 1 both fall in level-1 voxel 0 (centres -0.25, +0.25)
    got = xframe.apply_affine_zyx(m, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    np.testing.assert_allclose(got[0], [-0.25] * 3)
    np.testing.assert_allclose(got[1], [0.25] * 3)
    np.testing.assert_allclose(got[2], [0.75] * 3)
    np.testing.assert_allclose(xframe.level_scale_matrix(0), np.eye(4))


def test_label_aabb_covers_mapped_corners():
    m = np.vstack([np.asarray(_rot_z_xyz(39.0, 0.31, (500.0, 300.0, 20.0)), float),
                   [0, 0, 0, 1.0]])
    fwd = xframe.image_to_label_matrix_zyx(m)
    start, size = (10, 200, 300), (16, 64, 48)
    lo, hi = xframe.label_aabb_for_image_box(fwd, start, size, margin=1)

    zz, yy, xx = np.meshgrid(*[np.arange(s, s + d) for s, d in zip(start, size)], indexing="ij")
    pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], 1).astype(float)
    mapped = np.rint(xframe.apply_affine_zyx(fwd, pts)).astype(int)
    assert (mapped >= np.asarray(lo)).all()
    assert (mapped < np.asarray(hi)).all()


def test_label_aabb_clips_to_shape():
    m = np.eye(4)
    lo, hi = xframe.label_aabb_for_image_box(m, (0, 0, 0), (10, 10, 10),
                                             label_shape_zyx=(5, 5, 5), margin=1)
    assert lo == (0, 0, 0) and hi == (5, 5, 5)


def _synth_label_zarr(tmp_path, shape=(48, 40, 44), seed=3):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4, size=shape, dtype=np.uint8)
    root = zarr.open_group(store=str(tmp_path / "labels.zarr"), mode="w")
    root.create_array("0", shape=shape, chunks=(16, 16, 16), dtype=np.uint8)[:] = data
    half = data[::2, ::2, ::2]
    root.create_array("1", shape=half.shape, chunks=(16, 16, 16), dtype=np.uint8)[:] = half
    return str(tmp_path / "labels.zarr"), data, half


def test_resample_identity_returns_the_labels(tmp_path):
    url, data, _ = _synth_label_zarr(tmp_path)
    out = xframe.resample_labels_to_image_box(url, 0, np.eye(4), (4, 5, 6), (8, 9, 10))
    np.testing.assert_array_equal(out, data[4:12, 5:14, 6:16])
    assert out.dtype == np.uint8


def test_resample_matches_brute_force_nearest(tmp_path):
    url, data, _ = _synth_label_zarr(tmp_path)
    # label = 0.5*image + shift, plus a small rotation about z (XYZ frame)
    m = np.vstack([np.asarray(_rot_z_xyz(7.0, 0.5, (6.0, 5.0, 4.0)), float), [0, 0, 0, 1.0]])
    fwd = xframe.image_to_label_matrix_zyx(m)
    start, size = (3, 7, 11), (12, 13, 14)
    out = xframe.resample_labels_to_image_box(url, 0, fwd, start, size, sub_box=(5, 5, 5))

    zz, yy, xx = np.meshgrid(*[np.arange(s, s + d) for s, d in zip(start, size)], indexing="ij")
    pts = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], 1).astype(float)
    idx = np.rint(xframe.apply_affine_zyx(fwd, pts)).astype(int)
    ok = ((idx >= 0) & (idx < np.asarray(data.shape))).all(1)
    ref = np.zeros(idx.shape[0], np.uint8)
    ref[ok] = data[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    np.testing.assert_array_equal(out, ref.reshape(size))


def test_resample_sub_box_and_split_agree(tmp_path):
    url, _, _ = _synth_label_zarr(tmp_path)
    m = np.vstack([np.asarray(_rot_z_xyz(11.0, 0.7, (2.0, 3.0, 1.0)), float), [0, 0, 0, 1.0]])
    fwd = xframe.image_to_label_matrix_zyx(m)
    args = (url, 0, fwd, (2, 3, 4), (16, 17, 18))
    whole = xframe.resample_labels_to_image_box(*args, sub_box=(64, 64, 64))
    tiled = xframe.resample_labels_to_image_box(*args, sub_box=(5, 4, 7))
    split = xframe.resample_labels_to_image_box(*args, sub_box=(4, 4, 4), max_read_bytes=1)
    np.testing.assert_array_equal(whole, tiled)
    np.testing.assert_array_equal(whole, split)


def test_resample_level1_halves_coordinates(tmp_path):
    url, _, half = _synth_label_zarr(tmp_path)
    out = xframe.resample_labels_to_image_box(url, 1, np.eye(4), (0, 0, 0), (6, 6, 6))
    # level-0 index i -> level-1 (i+0.5)/2-0.5 -> rint
    idx = np.rint((np.arange(6) + 0.5) / 2 - 0.5).astype(int)
    np.testing.assert_array_equal(out, half[np.ix_(idx, idx, idx)])


def test_resample_outside_volume_is_fill(tmp_path):
    url, _, _ = _synth_label_zarr(tmp_path)
    out = xframe.resample_labels_to_image_box(url, 0, np.eye(4), (500, 500, 500), (4, 4, 4))
    assert out.shape == (4, 4, 4)
    assert not out.any()


def test_resample_partially_outside_volume(tmp_path):
    url, data, _ = _synth_label_zarr(tmp_path)
    out = xframe.resample_labels_to_image_box(url, 0, np.eye(4), (44, 36, 40), (8, 8, 8))
    np.testing.assert_array_equal(out[:4, :4, :4], data[44:48, 36:40, 40:44])
    assert not out[4:].any()


def test_resample_accepts_open_array(tmp_path):
    url, data, _ = _synth_label_zarr(tmp_path)
    arr = xframe.open_label_array(url, 0)
    out = xframe.resample_labels_to_image_box(arr, 0, np.eye(4), (0, 0, 0), (4, 4, 4))
    np.testing.assert_array_equal(out, data[:4, :4, :4])
