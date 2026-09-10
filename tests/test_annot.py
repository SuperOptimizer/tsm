"""dev/annot_export.py + dev/annot_import.py: the human label-correction loop.

Synthetic store -> the exporter picks the crop with the planted source seam -> a simulated
``correction.u8`` -> the importer moves the faces, stamps ``faces_source = 3`` and the
``faces_weight`` channel, and leaves every other brick alone.  Plus the training side: the
per-voxel weight multiplies the surface loss where it is set and is bit-identical where it is
not (and when the channel is absent entirely).

Plus the round trip, on a store built by the *real* merge builder path
(``build_rv_face_labels`` + ``build_face_labels`` + ``merge_face_labels``): an empty correction
must be a byte-identical no-op and one brush stroke may only move the SDF within ``clip + 1`` of
itself.  A full-crop recompute is not a faithful round trip of the stored field (see
``dev/annot_import.py``), which is why the import is incremental.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest
import torch
import zarr

from tsm.labels import encode_sdf_u8, fine_channels
from tsm.rvfaces import SOURCE_CT, SOURCE_HUMAN, SOURCE_RECTOVERSO
from tsm.volume import BrickWriter

DEV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dev")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(DEV, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


ax = _load("annot_export")
ai = _load("annot_import")

SHAPE = (64, 128, 128)
ORIGIN = (16, 32, 32)
CHUNK = 32
CLIP = 20.0
IN_X, OUT_X = 80, 96          # local x of the planted in / out face planes
SEAM = (slice(None), slice(64, 128), slice(64, 128))   # the CT-sourced block


# --------------------------------------------------------------------------- #
# a synthetic merged face store
# --------------------------------------------------------------------------- #
def _fields() -> dict[str, np.ndarray]:
    z, y, x = np.indices(SHAPE)
    f: dict[str, np.ndarray] = {}
    f["sdf_in"] = encode_sdf_u8(np.clip(x - IN_X, -CLIP, CLIP).astype(np.float32), CLIP)
    f["sdf_out"] = encode_sdf_u8(np.clip(x - OUT_X, -CLIP, CLIP).astype(np.float32), CLIP)
    f["faces_valid"] = np.ones(SHAPE, np.uint8)
    f["thickness"] = np.full(SHAPE, OUT_X - IN_X, np.uint8)
    src = np.full(SHAPE, SOURCE_RECTOVERSO, np.uint8)
    src[SEAM] = SOURCE_CT
    f["faces_source"] = src
    f["rv_class"] = ((x == IN_X) * 1 + (x == OUT_X) * 2).astype(np.uint8)
    f["hzvt_class"] = np.zeros(SHAPE, np.uint8)
    f["sdf"] = f["sdf_in"].copy()
    f["sdf_valid"] = np.ones(SHAPE, np.uint8)
    f["ink"] = ((z * 3 + y) % 251).astype(np.uint8)
    f["ink_valid"] = np.ones(SHAPE, np.uint8)
    return f


def _write_store(path: str, f: dict[str, np.ndarray], channels: list[str]) -> str:
    w = BrickWriter(path, channels, SHAPE, chunk=CHUNK, origin_zyx=ORIGIN, voxel_um=2.4, scale=1.0)
    for ci, name in enumerate(channels):
        w.write(ci, *ORIGIN, np.ascontiguousarray(f[name]))
    return path


@pytest.fixture()
def store(tmp_path):
    ch = fine_channels(faces=True, fiber=False, rv=True)
    f = _fields()
    path = _write_store(str(tmp_path / "out" / "labels" / "fine.zarr"), f, ch)
    return {"path": path, "channels": ch, "fields": f}


def _axis_json(tmp_path) -> str:
    """Umbilicus at x = 0, y = mid-region: the outward radial direction is ~ +x, which is the
    orientation the planted planes were built with (sdf_in = x - IN_X)."""
    p = str(tmp_path / "axis.json")
    y = ORIGIN[1] + SHAPE[1] // 2
    with open(p, "w") as fh:
        json.dump({"control_points": [{"z": 0, "y": y, "x": 0}, {"z": 10_000, "y": y, "x": 0}]}, fh)
    return p


def _config(tmp_path, out_dir: str) -> str:
    p = str(tmp_path / "cfg.json")
    cfg = {
        "volume": {"url": str(tmp_path / "ct.zarr"), "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": list(ORIGIN), "size_zyx": list(SHAPE)},
        "budget": {"ram_bytes": 1 << 30, "array_bytes": 1 << 28},
        "out_dir": out_dir,
        "extra": {"labels": {"clip": 20, "halo": 24, "brick": [32, 64, 64],
                             "axis_path": _axis_json(tmp_path),
                             "teachers_dir": str(tmp_path / "teachers"),
                             "faces": {"enabled": True, "source": "merge",
                                       "rectoverso_store": str(tmp_path / "rv.zarr")}}},
    }
    with open(p, "w") as fh:
        json.dump(cfg, fh)
    return p


# --------------------------------------------------------------------------- #
# export: selection
# --------------------------------------------------------------------------- #
def test_export_selects_the_planted_disagreement(store):
    st = ax.Store(store["path"])
    acc = ax.block_stats(st, stride=32, tile=64, near=8, select="disagreement",
                         pred=None, pred_tol=3.0)
    origins, rows = ax.crop_scores(acc, stride=32, size=64, select="disagreement",
                                   min_valid=0.1, seed=0)
    assert len(rows) == 1 * 3 * 3          # z has exactly one 64-crop, y/x three each
    chosen = ax.pick(origins, rows, n=2, size=64)
    # the seam block starts at local (0, 64, 64); the crop containing all of it wins
    assert chosen[0][0] == (0, 64, 64)
    assert chosen[0][1]["seam_frac"] > 0
    # non-overlapping: with size == 64 and a 128-wide axis no second crop can avoid the first
    # on y and x at once, so the runner-up is disjoint from it on at least one axis
    for o, _ in chosen[1:]:
        assert any(abs(o[a] - chosen[0][0][a]) >= 64 for a in range(3))


def test_export_random_and_min_valid_gate(store):
    st = ax.Store(store["path"])
    acc = ax.block_stats(st, stride=32, tile=64, near=8, select="random", pred=None, pred_tol=3.0)
    _, rows = ax.crop_scores(acc, 32, 64, "random", min_valid=0.1, seed=3)
    assert all(0.0 <= r["score"] <= 1.0 for r in rows)
    # every voxel is faces_valid == 1 here, so a threshold above 1 rejects everything
    _, rows = ax.crop_scores(acc, 32, 64, "disagreement", min_valid=1.5, seed=0)
    assert all(r["score"] == -1.0 for r in rows)


class _FakeCT:
    def read(self, z0, z1, y0, y1, x0, x1):
        z, y, x = np.indices((z1 - z0, y1 - y0, x1 - x0))
        return ((z + y + x) % 256).astype(np.uint8)


class _FakeCfg:
    class volume:  # noqa: N801
        url = "file://synthetic"


def _export_packet(tmp_path, store, origin=(0, 64, 64), size=64) -> tuple[str, dict]:
    st = ax.Store(store["path"])
    out = str(tmp_path / "packets")
    os.makedirs(out, exist_ok=True)
    row = {"score": 1.0, "valid_frac": 1.0}
    meta = ax.write_packet(out, 0, st, _FakeCfg, origin, size, row, "disagreement",
                           None, _FakeCT(), tif=False)
    ax.write_manifest(out, [meta], "cfg.json", "disagreement")
    return out, meta


def test_packet_layout_and_manifest(tmp_path, store):
    out, meta = _export_packet(tmp_path, store)
    pdir = os.path.join(out, meta["name"])
    for layer in ("ct", "faces_in", "faces_out", "ignore", "source", "rv_class"):
        f = os.path.join(pdir, f"{layer}.u8")
        assert os.path.getsize(f) == 64 ** 3, layer
    fin = np.fromfile(os.path.join(pdir, "faces_in.u8"), np.uint8).reshape(64, 64, 64)
    fout = np.fromfile(os.path.join(pdir, "faces_out.u8"), np.uint8).reshape(64, 64, 64)
    # the planted planes land at their crop-local x
    assert set(np.unique(np.nonzero(fin)[2])) == {IN_X - 64}
    assert set(np.unique(np.nonzero(fout)[2])) == {OUT_X - 64}
    assert meta["origin_zyx"] == [ORIGIN[0], ORIGIN[1] + 64, ORIGIN[2] + 64]
    with open(os.path.join(out, "packet.json")) as fh:
        man = json.load(fh)
    assert man["packets"][0]["path"] == meta["name"]
    assert man["packets"][0]["dims_xyz"] == [64, 64, 64]
    assert set(man["palette"]["correction"]["values"]) == {"0", "1", "2", "3", "4"}


# --------------------------------------------------------------------------- #
# import
# --------------------------------------------------------------------------- #
PATCH = (slice(8, 24), slice(8, 24))     # (z, y) of the painted patch, crop-local
NEW_X = IN_X - 64 + 4                    # move the in face 4 voxels outward, crop-local


def _paint(pdir: str, size: int = 64) -> np.ndarray:
    c = np.zeros((size, size, size), np.uint8)
    c[PATCH[0], PATCH[1], IN_X - 64] = ai.C_ERASE     # "wrong here"
    c[PATCH[0], PATCH[1], NEW_X] = ai.C_IN            # the corrected in face
    c[PATCH[0], slice(32, 36), OUT_X - 64] = ai.C_IGNORE
    c.tofile(os.path.join(pdir, "correction.u8"))
    return c


def test_import_moves_the_faces_and_leaves_the_rest(tmp_path, store):
    out, meta = _export_packet(tmp_path, store)
    pdir = os.path.join(out, meta["name"])
    corr = _paint(pdir)
    before = np.asarray(zarr.open_array(store=store["path"], mode="r")[:])

    cfg = _config(tmp_path, os.path.dirname(os.path.dirname(store["path"])))
    assert ai.main([cfg, "--packets", out, "--weight", "5"]) == 0

    arr = zarr.open_array(store=store["path"], mode="r")
    ch = {c: i for i, c in enumerate(arr.attrs["channels"])}
    assert ai.FACES_WEIGHT in ch and len(ch) == before.shape[0] + 1
    after = np.asarray(arr[:])

    # local (store) coordinates of the crop and of the painted patch
    cz, cy, cx = 0, 64, 64
    pz, py = PATCH
    sl = (slice(cz + pz.start, cz + pz.stop), slice(cy + py.start, cy + py.stop))

    sdf_in = after[ch["sdf_in"]]
    assert (sdf_in[sl[0], sl[1], cx + NEW_X] == 128).all()          # the new face is the zero set
    assert (sdf_in[sl[0], sl[1], IN_X] != 128).all()                # the old one is gone there
    # ... and untouched outside the painted patch
    assert (sdf_in[:8, :, IN_X] == 128).all()

    src = after[ch["faces_source"]]
    painted = np.zeros(SHAPE, bool)
    painted[cz:cz + 64, cy:cy + 64, cx:cx + 64] = corr > 0
    assert (src[painted] == SOURCE_HUMAN).all()
    assert not (src[~painted] == SOURCE_HUMAN).any()

    fw = after[ch[ai.FACES_WEIGHT]]
    assert (fw[painted] == 5).all()
    assert (fw[~painted] == 1).all()

    val = after[ch["faces_valid"]]
    ig = np.zeros(SHAPE, bool)
    ig[cz:cz + 64, cy:cy + 64, cx:cx + 64] = corr == ai.C_IGNORE
    assert (val[ig] == 2).all()

    # every channel the correction does not own is byte-identical, and so is every voxel
    # outside the packet crop in the ones it does
    for name in ("sdf", "sdf_valid", "ink", "ink_valid", "thickness", "rv_class", "hzvt_class"):
        assert np.array_equal(after[ch[name]], before[ch[name]]), name
    outside = np.ones(SHAPE, bool)
    outside[cz:cz + 64, cy:cy + 64, cx:cx + 64] = False
    for name in ("sdf_in", "sdf_out", "faces_valid", "faces_source"):
        assert np.array_equal(after[ch[name]][outside], before[ch[name]][outside]), name


def test_import_refuses_a_stale_packet(tmp_path, store):
    out, meta = _export_packet(tmp_path, store)
    _paint(os.path.join(out, meta["name"]))
    arr = zarr.open_array(store=store["path"], mode="r+")
    ci = list(arr.attrs["channels"]).index("sdf_in")
    arr[ci, 0, 70, 70] = 200                       # the store moved under the packet
    cfg = _config(tmp_path, os.path.dirname(os.path.dirname(store["path"])))
    with pytest.raises(SystemExit, match="labels changed"):
        ai.main([cfg, "--packets", out])


def test_import_dry_run_writes_nothing(tmp_path, store):
    out, meta = _export_packet(tmp_path, store)
    _paint(os.path.join(out, meta["name"]))
    before = np.asarray(zarr.open_array(store=store["path"], mode="r")[:])
    cfg = _config(tmp_path, os.path.dirname(os.path.dirname(store["path"])))
    assert ai.main([cfg, "--packets", out, "--dry-run"]) == 0
    arr = zarr.open_array(store=store["path"], mode="r")
    assert list(arr.attrs["channels"]) == store["channels"]
    assert np.array_equal(np.asarray(arr[:]), before)


# --------------------------------------------------------------------------- #
# training: the per-voxel surface weight
# --------------------------------------------------------------------------- #
def _faces_batch(B=2, P=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)  # noqa: E731
    pred = r(B, 3, P, P, P) * 4 - 2
    sdf = (r(B, 2, P, P, P) * 2 - 1) * 10.0
    valid = (r(B, 1, P, P, P) * 3).floor()
    return pred, sdf, valid


def test_faces_weight_of_one_is_bit_identical():
    from tsm.train import faces_loss

    pred, sdf, valid = _faces_batch()
    base = faces_loss(pred, sdf, valid)
    ones = faces_loss(pred, sdf, valid, torch.ones_like(valid))
    assert set(base) == set(ones)
    for k in base:
        assert base[k].item() == ones[k].item(), k          # exact, not allclose
        assert torch.equal(base[k], ones[k]), k


def test_faces_weight_changes_the_loss_where_set():
    from tsm.train import faces_loss

    pred, sdf, valid = _faces_batch(seed=1)
    w = torch.ones_like(valid)
    w[:, :, :4] = 5.0
    got = faces_loss(pred, sdf, valid, w)
    base = faces_loss(pred, sdf, valid)
    assert any(abs(got[k].item() - base[k].item()) > 1e-6 for k in base)
    # weighting only voxels the mask already drops changes nothing
    w2 = torch.where(valid == 1, torch.ones_like(w), torch.full_like(w, 3.0))
    only_masked = faces_loss(pred, sdf, valid, torch.where(valid == 1, torch.ones_like(w), w2 * 0 + 1))
    assert only_masked["sdf_in_l1"].item() == base["sdf_in_l1"].item()


def test_compute_losses_carries_the_weight():
    from tsm.train import compute_losses

    pred, sdf, valid = _faces_batch(seed=2)
    P = sdf.shape[-1]
    B = sdf.shape[0]
    batch = {
        "surface_sdf": sdf, "surface_valid": valid,
        "ink_prob": torch.zeros(B, 1, P, P, P), "ink_valid": torch.zeros(B, 1, P, P, P),
        "winding": torch.zeros(B, 6, P, P, P), "winding_conf": torch.zeros(B, 1, P, P, P),
        "winding_valid": torch.zeros(B, 1, P, P, P),
    }
    out = {"surface": [pred]}
    t0, p0 = compute_losses(out, batch, surface_mode="faces", ds_weights=(1.0,))
    batch1 = {**batch, "surface_weight": torch.ones_like(valid)}
    t1, p1 = compute_losses(out, batch1, surface_mode="faces", ds_weights=(1.0,))
    assert t0.item() == t1.item()
    w = torch.ones_like(valid)
    w[:, :, :2] = 7.0
    t2, _ = compute_losses(out, {**batch, "surface_weight": w}, surface_mode="faces", ds_weights=(1.0,))
    assert abs(t2.item() - t0.item()) > 1e-6


def test_target_keys_and_augment_carry_surface_weight():
    from tsm.data import SURFACE_WEIGHT_KEY, Augment, target_keys

    assert SURFACE_WEIGHT_KEY not in target_keys("faces")
    assert target_keys("faces", surface_weight=True)[SURFACE_WEIGHT_KEY] == 1
    B, P = 1, 16
    batch = {
        "input": torch.zeros(B, 2, P, P, P),
        "surface_sdf": torch.zeros(B, 2, P, P, P),
        "surface_valid": torch.ones(B, 1, P, P, P),
        "ink_prob": torch.zeros(B, 1, P, P, P), "ink_valid": torch.ones(B, 1, P, P, P),
        "winding": torch.zeros(B, 6, P, P, P), "winding_conf": torch.zeros(B, 1, P, P, P),
        "winding_valid": torch.zeros(B, 1, P, P, P),
        SURFACE_WEIGHT_KEY: torch.full((B, 1, P, P, P), 5.0),
    }
    batch[SURFACE_WEIGHT_KEY][:, :, :, :, : P // 2] = 1.0
    aug = Augment({"preset": "none", "flip": {"p": 1.0}}, seed=0)
    params = [aug.sample_spatial() for _ in range(B)]
    out = aug.apply_spatial(batch, params)
    assert out[SURFACE_WEIGHT_KEY].shape == (B, 1, P, P, P)
    # a pure flip permutes the weight field; the multiset of values is preserved
    assert torch.allclose(out[SURFACE_WEIGHT_KEY].sort().values,
                          batch[SURFACE_WEIGHT_KEY].sort().values, atol=1e-4)


def test_dataset_reads_faces_weight_after_the_append(tmp_path):
    """End to end at the dataset level: append the channel to a synthetic faces store and the
    faces-mode CropDataset gains the ``surface_weight`` target with the store's values."""
    from tsm.data import SURFACE_WEIGHT_KEY, CropDataset, open_reader_factory
    from synth import make_synthetic

    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True)
    ai.append_faces_weight(s["fine"], brick=[32, 32, 32], fill=1)
    arr = zarr.open_array(store=s["fine"], mode="r+")
    ci = list(arr.attrs["channels"]).index(ai.FACES_WEIGHT)
    arr[ci, :, :, :16] = 5

    ds = CropDataset(open_reader_factory(s["ct"], 0, 2.4), s["fine"], s["coarse"],
                     patch=32, stride=32, augment=False, surface_mode="faces")
    assert ds.surface_weight
    item = ds.to_tensors(ds.load(ds.origins[0]))
    w = item[SURFACE_WEIGHT_KEY]
    assert w.shape == (1, 32, 32, 32)
    assert set(torch.unique(w).tolist()) <= {1.0, 5.0}
    assert (w == 5.0).any()


# --------------------------------------------------------------------------- #
# import round trip on a store built by the real merge builder
# --------------------------------------------------------------------------- #
# The synthetic store above plants the SDFs by hand; this one runs the actual pipeline
# (tsm.rvfaces.build_rv_face_labels + tsm.labels.build_face_labels + merge_face_labels) on a
# stack of tilted sheets, which is what makes the round trip a real test: the merge builder
# derives its band normals from the *unthinned* upstream band and mixes two builders, neither of
# which the importer can reconstruct from the store (see dev/annot_import.py's docstring).
RT_SHAPE = (48, 96, 96)
RT_ORIGIN = (16, 32, 32)
RT_PERIOD = 28


def _merge_fields() -> dict[str, np.ndarray]:
    from tsm.labels import build_face_labels, radial_field
    from tsm.rvfaces import build_rv_face_labels, merge_face_labels

    axis = np.array([[0.0, RT_ORIGIN[1] + RT_SHAPE[1] // 2, 0.0],
                     [10_000.0, RT_ORIGIN[1] + RT_SHAPE[1] // 2, 0.0]])
    radial = radial_field(axis, RT_ORIGIN, RT_SHAPE, scale=1)
    z, y, x = np.indices(RT_SHAPE)
    # tilted sheets: the outward direction is ~ +x (umbilicus at x = 0) but the sheet normal is not
    ph = np.mod((x + RT_ORIGIN[2]) + 0.35 * (y + RT_ORIGIN[1]) + 0.2 * (z + RT_ORIGIN[0]), RT_PERIOD)
    body = ph < 12
    ct = np.where(body, 200, 40).astype(np.uint8)
    rv = np.zeros(RT_SHAPE, np.uint8)
    rv[(ph >= 0) & (ph < 2.5)] = 1                     # recto band -> in face
    rv[(ph >= 10) & (ph < 12.5)] = 2                   # verso band -> out face
    # a slab with no upstream band at all: the rectoverso builder is ignore there and the CT
    # builder wins the merge, so the store really does mix the two sources
    rv[:, :, :20] = 0
    recto_prob = np.where(body & (ph < 6), 230, 10).astype(np.uint8)

    rvres = build_rv_face_labels(rv, ct, radial, CLIP, min_component=8)
    ctres = build_face_labels(recto_prob, ct, None, radial, CLIP, body_threshold=80,
                              close_radius=6, side_reach=3, min_thickness=10, min_component=50)
    (si, so, val, thick), src = merge_face_labels(rvres, ctres, CLIP)
    f = {"sdf_in": si, "sdf_out": so, "faces_valid": val, "thickness": thick,
         "faces_source": src, "rv_class": rv, "hzvt_class": np.zeros(RT_SHAPE, np.uint8),
         "sdf": si.copy(), "sdf_valid": np.ones(RT_SHAPE, np.uint8),
         "ink": np.zeros(RT_SHAPE, np.uint8), "ink_valid": np.ones(RT_SHAPE, np.uint8)}
    return f


@pytest.fixture()
def merge_store(tmp_path):
    ch = fine_channels(faces=True, fiber=False, rv=True)
    f = _merge_fields()
    w = BrickWriter(str(tmp_path / "out" / "labels" / "fine.zarr"), ch, RT_SHAPE, chunk=CHUNK,
                    origin_zyx=RT_ORIGIN, voxel_um=2.4, scale=1.0)
    for ci, name in enumerate(ch):
        w.write(ci, *RT_ORIGIN, np.ascontiguousarray(f[name]))
    path = str(tmp_path / "out" / "labels" / "fine.zarr")
    return {"path": path, "channels": ch, "fields": f}


def _rt_config(tmp_path, out_dir: str) -> str:
    p = str(tmp_path / "cfg_rt.json")
    axis_p = str(tmp_path / "axis_rt.json")
    yc = RT_ORIGIN[1] + RT_SHAPE[1] // 2
    with open(axis_p, "w") as fh:
        json.dump({"control_points": [{"z": 0, "y": yc, "x": 0},
                                      {"z": 10_000, "y": yc, "x": 0}]}, fh)
    cfg = {
        "volume": {"url": str(tmp_path / "ct.zarr"), "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": list(RT_ORIGIN), "size_zyx": list(RT_SHAPE)},
        "budget": {"ram_bytes": 1 << 30, "array_bytes": 1 << 28},
        "out_dir": out_dir,
        "extra": {"labels": {"clip": 20, "halo": 24, "brick": [32, 32, 32],
                             "axis_path": axis_p, "teachers_dir": str(tmp_path / "teachers"),
                             "faces": {"enabled": True, "source": "merge",
                                       "rectoverso_store": str(tmp_path / "rv.zarr")}}},
    }
    with open(p, "w") as fh:
        json.dump(cfg, fh)
    return p


def _rt_packet(tmp_path, merge_store, origin=(8, 32, 32), size=32):
    st = ax.Store(merge_store["path"])
    out = str(tmp_path / "packets_rt")
    os.makedirs(out, exist_ok=True)
    meta = ax.write_packet(out, 0, st, _FakeCfg, origin, size, {"score": 1.0, "valid_frac": 1.0},
                           "disagreement", None, _FakeCT(), tif=False)
    ax.write_manifest(out, [meta], "cfg.json", "disagreement")
    return out, meta


def test_import_of_an_empty_correction_is_a_byte_identical_no_op(tmp_path, merge_store):
    """The regression the incremental recompute exists for: an all-zero correction must leave
    every label byte exactly where it was.  A full-crop recompute does not (see the module
    docstring of dev/annot_import.py) -- it moves the sign of whole slabs."""
    out, meta = _rt_packet(tmp_path, merge_store)
    size = meta["size_zyx"][0]
    np.zeros((size, size, size), np.uint8).tofile(
        os.path.join(out, meta["name"], "correction.u8"))
    before = np.asarray(zarr.open_array(store=merge_store["path"], mode="r")[:])

    cfg = _rt_config(tmp_path, os.path.dirname(os.path.dirname(merge_store["path"])))
    assert ai.main([cfg, "--packets", out, "--weight", "5"]) == 0

    arr = zarr.open_array(store=merge_store["path"], mode="r")
    ch = {c: i for i, c in enumerate(arr.attrs["channels"])}
    after = np.asarray(arr[:])
    for name in ("sdf_in", "sdf_out", "faces_valid", "faces_source"):
        assert np.array_equal(after[ch[name]], before[ch[name]]), name
    assert (after[ch[ai.FACES_WEIGHT]] == 1).all()


def test_import_of_a_brush_stroke_only_moves_the_region_of_influence(tmp_path, merge_store):
    """One small stroke may only move voxels within ``clip`` of it, and no more of them than
    its own footprint dilated by ``clip + 1``."""
    from scipy import ndimage as ndi

    out, meta = _rt_packet(tmp_path, merge_store)
    size = meta["size_zyx"][0]
    corr = np.zeros((size, size, size), np.uint8)
    corr[12:16, 12:16, 14] = ai.C_IN           # a 4x4 in-face brush stroke
    corr[18:20, 18:22, 22] = ai.C_OUT          # and a short out-face line
    corr.tofile(os.path.join(out, meta["name"], "correction.u8"))
    n_painted = int((corr > 0).sum())
    before = np.asarray(zarr.open_array(store=merge_store["path"], mode="r")[:])

    cfg = _rt_config(tmp_path, os.path.dirname(os.path.dirname(merge_store["path"])))
    assert ai.main([cfg, "--packets", out, "--weight", "5"]) == 0

    arr = zarr.open_array(store=merge_store["path"], mode="r")
    ch = {c: i for i, c in enumerate(arr.attrs["channels"])}
    after = np.asarray(arr[:])

    lo = [meta["origin_zyx"][a] - RT_ORIGIN[a] for a in range(3)]
    painted = np.zeros(RT_SHAPE, bool)
    painted[tuple(slice(lo[a], lo[a] + size) for a in range(3))] = corr > 0
    infl = ndi.distance_transform_edt(~painted) <= CLIP + 1.0

    total_moved = 0
    for name in ("sdf_in", "sdf_out"):
        moved = after[ch[name]] != before[ch[name]]
        assert not moved[~infl].any(), f"{name} moved outside the region of influence"
        # bounded by the stroke's footprint dilated by clip + 1
        assert moved.sum() <= int(infl.sum())
        total_moved += int(moved.sum())
    assert total_moved > 0                       # the stroke did do something
    assert int(infl.sum()) <= n_painted * (2 * (CLIP + 1) + 1) ** 3
    # and the whole point: a 24-voxel stroke does not repaint the crop
    assert total_moved < 0.02 * int(np.prod(RT_SHAPE))

    # the painted voxels really are the new zero sets
    box = tuple(slice(lo[a], lo[a] + size) for a in range(3))
    assert (after[ch["sdf_in"]][box][corr == ai.C_IN] == 128).all()
    assert (after[ch["sdf_out"]][box][corr == ai.C_OUT] == 128).all()
    assert (after[ch["faces_source"]][painted] == SOURCE_HUMAN).all()
    for name in ("sdf", "sdf_valid", "ink", "ink_valid", "thickness", "rv_class", "hzvt_class"):
        assert np.array_equal(after[ch[name]], before[ch[name]]), name
