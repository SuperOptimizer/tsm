"""Orientation-free (body) evaluation: ``dev/eval_region.py`` body sections, the unordered
upstream score, the fibre exclusivity metric, ``--axis`` plumbing and ``dev/tta_consistency.py``.

All CPU, all synthetic: no CT, no network, no GPU.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))

from synth import AXIS_YX, make_synthetic, synthetic_axis_tilted  # noqa: E402
from tsm.config import parse_config  # noqa: E402
from tsm.data import CLIP, encode_prob, encode_sdf, encode_signed  # noqa: E402
from tsm.volume import BrickWriter  # noqa: E402

eval_region = pytest.importorskip("eval_region")
tta_consistency = pytest.importorskip("tta_consistency")

SHAPE = (48, 32, 32)


def _write(path, channels, arrs, origin=(0, 0, 0), voxel_um=2.4, scale=1.0, chunk=16):
    w = BrickWriter(path, list(channels), arrs[0].shape, chunk=chunk, origin_zyx=origin,
                    voxel_um=voxel_um, scale=scale)
    for ci, a in enumerate(arrs):
        w.write(ci, origin[0], origin[1], origin[2], np.ascontiguousarray(a))
    return path


def _budget(root):
    return parse_config({"volume": {"url": "x"},
                         "region": {"start_zyx": [0, 0, 0], "size_zyx": list(SHAPE)},
                         "out_dir": str(root)}).budget


# --------------------------------------------------------------------------- #
# a body prediction store + the upstream rectoverso slab
# --------------------------------------------------------------------------- #
#: two sheets; each one's recto and verso band touch, so ``rectoverso > 0`` is ONE
#: 26-component per sheet -- the unordered reference object the body target is scored against.
SHEETS = ((9, 13), (25, 29))       # [z0, z1) of the 4-voxel band of each sheet
CORE = ((10, 12), (26, 28))        # the 2-voxel sheet body inside it


def _rv_volume() -> np.ndarray:
    rv = np.zeros(SHAPE, np.uint8)
    for z0, z1 in SHEETS:
        rv[z0:z0 + 2] = 1          # recto
        rv[z0 + 2:z1] = 2          # verso
    return rv


def _body_stores(root: str, weld: bool = False):
    """(rectoverso store, body prediction store).  The prediction is exact unless ``weld``."""
    from tsm.rvfaces import thin_band

    rv = _rv_volume()
    rv_path = _write(os.path.join(root, "rv.zarr"), ["rectoverso", "hzvt"],
                     [rv, np.zeros_like(rv)])
    thin = thin_band(rv > 0, 32)
    body = np.zeros(SHAPE, bool)
    for z0, z1 in CORE:
        body[z0:z1] = True
    if weld:                        # one column of body welding the two sheets into one component
        body[CORE[0][1] - 1:CORE[1][0] + 1, 0, 0] = True
    sdf = np.where(body, 1.0, -5.0).astype(np.float32)
    ch = ["sdf_body", "valid", "surface_body1"]
    pred_path = _write(os.path.join(root, "pred.zarr"), ch,
                       [encode_sdf(sdf), np.full(SHAPE, 255, np.uint8),
                        thin.astype(np.uint8) * 255])
    return rv_path, pred_path


def test_upstream_body_pass_scores_the_unordered_band(tmp_path):
    """A perfect body prediction: Dice 1 against the thinned recto U verso U contact union."""
    rv_path, pred_path = _body_stores(str(tmp_path))
    m = eval_region.upstream_body_pass(
        eval_region.open_store(pred_path), eval_region.open_store(rv_path),
        [0, 0, 0], list(SHAPE), _budget(tmp_path), brick=(16, 32, 32))
    assert m["teacher_independent"] is True and m["unordered"] is True
    assert m["prediction"] == "surface_body1"
    assert m["n_pred"] > 0 and m["n_band"] == m["n_pred"]
    assert m["dice_within2"] == pytest.approx(1.0)
    assert m["pred_to_band_median"] == 0.0 and m["band_to_pred_median"] == 0.0


def test_upstream_topology_body_entry_on_a_body_store(tmp_path):
    """Perfect -> no merges; a weld between the two sheet bodies -> merges_per_100 > 0."""
    budget = _budget(tmp_path)
    kw = dict(brick=(16, 32, 32), min_component=8, min_size=512, clip=CLIP)
    rv_path, pred_path = _body_stores(str(tmp_path / "ok"))
    ok = eval_region.upstream_topology_pass(
        eval_region.open_store(pred_path), eval_region.open_store(rv_path),
        [0, 0, 0], list(SHAPE), budget, **kw)
    assert "in" not in ok and "out" not in ok           # a body store has no named faces
    assert "inout_cross_links" not in ok
    b = ok["body"]
    assert b["n_ref_objects"] == 2 and b["n_student_components"] == 2
    assert (b["merges"], b["breaks"], b["missed"], b["spurious"]) == (0, 0, 0, 0)
    assert b["merges_per_100"] == 0.0

    rv2, pred2 = _body_stores(str(tmp_path / "weld"), weld=True)
    bad = eval_region.upstream_topology_pass(
        eval_region.open_store(pred2), eval_region.open_store(rv2),
        [0, 0, 0], list(SHAPE), budget, **kw)["body"]
    assert bad["n_student_components"] == 1
    assert bad["merges"] == 1 and bad["merges_per_100"] > 0.0

    md = eval_region.to_markdown({"region": {"start_zyx": [0, 0, 0], "size_zyx": list(SHAPE)},
                                  "clip": CLIP, "seconds": 1.0, "upstream_topology": ok,
                                  "upstream_body": {"dice_within2": 1.0}})
    assert "| metric | body |" in md and "## upstream_body" in md
    assert "inout_cross_links" not in md


# --------------------------------------------------------------------------- #
# the unordered score is invariant to a global in/out swap; the per-face one is not
# --------------------------------------------------------------------------- #
def _faces_stores(root: str):
    """A recto/verso slab with well separated faces + a two-face student on the thinned bands,
    and the SAME student with its two faces globally swapped."""
    rv = np.zeros(SHAPE, np.uint8)
    for z0 in (10, 26):
        rv[z0:z0 + 3] = 1                  # recto -> in face
    for z0 in (18, 34):
        rv[z0:z0 + 3] = 2                  # verso -> out face
    rv_path = _write(os.path.join(root, "rv.zarr"), ["rectoverso", "hzvt"], [rv, np.zeros_like(rv)])
    s_in = np.zeros(SHAPE, np.uint8)
    s_out = np.zeros(SHAPE, np.uint8)
    s_in[[11, 27]] = 255
    s_out[[19, 35]] = 255
    ch = ["sdf_in", "sdf_out", "valid", "surface_in1", "surface_out1"]
    flat = np.full(SHAPE, 128, np.uint8)
    valid = np.full(SHAPE, 255, np.uint8)
    a = _write(os.path.join(root, "pred.zarr"), ch, [flat, flat.copy(), valid, s_in, s_out])
    b = _write(os.path.join(root, "swap.zarr"), ch, [flat.copy(), flat.copy(), valid.copy(),
                                                     s_out, s_in])
    return rv_path, a, b


def test_unordered_dice_survives_a_global_face_swap_while_the_per_face_dice_collapses(tmp_path):
    rv_path, pred_path, swap_path = _faces_stores(str(tmp_path))
    budget = _budget(tmp_path)
    rv = eval_region.open_store(rv_path)
    args = ([0, 0, 0], list(SHAPE), budget)
    kw = dict(brick=(16, 32, 32))

    face_a = eval_region.upstream_faces_pass(eval_region.open_store(pred_path), rv, *args, **kw)
    face_b = eval_region.upstream_faces_pass(eval_region.open_store(swap_path), rv, *args, **kw)
    assert face_a["in"]["dice_within2"] == pytest.approx(1.0)
    assert face_a["out"]["dice_within2"] == pytest.approx(1.0)
    assert face_b["in"]["dice_within2"] == 0.0 and face_b["out"]["dice_within2"] == 0.0

    body_a = eval_region.upstream_body_pass(eval_region.open_store(pred_path), rv, *args, **kw)
    body_b = eval_region.upstream_body_pass(eval_region.open_store(swap_path), rv, *args, **kw)
    assert body_a["prediction"] == "surface_in1 | surface_out1"
    assert body_a["dice_within2"] == pytest.approx(1.0)
    assert body_b["dice_within2"] == body_a["dice_within2"]
    assert body_b["n_pred"] == body_a["n_pred"] and body_b["n_band"] == body_a["n_band"]


def test_two_face_topology_keeps_the_face_entries_and_gains_a_body_column(tmp_path):
    """A faces store keeps `in` / `out` / `inout_cross_links` (~/cmp_aug.py reads those keys)."""
    rv_path, pred_path, _ = _faces_stores(str(tmp_path))
    m = eval_region.upstream_topology_pass(
        eval_region.open_store(pred_path), eval_region.open_store(rv_path),
        [0, 0, 0], list(SHAPE), _budget(tmp_path), brick=(16, 32, 32), min_component=8,
        min_size=512, clip=CLIP)
    assert set(("in", "out", "body")) <= set(m) and "inout_cross_links" in m
    for face in ("in", "out"):
        assert m[face]["n_ref_objects"] == 2 and m[face]["merges"] == 0
    # the flat sdf bytes (128 = 0) give no body at all: every unordered object is missed
    assert m["body"]["n_ref_objects"] == 4 and m["body"]["n_student_components"] == 0
    md = eval_region.to_markdown({"region": {"start_zyx": [0, 0, 0], "size_zyx": list(SHAPE)},
                                  "clip": CLIP, "seconds": 1.0, "upstream_topology": m})
    assert "| metric | in | out | body |" in md
    assert "inout_cross_links.cross_links" in md


# --------------------------------------------------------------------------- #
# fibre exclusivity (fiber.overlap_frac) extremes
# --------------------------------------------------------------------------- #
def _fiber_pred(root: str, mode: str) -> str:
    """A full prediction store whose two fibre class channels either always or never coincide."""
    from tsm.infer import pred_channels

    ch = pred_channels("medial", True, "class")
    vt = np.zeros(SHAPE, np.uint8)
    hz = np.zeros(SHAPE, np.uint8)
    if mode == "all":
        vt[:] = 255
        hz[:] = 255
    else:                                   # disjoint: vt on one half of x, hz on the other
        vt[..., : SHAPE[2] // 2] = 255
        hz[..., SHAPE[2] // 2:] = 255
    z = np.zeros(SHAPE, np.float32)
    fields = {
        "sdf": np.full(SHAPE, 128, np.uint8), "valid": np.full(SHAPE, 255, np.uint8),
        "ink": np.zeros(SHAPE, np.uint8), "sin": encode_signed(z), "cos": encode_signed(z + 1),
        "density": np.full(SHAPE, 40, np.uint8), "nx": encode_signed(z), "ny": encode_signed(z),
        "nz": encode_signed(z + 1), "conf": encode_prob(z + 0.7), "spare": np.zeros(SHAPE, np.uint8),
        "fiber_vt": vt, "fiber_hz": hz, "surface1": np.zeros(SHAPE, np.uint8),
    }
    return _write(os.path.join(root, f"pred_{mode}.zarr"), ch, [fields[c] for c in ch])


@pytest.mark.parametrize("mode,expect", [("all", 1.0), ("disjoint", 0.0)])
def test_fiber_overlap_frac_extremes(tmp_path, mode, expect):
    p = eval_region.open_store(_fiber_pred(str(tmp_path), mode))
    m, _ = eval_region.voxel_pass(p, None, None, None, [0, 0, 0], list(SHAPE), CLIP,
                                  _budget(tmp_path), brick=(16, 32, 32))
    f = m["fiber"]
    assert f["overlap_frac"] == pytest.approx(expect)
    assert f["n_any"] == int(np.prod(SHAPE))
    assert f["firing_frac"] == pytest.approx(1.0)   # every voxel has data and one class firing


# --------------------------------------------------------------------------- #
# --axis plumbing
# --------------------------------------------------------------------------- #
def _axis_json(root: str, tilted: bool) -> str:
    pts = synthetic_axis_tilted() if tilted else np.array(
        [[-100000, AXIS_YX[0], AXIS_YX[1]], [100000, AXIS_YX[0], AXIS_YX[1]]], float)
    path = os.path.join(root, f"axis_{'tilt' if tilted else 'straight'}.json")
    with open(path, "w") as fh:
        json.dump({"coordinate_order": "xyz", "coordinate_space": "full_resolution",
                   "control_points": [{"z": float(z), "y": float(y), "x": float(x)}
                                      for z, y, x in pts]}, fh)
    return path


def test_resolve_axis_prefers_labels_then_train_then_none(tmp_path):
    root = str(tmp_path)
    lab, tr = _axis_json(root, True), _axis_json(root, False)
    mk = lambda extra: parse_config({  # noqa: E731
        "volume": {"url": "x"}, "region": {"start_zyx": [0, 0, 0], "size_zyx": list(SHAPE)},
        "out_dir": root, "extra": extra})
    assert eval_region.resolve_axis(mk({"labels": {"axis_path": lab}}), None) == lab
    assert eval_region.resolve_axis(mk({"train": {"axis_path": tr}}), None) == tr
    assert eval_region.resolve_axis(mk({"labels": {"axis_path": lab},
                                        "train": {"axis_path": tr}}), None) == lab
    assert eval_region.resolve_axis(mk({}), tr) == tr          # --axis wins
    assert eval_region.resolve_axis(mk({"labels": {"axis_path": lab}}), "none") is None
    assert eval_region.resolve_axis(mk({}), str(tmp_path / "missing.json")) is None


def _fiber_dir_store(root: str, P: int = 16):
    """A direction-mode prediction on a plane sheet + the human hz/vt band store."""
    from tsm.fiber import derive_direction_targets
    from tsm.infer import pred_channels, write_fiber_class_channels

    n0 = np.array([0.0, 1.0, 0.0], np.float32)     # sheet normal along y
    g = np.stack(np.meshgrid(*[np.arange(P, dtype=np.float32) - (P - 1) / 2.0] * 3, indexing="ij"))
    sdf = (g * n0.reshape(3, 1, 1, 1)).sum(0)[None]
    ones = np.ones((1, P, P, P), np.float32)
    band = np.zeros((1, P, P, P), np.uint8)
    band[:, : P // 2] = 1
    band[:, P // 2:] = 2
    t = derive_direction_targets(ones * 0.0, ones * 0.0, sdf, ones, band=band.astype(np.float32))
    ch = pred_channels("faces", True, "direction")
    fields = {c: np.zeros((P, P, P), np.uint8) for c in ch}
    fields["sdf_in"] = encode_sdf(sdf[0])
    fields["sdf_out"] = encode_sdf(sdf[0] - 4.0)
    for i, name in enumerate(("fiber_dz", "fiber_dy", "fiber_dx")):
        fields[name] = encode_signed(t["fiber_dir"][i])
    fields["fiber_strength"] = np.full((P, P, P), 255, np.uint8)
    pred = _write(os.path.join(root, "fpred.zarr"), ch, [fields[c] for c in ch], chunk=P)
    write_fiber_class_channels(pred, CLIP, brick=P)
    bandp = _write(os.path.join(root, "fband.zarr"), ["rectoverso", "hzvt"],
                   [np.zeros((P, P, P), np.uint8), band[0]], chunk=P)
    return pred, bandp, P


def test_fiber_upstream_pass_accepts_a_tilted_axis(tmp_path):
    """The per-brick axis tangent reaches ``fiber_basis``: finite numbers, and the tilted axis
    is not the same basis as the constant (1, 0, 0)."""
    root = str(tmp_path)
    pred, bandp, P = _fiber_dir_store(root)
    budget = _budget(tmp_path)
    args = (eval_region.Store(pred), eval_region.Store(bandp), "hzvt",
            [0, 0, 0], [P, P, P], CLIP, budget)
    flat = eval_region.fiber_upstream_pass(*args, brick=(P, P, P))
    tilt = eval_region.fiber_upstream_pass(*args, brick=(P, P, P),
                                           axis=synthetic_axis_tilted(0.4, -0.3))
    assert flat["axis_tangent"] is False and tilt["axis_tangent"] is True
    for m in (flat, tilt):
        assert m["n_band"] > 0 and np.isfinite(m["class_acc"])
        assert m["angle_deg_n"] > 0 and np.isfinite(m["angle_deg_median"])
        assert 0.0 <= m["angle_deg_median"] <= 180.0
    # the targets were derived with the constant axis, so the tilted basis must disagree more
    assert tilt["angle_deg_median"] > flat["angle_deg_median"]
    # the path form is recorded in the report
    ax = _axis_json(root, True)
    named = eval_region.fiber_upstream_pass(*args, brick=(P, P, P), axis=ax)
    assert named["axis"] == ax and named["axis_tangent"] is True


# --------------------------------------------------------------------------- #
# dev/tta_consistency.py
# --------------------------------------------------------------------------- #
def _grad3(c: torch.Tensor) -> torch.Tensor:
    """(B, 3, Z, Y, X) central differences (dz, dy, dx) with replicate padding -- genuinely
    equivariant under the flips and the y<->x transpose of the TTA group."""
    p = torch.nn.functional.pad(c, (1, 1, 1, 1, 1, 1), mode="replicate")
    gz = p[:, :, 2:, 1:-1, 1:-1] - p[:, :, :-2, 1:-1, 1:-1]
    gy = p[:, :, 1:-1, 2:, 1:-1] - p[:, :, 1:-1, :-2, 1:-1]
    gx = p[:, :, 1:-1, 1:-1, 2:] - p[:, :, 1:-1, 1:-1, :-2]
    return torch.cat([gz, gy, gx], 1) * 0.5


class _Equivariant(torch.nn.Module):
    """Every head a local function of the CT: scalars pointwise, vectors from its gradient."""

    def forward(self, x):
        c = x[:, 0:1]
        g = _grad3(c)
        one, zero = torch.ones_like(c), torch.zeros_like(c)
        return {
            "surface": torch.cat([3.0 * c, one], 1),
            "ink": c,
            # winding = [sin, cos, density, nx, ny, nz, conf, spare]; (nx, ny, nz) <-> (gx, gy, gz)
            "winding": torch.cat([c, one, one, g[:, 2:3], g[:, 1:2], g[:, 0:1], one, zero], 1),
            "fiber": torch.cat([g, one], 1),          # [dz, dy, dx, strength]
        }


class _ReadsOrientation(torch.nn.Module):
    """Every head read off the volume axes: a z ramp and constant (z, y, x) vectors."""

    def forward(self, x):
        c = x[:, 0:1]
        Z = c.shape[2]
        ramp = (torch.arange(Z, dtype=c.dtype, device=c.device) - (Z - 1) / 2.0)
        ramp = ramp.view(1, 1, Z, 1, 1).expand_as(c).contiguous()
        one, zero = torch.ones_like(c), torch.zeros_like(c)
        return {
            "surface": torch.cat([ramp, one], 1),
            "ink": c,
            "winding": torch.cat([ramp, one, one, one, zero, zero, one, zero], 1),
            # a constant direction that is NOT stabilised by the flips/transpose even axially
            "fiber": torch.cat([one, 2.0 * one, 3.0 * one, one], 1),
        }


@pytest.fixture(scope="module")
def _tta_case(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("tta"))
    s = make_synthetic(root, fine_shape=(64, 64, 64), fine_origin=(16, 32, 32))
    cfg = {
        "volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 4 << 30, "array_bytes": 1 << 30},
        "out_dir": root,
        "extra": {"infer": {"checkpoint": os.path.join(root, "fake.pt")},
                  "train": {"patch": 16, "stride": 16, "steps": 1, "input_radial": False,
                            "input_axis": False, "fine_store": s["fine"],
                            "coarse_store": s["coarse"], "holdout_origins": {"z_frac": 0.5}}},
    }
    path = os.path.join(root, "cfg.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return path


def _fake_loader(net):
    info = {"surface_mode": "medial", "fiber_mode": "direction", "input_radial": False,
            "input_axis": False, "axis_tangent": False, "axis_path": None, "step": 0}
    return lambda path, device=None, widths=None: (net.eval(), info)


def test_tta_consistency_is_zero_for_an_equivariant_net(_tta_case, monkeypatch):
    monkeypatch.setattr(tta_consistency, "load_student", _fake_loader(_Equivariant()))
    m = tta_consistency.run(_tta_case, crops=2, tta="flip8_rot4", seed=0)
    assert m["n_members"] == 16 and m["n_crops"] == 2 and m["holdout_origins"] is True
    assert m["sdf_std_p90"] < 1e-3, m
    assert m["body_mask_agreement"] > 0.999
    assert m["winding_normal_angle_p90"] < 0.1
    assert m["phase_std_deg_p90"] < 0.1
    assert m["fiber_dir_angle_p90"] < 0.1
    assert "sdf_std_p50" in m and m["sdf_channel"] == "sdf"
    assert "sdf_std_p90" in tta_consistency.to_table(m)


def test_tta_consistency_is_large_for_a_net_that_reads_the_volume_axes(_tta_case, monkeypatch):
    monkeypatch.setattr(tta_consistency, "load_student", _fake_loader(_ReadsOrientation()))
    m = tta_consistency.run(_tta_case, crops=2, tta="flip8_rot4", seed=0)
    assert m["sdf_std_p90"] > 1.0
    assert m["body_mask_agreement"] < 0.9
    assert m["winding_normal_angle_p90"] > 45.0
    assert m["fiber_dir_angle_p90"] > 20.0


def test_tta_consistency_refuses_the_laptop_gpu(monkeypatch):
    monkeypatch.delenv(tta_consistency.ALLOW_LOCAL_GPU_ENV, raising=False)
    assert tta_consistency.pick_device().type == "cpu"
    assert tta_consistency.pick_device("cpu").type == "cpu"
    with pytest.raises(SystemExit, match="laptop GPU"):
        tta_consistency.pick_device("cuda")
    monkeypatch.setenv(tta_consistency.ALLOW_LOCAL_GPU_ENV, "1")
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit, match="no CUDA device"):
            tta_consistency.pick_device("cuda")


# --------------------------------------------------------------------------- #
# end to end: eval_region.main on a body prediction store
# --------------------------------------------------------------------------- #
E2E_SHAPE = (64, 64, 64)
E2E_ORIGIN = (16, 32, 32)


def _make_body_case(root: str, noise: float = 0.0) -> str:
    """A body student that copies ``body_sdf(sdf_in, sdf_out)`` of the faces labels."""
    import zarr

    from tsm.data import body_sdf, decode_sdf
    from tsm.infer import BODY_PRED_CHANNELS, extract_surface
    from tsm.labels import decode_sdf_u8

    s = make_synthetic(root, fine_shape=E2E_SHAPE, fine_origin=E2E_ORIGIN, faces=True)
    arr = zarr.open_array(store=s["fine"], mode="r")
    ch = list(arr.attrs["channels"])
    g = lambda n: np.asarray(arr[ch.index(n)])  # noqa: E731
    b = body_sdf(decode_sdf(g("sdf_in")), decode_sdf(g("sdf_out")))
    if noise:
        b = b + np.random.default_rng(0).normal(0, noise, b.shape).astype(np.float32)
    sdf_u8 = encode_sdf(b)
    valid = np.full(E2E_SHAPE, 255, np.uint8)
    z = np.arange(E2E_SHAPE[0], dtype=np.float32)[:, None, None] * np.ones(E2E_SHAPE, np.float32)
    fields = {
        "sdf_body": sdf_u8, "valid": valid, "ink": g("ink"),
        "sin": encode_signed(np.sin(0.3 * z)), "cos": encode_signed(np.cos(0.3 * z)),
        "density": np.full(E2E_SHAPE, 40, np.uint8),
        "nx": encode_signed(np.zeros(E2E_SHAPE, np.float32)),
        "ny": encode_signed(np.zeros(E2E_SHAPE, np.float32)),
        "nz": encode_signed(np.ones(E2E_SHAPE, np.float32)),
        "conf": encode_prob(np.full(E2E_SHAPE, 0.7, np.float32)),
        "spare": np.zeros(E2E_SHAPE, np.uint8),
        "surface_body1": extract_surface(sdf_u8, valid, CLIP).astype(np.uint8) * 255,
    }
    _write(os.path.join(root, "student", "pred.zarr"), BODY_PRED_CHANNELS,
           [fields[c] for c in BODY_PRED_CHANNELS], E2E_ORIGIN, chunk=32)
    band = (np.abs(decode_sdf_u8(g("sdf"), CLIP)) <= 1.5) & (g("sdf") != 0)
    tdir = os.path.join(root, "teachers")
    _write(os.path.join(tdir, "recto.zarr"), ["recto"], [band.astype(np.uint8) * 255],
           E2E_ORIGIN, chunk=32)
    _write(os.path.join(tdir, "ink.zarr"), ["ink"], [g("ink")], E2E_ORIGIN, chunk=32)
    cfg = {"volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
           "region": {"start_zyx": list(E2E_ORIGIN), "size_zyx": list(E2E_SHAPE)},
           "budget": {"ram_bytes": 2 << 30, "array_bytes": 1 << 30},
           "out_dir": root, "extra": {}}
    path = os.path.join(root, "eval.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh)
    return path


def test_eval_region_body_mode_end_to_end(tmp_path):
    """A body student that copies the label body SDF: MAE ~ 0, body Dice ~ 1, `body` section."""
    cfgp = _make_body_case(str(tmp_path / "out"))
    m = eval_region.main([cfgp, "--no-gallery", "--brick", "32,32,32", "--axis", "none"])
    assert m["surface"]["student_surface"] == "body_sdf"
    assert m["surface"]["sdf_mae_band"] < 0.2
    # < 1 on purpose: the student writes a surface everywhere, the label zero set is restricted
    # to faces_valid == 1 (the fixture has both an "air" strip and ignore patches)
    assert m["surface"]["zc_dice_within1"] > 0.85
    b = m["body"]
    assert b["labels"].endswith(os.path.join("labels", "fine.zarr"))
    assert b["n_faces_valid1"] > 0 and b["n_label_body"] > 0
    assert b["body_dice"] > 0.99 and b["body_iou"] > 0.98
    assert b["body_vol_ratio"] == pytest.approx(1.0, abs=0.02)
    assert "faces" not in m and "surface_inface_vs_recto" not in m   # no named faces in body mode
    assert m["axis"] is None
    md = open(os.path.join(str(tmp_path / "out"), "eval", "metrics.md")).read()
    assert "## body" in md and "| body_dice |" in md


def test_eval_region_body_mode_noisy_student_is_worse(tmp_path):
    a = eval_region.main([_make_body_case(str(tmp_path / "a")), "--no-gallery",
                          "--brick", "32,32,32", "--axis", "none"])
    b = eval_region.main([_make_body_case(str(tmp_path / "b"), noise=6.0), "--no-gallery",
                          "--brick", "32,32,32", "--axis", "none"])
    assert b["surface"]["sdf_mae_band"] > a["surface"]["sdf_mae_band"]
    assert b["body"]["body_dice"] < a["body"]["body_dice"]
