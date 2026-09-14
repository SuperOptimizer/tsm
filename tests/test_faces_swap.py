"""``extra.train.faces_swap_invariant``: the two-face surface loss without a face *naming*.

The two-face target names its faces by a global outward convention (in = toward the umbilicus),
which no single crop reveals: a sheet seen from the other side is the same sheet.  With the flag
the loss picks, per sample, the cheaper of the two assignments, so the label carries no naming.

The swap is **negate and swap** -- ``(t_in, t_out) -> (-t_out, -t_in)`` (:func:`swap_faces_target`)
-- because both channels measure the signed distance to their own face along the SAME direction
(the one running from the in-face to the out-face).  A plain swap ``(t_out, t_in)`` is not even a
valid two-face target; the tests below pin both facts on the analytic two-sheet crop.

CPU only, no GPU, no network.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))

from synth import AXIS_YX, make_synthetic  # noqa: E402
from tsm.config import parse_config  # noqa: E402
from tsm.data import CLIP, CropDataset  # noqa: E402
from tsm.train import (  # noqa: E402
    body_mask,
    compute_losses,
    evaluate,
    faces_loss,
    faces_swap_choice,
    run_train,
    swap_faces_target,
    train_opts,
)

D, H, W = 48, 8, 8
ZIN, ZOUT = 24.0, 34.0
#: the same aux block the shipped `faces30k_gf_perm` config uses, plus the per-face terms
AUX = {"shell": 1.0, "shell_radius": 8, "shell_margin": 3.0, "crest": 1.0, "crest_tol": 1.0,
       "far": 1.0, "far_margin": 6.0, "gap": 1.0, "gap_radius": 3, "cldice": 1.0}
BODY_AUX = {"gap": 1.0, "gap_radius": 3, "cldice": 1.0}


def _z() -> torch.Tensor:
    z = torch.arange(D, dtype=torch.float32)[None, None, :, None, None]
    return z.expand(1, 1, D, H, W).clone()


def _labels() -> tuple[torch.Tensor, torch.Tensor]:
    """The analytic sheet pair: an IN face at z = ZIN and an OUT face at z = ZOUT."""
    z = _z()
    sdf = torch.cat([(z - ZIN).clamp(-CLIP, CLIP), (z - ZOUT).clamp(-CLIP, CLIP)], dim=1)
    return sdf, torch.ones(1, 1, D, H, W)


def _pred(sdf_in: torch.Tensor, sdf_out: torch.Tensor) -> torch.Tensor:
    return torch.cat([sdf_in, sdf_out, torch.full_like(sdf_in, 10.0)], dim=1)


def _wobbly() -> torch.Tensor:
    """An imperfect but plausible prediction (both faces shifted and softened)."""
    sdf, _ = _labels()
    a = (sdf[:, 0:1] * 0.9 + 0.7).clamp(-CLIP, CLIP)
    b = (sdf[:, 1:2] * 1.1 - 0.4).clamp(-CLIP, CLIP)
    return _pred(a, b)


def _total(terms: dict[str, torch.Tensor]) -> float:
    """The summed loss, added in a canonical order.

    Under a face relabelling the *terms* are bit-identical, only their keys are exchanged; adding
    them in dict order would then differ in the last ulp (float addition is commutative but not
    associative).  Summing the sorted values compares the multisets, which is the exact claim.
    """
    return float(sum(sorted(float(v) for v in terms.values())))


def _flip_pred(pred: torch.Tensor) -> torch.Tensor:
    """The prediction with its two face channels relabelled -- the same rule as the target."""
    return torch.cat([-pred[:, 1:2], -pred[:, 0:1], pred[:, 2:3]], dim=1)


# --------------------------------------------------------------------------- #
# the convention: negate AND swap
# --------------------------------------------------------------------------- #
def test_swapped_target_preserves_the_body_thickness_and_medial_set():
    sdf, valid = _labels()
    sw = swap_faces_target(sdf)
    t_in, t_out = sdf[:, 0:1], sdf[:, 1:2]
    assert torch.equal(sw[:, 0:1], -t_out) and torch.equal(sw[:, 1:2], -t_in)
    # the body is the SAME voxel set read with the roles exchanged
    b0 = body_mask(t_in, t_out, torch.ones_like(t_in))
    b1 = body_mask(sw[:, 0:1], sw[:, 1:2], torch.ones_like(t_in))
    assert torch.equal(b0, b1)
    assert float(b0.sum()) == (ZOUT - ZIN - 1) * H * W  # the open slab 24 < z < 34
    # thickness and medial zero set are untouched
    assert torch.equal(sw[:, 0:1] - sw[:, 1:2], t_in - t_out)
    assert torch.equal(sw[:, 0:1] + sw[:, 1:2], -(t_in + t_out))


def test_a_plain_swap_is_not_a_valid_two_face_target():
    """``(t_out, t_in)`` (no negation) empties the body: the naming is not a plain permutation."""
    sdf, _ = _labels()
    plain = torch.cat([sdf[:, 1:2], sdf[:, 0:1]], dim=1)
    b = body_mask(plain[:, 0:1], plain[:, 1:2], torch.ones(1, 1, D, H, W))
    assert float(b.sum()) == 0.0


# --------------------------------------------------------------------------- #
# exact symmetry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("aux", [None, AUX])
def test_loss_is_exactly_symmetric_under_flipping_the_prediction_or_the_label(aux):
    sdf, valid = _labels()
    pred = _wobbly()
    kw = dict(aux=aux, swap_invariant=True)
    base = _total(faces_loss(pred, sdf, valid, **kw))
    flip_p = _total(faces_loss(_flip_pred(pred), sdf, valid, **kw))
    flip_t = _total(faces_loss(pred, swap_faces_target(sdf), valid, **kw))
    assert flip_p == base
    assert flip_t == base
    # flipping BOTH keeps every term, with the two faces' keys exchanged (which is exactly what
    # "the naming is free" means: the numbers are the same, only their labels move)
    both = faces_loss(_flip_pred(pred), swap_faces_target(sdf), valid, **kw)
    ref = faces_loss(pred, sdf, valid, **kw)
    assert _total(both) == base
    for k_in, k_out in (("sdf_in_l1", "sdf_out_l1"), ("band_dice_in", "band_dice_out")):
        assert float(both[k_in]) == float(ref[k_out]) and float(both[k_out]) == float(ref[k_in])
    assert float(both["valid_bce"]) == float(ref["valid_bce"])


def test_without_the_flag_a_flipped_prediction_is_punished():
    """The control: with the historical loss the same flip is a disaster, which is the whole
    point of the flag (and shows the symmetry above is not vacuous)."""
    sdf, valid = _labels()
    pred = _wobbly()
    good = _total(faces_loss(pred, sdf, valid))
    bad = _total(faces_loss(_flip_pred(pred), sdf, valid))
    assert bad > 10.0 * good
    # with the flag the flipped prediction is scored exactly as well as the un-flipped one
    assert _total(faces_loss(_flip_pred(pred), sdf, valid, swap_invariant=True)) == good
    assert _total(faces_loss(pred, sdf, valid, swap_invariant=True)) == good


def test_the_aux_and_body_terms_match_under_the_swap():
    sdf, valid = _labels()
    pred = _wobbly()
    a = faces_loss(pred, sdf, valid, aux=AUX, swap_invariant=True)
    b = faces_loss(_flip_pred(pred), sdf, valid, aux=AUX, swap_invariant=True)
    # the body terms do not even depend on the assignment (the body mask and the soft body are
    # both invariant), so they match term by term...
    for k in ("gap", "cldice", "valid_bce"):
        assert float(a[k]) == float(b[k]), k
    # ...and the per-face terms are exchanged, not changed: the flipped prediction's channel 0
    # plays the role of the label's OUT face, so its terms carry the other face's key
    for k_in, k_out in (("sdf_in_l1", "sdf_out_l1"), ("band_dice_in", "band_dice_out"),
                        ("shell_in", "shell_out"), ("crest_in", "crest_out"), ("far_in", "far_out")):
        assert float(b[k_in]) == float(a[k_out]), k_in
        assert float(b[k_out]) == float(a[k_in]), k_out
    assert _total(a) == _total(b)


# --------------------------------------------------------------------------- #
# default off = bit-identical
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("aux", [None, AUX])
def test_default_off_is_bit_identical_to_the_historical_loss(aux):
    sdf, valid = _labels()
    for pred in (_wobbly(), _flip_pred(_wobbly())):
        old = faces_loss(pred, sdf, valid, aux=aux)
        new = faces_loss(pred, sdf, valid, aux=aux, swap_invariant=False)
        assert set(old) == set(new)
        for k in old:
            assert float(new[k]) == float(old[k]), k


def test_the_flag_changes_nothing_when_the_identity_assignment_already_wins():
    """A correctly oriented prediction keeps every term, bit for bit."""
    sdf, valid = _labels()
    pred = _wobbly()
    off = faces_loss(pred, sdf, valid, aux=AUX)
    on = faces_loss(pred, sdf, valid, aux=AUX, swap_invariant=True)
    for k in off:
        assert float(on[k]) == float(off[k]), k


# --------------------------------------------------------------------------- #
# the per-sample decision and its logging
# --------------------------------------------------------------------------- #
def _mixed_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch of two identical crops whose predictions are oriented the two opposite ways."""
    sdf, valid = _labels()
    pred = torch.cat([_wobbly(), _flip_pred(_wobbly())], dim=0)
    return pred, sdf.repeat(2, 1, 1, 1, 1), valid.repeat(2, 1, 1, 1, 1)


def test_the_choice_is_per_sample():
    pred, sdf, valid = _mixed_batch()
    chosen, swapped = faces_swap_choice(pred, sdf, (valid == 1).float())
    assert swapped.tolist() == [False, True]
    assert torch.equal(chosen[0:1], sdf[0:1])
    assert torch.equal(chosen[1:2], swap_faces_target(sdf)[1:2])
    # both crops end up as well scored as the well-oriented one alone (the batch means exchange
    # the two faces' keys for the flipped sample, so it is the total that must match)
    per = faces_loss(pred[0:1], sdf[0:1], valid[0:1], swap_invariant=True)
    both = faces_loss(pred, sdf, valid, swap_invariant=True)
    assert _total(both) == pytest.approx(_total(per), rel=1e-6)
    assert float(both["valid_bce"]) == pytest.approx(float(per["valid_bce"]), rel=1e-6)


def test_stats_report_the_swapped_fraction():
    pred, sdf, valid = _mixed_batch()
    st: dict[str, float] = {}
    faces_loss(pred, sdf, valid, swap_invariant=True, stats=st)
    assert st["faces_swapped_frac"] == pytest.approx(0.5)
    st2: dict[str, float] = {}
    faces_loss(pred, sdf, valid, swap_invariant=False, stats=st2)
    assert st2 == {}            # nothing is computed with the flag off


def test_compute_losses_logs_aug_faces_swapped_frac_and_does_deep_supervision():
    pred, sdf, valid = _mixed_batch()
    batch = {
        "surface_sdf": sdf, "surface_valid": valid,
        "ink_prob": torch.zeros(2, 1, D, H, W), "ink_valid": torch.ones(2, 1, D, H, W),
        "winding": torch.zeros(2, 6, D, H, W), "winding_conf": torch.zeros(2, 1, D, H, W),
        "winding_valid": torch.zeros(2, 1, D, H, W),
    }
    out = {"surface": [pred, torch.nn.functional.avg_pool3d(pred, 2)],
           "ink": [torch.zeros(2, 1, D, H, W), torch.zeros(2, 1, D // 2, H // 2, W // 2)],
           "winding": [torch.zeros(2, 8, D, H, W), torch.zeros(2, 8, D // 2, H // 2, W // 2)]}
    _, parts = compute_losses(out, batch, surface_mode="faces", surface_aux=BODY_AUX,
                              faces_swap_invariant=True)
    assert parts["aug/faces_swapped_frac"] == pytest.approx(0.5)
    _, off = compute_losses(out, batch, surface_mode="faces", surface_aux=BODY_AUX)
    assert "aug/faces_swapped_frac" not in off
    # deep supervision goes through the same rule: the flipped sample is not punished at either
    # level, so the surface head total is below the un-invariant one
    assert parts["surface"] < off["surface"]


# --------------------------------------------------------------------------- #
# config plumbing
# --------------------------------------------------------------------------- #
def _cfg(extra: dict) -> object:
    return parse_config({"volume": {"url": "x"}, "region": {"start_zyx": [0, 0, 0], "size_zyx": [16, 16, 16]},
                         "out_dir": "/tmp/x", "extra": {"train": extra}})


def test_train_opts_defaults_off_and_requires_faces_mode():
    assert train_opts(_cfg({}))["faces_swap_invariant"] is False
    assert train_opts(_cfg({"surface_mode": "faces", "faces_swap_invariant": True}))["faces_swap_invariant"]
    with pytest.raises(ValueError, match="surface_mode='faces'"):
        train_opts(_cfg({"surface_mode": "body", "faces_swap_invariant": True}))
    with pytest.raises(ValueError, match="must be a bool"):
        train_opts(_cfg({"surface_mode": "faces", "faces_swap_invariant": "yes"}))


def test_the_shipped_configs_pair_a_swap_run_with_its_control():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    perm = json.load(open(os.path.join(root, "faces30k_gf_perm.json")))
    ctrl = json.load(open(os.path.join(root, "faces30k_gf_aux.json")))
    assert perm["extra"]["train"]["faces_swap_invariant"] is True
    assert "faces_swap_invariant" not in ctrl["extra"]["train"]
    for c in (perm, ctrl):
        assert c["extra"]["train"]["surface_mode"] == "faces"
        assert c["extra"]["train"]["surface_aux"] == {"gap": 1.0, "gap_radius": 3, "cldice": 1.0}
    assert perm["out_dir"].endswith("faces30k_gf_perm") and ctrl["out_dir"].endswith("faces30k_gf_aux")


# --------------------------------------------------------------------------- #
# evaluate: the metrics under the better assignment
# --------------------------------------------------------------------------- #
class _Flip(torch.nn.Module):
    """A "student" that returns the label faces, optionally relabelled the other way round."""

    def __init__(self, ds, origins, flip: bool) -> None:
        super().__init__()
        self.t = [ds.to_tensors(ds.load(o)) for o in origins]
        self.flip = flip
        self.i = 0
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n = x.shape[0]
        items = [self.t[(self.i + k) % len(self.t)] for k in range(n)]
        self.i += n
        sdf = torch.stack([it["surface_sdf"] for it in items])
        if self.flip:
            sdf = torch.cat([-sdf[:, 1:2], -sdf[:, 0:1]], dim=1)
        big = torch.full((n, 1) + tuple(sdf.shape[2:]), 10.0)
        return {"surface": torch.cat([sdf, big], 1),
                "ink": torch.zeros_like(big), "winding": torch.zeros((n, 8) + tuple(sdf.shape[2:]))}


def _faces_ds(root: str):
    from tsm.volume import VolumeReader

    s = make_synthetic(root, fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32, faces=True)
    return CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16,
                       length=4, augment=False, surface_mode="faces")


def test_evaluate_scores_a_flipped_student_under_the_better_assignment(tmp_path):
    ds = _faces_ds(str(tmp_path))
    origins = ds.origins[:2]
    kw = dict(batch=1, surface_mode="faces")
    good = evaluate(_Flip(ds, origins, False), ds, origins, **kw)
    bad = evaluate(_Flip(ds, origins, True), ds, origins, **kw)
    fixed = evaluate(_Flip(ds, origins, True), ds, origins, faces_swap_invariant=True, **kw)
    assert bad["surface/in_sdf_mae"] > good["surface/in_sdf_mae"]
    assert fixed["surface/in_sdf_mae"] == pytest.approx(good["surface/in_sdf_mae"], abs=1e-6)
    assert fixed["surface/swapped_frac"] == pytest.approx(1.0)
    assert "surface/swapped_frac" not in good       # only reported when the flag is on
    ok = evaluate(_Flip(ds, origins, False), ds, origins, faces_swap_invariant=True, **kw)
    assert ok["surface/swapped_frac"] == pytest.approx(0.0)
    assert ok["surface/in_sdf_mae"] == pytest.approx(good["surface/in_sdf_mae"], abs=1e-6)


# --------------------------------------------------------------------------- #
# end to end: two training steps
# --------------------------------------------------------------------------- #
def _axis_json(root: str) -> str:
    path = os.path.join(root, "axis.json")
    os.makedirs(root, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"coordinate_order": "xyz", "coordinate_space": "full_resolution",
                   "control_points": [{"z": -100000, "y": AXIS_YX[0], "x": AXIS_YX[1]},
                                      {"z": 100000, "y": AXIS_YX[0], "x": AXIS_YX[1]}]}, fh)
    return path


def test_two_train_steps_with_the_flag_log_the_swapped_fraction(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root, faces=True, rv=True)
    extra = {"steps": 2, "patch": 32, "batch": 2, "accum": 1, "widths": [8, 16, 16, 16, 16],
             "ckpt_every": 2, "num_workers": 0, "stride": 16, "log_every": 1, "warmup": 1,
             "coarse_store": s["coarse"], "fine_store": s["fine"], "augment": False,
             "surface_mode": "faces", "faces_swap_invariant": True,
             "surface_aux": {"gap": 1.0, "gap_radius": 3, "cldice": 1.0},
             "input_radial": False, "axis_path": _axis_json(root),
             "holdout_origins": {"z_frac": 0.5}, "eval_max_crops": 2}
    cfg = parse_config({
        "volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root, "extra": {"train": extra},
    })
    r = run_train(cfg)
    assert r["step"] == 2
    assert "aug/faces_swapped_frac" in r["losses"]
    recs = [json.loads(l) for l in open(os.path.join(root, "train", "log.jsonl"))]
    fr = [rec["aug/faces_swapped_frac"] for rec in recs if "aug/faces_swapped_frac" in rec]
    assert len(fr) == 2
    # a FRACTION, not one of the aug/* sample counts: it must survive the log's int() rule
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in fr)
    assert "surface/swapped_frac" in r["holdout"]
    # the flag is recorded in the checkpoint config
    ck = torch.load(os.path.join(root, "train", "latest.pt"), map_location="cpu", weights_only=False)
    assert ck["config"]["faces_swap_invariant"] is True
    assert ck["config"]["train"]["faces_swap_invariant"] is True


# --------------------------------------------------------------------------- #
# inference / scoring: the two channels are only ever combined symmetrically
# --------------------------------------------------------------------------- #
def test_body_pred_mask_already_ignores_the_face_naming(tmp_path):
    """``dev/eval_region.body_pred_mask`` needs no change for a swap-invariant student.

    Relabelling the two faces is *negate and swap*, and ``(sdf_in > 0) & (sdf_out < 0)`` maps to
    ``(-sdf_out > 0) & (-sdf_in < 0)`` -- the same voxels.  The "symmetric" union one might reach
    for, ``| (sdf_out > 0 > sdf_in)``, is a no-op on any consistent two-face field (it needs the
    out face BELOW the in face), so it would buy nothing and could only fire on garbage.
    """
    eval_region = pytest.importorskip("eval_region")

    from tsm.labels import encode_sdf_u8
    from tsm.volume import BrickWriter

    shape = (16, 8, 8)
    z = np.arange(shape[0], dtype=np.float32)[:, None, None] * np.ones(shape, np.float32)
    s_in, s_out = z - 4.0, z - 10.0           # one sheet: body = 4 < z < 10
    valid = np.full(shape, 255, np.uint8)

    def _store(path, a, b):
        w = BrickWriter(path, ["sdf_in", "sdf_out", "valid"], shape, chunk=8,
                        origin_zyx=(0, 0, 0), voxel_um=2.4, scale=1.0)
        for ci, arr in enumerate((encode_sdf_u8(a, CLIP), encode_sdf_u8(b, CLIP), valid)):
            w.write(ci, 0, 0, 0, np.ascontiguousarray(arr))
        return eval_region.open_store(path)

    lo, hi = [0, 0, 0], list(shape)
    ref = eval_region.body_pred_mask(_store(str(tmp_path / "a.zarr"), s_in, s_out), lo, hi, CLIP)
    assert ref.sum() == 5 * 8 * 8              # z in {5, ..., 9}
    # the SAME sheet emitted under the other face naming -> the same body, voxel for voxel
    flip = _store(str(tmp_path / "b.zarr"), -s_out, -s_in)
    assert np.array_equal(eval_region.body_pred_mask(flip, lo, hi, CLIP), ref)
    # the union one might add, "| (sdf_out > 0 > sdf_in)", selects nothing on a consistent field
    # either way -- it needs the out face below the in face, which no valid pair has
    for st in (_store(str(tmp_path / "c.zarr"), s_in, s_out), flip):
        u = st.read(lo, hi, ["sdf_in", "sdf_out"])
        si, so = (eval_region.decode_sdf_u8(x, CLIP) for x in u)
        assert not ((u[0] != 0) & (u[1] != 0) & (so > 0.0) & (si < 0.0)).any()


def test_the_store_records_the_flag_for_provenance(tmp_path):
    """``tsm.infer`` stamps ``faces_swap_invariant`` on a swap-invariant student's pred.zarr."""
    eval_region = pytest.importorskip("eval_region")
    import zarr

    from tsm.volume import BrickWriter

    path = str(tmp_path / "p.zarr")
    BrickWriter(path, ["sdf_in", "sdf_out", "valid"], (8, 8, 8), chunk=8,
                origin_zyx=(0, 0, 0), voxel_um=2.4, scale=1.0)
    assert eval_region.open_store(path).swap_invariant is False
    zarr.open_array(store=path, mode="r+").attrs["faces_swap_invariant"] = True
    assert eval_region.open_store(path).swap_invariant is True


def test_the_thin_face_union_and_the_thickness_ignore_the_naming():
    """``surface_in1 | surface_out1`` is a union (naming-free by construction) and the two-face
    thickness ``sdf_in - sdf_out`` is invariant under the negate-and-swap relabelling."""
    sdf, _ = _labels()
    sw = swap_faces_target(sdf)
    assert torch.equal(sdf[:, 0:1] - sdf[:, 1:2], sw[:, 0:1] - sw[:, 1:2])
    # the per-face zero sets are exchanged, so their union is the same set
    zs = lambda t: {round(float(v), 3) for v in (t.abs() < 0.5).nonzero()[:, 2].unique().float()}
    assert zs(sdf[:, 0:1]) | zs(sdf[:, 1:2]) == zs(sw[:, 0:1]) | zs(sw[:, 1:2])
