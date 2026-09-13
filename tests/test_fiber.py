"""The fibre-orientation teacher, label channels, student head and loss (CPU, no data)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from tsm.cli import TEACHER_CHANNELS, _teacher_opts
from tsm.config import load_config, parse_config
from tsm.data import FIBER_CHANNELS, Augment, CropDataset, LabelStore, target_keys
from tsm.infer import FIBER_PRED_CHANNELS, activate_heads, n_head_ch, pred_channels, student_build_kw
from tsm.labels import FIBER_CHANNELS as LABEL_FIBER_CHANNELS
from tsm.labels import fine_channels
from tsm.student import build_model, heads_for, load_student_state
from tsm.teachers import TEACHERS, build_teacher, infer_vesuvius_arch
from tsm.train import TRAIN_DEFAULTS, compute_losses, fiber_loss, synthetic_batch, train_opts
from tsm.volume import VolumeReader

from tests.synth import make_synthetic, synthetic_axis, synthetic_axis_tilted

INVENTORY = os.path.join(os.path.dirname(__file__), "data", "fiber_tensor_inventory.json")


def _inventory() -> dict[str, tuple[int, ...]]:
    if not os.path.exists(INVENTORY):
        pytest.skip(f"no fiber tensor inventory at {INVENTORY}")
    d = json.load(open(INVENTORY))["tensors"]
    return {k: tuple(int(i) for i in v["shape"]) for k, v in d.items()}


# --------------------------------------------------------------------------- #
# teacher
# --------------------------------------------------------------------------- #
def test_fiber_registered_like_the_other_vesuvius_teachers():
    spec = TEACHERS["fiber"]
    assert spec.kind == "vesuvius" and spec.target == "labels" and spec.state_key == "ema"
    assert spec.activation == "softmax" and spec.fg_channel is None
    assert spec.patch == (256, 256, 256) and spec.voxel_um == 2.4
    # villa scripts/fiber_5class/dataset.py: percentile_minmax_normalize(p_lo=1, p_hi=99)
    assert spec.normalizer.mode == "percentile_minmax"
    assert (spec.normalizer.lo_pct, spec.normalizer.hi_pct) == (1.0, 99.0)
    assert TEACHER_CHANNELS["fiber"] == ["fiber_bg", "fiber_vt", "fiber_hz", "fiber_ink"]


def test_fiber_arch_inference_matches_the_checkpoint_inventory():
    inv = _inventory()
    spec = infer_vesuvius_arch(inv)
    assert spec.in_channels == 1
    assert spec.features_per_stage == [32, 64, 128, 256, 320, 320, 320]
    assert spec.task_decoders == {}
    assert spec.shared_decoder is not None and spec.shared_decoder.out_channels is None
    assert spec.task_heads == {"labels": 4}  # 4-way softmax head, per the model card
    net = build_teacher("vesuvius", inv)
    assert {k: tuple(v.shape) for k, v in net.state_dict().items()} == inv


def test_fiber_forward_gives_four_softmax_channels():
    """Tiny stand-in for the real net: the registry's activation / select must give 4 channels."""
    from tsm.teachers import ArchSpec, DecoderSpec, VesuviusUNet, apply_activation

    spec = ArchSpec(in_channels=1, features_per_stage=[8, 16], n_blocks_per_stage=[1, 1], strides=[1, 2],
                    shared_decoder=DecoderSpec(None, [1]), task_heads={"labels": 4})
    net = VesuviusUNet(spec).eval()
    t = TEACHERS["fiber"]
    x = t.normalizer(np.random.default_rng(0).integers(0, 256, (16, 16, 16), dtype=np.uint8))[None, None]
    with torch.inference_mode():
        logits = t.select(net(x))
    assert logits.shape == (1, 4, 16, 16, 16)
    prob = apply_activation(logits, t.activation)
    assert torch.allclose(prob.sum(1), torch.ones(1, 16, 16, 16), atol=1e-5)


def test_teacher_stage_writes_four_channels_and_accepts_flip8():
    """`c_out` must be 4 (a softmax teacher without an fg_channel keeps all its channels),
    and the accumulator must not take the 2-class logit-difference shortcut."""
    from tsm.sliding import _acc_channels

    spec = TEACHERS["fiber"]
    channels = TEACHER_CHANNELS["fiber"]
    c_out = 2 if (spec.activation == "softmax" and spec.fg_channel is not None) else len(channels)
    assert c_out == 4
    assert _acc_channels(c_out, spec.activation, spec.fg_channel) == 4
    cfg = load_config("configs/paris4_fiber_teacher.json")
    opts = _teacher_opts(cfg)
    assert opts["models"] == ["fiber"] and opts["tta"] == {"fiber": "none"}
    assert opts["backend"] == "trt" and opts["trt_precision"] == "fp16" and opts["skip_existing"] is True
    assert cfg.out_dir == "/home/forrest/tsm-output/slab_tta"


# --------------------------------------------------------------------------- #
# label store
# --------------------------------------------------------------------------- #
def test_fine_channels_append_fiber_last_so_old_stores_stay_a_prefix():
    assert LABEL_FIBER_CHANNELS == FIBER_CHANNELS == ["fiber_vt", "fiber_hz", "fiber_valid"]
    base = fine_channels()
    assert fine_channels(fiber=True) == base + FIBER_CHANNELS
    assert fine_channels(True, True)[: len(base)] == base
    assert fine_channels(True, True)[-3:] == FIBER_CHANNELS
    assert fine_channels(True, False) == fine_channels(True)
    # the rectoverso block (if built) comes after the fibre block, so both stay prefix-compatible
    assert fine_channels(True, True, True)[: len(fine_channels(True, True))] == fine_channels(True, True)


def test_label_build_copies_the_fiber_teacher_through(tmp_path):
    """`tsm labels` copies fiber.zarr channels 1/2 through and reuses the CT data mask."""
    import zarr

    from tsm.labels import read_box_padded

    shape = (8, 12, 12)
    rng = np.random.default_rng(0)
    vt = rng.integers(0, 256, shape, dtype=np.uint8)
    hz = rng.integers(0, 256, shape, dtype=np.uint8)
    arr = zarr.create_array(store=str(tmp_path / "fiber.zarr"), shape=(4,) + shape,
                            chunks=(1, 4, 4, 4), dtype="uint8", fill_value=0)
    arr[0], arr[1], arr[2], arr[3] = 10, vt, hz, 20
    got = read_box_padded(arr, (0, 0, 0), shape, channels=slice(1, 3))
    assert got.shape == (2,) + shape
    np.testing.assert_array_equal(got[0], vt)
    np.testing.assert_array_equal(got[1], hz)


def test_dataset_reads_the_fiber_channels(tmp_path):
    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32, fiber=True)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16, length=2,
                     augment=False)
    assert ds.fiber and ds.has_fiber_channels
    assert ds.target_keys["fiber_prob"] == 2 and ds.target_keys["fiber_valid"] == 1
    b = ds.load(ds.origins[0])
    assert b["fiber_prob"].shape == (2, 16, 16, 16) and b["fiber_valid"].shape == (1, 16, 16, 16)
    assert 0.0 <= b["fiber_prob"].min() and b["fiber_prob"].max() <= 1.0
    fine = LabelStore(s["fine"])
    z, y, x = (int(v) for v in ds.origins[0])
    np.testing.assert_allclose(b["fiber_prob"][0], fine.read("fiber_vt", z, y, x, 16) / 255.0, atol=1e-6)


def test_dataset_without_fiber_channels_is_unchanged(tmp_path):
    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16, length=2,
                     augment=False)
    assert not ds.fiber and not ds.has_fiber_channels
    assert "fiber_prob" not in ds.target_keys
    assert set(ds.load(ds.origins[0])) >= set(target_keys("medial"))


# --------------------------------------------------------------------------- #
# student head, loss, augmentation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["medial", "faces"])
def test_fiber_head_shape_in_every_surface_mode(mode):
    assert heads_for(mode) == {k: v for k, v in heads_for(mode, True).items() if k != "fiber"}
    assert heads_for(mode, True)["fiber"] == 2
    net = build_model(widths=(8, 16, 32), in_ch=2, surface_mode=mode, fiber=True).eval()
    with torch.inference_mode():
        out = net(torch.zeros(1, 2, 32, 32, 32))
    assert out["fiber"].shape == (1, 2, 32, 32, 32)
    net.train()
    out = net(torch.zeros(1, 2, 32, 32, 32))
    assert len(out["fiber"]) == 2 and out["fiber"][1].shape == (1, 2, 16, 16, 16)  # deep supervision


def test_fiber_loss_is_masked_by_fiber_valid():
    torch.manual_seed(0)
    pred = torch.randn(2, 2, 8, 8, 8)
    prob = torch.rand(2, 2, 8, 8, 8)
    valid = torch.zeros(2, 1, 8, 8, 8)
    valid[:, :, :4] = 1
    base = fiber_loss(pred, prob, valid)
    assert set(base) == {"vt_bce", "vt_dice", "hz_bce", "hz_dice"}
    # changing the target where valid == 0 cannot move the loss
    prob2 = prob.clone()
    prob2[:, :, 4:] = 1.0 - prob2[:, :, 4:]
    for k, v in fiber_loss(pred, prob2, valid).items():
        assert torch.allclose(v, base[k], atol=1e-6), k
    # ... and changing it where valid == 1 must
    prob3 = prob.clone()
    prob3[:, :, :4] = 1.0 - prob3[:, :, :4]
    assert not torch.allclose(fiber_loss(pred, prob3, valid)["vt_bce"], base["vt_bce"])
    # valid == 2 (ignore) counts as masked out, like every other head
    v2 = valid.clone()
    v2[:, :, 4:] = 2
    for k, v in fiber_loss(pred, prob2, v2).items():
        assert torch.allclose(v, base[k], atol=1e-6), k
    # a perfect prediction on the supervised voxels drives both terms to their floor
    perfect = torch.logit(prob.clamp(1e-4, 1 - 1e-4))
    d = fiber_loss(perfect, prob, valid)
    assert float(d["vt_dice"]) < 1e-3 and float(d["hz_dice"]) < 1e-3


def test_fiber_joins_deep_supervision_and_the_loss_weight():
    net = build_model(widths=(8, 16, 32), in_ch=2, surface_mode="faces", fiber=True).train()
    b = synthetic_batch(1, 32, surface_mode="faces", fiber=True)
    out = net(torch.zeros(1, 2, 32, 32, 32))
    _, p1 = compute_losses(out, b, {"fiber": 1.0}, surface_mode="faces")
    _, p2 = compute_losses(out, b, {"fiber": 3.0}, surface_mode="faces")
    assert "fiber" in p1 and p1["fiber"] > 0
    assert p2["total"] - p1["total"] == pytest.approx(2.0 * p1["fiber"], rel=1e-4)
    # deep supervision: the level-1 term is what makes `fiber` bigger than the level-0 parts
    lvl0 = p1["fiber/vt_bce"] + p1["fiber/vt_dice"] + p1["fiber/hz_bce"] + p1["fiber/hz_dice"]
    assert p1["fiber"] > lvl0
    # a batch without fiber targets simply has no fiber term
    b0 = synthetic_batch(1, 32, surface_mode="faces", fiber=False)
    _, p0 = compute_losses(out, b0, surface_mode="faces")
    assert "fiber" not in p0


def test_fiber_targets_move_with_the_grid_like_ink_apart_from_the_class_swap():
    """The two probabilities are resampled exactly like ink (mask-aware trilinear, nearest
    validity).  What is *not* like ink is the class identity: a transform that moves the scroll
    axis swaps the pair (see test_class_targets_are_swapped_only_when_the_axis_moves).  With the
    swap opted out (the pre-fix behaviour) the two fields are identical to the ink rule."""
    torch.manual_seed(0)
    b = synthetic_batch(2, 16, surface_mode="medial", fiber=True)
    b["fiber_valid"] = torch.ones_like(b["fiber_valid"])
    b["ink_valid"] = torch.ones_like(b["ink_valid"])
    b["ink_prob"] = b["fiber_prob"][:, 0:1].clone()  # the ink rule is the reference
    a = Augment("strong", seed=0, fiber_swap=False)
    params = [a.sample_spatial() for _ in range(2)]
    out = a.apply_spatial(b, params)
    assert out["fiber_prob"].shape == b["fiber_prob"].shape
    assert out["fiber_valid"].shape == b["fiber_valid"].shape
    # channel 0 was copied into ink, so it must come back identical to the augmented ink
    torch.testing.assert_close(out["fiber_prob"][:, 0:1], out["ink_prob"], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out["fiber_valid"], out["ink_valid"], atol=0, rtol=0)
    # with the fix on, each sample is either the same field or the swapped pair
    fixed = Augment("strong", seed=0).apply_spatial(b, params)
    for i, p in enumerate(params):
        from tsm.fiber import axis_swap_needed as _swap

        ref = out["fiber_prob"][i].flip(0) if _swap(p.matrix()) else out["fiber_prob"][i]
        torch.testing.assert_close(fixed["fiber_prob"][i], ref, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# inference channels and backward compatibility
# --------------------------------------------------------------------------- #
def test_pred_channels_and_activation_carry_the_fibre_probabilities():
    assert FIBER_PRED_CHANNELS == ["fiber_vt", "fiber_hz"]
    for mode in ("medial", "faces"):
        ch = pred_channels(mode, True)
        assert ch.index("fiber_vt") == ch.index("spare") + 1 and ch.index("fiber_hz") == ch.index("spare") + 2
        assert pred_channels(mode) == [c for c in ch if c not in FIBER_PRED_CHANNELS]
        assert n_head_ch(mode, True) == n_head_ch(mode) + 2
    net = build_model(widths=(8, 16, 32), in_ch=2, surface_mode="faces", fiber=True).eval()
    with torch.inference_mode():
        phys = activate_heads(net(torch.zeros(1, 2, 16, 16, 16)), "faces")
    assert phys.shape[1] == n_head_ch("faces", True)
    assert float(phys[:, -2:].min()) >= 0.0 and float(phys[:, -2:].max()) <= 1.0


def test_old_checkpoints_still_load_into_a_fibre_model_and_back(capsys):
    a = build_model(widths=(8, 16, 32), in_ch=2, surface_mode="faces", fiber=False)
    b = build_model(widths=(8, 16, 32), in_ch=2, surface_mode="faces", fiber=True)
    load_student_state(b, a.state_dict(), "old ckpt")           # head missing -> notice, no raise
    assert "are NOT in the checkpoint" in capsys.readouterr().out
    load_student_state(a, b.state_dict(), "fibre ckpt")         # head unexpected -> notice, no raise
    assert "were dropped" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="outside the heads"):
        load_student_state(b, {k: v for k, v in b.state_dict().items() if "enc" not in k}, "broken")


def test_student_build_kw_reads_the_head_flag():
    assert student_build_kw({})["fiber"] is False
    assert student_build_kw({"train": {"heads": {"fiber": True}}})["fiber"] is True
    assert student_build_kw({"heads": {"fiber": True}})["fiber"] is True


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_train_opts_validates_the_heads_block():
    base = {"volume": {"url": "x", "voxel_um": 2.4}, "region": {"start_zyx": [0, 0, 0], "size_zyx": [8, 8, 8]},
            "out_dir": "/tmp/x"}
    mk = lambda tr: parse_config({**base, "extra": {"train": tr}})  # noqa: E731
    assert train_opts(mk({}))["heads"] == {"fiber": False, "lsd": False}
    assert train_opts(mk({"heads": {"fiber": True}}))["heads"] == {"fiber": True, "lsd": False}
    assert train_opts(mk({}))["loss_weights"]["fiber"] == 1.0
    assert train_opts(mk({"loss_weights": {"fiber": 4.0}}))["loss_weights"]["fiber"] == 4.0
    with pytest.raises(ValueError, match="unknown extra.train.heads keys"):
        train_opts(mk({"heads": {"nope": True}}))
    with pytest.raises(ValueError, match="extra.train.heads must be an object"):
        train_opts(mk({"heads": [1]}))


def test_paris4_faces_fiber_config_enables_the_head():
    cfg = load_config("configs/paris4_faces_fiber.json")
    opts = train_opts(cfg)
    assert opts["heads"]["fiber"] is True
    assert opts["surface_mode"] == "faces"
    assert opts["loss_weights"]["fiber"] == 1.0
    assert opts["fine_store"] == "/home/forrest/tsm-output/slab_faces_fiber/labels/fine.zarr"
    ref = train_opts(load_config("configs/paris4_faces_tta2.json"))
    for k in ("steps", "batch", "accum", "lr", "surface_mode", "input_radial", "ink_pos_weight"):
        assert opts[k] == ref[k]


# --------------------------------------------------------------------------- #
# end-to-end: store -> CropDataset -> DataLoader collation -> augment v2 -> loss
# --------------------------------------------------------------------------- #
def test_fiber_targets_survive_the_whole_train_path(tmp_path):
    """The regression the per-stage unit tests all missed: `CropDataset.load` produced the
    fibre targets but `to_tensors` copied only `TARGET_KEYS`, so every real batch (loader ->
    augment -> `compute_losses`) reached the loss without `fiber_prob` / `fiber_valid` and the
    fibre head trained on nothing (`fiber=0.0000` every step, no `fiber/*` terms in the log)."""
    from torch.utils.data import DataLoader

    from tsm.train import model_input

    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32, fiber=True)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16, length=4,
                     augment=False, fiber=True)
    assert ds.fiber and ds.has_fiber_channels

    # the real per-item path (to_tensors), then the real collation
    item = ds[0]
    assert item["fiber_prob"].shape == (2, 16, 16, 16) and item["fiber_valid"].shape == (1, 16, 16, 16)
    batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0, drop_last=True)))
    assert batch["fiber_prob"].shape == (2, 2, 16, 16, 16) and batch["fiber_valid"].shape == (2, 1, 16, 16, 16)

    # the real augmentation the trainer runs (v2 "strong", here on CPU)
    aug = Augment("strong", seed=0)
    batch, _fired = aug(batch)
    assert "fiber_prob" in batch and "fiber_valid" in batch
    assert batch["fiber_prob"].shape == (2, 2, 16, 16, 16)
    assert float((batch["fiber_valid"] == 1).float().sum()) > 0  # supervision left after the transform

    # and the real loss
    net = build_model(widths=(8, 16, 32), in_ch=int(batch["input"].shape[1]), surface_mode="medial", fiber=True).train()
    out = net(model_input(batch))
    _, parts = compute_losses(out, batch, TRAIN_DEFAULTS["loss_weights"], surface_mode="medial")
    assert "fiber" in parts and parts["fiber"] > 0.0
    assert {"fiber/vt_bce", "fiber/vt_dice", "fiber/hz_bce", "fiber/hz_dice"} <= set(parts)


def test_fiber_head_on_a_store_without_the_channels_warns_and_is_fully_masked(tmp_path, capsys):
    """heads.fiber=true on an old store: keep the keys (fixed shapes) but say so loudly."""
    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16, length=2,
                     augment=False, fiber=True)
    assert "WARNING fiber targets requested" in capsys.readouterr().out
    assert ds.fiber and not ds.has_fiber_channels
    item = ds[0]
    assert "fiber_prob" in item and float(item["fiber_valid"].sum()) == 0.0


# --------------------------------------------------------------------------- #
# fibre orientation: class swap, direction targets, equivariance (2026-09-07)
# --------------------------------------------------------------------------- #
from itertools import permutations  # noqa: E402

from tsm.data import SpatialParams  # noqa: E402
from tsm.fiber import (axis_swap_needed, class_from_direction, derive_direction_targets,  # noqa: E402
                       direction_from_class, fiber_basis, sheet_normal)
from tsm.infer import (FIBER_DIR_PRED_CHANNELS, to_unit, write_fiber_class_channels)  # noqa: E402
from tsm.student import fiber_head, in_channels, input_channels, input_vec_slices  # noqa: E402
from tsm.train import evaluate, fiber_dir_loss  # noqa: E402


def cube_rotations() -> list[torch.Tensor]:
    """The 24 proper signed permutation matrices in (z, y, x)."""
    out = []
    for perm in permutations(range(3)):
        for sz in (1, -1):
            for sy in (1, -1):
                for sx in (1, -1):
                    m = torch.zeros(3, 3)
                    for i, (p, sg) in enumerate(zip(perm, (sz, sy, sx))):
                        m[i, p] = float(sg)
                    if float(torch.det(m)) > 0:
                        out.append(m)
    assert len(out) == 24
    return out


def _plane_crop(n0, p_vt: float, p_hz: float, size: int = 12):
    """A crop whose sdf is the plane with unit normal ``n0`` (so grad sdf == n0 exactly)."""
    n0 = np.asarray(n0, np.float32)
    n0 = n0 / np.linalg.norm(n0)
    g = np.stack(np.meshgrid(*[np.arange(size, dtype=np.float32) - (size - 1) / 2.0] * 3, indexing="ij"))
    sdf = (g * n0.reshape(3, 1, 1, 1)).sum(0)[None]
    ones = np.ones((1, size, size, size), np.float32)
    return {"sdf": sdf, "p_vt": ones * p_vt, "p_hz": ones * p_hz, "valid": ones, "n0": n0}


def test_fiber_basis_is_an_orthonormal_in_plane_triad():
    for n0 in ((0, 1, 0), (0, 0, 1), (0.3, 0.7, -0.2), (0.9, 0.1, 0.0)):
        v0 = np.asarray(n0, np.float32)
        n = torch.tensor((v0 / np.linalg.norm(v0)).astype(np.float32)).view(3, 1, 1, 1).expand(3, 4, 4, 4)
        tv, th, ok = fiber_basis(n.contiguous())
        assert bool(ok.all())
        for a, b in ((tv, th), (tv, n), (th, n)):
            assert float((a * b).sum(0).abs().max()) < 1e-5
        for v in (tv, th):
            torch.testing.assert_close(v.norm(dim=0), torch.ones(4, 4, 4), atol=1e-5, rtol=0)
        # t_v is the axis projected into the sheet: it has the largest z component of the two
        assert float(tv[0].abs().mean()) >= float(th[0].abs().mean())
    # a sheet whose normal IS the scroll axis has no defined vertical/horizontal split
    n = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1, 1).expand(3, 4, 4, 4).contiguous()
    assert not bool(fiber_basis(n)[2].any())


def test_class_swap_rule_is_exact_on_the_24_cube_rotations():
    for m in cube_rotations():
        # the volume z axis lands on column 0 of the matrix; a swap iff it is no longer +-z
        swap = abs(float(m[0, 0])) < 0.5
        assert axis_swap_needed(m) is swap, m
    # axial: a pure z flip is NOT a swap (180 degrees, but the fibre direction has no sign)
    assert axis_swap_needed(torch.diag(torch.tensor([-1.0, -1.0, 1.0]))) is False
    # continuous rotations: nearest class, tie at exactly 45 degrees goes to "no swap"
    from tsm.data import rotation_matrix

    assert axis_swap_needed(rotation_matrix(torch.tensor([0.0, 1.0, 0.0]), math.radians(44.0))) is False
    assert axis_swap_needed(rotation_matrix(torch.tensor([0.0, 1.0, 0.0]), math.radians(46.0))) is True


def test_direction_target_matches_the_class_target_and_inverts():
    c = _plane_crop((0, 1, 0), 1.0, 0.0)
    t = derive_direction_targets(c["p_vt"], c["p_hz"], c["sdf"], c["valid"])
    d = torch.from_numpy(t["fiber_dir"])
    n, _ = sheet_normal(torch.from_numpy(c["sdf"]))
    tv, th, ok = fiber_basis(n)
    assert bool(ok.all())
    torch.testing.assert_close(d.abs(), tv.abs(), atol=1e-5, rtol=0)  # pure vertical -> t_v
    torch.testing.assert_close(torch.from_numpy(t["fiber_str"]), torch.ones_like(d[0:1]), atol=1e-6, rtol=0)
    pv, ph = class_from_direction(d, torch.from_numpy(t["fiber_str"]), tv, th)
    torch.testing.assert_close(pv, torch.ones_like(pv), atol=1e-5, rtol=0)
    torch.testing.assert_close(ph, torch.zeros_like(ph), atol=1e-5, rtol=0)
    # pure horizontal
    c = _plane_crop((0, 1, 0), 0.0, 0.8)
    t = derive_direction_targets(c["p_vt"], c["p_hz"], c["sdf"], c["valid"])
    torch.testing.assert_close(torch.from_numpy(t["fiber_dir"]).abs(), th.abs(), atol=1e-5, rtol=0)
    assert float(t["fiber_str"].max()) == pytest.approx(0.8)


def test_human_bands_override_the_teacher_and_carry_the_band_weight():
    c = _plane_crop((0, 1, 0), 0.9, 0.0)          # the teacher says "vertical" everywhere
    band = np.zeros_like(c["valid"])
    band[:, :4] = 1   # human: horizontal
    band[:, 4:8] = 3  # human: exclude
    t = derive_direction_targets(c["p_vt"], c["p_hz"], c["sdf"], c["valid"], band=band, band_weight=5.0)
    n, _ = sheet_normal(torch.from_numpy(c["sdf"]))
    tv, th, _ = fiber_basis(n)
    d = torch.from_numpy(t["fiber_dir"])
    torch.testing.assert_close(d[:, :4].abs(), th[:, :4].abs(), atol=1e-5, rtol=0)   # human wins
    torch.testing.assert_close(d[:, 8:].abs(), tv[:, 8:].abs(), atol=1e-5, rtol=0)   # teacher elsewhere
    assert float(t["fiber_valid"][0, 4:8].max()) == 0.0                              # exclude -> dropped
    assert float(t["fiber_weight"][0, :4].min()) == 5.0 and float(t["fiber_weight"][0, 8:].max()) == 1.0
    assert float(t["fiber_str"][0, :4].min()) == 1.0                                 # a human label is certain


@pytest.mark.parametrize("n0", [(0, 1, 0), (0.2, 0.9, -0.3)])
def test_direction_and_class_rules_agree_under_every_cube_rotation(n0):
    """The equivariance test: rotate the crop, and (a) the derived direction target rotates with
    it, (b) re-deriving it from the *swapped* class channels gives the same answer."""
    c = _plane_crop(n0, 0.9, 0.2, size=12)
    t0 = derive_direction_targets(c["p_vt"], c["p_hz"], c["sdf"], c["valid"])
    d0 = torch.from_numpy(t0["fiber_dir"])[None]
    aug = Augment("none", seed=0)
    P = 12
    for m in cube_rotations():
        batch = {
            "input": torch.zeros(1, 2, P, P, P),
            "surface_sdf": torch.from_numpy(c["sdf"])[None],
            "surface_valid": torch.ones(1, 1, P, P, P),
            "ink_prob": torch.zeros(1, 1, P, P, P), "ink_valid": torch.ones(1, 1, P, P, P),
            "winding": torch.zeros(1, 6, P, P, P), "winding_conf": torch.zeros(1, 1, P, P, P),
            "winding_valid": torch.zeros(1, 1, P, P, P),
            "fiber_prob": torch.from_numpy(np.concatenate([c["p_vt"], c["p_hz"]]))[None],
            "fiber_dir": d0.clone(), "fiber_str": torch.from_numpy(t0["fiber_str"])[None],
            "fiber_weight": torch.from_numpy(t0["fiber_weight"])[None],
            "fiber_valid": torch.ones(1, 1, P, P, P),
        }
        out = aug.apply_spatial(batch, [SpatialParams(rot90=m)])
        core = (slice(None), slice(2, -2), slice(2, -2), slice(2, -2))
        # (a) the stored direction target moved as a vector: R d0, resampled
        rot_d = torch.einsum("ij,bjdhw->bidhw", m, d0)
        got = out["fiber_dir"]
        cos = (got * _resample_like(rot_d, m)).sum(1).abs()
        assert float(cos[core].min()) > 1 - 1e-4, m
        # (b) the class-swap rule agrees with the direction rule: the class implied by the
        # ROTATED direction, read against the basis of the rotated crop, is the swapped class.
        # The two encodings agree exactly at the class level; the full vector cannot agree in
        # general -- two numbers cannot carry an in-plane direction, which is the bug the
        # direction mode fixes.
        n2, _ = sheet_normal(out["surface_sdf"][:, 0:1])
        tv2, th2, ok2 = fiber_basis(n2)
        sel = ok2[core] & (out["fiber_str"][core] > 0.5)
        if not bool(sel.any()):
            continue  # the rotated sheet normal IS the scroll axis: no vt/hz split exists
        pv2, ph2 = class_from_direction(got, out["fiber_str"], tv2, th2)
        want_vt = bool(out["fiber_prob"][0, 0].mean() > out["fiber_prob"][0, 1].mean())
        assert bool(((pv2[core] > ph2[core]) == want_vt)[sel].all()), m


def _resample_like(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """The same grid resampling ``Augment.apply_spatial`` applies, for a field already rotated
    componentwise (cube rotations are exact grid permutations, so this is a pure reindexing)."""
    aug = Augment("none", seed=0)
    B, C = v.shape[0], v.shape[1]
    grid, _oob, _L = aug._grid([SpatialParams(rot90=m)], v.shape[2:], v.device)
    return torch.nn.functional.grid_sample(v, grid, mode="nearest", padding_mode="border", align_corners=False)


def test_class_targets_are_swapped_only_when_the_axis_moves():
    P = 8
    prob = torch.zeros(1, 2, P, P, P)
    prob[:, 0] = 1.0  # pure vertical
    base = {
        "input": torch.zeros(1, 2, P, P, P),
        "surface_sdf": torch.zeros(1, 1, P, P, P), "surface_valid": torch.ones(1, 1, P, P, P),
        "ink_prob": torch.zeros(1, 1, P, P, P), "ink_valid": torch.ones(1, 1, P, P, P),
        "winding": torch.zeros(1, 6, P, P, P), "winding_conf": torch.zeros(1, 1, P, P, P),
        "winding_valid": torch.zeros(1, 1, P, P, P),
        "fiber_prob": prob, "fiber_valid": torch.ones(1, 1, P, P, P),
        "fiber_band": torch.full((1, 1, P, P, P), 2.0),  # human "vt"
    }
    aug = Augment("none", seed=0)
    for m in cube_rotations():
        out = aug.apply_spatial({k: v.clone() for k, v in base.items()}, [SpatialParams(rot90=m)])
        swapped = abs(float(m[0, 0])) < 0.5
        vt = float(out["fiber_prob"][0, 0].mean())
        assert vt == pytest.approx(0.0 if swapped else 1.0, abs=1e-4), m
        assert float(out["fiber_band"].max()) == (1.0 if swapped else 2.0)
    # the opt-out reproduces the bug: the classes stay put whatever the transform
    aug_bug = Augment("none", seed=0, fiber_swap=False)
    m = [r for r in cube_rotations() if abs(float(r[0, 0])) < 0.5][0]
    out = aug_bug.apply_spatial({k: v.clone() for k, v in base.items()}, [SpatialParams(rot90=m)])
    assert float(out["fiber_prob"][0, 0].mean()) == pytest.approx(1.0, abs=1e-4)


def test_fiber_direction_head_and_loss():
    assert fiber_head("direction") == {"fiber": 4}
    assert heads_for("faces", True, "direction")["fiber"] == 4
    net = build_model(widths=(8, 16, 32), in_ch=2, surface_mode="faces", fiber=True, fiber_mode="direction").eval()
    with torch.inference_mode():
        out = net(torch.zeros(1, 2, 32, 32, 32))
    assert out["fiber"].shape == (1, 4, 32, 32, 32)
    # a perfect prediction (either sign: the target is an axis) gives zero direction loss
    B, P = 2, 8
    d = F.normalize(torch.randn(B, 3, P, P, P), dim=1)
    s = torch.rand(B, 1, P, P, P)
    valid = torch.ones(B, 1, P, P, P)
    pred = torch.cat([d * -1.0, torch.logit(s.clamp(1e-4, 1 - 1e-4))], dim=1)
    terms = fiber_dir_loss(pred, d, s, valid)
    assert float(terms["dir_cos"]) < 1e-6
    perp = F.normalize(torch.randn(B, 3, P, P, P), dim=1)
    perp = F.normalize(perp - (perp * d).sum(1, keepdim=True) * d, dim=1)
    assert float(fiber_dir_loss(torch.cat([perp, pred[:, 3:]], 1), d, s, valid)["dir_cos"]) > 0.9
    # masked by fiber_valid == 1 ...
    assert float(fiber_dir_loss(torch.cat([perp, pred[:, 3:]], 1), d, s, torch.zeros_like(valid))["dir_cos"]) == 0.0
    # ... and weighted per voxel (the human bands weigh more)
    w = torch.ones_like(valid)
    w[:, :, :4] = 5.0
    bad = perp.clone()
    bad[:, :, 4:] = d[:, :, 4:]  # only the heavy half is wrong
    a = float(fiber_dir_loss(torch.cat([bad, pred[:, 3:]], 1), d, s, valid)["dir_cos"])
    b = float(fiber_dir_loss(torch.cat([bad, pred[:, 3:]], 1), d, s, valid, w)["dir_cos"])
    assert b > a


import torch.nn.functional as F  # noqa: E402,E401
import math  # noqa: E402


def test_direction_mode_survives_the_whole_train_path(tmp_path):
    from torch.utils.data import DataLoader

    from tsm.train import model_input

    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32,
                       faces=True, fiber=True, rv=True)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16, length=4,
                     augment=False, fiber=True, surface_mode="faces", fiber_mode="direction",
                     input_axis=True, axis=synthetic_axis())
    assert ds.fiber_band and ds.target_keys["fiber_dir"] == 3
    item = ds[0]
    assert item["input"].shape[0] == in_channels(False, True) == 5
    batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0, drop_last=True)))
    aug = Augment("strong", seed=0, input_vec=input_vec_slices(False, True))
    batch, _fired = aug(batch)
    assert batch["fiber_dir"].shape == (2, 3, 16, 16, 16)
    net = build_model(widths=(8, 16, 32), in_ch=int(batch["input"].shape[1]), surface_mode="faces",
                      fiber=True, fiber_mode="direction").train()
    out = net(model_input(batch))
    _, parts = compute_losses(out, batch, TRAIN_DEFAULTS["loss_weights"], surface_mode="faces")
    assert {"fiber/dir_cos", "fiber/str_bce"} <= set(parts) and parts["fiber"] > 0.0
    # holdout metrics: the angular error and the teacher-independent human-band numbers
    res = evaluate(net, ds, ds.origins[:2], surface_mode="faces")
    assert "fiber/angle_deg" in res and 0.0 <= res["fiber/angle_deg"] <= 180.0
    assert "fiber/vt_auprc" in res
    assert "fiber/band_class_acc" in res and 0.0 <= res["fiber/band_class_acc"] <= 1.0


def test_direction_prediction_channels_and_derived_classes(tmp_path):
    assert FIBER_DIR_PRED_CHANNELS == ["fiber_dz", "fiber_dy", "fiber_dx", "fiber_strength"]
    ch = pred_channels("faces", True, "direction")
    assert ch[12:16] == FIBER_DIR_PRED_CHANNELS and ch[-2:] == ["fiber_vt", "fiber_hz"]
    assert n_head_ch("faces", True, "direction") == 16
    # the head channels are the first n_head_ch of the store, so the sliding engine's view is contiguous
    assert ch[:n_head_ch("faces", True, "direction")][-1] == "fiber_strength"
    # activation: unit direction + sigmoid strength, encoded like a normal / a probability
    raw = {"surface": torch.randn(1, 3, 4, 4, 4), "ink": torch.randn(1, 1, 4, 4, 4),
           "winding": torch.randn(1, 8, 4, 4, 4), "fiber": torch.randn(1, 4, 4, 4, 4)}
    phys = activate_heads(raw, "faces", "direction")
    assert phys.shape[1] == 16
    torch.testing.assert_close(phys[:, 12:15].norm(dim=1), torch.ones(1, 4, 4, 4), atol=1e-5, rtol=0)
    u = to_unit(phys, 20.0, "faces", "direction")
    torch.testing.assert_close(u[:, 12:15], (phys[:, 12:15] + 1) * 0.5, atol=1e-6, rtol=0)
    torch.testing.assert_close(u[:, 15:16], phys[:, 15:16], atol=1e-6, rtol=0)

    # the derived vt/hz channels, written post hoc from the direction and grad sdf_in
    from tsm.volume import BrickWriter

    P = 16
    c = _plane_crop((0, 1, 0), 1.0, 0.0, size=P)
    t = derive_direction_targets(c["p_vt"], c["p_hz"], c["sdf"], c["valid"])
    path = str(tmp_path / "pred.zarr")
    w = BrickWriter(path, ch, (P, P, P), chunk=P, origin_zyx=(0, 0, 0), voxel_um=2.4, scale=1.0)
    from tsm.data import encode_sdf, encode_signed

    data = {name: np.zeros((P, P, P), np.uint8) for name in ch}
    data["sdf_in"] = encode_sdf(c["sdf"][0])
    for i, name in enumerate(("fiber_dz", "fiber_dy", "fiber_dx")):
        data[name] = encode_signed(t["fiber_dir"][i])
    data["fiber_strength"] = np.full((P, P, P), 255, np.uint8)
    for ci, name in enumerate(ch):
        w.write(ci, 0, 0, 0, np.ascontiguousarray(data[name]))
    info = write_fiber_class_channels(path, 20.0, brick=P)
    assert info["sdf_channel"] == "sdf_in" and info["basis_defined_frac"] > 0.9
    import zarr as _zarr

    arr = _zarr.open_array(store=path, mode="r")
    vt = np.asarray(arr[ch.index("fiber_vt")])[2:-2, 2:-2, 2:-2]
    hz = np.asarray(arr[ch.index("fiber_hz")])[2:-2, 2:-2, 2:-2]
    assert vt.min() > 250 and hz.max() < 5  # a pure vertical direction decodes back to vt = 1


def test_direction_mode_is_the_default_and_old_checkpoints_still_load():
    # 2026-09-12: the defaults flipped to the direction head + the real axis input (plan section 2).
    assert TRAIN_DEFAULTS["fiber_mode"] == "direction" and TRAIN_DEFAULTS["fiber_swap_fix"] is True
    assert TRAIN_DEFAULTS["input_axis"] is True and TRAIN_DEFAULTS["axis_tangent"] is True
    assert heads_for("faces", True)["fiber"] == 2  # heads_for still defaults to the class head
    # a checkpoint config from before fiber_mode / input_axis existed
    kw = student_build_kw({"surface_mode": "faces", "heads": {"fiber": True}, "widths": [8, 16, 32]})
    assert kw["fiber_mode"] == "class" and kw["in_ch"] == 2
    net = build_model(**kw)
    assert net.heads["fiber"] == 2
    old = build_model(widths=(8, 16, 32), in_ch=2, surface_mode="faces", fiber=True)
    load_student_state(net, old.state_dict(), what="old checkpoint")


def test_new_fiber_configs():
    cfg = load_config("configs/paris4_faces_rvfw_dir.json")
    o = train_opts(cfg)
    assert o["fiber_mode"] == "direction" and o["heads"]["fiber"] and o["input_axis"]
    assert cfg.out_dir.endswith("slab_faces_rvfw_dir")
    ab = load_config("configs/ablate_fiber.json")
    ao = train_opts(ab)
    assert ao["fiber_mode"] == "class" and ao["fiber_swap_fix"] is False  # the buggy baseline
    from tsm.ablate import ablate_opts

    op = ablate_opts(ab)
    assert op["steps"] == 6000 and op["include_base"] is True
    assert set(op["variants"]) == {"class_fixed", "direction"}
    assert op["variants"]["class_fixed"]["fiber_swap_fix"] is True
    assert op["out_dir"].endswith("slab_faces_rvfw/ablation_fiber")


# --------------------------------------------------------------------------- #
# per-voxel axis field: "vertical" is the LOCAL umbilicus tangent (2026-09-12)
# --------------------------------------------------------------------------- #
def test_fiber_basis_accepts_a_per_voxel_axis_field():
    from tsm.labels import axis_tangent_field

    P = 6
    n0 = np.array([0.0, 1.0, 0.0], np.float32)
    n = torch.from_numpy(n0).view(3, 1, 1, 1).expand(3, P, P, P).contiguous()
    a_np = axis_tangent_field(synthetic_axis_tilted(0.3, -0.2), (0, 0, 0), (P, P, P))
    a = torch.from_numpy(a_np)
    assert a.shape == (3, P, P, P)
    tv, th, ok = fiber_basis(n, a)
    assert bool(ok.all())
    for u, v in ((tv, th), (tv, n), (th, n)):
        assert float((u * v).sum(0).abs().max()) < 1e-5
    torch.testing.assert_close(tv.norm(dim=0), torch.ones(P, P, P), atol=1e-5, rtol=0)
    # t_v is the axis projected into the sheet, so it lies in span(a, n) and is NOT the
    # constant-axis answer any more
    tv_c, th_c, _ = fiber_basis(n)
    assert float((tv * tv_c).sum(0).abs().min()) < 0.999
    proj = a - (a * n).sum(0, keepdim=True) * n          # the axis projected into the sheet
    proj = proj / proj.norm(dim=0, keepdim=True)
    torch.testing.assert_close(tv, proj, atol=1e-5, rtol=0)
    # a batched field broadcasts against a batched normal
    tvb, _, okb = fiber_basis(n[None].expand(2, 3, P, P, P), a[None].expand(2, 3, P, P, P))
    assert tvb.shape == (2, 3, P, P, P) and bool(okb.all())
    torch.testing.assert_close(tvb[0], tv, atol=1e-6, rtol=0)
    # a constant (3,) axis still works, and a bad shape is rejected
    torch.testing.assert_close(fiber_basis(n, (1.0, 0.0, 0.0))[0], tv_c, atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        fiber_basis(n, torch.zeros(4, P, P))


def test_direction_targets_use_the_axis_they_are_given():
    """The bug this fixes: the targets were always built against (1, 0, 0), so a tilted scroll
    axis made "vertical" mean the wrong thing everywhere."""
    c = _plane_crop((0, 1, 0), 1.0, 0.0)
    ax = np.asarray([1.0, 0.3, -0.2], np.float32)
    ax = ax / np.linalg.norm(ax)
    fld = np.broadcast_to(ax[:, None, None, None], (3,) + c["sdf"].shape[1:]).copy()
    t_const = derive_direction_targets(c["p_vt"], c["p_hz"], c["sdf"], c["valid"])
    t_field = derive_direction_targets(c["p_vt"], c["p_hz"], c["sdf"], c["valid"], axis=fld)
    d_const, d_field = torch.from_numpy(t_const["fiber_dir"]), torch.from_numpy(t_field["fiber_dir"])
    cos = (d_const * d_field).sum(0).abs()
    assert float(cos.max()) < 0.999            # a different "vertical"
    n, _ = sheet_normal(torch.from_numpy(c["sdf"]))
    tv, _, ok = fiber_basis(n, torch.from_numpy(fld))
    torch.testing.assert_close(d_field.abs(), tv.abs(), atol=1e-5, rtol=0)
    assert bool(ok.all())


def test_direction_targets_are_equivariant_under_the_cube_rotations_with_a_real_axis():
    """Rotate the crop *and its axis field*: the derived direction must rotate with them
    (exactly for the 24 cube rotations, to <1 degree for a random SO(3) rotation)."""
    from tsm import equivariance as eq

    P = 16
    c = _plane_crop((0.2, 0.9, -0.3), 0.9, 0.2, size=P)
    ax = np.asarray([1.0, 0.3, -0.2], np.float32)
    ax /= np.linalg.norm(ax)
    fld = np.broadcast_to(ax[:, None, None, None], (3, P, P, P)).copy()
    t0 = derive_direction_targets(c["p_vt"], c["p_hz"], c["sdf"], c["valid"], axis=fld)
    d0 = t0["fiber_dir"]
    core = (slice(2, -2),) * 3
    for m in cube_rotations():
        g = np.round(m.numpy()).astype(np.int64)
        t1 = derive_direction_targets(
            np.asarray(eq.apply(c["p_vt"], g)), np.asarray(eq.apply(c["p_hz"], g)),
            np.asarray(eq.apply(c["sdf"], g)), np.asarray(eq.apply(c["valid"], g)),
            axis=np.asarray(eq.apply_vector(fld, g)))
        want = np.asarray(eq.apply_vector(d0, g))
        cos = np.abs((t1["fiber_dir"] * want).sum(0))[core]
        assert cos.min() > 1 - 1e-4, m
    # a random SO(3) rotation: the same, within interpolation error
    R = eq.random_rotation_matrix(3)
    t1 = derive_direction_targets(
        np.asarray(eq.apply_rotation(c["p_vt"], R)), np.asarray(eq.apply_rotation(c["p_hz"], R)),
        np.asarray(eq.apply_rotation(c["sdf"], R)), np.ones_like(c["valid"]),
        axis=np.asarray(eq.apply_vector(fld, R)))
    want = np.asarray(eq.apply_vector(d0, R))
    want /= np.maximum(np.linalg.norm(want, axis=0, keepdims=True), 1e-6)
    inner = (slice(4, -4),) * 3
    cos = np.abs((t1["fiber_dir"] * want).sum(0))[inner]
    assert cos.min() > 0.99


def test_sheet_normal_can_prefer_the_fallback():
    """``prefer_fallback`` (for the body surface mode, whose SDF gradient is degenerate on the
    medial ridge); the default is unchanged."""
    c = _plane_crop((0, 1, 0), 1.0, 0.0)
    sdf = torch.from_numpy(c["sdf"])
    fb = torch.zeros_like(sdf.expand(3, *sdf.shape[1:]).contiguous())
    fb[2] = 1.0  # a unit fallback normal along +x, everywhere
    n_def, ok_def = sheet_normal(sdf, fb)
    torch.testing.assert_close(n_def[1].abs(), torch.ones_like(n_def[1]), atol=1e-5, rtol=0)
    n_pref, ok_pref = sheet_normal(sdf, fb, prefer_fallback=True)
    torch.testing.assert_close(n_pref, fb, atol=1e-6, rtol=0)
    assert bool(ok_def.all()) and bool(ok_pref.all())
    # without a fallback the flag changes nothing
    torch.testing.assert_close(sheet_normal(sdf, None, prefer_fallback=True)[0], sheet_normal(sdf)[0])


def test_class_mode_still_trains_end_to_end(tmp_path):
    """Legacy ablation path: the class head must keep working now that direction is the default."""
    from torch.utils.data import DataLoader

    from tsm.train import model_input

    s = make_synthetic(str(tmp_path), fine_shape=(32, 64, 64), fine_origin=(0, 0, 0), chunk=32,
                       faces=True, fiber=True)
    ds = CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], None, patch=16, stride=16,
                     length=4, augment=False, fiber=True, surface_mode="faces", fiber_mode="class")
    assert ds.axis is None  # the class mode needs no umbilicus at all
    batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0, drop_last=True)))
    batch, _ = Augment("strong", seed=0)(batch)
    net = build_model(widths=(8, 16, 32), in_ch=int(batch["input"].shape[1]), surface_mode="faces",
                      fiber=True, fiber_mode="class").train()
    loss, parts = compute_losses(net(model_input(batch)), batch, TRAIN_DEFAULTS["loss_weights"],
                                 surface_mode="faces")
    loss.backward()
    assert parts["fiber"] > 0.0 and {"fiber/vt_bce", "fiber/hz_bce"} <= set(parts)


def test_write_fiber_class_channels_can_use_the_real_axis_tangent(tmp_path):
    """The inference-side derivation takes the same axis the targets were built against."""
    import zarr as _zarr

    from tsm.data import encode_prob, encode_sdf, encode_signed
    from tsm.infer import pred_channels
    from tsm.volume import BrickWriter

    P = 16
    ch = pred_channels("faces", True, "direction")
    path = str(tmp_path / "pred.zarr")
    org = (0, 40, 40)
    w = BrickWriter(path, ch, (P, P, P), chunk=P, origin_zyx=org, voxel_um=2.4, scale=1.0)
    ax = synthetic_axis_tilted(0.3, -0.2)
    from tsm.labels import axis_tangent_field

    a = axis_tangent_field(ax, (0, 40, 40), (P, P, P))
    n0 = np.zeros((3, P, P, P), np.float32)
    n0[1] = 1.0  # sheet normal along +y
    tv, _, ok = fiber_basis(torch.from_numpy(n0), torch.from_numpy(a))
    assert bool(ok.all())
    g = np.stack(np.meshgrid(*[np.arange(P, dtype=np.float32) - (P - 1) / 2.0] * 3, indexing="ij"))
    data = {c: np.zeros((P, P, P), np.uint8) for c in ch}
    data["sdf_in"] = encode_sdf((g * n0[:, :1, :1, :1].reshape(3, 1, 1, 1)).sum(0), 20.0)
    for i, c in enumerate(("fiber_dz", "fiber_dy", "fiber_dx")):
        data[c] = encode_signed(tv[i].numpy())          # a purely "vertical" prediction
    data["fiber_strength"] = encode_prob(np.ones((P, P, P), np.float32))
    for ci, name in enumerate(ch):
        w.write(ci, *org, np.ascontiguousarray(data[name]))
    info = write_fiber_class_channels(path, 20.0, brick=P, axis=ax)
    assert info["axis_tangent"] is True and info["basis_defined_frac"] > 0.9
    arr = _zarr.open_array(store=path, mode="r")
    core = (slice(2, -2),) * 3
    vt = np.asarray(arr[ch.index("fiber_vt")])[core]
    hz = np.asarray(arr[ch.index("fiber_hz")])[core]
    assert vt.min() > 250 and hz.max() < 5
    # the same store read against the constant (1, 0, 0) axis decodes a mixture instead
    info2 = write_fiber_class_channels(path, 20.0, brick=P)
    assert info2["axis_tangent"] is False
    arr2 = _zarr.open_array(store=path, mode="r")
    assert np.asarray(arr2[ch.index("fiber_hz")])[core].max() > 20
