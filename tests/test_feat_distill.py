"""Training-only feature distillation: student encoder features -> frozen teacher encoder features."""
import json
import os

import pytest
import torch

from synth import make_synthetic
from tsm.config import parse_config
from tsm.student import FeatureProjector, TSMNet, feature_distill_loss
from tsm.teachers import TEACHERS, ArchSpec, ResidualEncoder, encoder_bytes_estimate, encoder_channels, encoder_features, encoder_strides
from tsm.train import FeatDistiller, feat_distill_opts, model_input, run_train, synthetic_batch

W = (8, 16, 16, 16, 16)  # tiny student widths, 4 stride-2 stages like the real (32, 64, 128, 256, 256)
T_FEATS = [8, 16, 24, 40, 48]  # tiny "teacher" encoder widths (stage k at /2^k)


class TinyTeacher(torch.nn.Module):
    """Stands in for VesuviusUNet: only ``shared_encoder`` is used by the distiller."""

    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        spec = ArchSpec(in_channels=1, features_per_stage=list(T_FEATS), n_blocks_per_stage=[1] * 5, strides=[1, 2, 2, 2, 2])
        self.shared_encoder = ResidualEncoder(spec)


def tiny_loader(name, models_dir, device="cpu", dtype=torch.float32):
    m = TinyTeacher(seed={"recto": 1, "ink": 2}[name]).to(device=device, dtype=dtype).eval()
    m.requires_grad_(False)
    return m, TEACHERS[name]


def test_return_features_shapes_and_strides():
    net = TSMNet(widths=W)
    x = torch.randn(1, 2, 32, 32, 32)
    net.train()
    out, feats = net(x, return_features=True)
    assert set(out) == {"surface", "ink", "winding"} and isinstance(out["surface"], list)
    assert list(feats) == ["enc0", "enc1", "enc2", "enc3", "enc4"]
    for k, c in enumerate(W):
        assert tuple(feats[f"enc{k}"].shape) == (1, c, 32 >> k, 32 >> k, 32 >> k), k
    assert net.encoder_channels() == {f"enc{k}": c for k, c in enumerate(W)}
    net.eval()
    with torch.no_grad():
        out_e, feats_e = net(x, return_features=True)
        plain = net(x)
    assert isinstance(out_e["surface"], torch.Tensor) and set(feats_e) == set(feats)
    torch.testing.assert_close(plain["surface"], out_e["surface"])


def test_teacher_encoder_helpers():
    t, spec = tiny_loader("recto", None)
    assert encoder_channels(t) == T_FEATS and encoder_strides(t) == [1, 2, 4, 8, 16]
    x = torch.randn(1, 1, 32, 32, 32)
    f = encoder_features(t, x)
    assert [tuple(a.shape) for a in f] == [(1, c, 32 >> k, 32 >> k, 32 >> k) for k, c in enumerate(T_FEATS)]
    assert all(not a.requires_grad for a in f)
    e = encoder_bytes_estimate(t, 128, 2)
    assert e["total"] == e["params"] + e["activations"] and e["activations"] > 2 * 8 * 128 ** 3 * 2
    assert spec.normalizer.mode == "zscore_instance" and TEACHERS["ink"].normalizer.mode == "percentile_minmax"


def test_projector_maps_channels_and_loss_zero_through_identity():
    pairs = [(3, 3), (4, 4)]
    proj = FeatureProjector(pairs, W, T_FEATS)
    feats = {f"enc{k}": torch.randn(2, c, 32 >> k, 32 >> k, 32 >> k) for k, c in enumerate(W)}
    p = proj(feats)
    assert tuple(p[(3, 3)].shape) == (2, 40, 4, 4, 4) and tuple(p[(4, 4)].shape) == (2, 48, 2, 2, 2)
    assert sum(q.numel() for q in proj.parameters()) == 16 * 40 + 16 * 48 + 2 * (40 + 48)  # conv (no bias) + GN affine
    # identity projector (equal widths, no norm): student feature == teacher feature -> both losses ~0
    ident = FeatureProjector([(3, 3)], W, [0, 0, 0, 16], norm=False)
    with torch.no_grad():
        ident.proj["s3_t3"][0].weight.copy_(torch.eye(16).view(16, 16, 1, 1, 1))
        ident.proj["s3_t3"][0].bias.zero_()
    teacher = [None, None, None, feats["enc3"].clone()]
    assert feature_distill_loss(ident(feats), teacher, "cosine").item() < 1e-5
    assert feature_distill_loss(ident(feats), teacher, "mse").item() < 1e-5
    assert feature_distill_loss(ident(feats), [None, None, None, -feats["enc3"]], "cosine").item() == pytest.approx(2.0, abs=1e-4)
    # stride mismatch: teacher map at a different spatial size is trilinearly resampled
    coarse = torch.nn.functional.avg_pool3d(feats["enc3"], 2)
    l = feature_distill_loss(ident(feats), [None, None, None, coarse], "cosine")
    assert torch.isfinite(l) and 0 < l.item() < 2
    with pytest.raises(ValueError):
        feature_distill_loss(ident(feats), teacher, "l1")


def test_feat_distill_opts_validation():
    fd = feat_distill_opts({"enabled": True})
    assert fd["teachers"] == ["recto", "ink"] and fd["pairs"] == [[3, 3], [4, 4]] and fd["weight"] == 0.1
    assert feat_distill_opts(None)["enabled"] is False
    with pytest.raises(ValueError):
        feat_distill_opts({"loss": "l1"})
    with pytest.raises(ValueError):
        feat_distill_opts({"bogus": 1})
    with pytest.raises(ValueError):
        feat_distill_opts({"enabled": True, "teachers": []})


def test_distiller_per_crop_normalization_and_loss_decreases():
    torch.manual_seed(0)
    fd = feat_distill_opts({"enabled": True, "weight": 1.0, "pairs": [[3, 3], [4, 4]]})
    d = FeatDistiller(fd, W, "cpu", loader=tiny_loader)
    assert set(d.encoders) == {"recto", "ink"} and all(not p.requires_grad for e in d.encoders.values() for p in e.parameters())
    lines = d.describe(W)
    assert "recto: student enc3 (16 ch, /8) -> teacher stage 3 (40 ch, /8)" in lines
    assert "ink: student enc4 (16 ch, /16) -> teacher stage 4 (48 ch, /16)" in lines
    b = synthetic_batch(2, 32, seed=7)
    ct01 = b["input"][:, 0]
    # per-crop teacher normalization: recto is z-scored per sample, ink min-max per sample
    xr = d.teacher_input("recto", ct01)
    assert tuple(xr.shape) == (2, 1, 32, 32, 32)
    assert torch.allclose(xr.mean(dim=(2, 3, 4)), torch.zeros(2), atol=1e-4) and torch.allclose(xr.std(dim=(2, 3, 4), unbiased=False), torch.ones(2), atol=1e-3)
    xi = d.teacher_input("ink", ct01)
    assert xi.min() >= 0 and xi.max() <= 1 and xi.amax(dim=(2, 3, 4)).min() > 0.99
    net = TSMNet(widths=W).train()
    opt = torch.optim.AdamW([{"params": net.parameters()}, {"params": d.parameters()}], lr=3e-3)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        _, feats = net(model_input(b), return_features=True)
        loss, parts = d(feats, ct01)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert set(parts) == {"feat_recto", "feat_ink", "feat"} and all(0 <= parts[k] <= 2 for k in ("feat_recto", "feat_ink"))
    assert losses[-1] < 0.8 * losses[0], losses
    # every=2 -> active on even optimizer steps only
    d.every = 2
    assert d.active(2) and not d.active(3)


def _cfg(root, ct_url, extra):
    return parse_config({
        "volume": {"url": ct_url, "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root,
        "extra": {"train": extra},
    })


def test_run_train_with_feat_distill_logs_and_checkpoints(tmp_path, monkeypatch):
    import tsm.train as trainmod

    monkeypatch.setattr(trainmod, "feat_teacher_loader", tiny_loader)
    root = str(tmp_path / "out")
    s = make_synthetic(root)
    extra = {"steps": 2, "patch": 32, "batch": 1, "accum": 1, "widths": list(W), "ckpt_every": 1,
             "num_workers": 0, "stride": 16, "log_every": 1, "warmup": 1, "coarse_store": s["coarse"],
             "fine_store": s["fine"],
             "feat_distill": {"enabled": True, "teachers": ["recto", "ink"], "pairs": [[3, 3], [4, 4]], "weight": 0.1}}
    cfg = _cfg(root, s["ct"], extra)
    r = run_train(cfg, dry_run=True)
    assert "feat_recto" in r["losses"] and "feat_ink" in r["losses"] and "feat_time_s" in r["losses"]
    r1 = run_train(cfg)
    assert r1["step"] == 2
    lines = [json.loads(l) for l in open(os.path.join(root, "train", "log.jsonl"))]
    assert [l["step"] for l in lines] == [1, 2]
    for l in lines:
        assert 0 < l["feat_recto"] <= 2 and 0 < l["feat_ink"] <= 2
        assert l["feat"] == pytest.approx(0.1 * (l["feat_recto"] + l["feat_ink"]) / 2, abs=1e-4)
        assert l["total"] > l["surface"] + l["ink"] + l["winding"]  # feat term is in the total
    ck = torch.load(os.path.join(root, "train", "latest.pt"), weights_only=False)
    assert "feat_proj" in ck["extra"] and any(k.startswith("recto.proj.s3_t3") for k in ck["extra"]["feat_proj"])
    student_keys = set(ck["model"])
    assert not any("proj" in k for k in student_keys)  # projectors never enter the exported student
    extra["steps"] = 3
    r2 = run_train(_cfg(root, s["ct"], extra), resume=True)
    assert r2["step"] == 3 and "feat" in r2["losses"]
