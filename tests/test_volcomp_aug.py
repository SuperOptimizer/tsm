"""volcomp round-trip augmentation: the real lossy-codec residual, on the CPU in the worker.

The training CT may come from the lossy volcomp mirror (q = 8: |d| mean ~3 grey levels).
``extra.train.augment``'s ``volcomp`` family puts that exact residual into the augmentation
pipeline, so a student trained on the lossless volume still sees mirror-like inputs.
"""
import json
import os

import numpy as np
import pytest
import torch

pytest.importorskip("volcomp_zarr", reason="the volcomp codec is not installed here")

from synth import make_synthetic  # noqa: E402
from tsm.config import parse_config  # noqa: E402
from tsm.data import (  # noqa: E402
    TRANSFORM_NAMES,
    VOLCOMP_CHUNK,
    AugmentConfig,
    CropDataset,
    VolcompCfg,
    as_augment_config,
    volcomp_roundtrip,
)
from tsm.train import run_train, volcomp_spec  # noqa: E402
from tsm.volume import VolumeReader  # noqa: E402


def sheet_volume(shape=(VOLCOMP_CHUNK,) * 3, seed: int = 0, noise: float = 2.0) -> np.ndarray:
    """A sheet-like uint8 volume: wavy bright sheets in a dark matrix, plus a little noise."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    phase = x + 4.0 * np.sin(y / 25.0) + 2.0 * np.cos(z / 30.0)
    v = 60.0 + 140.0 * np.exp(-np.sin(np.pi * phase / 14.0) ** 2 / 0.15)
    return (v + rng.normal(0.0, noise, v.shape)).clip(0, 255).astype(np.uint8)


def mean_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def test_roundtrip_is_small_and_grows_with_q():
    v = sheet_volume()
    out = {q: volcomp_roundtrip(v, q) for q in (4.0, 8.0, 12.0)}
    for q, o in out.items():
        assert o.shape == v.shape and o.dtype == np.uint8 and o is not v
    d = {q: mean_abs_delta(o, v) for q, o in out.items()}
    assert d[4.0] < d[8.0] < d[12.0], d  # lossier q -> bigger residual
    assert 1.0 <= d[8.0] <= 6.0, d  # the mirror's own quality: a few grey levels
    assert not np.array_equal(out[8.0], v)  # it really is lossy


def test_roundtrip_tiles_shapes_that_are_not_one_chunk():
    for shape in ((32, 32, 32), (VOLCOMP_CHUNK, 48, 160)):
        v = sheet_volume(shape, seed=1)
        o = volcomp_roundtrip(v, 8.0)
        assert o.shape == shape and o.dtype == np.uint8
        assert 1.0 <= mean_abs_delta(o, v) <= 8.0
    assert np.array_equal(volcomp_roundtrip(np.zeros((16, 16, 16), np.uint8), 8.0), np.zeros((16, 16, 16), np.uint8))
    with pytest.raises(ValueError):
        volcomp_roundtrip(np.zeros((4, 4), np.uint8), 8.0)


def test_roundtrip_cost_per_128_crop():
    import time

    v = sheet_volume()
    volcomp_roundtrip(v, 8.0)  # warm up
    t0 = time.perf_counter()
    n = 3
    for _ in range(n):
        volcomp_roundtrip(v, 8.0)
    ms = 1e3 * (time.perf_counter() - t0) / n
    print(f"[volcomp] {ms:.1f} ms per {VOLCOMP_CHUNK}^3 crop")
    assert ms < 200.0  # the real budget is ~50 ms; loose here so a busy CI box does not flake


def test_config_family_preset_and_validation():
    assert "volcomp" in TRANSFORM_NAMES
    c = AugmentConfig.preset("strong")
    assert c.volcomp.p == 0.0 and tuple(c.volcomp.q) == (4.0, 12.0)  # "strong" is unchanged
    sv = AugmentConfig.preset("strong_volcomp")
    assert sv.volcomp.p == 0.3
    assert {k: v for k, v in sv.to_dict().items() if k != "volcomp"} == {
        k: v for k, v in c.to_dict().items() if k != "volcomp"}  # = strong + volcomp
    assert AugmentConfig.from_dict(sv.to_dict()).to_dict() == sv.to_dict()
    assert AugmentConfig.preset("none").volcomp.p == 0.0
    cfg = as_augment_config({"preset": "none", "volcomp": {"p": 1.0, "q": [8.0, 8.0]}})
    assert cfg.volcomp.p == 1.0 and not cfg.is_identity()
    for bad in ({"p": 0.5, "q": [0.0, 4.0]}, {"p": 0.5, "q": [12.0, 4.0]}, {"p": 0.5, "q": [4.0]},
                {"p": 2.0}, {"p": 0.5, "qq": 1}):
        with pytest.raises(ValueError):
            AugmentConfig.from_dict({"volcomp": bad})


def test_missing_codec_raises_at_config_time(monkeypatch):
    import tsm.data as D

    monkeypatch.setattr(D, "volcomp_codec", lambda: None)
    VolcompCfg(p=0.0)  # off: never needs the codec
    with pytest.raises(ValueError, match="not importable"):
        VolcompCfg(p=0.3)


def _dataset(s, volcomp, seed=0):
    return CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], s["coarse"], patch=32,
                       stride=16, length=8, seed=seed, augment=False, volcomp=volcomp)


def test_dataset_applies_it_only_on_the_training_path(tmp_path):
    s = make_synthetic(str(tmp_path / "out"))
    off, zero = _dataset(s, None), _dataset(s, {"p": 0.0})
    on = _dataset(s, {"p": 1.0, "q": [8.0, 8.0]})
    assert zero.volcomp is None and on.volcomp is not None  # p = 0 is not even wired up
    for i in range(3):
        a, b, c = off[i], zero[i], on[i]
        assert torch.equal(a["input"], b["input"])  # p = 0: bit-identical
        assert a["input"].shape == c["input"].shape and a["input"].dtype == c["input"].dtype
        d = 255.0 * float((a["input"][0, 0] - c["input"][0, 0]).abs().mean())
        assert 0.5 < d < 10.0, d  # the codec residual, in grey levels
        for k in ("surface_sdf", "surface_valid", "ink_prob", "winding"):
            assert torch.equal(a[k], c[k])  # targets never move
        assert float(c["volcomp_q"]) == 8.0 and float(b["volcomp_q"] if "volcomp_q" in b else 0.0) == 0.0
    # a direct load() (the held-out evaluation, --overfit-one) is never augmented
    assert np.array_equal(on.load(on.origins[0])["ct"], off.load(off.origins[0])["ct"])
    assert float(on.load(on.origins[0])["volcomp_q"]) == 0.0


def test_fire_rate_follows_p(tmp_path):
    s = make_synthetic(str(tmp_path / "out"))
    ds = _dataset(s, {"p": 0.5, "q": [4.0, 12.0]}, seed=3)
    qs = [float(ds[i]["volcomp_q"]) for i in range(24)]
    fired = [q for q in qs if q > 0]
    assert 4 <= len(fired) <= 20 and all(4.0 <= q <= 12.0 for q in fired)


def test_run_train_two_steps_with_volcomp(tmp_path):
    root = str(tmp_path / "out")
    s = make_synthetic(root)
    extra = {"steps": 2, "patch": 32, "batch": 2, "accum": 1, "widths": [8, 16, 16, 16, 16], "ckpt_every": 2,
             "num_workers": 0, "stride": 16, "log_every": 1, "warmup": 1, "coarse_store": s["coarse"],
             "fine_store": s["fine"],
             "augment": {"preset": "strong_volcomp", "volcomp": {"p": 1.0, "q": [8.0, 8.0]}}}
    cfg = parse_config({
        "volume": {"url": s["ct"], "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root,
        "extra": {"train": extra},
    })
    assert volcomp_spec({"augment": "strong_volcomp"}).p == 0.3
    assert volcomp_spec({"augment": "strong"}) is None and volcomp_spec({"augment": "v1"}) is None
    r = run_train(cfg)
    assert r["step"] == 2
    lines = [json.loads(l) for l in open(os.path.join(root, "train", "log.jsonl"))]
    assert [l["step"] for l in lines] == [1, 2]
    assert all(l["aug/volcomp"] == 2 for l in lines), [l.get("aug/volcomp") for l in lines]
