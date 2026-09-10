"""Training from several label stores at once (extra.train.stores / MultiStoreDataset)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from synth import make_synthetic, synthetic_axis
from tsm.config import parse_config
from tsm.data import CropDataset, MultiStoreDataset
from tsm.train import build_dataset, train_opts
from tsm.volume import VolumeReader


def _axis(y, x):
    a = synthetic_axis()
    a[:, 1], a[:, 2] = y, x
    return a.tolist()


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    root = tmp_path_factory.mktemp("multi")
    a = make_synthetic(str(root / "a"), fine_origin=(16, 32, 32), seed=0)
    b = make_synthetic(str(root / "b"), fine_origin=(32, 16, 16), seed=1)
    return {"a": a, "b": b, "root": str(root)}


def _ds(s, axis, **kw):
    kw.setdefault("patch", 32)
    kw.setdefault("stride", 16)
    kw.setdefault("length", 64)
    kw.setdefault("augment", False)
    return CropDataset(lambda: VolumeReader(s["ct"], 0, 2.4), s["fine"], s["coarse"],
                       input_radial=True, axis=axis, **kw)


def _cfg(root, extra):
    return parse_config({
        "volume": {"url": extra.pop("_ct"), "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [16, 32, 32], "size_zyx": [64, 64, 64]},
        "budget": {"ram_bytes": 8 << 30, "array_bytes": 2 << 30},
        "out_dir": root,
        "extra": {"train": extra},
    })


def _stores_extra(two, **over):
    e = {"steps": 1, "patch": 32, "batch": 1, "accum": 1, "widths": [8, 16, 16, 16, 16], "ckpt_every": 1,
         "num_workers": 0, "stride": 16, "log_every": 1, "warmup": 1, "augment": False,
         "eval_max_crops": 2, "_ct": two["a"]["ct"],
         "stores": [
             {"name": "a", "fine_store": two["a"]["fine"], "coarse_store": two["a"]["coarse"],
              "axis": _axis(100.0, 100.0), "voxel_um": 2.4},
             {"name": "b", "fine_store": two["b"]["fine"], "coarse_store": two["b"]["coarse"],
              "axis": _axis(60.0, 140.0), "voxel_um": 2.4,
              "volume": {"url": two["b"]["ct"], "level": 0, "voxel_um": 2.4}},
         ]}
    e.update(over)
    return e


# --------------------------------------------------------------------------- config
def test_stores_config_validation(two, tmp_path):
    root = str(tmp_path)
    o = train_opts(_cfg(root, _stores_extra(two)))
    assert [e["name"] for e in o["stores"]] == ["a", "b"] and o["holdout_stores"] == []
    assert o["stores"][1]["volume"]["url"] == two["b"]["ct"]
    with pytest.raises(ValueError, match="replaces fine_store"):
        train_opts(_cfg(root, _stores_extra(two, fine_store=two["a"]["fine"])))
    with pytest.raises(ValueError, match="unknown extra.train.stores"):
        e = _stores_extra(two)
        e["stores"][0]["fine"] = "x"
        train_opts(_cfg(root, e))
    with pytest.raises(ValueError, match="must be unique"):
        e = _stores_extra(two)
        e["stores"][1]["name"] = "a"
        train_opts(_cfg(root, e))
    with pytest.raises(ValueError, match="unknown stores"):
        train_opts(_cfg(root, _stores_extra(two, holdout_stores=["c"])))
    with pytest.raises(ValueError, match="every store"):
        train_opts(_cfg(root, _stores_extra(two, holdout_stores=["a", "b"])))
    with pytest.raises(ValueError, match="needs extra.train.stores"):
        train_opts(_cfg(root, {"_ct": two["a"]["ct"], "holdout_stores": ["a"], "fine_store": two["a"]["fine"]}))


def test_shipped_multi_configs_parse():
    from tsm.config import load_config

    o = train_opts(load_config("configs/multi_s1_s4.json"))
    assert [e["name"] for e in o["stores"]] == ["paris4", "s4"] and o["holdout_stores"] == []
    assert o["stores"][0]["axis"].endswith("PHercParis4/umbilicus-full-resolution.json")
    assert o["stores"][1]["voxel_um"] == pytest.approx(2.399) and "PHerc1667" in o["stores"][1]["volume"]["url"]
    o2 = train_opts(load_config("configs/multi_holdout_s4.json"))
    assert o2["holdout_stores"] == ["s4"]


# --------------------------------------------------------------------------- dataset
def test_multi_dataset_delegates_and_tags_origins(two):
    da, db = _ds(two["a"], _axis(100.0, 100.0)), _ds(two["b"], _axis(60.0, 140.0))
    m = MultiStoreDataset([da, db], ["a", "b"], weights=[1.0, 1.0], seed=0, length=32,
                          holdout=[da.origins[:2], db.origins[:3]])
    assert m.origins.shape[1] == 4 and set(m.origins[:, 0].tolist()) == {0, 1}
    assert len(m.origins) == len(da.origins) + len(db.origins)
    assert len(m.holdout_origins) == 5 and sorted(m.holdout_origins[:, 0].tolist()) == [0, 0, 1, 1, 1]
    for i in range(4):
        k = m.store_index(i)
        got, ref = m[i], [da, db][k][i]
        for key in ("input", "surface_sdf", "winding"):
            torch.testing.assert_close(got[key], ref[key])
    # the crops really come from the store the origin names (different fine origins + axes)
    o = m.holdout_origins[m.holdout_origins[:, 0] == 1][0]
    it = m.to_tensors(m.load(o))
    gz = db.fine.origin_zyx[0] + int(o[1])
    assert int(it["origin_zyx"][0]) == gz


def test_sampling_is_proportional_to_weight(two):
    da, db = _ds(two["a"], None), _ds(two["b"], None)
    m = MultiStoreDataset([da, db], ["a", "b"], weights=[3.0, 1.0], seed=0, length=4000)
    ks = np.array([m.store_index(i) for i in range(4000)])
    assert abs((ks == 0).mean() - 0.75) < 0.03
    m0 = MultiStoreDataset([da, db], ["a", "b"], weights=[1.0, 0.0], seed=0, length=500)
    assert all(m0.store_index(i) == 0 for i in range(500))  # a zero weight is never sampled
    # default weight = the store's number of training origins (uniform over crops)
    md = MultiStoreDataset([da, db], ["a", "b"], seed=0, length=8)
    assert md.weights == [float(len(da.origins)), float(len(db.origins))]


def test_channel_mismatch_is_rejected(tmp_path):
    a = make_synthetic(str(tmp_path / "a"))
    b = make_synthetic(str(tmp_path / "b"), faces=True)  # extra face channels
    da, db = _ds(a, None), _ds(b, None)
    with pytest.raises(ValueError, match="same channel set"):
        MultiStoreDataset([da, db], ["a", "b"])
    dc = _ds(a, None, patch=16)
    with pytest.raises(ValueError, match="patch="):
        MultiStoreDataset([da, dc], ["a", "c"])
    with pytest.raises(ValueError, match="unique"):
        MultiStoreDataset([da, da], ["a", "a"])


def test_build_dataset_multi_and_holdout_store(two, tmp_path):
    root = str(tmp_path / "multi")
    ds = build_dataset(_cfg(root, _stores_extra(two, holdout_origins={"z_frac": 0.5})),
                       train_opts(_cfg(root, _stores_extra(two, holdout_origins={"z_frac": 0.5}))), augment=False)
    assert isinstance(ds, MultiStoreDataset) and ds.names == ["a", "b"]
    assert set(ds.origins[:, 0].tolist()) == {0, 1} and set(ds.holdout_origins[:, 0].tolist()) == {0, 1}
    # holdout_stores: store b trains on nothing and is evaluated whole
    root2 = str(tmp_path / "hold")
    opts = train_opts(_cfg(root2, _stores_extra(two, holdout_stores=["b"])))
    ds2 = build_dataset(_cfg(root2, _stores_extra(two, holdout_stores=["b"])), opts, augment=False)
    assert set(ds2.origins[:, 0].tolist()) == {0} and set(ds2.holdout_origins[:, 0].tolist()) == {1}
    assert ds2.weights[1] == 0.0 and len(ds2.holdout_origins) == len(ds2.datasets[1].origins)


# --------------------------------------------------------------------------- training / eval
def test_run_train_multi_store_reports_per_store_metrics(two, tmp_path):
    from tsm.train import run_train

    root = str(tmp_path / "run")
    cfg = _cfg(root, _stores_extra(two, steps=2, holdout_stores=["b"], holdout_origins={"z_frac": 0.5}))
    r = run_train(cfg)
    m = r["holdout"]
    assert m["n_crops"] > 0
    for k in ("surface/sdf_mae", "a/surface/sdf_mae", "b/surface/sdf_mae", "a/n_crops", "b/n_crops"):
        assert k in m, k
    saved = json.load(open(os.path.join(root, "train", "holdout_metrics.json")))
    assert set(saved) == set(m)
    assert np.load(os.path.join(root, "train", "holdout.npy")).shape[1] == 4


def test_single_store_path_is_unchanged(two, tmp_path):
    """The old fine_store/coarse_store config still builds a plain CropDataset with identical items."""
    root = str(tmp_path / "single")
    extra = {"steps": 1, "patch": 32, "batch": 1, "accum": 1, "num_workers": 0, "stride": 16,
             "augment": False, "_ct": two["a"]["ct"], "fine_store": two["a"]["fine"],
             "coarse_store": two["a"]["coarse"], "input_radial": False, "seed": 3}
    cfg = _cfg(root, dict(extra))
    opts = train_opts(_cfg(root, dict(extra)))
    assert opts["stores"] is None
    ds = build_dataset(cfg, opts, augment=False)
    assert isinstance(ds, CropDataset) and not isinstance(ds, MultiStoreDataset)
    ref = CropDataset(lambda: VolumeReader(two["a"]["ct"], 0, 2.4), two["a"]["fine"], two["a"]["coarse"],
                      patch=32, seed=3, length=ds.length, stride=16, augment=False, origins=ds.origins,
                      input_radial=False, axis=opts["axis_path"])
    for i in (0, 1, 5):
        a, b = ds[i], ref[i]
        assert set(a) == set(b)
        for k in a:
            torch.testing.assert_close(a[k], b[k])
