"""dev/view_export.py + tsm.view: the model-view packet exporter (spec/view.md).

Synthetic stores with *different* origins, pitches and channel sets -- a faces+fibre
``pred.zarr``, the four teacher stores (including the 9.6 um lasagna), a label ``fine.zarr``
and an upstream ``rectoverso.zarr`` -- are cut by the exporter with ``--pred`` (no GPU).
The tests check the manifest (kinds, clip, palettes, vec/axis), every file's size, the bytes
against the stores they came from, the nearest upsampling of the coarse teacher, the
coverage-drop warning (never a padded layer) and the CUDA refusal of ``--checkpoint``.
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest
import zarr

from tsm.cli import LASAGNA_CHANNELS
from tsm.infer import pred_channels
from tsm.labels import FACE_CHANNELS, FINE_CHANNELS
from tsm.view import Store, layer_for
from tsm.volume import BrickWriter

DEV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dev")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(DEV, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


vx = _load("view_export")

CT_SHAPE = (128, 128, 128)
PRED_ORIGIN, PRED_SHAPE = (16, 16, 16), (96, 96, 96)
TEACH_ORIGIN, TEACH_SHAPE = (0, 0, 0), (128, 128, 128)
LAS_ORIGIN, LAS_SHAPE = (0, 0, 0), (32, 32, 32)  # 9.6 um = 4 level-0 voxels
LABEL_ORIGIN, LABEL_SHAPE = (24, 24, 24), (64, 64, 64)
RV_ORIGIN, RV_SHAPE = (8, 8, 8), (96, 96, 96)
BOX_ORIGIN, BOX_DIMS = (32, 40, 48), (16, 24, 32)
#: same origin, but past the far x face of the label and rectoverso stores
WIDE_DIMS = (16, 24, 64)
PRED_CHANNELS_T = pred_channels("faces", fiber=True, fiber_mode="class")


def _fill(shape, seed, n):
    return np.random.default_rng(seed).integers(0, 256, (n, *shape), dtype=np.uint8)


def _store(path, channels, origin, shape, seed, voxel_um=2.4, scale=1.0):
    a = _fill(shape, seed, len(channels))
    w = BrickWriter(path, channels, shape, chunk=32, origin_zyx=origin, voxel_um=voxel_um, scale=scale)
    for ci in range(len(channels)):
        w.write(ci, *origin, np.ascontiguousarray(a[ci]))
    return a


@pytest.fixture(scope="module")
def packet_root(tmp_path_factory):
    """Every store plus a config whose volume is a plain local CT zarr."""
    root = tmp_path_factory.mktemp("view")
    data: dict[str, np.ndarray] = {}
    ct = np.random.default_rng(0).integers(0, 256, CT_SHAPE, dtype=np.uint8)
    arr = zarr.create_array(store=str(root / "ct.zarr"), shape=CT_SHAPE, chunks=(32, 32, 32),
                            dtype="uint8", fill_value=0)
    arr[:] = ct
    data["ct"] = ct
    data["pred"] = _store(str(root / "pred.zarr"), PRED_CHANNELS_T, PRED_ORIGIN, PRED_SHAPE, 1)
    with open(str(root / "pred.summary.json"), "w") as fh:
        json.dump({"clip": 20.0, "tta": "none",
                   "student": {"checkpoint": "/tmp/latest.pt", "step": 12345}}, fh)
    teachers = root / "teachers"
    data["recto"] = _store(str(teachers / "recto.zarr"), ["recto"], TEACH_ORIGIN, TEACH_SHAPE, 2)
    data["ink"] = _store(str(teachers / "ink.zarr"), ["ink"], TEACH_ORIGIN, TEACH_SHAPE, 3)
    data["fiber"] = _store(str(teachers / "fiber.zarr"), ["fiber_bg", "fiber_vt", "fiber_hz", "fiber_ink"],
                           TEACH_ORIGIN, TEACH_SHAPE, 4)
    data["lasagna"] = _store(str(teachers / "lasagna.zarr"), LASAGNA_CHANNELS, LAS_ORIGIN, LAS_SHAPE, 5,
                             voxel_um=9.6, scale=0.25)
    data["fine"] = _store(str(root / "fine.zarr"), FINE_CHANNELS + FACE_CHANNELS,
                          LABEL_ORIGIN, LABEL_SHAPE, 6)
    rv = np.random.default_rng(7).integers(0, 4, (2, *RV_SHAPE), dtype=np.uint8)
    rva = zarr.create_array(store=str(root / "rectoverso.zarr"), shape=(2, *RV_SHAPE),
                            chunks=(1, 32, 32, 32), dtype="uint8", fill_value=0)
    rva[:] = rv
    rva.attrs["origin_zyx"] = list(RV_ORIGIN)
    rva.attrs["channels"] = ["rectoverso", "hzvt"]  # dev/rectoverso_slab.py's own names
    rva.attrs["voxel_um"] = 2.4
    data["rv"] = rv
    cfg = {
        "volume": {"url": str(root / "ct.zarr"), "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": list(BOX_ORIGIN), "size_zyx": list(BOX_DIMS)},
        "out_dir": str(root / "out"),
        "extra": {"scroll": "PHercTest"},
    }
    cfg_path = root / "cfg.json"
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh)
    return {"root": root, "cfg": str(cfg_path), "teachers": str(teachers), "data": data}


def _export(packet_root, out, dims=BOX_DIMS, extra=()):
    z, y, x = BOX_ORIGIN
    argv = [packet_root["cfg"], "--out", str(out),
            "--box", f"{z},{y},{x},{dims[0]},{dims[1]},{dims[2]}",
            "--pred", str(packet_root["root"] / "pred.zarr"),
            "--teachers", packet_root["teachers"],
            "--labels", str(packet_root["root"] / "fine.zarr"),
            "--rectoverso", str(packet_root["root"] / "rectoverso.zarr"),
            *extra]
    assert vx.main(argv) == 0
    with open(os.path.join(str(out), "v000", "meta.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def exported(packet_root, tmp_path_factory):
    out = tmp_path_factory.mktemp("packet")
    return {"out": str(out), "meta": _export(packet_root, out)}


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #
def test_meta_shape_and_ct(exported):
    m = exported["meta"]
    assert m["format"] == "tsm.view.v1"
    assert m["dims_zyx"] == list(BOX_DIMS)
    assert m["origin_zyx"] == list(BOX_ORIGIN)
    assert m["voxel_um"] == 2.4 and m["scroll"] == "PHercTest"
    cts = [lay for lay in m["layers"] if lay["kind"] == "ct"]
    assert len(cts) == 1 and cts[0]["file"] == "ct.u8" and m["layers"][0] is cts[0]
    keys = [(lay["group"], lay["name"]) for lay in m["layers"]]
    assert len(keys) == len(set(keys)), "group+name must be unique"


def test_meta_groups_and_kinds(exported):
    by = {(lay["group"], lay["name"]): lay for lay in exported["meta"]["layers"]}
    assert by[("student", "sdf_in")]["kind"] == "sdf" and by[("student", "sdf_in")]["clip"] == 20.0
    assert by[("student", "valid")]["kind"] == "prob"
    assert by[("student", "density")]["kind"] == "density"
    assert by[("student", "thickness")]["kind"] == "count"
    assert by[("student", "sin")]["kind"] == "signed" and "vec" not in by[("student", "sin")]
    for name, axis in (("nx", 0), ("ny", 1), ("nz", 2)):
        lay = by[("student", name)]
        assert (lay["kind"], lay["vec"], lay["axis"]) == ("signed", "normal", axis)
    assert by[("teacher", "recto")]["kind"] == "prob"
    assert by[("teacher", "cos")]["kind"] == "signed"
    assert {n for g, n in by if g == "teacher"} == {"recto", "ink", "fiber_vt", "fiber_hz", "cos"}
    assert by[("label", "faces_valid")]["palette"] == ["invalid", "valid", "ignore"]
    assert by[("human", "rv_class")]["palette"] == ["bg", "recto", "verso", "contact"]
    assert by[("human", "hzvt_class")]["palette"] == ["bg", "hz", "vt", "exclude"]
    assert {n for g, n in by if g == "label"} == {"sdf_in", "sdf_out", "faces_valid", "ink"}


def test_provenance_from_pred_summary(exported):
    p = exported["meta"]["provenance"]
    assert p["checkpoint"] == "/tmp/latest.pt" and p["step"] == 12345
    assert p["clip"] == 20.0 and p["pred"].endswith("pred.zarr")
    assert p["config"].endswith("cfg.json")


def test_view_json(exported):
    with open(os.path.join(exported["out"], "view.json")) as fh:
        doc = json.load(fh)
    assert doc["format"] == "tsm.view.v1"
    assert [p["path"] for p in doc["packets"]] == ["v000"]
    assert doc["packets"][0]["dims_zyx"] == list(BOX_DIMS)
    assert "student.sdf_in" in doc["packets"][0]["layers"]


# --------------------------------------------------------------------------- #
# the bytes
# --------------------------------------------------------------------------- #
def test_every_layer_file_has_the_right_size(exported):
    n = int(np.prod(BOX_DIMS))
    files = sorted(f for f in os.listdir(os.path.join(exported["out"], "v000")) if f.endswith(".u8"))
    assert len(files) == len(exported["meta"]["layers"])
    for lay in exported["meta"]["layers"]:
        p = os.path.join(exported["out"], "v000", lay["file"])
        assert os.path.getsize(p) == n, lay["file"]


def _read(out, file):
    return np.fromfile(os.path.join(out, "v000", file), dtype=np.uint8).reshape(BOX_DIMS)


def _sub(a, origin, dims, store_origin):
    lo = [origin[i] - store_origin[i] for i in range(3)]
    return a[lo[0]:lo[0] + dims[0], lo[1]:lo[1] + dims[1], lo[2]:lo[2] + dims[2]]


def test_bytes_round_trip(exported, packet_root):
    d = packet_root["data"]
    out = exported["out"]
    assert np.array_equal(_read(out, "ct.u8"), _sub(d["ct"], BOX_ORIGIN, BOX_DIMS, (0, 0, 0)))
    ci = PRED_CHANNELS_T.index("sdf_in")
    assert np.array_equal(_read(out, "student.sdf_in.u8"),
                          _sub(d["pred"][ci], BOX_ORIGIN, BOX_DIMS, PRED_ORIGIN))
    assert np.array_equal(_read(out, "teacher.recto.u8"),
                          _sub(d["recto"][0], BOX_ORIGIN, BOX_DIMS, TEACH_ORIGIN))
    assert np.array_equal(_read(out, "teacher.fiber_hz.u8"),
                          _sub(d["fiber"][2], BOX_ORIGIN, BOX_DIMS, TEACH_ORIGIN))
    li = (FINE_CHANNELS + FACE_CHANNELS).index("faces_valid")
    assert np.array_equal(_read(out, "label.faces_valid.u8"),
                          _sub(d["fine"][li], BOX_ORIGIN, BOX_DIMS, LABEL_ORIGIN))
    assert np.array_equal(_read(out, "human.hzvt_class.u8"),
                          _sub(d["rv"][1], BOX_ORIGIN, BOX_DIMS, RV_ORIGIN))


def test_coarse_teacher_is_nearest_upsampled(exported, packet_root):
    """teacher.cos comes from the 9.6 um store: byte (z, y, x) is the coarse voxel // 4."""
    cos = packet_root["data"]["lasagna"][LASAGNA_CHANNELS.index("cos")]
    got = _read(exported["out"], "teacher.cos.u8")
    z, y, x = np.meshgrid(*[np.arange(BOX_ORIGIN[a], BOX_ORIGIN[a] + BOX_DIMS[a]) for a in range(3)],
                          indexing="ij")
    want = cos[z // 4 - LAS_ORIGIN[0], y // 4 - LAS_ORIGIN[1], x // 4 - LAS_ORIGIN[2]]
    assert np.array_equal(got, want)


# --------------------------------------------------------------------------- #
# coverage
# --------------------------------------------------------------------------- #
def test_uncovered_store_is_dropped_with_a_warning(packet_root, tmp_path, capsys):
    meta = _export(packet_root, tmp_path / "wide", dims=WIDE_DIMS)
    err = capsys.readouterr().out
    groups = {lay["group"] for lay in meta["layers"]}
    assert "label" not in groups and "human" not in groups, "a store that stops inside the box must be dropped"
    assert {"ct", "student", "teacher"} <= groups
    assert "WARNING" in err and "fine.zarr" in err and "does not cover" in err
    n = int(np.prod(WIDE_DIMS))
    for lay in meta["layers"]:
        assert os.path.getsize(os.path.join(str(tmp_path / "wide"), "v000", lay["file"])) == n


def test_store_covers_is_pure_geometry(packet_root):
    st = Store(str(packet_root["root"] / "fine.zarr"))
    assert st.origin == LABEL_ORIGIN and st.shape == LABEL_SHAPE and st.factor == 1
    assert st.covers(LABEL_ORIGIN, LABEL_SHAPE)
    assert not st.covers((LABEL_ORIGIN[0] - 1, *LABEL_ORIGIN[1:]), LABEL_SHAPE)
    assert not st.covers(LABEL_ORIGIN, (LABEL_SHAPE[0] + 1, *LABEL_SHAPE[1:]))
    las = Store(str(packet_root["root"] / "teachers" / "lasagna.zarr"))
    assert las.factor == 4 and las.covers((0, 0, 0), (128, 128, 128))
    assert not las.covers((0, 0, 0), (132, 128, 128))


# --------------------------------------------------------------------------- #
# the kind table and the CLI
# --------------------------------------------------------------------------- #
def test_rectoverso_channel_names_are_aliased(packet_root):
    """The slab writes 'rectoverso' / 'hzvt'; the viewer sees the class names of the fine store."""
    from tsm.view import rectoverso_source

    src = rectoverso_source(str(packet_root["root"] / "rectoverso.zarr"))
    assert [(lay.name, ch) for lay, ch in src.picks] == [("rv_class", "rectoverso"),
                                                         ("hzvt_class", "hzvt")]


def test_kind_table_is_shared_between_groups():
    a = layer_for("sdf_in", "student", 20.0)
    b = layer_for("sdf_in", "label", 20.0)
    assert a.kind == b.kind == "sdf" and a.clip == b.clip == 20.0
    assert a.file == "student.sdf_in.u8" and b.file == "label.sdf_in.u8"
    assert layer_for("sdf_valid", "label").kind == "class"
    assert layer_for("no_such_channel", "student") is None


def test_parse_box():
    assert vx.parse_box("1,2,3,4,5,6") == ((1, 2, 3), (4, 5, 6))
    for bad in ("1,2,3", "1,2,3,4,5,0", "a,2,3,4,5,6"):
        with pytest.raises(Exception):
            vx.parse_box(bad)


def test_checkpoint_refuses_without_cuda(packet_root, monkeypatch, tmp_path):
    """The laptop GPU is never used: no CUDA device -> a hard error, never a CPU fallback."""
    import torch

    from tsm.config import load_config
    from tsm.view import StudentSource

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = load_config(packet_root["cfg"])
    with pytest.raises(RuntimeError, match="CUDA"):
        StudentSource(str(tmp_path / "latest.pt"), cfg)


def test_pred_and_checkpoint_are_exclusive(packet_root, tmp_path):
    z, y, x = BOX_ORIGIN
    with pytest.raises(SystemExit):
        vx.main([packet_root["cfg"], "--out", str(tmp_path / "x"), "--box", f"{z},{y},{x},8,8,8",
                 "--pred", str(packet_root["root"] / "pred.zarr"), "--checkpoint", "latest.pt"])
