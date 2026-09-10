"""Export model-view packets for render3d's ``--view`` mode (``spec/view.md``).

    uv run python dev/view_export.py configs/paris4_eval_multi.json \
        --out ~/tsm-output/view/multi_p4 \
        --box 34496,15360,18688,256,512,512 [--box ...] \
        --pred ~/tsm-output/eval_multi_p4/student/pred.zarr \
        --teachers ~/tsm-output/eval/teachers \
        --labels ~/tsm-output/eval_faces_labels/labels/fine.zarr \
        --rectoverso ~/tsm-output/rectoverso_eval/rectoverso.zarr

One packet directory (``v000/``, ``v001/`` ...) per ``--box``, each holding ``meta.json``,
the required ``ct.u8`` and one ``<group>.<name>.u8`` per layer, plus a ``view.json`` batch
manifest at the root.  Everything about *what* a layer is lives in :mod:`tsm.view` (the kind
table, the store readers, the packet assembly), which ``tsm serve`` shares, so a packet on
disk and a live response describe their bytes identically.

Groups
------
``student``  ``--pred`` copies the channels of an existing ``pred.zarr`` (no GPU), or
             ``--checkpoint`` runs the student on each box.  ``--checkpoint`` needs a CUDA
             device and refuses to run without one -- the laptop GPU is never used, the
             intended machines are forlindesk2 and the Thunder A6000.
``teacher``  ``--teachers DIR``: ``recto``, ``ink``, ``fiber_vt/hz`` and, nearest-upsampled
             from the 9.6 um ``lasagna.zarr``, ``cos``.
``label``    ``--labels fine.zarr``: ``sdf_in``, ``sdf_out``, ``faces_valid``, ``ink``.
``human``    ``--rectoverso rectoverso.zarr``: the upstream ``rv_class`` / ``hzvt_class`` bands.

Every store is read at its own ``origin_zyx``; a box a store does not fully cover **drops
that store's layers with a warning** rather than padding them with zeros (which every kind's
decoding would read as real "no data").  Layers are written one at a time, so the exporter
never holds more than one box (plus, with ``--checkpoint``, the student's own head box).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tsm.config import load_config  # noqa: E402
from tsm.view import (CTSource, FORMAT, Source, StudentSource, label_source, packet_meta,  # noqa: E402
                      plan, pred_source, rectoverso_source, scroll_name, teacher_sources,
                      write_packet_dir)


def _log(msg: str) -> None:
    print(f"[view-export] {msg}", flush=True)


def parse_box(s: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """``z,y,x,dz,dy,dx`` -> (origin, dims)."""
    parts = [p for p in str(s).replace(" ", "").split(",") if p != ""]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(f"--box wants z,y,x,dz,dy,dx (6 ints), got {s!r}")
    try:
        v = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--box wants 6 ints, got {s!r}")
    if min(v[3:]) <= 0:
        raise argparse.ArgumentTypeError(f"--box sizes must be positive, got {s!r}")
    return (v[0], v[1], v[2]), (v[3], v[4], v[5])


def build_sources(cfg: Any, a: argparse.Namespace) -> list[Source]:
    """The ordered sources of every packet: ct first (the viewer's base image), then the
    student, the teachers, the labels and the human bands."""
    from tsm.cli import open_ct

    sources: list[Source] = [CTSource(open_ct(cfg, 0))]
    if a.pred:
        sources.append(pred_source(a.pred, a.group_name, level0_um=float(cfg.volume.voxel_um)))
    elif a.checkpoint:
        sources.append(StudentSource(a.checkpoint, cfg, group=a.group_name, tta=a.tta, log=_log))
    if a.teachers:
        sources.extend(teacher_sources(a.teachers, level0_um=float(cfg.volume.voxel_um), log=_log))
    if a.labels:
        sources.append(label_source(a.labels, level0_um=float(cfg.volume.voxel_um)))
    if a.rectoverso:
        sources.append(rectoverso_source(a.rectoverso, level0_um=float(cfg.volume.voxel_um)))
    return sources


def write_view_json(out_dir: str, metas: list[dict[str, Any]], names: list[str], cfg_path: str) -> str:
    """The batch manifest: the viewer's Prev / Next list (``packets[].path``)."""
    path = os.path.join(out_dir, "view.json")
    doc = {
        "format": FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": os.path.abspath(cfg_path),
        "packets": [
            {"path": name, "origin_zyx": m["origin_zyx"], "dims_zyx": m["dims_zyx"],
             "voxel_um": m["voxel_um"], "scroll": m["scroll"],
             "layers": [f"{lay['group']}.{lay['name']}" for lay in m["layers"]]}
            for name, m in zip(names, metas)
        ],
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="export tsm.view.v1 packets for render3d --view")
    ap.add_argument("config")
    ap.add_argument("--out", required=True, help="packet root (v000/, v001/ ... + view.json)")
    ap.add_argument("--box", action="append", required=True, metavar="z,y,x,dz,dy,dx",
                    help="one packet box in level-0 voxels; repeatable")
    ap.add_argument("--pred", default=None, help="student pred.zarr to copy channels from (no GPU)")
    ap.add_argument("--checkpoint", default=None, help="run the student on each box (needs CUDA)")
    ap.add_argument("--tta", default="none", choices=("none", "flip8", "flip8_rot4"),
                    help="--checkpoint test-time augmentation")
    ap.add_argument("--teachers", default=None, help="teacher store directory (recto/ink/fiber/lasagna.zarr)")
    ap.add_argument("--labels", default=None, help="label fine.zarr")
    ap.add_argument("--rectoverso", default=None, help="upstream rectoverso.zarr")
    ap.add_argument("--group-name", default="student", help="group name of the --pred / --checkpoint layers")
    a = ap.parse_args(argv)

    if a.pred and a.checkpoint:
        raise SystemExit("[view-export] --pred and --checkpoint are exclusive")
    boxes = [parse_box(s) for s in a.box]
    cfg = load_config(a.config)
    out_dir = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(out_dir, exist_ok=True)

    sources = build_sources(cfg, a)
    if len(sources) == 1:
        _log("WARNING no student / teacher / label / human source: the packets hold only ct.u8")
    scroll = scroll_name(cfg)
    metas: list[dict[str, Any]] = []
    names: list[str] = []
    for i, (origin, dims) in enumerate(boxes):
        name = f"v{i:03d}"
        kept = plan(sources, origin, dims, _log)
        if not any(lay.kind == "ct" for s in kept for lay in s.layers()):
            raise SystemExit(f"[view-export] {name}: the CT does not cover box {origin} {dims} "
                             f"(ct.u8 is required by every packet)")
        t0 = time.perf_counter()
        meta = write_packet_dir(os.path.join(out_dir, name), kept, origin, dims,
                                float(cfg.volume.voxel_um), cfg.budget, scroll,
                                provenance={"config": os.path.abspath(a.config)}, log=_log)
        _log(f"{name}: origin={list(origin)} dims={list(dims)} "
             f"{len(meta['layers'])} layers in {time.perf_counter() - t0:.1f}s")
        metas.append(meta)
        names.append(name)
    path = write_view_json(out_dir, metas, names, a.config)
    _log(f"wrote {len(metas)} packet(s) and {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
