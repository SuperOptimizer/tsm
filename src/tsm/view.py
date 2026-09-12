"""Model-view packets (``tsm.view.v1``) shared by the exporter and the live server.

render3d's ``--view`` mode reads a *view packet*: one CT crop plus any number of named
uint8 **layers** -- student heads, teacher outputs, label channels, human bands -- each
tagged with a *kind* that says how the viewer decodes and draws it (see
``render3d/spec/view.md``).  The same layer vocabulary is served live over TCP by
``tsm serve``, so the viewer can ask a remote GPU for the student's prediction of any box.

This module holds everything both consumers need and nothing either of them owns alone:

* :class:`Layer` -- one manifest entry (file / name / group / kind + the kind's keys).
* :func:`layer_for` -- the **kind table**: TSM's channel names -> a viewer kind.  It is the
  single place that knows that ``sdf_in`` is an ``sdf`` with a clip, that ``nx/ny/nz`` are
  the three components of one ``normal`` vector, and that ``faces_valid`` is a ``class``
  with a palette; both the exporter's ``meta.json`` and the server's response JSON come
  from it, so a viewer cannot see two different decodings of the same bytes.
* :class:`Store` / :class:`StoreSource` -- an origin-aware channel-store reader and the
  group of layers cut from it.  Coverage is decided from the geometry alone, *before* any
  read, so a box a store does not fully cover drops that store's layers with a warning
  (:func:`plan`) instead of being padded with fabricated zeros.
* :class:`CTSource` -- ``ct.u8`` from the config's volume reader.
* :class:`StudentSource` -- the student run on the box (CUDA only, model loaded once), for
  ``--checkpoint`` and for the server.
* :func:`packet_meta` / :func:`iter_arrays` / :func:`write_packet_dir` -- assembly.  Layers
  are produced one at a time and written (or sent) immediately: a packet is never fully
  resident, and every read is checked against the run's :class:`~tsm.limits.Budget`.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import zarr

from tsm.config import RunCfg
from tsm.data import CLIP
from tsm.limits import Budget, check_alloc

__all__ = [
    "FORMAT",
    "MAX_DIM",
    "Layer",
    "layer_for",
    "Store",
    "Source",
    "StoreSource",
    "CTSource",
    "StudentSource",
    "scroll_name",
    "pred_source",
    "teacher_sources",
    "label_source",
    "rectoverso_source",
    "check_dims",
    "plan",
    "packet_meta",
    "iter_arrays",
    "write_packet_dir",
]

#: value of ``meta.json``'s ``format`` key; the viewer refuses anything else
FORMAT = "tsm.view.v1"
#: per-axis hard limit of a packet's ``dims_zyx`` (spec/view.md)
MAX_DIM = 4096

Log = Callable[[str], None]


def _log(msg: str) -> None:
    print(f"[view] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# layers and the kind table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Layer:
    """One ``meta.json`` layer entry.

    ``group`` + ``name`` is the viewer's identity of a layer and ``<group>.<name>.u8`` its
    file, except the single required ``ct`` layer, which is always ``ct.u8``."""

    name: str
    group: str
    kind: str
    clip: float | None = None
    palette: tuple[str, ...] | None = None
    vec: str | None = None
    axis: int | None = None

    @property
    def file(self) -> str:
        return "ct.u8" if self.kind == "ct" else f"{self.group}.{self.name}.u8"

    def meta(self) -> dict[str, Any]:
        d: dict[str, Any] = {"file": self.file, "name": self.name, "group": self.group, "kind": self.kind}
        if self.clip is not None:
            d["clip"] = float(self.clip)
        if self.palette is not None:
            d["palette"] = list(self.palette)
        if self.vec is not None:
            d["vec"] = self.vec
            d["axis"] = int(self.axis if self.axis is not None else 0)
        return d


#: channels whose byte is ``sigmoid``-scaled (or a 0/255 mask): kind ``prob``
_PROB = (
    "valid", "ink", "conf", "spare", "recto", "surface", "m7",
    "fiber_bg", "fiber_vt", "fiber_hz", "fiber_ink", "fiber_strength",
    "surface1", "surface_in1", "surface_out1",
)
#: signed [-1, 1] scalars that are not part of a vector
_SIGNED = ("sin", "cos", "phase_sin", "phase_cos", "grad_mag")
#: the three components of one vector field: name -> (vec name, axis; 0/1/2 = x/y/z -> R/G/B)
_VEC = {
    "nx": ("normal", 0), "ny": ("normal", 1), "nz": ("normal", 2),
    "fiber_dx": ("fiber_dir", 0), "fiber_dy": ("fiber_dir", 1), "fiber_dz": ("fiber_dir", 2),
}
#: signed-distance channels (kind ``sdf``, ``clip`` required)
_SDF = ("sdf", "sdf_in", "sdf_out")
#: categorical channels and their class names
_CLASS = {
    "sdf_valid": ("invalid", "valid", "ignore"),
    "ink_valid": ("invalid", "valid", "ignore"),
    "faces_valid": ("invalid", "valid", "ignore"),
    "fiber_valid": ("invalid", "valid", "ignore"),
    "faces_source": ("none", "rectoverso", "ct", "human"),
    "rv_class": ("bg", "recto", "verso", "contact"),
    "hzvt_class": ("bg", "hz", "vt", "exclude"),
}


def layer_for(channel: str, group: str, clip: float = CLIP, name: str | None = None) -> Layer | None:
    """The viewer layer for a TSM channel name, or ``None`` when the channel has no kind.

    The one table both the exporter and the server use, so ``student.sdf_in`` and
    ``label.sdf_in`` are guaranteed to reach the viewer with the same decoding."""
    ch = str(channel)
    nm = str(name if name is not None else ch)
    if ch in _SDF:
        return Layer(nm, group, "sdf", clip=float(clip))
    if ch in _VEC:
        vec, axis = _VEC[ch]
        return Layer(nm, group, "signed", vec=vec, axis=axis)
    if ch in _SIGNED:
        return Layer(nm, group, "signed")
    if ch in _CLASS:
        return Layer(nm, group, "class", palette=_CLASS[ch])
    if ch == "density":
        return Layer(nm, group, "density")
    if ch == "thickness":
        return Layer(nm, group, "count")
    if ch in _PROB:
        return Layer(nm, group, "prob")
    return None


def scroll_name(cfg: RunCfg) -> str:
    """Informational scroll name: ``extra.scroll`` when set, else the URL's dataset component.

    Volume URLs are ``.../<scroll>/volumes/<name>.zarr`` upstream and in our mirror, so the
    component before ``volumes`` is the scroll; anything else yields ``""`` (the key is
    display-only and the viewer must not depend on it)."""
    s = cfg.extra.get("scroll")
    if isinstance(s, str) and s:
        return s
    m = re.search(r"/([A-Za-z0-9_.-]+)/volumes/", str(cfg.volume.url))
    return m.group(1) if m else ""


# --------------------------------------------------------------------------- #
# stores
# --------------------------------------------------------------------------- #
class Store:
    """Channel-indexed reader for a (C, Z, Y, X) uint8 store written by ``BrickWriter``.

    Boxes are given in **global level-0 voxels**; the store's own ``origin_zyx`` (and, for a
    coarser store such as ``lasagna.zarr``, its ``voxel_um``) turn them into local indices."""

    def __init__(self, path: str, level0_um: float = 2.4, channels: Sequence[str] | None = None) -> None:
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        self.arr = zarr.open_array(store=self.path, mode="r")
        attrs = dict(self.arr.attrs)
        chans = attrs.get("channels", channels or [])
        self.channels = [str(c) for c in chans]
        self.origin = tuple(int(v) for v in attrs.get("origin_zyx", (0, 0, 0)))
        self.shape = tuple(int(s) for s in self.arr.shape[1:])
        self.voxel_um = float(attrs.get("voxel_um", level0_um))
        f = self.voxel_um / float(level0_um)
        k = int(round(f))
        if k < 1 or abs(k - f) > 1e-3 * max(f, 1.0):
            raise ValueError(f"{self.path}: voxel {self.voxel_um} um is not an integer multiple of {level0_um} um")
        #: level-0 voxels per store voxel (1 for a 2.4 um store, 4 for the 9.6 um lasagna)
        self.factor = k
        if len(self.channels) != int(self.arr.shape[0]):
            raise ValueError(f"{self.path}: {len(self.channels)} channel names for {self.arr.shape[0]} channels")

    def has(self, name: str) -> bool:
        return name in self.channels

    def covers(self, origin: Sequence[int], dims: Sequence[int]) -> bool:
        """True when the whole level-0 box lies inside the store (never padded, see plan())."""
        f = self.factor
        for a in range(3):
            lo = (int(origin[a]) - self.origin[a] * f)
            hi = lo + int(dims[a])
            if lo < 0 or hi > self.shape[a] * f:
                return False
        return True

    def read_global(self, name: str, origin: Sequence[int], dims: Sequence[int],
                    budget: Budget | None = None) -> np.ndarray:
        """One channel's level-0 box, nearest-upsampled from a coarser store.

        The caller has already checked :meth:`covers`, so no padding is possible here; a
        coarse store is read on its own grid and repeated by ``factor`` (the viewer never
        resamples, and nearest keeps the store's bytes exact)."""
        ci = self.channels.index(str(name))
        f = self.factor
        lo = [int(origin[a]) - self.origin[a] * f for a in range(3)]
        hi = [lo[a] + int(dims[a]) for a in range(3)]
        clo = [lo[a] // f for a in range(3)]
        chi = [-((-hi[a]) // f) for a in range(3)]  # ceil
        if budget is not None:
            check_alloc([chi[a] - clo[a] for a in range(3)], np.uint8, budget)
        blk = np.asarray(self.arr[ci, clo[0]:chi[0], clo[1]:chi[1], clo[2]:chi[2]], dtype=np.uint8)
        if f == 1:
            return np.ascontiguousarray(blk)
        if budget is not None:
            check_alloc([(chi[a] - clo[a]) * f for a in range(3)], np.uint8, budget)
        up = np.repeat(np.repeat(np.repeat(blk, f, axis=0), f, axis=1), f, axis=2)
        d = [lo[a] - clo[a] * f for a in range(3)]
        return np.ascontiguousarray(up[d[0]:d[0] + dims[0], d[1]:d[1] + dims[1], d[2]:d[2] + dims[2]])


# --------------------------------------------------------------------------- #
# sources: a group of layers cut from one thing
# --------------------------------------------------------------------------- #
class Source:
    """A named group of layers that can be cut from one store / reader / model."""

    group: str = ""
    #: what the manifest's ``provenance`` should say about this source
    provenance: dict[str, Any] = {}

    def layers(self) -> list[Layer]:
        raise NotImplementedError

    def covers(self, origin: Sequence[int], dims: Sequence[int]) -> bool:
        return True

    def why(self) -> str:
        """Short description used in the coverage warning."""
        return self.group

    def read(self, origin: Sequence[int], dims: Sequence[int], budget: Budget) -> Iterator[tuple[Layer, np.ndarray]]:
        raise NotImplementedError


class StoreSource(Source):
    """Layers cut from one :class:`Store` (one zarr = one coverage decision)."""

    def __init__(self, store: Store, group: str, picks: Sequence[tuple[Layer, str]],
                 provenance: dict[str, Any] | None = None) -> None:
        self.store = store
        self.group = str(group)
        self.picks = [(lay, str(ch)) for lay, ch in picks]
        self.provenance = dict(provenance or {})

    def layers(self) -> list[Layer]:
        return [lay for lay, _ in self.picks]

    def covers(self, origin: Sequence[int], dims: Sequence[int]) -> bool:
        return self.store.covers(origin, dims)

    def why(self) -> str:
        return self.store.path

    def read(self, origin: Sequence[int], dims: Sequence[int], budget: Budget) -> Iterator[tuple[Layer, np.ndarray]]:
        for lay, ch in self.picks:
            yield lay, self.store.read_global(ch, origin, dims, budget)


class CTSource(Source):
    """``ct.u8`` from the config's volume reader (cache, primary URL, then alt_url)."""

    def __init__(self, reader: Any, group: str = "ct") -> None:
        self.reader = reader
        self.group = str(group)
        self.provenance = {"ct": getattr(reader, "url", "")}

    def layers(self) -> list[Layer]:
        return [Layer("ct", self.group, "ct")]

    def covers(self, origin: Sequence[int], dims: Sequence[int]) -> bool:
        shape = tuple(int(s) for s in self.reader.shape)
        return all(0 <= int(origin[a]) and int(origin[a]) + int(dims[a]) <= shape[a] for a in range(3))

    def why(self) -> str:
        return str(getattr(self.reader, "url", "ct"))

    def read(self, origin: Sequence[int], dims: Sequence[int], budget: Budget) -> Iterator[tuple[Layer, np.ndarray]]:
        z0, y0, x0 = (int(v) for v in origin)
        dz, dy, dx = (int(v) for v in dims)
        ct = self.reader.read(z0, z0 + dz, y0, y0 + dy, x0, x0 + dx)
        yield self.layers()[0], np.ascontiguousarray(np.asarray(ct, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# the student on a box (GPU only)
# --------------------------------------------------------------------------- #
class _BoxSink:
    """``run_sliding`` writer that keeps the one box in memory instead of a zarr store."""

    def __init__(self, channels: Sequence[str], origin: Sequence[int], dims: Sequence[int],
                 budget: Budget) -> None:
        self.channels = list(channels)
        self.origin = tuple(int(v) for v in origin)
        check_alloc((len(self.channels), *(int(v) for v in dims)), np.uint8, budget)
        self.a = np.zeros((len(self.channels), *(int(v) for v in dims)), np.uint8)

    def write(self, c: int, z0: int, y0: int, x0: int, a: np.ndarray) -> None:
        d = [int(v) - o for v, o in zip((z0, y0, x0), self.origin)]
        s = a.shape
        self.a[int(c), d[0]:d[0] + s[0], d[1]:d[1] + s[1], d[2]:d[2] + s[2]] = a

    def has_brick(self, z0: int, y0: int, x0: int) -> bool:
        return False  # nothing is ever resumed: a live box is computed from scratch


class StudentSource(Source):
    """The student checkpoint run on the requested box, CUDA only.

    The model, its TTA mode and the sliding-window spec are built once (the server holds one
    instance for its lifetime) and every request re-runs only ``run_sliding`` into a
    :class:`_BoxSink`.  The derived channels the offline ``tsm infer`` writes in a second pass
    (``surface*1``, ``thickness``) are computed here from the box itself -- without the
    brick halo of the offline pass, so the thin-surface channels may differ by at most one
    voxel at the box faces; the viewer draws the SDF zero crossing anyway.

    The laptop GPU must never be used, so a missing CUDA device is a hard error rather than a
    silent CPU fallback (a 512**3 student pass on CPU would take hours)."""

    def __init__(self, checkpoint: str, cfg: RunCfg, group: str = "student", tta: str = "none",
                 clip: float = CLIP, patch: int = 128, out_tile: int = 192, batch: int = 1,
                 log: Log | None = None) -> None:
        import torch

        from tsm.infer import (INFER_DEFAULTS, _checkpoint_fiber, _checkpoint_fiber_mode,
                               _checkpoint_surface_mode, StudentNet, load_student, n_head_ch,
                               pred_channels, tta_transforms)
        from tsm.sliding import WindowSpec

        log = log or _log
        if not torch.cuda.is_available():
            raise RuntimeError(
                "the student needs a CUDA device (this machine has none); run the exporter with "
                "--pred, or run `tsm serve` on forlindesk2 / the A6000 and tunnel the port")
        self.group = str(group)
        self.cfg = cfg
        self.clip = float(clip)
        self.checkpoint = os.path.abspath(os.path.expanduser(str(checkpoint)))
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"student checkpoint {self.checkpoint} not found")
        self.tta = str(tta or "none")
        tta_transforms(self.tta)  # validated here, applied inside StudentNet
        smode = _checkpoint_surface_mode(self.checkpoint)
        fiber, fmode = _checkpoint_fiber(self.checkpoint), _checkpoint_fiber_mode(self.checkpoint)
        self.channels = pred_channels(smode, fiber, fmode)
        self.n_head = n_head_ch(smode, fiber, fmode)
        self.surface_mode = smode
        t0 = time.perf_counter()
        model, info = load_student(self.checkpoint, "cuda")
        self.model = model
        self.info = info
        self.net = StudentNet(model, float(cfg.volume.voxel_um), self.clip, tta=self.tta,
                              surface_mode=info.get("surface_mode", smode),
                              input_radial=bool(info.get("input_radial", False)),
                              axis=info.get("axis_path"), input_axis=bool(info.get("input_axis", False)),
                              axis_tangent=bool(info.get("axis_tangent", False)),
                              fiber_mode=str(info.get("fiber_mode", fmode))).to("cuda")
        self.spec = WindowSpec(patch=int(patch), step=int(patch) // 2, out_tile=int(out_tile),
                               halo=None, batch=int(batch), tta=False, dtype=torch.bfloat16,
                               norm_scope="window", prefetch=False)
        self.provenance = {"checkpoint": self.checkpoint, "step": int(info.get("step", -1)),
                           "tta": self.tta, "surface_mode": smode, "clip": self.clip}
        log(f"student {os.path.basename(self.checkpoint)} step {info.get('step')} on cuda "
            f"({time.perf_counter() - t0:.1f}s), channels={self.channels}")

    def layers(self) -> list[Layer]:
        out = []
        for ch in self.channels:
            lay = layer_for(ch, self.group, self.clip)
            if lay is not None:
                out.append(lay)
        return out

    def read(self, origin: Sequence[int], dims: Sequence[int], budget: Budget) -> Iterator[tuple[Layer, np.ndarray]]:
        from tsm.cli import open_ct
        from tsm.config import RegionCfg
        from tsm.infer import extract_surface
        from tsm.limits import free_cuda
        from tsm.sliding import run_sliding
        from tsm.student import normalize_ct

        region = RegionCfg(start_zyx=tuple(int(v) for v in origin), size_zyx=tuple(int(v) for v in dims))
        sink = _BoxSink(self.channels[:self.n_head], origin, dims, budget)
        reader = open_ct(self.cfg, 0)
        try:
            run_sliding(reader, self.net, normalize_ct, self.spec, region, sink, self.n_head, "none",
                        budget, device="cuda", log=lambda m: _log(m.replace("[sliding]", "[view]")))
        finally:
            free_cuda()
        vals = {name: sink.a[i] for i, name in enumerate(sink.channels)}
        # the derived channels of `tsm infer`'s second pass, on this box alone
        val = vals.get("valid", np.full(tuple(int(v) for v in dims), 255, np.uint8))
        for surf, sdf in (("surface1", "sdf"), ("surface_in1", "sdf_in"), ("surface_out1", "sdf_out")):
            if surf in self.channels and sdf in vals:
                vals[surf] = (extract_surface(vals[sdf], val, self.clip).astype(np.uint8) * 255)
        if "thickness" in self.channels and "sdf_in" in vals and "sdf_out" in vals:
            from tsm.data import decode_sdf

            data = (vals["sdf_in"] != 0) & (vals["sdf_out"] != 0)
            th = decode_sdf(vals["sdf_in"], self.clip) - decode_sdf(vals["sdf_out"], self.clip)
            vals["thickness"] = np.where(data, np.clip(np.rint(th), 0, 255), 0).astype(np.uint8)
        for lay in self.layers():
            yield lay, np.ascontiguousarray(vals.get(lay.name, np.zeros(tuple(int(v) for v in dims), np.uint8)))


# --------------------------------------------------------------------------- #
# the standard sources of spec/view.md
# --------------------------------------------------------------------------- #
def pred_source(path: str, group: str = "student", clip: float | None = None,
                level0_um: float = 2.4) -> StoreSource:
    """Every kinded channel of a ``pred.zarr``, as the ``student`` group.

    The clip comes from the run's ``pred.summary.json`` (``tsm.infer._pred_clip``), never
    guessed: the SDF byte decoding is meaningless without it."""
    from tsm.infer import PRED_CHANNELS, _pred_clip, _pred_summary

    st = Store(path, level0_um, channels=PRED_CHANNELS)
    c = _pred_clip(st.path, clip)
    picks = [(lay, ch) for ch in st.channels if (lay := layer_for(ch, group, c)) is not None]
    summary = _pred_summary(st.path)
    student = summary.get("student") or {}
    prov = {"pred": st.path, "clip": c}
    if student.get("checkpoint"):
        prov["checkpoint"] = student["checkpoint"]
        prov["step"] = student.get("step")
    if summary.get("tta"):
        prov["tta"] = summary["tta"]
    return StoreSource(st, group, picks, prov)


#: teacher store -> the channels spec/view.md puts in the ``teacher`` group.  ``lasagna.zarr``
#: is a 9.6 um store: only ``cos`` is taken and it is nearest-upsampled by :class:`Store`.
TEACHER_PICKS: dict[str, tuple[str, ...]] = {
    "recto.zarr": ("recto",),
    "ink.zarr": ("ink",),
    "fiber.zarr": ("fiber_vt", "fiber_hz"),
    "lasagna.zarr": ("cos",),
}


def teacher_sources(teachers_dir: str, group: str = "teacher", level0_um: float = 2.4,
                    log: Log | None = None) -> list[StoreSource]:
    """One :class:`StoreSource` per teacher store present in ``teachers_dir``.

    Each store is its own coverage unit: the teachers of a run are written independently and
    a box may be inside one and outside another."""
    log = log or _log
    root = os.path.abspath(os.path.expanduser(str(teachers_dir)))
    out: list[StoreSource] = []
    for fname, chans in TEACHER_PICKS.items():
        p = os.path.join(root, fname)
        if not os.path.exists(os.path.join(p, "zarr.json")):
            continue
        st = Store(p, level0_um)
        picks = [(lay, ch) for ch in chans if st.has(ch) and (lay := layer_for(ch, group)) is not None]
        if not picks:
            log(f"{p}: none of {chans} present (channels={st.channels}), skipped")
            continue
        out.append(StoreSource(st, group, picks, {f"teacher_{fname}": st.path}))
    if not out:
        log(f"{root}: no teacher store found ({sorted(TEACHER_PICKS)})")
    return out


#: label-store channels of the ``label`` group (spec/view.md: sdf_in/out, faces_valid, ink)
LABEL_PICKS = ("sdf_in", "sdf_out", "faces_valid", "ink")


def label_source(path: str, group: str = "label", clip: float = CLIP,
                 level0_um: float = 2.4) -> StoreSource:
    """The face labels of a ``fine.zarr`` as the ``label`` group."""
    st = Store(path, level0_um)
    picks = [(lay, ch) for ch in LABEL_PICKS if st.has(ch) and (lay := layer_for(ch, group, clip)) is not None]
    if not picks:
        raise ValueError(f"{st.path} has none of {LABEL_PICKS} (channels={st.channels})")
    return StoreSource(st, group, picks, {"labels": st.path})


#: rectoverso.zarr channel name -> the class channel name of the fine store / the viewer
RV_ALIAS = {"rectoverso": "rv_class", "rv_class": "rv_class",
            "hzvt": "hzvt_class", "hzvt_class": "hzvt_class"}
#: positional names for a slab written before the ``channels`` attribute existed
RV_ALIAS_ORDER = ["rectoverso", "hzvt"]


def rectoverso_source(path: str, group: str = "human", level0_um: float = 2.4) -> StoreSource:
    """The upstream human bands of ``rectoverso.zarr`` as the ``human`` group.

    ``dev/rectoverso_slab.py`` writes channel 0 as ``rectoverso`` and channel 1 as ``hzvt``,
    and an older slab has no ``channels`` attribute at all; both are mapped onto the
    ``rv_class`` / ``hzvt_class`` names the fine store (and the viewer) use, so the same layer
    means the same thing whichever store it came from."""
    st = Store(path, level0_um, channels=RV_ALIAS_ORDER)
    picks = []
    for ch in st.channels:
        alias = RV_ALIAS.get(ch)
        lay = layer_for(alias, group) if alias else None
        if lay is not None:
            picks.append((lay, ch))
    if not picks:
        raise ValueError(f"{st.path}: no recto/verso band channel among {st.channels}")
    return StoreSource(st, group, picks, {"rectoverso": st.path})


# --------------------------------------------------------------------------- #
# packet assembly
# --------------------------------------------------------------------------- #
def check_dims(dims: Sequence[int]) -> tuple[int, int, int]:
    d = tuple(int(v) for v in dims)
    if len(d) != 3 or min(d) <= 0 or max(d) > MAX_DIM:
        raise ValueError(f"dims_zyx must be 3 positive ints <= {MAX_DIM}, got {list(dims)}")
    return d  # type: ignore[return-value]


def plan(sources: Sequence[Source], origin: Sequence[int], dims: Sequence[int],
         log: Log | None = None) -> list[Source]:
    """Drop the sources that do not fully cover the box, with a warning.

    Never pads: a partially covered store would hand the viewer fabricated zeros, which read
    exactly like "no data" in every kind's decoding."""
    log = log or _log
    kept: list[Source] = []
    for s in sources:
        if not s.layers():
            continue
        if s.covers(origin, dims):
            kept.append(s)
        else:
            log(f"WARNING {s.why()} does not cover box origin={list(map(int, origin))} "
                f"dims={list(map(int, dims))}: dropping {len(s.layers())} {s.group} layer(s)")
    return kept


def packet_meta(sources: Sequence[Source], origin: Sequence[int], dims: Sequence[int],
                voxel_um: float, scroll: str = "", provenance: dict[str, Any] | None = None
                ) -> dict[str, Any]:
    """``meta.json`` for the (already coverage-filtered) sources, layers in source order."""
    layers = [lay.meta() for s in sources for lay in s.layers()]
    if sum(1 for lay in layers if lay["kind"] == "ct") != 1:
        raise ValueError("a view packet needs exactly one 'ct' layer")
    prov: dict[str, Any] = {}
    for s in sources:
        prov.update(s.provenance)
    prov.update(provenance or {})
    return {
        "format": FORMAT,
        "dims_zyx": [int(v) for v in dims],
        "origin_zyx": [int(v) for v in origin],
        "voxel_um": float(voxel_um),
        "scroll": str(scroll),
        "layers": layers,
        "provenance": prov,
    }


def iter_arrays(sources: Sequence[Source], origin: Sequence[int], dims: Sequence[int],
                budget: Budget) -> Iterator[tuple[Layer, np.ndarray]]:
    """Every layer's bytes in manifest order, one box at a time (never all at once)."""
    d = check_dims(dims)
    for s in sources:
        want = s.layers()
        got = 0
        for lay, a in s.read(origin, d, budget):
            if a.shape != d or a.dtype != np.uint8:
                raise ValueError(f"{lay.group}.{lay.name}: got {a.shape} {a.dtype}, expected {d} uint8")
            got += 1
            yield lay, a
        if got != len(want):
            raise ValueError(f"{s.group}: {got} arrays for {len(want)} layers")


def write_packet_dir(out_dir: str, sources: Sequence[Source], origin: Sequence[int],
                     dims: Sequence[int], voxel_um: float, budget: Budget, scroll: str = "",
                     provenance: dict[str, Any] | None = None, log: Log | None = None
                     ) -> dict[str, Any]:
    """Write ``meta.json`` + one ``.u8`` per layer into ``out_dir``; returns the meta."""
    log = log or _log
    os.makedirs(out_dir, exist_ok=True)
    meta = packet_meta(sources, origin, dims, voxel_um, scroll, provenance)
    n = int(np.prod([int(v) for v in dims]))
    for lay, a in iter_arrays(sources, origin, dims, budget):
        p = os.path.join(out_dir, lay.file)
        with open(p, "wb") as fh:
            fh.write(np.ascontiguousarray(a).tobytes())
        if os.path.getsize(p) != n:
            raise IOError(f"{p}: wrote {os.path.getsize(p)} bytes, expected {n}")
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    log(f"{out_dir}: {len(meta['layers'])} layers of {tuple(int(v) for v in dims)} "
        f"({len(meta['layers']) * n / (1 << 20):.1f} MiB)")
    return meta
