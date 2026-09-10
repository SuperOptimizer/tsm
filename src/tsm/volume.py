"""Zarr volume access: reader with edge padding, brick iterator, brick writer."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import zarr

from tsm.config import RegionCfg
from tsm.limits import Budget, check_alloc, zeros

VOLCOMP_PKG = (
    "volcomp_zarr: uv pip install "
    "'git+https://github.com/SuperOptimizer/volume-compressor.git#subdirectory=python', "
    "then build libvolcomp.so (cmake --preset release --target volcomp_shim) and copy it "
    "next to volcomp_zarr/__init__.py or set VOLCOMP_LIB"
)
_VOLCOMP_ERROR = ""


def _try_register_volcomp() -> bool:
    """Import volcomp_zarr (which registers the codec). Never raises."""
    global _VOLCOMP_ERROR
    try:
        import volcomp_zarr  # noqa: F401
    except Exception as exc:  # missing package, or missing/broken libvolcomp.so
        _VOLCOMP_ERROR = f"{type(exc).__name__}: {exc}"
        return False
    _VOLCOMP_ERROR = ""
    return True


# Chunk fetch concurrency for remote stores.  zarr issues one async GET per
# chunk and caps the number in flight at ``zarr.config['async.concurrency']``
# (default 10); a 512x640x640 box at 128^3 chunks is 100 objects, so a higher
# cap keeps more requests in flight.  ``TSM_READ_CONCURRENCY`` overrides.
READ_CONCURRENCY = int(os.environ.get("TSM_READ_CONCURRENCY", "32"))


def set_read_concurrency(n: int) -> None:
    global READ_CONCURRENCY
    READ_CONCURRENCY = max(1, int(n))
    zarr.config.set({"async.concurrency": READ_CONCURRENCY})


class VolumeReadError(RuntimeError):
    """A chunk fetch failed after every retry (usually a network/S3 outage)."""


# Transient-failure retry for remote chunk fetches.  s3fs/aiobotocore surface
# connection drops as EndpointConnectionError / ClientConnectorError / OSError,
# and zarr sometimes re-wraps them, so we also sniff the message text.
READ_RETRIES = int(os.environ.get("TSM_READ_RETRIES", "6"))
RETRY_BASE_S = float(os.environ.get("TSM_READ_RETRY_BASE", "1.0"))
RETRY_MAX_S = float(os.environ.get("TSM_READ_RETRY_MAX", "32.0"))

_TRANSIENT_SUBSTRINGS = (
    "endpoint", "connection", "connect", "timeout", "timed out", "reset by peer",
    "broken pipe", "temporarily", "throttl", "slowdown", "service unavailable",
    "incomplete read", "eof occurred", "ssl", "502", "503", "504",
)


def _transient_types() -> tuple[type[BaseException], ...]:
    """(botocore, aiohttp, builtin) exception types worth retrying; import-safe."""
    types: list[type[BaseException]] = [OSError, ConnectionError, TimeoutError]
    try:
        import botocore.exceptions as be

        types += [be.BotoCoreError, be.ClientError]
    except Exception:
        pass
    try:
        import aiohttp

        types.append(aiohttp.ClientError)
    except Exception:
        pass
    return tuple(types)


def is_transient(exc: BaseException) -> bool:
    """True if ``exc`` looks like a network hiccup rather than a real error."""
    from tsm.limits import BudgetError

    if isinstance(exc, (BudgetError, MemoryError, KeyboardInterrupt, SystemExit)):
        return False
    if isinstance(exc, _transient_types()):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(w in text for w in _TRANSIENT_SUBSTRINGS):
        return True
    cause = exc.__cause__ or exc.__context__
    return bool(cause is not None and cause is not exc and is_transient(cause))


def retry_fetch(
    fn: Callable[[], Any],
    what: str,
    attempts: int | None = None,
    base: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call ``fn`` with exponential backoff (1, 2, 4 ... s) on transient failures."""
    n = max(1, int(READ_RETRIES if attempts is None else attempts))
    b = float(RETRY_BASE_S if base is None else base)
    last: BaseException | None = None
    for attempt in range(1, n + 1):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            last = exc
            if not is_transient(exc) or attempt >= n:
                break
            delay = min(b * (2 ** (attempt - 1)), RETRY_MAX_S)
            print(
                f"[tsm] read retry {attempt}/{n - 1} in {delay:.0f}s for {what}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            sleep(delay)
    assert last is not None
    if not is_transient(last):
        raise last
    raise VolumeReadError(
        f"read failed after {n} attempts: {what} ({type(last).__name__}: {last})"
    ) from last


def _make_store(url: str) -> Any:
    if url.startswith(("http://", "https://", "s3://")):
        zarr.config.set({"async.concurrency": READ_CONCURRENCY})
    if url.startswith(("http://", "https://")):
        return zarr.storage.FsspecStore.from_url(url, read_only=True)
    if url.startswith("s3://"):
        try:
            import s3fs  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"reading {url} needs the s3fs package (pip install s3fs)"
            ) from exc
        return zarr.storage.FsspecStore.from_url(
            url, storage_options={"anon": True, "default_fill_cache": False,
                                  "default_cache_type": "none"},
            read_only=True,
        )
    path = os.path.abspath(os.path.expanduser(url))
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such volume path: {path}")
    return zarr.storage.LocalStore(path, read_only=True)


def _open_node(store: Any) -> Any:
    try:
        return zarr.open_array(store=store, mode="r")
    except Exception:
        return zarr.open_group(store=store, mode="r")


def open_array(url: str, level: int = 0) -> zarr.Array:
    """Open a zarr array, descending into an OME-Zarr multiscale group if needed."""
    _try_register_volcomp()
    store = _make_store(url)
    try:
        node = _open_node(store)
        if isinstance(node, zarr.Group):
            key = str(level)
            if key not in node:
                keys = sorted(node.array_keys())
                raise KeyError(f"level {key!r} not in group {url}; arrays present: {keys}")
            node = node[key]
        if not isinstance(node, zarr.Array):
            raise TypeError(f"{url} level {level} is not a zarr array")
        codecs = str(getattr(node.metadata, "codecs", "")) + str(
            getattr(node.metadata, "compressor", "")
        )
        if "volcomp" in codecs and not _try_register_volcomp():
            raise RuntimeError(f"{url} uses the 'volcomp' codec but it is unavailable ({_VOLCOMP_ERROR}); {VOLCOMP_PKG}")
        return node
    except Exception as exc:
        if "volcomp" in str(exc) and not _try_register_volcomp():
            raise RuntimeError(f"{url} uses the 'volcomp' codec but it is unavailable ({_VOLCOMP_ERROR}); {VOLCOMP_PKG}") from exc
        raise


def codec_names(arr: zarr.Array) -> str:
    md = arr.metadata
    for attr in ("codecs", "compressor", "compressors"):
        v = getattr(md, attr, None)
        if v:
            return str(v)
    return "none"


class VolumeReader:
    """uint8 view of a 3D zarr array; reads are clipped to bounds and zero-padded."""

    def __init__(
        self,
        url: str,
        level: int = 0,
        voxel_um: float = 2.4,
        budget: Budget | None = None,
        array: Any = None,
    ) -> None:
        self.url = url
        self.level = level
        self.voxel_um = float(voxel_um)
        self.budget = budget or Budget()
        # ``array`` injects an already-open (or local-cache) array; see
        # ``LocalRegionArray`` / ``open_cached_reader``.
        self.array = open_array(url, level) if array is None else array
        if self.array.ndim < 3:
            raise ValueError(f"{url} has ndim {self.array.ndim}, need >= 3")
        self._lead = self.array.ndim - 3
        self.shape: tuple[int, int, int] = tuple(int(s) for s in self.array.shape[self._lead :])  # type: ignore[assignment]
        self.chunks: tuple[int, int, int] = tuple(int(c) for c in self.array.chunks[self._lead :])  # type: ignore[assignment]
        self.dtype = np.dtype(self.array.dtype)

    def __repr__(self) -> str:
        return f"VolumeReader({self.url!r}, level={self.level}, shape={self.shape}, chunks={self.chunks}, dtype={self.dtype.name})"

    def _to_u8(self, a: np.ndarray) -> np.ndarray:
        if a.dtype == np.uint8:
            return a
        if a.dtype == np.uint16:
            return (a >> 8).astype(np.uint8)
        if np.issubdtype(a.dtype, np.floating):
            return np.clip(a * 255.0, 0, 255).astype(np.uint8)
        return np.clip(a, 0, 255).astype(np.uint8)

    def read(self, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        req = (int(z1 - z0), int(y1 - y0), int(x1 - x0))
        if min(req) <= 0:
            raise ValueError(f"empty read box {(z0, z1, y0, y1, x0, x1)}")
        check_alloc(req, np.uint8, self.budget)
        lo = [max(0, int(v)) for v in (z0, y0, x0)]
        hi = [min(int(s), int(v)) for s, v in zip(self.shape, (z1, y1, x1))]
        out = zeros(req, np.uint8, self.budget)
        if any(h <= l for l, h in zip(lo, hi)):
            return out
        sel: tuple[Any, ...] = (0,) * self._lead + tuple(
            slice(l, h) for l, h in zip(lo, hi)
        )
        check_alloc([h - l for l, h in zip(lo, hi)], self.dtype, self.budget)
        what = f"{self.url} L{self.level} z{lo[0]}:{hi[0]} y{lo[1]}:{hi[1]} x{lo[2]}:{hi[2]}"
        block = np.asarray(retry_fetch(lambda: self.array[sel], what))
        dz, dy, dx = (lo[0] - z0, lo[1] - y0, lo[2] - x0)
        out[
            dz : dz + block.shape[0], dy : dy + block.shape[1], dx : dx + block.shape[2]
        ] = self._to_u8(block)
        return out


@dataclass
class Brick:
    index_zyx: tuple[int, int, int]
    z0: int
    y0: int
    x0: int
    core_shape: tuple[int, int, int]
    data: np.ndarray  # core + halo on every side, zero padded

    @property
    def halo(self) -> int:
        return (self.data.shape[0] - self.core_shape[0]) // 2


def iter_bricks(
    reader: VolumeReader, region: RegionCfg, brick: int, halo: int = 0
) -> Iterator[Brick]:
    """Tile `region` with `brick`-sized cores (last ones clipped); one brick live."""
    if brick <= 0 or halo < 0:
        raise ValueError("brick must be positive and halo non-negative")
    start = tuple(int(v) for v in region.start_zyx)
    size = tuple(int(v) for v in region.size_zyx)
    counts = [(size[i] + brick - 1) // brick for i in range(3)]
    for iz in range(counts[0]):
        for iy in range(counts[1]):
            for ix in range(counts[2]):
                idx = (iz, iy, ix)
                core0 = [start[i] + idx[i] * brick for i in range(3)]
                core1 = [min(core0[i] + brick, start[i] + size[i]) for i in range(3)]
                core_shape = (core1[0] - core0[0], core1[1] - core0[1], core1[2] - core0[2])
                data = reader.read(
                    core0[0] - halo,
                    core1[0] + halo,
                    core0[1] - halo,
                    core1[1] + halo,
                    core0[2] - halo,
                    core1[2] + halo,
                )
                yield Brick(idx, core0[0], core0[1], core0[2], core_shape, data)
                del data


class WriterMismatch(ValueError):
    """An existing output store does not match the requested writer geometry/identity."""


class BrickWriter:
    """zarr v3 (C,Z,Y,X) uint8 output with a done.json sidecar for resume.

    Reopening an existing store validates shape, chunk, dtype, channel names *and order*,
    origin, voxel size and scale against the request and raises :class:`WriterMismatch`
    otherwise (R04): the object used to advertise the new geometry while the store kept
    the old one, so a rerun with a changed region/channel set silently mixed two runs.
    ``done.json`` carries a ``writer_fingerprint`` of that same identity and its
    completion keys are ignored when a *different* fingerprint is present (a stale
    sidecar cannot skip work that belongs to a different run).  A legacy flat sidecar
    (``{key: [channels]}``, no fingerprint) written before this check existed is still
    honoured when the reopened array passes the validation above -- that validation is
    what the fingerprint stands in for -- and is upgraded to the new format in place, so
    an interrupted production build resumes instead of restarting.
    """

    def __init__(
        self,
        path: str,
        channels: Sequence[str],
        shape_zyx: Sequence[int],
        chunk: int = 128,
        dtype: Any = np.uint8,
        origin_zyx: Sequence[int] = (0, 0, 0),
        voxel_um: float = 2.4,
        scale: float = 1.0,
        brick: int | None = None,
    ) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        self.channels = list(channels)
        self.shape_zyx = tuple(int(s) for s in shape_zyx)
        self.origin_zyx = tuple(int(s) for s in origin_zyx)
        self.chunk = int(chunk)
        self.dtype = np.dtype(dtype)
        self.voxel_um = float(voxel_um)
        self.scale = float(scale)
        self.brick = None if brick is None else int(brick)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        shape = (len(self.channels), *self.shape_zyx)
        chunks = (1, self.chunk, self.chunk, self.chunk)
        attrs = {
            "channels": self.channels,
            "voxel_um": self.voxel_um,
            "origin_zyx": list(self.origin_zyx),
            "scale": self.scale,
        }
        self.reopened = os.path.exists(os.path.join(self.path, "zarr.json"))
        if self.reopened:
            self.array = zarr.open_array(store=self.path, mode="r+")
            self._check_existing(shape, chunks)
        else:
            self.array = zarr.create_array(
                store=self.path,
                shape=shape,
                chunks=chunks,
                dtype=self.dtype,
                fill_value=0,
                attributes=attrs,
                overwrite=False,
            )
        self._done_path = os.path.join(self.path, "done.json")
        self.fingerprint = self._fingerprint()
        self._done: dict[str, list[int]] = self._load_done()

    def _fingerprint(self) -> str:
        payload = json.dumps({
            "shape_zyx": list(self.shape_zyx), "channels": self.channels,
            "origin_zyx": list(self.origin_zyx), "chunk": self.chunk,
            "dtype": self.dtype.name, "brick": self.brick,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _check_existing(self, shape: tuple[int, ...], chunks: tuple[int, ...]) -> None:
        attrs = dict(self.array.attrs)
        got_ch = list(attrs.get("channels", []))
        bad: list[str] = []
        if tuple(int(s) for s in self.array.shape) != tuple(int(s) for s in shape):
            bad.append(f"shape {tuple(int(s) for s in self.array.shape)} != requested {shape}")
        if tuple(int(c) for c in self.array.chunks) != tuple(int(c) for c in chunks):
            bad.append(f"chunk {tuple(int(c) for c in self.array.chunks)} != requested {chunks}")
        if np.dtype(self.array.dtype) != self.dtype:
            bad.append(f"dtype {np.dtype(self.array.dtype).name} != requested {self.dtype.name}")
        if got_ch and got_ch != self.channels:
            bad.append(f"channels {got_ch} != requested {self.channels} (names and order must match)")
        if "origin_zyx" in attrs and tuple(int(v) for v in attrs["origin_zyx"]) != self.origin_zyx:
            bad.append(f"origin_zyx {tuple(int(v) for v in attrs['origin_zyx'])} != requested {self.origin_zyx}")
        if "voxel_um" in attrs and float(attrs["voxel_um"]) != self.voxel_um:
            bad.append(f"voxel_um {float(attrs['voxel_um'])} != requested {self.voxel_um}")
        if "scale" in attrs and float(attrs["scale"]) != self.scale:
            bad.append(f"scale {float(attrs['scale'])} != requested {self.scale}")
        if bad:
            raise WriterMismatch(
                f"{self.path} was written by a different run: " + "; ".join(bad)
                + " -- write to a new out_dir, or delete this store (`--force` does not "
                  "reshape an existing array)")

    @classmethod
    def open_existing(cls, path: str) -> "BrickWriter":
        path = os.path.abspath(os.path.expanduser(path))
        arr = zarr.open_array(store=path, mode="r+")
        attrs = dict(arr.attrs)
        return cls(
            path,
            channels=list(attrs.get("channels", [])) or [str(i) for i in range(arr.shape[0])],
            shape_zyx=tuple(int(s) for s in arr.shape[1:]),
            chunk=int(arr.chunks[1]),
            dtype=arr.dtype,
            origin_zyx=tuple(attrs.get("origin_zyx", (0, 0, 0))),
            voxel_um=float(attrs.get("voxel_um", 2.4)),
            scale=float(attrs.get("scale", 1.0)),
        )

    @staticmethod
    def _bricks(raw: Any) -> dict[str, list[int]]:
        try:
            return {str(k): sorted({int(c) for c in v}) for k, v in dict(raw).items()}
        except (TypeError, ValueError, AttributeError):
            return {}

    def _load_done(self) -> dict[str, list[int]]:
        """Completion keys belonging to *this* writer identity.

        A sidecar carrying a different ``writer_fingerprint`` is ignored.  A legacy flat
        sidecar has no fingerprint to compare, but reaching here means the store itself
        already matched shape/chunk/dtype/channels/origin/voxel_um/scale -- the same facts
        the fingerprint encodes -- so its keys are accepted and the file is upgraded.
        """
        try:
            with open(self._done_path, "r") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return {}
        if not isinstance(d, dict):
            return {}
        if "writer_fingerprint" in d:
            if str(d["writer_fingerprint"]) != self.fingerprint:
                print(f"[tsm] {self._done_path}: written by a different run "
                      f"({d['writer_fingerprint']} != {self.fingerprint}); "
                      f"ignoring its completion keys", flush=True)
                return {}
            return self._bricks(d.get("bricks", {}))
        if not self.reopened:  # a sidecar with no store to validate against proves nothing
            print(f"[tsm] {self._done_path}: legacy sidecar next to a freshly created array; "
                  "ignoring its completion keys", flush=True)
            return {}
        done = self._bricks(d)
        print(f"[tsm] {self._done_path}: legacy sidecar ({len(done)} bricks) accepted -- the store "
              f"matches this run's geometry; upgrading it to the fingerprinted format", flush=True)
        self._done = done
        try:
            self._save_done()
        except OSError as exc:  # read-only sidecar: resume still works, just not upgraded
            print(f"[tsm] {self._done_path}: could not upgrade in place ({exc})", flush=True)
        return done

    def _save_done(self) -> None:
        tmp = self._done_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"writer_fingerprint": self.fingerprint,
                       "shape_zyx": list(self.shape_zyx), "channels": self.channels,
                       "origin_zyx": list(self.origin_zyx), "brick": self.brick,
                       "bricks": self._done}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self._done_path)

    def _key(self, z0: int, y0: int, x0: int) -> str:
        return f"{int(z0)},{int(y0)},{int(x0)}"

    def _local(self, z0: int, y0: int, x0: int) -> tuple[int, int, int]:
        return (z0 - self.origin_zyx[0], y0 - self.origin_zyx[1], x0 - self.origin_zyx[2])

    def write(self, channel_idx: int, z0: int, y0: int, x0: int, arr_u8: np.ndarray) -> None:
        if arr_u8.ndim != 3:
            raise ValueError(f"expected a 3D brick, got shape {arr_u8.shape}")
        lz, ly, lx = self._local(z0, y0, x0)
        if min(lz, ly, lx) < 0:
            raise ValueError(f"brick at {(z0, y0, x0)} is before origin {self.origin_zyx}")
        if not 0 <= int(channel_idx) < len(self.channels):
            raise ValueError(f"channel index {channel_idx} outside 0..{len(self.channels) - 1} "
                             f"({self.channels})")
        dz, dy, dx = arr_u8.shape
        end = (lz + dz, ly + dy, lx + dx)
        # per axis: a tuple comparison stops at the first differing coordinate, so a
        # small Z extent used to hide a Y or X overflow (R12)
        over = [i for i in range(3) if end[i] > self.shape_zyx[i]]
        if over:
            axes = "".join("zyx"[i] for i in over)
            raise ValueError(f"brick at {(z0, y0, x0)} shape {arr_u8.shape} exceeds {self.shape_zyx} "
                             f"on axis {axes} (local end {end})")
        self.array[int(channel_idx), lz : lz + dz, ly : ly + dy, lx : lx + dx] = arr_u8
        chans = self._done.setdefault(self._key(z0, y0, x0), [])
        if channel_idx not in chans:
            chans.append(int(channel_idx))
            chans.sort()
        self._save_done()

    def has_brick(self, z0: int, y0: int, x0: int) -> bool:
        """True once every channel has been written for the core at (z0, y0, x0)."""
        return len(self._done.get(self._key(z0, y0, x0), [])) == len(self.channels)


# --------------------------------------------------------------------------- #
# local cache: a zarr array holding one sub-box of a remote level
# --------------------------------------------------------------------------- #
DEFAULT_CACHE_ROOT = "/home/forrest/tsm-cache"
CACHE_CHUNK = 128
COMPLETE_JSON = "complete.json"


# --------------------------------------------------------------------------- #
# Cache completion manifest (R02)
#
# A cache array is created at its *final* shape before any brick is written, and zarr
# reads unwritten chunks back as zero, so "the array opens and its footprint covers the
# request" cannot distinguish a finished cache from an interrupted one -- an interrupted
# build used to be reported as a HIT and served fake air.  ``complete.json`` is written
# by :mod:`tsm.cache` only after every planned brick is committed, and is *required*
# here.  Caches built before this manifest existed are still accepted when their
# ``done.json`` brick list covers the whole planned brick grid (derived from the stored
# ``size_zyx`` and the brick pitch implied by the keys); the manifest is then written so
# the next open is a cheap file check.
# --------------------------------------------------------------------------- #
def _brick_grid(size: Sequence[int], brick: int) -> list[tuple[int, int, int]]:
    return [(z, y, x)
            for z in range(0, int(size[0]), brick)
            for y in range(0, int(size[1]), brick)
            for x in range(0, int(size[2]), brick)]


def _infer_brick(keys: Sequence[str], size: Sequence[int]) -> int:
    """Smallest positive brick origin seen on any axis = the brick pitch of that run.

    Returns 0 when no key carries a positive coordinate: a lone ``0,0,0`` is equally
    consistent with "one brick spanning the whole box" and with "the first of many
    bricks, interrupted", so the cache cannot be *proven* complete from it.
    """
    steps: set[int] = set()
    for k in keys:
        try:
            steps.update(v for v in (int(c) for c in k.split(",")) if v > 0)
        except ValueError:
            return 0
    return min(steps) if steps else 0


def read_complete(path: str) -> dict[str, Any] | None:
    try:
        with open(os.path.join(path, COMPLETE_JSON)) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def write_complete(path: str, level: int, origin_zyx: Sequence[int], size_zyx: Sequence[int],
                   brick: int, source_url: str = "", extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Publish the completion manifest for a fully written cache level."""
    info = {
        "levels": [int(level)],
        "level": int(level),
        "origin_zyx": [int(v) for v in origin_zyx],
        "size_zyx": [int(v) for v in size_zyx],
        "brick": int(brick),
        "source_url": str(source_url),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **(dict(extra) if extra else {}),
    }
    tmp = os.path.join(path, COMPLETE_JSON + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(info, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, os.path.join(path, COMPLETE_JSON))
    return info


def cache_completion(path: str, attrs: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """``complete.json`` for a cache array, upgrading a legacy full ``done.json`` in place.

    Returns None (with the reason logged) when the cache cannot be proven complete.
    """
    info = read_complete(path)
    if info is not None:
        return info
    if attrs is None:
        try:
            attrs = dict(zarr.open_array(store=path, mode="r").attrs)
        except Exception as exc:
            print(f"[tsm] cache {path}: cannot read attrs ({type(exc).__name__}: {exc})", flush=True)
            return None
    size = attrs.get("size_zyx")
    origin = attrs.get("origin_zyx", (0, 0, 0))
    if not size:
        print(f"[tsm] cache {path}: no completion manifest and no size_zyx to verify against",
              flush=True)
        return None
    try:
        with open(os.path.join(path, "done.json")) as fh:
            keys = json.load(fh)
        keys = [str(k) for k in keys]
    except (OSError, ValueError, TypeError):
        print(f"[tsm] cache {path}: no {COMPLETE_JSON} and no readable done.json "
              "(an interrupted build reads back as zeros)", flush=True)
        return None
    brick = _infer_brick(keys, size)
    if brick <= 0:
        print(f"[tsm] cache {path}: no {COMPLETE_JSON}, and done.json ({len(keys)} key(s)) does not "
              "pin down a brick pitch, so its coverage of the box cannot be verified", flush=True)
        return None
    grid = _brick_grid(size, brick)
    if len(set(keys)) != len(grid):
        print(f"[tsm] cache {path}: incomplete -- done.json lists {len(set(keys))} bricks but the "
              f"{brick}^3 grid over {tuple(int(v) for v in size)} has {len(grid)}", flush=True)
        return None
    missing = [g for g in grid if f"{g[0]},{g[1]},{g[2]}" not in set(keys)]
    if missing:
        print(f"[tsm] cache {path}: incomplete -- {len(missing)}/{len(grid)} bricks of the "
              f"{brick}^3 grid are missing from done.json (e.g. {missing[0]})", flush=True)
        return None
    print(f"[tsm] cache {path}: done.json covers the full {len(grid)}-brick grid; "
          f"writing {COMPLETE_JSON}", flush=True)
    try:
        return write_complete(path, int(attrs.get("level", 0)), origin, size, brick,
                              str(attrs.get("source_url", "")), {"derived_from": "done.json"})
    except OSError as exc:  # read-only cache root: still a hit, just not memoised
        print(f"[tsm] cache {path}: could not write {COMPLETE_JSON} ({exc})", flush=True)
        return {"levels": [int(attrs.get("level", 0))], "origin_zyx": [int(v) for v in origin],
                "size_zyx": [int(v) for v in size], "brick": int(brick)}


def cache_volume_name(url: str) -> str:
    """Stable directory name for a volume URL (primary and alt_url agree)."""
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    if tail.endswith(".zarr"):
        tail = tail[: -len(".zarr")]
    return tail or "volume"


def cache_dir(cache_root: str, url: str) -> str:
    return os.path.join(os.path.abspath(os.path.expanduser(cache_root)), cache_volume_name(url))


def cache_array_path(cache_root: str, url: str, level: int, origin_zyx: Sequence[int] | None = None) -> str:
    """``<cache_root>/<volume>/L<level>.zarr``; a second, disjoint region at the same
    level gets an origin suffix so it does not clobber the first."""
    d = cache_dir(cache_root, url)
    base = os.path.join(d, f"L{int(level)}.zarr")
    if origin_zyx is None:
        return base
    origin = tuple(int(v) for v in origin_zyx)
    meta = os.path.join(base, "zarr.json")
    if not os.path.exists(meta):
        return base
    try:
        existing = tuple(int(v) for v in dict(zarr.open_array(store=base, mode="r").attrs)["origin_zyx"])
    except Exception:
        return base
    if existing == origin:
        return base
    return os.path.join(d, "L{}_{}_{}_{}.zarr".format(int(level), *origin))


def cache_candidates(cache_root: str, url: str, level: int) -> list[str]:
    """Every cached array for this volume level, plain path first."""
    d = cache_dir(cache_root, url)
    base = os.path.join(d, f"L{int(level)}.zarr")
    out = [base] if os.path.exists(os.path.join(base, "zarr.json")) else []
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    for n in names:
        if n.startswith(f"L{int(level)}_") and n.endswith(".zarr"):
            p = os.path.join(d, n)
            if os.path.exists(os.path.join(p, "zarr.json")):
                out.append(p)
    return out


class LocalRegionArray:
    """zarr-Array-like view of a cached sub-box, indexed in ABSOLUTE level coords.

    ``shape`` is the full remote level shape, so ``VolumeReader`` clips exactly as it
    would against the remote array; voxels inside the volume but outside the cached
    footprint read back as zero (``strict=True`` raises instead).
    """

    def __init__(
        self,
        array: Any,
        origin_zyx: Sequence[int],
        full_shape_zyx: Sequence[int] | None = None,
        strict: bool = False,
        path: str = "",
    ) -> None:
        self.array = array
        self.path = path
        self.strict = bool(strict)
        self.origin = tuple(int(v) for v in origin_zyx)
        self.local_shape = tuple(int(s) for s in array.shape[-3:])
        self.stop = tuple(o + s for o, s in zip(self.origin, self.local_shape))
        self.shape = (
            tuple(int(s) for s in full_shape_zyx)
            if full_shape_zyx is not None
            else self.stop
        )
        self.ndim = 3
        self.chunks = tuple(int(c) for c in array.chunks[-3:])
        self.dtype = np.dtype(array.dtype)

    def __repr__(self) -> str:
        return (f"LocalRegionArray({self.path!r}, origin={self.origin}, "
                f"size={self.local_shape}, shape={self.shape})")

    def covers(self, lo: Sequence[int], hi: Sequence[int]) -> bool:
        return all(int(l) >= o and int(h) <= s for l, h, o, s in zip(lo, hi, self.origin, self.stop))

    def __getitem__(self, sel: Any) -> np.ndarray:
        if not isinstance(sel, tuple):
            sel = (sel,)
        sel = tuple(s for s in sel if not isinstance(s, int))  # drop leading channel indices
        if len(sel) != 3 or not all(isinstance(s, slice) for s in sel):
            raise TypeError(f"LocalRegionArray takes 3 slices in absolute coords, got {sel!r}")
        lo = [int(s.start or 0) for s in sel]
        hi = [int(s.stop if s.stop is not None else d) for s, d in zip(sel, self.shape)]
        out = np.zeros([max(0, h - l) for l, h in zip(lo, hi)], dtype=self.dtype)
        if self.strict and not self.covers(lo, hi):
            raise VolumeReadError(
                f"request z{lo[0]}:{hi[0]} y{lo[1]}:{hi[1]} x{lo[2]}:{hi[2]} is outside the "
                f"cached box origin={self.origin} size={self.local_shape} ({self.path})")
        clo = [max(l, o) for l, o in zip(lo, self.origin)]
        chi = [min(h, e) for h, e in zip(hi, self.stop)]
        if any(h <= l for l, h in zip(clo, chi)):
            return out
        block = np.asarray(self.array[tuple(
            slice(l - o, h - o) for l, h, o in zip(clo, chi, self.origin))])
        d = [l - a for l, a in zip(clo, lo)]
        out[d[0]:d[0] + block.shape[0], d[1]:d[1] + block.shape[1], d[2]:d[2] + block.shape[2]] = block
        return out


def open_local_region(path: str, strict: bool = False) -> LocalRegionArray:
    """Open a cache array written by the ``cache`` stage (attrs carry the origin)."""
    arr = zarr.open_array(store=os.path.abspath(os.path.expanduser(path)), mode="r")
    attrs = dict(arr.attrs)
    full = attrs.get("full_shape_zyx")
    return LocalRegionArray(arr, attrs.get("origin_zyx", (0, 0, 0)), full, strict=strict, path=path)


def open_cached_reader(
    cache_root: str | None,
    url: str,
    level: int,
    voxel_um: float,
    region: RegionCfg | None = None,
    budget: Budget | None = None,
) -> VolumeReader | None:
    """A VolumeReader backed by the local cache, or None if it is absent/too small."""
    if not cache_root:
        return None
    paths = cache_candidates(cache_root, url, level)
    if not paths:
        return None
    for path in paths:
        try:
            loc = open_local_region(path)
        except Exception as exc:
            print(f"[tsm] cache {path}: unusable ({type(exc).__name__}: {exc})", flush=True)
            continue
        if region is not None and not loc.covers(region.start_zyx, region.stop_zyx):
            continue
        info = cache_completion(path)
        if info is None:
            print(f"[tsm] cache {path}: not proven complete; treating as a MISS", flush=True)
            continue
        c_lo = tuple(int(v) for v in info.get("origin_zyx", loc.origin))
        c_hi = tuple(a + int(b) for a, b in zip(c_lo, info.get("size_zyx", loc.local_shape)))
        if region is not None and not all(
                int(l) >= a and int(h) <= b
                for l, h, a, b in zip(region.start_zyx, region.stop_zyx, c_lo, c_hi)):
            print(f"[tsm] cache {path}: the completed footprint {c_lo}..{c_hi} does not cover "
                  f"start={tuple(region.start_zyx)} stop={tuple(region.stop_zyx)}; MISS", flush=True)
            continue
        r = VolumeReader(url, level, voxel_um, budget, array=loc)
        print(f"[tsm] cache HIT level {level}: {path} origin={loc.origin} size={loc.local_shape}", flush=True)
        return r
    if region is not None:
        print(f"[tsm] cache MISS level {level}: no cached box covers start={tuple(region.start_zyx)} "
              f"size={tuple(region.size_zyx)} (tried {len(paths)}); using the remote volume", flush=True)
    return None
