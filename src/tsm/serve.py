"""``tsm serve``: the live view protocol of ``spec/view.md`` over TCP.

The viewer (render3d ``--view-serve host:port``) asks a GPU machine for the student's
prediction of any box and gets back exactly what it would have read from a packet directory:
a ``tsm.view.v1`` ``meta.json`` followed by its layers' raw bytes, in manifest order.  The
layer vocabulary, the kind table and the store readers are :mod:`tsm.view`'s, shared with
``dev/view_export.py``, so live and on-disk packets can never disagree about a byte's meaning.

Framing (mirrors ``tools/surf/surfserver.py``), 4-byte magic + little-endian u32 length::

    request : 'TSV1' u32 hdr_len   then hdr_len bytes of JSON
    response: 'TSVR' u32 hdr_len   then hdr_len bytes of JSON, then the layer bytes
    error   : 'TSVE' u32 msg_len   then msg_len bytes of UTF-8 text

Request JSON is ``{"origin_zyx": [...], "dims_zyx": [...], "want": ["student", "ct"],
"tta": "none"}``; ``{"hello": true}`` answers with an empty layer list whose JSON carries
``groups`` / ``max_dims_zyx`` / ``scroll`` / ``checkpoint``.

Serialisation is deliberate: the student model is loaded **once** and one request is served at
a time (connections are handled in turn, and a lock guards the model), because a request holds
the box's layers only one array at a time and a second concurrent box would double both the
host RAM and the VRAM high-water mark.  ``run_serve`` refuses to start without a CUDA device --
the laptop GPU is never used; the intended hosts are forlindesk2 and the Thunder A6000, reached
through ``ssh -L 9760:localhost:9760 host``.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np

from tsm.config import RunCfg
from tsm.limits import Budget
from tsm.view import (CTSource, FORMAT, Source, StudentSource, check_dims, iter_arrays,
                      packet_meta, plan, scroll_name, teacher_sources)

__all__ = ["MAGIC_REQ", "MAGIC_RES", "MAGIC_ERR", "DEFAULT_PORT", "DEFAULT_MAX_DIMS",
           "ProtocolError", "RequestError", "recv_exact", "read_request", "send_response",
           "send_error", "ViewServer", "run_serve", "main"]

MAGIC_REQ = b"TSV1"
MAGIC_RES = b"TSVR"
MAGIC_ERR = b"TSVE"
DEFAULT_PORT = 9760
DEFAULT_MAX_DIMS = (512, 512, 512)
#: refuse absurd headers outright: a request header is a few hundred bytes of JSON
MAX_HDR = 1 << 20

Log = Callable[[str], None]


def _log(msg: str) -> None:
    print(f"[serve] {msg}", flush=True)


class ProtocolError(Exception):
    """The peer sent something that is not a framed request (the connection is dropped)."""


class RequestError(Exception):
    """A well-formed request that cannot be served (answered with ``TSVE``)."""


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #
def recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Exactly ``n`` bytes, or ``None`` when the peer closed cleanly at a frame boundary."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            if not buf:
                return None  # clean close at a frame boundary
            raise ProtocolError(f"stream ended after {len(buf)} of {n} bytes")
        buf += chunk
    return bytes(buf)


def read_request(sock: socket.socket) -> dict[str, Any] | None:
    """One ``TSV1`` frame -> its JSON, or ``None`` at end of stream."""
    head = recv_exact(sock, 8)
    if head is None:
        return None
    magic, n = head[:4], struct.unpack("<I", head[4:8])[0]
    if magic != MAGIC_REQ:
        raise ProtocolError(f"bad magic {magic!r} (expected {MAGIC_REQ!r})")
    if n > MAX_HDR:
        raise ProtocolError(f"header of {n} bytes exceeds {MAX_HDR}")
    body = recv_exact(sock, n) if n else b""
    if body is None:
        raise ProtocolError("truncated request header")
    try:
        req = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"request header is not JSON: {exc}")
    if not isinstance(req, dict):
        raise ProtocolError("request header must be a JSON object")
    return req


def send_response(sock: socket.socket, meta: dict[str, Any],
                  arrays: Sequence[np.ndarray] | Any = ()) -> int:
    """``TSVR`` + JSON + every layer's bytes in manifest order; returns the bytes sent."""
    body = json.dumps(meta).encode("utf-8")
    sock.sendall(MAGIC_RES + struct.pack("<I", len(body)) + body)
    sent = 8 + len(body)
    for a in arrays:
        b = np.ascontiguousarray(np.asarray(a, dtype=np.uint8)).tobytes()
        sock.sendall(b)
        sent += len(b)
    return sent


def send_error(sock: socket.socket, msg: str) -> None:
    body = str(msg).encode("utf-8")[:MAX_HDR]
    sock.sendall(MAGIC_ERR + struct.pack("<I", len(body)) + body)


# --------------------------------------------------------------------------- #
# the server
# --------------------------------------------------------------------------- #
class ViewServer:
    """Serves view packets for boxes of ``cfg``'s volume.

    ``student`` is any :class:`~tsm.view.Source` (normally a
    :class:`~tsm.view.StudentSource`); injecting one keeps the protocol testable without a
    GPU.  ``sources`` are the extra store-backed groups (teachers), each with its own
    coverage check."""

    def __init__(self, cfg: RunCfg, student: Source | None = None,
                 sources: Sequence[Source] | None = None, ct: Source | None = None,
                 max_dims: Sequence[int] = DEFAULT_MAX_DIMS, host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT, budget: Budget | None = None,
                 log: Log | None = None) -> None:
        from tsm.cli import open_ct

        self.cfg = cfg
        self.log = log or _log
        self.budget = budget or cfg.budget
        self.max_dims = tuple(int(v) for v in max_dims)
        check_dims(self.max_dims)
        self.host, self.port = str(host), int(port)
        self.ct = ct if ct is not None else CTSource(open_ct(cfg, 0))
        self.student = student
        self.sources: list[Source] = [self.ct] + ([student] if student is not None else []) + list(sources or [])
        self.scroll = scroll_name(cfg)
        self.voxel_um = float(cfg.volume.voxel_um)
        self._lock = threading.Lock()  # one box at a time: one model, one budget
        self._sock: socket.socket | None = None
        self.requests = 0

    # ------------------------------------------------------------------ #
    @property
    def groups(self) -> list[str]:
        out: list[str] = []
        for s in self.sources:
            if s.group not in out:
                out.append(s.group)
        return out

    @property
    def checkpoint(self) -> str:
        return str((self.student.provenance if self.student is not None else {}).get("checkpoint", ""))

    def hello(self) -> dict[str, Any]:
        """The handshake response JSON: a packet meta with no layers plus the server's terms."""
        return {
            "format": FORMAT,
            "hello": True,
            "dims_zyx": [0, 0, 0],
            "origin_zyx": [0, 0, 0],
            "voxel_um": self.voxel_um,
            "scroll": self.scroll,
            "layers": [],
            "groups": self.groups,
            "max_dims_zyx": [int(v) for v in self.max_dims],
            "checkpoint": self.checkpoint,
            "provenance": {k: v for s in self.sources for k, v in s.provenance.items()},
        }

    # ------------------------------------------------------------------ #
    def box(self, req: dict[str, Any]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Validate and clamp the requested box, in level-0 voxels.

        ``dims_zyx`` is clamped to the server's maximum (the viewer may ask for more than the
        host can hold); an origin outside the volume, or a box that runs past its far face,
        is a :class:`RequestError` -- padding a CT crop would show the viewer black voxels
        that look like real air."""
        try:
            origin = [int(v) for v in req.get("origin_zyx", [])]
            dims = [int(v) for v in req.get("dims_zyx", [])]
        except (TypeError, ValueError):
            raise RequestError(f"origin_zyx / dims_zyx must be 3 ints each: {req!r}")
        if len(origin) != 3 or len(dims) != 3:
            raise RequestError(f"origin_zyx / dims_zyx must be 3 ints each: {req!r}")
        if min(dims) <= 0:
            raise RequestError(f"dims_zyx must be positive, got {dims}")
        dims = [min(d, m) for d, m in zip(dims, self.max_dims)]
        shape = tuple(int(s) for s in self.ct.reader.shape) if isinstance(self.ct, CTSource) else None
        if shape is not None:
            for a in range(3):
                if origin[a] < 0 or origin[a] + dims[a] > shape[a]:
                    raise RequestError(
                        f"box origin={origin} dims={dims} is outside the volume {list(shape)}")
        return (origin[0], origin[1], origin[2]), (dims[0], dims[1], dims[2])

    def selected(self, want: Any) -> list[Source]:
        """The sources whose group the request asked for (default: all of them)."""
        if want is None:
            return list(self.sources)
        if not isinstance(want, (list, tuple)) or not all(isinstance(w, str) for w in want):
            raise RequestError("want must be a list of group names")
        names = set(want)
        unknown = sorted(names - set(self.groups))
        if unknown:
            raise RequestError(f"unknown group(s) {unknown}; this server serves {self.groups}")
        return [s for s in self.sources if s.group in names]

    def handle(self, sock: socket.socket, req: dict[str, Any]) -> None:
        """Serve one request onto ``sock`` (errors are the caller's to turn into ``TSVE``)."""
        if req.get("hello"):
            send_response(sock, self.hello())
            return
        origin, dims = self.box(req)
        tta = req.get("tta")
        if tta is not None and self.student is not None and str(tta) != str(self.student.provenance.get("tta", "none")):
            raise RequestError(f"this server runs tta={self.student.provenance.get('tta', 'none')!r}, "
                               f"restart it with --tta {tta} to change that")
        with self._lock:
            sources = plan(self.selected(req.get("want")), origin, dims, self.log)
            meta = packet_meta(sources, origin, dims, self.voxel_um, self.scroll)
            t0 = time.perf_counter()
            body = json.dumps(meta).encode("utf-8")
            sock.sendall(MAGIC_RES + struct.pack("<I", len(body)) + body)
            n = 0
            for _lay, a in iter_arrays(sources, origin, dims, self.budget):
                sock.sendall(np.ascontiguousarray(a).tobytes())
                n += 1
            self.requests += 1
        self.log(f"served origin={list(origin)} dims={list(dims)} {n} layers "
                 f"in {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------ #
    def bind(self) -> int:
        """Bind and listen; returns the actual port (``--port 0`` picks a free one)."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(4)
        self.port = int(s.getsockname()[1])
        self._sock = s
        return self.port

    def serve_forever(self, ready: threading.Event | None = None) -> None:
        """Accept connections one at a time and answer every framed request on each."""
        if self._sock is None:
            self.bind()
        assert self._sock is not None
        self.log(f"listening on {self.host}:{self.port}; groups={self.groups} "
                 f"max_dims={list(self.max_dims)} checkpoint={self.checkpoint or '-'}")
        if ready is not None:
            ready.set()
        try:
            while True:
                sock = self._sock  # a local reference: close() may clear the attribute
                if sock is None:
                    return
                try:
                    conn, addr = sock.accept()
                except OSError:
                    return  # the listening socket was closed (close())
                with conn:
                    self.serve_connection(conn, addr)
        finally:
            self.close()

    def serve_connection(self, conn: socket.socket, addr: Any = None) -> None:
        try:
            while True:
                req = read_request(conn)
                if req is None:
                    return
                try:
                    self.handle(conn, req)
                except RequestError as exc:
                    self.log(f"reject {addr}: {exc}")
                    send_error(conn, str(exc))
                except Exception as exc:  # a failed read must not take the server down
                    self.log(f"error {addr}: {type(exc).__name__}: {exc}")
                    send_error(conn, f"{type(exc).__name__}: {exc}")
        except ProtocolError as exc:
            self.log(f"protocol error from {addr}: {exc}")
            try:
                send_error(conn, str(exc))
            except OSError:
                pass
        except OSError as exc:
            self.log(f"connection {addr} dropped: {exc}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_serve(cfg: RunCfg, checkpoint: str, port: int = DEFAULT_PORT,
              max_dims: Sequence[int] = DEFAULT_MAX_DIMS, teachers: str | None = None,
              tta: str = "none", host: str = "127.0.0.1", group: str = "student",
              dry_run: bool = False) -> ViewServer:
    """Build the server (loading the student once) and run it until interrupted."""
    extra = teacher_sources(teachers, level0_um=float(cfg.volume.voxel_um), log=_log) if teachers else []
    student = None if dry_run else StudentSource(checkpoint, cfg, group=group, tta=tta, log=_log)
    srv = ViewServer(cfg, student=student, sources=extra, max_dims=max_dims, host=host, port=port)
    if dry_run:
        _log(f"dry run: would listen on {host}:{port}, groups={srv.groups}")
        return srv
    srv.bind()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        _log("interrupted")
    finally:
        srv.close()
    return srv


def parse_dims(s: str) -> tuple[int, int, int]:
    v = [int(p) for p in str(s).split(",") if p != ""]
    if len(v) != 3:
        raise ValueError(f"--max-dims wants z,y,x, got {s!r}")
    return (v[0], v[1], v[2])


def main(argv: list[str] | None = None, cfg: RunCfg | None = None) -> int:
    """``tsm serve`` / ``python -m tsm.serve``; ``cfg`` is passed by the CLI stage, which has
    already loaded the config (and then the positional argument is not accepted twice)."""
    import argparse

    from tsm.config import load_config

    ap = argparse.ArgumentParser(prog="tsm serve", description="live tsm.view.v1 prediction server")
    if cfg is None:
        ap.add_argument("config")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (the viewer tunnels with ssh -L)")
    ap.add_argument("--max-dims", default="512,512,512", metavar="Z,Y,X")
    ap.add_argument("--teachers", default=None, help="teacher store directory to serve as the 'teacher' group")
    ap.add_argument("--tta", default="none", choices=("none", "flip8", "flip8_rot4"))
    ap.add_argument("--group-name", default="student")
    ap.add_argument("--dry-run", action="store_true", help="build nothing, print what would be served")
    a = ap.parse_args(argv)
    run_cfg = cfg if cfg is not None else load_config(a.config)
    run_serve(run_cfg, os.path.expanduser(a.checkpoint), port=int(a.port),
              max_dims=parse_dims(a.max_dims), teachers=a.teachers, tta=a.tta, host=a.host,
              group=a.group_name, dry_run=bool(a.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
