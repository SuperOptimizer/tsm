"""tsm.serve: the live view protocol (spec/view.md) over a real socket, byte for byte.

The server is started in a thread on a free port with a stub CT reader and a stub student
source (no GPU, no zarr, no model): what is under test is the framing, the handshake, the box
validation and the layer order -- everything the viewer's C client parses.  The GPU path is a
:class:`~tsm.view.StudentSource` the server merely holds, so it is covered in test_view.py.
"""

from __future__ import annotations

import json
import socket
import struct
import threading

import numpy as np
import pytest

from tsm.config import parse_config
from tsm.limits import Budget
from tsm.serve import MAGIC_ERR, MAGIC_REQ, MAGIC_RES, ViewServer
from tsm.view import CTSource, Layer, Source

VOL_SHAPE = (64, 64, 64)
DIMS = (8, 12, 16)
ORIGIN = (16, 8, 4)


class FakeReader:
    """The config's volume reader, replaced by a deterministic in-memory volume."""

    url = "file://fake"
    shape = VOL_SHAPE

    def __init__(self) -> None:
        self.vol = np.arange(int(np.prod(VOL_SHAPE)), dtype=np.int64).reshape(VOL_SHAPE).astype(np.uint8)

    def read(self, z0, z1, y0, y1, x0, x1):
        return self.vol[z0:z1, y0:y1, x0:x1]


class StubStudent(Source):
    """Two layers whose bytes are a pure function of the box, so the client can check them."""

    group = "student"
    provenance = {"checkpoint": "/fake/latest.pt", "step": 7, "tta": "none"}

    def layers(self):
        return [Layer("sdf_in", "student", "sdf", clip=20.0), Layer("valid", "student", "prob")]

    def read(self, origin, dims, budget):
        for i, lay in enumerate(self.layers()):
            yield lay, np.full(tuple(int(v) for v in dims), (int(origin[0]) + i) % 256, np.uint8)


def _cfg():
    return parse_config({
        "volume": {"url": "file://fake", "level": 0, "voxel_um": 2.4},
        "region": {"start_zyx": [0, 0, 0], "size_zyx": list(VOL_SHAPE)},
        "out_dir": "/tmp/tsm-serve-test",
        "extra": {"scroll": "PHercTest"},
    })


@pytest.fixture()
def server():
    srv = ViewServer(_cfg(), student=StubStudent(), ct=CTSource(FakeReader()),
                     max_dims=(32, 32, 32), port=0, budget=Budget())
    srv.bind()
    ready = threading.Event()
    t = threading.Thread(target=srv.serve_forever, args=(ready,), daemon=True)
    t.start()
    ready.wait(5)
    yield srv
    srv.close()
    t.join(2)


def _connect(srv):
    s = socket.create_connection(("127.0.0.1", srv.port), timeout=10)
    s.settimeout(10)
    return s


def _send(sock, req: dict) -> None:
    body = json.dumps(req).encode("utf-8")
    sock.sendall(MAGIC_REQ + struct.pack("<I", len(body)) + body)


def _recv(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        assert c, f"stream ended after {len(buf)} of {n} bytes"
        buf += c
    return buf


def _frame(sock) -> tuple[bytes, bytes]:
    head = _recv(sock, 8)
    return head[:4], _recv(sock, struct.unpack("<I", head[4:8])[0])


def test_hello_then_request_then_error(server):
    with _connect(server) as sock:
        # 1. handshake: an empty layer list plus the server's terms
        _send(sock, {"hello": True})
        magic, body = _frame(sock)
        assert magic == MAGIC_RES
        hello = json.loads(body)
        assert hello["format"] == "tsm.view.v1" and hello["layers"] == []
        assert hello["groups"] == ["ct", "student"]
        assert hello["max_dims_zyx"] == [32, 32, 32]
        assert hello["scroll"] == "PHercTest"
        assert hello["checkpoint"] == "/fake/latest.pt"

        # 2. one box: meta.json then exactly Z*Y*X bytes per layer, in manifest order
        _send(sock, {"origin_zyx": list(ORIGIN), "dims_zyx": list(DIMS),
                     "want": ["ct", "student"], "tta": "none"})
        magic, body = _frame(sock)
        assert magic == MAGIC_RES
        meta = json.loads(body)
        assert meta["format"] == "tsm.view.v1"
        assert meta["dims_zyx"] == list(DIMS) and meta["origin_zyx"] == list(ORIGIN)
        assert [(lay["group"], lay["name"]) for lay in meta["layers"]] == [
            ("ct", "ct"), ("student", "sdf_in"), ("student", "valid")]
        assert meta["layers"][1]["clip"] == 20.0
        n = int(np.prod(DIMS))
        got = [np.frombuffer(_recv(sock, n), np.uint8).reshape(DIMS) for _ in meta["layers"]]
        assert np.array_equal(got[0], FakeReader().vol[16:24, 8:20, 4:20])
        assert np.array_equal(got[1], np.full(DIMS, ORIGIN[0] % 256, np.uint8))
        assert np.array_equal(got[2], np.full(DIMS, (ORIGIN[0] + 1) % 256, np.uint8))

        # 3. a box outside the volume: TSVE, and the connection stays usable
        _send(sock, {"origin_zyx": [0, 0, 60], "dims_zyx": [8, 8, 8]})
        magic, body = _frame(sock)
        assert magic == MAGIC_ERR
        assert "outside the volume" in body.decode("utf-8")

        _send(sock, {"hello": True})
        assert _frame(sock)[0] == MAGIC_RES
    assert server.requests == 1  # only the box request counts as a served box


def test_dims_are_clamped_to_max(server):
    with _connect(server) as sock:
        _send(sock, {"origin_zyx": [0, 0, 0], "dims_zyx": [64, 64, 64], "want": ["ct"]})
        magic, body = _frame(sock)
        assert magic == MAGIC_RES
        meta = json.loads(body)
        assert meta["dims_zyx"] == [32, 32, 32]
        assert len(_recv(sock, 32 ** 3)) == 32 ** 3


def test_unknown_group_and_bad_box_are_errors(server):
    with _connect(server) as sock:
        _send(sock, {"origin_zyx": [0, 0, 0], "dims_zyx": [8, 8, 8], "want": ["teacher"]})
        magic, body = _frame(sock)
        assert magic == MAGIC_ERR and "unknown group" in body.decode("utf-8")
        _send(sock, {"origin_zyx": [0, 0], "dims_zyx": [8, 8, 8]})
        assert _frame(sock)[0] == MAGIC_ERR
        _send(sock, {"origin_zyx": [0, 0, 0], "dims_zyx": [0, 8, 8]})
        assert _frame(sock)[0] == MAGIC_ERR


def test_bad_magic_closes_with_an_error(server):
    with _connect(server) as sock:
        sock.sendall(b"XXXX" + struct.pack("<I", 0))
        magic, body = _frame(sock)
        assert magic == MAGIC_ERR and b"magic" in body


def test_tta_mismatch_is_refused(server):
    with _connect(server) as sock:
        _send(sock, {"origin_zyx": list(ORIGIN), "dims_zyx": list(DIMS), "tta": "flip8"})
        magic, body = _frame(sock)
        assert magic == MAGIC_ERR and "tta" in body.decode("utf-8")


def test_serve_refuses_without_cuda(monkeypatch):
    """`tsm serve` is a GPU service: no CUDA device is a hard error (never the laptop GPU)."""
    import torch

    from tsm.serve import run_serve

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("tsm.cli.open_ct", lambda cfg, *a, **k: FakeReader())
    with pytest.raises(RuntimeError, match="CUDA"):
        run_serve(_cfg(), "/fake/latest.pt", port=0)
