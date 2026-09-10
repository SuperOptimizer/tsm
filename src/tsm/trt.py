"""TensorRT backend for teacher inference (optional; ``uv pip install tensorrt-cu13 onnx onnxscript``).

The teacher is exported to ONNX once per (teacher, patch, batch, precision) at the
fixed window shape ``[B, 1, p, p, p]`` and compiled into a TensorRT engine cached
per GPU under ``~/.cache/tsm-models/trt/``.  Only the *raw logits* are exported
(``TeacherSpec.select`` applied inside the wrapper): activation, gaussian
blending and quantisation stay in :mod:`tsm.sliding`, so ``TRTTeacher`` is a
drop-in ``net`` for :func:`tsm.sliding.predict_box`.

TensorRT >= 10 (and only strongly-typed networks in TensorRT 11) take the
precision from the graph itself, so the net is exported in the target dtype
(``fp16`` by default; ``bf16`` has no TensorRT 11.2 tactic for 3D ConvTranspose) with
fp32 casts on the input and output tensors.  Engines are GPU specific and keyed
by (GPU name, SM, TensorRT version); a different card rebuilds transparently.

``patch`` may differ from the teacher's native window (the nets are fully
convolutional; instance norm makes the output mildly window-size dependent) -- the
engine cache key carries it.  The ONNX graph itself is shape-agnostic (Conv /
InstanceNorm / Resize-by-scale only), so it is exported once per (teacher, batch,
precision) at a small ``EXPORT_PATCH`` window (little VRAM) and the input/output
dims are rewritten to the requested patch right before the engine build.

Execution runs on a dedicated CUDA stream (``submit`` / ``wait``): the caller's
stream records an event once the input is ready, the engine waits on it, and
the output event is waited on by the caller's stream in ``wait``.  Two output
buffers alternate (with or without the extra stream), so a result stays valid
until the second-next ``submit`` (the sliding loop consumes it before that).  ``__call__`` is the synchronous
drop-in (submit + wait + clone).

The student (:class:`TRTStudent`, ``extra.infer.backend = "trt"``) uses the same
machinery: ``[B, in_ch, p, p, p]`` in (in_ch = 5 with the radial channels, 8 with the scroll-axis
channels too, which are
built torch-side and concatenated before the call), the 11- (or 12-, two-face)
channel *raw* head tensor out, split back into ``{surface, ink, winding}`` so
:class:`tsm.infer.StudentNet` is unchanged.  fp16 only (bf16 has no TensorRT 11 tactic
for the decoder's 3D ConvTranspose); engines are cached under the same key scheme with
the name ``student_i<in>o<out>_w<widths>_bs<body_stride>_f<fingerprint>`` (the
fingerprint binds the artifact to the exact weights *and* architecture, see
:func:`net_fingerprint`).  Its ONNX is exported once
per (net, batch, precision) with *symbolic* spatial dims (``_student_dynamic_shapes``):
GroupNorm decomposes into reshape / InstanceNorm / reshape, and a static export bakes
the export window into that second Reshape, so the graph would only build at one patch.

INT8 (``precision="int8"``): TensorRT 11 has no implicit calibrator any more, so the
net is quantised with NVIDIA Model Optimizer (``uv pip install nvidia-modelopt
huggingface_hub``): every ``Conv3d`` / ``ConvTranspose3d`` gets per-channel int8
weights and a per-tensor int8 input quantiser calibrated on real windows
(:func:`collect_calibration_windows`), except the final ``seg_layers`` /
``task_heads`` (kept fp16); the Q/DQ ONNX builds into an explicit-quantisation
engine.  The rest of the graph (norms, activations, upsampling) stays fp16.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn

from tsm.teachers import DEFAULT_MODELS_DIR, TeacherSpec

__all__ = [
    "TRTTeacher", "TRTStudent", "export_onnx", "export_student_onnx", "student_engine_name",
    "build_engine", "engine_cache_key", "auto_precision",
    "collect_calibration_windows", "quantize_int8",
    "weight_fingerprint", "arch_fingerprint", "net_fingerprint", "teacher_cache_name",
]

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
PRECISIONS = ("fp16", "bf16", "fp32", "int8")
_INT8_COMPUTE_DTYPE = torch.float16  # non-quantised ops of an int8 engine
EXPORT_PATCH = 128  # ONNX export window (shape-agnostic graph; re-dimmed at build time)


def _log(msg: str) -> None:
    print(f"[trt] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Model provenance (R01): ONNX / engine cache names must change when the *weights* or
# the *architecture* change, not merely when the shape contract does -- two checkpoints
# with the same widths/channels used to share one cached engine, so requesting
# checkpoint B silently ran checkpoint A.  ``weight_fingerprint`` hashes every
# state_dict tensor in sorted-key order; ``arch_fingerprint`` hashes the declared
# architecture config *and* ``repr(net)``, which spells out every submodule's norm
# kind, groups, kernel and width (so ``norm``/``blocks``/``fullres_width`` cannot
# collide even when the listed dimensions agree).
# --------------------------------------------------------------------------- #
_ARCH_KEYS = (
    "in_ch", "aux_ch", "widths", "blocks", "dec_blocks", "bottleneck_blocks", "groups",
    "ds_levels", "fullres_width", "body_stride", "norm_kind", "norm", "heads", "out_ch",
    "divisor", "n_down", "n_levels",
)


def _tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.numel() == 0:
        return b""
    try:  # works for bf16 too, which numpy cannot represent
        return t.reshape(-1).view(torch.uint8).numpy().tobytes()
    except Exception:
        return t.reshape(-1).float().numpy().tobytes()


def weight_fingerprint(net: nn.Module, n: int = 12) -> str:
    """sha256 over each state_dict tensor's bytes (sorted key order) -> ``n`` hex chars."""
    h = hashlib.sha256()
    sd = net.state_dict()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        if isinstance(v, torch.Tensor):
            h.update(f"|{tuple(v.shape)}|{v.dtype}|".encode())
            h.update(_tensor_bytes(v))
        else:
            h.update(f"|{v!r}|".encode())
    return h.hexdigest()[:n]


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(v[k]) for k in sorted(v, key=str)}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def arch_fingerprint(net: nn.Module, n: int = 12) -> str:
    """sha256 over the declared architecture config plus ``repr(net)`` -> ``n`` hex chars."""
    cfg = {k: _jsonable(getattr(net, k)) for k in _ARCH_KEYS if hasattr(net, k)}
    payload = json.dumps(cfg, sort_keys=True) + "\n" + repr(net)
    return hashlib.sha256(payload.encode()).hexdigest()[:n]


def net_fingerprint(net: nn.Module) -> str:
    """``<arch><weights>`` -- the identity an ONNX / engine cache entry is bound to."""
    return arch_fingerprint(net) + weight_fingerprint(net)


def teacher_cache_name(tspec: TeacherSpec, net: nn.Module | None = None, fp: str = "",
                       precision: str = "", calib_algorithm: str = "") -> str:
    """Cache stem for a teacher engine/ONNX: name x model fingerprint x int8 calibration."""
    f = fp or (net_fingerprint(net) if net is not None else "")
    parts = [tspec.name] + ([f"f{f}"] if f else [])
    if str(precision) == "int8" and calib_algorithm:
        parts.append(f"c{calib_algorithm}")
    return "_".join(parts)


def auto_precision(device: torch.device | int = 0) -> str:
    """fp16 everywhere: TensorRT 11.2 has no bf16 tactic for 3D ConvTranspose (the
    teachers' upsampling), so a bf16 build fails on Blackwell too; pass
    ``precision="bf16"`` explicitly to retry on a newer TensorRT."""
    return "fp16"


def _gpu_tag(device: torch.device | int = 0) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", torch.cuda.get_device_name(device)).strip("_")
    major, minor = torch.cuda.get_device_capability(device)
    return f"{name}_sm{major}{minor}"


def engine_cache_key(name: str, patch: int, batch: int, precision: str, device: torch.device | int = 0) -> str:
    import tensorrt as trt

    return f"{name}_p{patch}_b{batch}_{precision}_{_gpu_tag(device)}_trt{trt.__version__}"


class _LogitsWrapper(nn.Module):
    """fp32 in -> teacher in ``dtype`` -> selected logits as fp32."""

    def __init__(self, net: nn.Module, select: Callable[[Any], torch.Tensor], dtype: torch.dtype):
        super().__init__()
        self.net = net
        self.select = select
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.select(self.net(x.to(self.dtype)))
        return y.float()


def _onnx_export(wrapper: nn.Module, x: torch.Tensor, path: str, opset: int,
                 dynamic_shapes: Any | None = None, dynamic_axes: Any | None = None) -> None:
    """``dynamic_shapes`` (dynamo) / ``dynamic_axes`` (TorchScript fallback) keep the spatial
    dims symbolic so shape-dependent decompositions (GroupNorm's reshape) read them from
    ``Shape`` ops instead of baking the export window in; see :func:`export_student_onnx`."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with torch.inference_mode():
        try:
            prog = torch.onnx.export(
                wrapper, (x,), input_names=["x"], output_names=["y"], opset_version=opset,
                dynamo=True, optimize=True, dynamic_shapes=dynamic_shapes,
            )
            prog.save(path, external_data=True)
        except Exception as exc:  # fall back to the TorchScript exporter
            _log(f"dynamo export failed ({type(exc).__name__}: {str(exc)[:200]}); trying dynamo=False")
            torch.onnx.export(
                wrapper, (x,), path, input_names=["x"], output_names=["y"], opset_version=min(opset, 17),
                dynamo=False, do_constant_folding=True, dynamic_axes=dynamic_axes,
            )


def export_onnx(
    net: nn.Module, tspec: TeacherSpec, path: str, batch: int = 1, precision: str = "fp16",
    opset: int = 18, patch: int = EXPORT_PATCH,
) -> str:
    """Export ``net`` (any device) to ``path`` at window ``patch`` (default the small
    ``EXPORT_PATCH``; the graph is re-dimmed by :func:`build_engine`)."""
    dtype = _DTYPES[precision]
    p = int(patch)
    dev = next(net.parameters()).device
    # copy the weights into the export dtype on the same device (the caller keeps its fp32 net)
    net_c = copy.deepcopy(net).to(dtype=dtype).eval()
    wrapper = _LogitsWrapper(net_c, tspec.select, dtype).eval()
    x = torch.zeros((int(batch), 1, p, p, p), dtype=torch.float32, device=dev)
    t = time.perf_counter()
    _onnx_export(wrapper, x, path, opset)
    del net_c, wrapper, x
    _log(f"exported {os.path.basename(path)} ({precision}, batch {batch}, {p}^3) in {time.perf_counter() - t:.1f}s")
    return path


# --------------------------------------------------------------------------- #
# INT8: calibration windows + Model Optimizer PTQ
# --------------------------------------------------------------------------- #
def collect_calibration_windows(
    reader: Any, start_zyx: Sequence[int], patch: int, n: int = 32, cache_path: str | None = None,
    empty_max: int = 0, layout: tuple[int, int, int] = (2, 4, 4),
) -> np.ndarray:
    """``n`` real uint8 windows [n, p, p, p] read from ``reader`` (a ``tsm.volume.VolumeReader``)
    starting at ``start_zyx`` in the reader's level grid.  A box of ``layout`` x ``patch`` is read
    once and split into non-overlapping windows; windows with ``max <= empty_max`` (air) are
    dropped; the result is cached as ``.npy`` (memory-mapped on reload)."""
    if cache_path and os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape[1:] == (patch,) * 3 and arr.shape[0] >= min(n, 1):
            return arr[:n]
    p = int(patch)
    z0, y0, x0 = (int(v) for v in start_zyx)
    lz, ly, lx = layout
    t = time.perf_counter()
    box = reader.read(z0, z0 + lz * p, y0, y0 + ly * p, x0, x0 + lx * p)
    _log(f"calibration box {box.shape} read in {time.perf_counter() - t:.1f}s (mean {box.mean():.1f})")
    wins = []
    for iz in range(lz):
        for iy in range(ly):
            for ix in range(lx):
                w = box[iz * p : (iz + 1) * p, iy * p : (iy + 1) * p, ix * p : (ix + 1) * p]
                if int(w.max()) > empty_max:
                    wins.append(np.ascontiguousarray(w))
    del box
    if not wins:
        raise RuntimeError("no non-empty calibration windows")
    arr = np.stack(wins[:n])
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, arr)
    _log(f"{arr.shape[0]} calibration windows of {p}^3 ({arr.nbytes / (1 << 20):.0f} MiB)")
    return arr


def quantize_int8(
    net: nn.Module, tspec: TeacherSpec, calib: np.ndarray, patch: int | None = None, batch: int = 1,
    device: torch.device | str = "cuda", algorithm: str = "max", keep_heads_fp16: bool = True,
    quantize: str = "both",
) -> tuple[nn.Module, dict[str, Any]]:
    """Return an int8-quantised (Model Optimizer, PTQ) fp16 copy of ``net`` wrapped as
    :class:`_LogitsWrapper`, calibrated on ``calib`` [n, p, p, p] uint8 windows through
    ``tspec.normalizer``.  ``algorithm``: "max" (per-tensor absmax), "mse" (amax search), or the
    histogram methods "percentile" (99.99th), "entropy", "mse_hist" for the activations."""
    import modelopt.torch.quantization as mtq

    device = torch.device(device)
    p = int(patch or calib.shape[-1])
    if calib.shape[1:] != (p,) * 3:
        raise ValueError(f"calibration windows {calib.shape} do not match patch {p}")
    net_c = copy.deepcopy(net).to(device=device, dtype=_INT8_COMPUTE_DTYPE).eval()
    wrapper = _LogitsWrapper(net_c, tspec.select, _INT8_COMPUTE_DTYPE).eval()
    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    # transposed convs stay fp16: their per-channel weight axis differs from Conv's and they are cheap
    cfg["quant_cfg"] += [
        {"quantizer_name": "*transpconv*", "enable": False},
        {"quantizer_name": "*", "parent_class": "nn.ConvTranspose3d", "enable": False},
        # norm inputs: no speedup and TensorRT rejects Q/DQ feeding InstanceNormalization
        {"quantizer_name": "*norm*", "enable": False},
        {"quantizer_name": "*", "parent_class": "nn.InstanceNorm3d", "enable": False},
    ]
    if keep_heads_fp16:
        cfg["quant_cfg"] += [
            {"quantizer_name": "*seg_layers*", "enable": False},
            {"quantizer_name": "*task_heads*", "enable": False},
        ]
    if quantize == "weights":  # diagnostics: int8 weights, fp16 activations
        cfg["quant_cfg"].append({"quantizer_name": "*input_quantizer", "enable": False})
    elif quantize == "inputs":
        cfg["quant_cfg"].append({"quantizer_name": "*weight_quantizer", "enable": False})
    histogram = algorithm in ("percentile", "entropy", "mse_hist")
    if histogram:  # activation ranges from a histogram (outlier-robust) instead of the absolute max
        for rule in cfg["quant_cfg"]:
            if rule.get("quantizer_name") == "*input_quantizer":
                rule["cfg"] = dict(rule["cfg"], calibrator="histogram")
        cfg["algorithm"] = None
    else:
        cfg["algorithm"] = algorithm if algorithm == "max" else {"method": algorithm}

    def forward_loop(model: nn.Module) -> None:
        with torch.inference_mode():
            for i in range(0, calib.shape[0], batch):
                chunk = calib[i : i + batch]
                x = torch.stack([tspec.normalizer(torch.from_numpy(np.asarray(w)).to(device))[None] for w in chunk])
                if x.shape[0] < batch:
                    x = torch.cat([x, x[-1:].expand(batch - x.shape[0], -1, -1, -1, -1)])
                model(x)

    t = time.perf_counter()
    if histogram:
        from modelopt.torch.quantization.calib import HistogramCalibrator
        from modelopt.torch.quantization.model_calib import enable_stats_collection

        mtq.quantize(wrapper, cfg, forward_loop=None)
        enable_stats_collection(wrapper)
        forward_loop(wrapper)
        method = {"percentile": "percentile", "entropy": "entropy", "mse_hist": "mse"}[algorithm]
        for m in wrapper.modules():
            if type(m).__name__ != "TensorQuantizer" or getattr(m, "_disabled", False):
                continue
            cal = getattr(m, "_calibrator", None)
            if isinstance(cal, HistogramCalibrator):
                m.load_calib_amax(method, **({"percentile": 99.99} if method == "percentile" else {}))
            elif cal is not None:
                m.load_calib_amax()
            m.enable_quant()
    else:
        mtq.quantize(wrapper, cfg, forward_loop)
    n_q = sum(1 for m in wrapper.modules() if type(m).__name__ == "TensorQuantizer" and m.is_enabled)
    info = {"algorithm": algorithm, "n_calib": int(calib.shape[0]), "patch": p, "quantizers": n_q,
            "calib_seconds": round(time.perf_counter() - t, 1), "keep_heads_fp16": keep_heads_fp16}
    _log(f"int8 PTQ ({algorithm}) on {calib.shape[0]} windows in {info['calib_seconds']}s, {n_q} quantizers")
    return wrapper, info


class _upcast_cpu_half_ops:
    """CPU tracing in fp16: ops without a Half CPU kernel (avg_pool3d, the student's
    trilinear interpolate) run in fp32 and cast back, which lands as Cast -> op -> Cast in
    the graph (TensorRT folds them)."""

    _NAMES = ("avg_pool3d", "interpolate")

    def __enter__(self):
        import torch.nn.functional as F

        self._orig = {n: getattr(F, n) for n in self._NAMES}
        for n, fn in self._orig.items():
            def wrapped(x, *a, _fn=fn, **k):
                if x.dtype in (torch.float16, torch.bfloat16) and x.device.type == "cpu":
                    return _fn(x.float(), *a, **k).to(x.dtype)
                return _fn(x, *a, **k)
            setattr(F, n, wrapped)
        return self

    def __exit__(self, *exc):
        import torch.nn.functional as F

        for n, fn in self._orig.items():
            setattr(F, n, fn)
        return False


def _fix_qdq_dtypes(path: str, dtype: torch.dtype = _INT8_COMPUTE_DTYPE) -> int:
    """Model Optimizer's quantized convs run their inputs through fp32 (Q/DQ scales, the casts
    feeding QuantizeLinear, and the conv biases/weights folded by the tracer come out fp32),
    which makes TensorRT type the graph fp32 while the remaining fp16 initializers (norm
    weights) stay half -> strongly-typed parse errors.  Rewrite every fp32 constant /
    initializer to ``dtype`` (except Resize scales, which must be float32), clamp Q/DQ scales
    positive (TensorRT rejects zero scales) and retarget fp32 casts that do not produce a
    graph output.  Returns the number of edits."""
    import onnx
    from onnx import numpy_helper

    onnx_dtype = {torch.float16: onnx.TensorProto.FLOAT16, torch.bfloat16: onnx.TensorProto.BFLOAT16,
                  torch.float32: onnx.TensorProto.FLOAT}[dtype]
    np_dtype = {torch.float16: np.float16, torch.float32: np.float32}.get(dtype, np.float16)
    tiny = float(np.finfo(np_dtype).tiny)
    model = onnx.load(path)
    g = model.graph
    outputs = {o.name for o in g.output}
    keep_f32 = set()
    for n in g.node:
        if n.op_type == "Resize":
            keep_f32.update(n.input[1:])
    scale_names = {n.input[1] for n in g.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")}
    changed = 0

    def _convert(t: "onnx.TensorProto", name: str) -> None:
        nonlocal changed
        if t.data_type != onnx.TensorProto.FLOAT or name in keep_f32:
            return
        arr = numpy_helper.to_array(t).astype(np.float32)
        if name in scale_names:
            arr = np.maximum(arr, tiny)
        t.CopyFrom(numpy_helper.from_array(arr.astype(np_dtype), t.name))
        changed += 1

    for t in g.initializer:
        _convert(t, t.name)
    for n in g.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    _convert(attr.t, n.output[0])
        elif n.op_type == "Cast" and n.output[0] not in outputs:
            for attr in n.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.FLOAT:
                    attr.i = onnx_dtype
                    changed += 1
    if changed:
        onnx.save(model, path)
    return changed


def export_onnx_int8(
    wrapper: nn.Module, path: str, patch: int = EXPORT_PATCH, batch: int = 1, opset: int = 17,
    device: torch.device | str = "cpu",
) -> str:
    """Q/DQ ONNX of a Model-Optimizer-quantised wrapper.  Only the TorchScript exporter
    (``dynamo=False``, *without* ``export_torch_mode``) emits the quantizers as
    QuantizeLinear/DequantizeLinear here: the dynamo path trips on a data-dependent
    guard inside the fake-quant kernel and ``export_torch_mode`` turns the amax buffers
    into graph params the symbolic cannot fold.  Tracing on CUDA segfaults (modelopt's
    CPU fallback extension), so the wrapper is traced on the CPU in fp16 at a small
    window (the graph is shape-agnostic)."""
    device = torch.device(device)
    wrapper = wrapper.to(device)
    x = torch.zeros((int(batch), 1, patch, patch, patch), dtype=torch.float32, device=device)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t = time.perf_counter()
    with torch.inference_mode(), _upcast_cpu_half_ops():
        torch.onnx.export(
            wrapper, (x,), path, input_names=["x"], output_names=["y"], opset_version=min(opset, 17),
            dynamo=False, do_constant_folding=True,
        )
    n_fix = _fix_qdq_dtypes(path)
    _log(f"exported {os.path.basename(path)} (int8 Q/DQ, batch {batch}, {patch}^3, {n_fix} scale/cast dtype fixes) "
         f"in {time.perf_counter() - t:.1f}s")
    return path


# --------------------------------------------------------------------------- #
# Engine build / runtime
# --------------------------------------------------------------------------- #
def _redim_onnx(onnx_path: str, patch: int | None, batch: int | None) -> bytes:
    """Serialised model with the (shape-agnostic) graph's input / output dims set to
    ``[batch, C, patch, patch, patch]``; intermediate shape annotations are dropped."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=True)
    changed = False
    for vi in list(model.graph.input) + list(model.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if len(dims) != 5:
            continue
        want = [batch or dims[0].dim_value, dims[1].dim_value] + [patch or d.dim_value for d in dims[2:]]
        for d, v in zip(dims, want):
            if not int(v):  # symbolic dim and nothing to pin it to: leave it dynamic
                continue
            if d.dim_value != int(v) or d.HasField("dim_param"):
                d.dim_value = int(v)  # setting dim_value clears dim_param (protobuf oneof)
                changed = True
    if changed:
        del model.graph.value_info[:]
    return model.SerializeToString()


def build_engine(onnx_path: str, engine_path: str, workspace_gb: float = 6.0, patch: int | None = None,
                 batch: int | None = None) -> tuple[str, float]:
    """Parse ``onnx_path`` (re-dimmed to ``patch`` / ``batch`` when given) and serialise a
    strongly-typed engine to ``engine_path``.

    Returns (engine_path, build_seconds)."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    t = time.perf_counter()
    if patch is None and batch is None:
        ok = parser.parse_from_file(onnx_path)
    else:
        ok = parser.parse(_redim_onnx(onnx_path, patch, batch))
    if not ok:
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}: {errs}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if hasattr(trt.BuilderFlag, "TF32"):
        config.set_flag(trt.BuilderFlag.TF32)
    cache_path = engine_path + ".timing"
    tcache = None
    try:
        with open(cache_path, "rb") as fh:
            tcache = config.create_timing_cache(fh.read())
    except OSError:
        tcache = config.create_timing_cache(b"")
    if tcache is not None:
        config.set_timing_cache(tcache, ignore_mismatch=False)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    tmp = engine_path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(bytes(serialized))
    os.replace(tmp, engine_path)
    if tcache is not None:
        try:
            with open(cache_path, "wb") as fh:
                fh.write(bytes(tcache.serialize()))
        except Exception:
            pass
    dt = time.perf_counter() - t
    _log(f"built {os.path.basename(engine_path)} ({serialized.nbytes / (1 << 20):.0f} MiB) in {dt:.1f}s")
    return engine_path, dt


class _TRTEngine:
    """Loaded engine + I/O buffers: ``submit(x) -> handle`` enqueues on the wrapper's own
    CUDA stream, ``wait(handle)`` makes the caller's current stream wait and returns the
    raw output buffer (overwritten by the second-next ``submit``).  Subclasses turn that
    tensor into whatever the torch net returned (``_wrap``)."""

    def __init__(self, engine_path: str, batch: int, precision: str,
                 device: torch.device | str = "cuda", use_stream: bool = True) -> None:
        import tensorrt as trt

        self.engine_path = engine_path
        self.batch = int(batch)
        self.precision = precision
        self.device = torch.device(device)
        self.build_seconds = 0.0
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as fh:
            self.engine = self.runtime.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialise {engine_path}")
        self.context = self.engine.create_execution_context()
        self.in_name, self.out_name = "x", "y"
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        if self.in_name not in names or self.out_name not in names:
            raise RuntimeError(f"unexpected engine tensors {names}")
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        self.patch = int(self.in_shape[-1])
        self.stream = torch.cuda.Stream(device=self.device) if use_stream else None
        # two output buffers: the pipelined sliding loop submits pass j+1 before it consumes pass j
        self._outs = [torch.empty(self.out_shape, dtype=torch.float32, device=self.device) for _ in range(2)]
        self._slot = 0
        self.device_memory_mb = self.engine.device_memory_size_v2 / (1 << 20) if hasattr(
            self.engine, "device_memory_size_v2") else float("nan")

    def _wrap(self, y: torch.Tensor) -> Any:
        return y

    def submit(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Any]:
        if tuple(x.shape) != self.in_shape:
            raise ValueError(f"TRT engine expects input {self.in_shape}, got {tuple(x.shape)}")
        x = x.float().contiguous()
        out = self._outs[self._slot]
        self._slot = (self._slot + 1) % len(self._outs)
        self.context.set_tensor_address(self.in_name, x.data_ptr())
        self.context.set_tensor_address(self.out_name, out.data_ptr())
        cur = torch.cuda.current_stream(self.device)
        if self.stream is None:
            if not self.context.execute_async_v3(cur.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            return x, out, None
        # input ready (and the buffer's previous consumer, which ran on `cur`, done) -> engine stream
        ev_in = torch.cuda.Event()
        ev_in.record(cur)
        self.stream.wait_event(ev_in)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        x.record_stream(self.stream)  # keep the input's memory until the engine has read it
        ev_out = torch.cuda.Event()
        ev_out.record(self.stream)
        return x, out, ev_out

    def wait(self, handle: tuple[torch.Tensor, torch.Tensor, Any]) -> Any:
        _x, out, ev_out = handle
        if ev_out is not None:
            torch.cuda.current_stream(self.device).wait_event(ev_out)
        return self._wrap(out)

    def __call__(self, x: torch.Tensor, **kw: Any) -> Any:
        _x, out, ev_out = self.submit(x, **kw)
        if ev_out is not None:
            torch.cuda.current_stream(self.device).wait_event(ev_out)
        return self._wrap(out.clone())  # the buffer is reused on a later call


class TRTTeacher(_TRTEngine):
    """Callable ``net`` replacement: ``x`` [B,1,p,p,p] cuda tensor -> logits (dict or tensor
    shaped like the torch teacher's output so ``TeacherSpec.select`` still applies)."""

    def __init__(self, engine_path: str, tspec: TeacherSpec, batch: int, precision: str,
                 device: torch.device | str = "cuda", use_stream: bool = True) -> None:
        self.tspec = tspec
        super().__init__(engine_path, batch, precision, device, use_stream=use_stream)

    def _wrap(self, y: torch.Tensor) -> Any:
        if self.tspec.target is not None:
            return {self.tspec.target: y}
        return y

    @staticmethod
    def onnx_path(tspec: TeacherSpec, batch: int, precision: str, cache_dir: str, patch: int | None = None,
                  net: nn.Module | None = None, fp: str = "", calib_algorithm: str = "") -> str:
        """Shape-agnostic ONNX cache file, bound to the teacher's weight+architecture
        fingerprint (from ``net``, or a precomputed ``fp``); a legacy per-patch export of
        the *same* fingerprint is reused when present."""
        name = teacher_cache_name(tspec, net=net, fp=fp, precision=precision,
                                  calib_algorithm=calib_algorithm)
        if patch is not None:
            legacy = os.path.join(cache_dir, f"{name}_p{patch}_b{batch}_{precision}.onnx")
            if os.path.exists(legacy):
                return legacy
        return os.path.join(cache_dir, f"{name}_b{batch}_{precision}.onnx")

    @classmethod
    def build_or_load(
        cls, net: nn.Module, tspec: TeacherSpec, batch: int = 1, precision: str = "auto",
        cache_dir: str | os.PathLike = os.path.join(DEFAULT_MODELS_DIR, "trt"),
        device: torch.device | str = "cuda", workspace_gb: float = 6.0, patch: int | None = None,
        use_stream: bool = True, calib: np.ndarray | None = None, calib_algorithm: str = "max",
    ) -> "TRTTeacher":
        """Return a ready engine for ``tspec`` at ``batch`` / ``patch``, building (ONNX -> plan)
        on a cache miss.  ``precision="int8"`` needs ``calib`` (uint8 windows [n, p, p, p]) unless
        the ONNX / engine is already cached."""
        device = torch.device(device)
        if precision == "auto":
            precision = auto_precision(device)
        if precision not in PRECISIONS:
            raise ValueError(f"unknown TRT precision {precision!r}; known: {PRECISIONS}")
        p = int(patch or tspec.patch[0])
        cache_dir = os.fspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        fp = net_fingerprint(net)
        name = teacher_cache_name(tspec, fp=fp, precision=precision, calib_algorithm=calib_algorithm)
        key = engine_cache_key(name, p, batch, precision, device)
        engine_path = os.path.join(cache_dir, key + ".plan")
        build_s = 0.0
        if not os.path.exists(engine_path):
            onnx_path = cls.onnx_path(tspec, batch, precision, cache_dir, p, fp=fp,
                                      calib_algorithm=calib_algorithm)
            if not os.path.exists(onnx_path):
                if precision == "int8":
                    if calib is None:
                        raise ValueError("int8 build needs calibration windows (calib=...)")
                    wrapper, info = quantize_int8(net, tspec, calib, patch=int(calib.shape[-1]), batch=batch,
                                                  device=device, algorithm=calib_algorithm)
                    export_onnx_int8(wrapper, onnx_path, batch=batch)
                    with open(onnx_path + ".calib.json", "w") as fh:
                        json.dump({**info, "algorithm": calib_algorithm,
                                   "calib_shape": list(calib.shape),
                                   "calib_sha256": hashlib.sha256(
                                       np.ascontiguousarray(calib).tobytes()).hexdigest()[:16],
                                   "model_fingerprint": fp}, fh, indent=1)
                    del wrapper
                    torch.cuda.empty_cache()
                else:
                    export_onnx(net, tspec, onnx_path, batch=batch, precision=precision)
            _, build_s = build_engine(onnx_path, engine_path, workspace_gb=workspace_gb, patch=p, batch=batch)
        obj = cls(engine_path, tspec, batch, precision, device, use_stream=use_stream)
        obj.build_seconds = build_s
        return obj


def onnx_fingerprint(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Student (TSMNet) engine
# --------------------------------------------------------------------------- #
STUDENT_HEADS = ("surface", "ink", "winding")  # concatenation order of the raw head output


class _StudentHeadWrapper(nn.Module):
    """fp32 [B, in_ch, p, p, p] -> student in ``dtype`` -> raw heads concatenated as fp32.

    The student's eval forward returns ``{head: tensor}``; the engine has one output, so the
    heads are concatenated in :data:`STUDENT_HEADS` order (surface 2|3 + ink 1 + winding 8 =
    11 or 12 channels, exactly the tensor :func:`tsm.infer.activate_heads` consumes).  The
    radial input channels are computed torch-side and are part of ``x``."""

    def __init__(self, net: nn.Module, dtype: torch.dtype, heads: Sequence[str] = STUDENT_HEADS):
        super().__init__()
        self.net = net
        self.dtype = dtype
        self.head_names = tuple(heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.to(self.dtype))
        parts = []
        for name in self.head_names:
            v = out[name]
            parts.append(v[0] if isinstance(v, (list, tuple)) else v)
        return torch.cat(parts, dim=1).float()


def student_head_channels(net: nn.Module) -> "dict[str, int]":
    """``{head: channels}`` of a TSMNet, in the engine's concatenation order."""
    heads = dict(getattr(net, "heads", {"surface": 2, "ink": 1, "winding": 8}))
    return {k: int(heads[k]) for k in STUDENT_HEADS if k in heads}


def student_engine_name(in_ch: int, out_ch: int, widths: Sequence[int], body_stride: int = 1,
                        fp: str = "", net: nn.Module | None = None) -> str:
    """Cache name for a student engine: the shape contract the engine encodes **plus** the
    model fingerprint (:func:`net_fingerprint`) of the exact checkpoint/architecture, so
    two nets with identical widths but different weights never share an ONNX or plan."""
    w = "x".join(str(int(v)) for v in widths)
    base = f"student_i{int(in_ch)}o{int(out_ch)}_w{w}_bs{int(body_stride)}"
    f = fp or (net_fingerprint(net) if net is not None else "")
    return f"{base}_f{f}" if f else base


def _student_dynamic_shapes(net: nn.Module, patch: int) -> tuple[Any, Any]:
    """(``dynamic_shapes``, ``dynamic_axes``) making the three spatial dims symbolic
    multiples of the net's ``divisor``.

    Without them the GroupNorm decomposition (reshape to groups -> InstanceNorm ->
    reshape back) bakes the *export* window into the second Reshape's constant shape,
    and :func:`_redim_onnx` -- which only rewrites the graph's input/output dims --
    produces a graph TensorRT rejects ("Reshaping [1,8,8388608] to [1,32,64,64,64]").
    With them the reshape target is read from ``Shape`` ops, so one export serves every
    patch (TensorRT constant-folds them once the input is re-dimmed to a static size)."""
    d = max(1, int(getattr(net, "divisor", 1)))
    n = max(2, int(patch) // d)
    dim = torch.export.Dim("p", min=2, max=max(n, 8192 // d))
    ax = d * dim if d > 1 else dim
    return ({"x": {2: ax, 3: ax, 4: ax}},
            {"x": {2: "d", 3: "h", 4: "w"}, "y": {2: "d", 3: "h", 4: "w"}})


def export_student_onnx(net: nn.Module, path: str, in_ch: int | None = None, batch: int = 1,
                        precision: str = "fp16", opset: int = 18, patch: int = 64) -> str:
    """Export a TSMNet to ONNX with symbolic spatial dims (one graph for every window;
    re-dimmed to the build patch by :func:`build_engine`).  ``precision`` only chooses the weight dtype of the exported
    copy -- fp16, since TensorRT 11 has no bf16 tactic for the decoder's 3D ConvTranspose.

    Runs on whatever device ``net`` lives on, so the CPU export path can be exercised
    without a GPU (the engine build in :func:`build_engine` is CUDA-only)."""
    dtype = _DTYPES[precision]
    p = int(patch)
    dev = next(net.parameters()).device
    cin = int(in_ch if in_ch is not None else getattr(net, "in_ch"))
    net_c = copy.deepcopy(net).to(dtype=dtype).eval()
    wrapper = _StudentHeadWrapper(net_c, dtype).eval()
    x = torch.zeros((int(batch), cin, p, p, p), dtype=torch.float32, device=dev)
    t = time.perf_counter()
    with _upcast_cpu_half_ops():  # trilinear upsample of the full-res block has no CPU half kernel
        _onnx_export(wrapper, x, path, opset, *_student_dynamic_shapes(net, p))
    del net_c, wrapper, x
    _log(f"exported {os.path.basename(path)} (student, {precision}, batch {batch}, {cin} ch, {p}^3) "
         f"in {time.perf_counter() - t:.1f}s")
    return path


class TRTStudent(_TRTEngine):
    """TensorRT student: ``x`` [B, in_ch, p, p, p] fp32 cuda tensor -> ``{head: tensor}``
    exactly as ``TSMNet.eval()`` returns it, so :class:`tsm.infer.StudentNet` (activation,
    TTA, to_unit) and :func:`tsm.sliding.predict_box` are unchanged.

    ``in_ch`` includes the radial and scroll-axis channels (5 with ``input_radial``, 8 with
    ``input_axis`` as well; both are part of the engine cache key): they are built
    torch-side by ``StudentNet.radial`` and concatenated before the engine call."""

    def __init__(self, engine_path: str, heads: "dict[str, int]", batch: int, precision: str,
                 device: torch.device | str = "cuda", use_stream: bool = True) -> None:
        self.heads = dict(heads)
        super().__init__(engine_path, batch, precision, device, use_stream=use_stream)
        self.in_ch = int(self.in_shape[1])
        if int(self.out_shape[1]) != sum(self.heads.values()):
            raise RuntimeError(f"engine output {self.out_shape} does not match heads {self.heads}")

    def _wrap(self, y: torch.Tensor) -> Any:
        out, i = {}, 0
        for name, c in self.heads.items():
            out[name] = y[:, i : i + c]
            i += c
        return out

    @staticmethod
    def onnx_path(name: str, batch: int, precision: str, cache_dir: str) -> str:
        # "_dyn": symbolic spatial dims (see _student_dynamic_shapes); older static
        # exports under the previous name only build at their export patch.
        return os.path.join(cache_dir, f"{name}_b{batch}_{precision}_dyn.onnx")

    @classmethod
    def build_or_load(
        cls, net: nn.Module, patch: int, batch: int = 1, precision: str = "fp16",
        cache_dir: str | os.PathLike = os.path.join(DEFAULT_MODELS_DIR, "trt"),
        device: torch.device | str = "cuda", workspace_gb: float = 6.0, use_stream: bool = True,
        export_patch: int = 64,
    ) -> "TRTStudent":
        """Ready engine for ``net`` at ``patch`` / ``batch``, building (ONNX -> plan) on a
        cache miss.  The engine is keyed like the teachers': name (channels, widths,
        body_stride) x patch x batch x precision x (GPU, SM, TensorRT version)."""
        device = torch.device(device)
        if precision == "auto":
            precision = auto_precision(device)
        if precision not in ("fp16", "fp32"):
            raise ValueError(f"student TRT precision must be fp16 or fp32, got {precision!r}")
        heads = student_head_channels(net)
        in_ch, out_ch = int(net.in_ch), sum(heads.values())
        name = student_engine_name(in_ch, out_ch, getattr(net, "widths", ()),
                                   getattr(net, "body_stride", 1), fp=net_fingerprint(net))
        cache_dir = os.fspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        engine_path = os.path.join(cache_dir, engine_cache_key(name, int(patch), batch, precision, device) + ".plan")
        build_s = 0.0
        if not os.path.exists(engine_path):
            onnx_path = cls.onnx_path(name, batch, precision, cache_dir)
            if not os.path.exists(onnx_path):
                export_student_onnx(net, onnx_path, in_ch=in_ch, batch=batch, precision=precision,
                                    patch=int(export_patch))
            _, build_s = build_engine(onnx_path, engine_path, workspace_gb=workspace_gb, patch=int(patch), batch=batch)
        obj = cls(engine_path, heads, batch, precision, device, use_stream=use_stream)
        obj.build_seconds = build_s
        return obj
