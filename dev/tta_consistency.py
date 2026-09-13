"""How much does the student's answer change when the *same* crop is shown rotated?

usage: uv run python dev/tta_consistency.py configs/paris4_eval_body.json [--crops 32]
                                            [--tta flip8_rot4] [--checkpoint ...] [--device cpu]

Every member of the TTA group (``tsm.infer.tta_transforms``, 16 elements for ``flip8_rot4``) is a
symmetry of the *volume*, not of the scroll: the CT crop is flipped / transposed, the student runs
on it, and ``tta_inverse`` brings the heads back to the original frame.  A model whose prediction
is a property of the tissue returns the same field 16 times; a model that has learned "recto is the
face toward -z of this crop", or that reads the fibre classes off the volume axes, returns 16
different fields.  The spread of the de-rotated members is therefore a *label-free* measure of how
much of the prediction is orientation bookkeeping -- it needs no teacher, no upstream band and no
holdout label, only the CT.

Reported (JSON + a short table), over ``--crops`` crops of the config's **held-out** origins:

* ``sdf_std_p50/p90``      -- per-voxel standard deviation across the members of the primary SDF
  (``sdf_body`` in body mode, ``sdf_in`` in faces mode, ``sdf`` in medial mode), in voxels, on the
  band where the mean SDF is within ``--band`` (6) voxels of a surface;
* ``body_mask_agreement``  -- mean pairwise IoU of the predicted body ``sdf > 0`` across members;
* ``fiber_dir_angle_p50/p90`` -- axial angle (degrees) between each member's de-rotated fibre
  direction and the members' mean direction (``fiber_mode="direction"`` only);
* ``winding_normal_angle_p90`` -- the same for the winding normal (a signed vector);
* ``phase_std_deg_p90``    -- circular standard deviation of the winding phase across members.

**The laptop GPU is never used.**  Without ``TSM_ALLOW_LOCAL_GPU=1`` in the environment this runs
on the CPU (slow but correct: 32 crops x 16 passes of a 128-cube); ``--device cuda`` without that
variable is a hard error rather than a silent grab of the display GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))

from tsm.config import load_config  # noqa: E402
from tsm.infer import (  # noqa: E402
    INFER_DEFAULTS,
    StudentNet,
    _opts,
    activate_heads,
    load_student,
    tta_transforms,
)

#: the project rule (``tsm.view`` / ``tsm.serve``): the student runs on forlindesk2 or the A6000,
#: never on this machine's display GPU.  Set this to 1 to override on a machine that has a spare one.
ALLOW_LOCAL_GPU_ENV = "TSM_ALLOW_LOCAL_GPU"
DEFAULT_CROPS = 32
DEFAULT_BAND = 6.0          # |mean sdf| <= this many voxels: the sheet neighbourhood
SAMPLE_PER_CROP = 200_000   # bounded sample of the per-voxel spreads, per crop and per metric


def log(msg: str) -> None:
    print(f"[tta] {msg}", flush=True)


def pick_device(requested: str | None = None) -> torch.device:
    """CPU unless ``TSM_ALLOW_LOCAL_GPU`` is set; an explicit ``--device cuda`` without it fails."""
    allowed = bool(os.environ.get(ALLOW_LOCAL_GPU_ENV))
    req = str(requested or "").strip().lower()
    if req.startswith("cuda"):
        if not allowed:
            raise SystemExit(
                f"--device {requested} refused: the laptop GPU must never be used.  Run this on "
                f"forlindesk2 / the A6000, or set {ALLOW_LOCAL_GPU_ENV}=1 if this machine has a "
                f"spare GPU; --device cpu always works.")
        if not torch.cuda.is_available():
            raise SystemExit(f"--device {requested} but no CUDA device is available")
        return torch.device(req)
    if req:
        return torch.device(req)
    if allowed and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        log(f"a CUDA device is present but {ALLOW_LOCAL_GPU_ENV} is not set: running on the CPU")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# bounded samples -> percentiles
# --------------------------------------------------------------------------- #
class Sample:
    """Bounded, seeded sample of a scalar spread, pooled over crops."""

    def __init__(self, cap: int = SAMPLE_PER_CROP, seed: int = 0) -> None:
        self.cap, self.rng = int(cap), np.random.default_rng(seed)
        self.parts: list[np.ndarray] = []
        self.n = 0

    def add(self, v: np.ndarray) -> None:
        v = np.asarray(v, np.float32).ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return
        self.n += int(v.size)
        if v.size > self.cap:
            v = v[self.rng.choice(v.size, self.cap, replace=False)]
        self.parts.append(v)

    def stats(self, name: str) -> dict[str, Any]:
        v = np.concatenate(self.parts) if self.parts else np.zeros(0, np.float32)
        if v.size == 0:
            return {f"{name}_n": 0, f"{name}_p50": None, f"{name}_p90": None, f"{name}_mean": None}
        return {f"{name}_n": int(self.n), f"{name}_p50": float(np.percentile(v, 50)),
                f"{name}_p90": float(np.percentile(v, 90)), f"{name}_mean": float(v.mean())}


def _angle_to_mean(v: torch.Tensor, axial: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``(angle in degrees, defined)`` of each member's unit vector to the members' mean.

    ``v`` is (M, B, 3, Z, Y, X) -- the member axis first, the component axis at 2.  ``axial``
    (fibre directions): the sign is meaningless, so every member is first aligned with member 0
    and the angle is taken from ``|cos|``.  ``defined`` is False where *any* member produced a
    zero vector: ``activate_heads`` normalises with an epsilon, so a head that predicts nothing
    there says "no direction", which is not a disagreement between the members."""
    ok = (v.norm(dim=2) > 0.5).all(0)          # (B, Z, Y, X)
    w = v
    if axial:
        s = torch.sign((v * v[0:1]).sum(2, keepdim=True))
        w = v * torch.where(s == 0, torch.ones_like(s), s)
    m = w.mean(0, keepdim=True)
    m = m / m.norm(dim=2, keepdim=True).clamp_min(1e-9)
    c = (w * m).sum(2)
    if axial:
        c = c.abs()
    return torch.rad2deg(torch.arccos(c.clamp(-1.0, 1.0))), ok.unsqueeze(0).expand_as(c)


def _pairwise_iou(mask: torch.Tensor) -> float:
    """Mean IoU over the ``M*(M-1)/2`` member pairs of a (M, N) bool mask (0 pairs -> 1.0)."""
    a = mask.reshape(mask.shape[0], -1).float()
    inter = a @ a.T
    n = a.sum(1)
    union = n[:, None] + n[None, :] - inter
    iou = torch.where(union > 0, inter / union.clamp_min(1.0), torch.ones_like(union))
    m = mask.shape[0]
    iu = torch.triu_indices(m, m, offset=1)
    return float(iou[iu[0], iu[1]].mean()) if iu.shape[1] else 1.0


# --------------------------------------------------------------------------- #
def member_outputs(sn: StudentNet, x: torch.Tensor, origins, transforms,
                   surface_mode: str, fiber_mode: str) -> torch.Tensor:
    """(M, B, C, Z, Y, X) physical head values, one per TTA member, all in the *original* frame.

    Each member is run through the real inference path (``StudentNet.raw`` -> ``tta_forward``,
    net, ``tta_inverse``) with the TTA list temporarily set to that one element, so the geometry
    input channels are transported exactly as they are in production."""
    saved = sn.tta
    outs = []
    try:
        for t in transforms:
            sn.tta = [t]
            with torch.no_grad():
                outs.append(activate_heads(sn.raw(x, origins), surface_mode, fiber_mode).float())
    finally:
        sn.tta = saved
    return torch.stack(outs, dim=0)


def crop_metrics(phys: torch.Tensor, surface_mode: str, fiber_mode: str, band: float,
                 acc: dict[str, Any]) -> None:
    """Accumulate the per-voxel spreads of one crop's member stack into ``acc``."""
    ns = 2 if surface_mode == "faces" else 1
    o = ns - 1
    sdf = phys[:, :, 0]                                   # (M, B, Z, Y, X)
    mean_sdf = sdf.mean(0)
    sel = mean_sdf.abs() < float(band)
    if sel.any():
        acc["sdf"].add(sdf.std(0, unbiased=False)[sel].cpu().numpy())
    acc["iou"].append(_pairwise_iou(sdf > 0))

    sn_, cs_ = phys[:, :, 3 + o], phys[:, :, 4 + o]       # winding sin / cos, already renormalised
    ang = torch.atan2(sn_, cs_)
    r = torch.stack([torch.cos(ang).mean(0), torch.sin(ang).mean(0)]).norm(dim=0).clamp(1e-6, 1.0)
    acc["phase"].add(torch.rad2deg(torch.sqrt(-2.0 * torch.log(r))).cpu().numpy())

    nrm = phys[:, :, 6 + o:9 + o]                         # winding normal (nx, ny, nz)
    ang_n, ok_n = _angle_to_mean(nrm, axial=False)
    if ok_n.any():
        acc["normal"].add(ang_n[ok_n].cpu().numpy())

    if fiber_mode == "direction" and phys.shape[2] >= 11 + o + 4:
        d = phys[:, :, 11 + o:14 + o]                     # fibre direction (dz, dy, dx)
        strength = phys[:, :, 14 + o].mean(0)
        ang_f, ok_f = _angle_to_mean(d, axial=True)
        m = ok_f & (strength >= 0.5).unsqueeze(0).expand_as(ang_f)
        if not m.any():
            m = ok_f
        if m.any():
            acc["fiber"].add(ang_f[m].cpu().numpy())


def run(config: str, crops: int = DEFAULT_CROPS, tta: str = "flip8_rot4", checkpoint: str | None = None,
        device: str | None = None, band: float = DEFAULT_BAND, seed: int = 0,
        patch: int | None = None) -> dict[str, Any]:
    from tsm.train import build_dataset, model_input, train_opts

    cfg = load_config(config)
    iopts = _opts(cfg, "infer", INFER_DEFAULTS)
    ckpt = os.path.expanduser(str(checkpoint or iopts["checkpoint"]
                                  or os.path.join(cfg.out_dir, "train", "latest.pt")))
    clip = float(iopts["clip"])
    dev = pick_device(device)
    transforms = tta_transforms(tta)
    log(f"{len(transforms)} TTA members ({tta}) on {dev}")

    model, info = load_student(ckpt, dev)
    smode = str(info.get("surface_mode", "medial"))
    fmode = str(info.get("fiber_mode", "class"))
    sn = StudentNet(model, float(cfg.volume.voxel_um), clip, tta="none", surface_mode=smode,
                    input_radial=bool(info.get("input_radial", False)),
                    axis=info.get("axis_path"), input_axis=bool(info.get("input_axis", False)),
                    axis_tangent=bool(info.get("axis_tangent", False)), fiber_mode=fmode,
                    gap_class=bool(info.get("gap_class", False)),
                    lsd=bool(info.get("lsd", False))).to(dev)
    log(f"student {os.path.basename(ckpt)} step {info.get('step')}: surface_mode={smode} "
        f"fiber_mode={fmode} input_radial={info.get('input_radial')} input_axis={info.get('input_axis')}")

    topts = train_opts(cfg)
    if patch:
        topts = {**topts, "patch": int(patch)}
    ds = build_dataset(cfg, topts, augment=False)
    origins = np.asarray(getattr(ds, "holdout_origins", None), dtype=np.int64).reshape(-1, ds.origins.shape[1]) \
        if getattr(ds, "holdout_origins", None) is not None and len(ds.holdout_origins) else None
    held_out = origins is not None
    if not held_out:
        log("WARNING no holdout origins in this config; sampling the training origins instead")
        origins = np.asarray(ds.origins, dtype=np.int64).reshape(-1, ds.origins.shape[1])
    rng = np.random.default_rng(int(seed))
    n = min(int(crops), len(origins))
    pick = origins[rng.choice(len(origins), n, replace=False)] if len(origins) > n else origins

    acc: dict[str, Any] = {k: Sample(seed=i) for i, k in enumerate(("sdf", "phase", "normal", "fiber"))}
    acc["iou"] = []
    for i, org in enumerate(pick):
        item = ds.to_tensors(ds.load([int(v) for v in org]))
        batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
        x = model_input(batch)[:, 0:1].to(dev)
        go = [[int(v) for v in batch["origin_zyx"][0].tolist()]]
        phys = member_outputs(sn, x, go, transforms, smode, fmode)
        crop_metrics(phys, smode, fmode, band, acc)
        del phys, x, batch, item
        log(f"crop {i + 1}/{n} at {list(int(v) for v in org)} done")

    out: dict[str, Any] = {
        "config": os.path.abspath(config), "checkpoint": ckpt, "tta": tta,
        "n_members": len(transforms), "n_crops": int(n), "patch": int(ds.patch),
        "device": str(dev), "surface_mode": smode, "fiber_mode": fmode, "band": float(band),
        "holdout_origins": bool(held_out),
        "sdf_channel": {"body": "sdf_body", "faces": "sdf_in"}.get(smode, "sdf"),
        **acc["sdf"].stats("sdf_std"),
        "body_mask_agreement": float(np.mean(acc["iou"])) if acc["iou"] else None,
        **acc["fiber"].stats("fiber_dir_angle"),
        **acc["normal"].stats("winding_normal_angle"),
        **acc["phase"].stats("phase_std_deg"),
    }
    return out


ROWS = ("sdf_std_p50", "sdf_std_p90", "body_mask_agreement", "fiber_dir_angle_p50",
        "fiber_dir_angle_p90", "winding_normal_angle_p90", "phase_std_deg_p90")


def to_table(m: dict[str, Any]) -> str:
    w = max(len(k) for k in ROWS)
    head = (f"TTA consistency: {m['n_crops']} crops x {m['n_members']} members ({m['tta']}), "
            f"surface_mode={m['surface_mode']} fiber_mode={m['fiber_mode']}")
    lines = [head, "-" * len(head)]
    for k in ROWS:
        v = m.get(k)
        lines.append(f"{k:<{w}}  " + ("-" if v is None else f"{v:.4g}"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--crops", type=int, default=DEFAULT_CROPS)
    ap.add_argument("--tta", default="flip8_rot4")
    ap.add_argument("--checkpoint", default=None, help="default: extra.infer.checkpoint")
    ap.add_argument("--device", default=None, help="cpu (default) | cuda (needs TSM_ALLOW_LOCAL_GPU)")
    ap.add_argument("--band", type=float, default=DEFAULT_BAND)
    ap.add_argument("--patch", type=int, default=None, help="override extra.train.patch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the JSON here (default <out_dir>/eval/tta_consistency.json)")
    a = ap.parse_args(argv)
    m = run(a.config, crops=a.crops, tta=a.tta, checkpoint=a.checkpoint, device=a.device,
            band=a.band, seed=a.seed, patch=a.patch)
    cfg = load_config(a.config)
    path = a.out or os.path.join(os.path.expanduser(cfg.out_dir), "eval", "tta_consistency.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(m, fh, indent=2, default=float)
    print(to_table(m))
    log(f"wrote {path}")
    return m


if __name__ == "__main__":
    main()
