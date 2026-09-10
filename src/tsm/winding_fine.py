"""Native 2.4 um winding phase from the two face SDFs (``wf_*`` fine-store channels).

The winding targets the student trains on today are the level-2 (9.6 um) lasagna field
upsampled on the fly (:func:`tsm.data.upsample_coarse_targets`): one phase value per
9.6 um voxel, trilinearly smeared over 4x4x4 fine voxels.  Wherever the fine store
carries the two-face channels (``sdf_in`` / ``sdf_out``, see docs/label_store.md) the
same field can be reconstructed *at the fine pitch* from the geometry, with the coarse
field used only as a prior for the integer turn.

Geometry
--------
Both SDFs are positive on the **outward** side of their own face, so along the outward
normal one period of the sheet stack looks like (``d`` = distance from this sheet's in
face, ``t`` = sheet thickness, ``g`` = air gap, ``P = t + g`` the sheet spacing)::

    in face          out face            next in face
    d=0              d=t                 d=P
    |================|-------------------|
     sheet body        air gap

Each SDF is the signed distance to the *nearest* voxel of its own face set, so both are
sawtooths of period ``P`` with values in ``[-P/2, P/2]``:

* ``sdf_in(d)  = d``      for ``d < P/2``,  ``d - P``      beyond,
* ``sdf_out(d) = d - t``  for ``|d - t| < P/2``, shifted by ``P`` beyond.

The winding number increases by exactly 1 per sheet and is linear in ``d`` (that is what
makes ``|grad w| = 1 / P`` -- one over the local sheet spacing -- as the spec asks), so
its fractional part is simply

    f = (sdf_in / P) mod 1

(with the integer turn at the **in** face, i.e. the face toward the umbilicus / the recto
side).  The ``mod`` is exactly what selects the right branch of the sawtooth: inside the
sheet ``sdf_in = d >= 0`` gives ``f = d / P``, and past the half-period the branch
``sdf_in = d - P < 0`` gives ``f = 1 + (d - P)/P = d / P`` again.  No sheet/gap
classification is needed, and ``f`` is continuous and monotone along the normal.

``P`` is recovered from the two SDFs alone.  With ``D = sdf_in - sdf_out``, elementary
algebra on the two sawtooths gives ``D = +t`` on part of the period and ``D = -g`` on the
rest (both parts non-empty for any ``0 < t < P``), so ``t`` and ``g`` are each measured
somewhere in every period and filled to the whole box by a nearest-voxel (EDT feature)
transform; ``P = t + g``.

The integer turn, and why the coarse prior does not gate anything
-----------------------------------------------------------------
The store encodes the phase as ``sin(2*pi*w)`` / ``cos(2*pi*w)``, i.e. **modulo 1**, so the
integer turn never reaches the store: the snap ``w_fine = round(w_coarse - f) + f`` is
invisible in the encoding and ``f`` above is fully determined by the two faces alone.  The
prior is therefore **not** consulted by default (``snap_max = None``): a voxel is supervised
wherever ``faces_valid == 1``, ``sdf_in`` is inside the clip and the period is determined.
The coarse field is used only to orient the normal outward where the radial direction is
ambiguous.

The reason is measured, not stylistic.  On the crumpled crop the sheet spacing the faces imply
is ~28 fine voxels while the coarse lasagna field's own ``density`` implies ~167 -- a factor of
6: at 9.6 um the teacher cannot resolve a 28-voxel period and its phase is aliased there.
Gating the (correct) fine phase on the (aliased) coarse one would throw away good supervision
and keep bad.

Setting ``snap_max`` to a number turns the old gate back on as an **opt-in diagnostic**:

    resid = wrap(phase_coarse - f) in [-0.5, 0.5],   |resid| = |w_fine - w_coarse|,

and the voxel is refused when ``|resid| >= snap_max`` (the spec's "|w_fine - w_coarse| < 0.5
else invalid") and additionally required to have coarse ``valid == 1``.  ``resid`` is
accumulated into ``stats`` either way, so the aliasing stays measurable.

Phase origin
------------
``f = 0`` sits on the in face, whereas the coarse lasagna phase has its own (unrelated)
zero, so the two conventions differ by a roughly constant offset -- that offset *is* the
median ``|resid|`` the dev script reports.  ``phase_offset`` (turns, added to ``f``) lets
a build re-anchor the fine phase onto the coarse convention if the two are to be mixed in
one training run; the default 0 keeps the geometric definition.

Encodings are exactly the coarse store's (docs/label_store.md), so a consumer decodes
``wf_*`` with the same rules as ``phase_sin`` ... ``valid``.  The one unit difference is
``wf_density``: like the coarse ``density`` it is ``round(1000 * v)``, but ``v`` is in
wraps per **fine** (2.4 um) voxel -- which is what the training target is after
``upsample_coarse_targets`` divides the coarse density by the pitch factor.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import ndimage as ndi

from tsm.labels import decode_sdf_u8, decode_signed_u8, encode_signed_u8

__all__ = [
    "WF_CHANNELS",
    "WF_DEFAULTS",
    "wf_opts",
    "encode_density_u8",
    "decode_density_u8",
    "sheet_period",
    "winding_fine_box",
    "upsample_coarse_prior",
]

#: fine-store channels appended by ``extra.labels.winding_fine`` (always last)
WF_CHANNELS = ["wf_sin", "wf_cos", "wf_density", "wf_nx", "wf_ny", "wf_nz", "wf_conf", "wf_valid"]

#: the byte a signed channel uses for 0.0 (the neutral value of an invalid voxel)
_SIGNED_ZERO = int(encode_signed_u8(np.float32(0.0)))

WF_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    # None (default) = the coarse prior never gates the fine phase (it is aliased wherever
    # its period is coarser than the faces'; see the module docstring).  A number turns the
    # spec's consistency gate back on as a diagnostic: |wrap(phase_coarse - f)| >= it -> invalid.
    "snap_max": None,
    "min_period": 6.0,     # sheet spacing (voxels) outside [min, max] -> wf_valid = 0
    "max_period": 160.0,
    "sat_margin": 0.5,     # |sdf| >= clip - this counts as saturated (unusable)
    "min_grad": 0.5,       # |grad sdf| must be in [min_grad, max_grad] to vote for the normal
    "max_grad": 1.5,
    "phase_offset": 0.0,   # turns added to f (re-anchor onto the coarse phase convention)
    "require_faces_valid": True,   # False also accepts faces_valid == 2 (ignore) voxels
}


# --------------------------------------------------------------------------- #
# encodings
# --------------------------------------------------------------------------- #
def encode_density_u8(v: np.ndarray) -> np.ndarray:
    """wraps per voxel -> u8 = round(1000 v), clipped to 255 (the coarse ``density`` rule)."""
    return np.clip(np.rint(np.asarray(v, dtype=np.float32) * 1000.0), 0, 255).astype(np.uint8)


def decode_density_u8(u: np.ndarray) -> np.ndarray:
    return np.asarray(u, dtype=np.float32) / 1000.0


# --------------------------------------------------------------------------- #
# sheet spacing
# --------------------------------------------------------------------------- #
def _nearest_fill(values: np.ndarray, src: np.ndarray) -> np.ndarray | None:
    """``values`` propagated to every voxel from its nearest ``src`` voxel (EDT features)."""
    if not src.any():
        return None
    if src.all():
        return values
    idx = ndi.distance_transform_edt(~src, return_distances=False, return_indices=True)
    return np.ascontiguousarray(values[tuple(idx)])


def sheet_period(sdf_in: np.ndarray, sdf_out: np.ndarray, usable: np.ndarray, eps: float = 0.5
                 ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``(thickness, gap)`` fields (voxels) from the two SDFs of a box.

    ``D = sdf_in - sdf_out`` equals ``+t`` on one part of every period and ``-g`` on the
    rest, so each of the two is measured somewhere in every period and filled to the whole
    box from its nearest measurement.  Returns ``(None, None)`` when the box never shows
    one of the two (a box smaller than half a period, or no air gap at all).
    """
    d = np.asarray(sdf_in, dtype=np.float32) - np.asarray(sdf_out, dtype=np.float32)
    t = _nearest_fill(d, usable & (d > float(eps)))
    g = _nearest_fill(-d, usable & (d < -float(eps)))
    return t, g


# --------------------------------------------------------------------------- #
# the refined field
# --------------------------------------------------------------------------- #
def winding_fine_box(
    sdf_in_u8: np.ndarray,
    sdf_out_u8: np.ndarray,
    faces_valid: np.ndarray,
    prior: dict[str, np.ndarray] | None,
    radial: np.ndarray,
    clip: float = 20.0,
    *,
    snap_max: float | None = None,
    min_period: float = 6.0,
    max_period: float = 160.0,
    sat_margin: float = 0.5,
    min_grad: float = 0.5,
    max_grad: float = 1.5,
    phase_offset: float = 0.0,
    require_faces_valid: bool = True,
    stats: dict[str, Any] | None = None,
) -> tuple[np.ndarray, ...]:
    """Refined winding field of one (haloed) box: the 8 :data:`WF_CHANNELS` as uint8.

    Parameters
    ----------
    sdf_in_u8, sdf_out_u8, faces_valid
        The fine store's face channels for the box (uint8, store encoding; ``sdf`` byte 0 =
        no data, ``faces_valid`` 0 none / 1 supervise / 2 ignore).
    prior
        The coarse field sampled at the box, or ``None``.  Keys: ``"normal"`` (3, Z, Y, X)
        outward unit normal (zero where the coarse field is invalid) -- the **only** thing the
        default path uses, and only to orient the refined normal outward -- plus ``"phase"``
        (Z, Y, X) turns in [0, 1) and ``"valid"`` (Z, Y, X) uint8, needed when ``snap_max`` is
        set and used for the ``resid`` diagnostic otherwise.  See :func:`upsample_coarse_prior`.
    radial
        (3, Z, Y, X) outward radial direction (``tsm.labels.radial_field``), the orientation
        reference wherever the prior normal is missing.
    snap_max
        ``None`` (default): the prior never gates the result.  A number: refuse voxels whose
        phase disagrees with the prior by that much (and voxels the prior does not cover).

    Returns
    -------
    ``(wf_sin, wf_cos, wf_density, wf_nx, wf_ny, wf_nz, wf_conf, wf_valid)``, all uint8,
    in the coarse store's encodings (``wf_density`` in wraps per *fine* voxel).
    """
    shape = tuple(int(v) for v in np.shape(sdf_in_u8))
    if tuple(np.shape(sdf_out_u8)) != shape or tuple(np.shape(faces_valid)) != shape:
        raise ValueError(f"face channels must share a shape, got {shape}, "
                         f"{np.shape(sdf_out_u8)}, {np.shape(faces_valid)}")
    radial = np.asarray(radial, dtype=np.float32)
    if radial.shape != (3,) + shape:
        raise ValueError(f"radial must be (3,) + {shape}, got {radial.shape}")
    prior = prior or {}
    gated = snap_max is not None
    ph = prior.get("phase")
    pv = prior.get("valid")
    if gated and (ph is None or pv is None):
        raise ValueError("snap_max needs a prior with 'phase' and 'valid'")
    if ph is not None:
        ph = np.asarray(ph, dtype=np.float32)
        pv = np.ones(shape, np.uint8) if pv is None else np.asarray(pv)
        if ph.shape != shape or pv.shape != shape:
            raise ValueError(f"prior phase/valid must be {shape}, got {ph.shape}, {pv.shape}")
    pn = prior.get("normal")
    if pn is not None:
        pn = np.asarray(pn, dtype=np.float32)
        if pn.shape != (3,) + shape:
            raise ValueError(f"prior normal must be (3,) + {shape}, got {pn.shape}")
    clip = float(clip)

    zero = np.zeros(shape, np.uint8)
    neutral = np.full(shape, _SIGNED_ZERO, np.uint8)
    out_invalid = (neutral, neutral.copy(), zero, zero.copy(), neutral.copy(), neutral.copy(),
                   neutral.copy(), zero.copy())

    si = decode_sdf_u8(sdf_in_u8, clip)
    so = decode_sdf_u8(sdf_out_u8, clip)
    fv = np.asarray(faces_valid)
    data = (np.asarray(sdf_in_u8) > 0) & (np.asarray(sdf_out_u8) > 0) & (fv > 0)
    sat_in = np.abs(si) >= clip - float(sat_margin)
    sat_out = np.abs(so) >= clip - float(sat_margin)
    usable = data & ~sat_in & ~sat_out
    if stats is not None:
        stats["wf_boxes"] = stats.get("wf_boxes", 0) + 1

    t_f, g_f = sheet_period(si, so, usable)
    if t_f is None or g_f is None:
        if stats is not None:
            stats["wf_no_period_boxes"] = stats.get("wf_no_period_boxes", 0) + 1
            stats["wf_valid_hist"] = (stats.get("wf_valid_hist", np.zeros(2, np.int64))
                                      + np.array([int(np.prod(shape)), 0], np.int64))
        return out_invalid
    period = t_f + g_f
    ok_period = (period >= float(min_period)) & (period <= float(max_period))
    period = np.clip(period, float(min_period), float(max_period))
    del t_f, g_f

    # ---- phase -------------------------------------------------------------
    f = np.mod(si / period + float(phase_offset), 1.0).astype(np.float32)

    # ---- normal: grad of whichever SDF is locally well behaved -------------
    n = np.zeros((3,) + shape, np.float32)
    magsum = np.zeros(shape, np.float32)
    for fld, sat in ((si, sat_in), (so, sat_out)):
        gr = np.stack([np.gradient(fld, axis=a).astype(np.float32) if fld.shape[a] > 1
                       else np.zeros(shape, np.float32) for a in range(3)])
        mag = np.sqrt((gr * gr).sum(0))
        w = ((mag >= float(min_grad)) & (mag <= float(max_grad)) & data & ~sat).astype(np.float32)
        n += gr * w
        magsum += mag * w
        del gr, mag, w
    nrm = np.sqrt((n * n).sum(0))
    have_n = nrm > 1e-3
    # geometric confidence: the coherence |sum g| / sum |g| of the two SDF gradient votes --
    # 1 where only one SDF voted (a sawtooth seam of the other one is not a defect), and
    # cos(angle/2) where both did, so it falls off exactly where the local sheet geometry is
    # inconsistent (contacts, ambiguous faces, noise).
    geo = np.zeros(shape, np.float32)
    geo[have_n] = np.clip(nrm[have_n] / np.maximum(magsum[have_n], 1e-6), 0.0, 1.0)
    del magsum
    n[:, have_n] /= nrm[have_n]
    # fall back to the prior normal, else to the radial direction
    ref = radial if pn is None else np.where(np.sqrt((pn * pn).sum(0)) >= 0.5, pn, radial)
    n = np.where(have_n[None], n, ref)
    # orient outward (the same convention as the coarse store: n . r_hat >= 0)
    flip = (n * ref).sum(0) < 0
    n[:, flip] *= -1.0
    del flip, nrm

    # ---- validity, and (opt-in) the consistency gate ------------------------
    resid = None
    if ph is not None:
        diff = ph - f
        resid = (diff - np.rint(diff)).astype(np.float32)  # in [-0.5, 0.5]
        del diff
    face_ok = (fv == 1) if require_faces_valid else (fv > 0)
    # the geometry decides on its own: a face-supervised voxel inside the clip whose period
    # and normal are determined.  The coarse prior is not consulted unless snap_max is set.
    valid = data & face_ok & ~sat_in & ok_period & have_n
    conf = geo
    if gated:
        conf = geo * np.clip(1.0 - np.abs(resid) / max(float(snap_max), 1e-6), 0.0, 1.0)
        valid &= (pv == 1) & (np.abs(resid) < float(snap_max))
    del face_ok, ok_period, have_n, geo

    # ---- encode ------------------------------------------------------------
    ang = (2.0 * np.pi) * f
    sin_u8 = encode_signed_u8(np.sin(ang))
    cos_u8 = encode_signed_u8(np.cos(ang))
    dens_u8 = encode_density_u8(1.0 / period)
    nx_u8, ny_u8, nz_u8 = (encode_signed_u8(n[a]) for a in (2, 1, 0))
    conf_u8 = np.clip(np.rint(255.0 * conf), 0, 255).astype(np.uint8)
    valid_u8 = valid.astype(np.uint8)
    # invalid voxels carry the neutral encoding: 0 for density / conf, 0.0 (byte 128) for
    # every signed channel.  Consumers mask on wf_valid; this only keeps the bytes tidy.
    bad = ~valid
    for a in (dens_u8, conf_u8):
        a[bad] = 0
    for a in (sin_u8, cos_u8, nx_u8, ny_u8, nz_u8):
        a[bad] = _SIGNED_ZERO

    if stats is not None:
        stats["wf_valid_hist"] = (stats.get("wf_valid_hist", np.zeros(2, np.int64))
                                  + np.bincount(valid_u8.ravel(), minlength=2)[:2])
        if resid is not None:  # a pure diagnostic unless snap_max is set
            cand = valid & (pv == 1)
            stats["wf_resid_hist"] = (stats.get("wf_resid_hist", np.zeros(50, np.int64))
                                      + np.histogram(np.abs(resid[cand]), bins=50, range=(0.0, 0.5))[0])
            stats["wf_signed_resid_sum"] = stats.get("wf_signed_resid_sum", 0.0) + float(resid[cand].sum())
            stats["wf_resid_n"] = stats.get("wf_resid_n", 0) + int(cand.sum())
        stats["wf_period_sum"] = stats.get("wf_period_sum", 0.0) + float(period[valid].sum())
        stats["wf_period_n"] = stats.get("wf_period_n", 0) + int(valid.sum())
    return sin_u8, cos_u8, dens_u8, nx_u8, ny_u8, nz_u8, conf_u8, valid_u8


# --------------------------------------------------------------------------- #
# the coarse prior, sampled on a fine box
# --------------------------------------------------------------------------- #
def upsample_coarse_prior(coarse_u8: Any, lo: Sequence[int], hi: Sequence[int], scale: int = 4,
                          normal: bool = False, density: bool = False) -> dict[str, np.ndarray]:
    """Coarse phase (and validity, optionally the normal / density) on the fine box ``[lo, hi)``.

    Same alignment as :func:`tsm.labels.upsample_coarse_normal` (coarse voxel ``i`` covers
    fine voxels ``[i*k, (i+1)*k)``): sin/cos and density are mask-aware trilinear over
    ``valid == 1`` coarse voxels, ``valid`` is nearest.  Returns ``{"phase", "valid"}``
    (+ ``"normal"``, ``"density"``), with ``phase`` in turns in [0, 1), ``density`` in wraps
    per *fine* voxel (the coarse channel divided by ``scale``) and 0 / ``valid = 0`` where the
    coarse field is not valid in the neighbourhood.
    """
    from tsm.labels import upsample_coarse_normal

    k = int(scale)
    cshape = tuple(int(s) for s in coarse_u8.shape[1:])
    lo = [int(v) for v in lo]
    hi = [int(v) for v in hi]
    c0 = [max(0, int(np.floor((lo[a] + 0.5) / k - 0.5)) - 1) for a in range(3)]
    c1 = [min(cshape[a], int(np.ceil((hi[a] - 0.5) / k - 0.5)) + 2) for a in range(3)]
    c1 = [max(c1[a], c0[a] + 1) for a in range(3)]
    box = (slice(c0[0], c1[0]), slice(c0[1], c1[1]), slice(c0[2], c1[2]))
    v = np.asarray(coarse_u8[(7,) + box])
    m = (v == 1).astype(np.float32)
    sc = decode_signed_u8(np.asarray(coarse_u8[(slice(0, 2),) + box])) * m

    def zoom(a: np.ndarray, order: int) -> np.ndarray:
        return ndi.zoom(a, zoom=k, order=order, mode="nearest", grid_mode=True)

    take = []
    for a in range(3):
        i = np.arange(lo[a], hi[a]) - c0[a] * k
        take.append(np.clip(i, 0, (c1[a] - c0[a]) * k - 1))

    def crop(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(a[take[0]][:, take[1]][:, :, take[2]])

    mu = crop(zoom(m, 1))
    su = crop(zoom(sc[0], 1))
    cu = crop(zoom(sc[1], 1))
    vu = crop(zoom(v, 0))
    ok = mu >= 0.5
    phase = np.zeros(su.shape, np.float32)
    phase[ok] = np.mod(np.arctan2(su[ok], cu[ok]) / (2.0 * np.pi), 1.0)
    vu = np.where(ok, vu, np.uint8(0)).astype(np.uint8)
    out = {"phase": phase, "valid": vu}
    if density:
        dc = np.asarray(coarse_u8[(2,) + box]).astype(np.float32) / 1000.0 * m
        du = crop(zoom(dc, 1))
        out["density"] = np.where(ok, du / np.maximum(mu, 1e-6) / k, 0.0).astype(np.float32)
    if normal:
        out["normal"] = upsample_coarse_normal(coarse_u8, lo, hi, scale=k)
    return out


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
def wf_opts(raw: Any) -> dict[str, Any]:
    """``extra.labels.winding_fine``: ``true`` / ``false`` / an options object -> full dict."""
    opts = dict(WF_DEFAULTS)
    if raw is None or raw is False:
        return opts
    if raw is True:
        opts["enabled"] = True
        return opts
    if not isinstance(raw, dict):
        raise ValueError(f"extra.labels.winding_fine must be a bool or an object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(WF_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown extra.labels.winding_fine keys: {unknown} (known: {sorted(WF_DEFAULTS)})")
    enabled = bool(raw.get("enabled", True))  # an options object means "on" unless it says otherwise
    opts.update(raw)
    opts["enabled"] = enabled
    return opts
