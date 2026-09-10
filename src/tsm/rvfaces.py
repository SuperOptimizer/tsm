"""Two-face surface labels from the **upstream** Scroll-1 recto/verso band labels.

``tsm.labels.build_face_labels`` derives the two faces of a sheet from the CT itself
(threshold -> body -> boundary -> local march along the outward normal).  This module is the
second, *teacher-independent* source: the voxelised recto/verso mesh surfaces published
upstream, resampled onto our 2.4 um grid by ``dev/rectoverso_slab.py`` (see
:mod:`tsm.xframe`).  ``extra.labels.faces.source`` selects which one is used:

``"ct"`` (default)
    unchanged, :func:`tsm.labels.build_face_labels`.
``"rectoverso"``
    :func:`build_rv_face_labels` below.
``"merge"``
    both; the rectoverso result wins wherever it is valid, the CT builder fills the rest, and
    a ``faces_source`` channel records which one supplied every voxel
    (:func:`merge_face_labels`).

Upstream class encoding (channel 0 of ``rectoverso.zarr``): 0 background, 1 recto, 2 verso,
3 intersection = *contact*, i.e. the recto of one sheet touching the verso of the next.  Recto
is the face toward the scroll umbilicus, so **recto = our IN face and verso = our OUT face**;
a contact voxel belongs to *both* surfaces (it is the in face of the outer sheet and the out
face of the inner one at the same time), hence ``recto_band = {1, 3}`` and
``verso_band = {2, 3}``.

The bands are voxelised mesh surfaces from the coarser ~7.9 um frame, so on our grid they are
3-7 voxels thick and registered to ~1-2 voxels; they are thinned to a ~1-voxel medial sheet by
the 3-D EDT ridge (:func:`tsm.labels.medial_surface`) before anything else -- a genuine 3-D
operation, never per-slice.

Sign convention (docs/label_store.md): each SDF is positive on the **outward** side of its own
face (away from the umbilicus), so inside a sheet ``sdf_in > 0 > sdf_out`` and
``thickness = sdf_in - sdf_out``.  Both SDFs are distances to the *nearest* voxel of their own
face set -- the same rule ``build_face_labels`` uses -- so a sheet's interior is resolved only as
far as the point equidistant from its own face and the next sheet's; at a **contact** plane, which
belongs to both face sets, the two SDFs share a sign immediately on either side of it.  That is a
property of the representation, not of this builder.  The sign is decided from *geometry only* -- never from the
CT -- as ``sign((p - q) . n_out(q))`` with ``q`` the nearest face voxel of that face and
``n_out`` the outward direction there: the radial direction from the umbilicus axis, refined by
the local band normal (:func:`local_band_normals`) and oriented to agree with the radial one.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
from scipy import ndimage as ndi

from tsm.labels import (
    _drop_small_components,
    _signed_side,
    ct_mask,
    decode_sdf_u8,
    encode_sdf_u8,
    hist_percentiles,
    medial_surface,
)

__all__ = [
    "RV_CHANNELS",
    "FACE_SOURCES",
    "SOURCE_NONE",
    "SOURCE_RECTOVERSO",
    "SOURCE_CT",
    "SOURCE_HUMAN",
    "RV_BG",
    "RV_RECTO",
    "RV_VERSO",
    "RV_CONTACT",
    "thin_band",
    "local_band_normals",
    "build_rv_face_labels",
    "merge_face_labels",
    "open_rv_store",
    "read_rv_box",
    "disagree_summary",
]

#: extra fine-store channels written whenever ``faces.source != "ct"`` (appended last)
RV_CHANNELS = ["faces_source", "rv_class", "hzvt_class"]

FACE_SOURCES = ("ct", "rectoverso", "merge")
SOURCE_NONE, SOURCE_RECTOVERSO, SOURCE_CT = 0, 1, 2
#: ``faces_source`` code written by ``dev/annot_import.py`` where a human corrected the faces.
#: No builder ever produces it; it exists so a corrected voxel stays distinguishable from both
#: automatic sources for ever after (and can be weighted or held out on its own).
SOURCE_HUMAN = 3

RV_BG, RV_RECTO, RV_VERSO, RV_CONTACT = 0, 1, 2, 3

#: bins of the merge disagreement histograms: |delta sdf| in whole voxels, 0 .. DISAGREE_MAX
DISAGREE_MAX = 40


# --------------------------------------------------------------------------- #
# band thinning
# --------------------------------------------------------------------------- #
def thin_band(band: np.ndarray, min_component: int = 32, tol: float = 0.5) -> np.ndarray:
    """Thin a 3-7 voxel thick voxelised band to its ~1-voxel medial sheet.

    26-components smaller than ``min_component`` are dropped first (voxelisation specks), then
    the 3-D EDT ridge (:func:`tsm.labels.medial_surface`: foreground voxels whose distance to
    the band's complement is within ``tol`` of the 26-neighbourhood maximum) keeps the local
    maximum of the band's EDT *along the band normal* -- for a slab of odd width exactly the
    centre plane, and a sheet stays a sheet (unlike a curve skeleton).  Fully 3-D: no per-slice
    morphology anywhere.
    """
    band = np.asarray(band, dtype=bool)
    if not band.any():
        return band
    b = _drop_small_components(band, int(min_component))
    if not b.any():
        return b
    return medial_surface(b, tol=float(tol))


# --------------------------------------------------------------------------- #
# local outward direction
# --------------------------------------------------------------------------- #
def _window_moment(band_f: np.ndarray, w: int, *a: np.ndarray) -> np.ndarray:
    """Box-filtered sum of ``band * prod(a)`` over a ``w**3`` window (float64, zero outside)."""
    p = band_f
    for x in a:
        p = p * x
    return ndi.uniform_filter(p, size=int(w), mode="constant", cval=0.0)


def local_band_normals(band: np.ndarray, radial: np.ndarray, idx_zyx: tuple,
                       window: int = 7, min_count: int = 6, planarity_max: float = 0.5
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normal of the band sheet at the voxels ``idx_zyx`` (3 index arrays).

    Local PCA of the *band voxel positions* inside a ``window**3`` box centred on each voxel:
    the second-moment matrix is accumulated with box filters (so the whole field costs ten
    separable passes, and only the requested voxels are gathered out of it) and its eigenvector
    for the **smallest** eigenvalue is the sheet normal -- the band spreads in the two in-plane
    directions and is thin across, so the least-variance direction is the normal.

    The eigenvector is then oriented to agree with the radial direction from the umbilicus
    (``n . r_hat >= 0``), which is what makes it an *outward* normal: PCA alone gives an
    unsigned axis.  Where the window holds fewer than ``min_count`` band voxels, or the sheet
    is not clearly planar (``lambda0 > planarity_max * lambda1``, e.g. at a band's ragged edge
    or where two bands cross), the radial direction is used unrefined.

    Returns ``(normals (3, N), used_pca (N,))``.
    """
    n_pts = len(idx_zyx[0])
    rad = np.ascontiguousarray(np.asarray(radial, dtype=np.float32)[:, idx_zyx[0], idx_zyx[1], idx_zyx[2]])
    if n_pts == 0:
        return rad, np.zeros(0, dtype=bool)
    b = np.asarray(band, dtype=np.float64)
    shape = b.shape
    w = int(window)
    # centred coordinates keep the second moments small (float64 is exact enough either way)
    coords = [np.arange(s, dtype=np.float64).reshape([-1 if k == ax else 1 for k in range(3)])
              - 0.5 * (s - 1) for ax, s in enumerate(shape)]

    def gather(arr: np.ndarray) -> np.ndarray:
        return arr[idx_zyx[0], idx_zyx[1], idx_zyx[2]].copy()

    s0 = gather(_window_moment(b, w))
    s1 = np.stack([gather(_window_moment(b, w, coords[a])) for a in range(3)])
    cov = np.zeros((n_pts, 3, 3), dtype=np.float64)
    n0 = np.maximum(s0, 1e-12)
    mean = s1 / n0
    for a in range(3):
        for c in range(a, 3):
            m2 = gather(_window_moment(b, w, coords[a], coords[c])) / n0
            v = m2 - mean[a] * mean[c]
            cov[:, a, c] = v
            cov[:, c, a] = v
            del m2, v
    del b
    count = s0 * (w ** 3)  # uniform_filter returns the window *mean*
    evals, evecs = np.linalg.eigh(cov)
    del cov
    n = evecs[:, :, 0].T.astype(np.float32)            # smallest eigenvalue -> sheet normal
    planar = evals[:, 0] <= float(planarity_max) * np.maximum(evals[:, 1], 1e-12)
    ok = (count >= float(min_count)) & planar & np.isfinite(n).all(axis=0)
    del evals, evecs
    nrm = np.sqrt((n * n).sum(0))
    ok &= nrm > 1e-6
    n = n / np.maximum(nrm, 1e-6)
    # PCA gives an unsigned axis: orient it outward (agreeing with the radial direction)
    flip = (n * rad).sum(0) < 0
    n = np.where(flip[None], -n, n).astype(np.float32)
    out = np.where(ok[None], n, rad).astype(np.float32)
    return np.ascontiguousarray(out), ok


# --------------------------------------------------------------------------- #
# the builder
# --------------------------------------------------------------------------- #
def _signed_face(face: np.ndarray, n_field: np.ndarray, shape: tuple, clip: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(clipped signed distance to ``face``, unclipped distance) with the store's sign rule."""
    dist, idx = ndi.distance_transform_edt(~face, return_indices=True)
    idx = idx.astype(np.int32)
    n_f = np.ascontiguousarray(n_field[:, idx[0], idx[1], idx[2]])
    s = _signed_side(idx, n_f, shape)
    del n_f, idx
    dist = dist.astype(np.float32)
    sdf = s * np.minimum(dist, float(clip))
    sdf[face] = 0.0
    del s
    return sdf, dist


def build_rv_face_labels(
    rv_class: np.ndarray,
    ct_u8: np.ndarray,
    radial: np.ndarray,
    clip: float = 20.0,
    *,
    min_component: int = 32,
    normal_window: int = 7,
    min_normal_count: int = 6,
    planarity_max: float = 0.5,
    thin_tol: float = 0.5,
    ct_mask_threshold: float = 30.0,
    ct_sigma: float = 2.0,
    ct_mask_dilate: int = 2,
    stats: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(sdf_in_u8, sdf_out_u8, faces_valid_u8, thickness_u8)`` for one (haloed) box from the
    upstream recto/verso classes.

    * **Surfaces.**  in face = thinned ``{recto, contact}`` band, out face = thinned
      ``{verso, contact}`` band (:func:`thin_band`); a contact voxel is on *both* faces.
    * **SDFs.**  Signed EDT to each surface, clipped to +-``clip``, positive on the outward side
      of its own face.  The sign is geometric, never from the CT: for every voxel ``p`` with
      nearest face voxel ``q``, ``sign((p - q) . n_out(q))`` where ``n_out`` is the radial
      direction refined by the local band normal (:func:`local_band_normals`).
    * **faces_valid.**  1 where the voxel lies within ``clip`` of **both** an in-face and an
      out-face voxel (the sheet interior and its near exterior are then determined by two real
      surfaces), 2 (ignore) elsewhere; 0 outside the CT data mask, as in the CT builder.
    * **thickness.**  ``sdf_in - sdf_out`` (unclipped distances) inside the sheet
      (``sdf_in >= 0 >= sdf_out``), rounded and clipped to 255; 0 elsewhere.
    * **Guard.**  A box with no band voxels at all is entirely ignore.
    """
    rv = np.asarray(rv_class)
    shape = tuple(int(v) for v in rv.shape)
    if tuple(ct_u8.shape) != shape or tuple(radial.shape) != (3,) + shape:
        raise ValueError(f"shape mismatch: rv {shape} ct {ct_u8.shape} radial {radial.shape}")
    clip = float(clip)
    data = ct_mask(ct_u8, ct_mask_threshold, ct_sigma, ct_mask_dilate)

    recto = (rv == RV_RECTO) | (rv == RV_CONTACT)
    verso = (rv == RV_VERSO) | (rv == RV_CONTACT)
    thin_in = thin_band(recto, min_component, thin_tol)
    thin_out = thin_band(verso, min_component, thin_tol)
    del recto, verso

    if stats is not None:
        stats["rv_boxes"] = stats.get("rv_boxes", 0) + 1
        stats["rv_band_voxels"] = stats.get("rv_band_voxels", 0) + int((rv > 0).sum())
        stats["rv_thin_in_voxels"] = stats.get("rv_thin_in_voxels", 0) + int(thin_in.sum())
        stats["rv_thin_out_voxels"] = stats.get("rv_thin_out_voxels", 0) + int(thin_out.sum())

    if not thin_in.any() or not thin_out.any():
        # no band (or only one of the two faces) in this box: nothing is determined
        if stats is not None:
            stats["rv_no_band_boxes"] = stats.get("rv_no_band_boxes", 0) + 1
        flat = np.full(shape, encode_sdf_u8(np.float32(clip), clip), dtype=np.uint8)
        flat[~data] = 0
        return (flat, flat.copy(), np.where(data, np.uint8(2), np.uint8(0)),
                np.zeros(shape, np.uint8))

    # ---- outward direction: radial, refined by the local band normal -------
    pts = np.nonzero(thin_in | thin_out)
    band_all = (rv > 0)
    n_pts, used_pca = local_band_normals(band_all, radial, pts, int(normal_window),
                                         int(min_normal_count), float(planarity_max))
    del band_all
    n_field = np.array(radial, dtype=np.float32, copy=True)
    n_field[:, pts[0], pts[1], pts[2]] = n_pts
    if stats is not None:
        stats["rv_normal_pts"] = stats.get("rv_normal_pts", 0) + int(used_pca.size)
        stats["rv_normal_pca_used"] = stats.get("rv_normal_pca_used", 0) + int(used_pca.sum())
    del pts, n_pts, used_pca

    sdf_in, dist_in = _signed_face(thin_in, n_field, shape, clip)
    sdf_out, dist_out = _signed_face(thin_out, n_field, shape, clip)
    del n_field, thin_in, thin_out

    # ---- validity: both faces reachable within `clip` ----------------------
    determined = (dist_in <= clip) & (dist_out <= clip)
    faces_valid = np.where(data, np.where(determined, np.uint8(1), np.uint8(2)), np.uint8(0))
    del determined

    inside = (sdf_in >= 0.0) & (sdf_out <= 0.0)
    thickness = np.where(inside, dist_in + dist_out, 0.0)
    del inside, dist_in, dist_out

    sdf_in_u8 = encode_sdf_u8(sdf_in, clip)
    sdf_out_u8 = encode_sdf_u8(sdf_out, clip)
    del sdf_in, sdf_out
    sdf_in_u8[~data] = 0
    sdf_out_u8[~data] = 0
    thickness_u8 = np.clip(np.rint(thickness), 0, 255).astype(np.uint8)
    thickness_u8[~data] = 0
    if stats is not None:
        stats["rv_valid_hist"] = (stats.get("rv_valid_hist", np.zeros(3, np.int64))
                                  + np.bincount(faces_valid.ravel(), minlength=3)[:3])
    return sdf_in_u8, sdf_out_u8, faces_valid, thickness_u8


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def merge_face_labels(rv: Sequence[np.ndarray], ct: Sequence[np.ndarray], clip: float,
                      band: np.ndarray | None = None, stats: dict[str, Any] | None = None
                      ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Merge the two face builders: rectoverso wins where it is valid, CT fills the rest.

    ``rv`` / ``ct`` are the ``(sdf_in_u8, sdf_out_u8, faces_valid_u8, thickness_u8)`` tuples.
    Returns ``(merged 4-tuple, faces_source)`` with ``faces_source`` 0 none / 1 rectoverso /
    2 ct, so training and evaluation can weight or ablate by source.

    ``faces_valid`` of the merge is 1 where either source supervises, 0 where **both** report
    no data, else 2.  Where both are valid the per-voxel disagreement ``|delta sdf_in|`` /
    ``|delta sdf_out|`` is histogrammed into ``stats`` (restricted to ``band`` as well, when
    given) -- that is the calibration number between the CT geometry and the upstream meshes.
    """
    rvv, ctv = rv[2], ct[2]
    use_rv = rvv == 1
    use_ct = (~use_rv) & (ctv == 1)
    src = np.where(use_rv, np.uint8(SOURCE_RECTOVERSO),
                   np.where(use_ct, np.uint8(SOURCE_CT), np.uint8(SOURCE_NONE)))
    no_data = (rvv == 0) & (ctv == 0)
    valid = np.where(use_rv | use_ct, np.uint8(1), np.where(no_data, np.uint8(0), np.uint8(2)))
    out = [np.where(use_rv, rv[i], ct[i]).astype(np.uint8) for i in (0, 1)]
    out.append(valid)
    out.append(np.where(use_rv, rv[3], ct[3]).astype(np.uint8))
    for i in (0, 1, 3):
        out[i][valid == 0] = 0
    src[valid == 0] = SOURCE_NONE

    if stats is not None:
        both = (rvv == 1) & (ctv == 1)
        stats["merge_source_hist"] = (stats.get("merge_source_hist", np.zeros(3, np.int64))
                                      + np.bincount(src.ravel(), minlength=3)[:3])
        stats["merge_both_valid"] = stats.get("merge_both_valid", 0) + int(both.sum())
        # Per-source sheet thickness: the sanity check that the CT builder is measuring single
        # sheets and not welded sheet pairs.  A CT median around twice the rectoverso median
        # means the body closing bridged the inter-sheet gap again.
        for tag, arr, sel in (("rv", rv[3], rvv == 1), ("ct", ct[3], ctv == 1),
                              ("rv_both", rv[3], both), ("ct_both", ct[3], both)):
            v = arr[sel & (arr > 0)]
            if v.size:
                k = f"thickness_hist_{tag}"
                stats[k] = (stats.get(k, np.zeros(256, np.int64))
                            + np.bincount(v.ravel(), minlength=256)[:256])
            del v
        if both.any():
            for j, name in ((0, "sdf_in"), (1, "sdf_out")):
                d = np.abs(decode_sdf_u8(rv[j], clip) - decode_sdf_u8(ct[j], clip))
                for key, sel in (("all", both), ("band", both & band if band is not None else None)):
                    if sel is None or not sel.any():
                        continue
                    h = np.bincount(np.clip(np.rint(d[sel]), 0, DISAGREE_MAX).astype(np.int64),
                                    minlength=DISAGREE_MAX + 1)[:DISAGREE_MAX + 1]
                    k = f"disagree_{name}_{key}"
                    stats[k] = stats.get(k, np.zeros(DISAGREE_MAX + 1, np.int64)) + h
                del d
        del both
    return tuple(out), src


def disagree_summary(stats: dict[str, Any]) -> dict[str, Any]:
    """The merge calibration block for ``labels.summary.json`` (medians of the histograms)."""
    out: dict[str, Any] = {"both_valid_voxels": int(stats.get("merge_both_valid", 0))}
    sh = np.asarray(stats.get("merge_source_hist", np.zeros(3, np.int64)))
    tot = max(int(sh.sum()), 1)
    out["source_fraction"] = {"none": float(sh[0] / tot), "rectoverso": float(sh[1] / tot),
                              "ct": float(sh[2] / tot)}
    out["source_voxels"] = {"none": int(sh[0]), "rectoverso": int(sh[1]), "ct": int(sh[2])}
    th: dict[str, Any] = {}
    for tag in ("rv", "ct", "rv_both", "ct_both"):
        h = stats.get(f"thickness_hist_{tag}")
        if h is None:
            continue
        h = np.asarray(h, dtype=np.int64)
        p = hist_percentiles(h, (10, 50, 90))
        th[tag] = {"n": int(h[1:].sum()), "p10": p[0], "median": p[1], "p90": p[2]}
    if th:
        out["thickness_voxels_by_source"] = th
    for name in ("sdf_in", "sdf_out"):
        for key in ("all", "band"):
            h = stats.get(f"disagree_{name}_{key}")
            if h is None:
                continue
            h = np.asarray(h, dtype=np.int64)
            p = hist_percentiles(h, (50, 90), drop_zero=False)
            out[f"abs_delta_{name}_{key}"] = {
                "n": int(h.sum()), "median": p[0], "p90": p[1],
                "mean": float((h * np.arange(h.size)).sum() / max(h.sum(), 1)),
            }
    return out


# --------------------------------------------------------------------------- #
# store access
# --------------------------------------------------------------------------- #
def open_rv_store(path: str):
    """Open ``rectoverso.zarr`` -> ``(array, origin_zyx)``; channel 0 = rectoverso, 1 = hzvt."""
    import zarr

    p = os.path.expanduser(str(path))
    arr = zarr.open_array(store=p, mode="r")
    if int(arr.shape[0]) < 2:
        raise ValueError(f"{p}: expected >= 2 channels, got shape {arr.shape}")
    origin = tuple(int(v) for v in dict(arr.attrs).get("origin_zyx", (0, 0, 0)))
    return arr, origin


def read_rv_box(arr, origin_zyx: Sequence[int], lo: Sequence[int], hi: Sequence[int]) -> np.ndarray:
    """``(2, dz, dy, dx)`` block at **global** level-0 voxel coordinates, 0 (background)
    outside the store -- an unlabelled halo simply becomes ignore, never a fabricated band."""
    lo = [int(v) for v in lo]
    hi = [int(v) for v in hi]
    o = [int(v) for v in origin_zyx]
    shape = [int(s) for s in arr.shape[1:]]
    out = np.zeros((2,) + tuple(h - l for l, h in zip(lo, hi)), np.uint8)
    a = [max(0, l - oo) for l, oo in zip(lo, o)]
    b = [min(s, h - oo) for h, oo, s in zip(hi, o, shape)]
    if all(bb > aa for aa, bb in zip(a, b)):
        blk = np.asarray(arr[0:2, a[0]:b[0], a[1]:b[1], a[2]:b[2]])
        d = [aa + oo - l for aa, oo, l in zip(a, o, lo)]
        out[:, d[0]:d[0] + blk.shape[1], d[1]:d[1] + blk.shape[2], d[2]:d[2] + blk.shape[3]] = blk
    return out


class SubView:
    """Read-only ``(C, Z, Y, X)`` view of a sub-box of a larger store.

    A crop config reuses the whole-slab teacher stores: the arrays then cover a *larger* region
    than ``cfg.region``, so index 0 of the view is store index ``offset`` and the view reports
    the region's shape.  That keeps :func:`tsm.labels.read_box_padded` (which edge-replicates
    outside ``shape``) behaving exactly as it does for an exactly-matching store -- halos are
    replicated at the *region* faces, not silently filled with neighbouring data.
    """

    def __init__(self, arr, offset_zyx: Sequence[int], shape_zyx: Sequence[int]) -> None:
        self.arr = arr
        self.offset = tuple(int(v) for v in offset_zyx)
        self.ndim = int(arr.ndim)
        self.shape = tuple(int(s) for s in arr.shape[:self.ndim - 3]) + tuple(int(v) for v in shape_zyx)
        self.dtype = arr.dtype
        self.attrs = getattr(arr, "attrs", {})

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        lead, tail = key[:len(key) - 3], key[len(key) - 3:]
        sl = tuple(slice(s.start + o, s.stop + o) for s, o in zip(tail, self.offset))
        return self.arr[lead + sl]
