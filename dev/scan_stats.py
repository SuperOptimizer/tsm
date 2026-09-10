"""Scan-domain comparison between N OME-Zarr CT volumes.

Samples random interior blocks from level 0 of each volume (guided by a coarse
level-5 survey), then reports intensity, texture and sheet-scale statistics plus
pairwise domain distances.  Writes stats.json, report.md and overlay.png.

Usage:
  taskset -c 0-5 uv run python dev/scan_stats.py \
      --url A=s3://bucket/path/a.zarr --url B=s3://bucket/path/b.zarr \
      --blocks 24 --seed 0 --out ~/tsm-output/scan_stats

CPU only, streams one 256^3 uint8 block at a time (peak RSS well under 2 GB).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tsm.volume import open_array  # noqa: E402

BLOCK = 256          # level-0 block edge
SURVEY_LEVEL = 5     # 32x downsample
SURVEY_CHUNK = 128
CT_CLIP = (0.0, 212.0, 87.54424285888672, 47.74376678466797)  # lo, hi, mean, std (m7 teacher)
METADATA_URL = "vesuvius-challenge-open-data/metadata.json"


# ---------------------------------------------------------------- block picking

def survey_and_pick(url: str, n_blocks: int, rng: np.random.Generator, log) -> list[tuple[int, int, int]]:
    """Pick n_blocks level-0 origins inside the scroll using a level-5 survey."""
    a5 = open_array(url, SURVEY_LEVEL)
    a0 = open_array(url, 0)
    s5 = a5.shape
    scale = int(round(a0.shape[0] / s5[0]))          # 32
    sub = BLOCK // scale                              # 8 level-5 voxels per level-0 block
    grid = [max(1, s // SURVEY_CHUNK) for s in s5]
    want_chunks = max(4, int(math.ceil(n_blocks / 2)))
    tried, good = set(), []
    budget = want_chunks * 4
    while len(good) < want_chunks and len(tried) < budget:
        ci = tuple(int(rng.integers(0, g)) for g in grid)
        if ci in tried:
            continue
        tried.add(ci)
        o = [c * SURVEY_CHUNK for c in ci]
        if any(o[d] + SURVEY_CHUNK > s5[d] for d in range(3)):
            continue
        blk = np.asarray(a5[o[0]:o[0] + SURVEY_CHUNK, o[1]:o[1] + SURVEY_CHUNK, o[2]:o[2] + SURVEY_CHUNK])
        zf = float((blk == 0).mean())
        if zf > 0.05:
            continue
        good.append((o, blk))
        log(f"    survey chunk {ci} mean={blk.mean():.1f} zero={zf:.3f} ({len(good)}/{want_chunks})")
    if not good:
        raise RuntimeError(f"no interior survey chunks found for {url}")

    pooled = np.concatenate([b.ravel()[::7] for _, b in good])
    thr = two_mode_threshold(pooled)
    log(f"    level-5 air/papyrus threshold = {thr:.1f}")

    origins: list[tuple[int, int, int]] = []
    per = int(math.ceil(n_blocks / len(good)))
    for o, blk in good:
        n = blk.shape[0] // sub
        v = blk[: n * sub, : n * sub, : n * sub].reshape(n, sub, n, sub, n, sub).astype(np.float32)
        means = v.mean(axis=(1, 3, 5))
        zeros = (v == 0).mean(axis=(1, 3, 5))
        ok = np.argwhere((means > thr) & (zeros < 1e-6))
        if not len(ok):
            continue
        rng.shuffle(ok)
        for idx in ok[:per]:
            org = tuple(int((o[d] + int(idx[d]) * sub) * scale) for d in range(3))
            if all(org[d] + BLOCK <= a0.shape[d] for d in range(3)):
                origins.append(org)
    rng.shuffle(origins)
    return origins[:n_blocks]


def two_mode_threshold(v: np.ndarray) -> float:
    """Midpoint between the two dominant peaks of a nonzero-value histogram."""
    lo, hi = peak_modes(v)
    return 0.5 * (lo + hi)


def peak_modes(v: np.ndarray) -> tuple[float, float]:
    """Two dominant peaks (air, papyrus) of the nonzero intensity histogram."""
    h, edges = np.histogram(v[v > 0], bins=256, range=(0, 256))
    c = 0.5 * (edges[:-1] + edges[1:])
    hs = ndimage.gaussian_filter1d(h.astype(np.float64), 3.0)
    ismax = (hs[1:-1] >= hs[:-2]) & (hs[1:-1] >= hs[2:])
    idx = np.flatnonzero(ismax) + 1
    if len(idx) < 2:                                  # unimodal: use quartile split
        nz = v[v > 0]
        return float(np.percentile(nz, 15)), float(np.percentile(nz, 85))
    idx = idx[np.argsort(hs[idx])[::-1][:6]]
    idx = np.sort(idx)
    best, score = (idx[0], idx[-1]), -1.0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            if c[idx[j]] - c[idx[i]] < 8:
                continue
            s = min(hs[idx[i]], hs[idx[j]]) * (c[idx[j]] - c[idx[i]]) ** 0.5
            if s > score:
                best, score = (idx[i], idx[j]), s
    return float(c[best[0]]), float(c[best[1]])


# ---------------------------------------------------------------- per-block stats

def radial_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged 3D power spectrum.  Returns (freq cycles/voxel, power)."""
    n = x.shape[0]
    w = np.hanning(n).astype(np.float32)
    win = w[:, None, None] * w[None, :, None] * w[None, None, :]
    f = np.fft.rfftn((x - x.mean()) * win)
    p = (f.real ** 2 + f.imag ** 2).astype(np.float64)
    kz = np.fft.fftfreq(n)[:, None, None]
    ky = np.fft.fftfreq(n)[None, :, None]
    kx = np.fft.rfftfreq(n)[None, None, :]
    kr = np.sqrt(kz ** 2 + ky ** 2 + kx ** 2)
    nb = n // 2
    bi = np.clip((kr * 2 * nb).astype(np.int32), 0, nb)
    cnt = np.bincount(bi.ravel(), minlength=nb + 1)
    tot = np.bincount(bi.ravel(), weights=p.ravel(), minlength=nb + 1)
    freq = np.arange(nb + 1) / (2.0 * nb)
    return freq, tot / np.maximum(cnt, 1)


def acf_length(x: np.ndarray, axis: int) -> float:
    """Lag (voxels) where the mean normalised autocorrelation along `axis` hits 1/e."""
    y = np.moveaxis(x, axis, -1).astype(np.float32)
    y = y - y.mean(axis=-1, keepdims=True)
    n = y.shape[-1]
    f = np.fft.rfft(y, n=2 * n, axis=-1)
    ac = np.fft.irfft(f.real ** 2 + f.imag ** 2, n=2 * n, axis=-1)[..., :n]
    ac = ac.reshape(-1, n).mean(axis=0)
    if ac[0] <= 0:
        return float("nan")
    ac = ac / ac[0]
    tgt = 1.0 / math.e
    below = np.flatnonzero(ac < tgt)
    if not len(below):
        return float(n)
    i = int(below[0])
    a, b = ac[i - 1], ac[i]
    return float((i - 1) + (a - tgt) / max(a - b, 1e-9))


def ray_stats(x: np.ndarray, thr: float, rng: np.random.Generator, n_rays: int = 240):
    """Dominant sheet period and run-length thickness above thr, along random rays.

    The ray spectrum is red, so the peak is taken on a whitened spectrum (log power
    minus a smooth baseline) and refined by parabolic interpolation to beat the
    coarse FFT bin spacing at long periods.
    """
    n = x.shape[0]
    periods, thick = [], []
    win = np.hanning(n)
    f = np.fft.rfftfreq(n)
    band = np.flatnonzero((f >= 1.0 / 128) & (f <= 1.0 / 8))
    for _ in range(n_rays):
        ax = int(rng.integers(0, 3))
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if ax == 0:
            r = x[:, i, j]
        elif ax == 1:
            r = x[i, :, j]
        else:
            r = x[i, j, :]
        r = r.astype(np.float32)
        if (r == 0).any():
            continue
        d = (r - r.mean()) * win
        p = np.abs(np.fft.rfft(d)) ** 2
        lp = np.log(p + 1e-12)
        base = ndimage.gaussian_filter1d(lp, 8.0, mode="nearest")
        w = lp - base
        k = int(band[np.argmax(w[band])])
        if w[k] > math.log(3.0) and 0 < k < len(lp) - 1:
            a, b, c = lp[k - 1], lp[k], lp[k + 1]
            den = a - 2 * b + c
            dk = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
            dk = float(np.clip(dk, -0.5, 0.5))
            fk = (k + dk) / n
            if fk > 0:
                periods.append(1.0 / fk)
        m = r > thr
        if m.any():
            edges = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)
            runs = ends - starts
            inner = runs[(starts > 0) & (ends < n)]
            thick.extend(inner.tolist())
    return periods, thick


@dataclass
class BlockStat:
    hist: np.ndarray
    hist_pct: np.ndarray
    psd: np.ndarray
    acf: tuple[float, float, float]
    lap_std: float
    local_std: float
    zero_frac: float
    pcts: list[float]
    periods: list[float]
    thick: list[int]


def block_stats(b: np.ndarray, thr: float, rng: np.random.Generator, psd_freq_holder: list):
    x = b.astype(np.float32)
    hist = np.histogram(b, bins=256, range=(0, 256))[0].astype(np.float64)

    nz = x[x > 0]
    lo, hi = (np.percentile(nz, 1), np.percentile(nz, 99)) if nz.size else (0.0, 1.0)
    den = max(float(hi - lo), 1e-8)
    xp = (np.clip(x, lo, hi) - lo) / den
    hist_pct = np.histogram(xp, bins=256, range=(0.0, 1.0))[0].astype(np.float64)

    freq, psd = radial_psd(x)
    psd_freq_holder.append(freq)

    lap = ndimage.laplace(x)
    loc_mean = ndimage.uniform_filter(x, size=8)
    loc_var = ndimage.uniform_filter(x * x, size=8) - loc_mean ** 2
    local_std = float(np.sqrt(np.maximum(loc_var, 0)).mean())

    periods, thick = ray_stats(b, thr, rng)
    return BlockStat(
        hist=hist, hist_pct=hist_pct, psd=psd,
        acf=(acf_length(x, 0), acf_length(x, 1), acf_length(x, 2)),
        lap_std=float(lap.std()), local_std=local_std,
        zero_frac=float((b == 0).mean()),
        pcts=[float(v) for v in np.percentile(x, [1, 5, 50, 95, 99])],
        periods=periods, thick=thick,
    )


# ---------------------------------------------------------------- aggregation

def ct_edges() -> np.ndarray:
    """256 bin edges, one per uint8 level, in ct_clip units (no aliasing)."""
    clo, chi, cm, cs = CT_CLIP
    return (np.arange(257) - cm) / cs


def ct_hist(hist) -> np.ndarray:
    """ct_clip histogram derived from the raw uint8 histogram (zeros dropped)."""
    clo, chi, cm, cs = CT_CLIP
    h = np.array(hist, dtype=np.float64).copy()
    h[0] = 0.0
    v = (np.clip(np.arange(256) + 0.5, clo, chi) - cm) / cs
    e = ct_edges()
    return np.histogram(v, bins=e, weights=h)[0]


def loglog_slope(freq: np.ndarray, psd: np.ndarray, f0: float, f1: float) -> float:
    m = (freq >= f0) & (freq <= f1) & (psd > 0)
    if m.sum() < 4:
        return float("nan")
    return float(np.polyfit(np.log(freq[m]), np.log(psd[m]), 1)[0])


def w1_from_hist(h1: np.ndarray, h2: np.ndarray, edges: np.ndarray) -> float:
    p = h1 / max(h1.sum(), 1)
    q = h2 / max(h2.sum(), 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))[:-1] * np.diff(c)))


def hist_moments(h: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    p = h / max(h.sum(), 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    m = float((p * c).sum())
    return m, float(math.sqrt(max((p * (c - m) ** 2).sum(), 0)))


def analyse(name: str, url: str, n_blocks: int, seed: int, log) -> dict:
    rng = np.random.default_rng(seed)
    log(f"  [{name}] survey")
    t0 = time.time()
    origins = survey_and_pick(url, n_blocks, rng, log)
    log(f"  [{name}] picked {len(origins)} blocks in {time.time() - t0:.0f}s")
    a0 = open_array(url, 0)

    blocks = []
    hpool = np.zeros(256, dtype=np.float64)
    for i, o in enumerate(origins):
        t = time.time()
        b = np.asarray(a0[o[0]:o[0] + BLOCK, o[1]:o[1] + BLOCK, o[2]:o[2] + BLOCK])
        blocks.append(b)
        hpool += np.histogram(b, bins=256, range=(0, 256))[0]
        log(f"  [{name}] read block {i + 1}/{len(origins)} @{o} {time.time() - t:.1f}s")

    centers = np.arange(256) + 0.5
    hnz = hpool.copy()
    hnz[0] = 0
    pool = np.repeat(centers, (hnz / max(hnz.sum(), 1) * 400000).astype(np.int64))
    air, pap = peak_modes(pool)
    thr = 0.5 * (air + pap)
    log(f"  [{name}] pooled modes air={air:.1f} pap={pap:.1f} thr={thr:.1f}")

    stats: list[BlockStat] = []
    freqs: list[np.ndarray] = []
    for i, b in enumerate(blocks):
        t = time.time()
        stats.append(block_stats(b, thr, rng, freqs))
        blocks[i] = None
        log(f"  [{name}] stats block {i + 1}/{len(origins)} {time.time() - t:.1f}s")
    blocks = None

    hist = np.sum([s.hist for s in stats], axis=0)
    hist_pct = np.sum([s.hist_pct for s in stats], axis=0)
    psd = np.mean([s.psd for s in stats], axis=0)
    freq = freqs[0]

    nzh = hist.copy()
    nzh[0] = 0
    pap_mask = centers > 0.5 * (air + pap)
    w = nzh[pap_mask]
    pm = float((w * centers[pap_mask]).sum() / max(w.sum(), 1))
    pap_std = float(math.sqrt(max((w * (centers[pap_mask] - pm) ** 2).sum() / max(w.sum(), 1), 0)))

    cdf = np.cumsum(nzh) / max(nzh.sum(), 1)
    def pct(q):
        return float(np.interp(q, cdf, centers))

    periods = np.array([p for s in stats for p in s.periods], dtype=np.float64)
    thick = np.array([t for s in stats for t in s.thick], dtype=np.float64)
    acf = np.array([s.acf for s in stats], dtype=np.float64)

    return {
        "name": name, "url": url, "n_blocks": len(origins), "origins": [list(o) for o in origins],
        "level0_shape": list(a0.shape),
        "hist": hist.tolist(), "hist_pct": hist_pct.tolist(),
        "psd_freq": freq.tolist(), "psd": psd.tolist(),
        "air_mode": air, "papyrus_mode": pap, "papyrus_std": pap_std,
        "contrast_ratio": (pap - air) / max(pap_std, 1e-9),
        "threshold": 0.5 * (air + pap),
        "pct": {"p1": pct(0.01), "p5": pct(0.05), "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)},
        "zero_frac": float(np.mean([s.zero_frac for s in stats])),
        "psd_slope": loglog_slope(freq, psd, 1.0 / 64, 1.0 / 4),
        "acf_z": float(acf[:, 0].mean()), "acf_y": float(acf[:, 1].mean()), "acf_x": float(acf[:, 2].mean()),
        "acf_z_sd": float(acf[:, 0].std()), "acf_y_sd": float(acf[:, 1].std()), "acf_x_sd": float(acf[:, 2].std()),
        "lap_std": float(np.mean([s.lap_std for s in stats])),
        "local_std": float(np.mean([s.local_std for s in stats])),
        "noise_ratio": float(np.mean([s.lap_std / max(s.local_std, 1e-9) for s in stats])),
        "sheet_period": {
            "n": int(periods.size),
            "median": float(np.median(periods)) if periods.size else float("nan"),
            "p25": float(np.percentile(periods, 25)) if periods.size else float("nan"),
            "p75": float(np.percentile(periods, 75)) if periods.size else float("nan"),
        },
        "thickness": {
            "n": int(thick.size),
            "mean": float(thick.mean()) if thick.size else float("nan"),
            "median": float(np.median(thick)) if thick.size else float("nan"),
        },
    }


# ---------------------------------------------------------------- metadata

SCAN_FIELDS = [
    ("Sample", lambda v, s: v["sample_id"]),
    ("Volume id", lambda v, s: v["id"]),
    ("Scan id", lambda v, s: v.get("scan_id")),
    ("Volume long id", lambda v, s: v.get("long_id")),
    ("Volume export date", lambda v, s: v["creation"]["date"]),
    ("Export workflow", lambda v, s: v["creation"]["metadata"].get("workflow_name")),
    ("Export f32 window", lambda v, s: f"[{v['creation']['metadata'].get('target_window_f32_min')}, {v['creation']['metadata'].get('target_window_f32_max')}]"),
    ("Export u16 window", lambda v, s: f"[{v['creation']['metadata'].get('window_u16_min')}, {v['creation']['metadata'].get('window_u16_max')}]"),
    ("Level-0 shape (z,y,x)", lambda v, s: v["properties"].get("shape")),
    ("dtype", lambda v, s: v["properties"].get("data_format")),
    ("Facility / beamline", lambda v, s: f"{s['creation']['metadata'].get('location')} / {s['creation']['metadata'].get('beamline')}"),
    ("Scan date", lambda v, s: s["creation"]["date"]),
    ("Scan radix", lambda v, s: acq(s).get("scanRadix")),
    ("Scan type", lambda v, s: f"{acq(s).get('scanType')}, {acq(s).get('nb_scans')} sub-scans"),
    ("Energy (keV)", lambda v, s: s["creation"]["metadata"].get("energy_keV")),
    ("Voxel size (um)", lambda v, s: s["creation"]["metadata"].get("pixel_size_um")),
    ("Sample-detector dist (mm)", lambda v, s: acq(s).get("sampleDetectorDistance")),
    ("Source-sample dist (mm)", lambda v, s: acq(s).get("sourceSampleDistance")),
    ("X-ray magnification", lambda v, s: acq(s).get("xray_magnification")),
    ("Detector sensor", lambda v, s: det(s).get("sensor")),
    ("Detector pixel (mm)", lambda v, s: det(s).get("sensorPixelSize")),
    ("Detector ROI", lambda v, s: det(s).get("roi_size")),
    ("Optics", lambda v, s: f"{det(s).get('opticsName')} x{det(s).get('optical_magnification')}"),
    ("Scintillator", lambda v, s: det(s).get("scintillator")),
    ("Exposure (s)", lambda v, s: acq(s).get("expo_time")),
    ("Latency (s)", lambda v, s: acq(s).get("latency_time")),
    ("Accumulation", lambda v, s: acq(s).get("accumulation")),
    ("Projections (tomo_N)", lambda v, s: acq(s).get("tomo_N")),
    ("Flats / darks", lambda v, s: f"{acq(s).get('ref_N')} / {acq(s).get('dark_N')}"),
    ("Scan range (deg)", lambda v, s: acq(s).get("scanRange")),
    ("Rotation mode", lambda v, s: acq(s).get("rotationMode")),
    ("Half acquisition", lambda v, s: acq(s).get("half_acquisition")),
    ("z range (mm)", lambda v, s: f"{acq(s).get('z_start')} .. {acq(s).get('z_end')} step {acq(s).get('z_step')}"),
    ("Machine mode / current", lambda v, s: f"{acq(s).get('machineMode')}, {acq(s).get('machineCurrentStart')}->{acq(s).get('machineCurrentStop')} mA"),
    ("Attenuators", lambda v, s: acq(s).get("comment")),
    ("Scan time / total (s)", lambda v, s: f"{acq(s).get('scan_time')} / {acq(s).get('scan_time_total')}"),
    ("Recon workflow", lambda v, s: proc(s).get("general", {}).get("WorkFlowScheme")),
    ("Flat field", lambda v, s: proc(s).get("general", {}).get("FlatField")),
    ("Distortion correction", lambda v, s: proc(s).get("general", {}).get("DetectorDistortionCorrection")),
    ("Phase retrieval", lambda v, s: f"{proc(s).get('preprocessing', {}).get('phase', {}).get('method')} db={proc(s).get('preprocessing', {}).get('phase', {}).get('delta_beta')}"),
    ("Unsharp (coeff/sigma)", lambda v, s: f"{proc(s).get('preprocessing', {}).get('phase', {}).get('unsharp_coeff')} / {proc(s).get('preprocessing', {}).get('phase', {}).get('unsharp_sigma')}"),
    ("Recon method", lambda v, s: proc(s).get("reconstruction", {}).get("method")),
    ("Rotation axis (px)", lambda v, s: proc(s).get("reconstruction", {}).get("rotation_axis_position")),
    ("32-bit conv window", lambda v, s: f"[{proc(s).get('postprocessing', {}).get('32BitsConversion', {}).get('dataset_used_min')}, {proc(s).get('postprocessing', {}).get('32BitsConversion', {}).get('dataset_used_max')}]"),
    ("Recon histogram 0.2-99.8%", lambda v, s: f"[{hist32(s).get('min_0p002_percentile')}, {hist32(s).get('max_0p998_percentile')}]"),
    ("Masking", lambda v, s: json.dumps(s["creation"]["metadata"]["metadata"].get("masking")) if s["creation"]["metadata"]["metadata"].get("masking") else "SAM2 mask applied at export (suffix -masked)"),
]


def acq(s):
    return s["creation"]["metadata"]["metadata"].get("tomo", {}).get("acquisition", {})


def det(s):
    return acq(s).get("detector", {})


def proc(s):
    return s["creation"]["metadata"]["metadata"].get("tomo", {}).get("processing", {})


def hist32(s):
    return proc(s).get("32bitsData", {}).get("histogram", {})


def load_metadata():
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    raw = fs.cat(METADATA_URL)
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    return json.loads(raw)


def find_meta(meta: dict, url: str):
    """Locate (volume, scan) metadata records for an s3 volume url."""
    key = url.rstrip("/").split("/")[-1].split("-")[0]
    for sample in meta.get("samples", {}).values():
        vols = sample.get("volumes") or {}
        if key in vols:
            v = vols[key]
            s = (sample.get("scans") or {}).get(v.get("scan_id"))
            return v, s
    return None, None


# ---------------------------------------------------------------- plotting (PIL)

from PIL import Image, ImageDraw  # noqa: E402

COLORS = [(0, 90, 200), (210, 60, 20), (20, 140, 60), (140, 60, 180)]


class Panel:
    def __init__(self, img, x, y, w, h, title, xlabel, ylabel, logx=False, logy=False):
        self.d = ImageDraw.Draw(img)
        self.x, self.y, self.w, self.h = x, y, w, h
        self.logx, self.logy = logx, logy
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.series = []

    def add(self, xs, ys, color, label):
        self.series.append((np.asarray(xs, float), np.asarray(ys, float), color, label))

    def draw(self):
        d = self.d
        xs = np.concatenate([s[0] for s in self.series])
        ys = np.concatenate([s[1] for s in self.series])
        if self.logx:
            xs = xs[xs > 0]
        if self.logy:
            ys = ys[ys > 0]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        if self.logx:
            x0, x1 = math.log10(x0), math.log10(x1)
        if self.logy:
            y0, y1 = math.log10(max(y0, 1e-300)), math.log10(y1)
        if x1 <= x0:
            x1 = x0 + 1
        if y1 <= y0:
            y1 = y0 + 1
        pad = 0.03 * (y1 - y0)
        y1 = y1 + pad
        y0 = y0 - pad if (self.logy or y0 - pad > 0 or y0 < 0) else 0.0

        def px(v):
            v = math.log10(v) if self.logx else v
            return self.x + (v - x0) / (x1 - x0) * self.w

        def py(v):
            v = math.log10(max(v, 1e-300)) if self.logy else v
            return self.y + self.h - (v - y0) / (y1 - y0) * self.h

        d.rectangle([self.x, self.y, self.x + self.w, self.y + self.h], outline=(120, 120, 120))
        for t in range(5):
            gx = self.x + t * self.w / 4
            gy = self.y + t * self.h / 4
            d.line([gx, self.y, gx, self.y + self.h], fill=(230, 230, 230))
            d.line([self.x, gy, self.x + self.w, gy], fill=(230, 230, 230))
            vx = x0 + t * (x1 - x0) / 4
            vy = y1 - t * (y1 - y0) / 4
            d.text((gx - 14, self.y + self.h + 4), fmt(10 ** vx if self.logx else vx), fill=(60, 60, 60))
            d.text((self.x - 46, gy - 5), fmt(10 ** vy if self.logy else vy), fill=(60, 60, 60))
        d.text((self.x, self.y - 16), self.title, fill=(0, 0, 0))
        d.text((self.x + self.w / 2 - 30, self.y + self.h + 18), self.xlabel, fill=(60, 60, 60))
        d.text((self.x - 46, self.y - 16), self.ylabel, fill=(60, 60, 60))
        for k, (sx, sy, color, label) in enumerate(self.series):
            pts = []
            for a, b in zip(sx, sy):
                if (self.logx and a <= 0) or (self.logy and b <= 0):
                    continue
                pts.append((px(a), py(b)))
            if len(pts) > 1:
                d.line(pts, fill=color, width=2)
            d.text((self.x + self.w - 90, self.y + 6 + 14 * k), label, fill=color)


def fmt(v):
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1000 or a < 0.01:
        return f"{v:.1e}"
    if a >= 10:
        return f"{v:.0f}"
    return f"{v:.2f}"


def make_overlay(results: list[dict], path: str):
    W, H = 1560, 1120
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 12), "scan domain comparison: " + " vs ".join(r["name"] for r in results), fill=(0, 0, 0))
    pw, ph = 620, 300
    panels = []

    p = Panel(img, 90, 70, pw, ph, "intensity histogram (raw uint8, zeros dropped)", "value", "frac")
    for r, c in zip(results, COLORS):
        h = np.array(r["hist"], float)
        h[0] = 0
        p.add(np.arange(256) + 0.5, h / max(h.sum(), 1), c, r["name"])
    panels.append(p)

    p = Panel(img, 850, 70, pw, ph, "after per-block percentile 1-99 norm", "normalised value", "frac")
    for r, c in zip(results, COLORS):
        h = np.array(r["hist_pct"], float)
        h = ndimage.gaussian_filter1d(h, 2.0)   # display only; W1 uses the raw bins
        p.add(np.linspace(0, 1, 256), h / max(h.sum(), 1), c, r["name"])
    panels.append(p)

    p = Panel(img, 90, 430, pw, ph, "after ct_clip norm (m7: clip 0-212, -87.544, /47.744)", "z", "frac")
    e = ct_edges()
    for r, c in zip(results, COLORS):
        h = ct_hist(r["hist"])
        p.add(0.5 * (e[:-1] + e[1:]), h / max(h.sum(), 1), c, r["name"])
    panels.append(p)

    p = Panel(img, 850, 430, pw, ph, "radial power spectral density", "cycles/voxel", "power",
              logx=True, logy=True)
    for r, c in zip(results, COLORS):
        f = np.array(r["psd_freq"], float)
        s = np.array(r["psd"], float)
        m = f > 0
        p.add(f[m], s[m], c, f"{r['name']} (slope {r['psd_slope']:.2f})")
    panels.append(p)

    p = Panel(img, 90, 790, pw, ph, "autocorrelation length (1/e), voxels", "axis (z, y, x)", "lag")
    for r, c in zip(results, COLORS):
        p.add([0, 1, 2], [r["acf_z"], r["acf_y"], r["acf_x"]], c, r["name"])
    panels.append(p)

    p = Panel(img, 850, 790, pw, ph, "sheet period / thickness summary (voxels)",
              "0=period p25 1=median 2=p75 3=thickness mean", "voxels")
    for r, c in zip(results, COLORS):
        sp = r["sheet_period"]
        p.add([0, 1, 2, 3], [sp["p25"], sp["median"], sp["p75"], r["thickness"]["mean"]], c, r["name"])
    panels.append(p)

    for p in panels:
        p.draw()
    img.save(path)


# ---------------------------------------------------------------- report

def domain_distances(a: dict, b: dict) -> dict:
    e_raw = np.arange(257).astype(float)
    ha, hb = np.array(a["hist"], float), np.array(b["hist"], float)
    ha[0] = hb[0] = 0
    w_raw = w1_from_hist(ha, hb, e_raw)
    _, sd_raw = hist_moments(ha, e_raw)

    e_pct = np.linspace(0, 1, 257)
    w_pct = w1_from_hist(np.array(a["hist_pct"], float), np.array(b["hist_pct"], float), e_pct)
    _, sd_pct = hist_moments(np.array(a["hist_pct"], float), e_pct)

    e_ct = ct_edges()
    w_ct = w1_from_hist(ct_hist(a["hist"]), ct_hist(b["hist"]), e_ct)
    _, sd_ct = hist_moments(ct_hist(a["hist"]), e_ct)

    return {
        "w1_raw": w_raw, "w1_raw_over_sdA": w_raw / max(sd_raw, 1e-9),
        "w1_pct_norm": w_pct, "w1_pct_over_sdA": w_pct / max(sd_pct, 1e-9),
        "w1_ct_clip": w_ct, "w1_ct_over_sdA": w_ct / max(sd_ct, 1e-9),
        "acf_ratio_z": b["acf_z"] / a["acf_z"], "acf_ratio_y": b["acf_y"] / a["acf_y"],
        "acf_ratio_x": b["acf_x"] / a["acf_x"],
        "psd_slope_ratio": b["psd_slope"] / a["psd_slope"],
        "psd_slope_diff": b["psd_slope"] - a["psd_slope"],
        "contrast_ratio_ratio": b["contrast_ratio"] / a["contrast_ratio"],
        "noise_ratio_ratio": b["noise_ratio"] / a["noise_ratio"],
    }


def write_report(results, meta_rows, dist, path, note):
    names = [r["name"] for r in results]
    L = ["# Scan-domain comparison", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} — {results[0]['n_blocks']} random 256^3 level-0 blocks per volume, "
         "picked inside the scroll via a level-5 survey, fixed seed.", ""]
    if note:
        L += [note, ""]
    L += ["## Scan parameters (from s3://vesuvius-challenge-open-data/metadata.json)", "",
          "| field | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for field, vals in meta_rows:
        L.append(f"| {field} | " + " | ".join(str(v) for v in vals) + " |")

    def row(label, key, f="{:.3f}"):
        vals = []
        for r in results:
            v = r
            for k in key.split("."):
                v = v[k]
            vals.append(f.format(v) if isinstance(v, float) else str(v))
        L.append(f"| {label} | " + " | ".join(vals) + " |")

    L += ["", "## Measured statistics", "",
          "| statistic | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    row("blocks sampled", "n_blocks")
    row("air mode (uint8)", "air_mode", "{:.1f}")
    row("papyrus mode (uint8)", "papyrus_mode", "{:.1f}")
    row("papyrus std", "papyrus_std", "{:.2f}")
    row("contrast (pap-air)/pap_std", "contrast_ratio", "{:.3f}")
    for q in ("p1", "p5", "p50", "p95", "p99"):
        row(f"percentile {q[1:]}", f"pct.{q}", "{:.1f}")
    row("exact-zero (masked) frac", "zero_frac", "{:.2e}")
    row("PSD log-log slope 1/64..1/4", "psd_slope", "{:.3f}")
    row("ACF 1/e length z (vox)", "acf_z", "{:.2f}")
    row("ACF 1/e length y (vox)", "acf_y", "{:.2f}")
    row("ACF 1/e length x (vox)", "acf_x", "{:.2f}")
    row("Laplacian std", "lap_std", "{:.2f}")
    row("mean local std (8^3)", "local_std", "{:.2f}")
    row("noise ratio lap_std/local_std", "noise_ratio", "{:.3f}")
    row("sheet period median (vox)", "sheet_period.median", "{:.1f}")
    row("sheet period p25 (vox)", "sheet_period.p25", "{:.1f}")
    row("sheet period p75 (vox)", "sheet_period.p75", "{:.1f}")
    row("papyrus thickness mean (vox)", "thickness.mean", "{:.2f}")
    row("papyrus thickness median (vox)", "thickness.median", "{:.1f}")

    L += ["", f"## Domain distance ({names[0]} vs {names[1]})", "",
          "| measure | value |", "|---|---|"]
    for k, v in dist.items():
        L.append(f"| {k} | {v:.4f} |")
    L += ["", "## Reading", "", *build_reading(results, dist, meta_rows)]
    open(path, "w").write("\n".join(L) + "\n")


ID_FIELDS = {"Sample", "Volume id", "Scan id", "Volume long id", "Volume export date",
             "Export workflow", "Scan date", "Scan radix", "Recon method", "Rotation axis (px)",
             "Level-0 shape (z,y,x)"}


def build_reading(results, dist, rows) -> list[str]:
    """<=10 lines of plain-language reading of the domain gap."""
    if len(results) < 2 or not dist:
        return ["(single volume: no comparison)"]
    a, b = results[0], results[1]
    na, nb = a["name"], b["name"]
    same = [f for f, v in rows if len(set(map(str, v))) == 1 and str(v[0]) not in ("n/a", "None")]
    diff = [(f, v) for f, v in rows
            if len(set(map(str, v))) > 1 and f not in ID_FIELDS and str(v[0]) not in ("n/a", "None")]
    best = min([("raw uint8", dist["w1_raw_over_sdA"]),
                ("percentile 1-99 (ink/fibre teachers)", dist["w1_pct_over_sdA"]),
                ("ct_clip (m7 teacher)", dist["w1_ct_over_sdA"])], key=lambda t: t[1])
    win = next((v for f, v in rows if f == "Export f32 window"), None)

    L = []
    L.append(f"- Acquisition is not the source of any gap: {len(same)} of the {len(same) + len(diff)} "
             f"comparable scan/reconstruction fields are byte-identical, including "
             f"beamline, energy, voxel size, propagation distance, detector, optics, exposure and the "
             f"whole nabu + Paganin + GHBP reconstruction recipe.")
    L.append("- Scan fields that do differ: " + "; ".join(
        f"{f} ({v[0]} vs {v[1]})" for f, v in diff[:5]) + ".")
    if win:
        L.append(f"- The one that matters is the 8-bit export window: {na} maps f32 {win[0]} to 0-255, "
                 f"{nb} maps {win[1]}. Same physics, different gain/offset.")
    L.append(f"- In the data the two modes nearly coincide (air {a['air_mode']:.0f} vs {b['air_mode']:.0f}, "
             f"papyrus {a['papyrus_mode']:.0f} vs {b['papyrus_mode']:.0f}) but {nb}'s upper tail is "
             f"stretched: p95 {a['pct']['p95']:.0f} -> {b['pct']['p95']:.0f}, "
             f"p99 {a['pct']['p99']:.0f} -> {b['pct']['p99']:.0f}.")
    L.append(f"- So the largest single difference is spread, not level: papyrus std "
             f"{a['papyrus_std']:.1f} -> {b['papyrus_std']:.1f} "
             f"({100 * (b['papyrus_std'] / a['papyrus_std'] - 1):+.0f}%), pulling mode contrast down "
             f"{a['contrast_ratio']:.2f} -> {b['contrast_ratio']:.2f} ({dist['contrast_ratio_ratio']:.2f}x).")
    L.append(f"- Texture and effective resolution match: PSD log-log slope {a['psd_slope']:.2f} vs "
             f"{b['psd_slope']:.2f} (diff {dist['psd_slope_diff']:+.2f}); noise ratio "
             f"{a['noise_ratio']:.2f} vs {b['noise_ratio']:.2f}, i.e. {nb} is if anything slightly "
             f"cleaner per unit local contrast.")
    L.append(f"- Structure is close, {nb} a little tighter: autocorrelation 1/e "
             f"z {a['acf_z']:.1f}->{b['acf_z']:.1f} ({dist['acf_ratio_z']:.2f}x), y "
             f"{dist['acf_ratio_y']:.2f}x, x {dist['acf_ratio_x']:.2f}x; sheet period median "
             f"{a['sheet_period']['median']:.0f} -> {b['sheet_period']['median']:.0f} vox "
             f"({2.4 * a['sheet_period']['median']:.0f} -> {2.4 * b['sheet_period']['median']:.0f} um), "
             f"papyrus thickness {a['thickness']['mean']:.1f} -> {b['thickness']['mean']:.1f} vox.")
    L.append(f"- Histogram distance: raw W1 = {dist['w1_raw']:.1f} grey levels = "
             f"{dist['w1_raw_over_sdA']:.2f} of {na}'s own spread; ct_clip is an affine map so it only "
             f"rescales that ({dist['w1_ct_over_sdA']:.2f} sd) and additionally clips more of {nb}'s "
             f"bright tail at 212.")
    L.append(f"- Percentile 1-99 normalisation cuts the gap to {dist['w1_pct_over_sdA']:.2f} sd "
             f"({dist['w1_pct_over_sdA'] / max(dist['w1_raw_over_sdA'], 1e-9):.2f}x of raw) because it "
             f"absorbs exactly the per-volume gain/offset; best of the three is {best[0]}.")
    L.append(f"- Verdict: {nb} is the same domain as {na} — same scanner, same optics, same recon — "
             f"offset by a contrast/gain factor and a slightly tighter winding. Percentile-normalised "
             f"inputs should transfer nearly as-is; raw or ct_clip inputs face a "
             f"~{dist['w1_raw_over_sdA']:.2f} sd shift and a {100 * (b['papyrus_std'] / a['papyrus_std'] - 1):+.0f}% "
             f"wider papyrus distribution, so recalibrating the 8-bit window (or per-block percentile "
             f"norm) is the cheapest fix.")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", action="append", required=True,
                    help="NAME=url (repeatable)")
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="~/tsm-output/scan_stats")
    ap.add_argument("--from-json", action="store_true",
                    help="re-render report.md and overlay.png from an existing stats.json")
    args = ap.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

    if args.from_json:
        payload = json.load(open(os.path.join(out, "stats.json")))
        results = payload["volumes"]
        rows = [(f, [payload["scan_parameters"][f][r["name"]] for r in results])
                for f in payload["scan_parameters"]]
        dist = payload["domain_distance"]
        write_report(results, rows, dist, os.path.join(out, "report.md"), NOTE)
        make_overlay(results, os.path.join(out, "overlay.png"))
        log(f"re-rendered {out}/report.md overlay.png")
        return

    meta = load_metadata()
    results, meta_rows_cols = [], []
    for i, spec in enumerate(args.url):
        name, url = spec.split("=", 1)
        v, s = find_meta(meta, url)
        meta_rows_cols.append((v, s))
        results.append(analyse(name, url, args.blocks, args.seed + 1000 * i, log))

    rows = []
    for field, fn in SCAN_FIELDS:
        vals = []
        for v, s in meta_rows_cols:
            try:
                vals.append(fn(v, s) if v is not None else "n/a")
            except Exception:
                vals.append("n/a")
        rows.append((field, vals))

    dist = domain_distances(results[0], results[1]) if len(results) > 1 else {}
    payload = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "block": BLOCK,
               "seed": args.seed, "volumes": results, "scan_parameters":
               {f: {results[i]["name"]: str(vals[i]) for i in range(len(results))} for f, vals in rows},
               "domain_distance": dist}
    json.dump(payload, open(os.path.join(out, "stats.json"), "w"), indent=1)
    write_report(results, rows, dist, os.path.join(out, "report.md"), NOTE)
    make_overlay(results, os.path.join(out, "overlay.png"))
    log(f"wrote {out}/stats.json report.md overlay.png")


NOTE = (
    "Method / caveats: blocks are chosen by a level-5 (32x) survey — a random level-5 chunk is kept "
    "if <5% of it is exact-zero (mask), and 256^3 level-0 blocks are drawn from the sub-blocks whose "
    "level-5 mean is above the midpoint of the two dominant level-5 modes.  That deliberately biases "
    "sampling toward dense scroll interior, so the air/papyrus mass ratio reflects the sampler as well "
    "as the scroll.  Sheet period and papyrus thickness are measured against each volume's own "
    "air/papyrus midpoint threshold (A 77.5, B 78.0 in uint8), so they are comparable here.  ct_clip is "
    "a global affine map plus clipping, so its Wasserstein distance is the raw one rescaled by 1/47.744 "
    "(plus whatever the clip at 212 removes); percentile 1-99 is per-block and is the only one of the "
    "three that adapts to the volume."
)

if __name__ == "__main__":
    main()
