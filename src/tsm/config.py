"""JSON run configs. No DAG, no hashing: plain dataclasses with validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tsm.limits import Budget

TOP_KEYS = {"volume", "region", "budget", "out_dir", "extra"}


@dataclass
class VolumeCfg:
    url: str
    level: int = 0
    voxel_um: float = 2.4
    alt_url: str | None = None
    # local zarr cache of the region (see ``tsm cache``); reads prefer it over the URLs
    cache_root: str | None = None
    cache_margin: int = 64


@dataclass
class RegionCfg:
    start_zyx: tuple[int, int, int]
    size_zyx: tuple[int, int, int]

    @property
    def stop_zyx(self) -> tuple[int, int, int]:
        return tuple(a + b for a, b in zip(self.start_zyx, self.size_zyx))  # type: ignore[return-value]


@dataclass
class RunCfg:
    volume: VolumeCfg
    region: RegionCfg
    budget: Budget
    out_dir: str
    extra: dict[str, Any] = field(default_factory=dict)


def _triple(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a list of 3 ints, got {value!r}")
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{name} must contain ints, got {value!r}")
        if v < 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        out.append(int(v))
    return (out[0], out[1], out[2])


def _volume(d: Any) -> VolumeCfg:
    if not isinstance(d, dict):
        raise ValueError("volume must be an object")
    known = {"url", "level", "voxel_um", "alt_url", "cache_root", "cache_margin"}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(f"unknown volume keys: {unknown}")
    url = d.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("volume.url must be a non-empty string")
    level = d.get("level", 0)
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise ValueError("volume.level must be a non-negative int")
    voxel_um = d.get("voxel_um", 2.4)
    if isinstance(voxel_um, bool) or not isinstance(voxel_um, (int, float)) or voxel_um <= 0:
        raise ValueError("volume.voxel_um must be a positive number")
    alt = d.get("alt_url")
    if alt is not None and not isinstance(alt, str):
        raise ValueError("volume.alt_url must be a string or null")
    cache_root = d.get("cache_root")
    if cache_root is not None and (not isinstance(cache_root, str) or not cache_root):
        raise ValueError("volume.cache_root must be a non-empty string or null")
    margin = d.get("cache_margin", 64)
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("volume.cache_margin must be a non-negative int")
    return VolumeCfg(url=url, level=level, voxel_um=float(voxel_um), alt_url=alt,
                     cache_root=cache_root, cache_margin=int(margin))


def _region(d: Any) -> RegionCfg:
    if not isinstance(d, dict):
        raise ValueError("region must be an object")
    unknown = sorted(set(d) - {"start_zyx", "size_zyx"})
    if unknown:
        raise ValueError(f"unknown region keys: {unknown}")
    size = _triple(d.get("size_zyx"), "region.size_zyx")
    if min(size) <= 0:
        raise ValueError("region.size_zyx must be positive")
    return RegionCfg(start_zyx=_triple(d.get("start_zyx"), "region.start_zyx"), size_zyx=size)


def parse_config(raw: dict[str, Any]) -> RunCfg:
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    unknown = sorted(set(raw) - TOP_KEYS)
    if unknown:
        raise ValueError(f"unknown top-level config keys: {unknown}")
    out_dir = raw.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir:
        raise ValueError("out_dir must be a non-empty string")
    extra = raw.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("extra must be an object")
    return RunCfg(
        volume=_volume(raw.get("volume")),
        region=_region(raw.get("region")),
        budget=Budget.from_dict(raw.get("budget")),
        out_dir=out_dir,
        extra=extra,
    )


def load_config(path: str) -> RunCfg:
    with open(path, "r") as fh:
        raw = json.load(fh)
    return parse_config(raw)
