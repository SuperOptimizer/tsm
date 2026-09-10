"""Memory guards. Every large numpy allocation in TSM goes through here."""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, fields
from typing import Any, Callable, Iterable, Sequence

import numpy as np

GB = 1 << 30
MB = 1 << 20


class BudgetError(MemoryError):
    """Raised before an allocation that would exceed the configured budget."""


@dataclass
class Budget:
    ram_bytes: int = 8 << 30
    array_bytes: int = 2 << 30
    vram_frac: float = 0.8
    min_avail_mb: int = 2048

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Budget":
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown budget keys: {unknown}")
        b = cls(**{k: d[k] for k in d})
        if b.ram_bytes <= 0 or b.array_bytes <= 0:
            raise ValueError("budget byte limits must be positive")
        if not 0.0 < b.vram_frac <= 1.0:
            raise ValueError("vram_frac must be in (0, 1]")
        return b


def nbytes(shape: Sequence[int], dtype: Any) -> int:
    n = int(np.dtype(dtype).itemsize)
    for s in shape:
        if int(s) < 0:
            raise ValueError(f"negative dimension in shape {tuple(shape)}")
        n *= int(s)
    return n


def check_alloc(shape: Sequence[int], dtype: Any, budget: Budget) -> int:
    n = nbytes(shape, dtype)
    if n > budget.array_bytes:
        raise BudgetError(
            f"array {tuple(shape)} {np.dtype(dtype).name} = {n / GB:.3f} GiB "
            f"exceeds array_bytes {budget.array_bytes / GB:.3f} GiB"
        )
    if n > budget.ram_bytes:
        raise BudgetError(
            f"array {tuple(shape)} {np.dtype(dtype).name} = {n / GB:.3f} GiB "
            f"exceeds ram_bytes {budget.ram_bytes / GB:.3f} GiB"
        )
    return n


def zeros(shape: Sequence[int], dtype: Any, budget: Budget) -> np.ndarray:
    check_alloc(shape, dtype, budget)
    return np.zeros(tuple(int(s) for s in shape), dtype=dtype)


def empty(shape: Sequence[int], dtype: Any, budget: Budget) -> np.ndarray:
    check_alloc(shape, dtype, budget)
    return np.empty(tuple(int(s) for s in shape), dtype=dtype)


def read_mem_avail_mb() -> int:
    """MemAvailable from /proc/meminfo, in MiB. -1 if unavailable."""
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def read_rss_bytes() -> int:
    """VmRSS of this process, in bytes. -1 if unavailable."""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


def peak_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def _die(msg: str) -> None:
    sys.stderr.write(f"[tsm] FATAL {msg}\n")
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(3)


class MemWatchdog:
    """Polls MemAvailable and own RSS; trips rather than letting the box swap."""

    def __init__(
        self,
        budget: Budget,
        interval: float = 2.0,
        avail_mb_fn: Callable[[], int] = read_mem_avail_mb,
        rss_fn: Callable[[], int] = read_rss_bytes,
        on_trip: Callable[[str], None] = _die,
    ) -> None:
        self.budget = budget
        self.interval = float(interval)
        self.avail_mb_fn = avail_mb_fn
        self.rss_fn = rss_fn
        self.on_trip = on_trip
        self.tripped = False
        self.reason = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def check_once(self) -> bool:
        avail = self.avail_mb_fn()
        rss = self.rss_fn()
        reason = ""
        if avail >= 0 and avail < self.budget.min_avail_mb:
            reason = f"MemAvailable {avail} MiB < min_avail_mb {self.budget.min_avail_mb}"
        elif rss >= 0 and rss > self.budget.ram_bytes:
            reason = f"RSS {rss / GB:.2f} GiB > ram_bytes {self.budget.ram_bytes / GB:.2f} GiB"
        if reason and not self.tripped:
            self.tripped = True
            self.reason = reason
            print(f"[tsm] memory watchdog tripped: {reason}", flush=True)
            self.on_trip(reason)
        return self.tripped

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            if self.check_once():
                return

    def start(self) -> "MemWatchdog":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="tsm-memwatch", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=self.interval + 1.0)
            self._thread = None

    def __enter__(self) -> "MemWatchdog":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def cuda_guard(budget: Budget) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    torch.cuda.set_per_process_memory_fraction(budget.vram_frac)
    return True


def free_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_table(rows: Iterable[Sequence[object]]) -> str:
    rows = [[str(c) for c in r] for r in rows]
    if not rows:
        return ""
    width = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(width[i]) for i, c in enumerate(r)).rstrip() for r in rows)


def estimate_and_assert(
    items: list[tuple[str, Sequence[int], Any]], budget: Budget
) -> int:
    rows: list[Sequence[object]] = [["name", "shape", "dtype", "bytes", "GiB"]]
    total = 0
    over: list[str] = []
    for name, shape, dtype in items:
        n = nbytes(shape, dtype)
        total += n
        if n > budget.array_bytes:
            over.append(f"{name} {n / GB:.3f} GiB > array_bytes {budget.array_bytes / GB:.3f} GiB")
        rows.append([name, tuple(int(s) for s in shape), np.dtype(dtype).name, n, f"{n / GB:.3f}"])
    rows.append(["TOTAL", "", "", total, f"{total / GB:.3f}"])
    print(format_table(rows), flush=True)
    if total > budget.ram_bytes:
        over.append(f"total {total / GB:.3f} GiB > ram_bytes {budget.ram_bytes / GB:.3f} GiB")
    if over:
        raise BudgetError("; ".join(over))
    return total
