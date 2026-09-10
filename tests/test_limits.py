import numpy as np
import pytest

from tsm.limits import (
    Budget,
    BudgetError,
    MemWatchdog,
    check_alloc,
    estimate_and_assert,
    format_table,
    nbytes,
    peak_rss_mb,
    zeros,
)


def test_nbytes_and_check_alloc_ok():
    b = Budget(ram_bytes=1 << 20, array_bytes=1 << 20)
    assert nbytes((10, 10, 10), np.uint8) == 1000
    assert check_alloc((10, 10, 10), np.uint16, b) == 2000
    a = zeros((4, 4, 4), np.uint8, b)
    assert a.shape == (4, 4, 4) and a.dtype == np.uint8


def test_check_alloc_raises_above_array_bytes():
    b = Budget(ram_bytes=1 << 30, array_bytes=1000)
    with pytest.raises(BudgetError):
        check_alloc((100, 100), np.uint8, b)
    with pytest.raises(BudgetError):
        zeros((100, 100), np.uint8, b)


def test_budget_from_dict_rejects_unknown():
    assert Budget.from_dict({"ram_bytes": 5}).ram_bytes == 5
    with pytest.raises(ValueError):
        Budget.from_dict({"nope": 1})


def test_estimate_and_assert(capsys):
    b = Budget(ram_bytes=1 << 20, array_bytes=1 << 20)
    total = estimate_and_assert([("a", (10, 10, 10), np.uint8), ("b", (2, 2), np.float32)], b)
    assert total == 1016
    out = capsys.readouterr().out
    assert "TOTAL" in out and "uint8" in out
    with pytest.raises(BudgetError):
        estimate_and_assert([("big", (2000, 2000), np.uint8)], b)


def test_format_table():
    t = format_table([["a", "bb"], ["ccc", "d"]])
    assert t.splitlines()[0].startswith("a")


def test_watchdog_trips_on_low_avail():
    tripped = []
    w = MemWatchdog(
        Budget(min_avail_mb=1024),
        avail_mb_fn=lambda: 10,
        rss_fn=lambda: 0,
        on_trip=tripped.append,
    )
    assert w.check_once() is True
    assert w.tripped and "MemAvailable" in tripped[0]


def test_watchdog_trips_on_rss():
    tripped = []
    w = MemWatchdog(
        Budget(ram_bytes=1 << 20, min_avail_mb=0),
        avail_mb_fn=lambda: 100000,
        rss_fn=lambda: 1 << 30,
        on_trip=tripped.append,
    )
    assert w.check_once() is True
    assert "RSS" in tripped[0]


def test_watchdog_quiet_and_context_manager():
    tripped = []
    w = MemWatchdog(
        Budget(ram_bytes=1 << 40, min_avail_mb=0),
        interval=0.01,
        avail_mb_fn=lambda: 100000,
        rss_fn=lambda: 1 << 20,
        on_trip=tripped.append,
    )
    with w:
        assert w.check_once() is False
    assert not tripped and not w.tripped


def test_peak_rss_positive():
    assert peak_rss_mb() > 0
