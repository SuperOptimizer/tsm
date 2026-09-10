"""The one shared average-precision implementation (R15): grouped thresholds, documented
degenerate behaviour, and the same numbers wherever it is re-exported."""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))

from tsm.equivariance import average_precision as ap  # noqa: E402
from tsm.train import average_precision as ap_train  # noqa: E402


def test_tied_scores_do_not_depend_on_voxel_order():
    """Four identical scores, two positives: one threshold -> precision 0.5, recall 1 -> AP 0.5."""
    s = np.full(4, 0.5)
    for labels in set(itertools.permutations([1, 1, 0, 0])):
        y = np.array(labels, bool)
        assert ap(s, y) == pytest.approx(0.5), labels
        assert ap_train(s, y) == pytest.approx(0.5), labels
    # the review's two cases explicitly (they used to give 1.0 and 0.41666...)
    assert ap(np.ones(4), np.array([1, 1, 0, 0], bool)) == pytest.approx(0.5)
    assert ap(np.ones(4), np.array([0, 0, 1, 1], bool)) == pytest.approx(0.5)


def test_partial_ties_are_one_threshold_per_distinct_score():
    # scores 1,1,0,0 with labels 1,0,1,0: two thresholds -> (0.5*0.5) + (0.5*0.5) = 0.5
    s = np.array([1.0, 1.0, 0.0, 0.0])
    y = np.array([1, 0, 1, 0], bool)
    assert ap(s, y) == pytest.approx(0.5)
    assert ap(s, y) == ap(s[::-1], y[::-1])


def test_perfect_and_reversed_rankings():
    s = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    y = np.array([1, 1, 0, 0, 0], bool)
    assert ap(s, y) == pytest.approx(1.0)
    assert ap(-s, y) == pytest.approx((1 / 4 + 2 / 5) / 2)  # worst ranking: the positives rank last
    # unique scores: unchanged from the ungrouped formula
    y2 = np.array([1, 0, 1, 0, 0], bool)
    assert ap(s, y2) == pytest.approx((1.0 + 2 / 3) / 2)
    assert ap_train(s, y2) == pytest.approx((1.0 + 2 / 3) / 2)


def test_degenerate_inputs_return_none_or_nan_and_never_raise():
    s = np.array([0.1, 0.9])
    assert ap(s, np.zeros(2, bool)) is None          # no positive
    assert ap(s, np.ones(2, bool)) is None           # no negative
    assert ap(np.zeros(0), np.zeros(0, bool)) is None  # empty
    assert np.isnan(ap_train(s, np.zeros(2, bool)))
    assert np.isnan(ap_train(np.zeros(0), np.zeros(0, bool)))
    with pytest.raises(ValueError):
        ap(np.zeros(3), np.zeros(2, bool))


def test_sklearn_agreement_including_ties():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for q in (4, 16, 256):
        s = np.round(rng.random(500) * q) / q      # heavily quantized -> many ties
        y = rng.random(500) < 0.3
        assert ap(s, y) == pytest.approx(sk.average_precision_score(y, s))


def test_every_entry_point_shares_the_implementation():
    eval_region = pytest.importorskip("eval_region")
    s = np.full(6, 0.25)
    y = np.array([1, 0, 1, 0, 0, 0], bool)
    assert eval_region.average_precision is ap
    r = eval_region.Reservoir(cap=100, seed=0)
    r.add(s, y)
    assert r.auprc() == pytest.approx(ap(s, y)) == pytest.approx(1 / 3)
