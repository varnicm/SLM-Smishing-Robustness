"""Unit tests for the per-observation pairwise gap: gap = sim(A,A') - sim(A,B).

The key property: a valid ZERO score must NOT be treated as missing. A truthiness
check (`if true and base:`) would break this; valid_number()/pairwise_gap() use an
explicit finite-number test instead.

Run: python tests/test_pairwise_gap.py   (prints 'ALL PAIRWISE-GAP TESTS PASSED')
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from analyze_explanation import valid_number, pairwise_gap  # noqa: E402


def test_normal():
    # the supervisor's example: 0.4 - 0.3 == 0.1 (not null)
    assert math.isclose(pairwise_gap(0.4, 0.3), 0.1, abs_tol=1e-9)


def test_zero_scores():
    # valid zeros must yield a valid zero gap, NOT None
    g = pairwise_gap(0.0, 0.0)
    assert g is not None and math.isclose(g, 0.0, abs_tol=1e-12)


def test_zero_on_one_side():
    assert math.isclose(pairwise_gap(0.0, 0.3), -0.3, abs_tol=1e-9)
    assert math.isclose(pairwise_gap(0.4, 0.0), 0.4, abs_tol=1e-9)


def test_missing_or_invalid_is_none():
    assert pairwise_gap(None, 0.3) is None
    assert pairwise_gap(0.4, None) is None
    assert pairwise_gap(float("nan"), 0.3) is None
    assert pairwise_gap(0.4, float("inf")) is None


def test_valid_number():
    assert valid_number(0.0) is True          # a real zero is valid
    assert valid_number(0.87) is True
    assert valid_number(None) is False
    assert valid_number(float("nan")) is False
    assert valid_number(float("inf")) is False
    assert valid_number(True) is False        # bool is not a score


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS {name}")
    print("\nALL PAIRWISE-GAP TESTS PASSED")
