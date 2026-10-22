#!/usr/bin/env python3
# =============================================================================
#  stats.py  -  small, dependency-free statistics for aggregating repeats
# -----------------------------------------------------------------------------
#  A benchmark is run several times (independent REPEATS); this aggregates the
#  per-repeat means into mean / sample-std / 95% confidence interval. The CI
#  uses the Student-t critical value (correct for the small sample sizes typical
#  of repeats, e.g. 3-10), via a tiny embedded table so there is no scipy/numpy
#  dependency here. For n > 30 it falls back to the normal approximation (1.96).
#
#  Reporting mean +/- CI95 across independent repeats - not just the spread
#  within one run - is what lets a reviewer trust the numbers.
# =============================================================================
import math

# Two-sided 95% Student-t critical values by degrees of freedom (n-1).
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t95(df):
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    return 1.96  # normal approximation for large df


def summarize(values):
    """Return mean, sample std, n, and 95% CI half-width for a list of numbers.

    half_width is such that the CI is [mean - half_width, mean + half_width].
    With a single value, std and CI are None (no spread to estimate).
    """
    n = len(values)
    if n == 0:
        return {
            "mean": None,
            "std": None,
            "n": 0,
            "ci95_half": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    mean = sum(values) / n
    if n == 1:
        return {
            "mean": round(mean, 6),
            "std": None,
            "n": 1,
            "ci95_half": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    var = sum((x - mean) ** 2 for x in values) / (n - 1)  # sample variance
    std = math.sqrt(var)
    half = _t95(n - 1) * std / math.sqrt(n)
    return {
        "mean": round(mean, 6),
        "std": round(std, 6),
        "n": n,
        "ci95_half": round(half, 6),
        "ci95_low": round(mean - half, 6),
        "ci95_high": round(mean + half, 6),
    }
