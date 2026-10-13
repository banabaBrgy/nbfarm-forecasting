"""Ordinary least squares: one feature (calendar day index) → eggs."""

from __future__ import annotations


def fit_line(x: list[float], y: list[float]) -> tuple[float, float]:
    """Return (slope, intercept). If X has no variance, slope is 0 and intercept is mean(y)."""
    n = len(x)
    if n == 0:
        raise ValueError("need at least one point")
    if n == 1:
        return 0.0, float(y[0])

    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(a * b for a, b in zip(x, y))
    sum_xx = sum(a * a for a in x)
    denominator = n * sum_xx - sum_x * sum_x

    if denominator == 0:
        return 0.0, sum_y / n

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept


def predict_y(slope: float, intercept: float, x: float) -> int:
    return max(0, round(slope * x + intercept))
