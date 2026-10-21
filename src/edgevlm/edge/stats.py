"""Small, dependency-free statistics used by the on-device evaluation (Wilson interval, exact McNemar test)."""

from __future__ import annotations

from math import comb, sqrt


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion, as fractions."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def mcnemar_exact(only_a_correct: int, only_b_correct: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts (binomial test with p = 0.5)."""
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    k = min(only_a_correct, only_b_correct)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_counts(correct_a, correct_b) -> dict[str, int]:
    """Both-correct / only-A / only-B / both-wrong counts from two boolean sequences."""
    a, b = list(correct_a), list(correct_b)
    if len(a) != len(b):
        raise ValueError("outcome sequences must have the same length")
    return {
        "both_correct": sum(x and y for x, y in zip(a, b)),
        "only_a_correct": sum(x and not y for x, y in zip(a, b)),
        "only_b_correct": sum(y and not x for x, y in zip(a, b)),
        "both_wrong": sum(not x and not y for x, y in zip(a, b)),
    }
