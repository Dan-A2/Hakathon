"""Cluster bootstrap CIs and McNemar's test."""
from __future__ import annotations

import math

import numpy as np


def cluster_bootstrap(values, groups, n_boot: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    """Bootstrap the mean of per-case values by resampling GROUPS (twins and linked claims are correlated)."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    ok = ~np.isnan(values)
    values, groups = values[ok], groups[ok]
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "n_groups": 0}
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    sums = np.bincount(inv, weights=values, minlength=G)
    counts = np.bincount(inv, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1 - ci) / 2
    return {"mean": float(values.mean()), "lo": float(np.quantile(means, alpha)), "hi": float(np.quantile(means, 1 - alpha)),
            "n": int(len(values)), "n_groups": int(G)}


def paired_cluster_bootstrap(a, b, groups, n_boot: int = 10000, seed: int = 0) -> dict:
    """CI for mean(a - b) with group resampling (same groups for both policies)."""
    return cluster_bootstrap(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), groups, n_boot, seed)


def mcnemar(correct_a, correct_b) -> dict:
    """Exact two-sided McNemar test on paired per-case correctness (a vs b)."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n_ab = int(np.sum(a & ~b))     # a right, b wrong
    n_ba = int(np.sum(~a & b))     # a wrong, b right
    n = n_ab + n_ba
    if n == 0:
        p = 1.0
    else:
        k = min(n_ab, n_ba)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return {"a_right_b_wrong": n_ab, "a_wrong_b_right": n_ba, "discordant": n, "p_value": float(p)}


def fmt_ci(d: dict, pct: bool = False, digits: int = 3) -> str:
    if d is None or d.get("n", 0) == 0 or d["mean"] != d["mean"]:
        return "n/a"
    if pct:
        return f"{100 * d['mean']:.1f} [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}]"
    return f"{d['mean']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"
