"""Calibrated P(correct): logistic regression on [z-scored signals, one-hot action], fit on val only.

Kept separate from the controller.  Controller probabilities are "action preferences";
this is the uncertainty number the deployed agent reports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import config as C

ACTION_COLS = ["act_answer", "act_verify"]   # abstain delivers no verdict, so it has no P(correct)


def design(Xz: np.ndarray, actions: np.ndarray) -> np.ndarray:
    a = np.asarray(actions)
    return np.hstack([Xz, (a == C.ANSWER)[:, None].astype(float), (a == C.VERIFY)[:, None].astype(float)])


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> np.ndarray:
    """L2-regularised logistic regression by Newton's method (X already has a bias column)."""
    n, d = X.shape
    beta = np.zeros(d)
    reg = l2 * np.eye(d)
    reg[0, 0] = 0.0    # don't penalise the bias (column 0 is the 'bias' feature)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (p - y) + reg @ beta
        Wd = p * (1 - p)
        H = (X * Wd[:, None]).T @ X + reg + 1e-9 * np.eye(d)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


@dataclass
class Calibrator:
    beta: np.ndarray
    feature_names: list[str]
    mu: np.ndarray
    sd: np.ndarray

    def _z(self, xs: list[dict]) -> np.ndarray:
        from controller.policy import feature_matrix

        raw = feature_matrix(xs, self.feature_names, fill={f: float(m) for f, m in zip(self.feature_names, self.mu)})
        return (raw - self.mu) / self.sd

    def predict(self, xs: list[dict], actions) -> np.ndarray:
        Xd = design(self._z(xs), np.asarray(actions))
        return 1 / (1 + np.exp(-(Xd @ self.beta)))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, beta=self.beta, mu=self.mu, sd=self.sd,
                 meta=json.dumps({"feature_names": self.feature_names, "columns": self.feature_names + ACTION_COLS}))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Calibrator":
        d = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        return cls(d["beta"], meta["feature_names"], d["mu"], d["sd"])


def fit_calibrator(xs: list[dict], outcomes: list[tuple[int, bool]], feature_names: list[str],
                   mu: np.ndarray, sd: np.ndarray, l2: float = 1e-2) -> Calibrator:
    """xs[i] are the signals of a val case; outcomes[i] = (action, correct) for a committed action."""
    cal = Calibrator(np.zeros(len(feature_names) + len(ACTION_COLS)), list(feature_names), np.asarray(mu), np.asarray(sd))
    Xd = design(cal._z(xs), np.asarray([a for a, _ in outcomes]))
    y = np.asarray([1.0 if ok else 0.0 for _, ok in outcomes])
    cal.beta = fit_logreg(Xd, y, l2=l2)
    return cal


# ----------------------------------------------------------------------------- metrics
def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & ((probs < hi) if i < n_bins - 1 else (probs <= hi))
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "confidence": float(probs[m].mean()) if m.any() else None,
                    "accuracy": float(labels[m].mean()) if m.any() else None})
    return out


def ece(probs, labels, n_bins: int = 10) -> float:
    bins = reliability_bins(probs, labels, n_bins)
    n = sum(b["n"] for b in bins)
    return float(sum(b["n"] / n * abs(b["accuracy"] - b["confidence"]) for b in bins if b["n"])) if n else float("nan")


def brier(probs, labels) -> float:
    p, y = np.asarray(probs, dtype=float), np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")
