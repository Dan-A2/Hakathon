import numpy as np

from controller.calibrate import brier, ece, fit_calibrator, reliability_bins
from eval.stats import cluster_bootstrap, mcnemar


def test_cluster_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.5, 0.1, 400)
    groups = np.repeat(np.arange(100), 4)
    d = cluster_bootstrap(vals, groups, n_boot=2000)
    assert d["lo"] <= d["mean"] <= d["hi"] and d["n_groups"] == 100 and d["hi"] - d["lo"] < 0.1


def test_mcnemar_exact():
    a = [True] * 10 + [False] * 10
    b = [True] * 10 + [False] * 10
    assert mcnemar(a, b)["p_value"] == 1.0
    r = mcnemar([True] * 8 + [False] * 2, [False] * 8 + [True] * 2)
    assert r["a_right_b_wrong"] == 8 and r["a_wrong_b_right"] == 2 and 0.1 < r["p_value"] < 0.12


def test_calibrator_learns_separable_signal():
    rng = np.random.default_rng(0)
    xs, outs = [], []
    for _ in range(600):
        conf = rng.uniform(0, 1)
        x = {"bias": 1.0, "conf_mean": conf, "agree": 1.0}
        xs.append(x)
        outs.append((0, bool(rng.uniform() < conf)))       # P(correct) == conf
    names = ["bias", "conf_mean", "agree"]
    mu, sd = np.array([0, 0.5, 1.0]), np.array([1, 0.29, 1.0])
    cal = fit_calibrator(xs, outs, names, mu, sd)
    p = cal.predict(xs, [0] * len(xs))
    y = np.array([o[1] for o in outs], dtype=float)
    assert ece(p, y) < 0.08 and brier(p, y) < 0.22
    assert sum(b["n"] for b in reliability_bins(p, y)) == len(xs)
