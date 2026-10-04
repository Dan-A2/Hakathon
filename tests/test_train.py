import numpy as np

from controller.train import argmax_actions, choose_median, train


def test_reinforce_learns_predictable_policy():
    rng = np.random.default_rng(1)
    n = 400
    f = rng.integers(0, 2, n).astype(float)
    X = np.stack([np.ones(n), (f - f.mean()) / f.std()], axis=1)
    R = np.stack([np.where(f == 1, 1.0, -1.0), np.full(n, -1.1), np.zeros(n)], axis=1)
    res = train(X, R, seed=0, epochs=80, patience=1000)
    a = argmax_actions(res["W"], X)
    assert (a[f == 1] == 0).mean() > 0.95 and (a[f == 0] == 2).mean() > 0.95
    assert res["best_val"] > 0.4


def test_choose_median_picks_middle_seed():
    res = [{"best_val": 0.3}, {"best_val": 0.9}, {"best_val": 0.5}]
    assert choose_median(res) == 2
