import numpy as np
import pytest

from controller.policy import Controller, ControllerMismatch, Standardizer, feature_matrix, softmax


def _ctrl(prompt_hash):
    names = ["bias", "conf_mean", "agree"]
    std = Standardizer.fit(np.array([[1, 0.5, 1.0], [1, 0.9, 0.0], [1, 0.1, 1.0]]), names)
    W = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, -2.0], [1.0, -2.0, 0.0]])
    return Controller(W, std, 1.0, 0.05, 4, prompt_hash, {"backend": "mock", "model": "m", "revision": "r"}, "cachehash")


def test_softmax_and_standardizer():
    p = softmax(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(p, 1 / 3)
    std = Standardizer.fit(np.array([[1, 2.0], [1, 4.0]]), ["bias", "f"])
    assert std.mu[0] == 0 and std.sd[0] == 1 and np.allclose(std.transform(np.array([[1, 3.0]])), [[1, 0]])


def test_feature_matrix_fills_missing():
    X = feature_matrix([{"bias": 1, "a": None}], ["bias", "a"], fill={"a": 0.25})
    assert X.tolist() == [[1.0, 0.25]]


def test_save_load_roundtrip_and_hash_guard(tmp_path, monkeypatch):
    import agent.prompts as P

    ctrl = _ctrl(P.prompt_hash())
    path = ctrl.save(tmp_path / "c.npz")
    back = Controller.load(path)
    assert np.allclose(back.W, ctrl.W) and back.feature_names == ctrl.feature_names and back.w == 1.0
    a, p = back.act({"bias": 1, "conf_mean": 0.95, "agree": 1})
    assert a == 0 and p.shape == (3,)
    monkeypatch.setattr(P, "prompt_hash", lambda: "deadbeef")
    with pytest.raises(ControllerMismatch):
        Controller.load(path)
    Controller.load(path, check_prompt=False)
    with pytest.raises(ControllerMismatch):
        back.check_llm({"model": "other", "revision": "r"})
