"""softmax(W x) policy with z-scored features, saved as controller.npz with provenance."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import config as C


class ControllerMismatch(RuntimeError):
    """The controller was trained against a different prompt or model than the one in use."""


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)


@dataclass
class Standardizer:
    feature_names: list[str]
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, feature_names: list[str]) -> "Standardizer":
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        if "bias" in feature_names:           # the bias column is left untouched
            i = feature_names.index("bias")
            mu[i], sd[i] = 0.0, 1.0
        return cls(list(feature_names), mu, sd)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


def feature_matrix(xs: list[dict], feature_names: list[str], fill: dict | None = None) -> np.ndarray:
    """Stack feature dicts into a matrix; missing/None values are filled with `fill[name]` (default 0.5)."""
    fill = fill or {}
    rows = []
    for x in xs:
        row = []
        for f in feature_names:
            v = x.get(f)
            if v is None:
                v = fill.get(f, 0.5)
            row.append(float(v))
        rows.append(row)
    return np.asarray(rows, dtype=float).reshape(len(xs), len(feature_names))


class Controller:
    def __init__(self, W: np.ndarray, std: Standardizer, w: float, c: float, K: int, prompt_hash: str,
                 llm: dict, cache_hash: str, meta: dict | None = None, action_names=tuple(C.ACTIONS)):
        self.W = np.asarray(W, dtype=float)
        self.std = std
        self.feature_names = list(std.feature_names)
        self.action_names = list(action_names)
        self.w, self.c, self.K = float(w), float(c), int(K)
        self.prompt_hash = prompt_hash
        self.llm = dict(llm)
        self.cache_hash = cache_hash
        self.meta = dict(meta or {})
        assert self.W.shape == (len(self.action_names), len(self.feature_names)), self.W.shape

    # ---- inference -------------------------------------------------------------------------
    def featurize(self, x: dict) -> np.ndarray:
        raw = feature_matrix([x], self.feature_names, fill={f: float(m) for f, m in zip(self.feature_names, self.std.mu)})
        return self.std.transform(raw)[0]

    def probs(self, x: dict) -> np.ndarray:
        return softmax(self.W @ self.featurize(x))

    def act(self, x: dict) -> tuple[int, np.ndarray]:
        p = self.probs(x)
        return int(np.argmax(p)), p

    # ---- persistence -------------------------------------------------------------------
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "feature_names": self.feature_names, "action_names": self.action_names,
            "w": self.w, "c": self.c, "K": self.K, "prompt_hash": self.prompt_hash, "llm": self.llm,
            "cache_hash": self.cache_hash, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **self.meta,
        }
        np.savez(path, W=self.W, mu=self.std.mu, sd=self.std.sd, meta=json.dumps(meta))
        return path

    @classmethod
    def load(cls, path: Path | str, check_prompt: bool = True) -> "Controller":
        from agent.prompts import prompt_hash as current_prompt_hash

        d = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        std = Standardizer(meta["feature_names"], d["mu"], d["sd"])
        extra = {k: v for k, v in meta.items() if k not in
                 {"feature_names", "action_names", "w", "c", "K", "prompt_hash", "llm", "cache_hash"}}
        ctrl = cls(d["W"], std, meta["w"], meta["c"], meta["K"], meta["prompt_hash"], meta["llm"], meta["cache_hash"],
                   extra, meta["action_names"])
        if check_prompt and meta["prompt_hash"] != current_prompt_hash():
            raise ControllerMismatch(
                f"prompt hash {current_prompt_hash()} differs from the controller's {meta['prompt_hash']}; "
                "the prompts changed after the cache was built (set KWTC_SKIP_HASH_CHECK=1 to override)")
        return ctrl

    def check_llm(self, identity: dict) -> None:
        """Refuse to run on a different frozen model than the one the cache was built with."""
        want = (self.llm.get("model"), self.llm.get("revision"))
        have = (identity.get("model"), identity.get("revision"))
        if want != have:
            raise ControllerMismatch(f"controller trained with {want}, but the LLM in use is {have}")

    def describe(self) -> str:
        lines = [f"controller: {len(self.action_names)} x {len(self.feature_names)}  (w={self.w}, c={self.c}, K={self.K})",
                 f"  llm={self.llm}  prompt_hash={self.prompt_hash}  cache_hash={self.cache_hash[:12]}",
                 "  W (rows=actions, cols=features):", "    " + " ".join(f"{f:>11s}" for f in self.feature_names)]
        for a, row in zip(self.action_names, self.W):
            lines.append(f"    {a:>8s} " + " ".join(f"{v:11.3f}" for v in row))
        return "\n".join(lines)
