"""Stage 2: REINFORCE replay over the counterfactual cache (pure NumPy).

    python -m controller.train --w 1 --c 0.05 --seeds 5 --out art/controller.npz
    python -m controller.train --sweep --sweep-out art/sweep.jsonl

Each training step uses only the reward of the sampled action (an honest bandit);
the cache is a simulator of the frozen agent.  Model selection: every 10 epochs the
argmax policy is scored on val (mean reward); the best W per seed is kept; the seed
with the MEDIAN val reward is shipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import config as C
from common.io import save_json, sha256_files
from controller.policy import Controller, Standardizer, entropy, feature_matrix, softmax
from scorer.score import Scored, load_scored, outcome, reward


def feature_names_for(use_tok_prob: bool) -> list[str]:
    return list(C.FEATURE_NAMES) + ([C.TOK_PROB_FEATURE] if use_tok_prob else [])


def reward_table(scored: list[Scored], w: float, c: float) -> np.ndarray:
    return np.asarray([[reward(sc, a, w, c) for a in range(3)] for sc in scored], dtype=float)


def adam_step(W, G, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    m[:] = b1 * m + (1 - b1) * G
    v[:] = b2 * v + (1 - b2) * G * G
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    return W - lr * mhat / (np.sqrt(vhat) + eps)


def argmax_actions(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.argmax(X @ W.T, axis=1)


def argmax_reward(W: np.ndarray, X: np.ndarray, R: np.ndarray) -> float:
    a = argmax_actions(W, X)
    return float(R[np.arange(len(X)), a].mean())


def action_mix(W: np.ndarray, X: np.ndarray) -> list[float]:
    a = argmax_actions(W, X)
    return [float((a == k).mean()) for k in range(3)]


def train(X: np.ndarray, R: np.ndarray, seed: int, epochs: int = 300, lr: float = 0.05, batch: int = 32,
          beta_ema: float = 0.95, ent0: float = 0.01, X_val: np.ndarray | None = None, R_val: np.ndarray | None = None,
          eval_every: int = 10, patience: int = 50) -> dict:
    """REINFORCE with a running (EMA) baseline, entropy bonus decaying to 0 by half-way, Adam."""
    rng = np.random.default_rng(seed)
    n, F = X.shape
    W = np.zeros((3, F))                     # uniform policy at start
    m, v = np.zeros_like(W), np.zeros_like(W)
    b, t = 0.0, 0
    Xv, Rv = (X_val, R_val) if X_val is not None else (X, R)
    best_val, best_W, best_ep = -np.inf, W.copy(), 0
    last_improve = 0
    curve = []
    ep = 0
    for ep in range(epochs):
        ent = ent0 * max(0.0, 1 - ep / (0.5 * epochs))
        for idx in np.array_split(rng.permutation(n), max(1, n // batch)):
            Xb = X[idx]
            P = softmax(Xb @ W.T)
            a = (P.cumsum(axis=1) > rng.random(len(idx))[:, None]).argmax(axis=1)    # sample a ~ pi
            r = R[idx, a]                                                            # reward of the sampled action only
            bs = np.empty(len(idx))
            for i, ri in enumerate(r):          # baseline from previous rewards, updated per sample
                bs[i] = b
                b = beta_ema * b + (1 - beta_ema) * ri
            adv = r - bs
            onehot = np.eye(3)[a]
            G = -((adv[:, None] * (onehot - P)).T @ Xb)                              # grad of -(r-b) log pi(a|x)
            if ent > 0:
                H = entropy(P)
                G += -ent * ((P * (-np.log(np.clip(P, 1e-12, None)) - H[:, None])).T @ Xb)   # grad of -H
            G /= len(idx)
            t += 1
            W = adam_step(W, G, m, v, t, lr)
        if ep % eval_every == 0 or ep == epochs - 1:
            val_r = argmax_reward(W, Xv, Rv)
            curve.append({"epoch": ep, "val_reward": val_r, "train_reward": argmax_reward(W, X, R),
                          "val_mix": action_mix(W, Xv), "entropy_bonus": ent})
            if val_r > best_val + 1e-12:
                best_val, best_W, best_ep, last_improve = val_r, W.copy(), ep, ep
            elif ep - last_improve >= patience:
                break
    return {"W": best_W, "best_val": float(best_val), "best_epoch": best_ep, "curve": curve,
            "epochs_run": ep + 1, "final_W": W}


def choose_median(results: list[dict]) -> int:
    order = sorted(range(len(results)), key=lambda i: results[i]["best_val"])
    return order[len(order) // 2]


def test_metrics(W: np.ndarray, X: np.ndarray, scored: list[Scored], w: float, c: float) -> dict:
    a = argmax_actions(W, X)
    outs = [outcome(sc, int(k)) for sc, k in zip(scored, a)]
    R = reward_table(scored, w, c)
    mix = [float((a == k).mean()) for k in range(3)]
    return {"reward": float(R[np.arange(len(a)), a].mean()),
            "accuracy": float(np.mean([o["correct"] for o in outs])),
            "coverage": float(np.mean([o["committed"] for o in outs])),
            "tool_calls_per_case": float(np.mean([o["tool_calls"] for o in outs])),
            "mix": mix, "majority_action": C.ACTIONS[int(np.argmax(mix))]}


class Data:
    """Loaded, standardised splits (standardisation uses train statistics only)."""

    def __init__(self, cases_dir: Path, cache_dir: Path, use_tok_prob: bool = False, splits=("train", "val", "test_id")):
        self.cases_dir, self.cache_dir = Path(cases_dir), Path(cache_dir)
        self.feature_names = feature_names_for(use_tok_prob)
        self.scored = {s: load_scored(s, cases_dir, cache_dir) for s in splits if (Path(cache_dir) / f"{s}.jsonl").exists()}
        if "train" not in self.scored or "val" not in self.scored:
            raise FileNotFoundError("train and val caches are required for training")
        if use_tok_prob and all(sc.x.get(C.TOK_PROB_FEATURE) is None for sc in self.scored["train"]):
            raise ValueError("--use-tok-prob requested but the train cache has no tok_prob (vLLM only)")
        Xraw = feature_matrix([sc.x for sc in self.scored["train"]], self.feature_names)
        self.std = Standardizer.fit(Xraw, self.feature_names)
        fill = {f: float(mu) for f, mu in zip(self.feature_names, self.std.mu)}
        self.X = {s: self.std.transform(feature_matrix([sc.x for sc in scs], self.feature_names, fill))
                  for s, scs in self.scored.items()}
        first = self.scored["train"][0].raw
        self.prompt_hash = first.get("prompt_hash", "")
        self.llm = first.get("llm", {})
        self.K = int(first.get("K", C.DEFAULT_K))
        self.cache_hash = sha256_files([self.cache_dir / "train.jsonl", self.cache_dir / "val.jsonl"])

    def R(self, split: str, w: float, c: float) -> np.ndarray:
        return reward_table(self.scored[split], w, c)


def train_seeds(data: Data, w: float, c: float, seeds: list[int], **kw) -> list[dict]:
    R, Rv = data.R("train", w, c), data.R("val", w, c)
    out = []
    for s in seeds:
        t0 = time.time()
        res = train(data.X["train"], R, s, X_val=data.X["val"], R_val=Rv, **kw)
        res["seed"], res["train_s"] = s, round(time.time() - t0, 2)
        res["val_mix"] = action_mix(res["W"], data.X["val"])
        out.append(res)
    return out


def fit_and_save(data: Data, w: float, c: float, seeds: list[int], out: Path, **kw) -> Controller:
    results = train_seeds(data, w, c, seeds, **kw)
    k = choose_median(results)
    chosen = results[k]
    Rv = data.R("val", w, c)
    baselines = {"always_answer": float(Rv[:, 0].mean()), "always_verify": float(Rv[:, 1].mean()), "always_abstain": 0.0}
    vals = [r["best_val"] for r in results]
    meta = {
        "seeds": seeds, "chosen_seed": chosen["seed"], "chosen_rule": "median val reward across seeds",
        "val_reward_per_seed": vals, "val_reward_mean": float(np.mean(vals)), "val_reward_std": float(np.std(vals)),
        "val_baselines": baselines, "best_epoch": chosen["best_epoch"], "epochs_run": chosen["epochs_run"],
        "train_config": kw, "n_train": len(data.scored["train"]), "n_val": len(data.scored["val"]),
    }
    ctrl = Controller(chosen["W"], data.std, w, c, data.K, data.prompt_hash, data.llm, data.cache_hash, meta)
    ctrl.save(out)
    save_json(Path(out).with_suffix(".train.json"), {
        "meta": meta, "per_seed": [{"seed": r["seed"], "best_val": r["best_val"], "best_epoch": r["best_epoch"],
                                    "epochs_run": r["epochs_run"], "train_s": r["train_s"], "val_mix": r["val_mix"],
                                    "curve": r["curve"]} for r in results]})
    return ctrl


def train_row(data: Data, w: float, c: float, seed: int, **kw) -> dict:
    """One sweep cell: train on train, select on val, report test_id metrics."""
    res = train_seeds(data, w, c, [seed], **kw)[0]
    row = {"w": w, "c": c, "seed": seed, "val_reward": res["best_val"], "best_epoch": res["best_epoch"],
           "val_mix": res["val_mix"], "train_s": res["train_s"]}
    if "test_id" in data.scored:
        row["test"] = test_metrics(res["W"], data.X["test_id"], data.scored["test_id"], w, c)
    row["W"] = res["W"].tolist()
    return row


def train_one(w: float, c: float, seed: int, cases_dir=C.CASES_DIR, cache_dir=C.CACHE_DIR, use_tok_prob=False, **kw) -> dict:
    """Self-contained sweep cell (loads the caches itself) - used by the Modal starmap sweep."""
    return train_row(Data(Path(cases_dir), Path(cache_dir), use_tok_prob), w, c, seed, **kw)


def run_sweep(data: Data, ws, cs, seeds, out_path: Path, **kw) -> list[dict]:
    rows = []
    t0 = time.time()
    total = len(ws) * len(cs) * len(seeds)
    with Path(out_path).open("w") as f:
        for w in ws:
            for c in cs:
                for s in seeds:
                    row = train_row(data, w, c, s, **kw)
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    print(f"[sweep] {len(rows)}/{total} w={w} c={c} seed={s} val={row['val_reward']:.3f} "
                          f"test={row.get('test', {}).get('reward', float('nan')):.3f} mix={row['val_mix']} "
                          f"({time.time() - t0:.0f}s)", file=sys.stderr)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w", type=float, default=C.DEFAULT_W)
    ap.add_argument("--c", type=float, default=C.DEFAULT_C)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=str(C.ART_DIR / "controller.npz"))
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--cache-dir", default=str(C.CACHE_DIR))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--use-tok-prob", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="run the w x c x seeds grid instead of fitting one controller")
    ap.add_argument("--sweep-w", default="0.5,1,2,4")
    ap.add_argument("--sweep-c", default="0,0.025,0.05,0.1,0.2")
    ap.add_argument("--sweep-out", default=str(C.ART_DIR / "sweep.jsonl"))
    args = ap.parse_args(argv)

    kw = dict(epochs=args.epochs, lr=args.lr, batch=args.batch, ent0=args.ent, patience=args.patience)
    data = Data(Path(args.cases_dir), Path(args.cache_dir), args.use_tok_prob)
    seeds = list(range(args.seeds))
    if args.sweep:
        ws = [float(x) for x in args.sweep_w.split(",")]
        cs = [float(x) for x in args.sweep_c.split(",")]
        run_sweep(data, ws, cs, seeds, Path(args.sweep_out), **kw)
        print(f"[sweep] wrote {args.sweep_out}")
        return
    t0 = time.time()
    ctrl = fit_and_save(data, args.w, args.c, seeds, Path(args.out), **kw)
    m = ctrl.meta
    print(ctrl.describe())
    print(f"val reward per seed: {[round(v, 3) for v in m['val_reward_per_seed']]}  -> shipped seed {m['chosen_seed']} "
          f"(median). mean {m['val_reward_mean']:.3f} +/- {m['val_reward_std']:.3f}; "
          f"baselines {json.dumps({k: round(v, 3) for k, v in m['val_baselines'].items()})}; {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
