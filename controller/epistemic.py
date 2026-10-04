"""Two epistemic controllers that decide in two stages: before the check (answer / check / abstain) and,
if they checked, after it (commit the checked verdict / keep the prior / abstain).

  * TwoStageController - REINFORCE replay over the cache with two softmax heads (W1: 3 x 6 pre-check
    signals, W2: 3 x 13 post-check signals).  Same trainer discipline as controller/train.py.
  * EUController - decision theory from calibrated credences.  Logistic models give P(correct) for
    each option; the action with the highest expected reward under (w, c) is taken, so "answer iff
    P(correct) > w/(1+w)" is the literal mechanism and the reported number is a credence.

    python -m controller.epistemic --cache-dir art/cache_llama3b --out-dir art/models/llama3b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import config as C
from common.io import save_json, sha256_files
from controller.calibrate import fit_logreg
from controller.policy import Standardizer, entropy, feature_matrix, softmax
from controller.train import adam_step, choose_median
from scorer.score import Scored, load_scored, reward

S1_ANSWER, S1_CHECK, S1_ABSTAIN = 0, 1, 2
S2_TO_DECISION = {0: C.CHECK_COMMIT, 1: C.CHECK_KEEP, 2: C.CHECK_ABSTAIN}


# ----------------------------------------------------------------------------- data
class EpiData:
    """Pre- and post-check feature matrices (standardised with train statistics) and reward tables."""

    def __init__(self, cases_dir: Path | str, cache_dir: Path | str, splits=("train", "val", "test_id")):
        self.cases_dir, self.cache_dir = Path(cases_dir), Path(cache_dir)
        self.pre_names, self.post_names = list(C.FEATURE_NAMES), list(C.POST_FEATURE_NAMES)
        self.scored = {s: load_scored(s, cases_dir, cache_dir) for s in splits if (Path(cache_dir) / f"{s}.jsonl").exists()}
        tr = self.scored["train"]
        self.std1 = Standardizer.fit(feature_matrix([x.x for x in tr], self.pre_names), self.pre_names)
        self.std2 = Standardizer.fit(feature_matrix([x.z for x in tr], self.post_names), self.post_names)
        self.X1 = {s: self.std1.transform(feature_matrix([x.x for x in v], self.pre_names)) for s, v in self.scored.items()}
        self.X2 = {s: self.std2.transform(feature_matrix([x.z for x in v], self.post_names)) for s, v in self.scored.items()}
        first = tr[0].raw
        self.prompt_hash, self.llm, self.K = first.get("prompt_hash", ""), first.get("llm", {}), int(first.get("K", C.DEFAULT_K))
        self.cache_hash = sha256_files([self.cache_dir / "train.jsonl", self.cache_dir / "val.jsonl"])

    def rewards(self, split: str, w: float, c: float) -> tuple[np.ndarray, np.ndarray]:
        """Ra[i] = reward of the grounded answer; R2[i, j] = reward of stage-2 decision j."""
        sc = self.scored[split]
        Ra = np.array([reward(x, C.D_ANSWER, w, c) for x in sc])
        R2 = np.array([[reward(x, d, w, c) for d in C.STAGE2_DECISIONS] for x in sc])
        return Ra, R2


# ----------------------------------------------------------------------------- REINFORCE, two stages
def argmax_two_stage(W1, W2, X1, X2) -> np.ndarray:
    a1 = np.argmax(X1 @ W1.T, axis=1)
    a2 = np.argmax(X2 @ W2.T, axis=1)
    out = np.where(a1 == S1_ANSWER, C.D_ANSWER, np.where(a1 == S1_ABSTAIN, C.ABSTAIN, 0))
    for j, d in S2_TO_DECISION.items():
        out = np.where((a1 == S1_CHECK) & (a2 == j), d, out)
    return out


def decisions_reward(dec: np.ndarray, Ra: np.ndarray, R2: np.ndarray) -> float:
    r = np.where(dec == C.D_ANSWER, Ra, 0.0)
    for j, d in S2_TO_DECISION.items():
        r = np.where(dec == d, R2[:, j], r)
    return float(r.mean())


def train_two_stage(X1, X2, Ra, R2, seed, epochs=300, lr=0.05, batch=32, beta_ema=0.95, ent0=0.01,
                    val=None, eval_every=10, patience=50) -> dict:
    rng = np.random.default_rng(seed)
    n = len(X1)
    W1, W2 = np.zeros((3, X1.shape[1])), np.zeros((3, X2.shape[1]))
    m1, v1, m2, v2 = (np.zeros_like(W1), np.zeros_like(W1), np.zeros_like(W2), np.zeros_like(W2))
    b, t = 0.0, 0
    X1v, X2v, Rav, R2v = val if val is not None else (X1, X2, Ra, R2)
    best, best_W, best_ep, last_improve, curve, ep = -np.inf, (W1.copy(), W2.copy()), 0, 0, [], 0
    for ep in range(epochs):
        ent = ent0 * max(0.0, 1 - ep / (0.5 * epochs))
        for idx in np.array_split(rng.permutation(n), max(1, n // batch)):
            P1 = softmax(X1[idx] @ W1.T)
            a1 = (P1.cumsum(1) > rng.random(len(idx))[:, None]).argmax(1)
            P2 = softmax(X2[idx] @ W2.T)
            a2 = (P2.cumsum(1) > rng.random(len(idx))[:, None]).argmax(1)
            r = np.where(a1 == S1_ANSWER, Ra[idx], 0.0)
            chk = a1 == S1_CHECK
            r = np.where(chk, R2[idx, a2], r)
            bs = np.empty(len(idx))
            for i, ri in enumerate(r):
                bs[i] = b
                b = beta_ema * b + (1 - beta_ema) * ri
            adv = r - bs
            G1 = -((adv[:, None] * (np.eye(3)[a1] - P1)).T @ X1[idx])
            G2 = -(((adv * chk)[:, None] * (np.eye(3)[a2] - P2)).T @ X2[idx])       # stage 2 only credits checked samples
            if ent > 0:
                H1, H2 = entropy(P1), entropy(P2)
                G1 += -ent * ((P1 * (-np.log(np.clip(P1, 1e-12, None)) - H1[:, None])).T @ X1[idx])
                G2 += -ent * (((P2 * (-np.log(np.clip(P2, 1e-12, None)) - H2[:, None])) * chk[:, None]).T @ X2[idx])
            t += 1
            W1 = adam_step(W1, G1 / len(idx), m1, v1, t, lr)
            W2 = adam_step(W2, G2 / len(idx), m2, v2, t, lr)
        if ep % eval_every == 0 or ep == epochs - 1:
            val_r = decisions_reward(argmax_two_stage(W1, W2, X1v, X2v), Rav, R2v)
            curve.append({"epoch": ep, "val_reward": val_r})
            if val_r > best + 1e-12:
                best, best_W, best_ep, last_improve = val_r, (W1.copy(), W2.copy()), ep, ep
            elif ep - last_improve >= patience:
                break
    return {"W1": best_W[0], "W2": best_W[1], "best_val": float(best), "best_epoch": best_ep, "curve": curve, "epochs_run": ep + 1}


class TwoStageController:
    def __init__(self, W1, W2, std1: Standardizer, std2: Standardizer, w, c, K, prompt_hash, llm, cache_hash, meta=None):
        self.W1, self.W2, self.std1, self.std2 = np.asarray(W1), np.asarray(W2), std1, std2
        self.w, self.c, self.K, self.prompt_hash, self.llm, self.cache_hash = float(w), float(c), int(K), prompt_hash, dict(llm), cache_hash
        self.meta = dict(meta or {})

    def _z(self, std, names, d):
        return std.transform(feature_matrix([d], names, fill={f: float(m) for f, m in zip(names, std.mu)}))[0]

    def decide(self, x: dict, z: dict) -> tuple[int, dict]:
        p1 = softmax(self.W1 @ self._z(self.std1, self.std1.feature_names, x))
        a1 = int(np.argmax(p1))
        info = {"stage1": {n: float(v) for n, v in zip(C.STAGE1_NAMES, p1)}}
        if a1 == S1_ANSWER:
            return C.D_ANSWER, info
        if a1 == S1_ABSTAIN:
            return C.ABSTAIN, info
        p2 = softmax(self.W2 @ self._z(self.std2, self.std2.feature_names, z))
        info["stage2"] = {n: float(v) for n, v in zip(C.STAGE2_NAMES, p2)}
        return S2_TO_DECISION[int(np.argmax(p2))], info

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"kind": "two_stage_reinforce", "pre_names": self.std1.feature_names, "post_names": self.std2.feature_names,
                "w": self.w, "c": self.c, "K": self.K, "prompt_hash": self.prompt_hash, "llm": self.llm, "cache_hash": self.cache_hash,
                **self.meta}
        np.savez(path, W1=self.W1, W2=self.W2, mu1=self.std1.mu, sd1=self.std1.sd, mu2=self.std2.mu, sd2=self.std2.sd, meta=json.dumps(meta))
        return path

    @classmethod
    def load(cls, path):
        d = np.load(Path(path), allow_pickle=False)
        m = json.loads(str(d["meta"]))
        extra = {k: v for k, v in m.items() if k not in {"kind", "pre_names", "post_names", "w", "c", "K", "prompt_hash", "llm", "cache_hash"}}
        return cls(d["W1"], d["W2"], Standardizer(m["pre_names"], d["mu1"], d["sd1"]), Standardizer(m["post_names"], d["mu2"], d["sd2"]),
                   m["w"], m["c"], m["K"], m["prompt_hash"], m["llm"], m["cache_hash"], extra)


# ----------------------------------------------------------------------------- decision theory from credences
def _ridge(X, y, lam=1.0):
    reg = lam * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def _sigmoid(v):
    return 1 / (1 + np.exp(-np.clip(v, -30, 30)))


class EUController:
    """Credences P_answer(x), P_commit(z), P_keep(z) fit on val; decisions maximise expected reward."""

    def __init__(self, beta_a, beta_cv, beta_kp, beta_check, std1: Standardizer, std2: Standardizer, w, c, K,
                 prompt_hash="", llm=None, cache_hash="", meta=None):
        self.beta_a, self.beta_cv, self.beta_kp, self.beta_check = (np.asarray(beta_a), np.asarray(beta_cv),
                                                                    np.asarray(beta_kp), np.asarray(beta_check))
        self.std1, self.std2 = std1, std2
        self.w, self.c, self.K = float(w), float(c), int(K)
        self.prompt_hash, self.llm, self.cache_hash, self.meta = prompt_hash, dict(llm or {}), cache_hash, dict(meta or {})

    # ---- fitting ------------------------------------------------------------------------------
    @classmethod
    def fit(cls, scored: list[Scored], X1: np.ndarray, X2: np.ndarray, std1, std2, w: float, c: float, K: int = C.DEFAULT_K,
            prompt_hash="", llm=None, cache_hash="", l2: float = 1e-2) -> "EUController":
        ya = np.array([x.dprov_correct for x in scored], float)
        ycv = np.array([x.dver_correct for x in scored], float)
        beta_a, beta_cv, beta_kp = fit_logreg(X1, ya, l2), fit_logreg(X2, ycv, l2), fit_logreg(X2, ya, l2)
        # value of checking = predicted ADVANTAGE of checking over answering: realised reward of the stage-2 rule
        # minus the realised reward of the grounded answer, regressed on the pre-check signals.  Fitting the
        # difference removes the shared "this case is hard" variance and leaves what checking adds.
        p_cv, p_kp = _sigmoid(X2 @ beta_cv), _sigmoid(X2 @ beta_kp)
        s2 = cls._stage2(p_cv, p_kp, w)
        advantage = np.array([reward(x, d, w, c) - reward(x, C.D_ANSWER, w, c) for x, d in zip(scored, s2)])
        beta_check = _ridge(X1, advantage)
        return cls(beta_a, beta_cv, beta_kp, beta_check, std1, std2, w, c, K, prompt_hash, llm, cache_hash,
                   meta={"n_fit": len(scored), "fit_on": "val"})

    @staticmethod
    def _stage2(p_cv, p_kp, w) -> np.ndarray:
        ev = np.stack([(1 + w) * p_cv - w, (1 + w) * p_kp - w, np.zeros_like(p_cv)], axis=1)   # tool cost is sunk here
        return np.array([C.STAGE2_DECISIONS[j] for j in np.argmax(ev, axis=1)])

    # ---- deciding -----------------------------------------------------------------------------
    def logits(self, X1: np.ndarray, X2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return X1 @ self.beta_a, X2 @ self.beta_cv, X2 @ self.beta_kp

    def decide_batch(self, X1: np.ndarray, X2: np.ndarray, w: float | None = None,
                     platt: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Decisions and credences. `platt={'a': (a, b), 'cv': (a, b), 'kp': (a, b)}` recalibrates each credence
        family as sigmoid(a * logit + b) (few-shot recalibration on a new domain); the in-domain
        value-of-checking model is kept."""
        w = self.w if w is None else w
        la, lcv, lkp = self.logits(X1, X2)
        if platt is not None:
            la = platt["a"][0] * la + platt["a"][1]
            lcv = platt["cv"][0] * lcv + platt["cv"][1]
            lkp = platt["kp"][0] * lkp + platt["kp"][1]
        p_a = _sigmoid(la)
        ev_answer = (1 + w) * p_a - w
        v_check = ev_answer + X1 @ self.beta_check            # expected value of answering plus the predicted gain from checking
        p_cv, p_kp = _sigmoid(lcv), _sigmoid(lkp)
        s2 = self._stage2(p_cv, p_kp, w)
        stage1 = np.argmax(np.stack([ev_answer, v_check, np.zeros_like(p_a)], axis=1), axis=1)
        dec = np.where(stage1 == S1_ANSWER, C.D_ANSWER, np.where(stage1 == S1_ABSTAIN, C.ABSTAIN, s2))
        cred = np.where(dec == C.D_ANSWER, p_a, np.where(dec == C.CHECK_COMMIT, p_cv, np.where(dec == C.CHECK_KEEP, p_kp, np.nan)))
        return dec, cred, {"p_answer": p_a, "ev_answer": ev_answer, "v_check": v_check, "p_commit": p_cv, "p_keep": p_kp}

    def decide(self, x: dict, z: dict) -> tuple[int, float | None, dict]:
        X1 = self.std1.transform(feature_matrix([x], self.std1.feature_names, fill={f: float(m) for f, m in zip(self.std1.feature_names, self.std1.mu)}))
        X2 = self.std2.transform(feature_matrix([z], self.std2.feature_names, fill={f: float(m) for f, m in zip(self.std2.feature_names, self.std2.mu)}))
        dec, cred, info = self.decide_batch(X1, X2)
        return int(dec[0]), (None if np.isnan(cred[0]) else float(cred[0])), {k: float(v[0]) for k, v in info.items()}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"kind": "expected_utility", "pre_names": self.std1.feature_names, "post_names": self.std2.feature_names,
                "w": self.w, "c": self.c, "K": self.K, "prompt_hash": self.prompt_hash, "llm": self.llm, "cache_hash": self.cache_hash, **self.meta}
        np.savez(path, beta_a=self.beta_a, beta_cv=self.beta_cv, beta_kp=self.beta_kp, beta_check=self.beta_check,
                 mu1=self.std1.mu, sd1=self.std1.sd, mu2=self.std2.mu, sd2=self.std2.sd, meta=json.dumps(meta))
        return path

    @classmethod
    def load(cls, path):
        d = np.load(Path(path), allow_pickle=False)
        m = json.loads(str(d["meta"]))
        extra = {k: v for k, v in m.items() if k not in {"kind", "pre_names", "post_names", "w", "c", "K", "prompt_hash", "llm", "cache_hash"}}
        return cls(d["beta_a"], d["beta_cv"], d["beta_kp"], d["beta_check"], Standardizer(m["pre_names"], d["mu1"], d["sd1"]),
                   Standardizer(m["post_names"], d["mu2"], d["sd2"]), m["w"], m["c"], m["K"], m["prompt_hash"], m["llm"], m["cache_hash"], extra)

    def describe(self) -> str:
        rows = [("P(answer right)", self.beta_a, self.std1.feature_names), ("gain from checking", self.beta_check, self.std1.feature_names),
                ("P(checked verdict right)", self.beta_cv, self.std2.feature_names), ("P(prior right | check)", self.beta_kp, self.std2.feature_names)]
        return "\n".join(f"  {name:26s} " + ", ".join(f"{f}={b:+.2f}" for f, b in zip(names, beta)) for name, beta, names in rows)


# ----------------------------------------------------------------------------- CLI
def fit_all(data: EpiData, w: float, c: float, seeds: list[int], out_dir: Path, **kw) -> tuple[TwoStageController, EUController]:
    Ra, R2 = data.rewards("train", w, c)
    Rav, R2v = data.rewards("val", w, c)
    results = []
    for s in seeds:
        t0 = time.time()
        res = train_two_stage(data.X1["train"], data.X2["train"], Ra, R2, s, val=(data.X1["val"], data.X2["val"], Rav, R2v), **kw)
        res["seed"], res["train_s"] = s, round(time.time() - t0, 2)
        results.append(res)
    k = choose_median(results)
    chosen = results[k]
    vals = [r["best_val"] for r in results]
    rl = TwoStageController(chosen["W1"], chosen["W2"], data.std1, data.std2, w, c, data.K, data.prompt_hash, data.llm, data.cache_hash,
                            meta={"seeds": seeds, "chosen_seed": chosen["seed"], "val_reward_per_seed": vals,
                                  "val_reward_mean": float(np.mean(vals)), "val_reward_std": float(np.std(vals)), "train_config": kw})
    rl.save(out_dir / "epistemic_rl.npz")
    eu = EUController.fit(data.scored["val"], data.X1["val"], data.X2["val"], data.std1, data.std2, w, c, data.K,
                          data.prompt_hash, data.llm, data.cache_hash)
    eu.save(out_dir / "epistemic_eu.npz")
    dec_eu, _, _ = eu.decide_batch(data.X1["val"], data.X2["val"])
    save_json(out_dir / "epistemic.train.json", {
        "rl": {"meta": rl.meta, "per_seed": [{k2: r[k2] for k2 in ("seed", "best_val", "best_epoch", "epochs_run", "train_s", "curve")} for r in results]},
        "eu": {"val_reward": decisions_reward(dec_eu, Rav, R2v), "val_mix": {n: float(np.mean(dec_eu == d)) for n, d in zip(C.DECISIONS, range(7)) if np.any(dec_eu == d)}},
        "val_baselines": {"grounded_answer": float(Rav.mean()), "always_check_commit": float(R2v[:, 0].mean()), "abstain": 0.0}})
    return rl, eu


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--cache-dir", default=str(C.CACHE_DIR))
    ap.add_argument("--out-dir", default=str(C.ART_DIR))
    ap.add_argument("--w", type=float, default=C.DEFAULT_W)
    ap.add_argument("--c", type=float, default=C.DEFAULT_C)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=50)
    args = ap.parse_args(argv)
    data = EpiData(args.cases_dir, args.cache_dir)
    rl, eu = fit_all(data, args.w, args.c, list(range(args.seeds)), Path(args.out_dir), epochs=args.epochs, lr=args.lr, ent0=args.ent, patience=args.patience)
    Rav, R2v = data.rewards("val", args.w, args.c)
    dec_eu, _, _ = eu.decide_batch(data.X1["val"], data.X2["val"])
    print(f"two-stage REINFORCE: val reward per seed {[round(v, 3) for v in rl.meta['val_reward_per_seed']]} -> shipped seed {rl.meta['chosen_seed']} (median)")
    print(f"expected-utility controller: val reward {decisions_reward(dec_eu, Rav, R2v):.3f}; grounded-answer baseline {Rav.mean():.3f}; always check+commit {R2v[:, 0].mean():.3f}")
    print(eu.describe())


if __name__ == "__main__":
    main()
