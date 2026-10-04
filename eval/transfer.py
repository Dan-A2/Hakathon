"""Cross-domain epistemics: does the agent know what it does not know in a domain it was never calibrated on?

    python -m eval.transfer [--models llama3b,llama8b,gemma26b] [--splits test_ood,test_climate,test_vitc]

For every model and every out-of-domain split with a cache, four questions are answered from the cache
(no model calls):

  zero-shot     The in-domain credence-based agent used as is (credences fit on SciFact validation).
  few-shot      Platt recalibration of its credences on k labelled cases from the new domain, held-out
                evaluation on the rest, repeated over random draws.  "Few-Shot Recalibration of Language
                Models" (Li et al., 2024) showed that calibration that looks fine in aggregate hides
                slice-specific miscalibration which a handful of examples can repair; the curve over k says
                how much labelled data the new domain needs before the agent's probabilities mean something.
  risk control  A distribution-free abstention threshold (Learn-then-Test style conformal risk control): on
                the k calibration cases, the largest coverage whose upper confidence bound on the error rate
                among delivered verdicts stays below a target alpha, then the realised error rate on held-out
                cases.  This is the mechanism a lab would use to state "no more than alpha of the verdicts
                this agent delivers are wrong".
  novelty       Mahalanobis distance of a case's pre-check signals to the in-domain training distribution:
                can the agent tell that it has left the range where its calibration holds, and does
                miscalibration concentrate in novel cases?
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from common import config as C
from common import models as M
from common.io import save_json
from controller.calibrate import ece
from controller.epistemic import EUController
from controller.policy import feature_matrix
from eval.report import SPLIT_NAMES
from eval.stats import cluster_bootstrap, fmt_ci
from scorer.score import Scored, load_scored, outcome, reward

KS = (0, 25, 50, 100, 150)
REPEATS = 20
ALPHAS = (0.2, 0.3, 0.4)
DELTA = 0.1
N_MIN = 20          # smallest calibration prefix a risk bound may be computed on
LOGIT_CLIP = 8.0


# ----------------------------------------------------------------------------- recalibration + risk control
def _logloss(a, b, logits, y, l2):
    z = np.clip(a * logits + b, -30, 30)
    return float(np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)) + 0.5 * l2 * ((a - 1) ** 2 + b ** 2))


def platt_fit(logits: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 100) -> tuple[float, float]:
    """Monotone two-parameter recalibration p' = sigmoid(a * logit + b), a >= 0.

    Damped Newton with backtracking on the log loss (plain Newton diverges when some logits are extreme
    and the probabilities saturate), logits clipped to +-LOGIT_CLIP, mild shrinkage toward the identity map.
    If the signal is anti-informative the constrained optimum is a = 0: a constant at the base rate, which
    is the honest answer when the in-domain signal carries nothing in the new domain."""
    logits = np.clip(np.asarray(logits, float), -LOGIT_CLIP, LOGIT_CLIP)
    y = np.asarray(y, float)
    a, b = 1.0, 0.0
    loss = _logloss(a, b, logits, y, l2)
    for _ in range(iters):
        z = np.clip(a * logits + b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = np.array([np.sum((p - y) * logits) + l2 * (a - 1), np.sum(p - y) + l2 * b])
        wgt = p * (1 - p) + 1e-6
        H = np.array([[np.sum(wgt * logits * logits) + l2, np.sum(wgt * logits)], [np.sum(wgt * logits), np.sum(wgt) + l2]])
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4:
            na, nb = max(0.0, a - t * step[0]), b - t * step[1]
            nl = _logloss(na, nb, logits, y, l2)
            if nl <= loss:
                break
            t /= 2
        if t <= 1e-4:
            break
        converged = abs(a - na) < 1e-7 and abs(b - nb) < 1e-7
        a, b, loss = na, nb, nl
        if converged:
            break
    if a == 0.0:                                    # anti-informative signal: calibrated constant
        m = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        b = float(np.log(m / (1 - m)))
    return float(a), float(b)


def wilson_upper(k_wrong: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) Wilson upper bound on a binomial proportion."""
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - delta)
    phat = k_wrong / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (centre + half) / denom


def conformal_threshold(credence: np.ndarray, correct: np.ndarray, alpha: float, delta: float = DELTA, n_min: int = N_MIN) -> float:
    """Largest-coverage credence threshold whose (1 - delta) upper confidence bound on the error rate among
    delivered verdicts is <= alpha.  Fixed-sequence testing from the strictest threshold (top n_min cases)
    downwards, stopping at the first failure; returns inf (abstain on everything) when nothing passes."""
    order = np.argsort(-credence)
    if len(order) < n_min:
        return math.inf
    best = math.inf
    for i in range(n_min, len(order) + 1):
        idx = order[:i]
        ucb = wilson_upper(int(np.sum(1 - correct[idx])), i, delta)
        if ucb <= alpha:
            best = float(credence[order[i - 1]])
        else:
            break
    return best


def mahalanobis(X_ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    mu = X_ref.mean(axis=0)
    cov = np.cov(X_ref, rowvar=False) + 1e-6 * np.eye(X_ref.shape[1])
    inv = np.linalg.inv(cov)
    d = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a positive scores above a negative (ties count half)."""
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    order = np.argsort(allv)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ----------------------------------------------------------------------------- per model / split
class Domain:
    """One out-of-domain split for one model: standardised signals and gold-scored outcomes."""

    def __init__(self, eu: EUController, scored: list[Scored]):
        self.scored = scored
        fill1 = {f: float(m) for f, m in zip(eu.std1.feature_names, eu.std1.mu)}
        fill2 = {f: float(m) for f, m in zip(eu.std2.feature_names, eu.std2.mu)}
        self.X1 = eu.std1.transform(feature_matrix([s.x for s in scored], eu.std1.feature_names, fill1))
        self.X2 = eu.std2.transform(feature_matrix([s.z for s in scored], eu.std2.feature_names, fill2))
        self.groups = np.array([s.group_id for s in scored])
        self.answer_right = np.array([s.dprov_correct for s in scored], float)


def metrics_for(dom: Domain, dec: np.ndarray, cred: np.ndarray, idx: np.ndarray, w: float, c: float) -> dict:
    sc = [dom.scored[i] for i in idx]
    d, cr = dec[idx], cred[idx]
    outs = [outcome(s, int(a)) for s, a in zip(sc, d)]
    r = np.array([reward(s, int(a), w, c) for s, a in zip(sc, d)])
    base = np.array([reward(s, C.ANSWER, w, c) for s in sc])
    committed = np.array([o["committed"] for o in outs])
    correct = np.array([o["correct"] for o in outs], float)
    m = {"utility": float(r.mean()), "gain_vs_always_answer": float((r - base).mean()),
         "accuracy": float(correct.mean()), "coverage": float(committed.mean()),
         "harmful": float(np.mean(committed & (correct == 0))),
         "selective_risk": float(np.mean(1 - correct[committed])) if committed.any() else float("nan"),
         "ece": ece(cr[committed & ~np.isnan(cr)], correct[committed & ~np.isnan(cr)]) if (committed & ~np.isnan(cr)).any() else float("nan")}
    return m


def platt_from_cases(eu: EUController, dom: Domain, idx: np.ndarray) -> dict:
    """One monotone recalibration per credence family (answer / checked verdict / kept prior)."""
    la, lcv, lkp = eu.logits(dom.X1[idx], dom.X2[idx])
    y_a = np.array([dom.scored[i].dprov_correct for i in idx], float)
    y_cv = np.array([dom.scored[i].dver_correct for i in idx], float)
    return {"a": platt_fit(la, y_a), "cv": platt_fit(lcv, y_cv), "kp": platt_fit(lkp, y_a)}


def risk_control_eval(cred_cal, ok_cal, cred_test, ok_test, com_test, alphas=ALPHAS) -> dict:
    out = {}
    for a in alphas:
        tau = conformal_threshold(cred_cal, ok_cal, a)
        sel = com_test & (np.nan_to_num(cred_test, nan=-1) >= tau)
        out[a] = {"coverage": float(sel.mean()), "risk": float(np.mean(1 - ok_test[sel])) if sel.any() else float("nan"),
                  "threshold": tau if math.isfinite(tau) else None}
    return out


def run_in_domain_risk(eu: EUController, d_val: Domain, d_test: Domain) -> dict:
    """Reference: threshold chosen on SciFact validation, realised on SciFact test (no domain shift)."""
    dec_v, cred_v, _ = eu.decide_batch(d_val.X1, d_val.X2)
    dec_t, cred_t, _ = eu.decide_batch(d_test.X1, d_test.X2)
    ov = [outcome(s, int(a)) for s, a in zip(d_val.scored, dec_v)]
    ot = [outcome(s, int(a)) for s, a in zip(d_test.scored, dec_t)]
    com_v = np.array([o["committed"] for o in ov])
    res = risk_control_eval(cred_v[com_v], np.array([o["correct"] for o in ov], float)[com_v], cred_t,
                            np.array([o["correct"] for o in ot], float), np.array([o["committed"] for o in ot]))
    res["n_cal"] = int(com_v.sum())
    res["uncontrolled_risk"] = float(np.mean([not o["correct"] for o in ot if o["committed"]]))
    res["uncontrolled_coverage"] = float(np.mean([o["committed"] for o in ot]))
    return res


def run_domain(eu: EUController, dom: Domain, w: float, c: float, ks=KS, repeats=REPEATS, alphas=ALPHAS, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(dom.scored)
    all_idx = np.arange(n)
    dec0, cred0, _ = eu.decide_batch(dom.X1, dom.X2)
    out = {"n": n, "zero_shot": metrics_for(dom, dec0, cred0, all_idx, w, c), "few_shot": {}, "risk_control": {}, "slopes": {}}
    # in-sample ceiling: refit the whole credence model on this domain and score it on the same cases
    eu_full = EUController.fit(dom.scored, dom.X1, dom.X2, eu.std1, eu.std2, w, c, eu.K)
    decf, credf, _ = eu_full.decide_batch(dom.X1, dom.X2)
    out["in_sample_ceiling"] = metrics_for(dom, decf, credf, all_idx, w, c)
    for k in ks:
        if k == 0 or k >= n - 20:
            continue
        rows, rc, slopes = [], {a: [] for a in alphas}, []
        for _ in range(repeats):
            perm = rng.permutation(n)
            cal, test = perm[:k], perm[k:]
            ab = platt_from_cases(eu, dom, cal)
            slopes.append([ab["a"][0], ab["cv"][0], ab["kp"][0]])
            dec, cred, _ = eu.decide_batch(dom.X1, dom.X2, platt=ab)
            rows.append(metrics_for(dom, dec, cred, test, w, c))
            # risk control: zero-shot credences, threshold from the k calibration cases, realised on the held-out cases
            outs_cal = [outcome(dom.scored[i], int(dec0[i])) for i in cal]
            com_cal = np.array([o["committed"] for o in outs_cal])
            if com_cal.sum() >= N_MIN:
                outs_t = [outcome(dom.scored[i], int(dec0[i])) for i in test]
                res = risk_control_eval(cred0[cal][com_cal], np.array([o["correct"] for o in outs_cal], float)[com_cal], cred0[test],
                                        np.array([o["correct"] for o in outs_t], float), np.array([o["committed"] for o in outs_t]), alphas)
                for a in alphas:
                    rc[a].append(res[a])
        agg = {key: (float(np.nanmean([r[key] for r in rows])), float(np.nanstd([r[key] for r in rows]))) for key in rows[0]}
        out["few_shot"][k] = agg
        out["slopes"][k] = [float(np.mean(s)) for s in np.array(slopes).T]       # mean fitted slope a per family
        out["risk_control"][k] = {a: {"coverage": float(np.nanmean([x["coverage"] for x in v])) if v else float("nan"),
                                      "risk": float(np.nanmean([x["risk"] for x in v])) if v else float("nan"),
                                      "share_abstain_all": float(np.mean([x["threshold"] is None for x in v])) if v else float("nan")}
                                  for a, v in rc.items()}
    return out


def novelty_analysis(eu: EUController, X1_train: np.ndarray, X1_val: np.ndarray, dom_in: Domain, dom_out: Domain, w: float, c: float) -> dict:
    cols = [i for i, f in enumerate(eu.std1.feature_names) if f != "bias"]
    d_in = mahalanobis(X1_train[:, cols], dom_in.X1[:, cols])
    d_out = mahalanobis(X1_train[:, cols], dom_out.X1[:, cols])
    d_val = mahalanobis(X1_train[:, cols], X1_val[:, cols])
    thr = float(np.quantile(d_val, 0.95))
    dec, cred, _ = eu.decide_batch(dom_out.X1, dom_out.X2)
    outs = [outcome(s, int(a)) for s, a in zip(dom_out.scored, dec)]
    committed = np.array([o["committed"] for o in outs])
    correct = np.array([o["correct"] for o in outs], float)
    q = np.quantile(d_out, [0.25, 0.5, 0.75])
    bins = np.digitize(d_out, q)
    per_q = []
    for b in range(4):
        m = (bins == b) & committed & ~np.isnan(cred)
        per_q.append({"quartile": b + 1, "n": int(m.sum()), "ece": ece(cred[m], correct[m]) if m.sum() >= 5 else float("nan"),
                      "accuracy": float(correct[bins == b].mean()) if (bins == b).any() else float("nan")})
    # rule: abstain when the case is more novel than 95% of in-domain validation cases
    dec_rule = np.where(d_out > thr, C.ABSTAIN, dec)
    r0 = np.array([reward(s, int(a), w, c) for s, a in zip(dom_out.scored, dec)])
    r1 = np.array([reward(s, int(a), w, c) for s, a in zip(dom_out.scored, dec_rule)])
    harm0 = float(np.mean(committed & (correct == 0)))
    outs1 = [outcome(s, int(a)) for s, a in zip(dom_out.scored, dec_rule)]
    harm1 = float(np.mean([o["committed"] and not o["correct"] for o in outs1]))
    return {"auroc_ood_vs_in": auroc(d_out, d_in), "share_flagged_novel": float(np.mean(d_out > thr)),
            "share_in_domain_flagged": float(np.mean(d_in > thr)), "per_quartile": per_q,
            "novelty_rule": {"utility_before": float(r0.mean()), "utility_after": float(r1.mean()), "harmful_before": harm0, "harmful_after": harm1,
                             "abstain_share_after": float(np.mean(dec_rule == C.ABSTAIN))}}


# ----------------------------------------------------------------------------- report
def render(results: dict, split_names: dict) -> str:
    L = ["# Cross-domain epistemics: does the agent know what it does not know in a new domain?", "",
         "## What this report says", ""]
    for key, mres in results.items():
        label = M.get(key)["label"]
        for split, r in mres["domains"].items():
            z, fs = r["zero_shot"], r["few_shot"]
            ks_sorted = sorted(fs)
            best_k = ks_sorted[-1] if ks_sorted else None
            ece_txt = f"calibration error {z['ece']:.3f} zero-shot"
            if best_k:
                ece_txt += f", {fs[best_k]['ece'][0]:.3f} after recalibrating on {best_k} labelled cases (in-sample ceiling {r['in_sample_ceiling']['ece']:.3f})"
            gain_txt = f"utility vs always answering {z['gain_vs_always_answer']:+.3f} zero-shot"
            if best_k:
                gain_txt += f", {fs[best_k]['gain_vs_always_answer'][0]:+.3f} with {best_k} cases"
            L.append(f"- **{label} on {split_names.get(split, split)}** ({r['n']} cases): {ece_txt}; {gain_txt}.")
            if best_k and r.get("slopes", {}).get(best_k):
                sa, scv, skp = r["slopes"][best_k]
                verdict = ("the pre-check confidence signal carries no usable information in this domain" if sa < 0.2 else
                           "the pre-check confidence signal is over-confident here but still informative")
                L.append(f"  - Recalibration slopes at k = {best_k}: answer {sa:.2f}, checked verdict {scv:.2f}, kept prior {skp:.2f} "
                         f"(1 = in-domain scale was right, 0 = no information). So {verdict}, while the post-check evidence signal "
                         f"keeps {'about half' if 0.35 <= scv <= 0.7 else f'{scv:.0%}'} of its weight.")
            # risk control: can the domain be certified at all?
            rc = r.get("risk_control", {})
            if rc:
                kmax = max(rc)
                certifiable = [a for a, v in rc[kmax].items() if v["share_abstain_all"] < 0.5]
                L.append(f"  - Risk control with {kmax} labelled cases: " + (
                    f"an error ceiling of {min(certifiable):.0%} can be certified at {100 * rc[kmax][min(certifiable)]['coverage']:.0f}% coverage."
                    if certifiable else "no error ceiling up to 40% can be certified with usable coverage; the agent should refuse this domain."))
        nv = mres.get("novelty", {})
        for split, v in nv.items():
            L.append(f"  - Novelty detector for {split_names.get(split, split)}: tells new-domain cases from in-domain ones with AUROC "
                     f"{v['auroc_ood_vs_in']:.2f}; flags {100 * v['share_flagged_novel']:.0f}% of them as outside the calibrated range "
                     f"(vs {100 * v['share_in_domain_flagged']:.0f}% in-domain). Abstaining on flagged cases moves utility "
                     f"{v['novelty_rule']['utility_before']:+.3f} -> {v['novelty_rule']['utility_after']:+.3f} and harmful answers "
                     f"{100 * v['novelty_rule']['harmful_before']:.0f}% -> {100 * v['novelty_rule']['harmful_after']:.0f}%.")
    L += ["", "## How to read this report", "",
          "The credence-based epistemic agent was calibrated on SciFact validation cases. Here it meets claims from other domains. "
          "**Zero-shot** uses it unchanged. **Few-shot** fits a two-parameter recalibration of its credences on k labelled cases "
          "from the new domain and scores the remaining cases; numbers are means over 20 random draws (standard deviation in "
          "brackets). The **in-sample ceiling** refits the whole credence model on all cases of the domain and scores the same "
          "cases, an optimistic bound. **Risk control** picks, on the same k cases, the strictest coverage whose Hoeffding upper "
          "bound (90% confidence) on the error rate among delivered verdicts is below a target; the realised error on held-out "
          "cases shows whether the guarantee holds and what coverage it costs (one-sided 90% Wilson bound, computed on at least 20 "
          "cases). An in-domain reference row (threshold from SciFact validation, realised on SciFact test) shows the mechanism where "
          "no shift is present. **Novelty** is the Mahalanobis distance of a case's six pre-check signals to the SciFact training "
          "distribution.", "",
          "- **Calibration error (ECE)**: average gap between the probability the agent reports and its observed accuracy; 0 is perfect.",
          "- **Utility**: mean reward (+1 right, -1 wrong, 0 abstain, -0.05 per tool call). **Gain**: utility minus always answering.",
          "- **Harmful**: share of all cases with a wrong verdict delivered. **Coverage**: share of cases with a verdict.",
          "- **Selective risk**: error rate among delivered verdicts.",
          "- **Fitted slope a**: the recalibration's multiplier on each in-domain credence family. 1 = the in-domain scale was right; "
          "below 1 = the signal was over-confident in the new domain; 0 = the signal carries no usable information there and the "
          "agent falls back to the domain's base rate."]
    L += ["", "### Table 1. Calibration and utility as labelled cases from the new domain are added", "",
          "*k = 0 is the zero-shot agent. Each further row recalibrates on k labelled cases and scores the rest. Watch the "
          "calibration error fall and the gain over always answering change sign.*", "",
          "| Model | Domain | k | Calibration error | Utility | Gain vs always answer | Accuracy % | Harmful % | Coverage % | Selective risk % | Fitted slope a (answer / checked / kept) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, mres in results.items():
        label = M.get(key)["label"]
        for split, r in mres["domains"].items():
            z = r["zero_shot"]
            L.append(f"| {label} | {split_names.get(split, split)} | 0 | {z['ece']:.3f} | {z['utility']:.3f} | {z['gain_vs_always_answer']:+.3f} | "
                     f"{100 * z['accuracy']:.1f} | {100 * z['harmful']:.1f} | {100 * z['coverage']:.0f} | {100 * z['selective_risk']:.1f} | 1 / 1 / 1 (in-domain) |")
            for k in sorted(r["few_shot"]):
                f = r["few_shot"][k]
                sl = r["slopes"].get(k, [float("nan")] * 3)
                L.append(f"| | | {k} | {f['ece'][0]:.3f} ({f['ece'][1]:.3f}) | {f['utility'][0]:.3f} ({f['utility'][1]:.3f}) | "
                         f"{f['gain_vs_always_answer'][0]:+.3f} ({f['gain_vs_always_answer'][1]:.3f}) | {100 * f['accuracy'][0]:.1f} | "
                         f"{100 * f['harmful'][0]:.1f} | {100 * f['coverage'][0]:.0f} | {100 * f['selective_risk'][0]:.1f} | "
                         f"{sl[0]:.2f} / {sl[1]:.2f} / {sl[2]:.2f} |")
            ic = r["in_sample_ceiling"]
            L.append(f"| | | all (in-sample ceiling) | {ic['ece']:.3f} | {ic['utility']:.3f} | {ic['gain_vs_always_answer']:+.3f} | "
                     f"{100 * ic['accuracy']:.1f} | {100 * ic['harmful']:.1f} | {100 * ic['coverage']:.0f} | {100 * ic['selective_risk']:.1f} | refit |")
    L += ["", "### Table 2. Risk control: a guaranteed ceiling on the error rate of delivered verdicts", "",
          "*Target alpha = the maximum share of delivered verdicts allowed to be wrong. The threshold is chosen on k labelled "
          "cases with a one-sided 90% Wilson bound (at least 20 cases) and applied to held-out cases. 'Realised risk' should sit "
          "at or below alpha in about 90% of draws; 'coverage' is what the guarantee costs. 'Abstain-all' is the share of draws "
          "where no threshold met the bound, i.e. the agent should refuse the whole domain at that error target.*", "",
          "| Model | Domain | k | Target alpha | Realised risk % | Coverage % | Abstain-all share % |", "|---|---|---|---|---|---|---|"]
    for key, mres in results.items():
        label = M.get(key)["label"]
        ir = mres.get("in_domain_risk")
        if ir:
            for a in ALPHAS:
                v = ir[a]
                L.append(f"| {label} | SciFact test (in-domain reference; uncontrolled risk {100 * ir['uncontrolled_risk']:.0f}% at "
                         f"{100 * ir['uncontrolled_coverage']:.0f}% coverage) | {ir['n_cal']} (validation) | {a:.2f} | "
                         f"{'n/a' if v['risk'] != v['risk'] else f'{100 * v['risk']:.1f}'} | {100 * v['coverage']:.0f} | "
                         f"{0 if v['threshold'] is not None else 100} |")
        for split, r in mres["domains"].items():
            for k in sorted(r["risk_control"]):
                for a, v in r["risk_control"][k].items():
                    L.append(f"| {label} | {split_names.get(split, split)} | {k} | {a:.2f} | {100 * v['risk']:.1f} | {100 * v['coverage']:.0f} | {100 * v['share_abstain_all']:.0f} |")
    L += ["", "### Table 3. Novelty: can the agent tell it has left its calibrated range?", "",
          "*AUROC is how well the distance of a case's signals from the training distribution separates new-domain cases from "
          "in-domain test cases (0.5 = no better than chance). The quartile columns give the calibration error of the zero-shot "
          "credence within each novelty quartile of the new domain (Q1 least novel). The last columns apply one rule: abstain "
          "when a case is more novel than 95% of in-domain validation cases.*", "",
          "| Model | Domain | AUROC | Flagged novel % | ECE Q1 | ECE Q2 | ECE Q3 | ECE Q4 | Utility before -> after | Harmful before -> after % |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, mres in results.items():
        label = M.get(key)["label"]
        for split, v in mres.get("novelty", {}).items():
            qs = " | ".join(f"{q['ece']:.3f}" if q["ece"] == q["ece"] else "n/a" for q in v["per_quartile"])
            nr = v["novelty_rule"]
            L.append(f"| {label} | {split_names.get(split, split)} | {v['auroc_ood_vs_in']:.2f} | {100 * v['share_flagged_novel']:.0f} | {qs} | "
                     f"{nr['utility_before']:+.3f} -> {nr['utility_after']:+.3f} | {100 * nr['harmful_before']:.0f} -> {100 * nr['harmful_after']:.0f} |")
    L += ["", "## Reading", "",
          "- If calibration error falls steeply with a few dozen labelled cases, the agent's *signals* carry over to the new domain and "
          "only their *scale* was wrong: the epistemic machinery works and needs a small calibration set per domain.",
          "- If it stays high even at the in-sample ceiling, the signals themselves do not transfer and no recalibration can help.",
          "- Risk control makes the trade explicit: with few calibration cases the bound is loose, so the guarantee is bought with low coverage.",
          "- A novelty AUROC well above 0.5 means the agent can at least recognise unfamiliar territory and widen its uncertainty there."]
    return "\n".join(L) + "\n"


def figures(results: dict, out: Path, split_names: dict) -> list[Path]:
    from eval import figures as F

    plt = F.plt
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    F._ax(axes[0], "Calibration error vs labelled cases from the new domain", "labelled cases k (0 = zero-shot)", "Calibration error (ECE)", grid_axis="both")
    F._ax(axes[1], "Utility gain over always answering vs k", "labelled cases k (0 = zero-shot)", "Utility gain", grid_axis="both")
    i = 0
    for key, mres in results.items():
        for split, r in mres["domains"].items():
            ks = [0] + sorted(r["few_shot"])
            e = [r["zero_shot"]["ece"]] + [r["few_shot"][k]["ece"][0] for k in sorted(r["few_shot"])]
            g = [r["zero_shot"]["gain_vs_always_answer"]] + [r["few_shot"][k]["gain_vs_always_answer"][0] for k in sorted(r["few_shot"])]
            lab = f"{M.get(key)['label']} / {split_names.get(split, split).split(' (')[0]}"
            col = F.SERIES[i % len(F.SERIES)]
            axes[0].plot(ks, e, color=col, linewidth=2, marker="o", markersize=5, markeredgecolor=F.SURFACE, label=lab)
            axes[1].plot(ks, g, color=col, linewidth=2, marker="o", markersize=5, markeredgecolor=F.SURFACE, label=lab)
            i += 1
    axes[1].axhline(0, color=F.AXIS, linewidth=1)
    axes[0].legend(fontsize=7)
    paths.append(F._save(fig, out / "1_recalibration_curves.png",
                         "How to read: each line is one model on one new domain. Left: how far the agent's reported probabilities are from its "
                         "real accuracy, before (k = 0) and after recalibrating on k labelled cases from that domain. Right: whether the agent "
                         "then beats simply always answering. Falling lines on the left mean the signals transfer and only their scale was off."))
    # risk control
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    F._ax(axes[0], "Realised error among delivered verdicts", "labelled cases k", "Error rate", grid_axis="both")
    F._ax(axes[1], "Coverage bought by the guarantee", "labelled cases k", "Share of cases answered", grid_axis="both")
    i = 0
    for key, mres in results.items():
        for split, r in mres["domains"].items():
            for a in ALPHAS:
                ks = sorted(r["risk_control"])
                if not ks:
                    continue
                risk = [r["risk_control"][k][a]["risk"] for k in ks]
                cov = [r["risk_control"][k][a]["coverage"] for k in ks]
                lab = f"{M.get(key)['label']} / {split_names.get(split, split).split(' (')[0]}, alpha {a}"
                col = F.SERIES[i % len(F.SERIES)]
                ls = "-" if a == ALPHAS[0] else "--"
                axes[0].plot(ks, risk, color=col, linewidth=2, linestyle=ls, marker="o", markersize=4.5, markeredgecolor=F.SURFACE, label=lab)
                axes[1].plot(ks, cov, color=col, linewidth=2, linestyle=ls, marker="o", markersize=4.5, markeredgecolor=F.SURFACE)
            i += 1
    for a in ALPHAS:
        axes[0].axhline(a, color=F.AXIS, linewidth=1)
    axes[0].legend(fontsize=6.5)
    paths.append(F._save(fig, out / "2_risk_control.png",
                         "How to read: solid lines target an error rate of 0.2 among delivered verdicts, dashed lines 0.3 (grey rules). Left: the error "
                         "actually realised on held-out cases should stay at or under the target. Right: the share of cases the agent still answers "
                         "under that guarantee; with few calibration cases the statistical bound is loose and coverage is low."))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(M.ORDER))
    ap.add_argument("--splits", default=",".join(C.OOD_SPLITS))
    ap.add_argument("--art", default=str(C.ART_DIR))
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    art = Path(args.art)
    results = {}
    for key in args.models.split(","):
        m = M.get(key)
        eu_path = art / "models" / key / "epistemic_eu.npz"
        cache_dir = art / m["cache"]
        if not eu_path.exists():
            print(f"[transfer] {key}: no credence-based agent at {eu_path}; skipping")
            continue
        eu = EUController.load(eu_path)
        w, c = eu.w, eu.c
        train = load_scored("train", args.cases_dir, cache_dir)
        val = load_scored("val", args.cases_dir, cache_dir)
        test_in = load_scored("test_id", args.cases_dir, cache_dir)
        d_train, d_val, d_in = Domain(eu, train), Domain(eu, val), Domain(eu, test_in)
        results[key] = {"domains": {}, "novelty": {}}
        for split in args.splits.split(","):
            if not (cache_dir / f"{split}.jsonl").exists():
                print(f"[transfer] {key}: no cache for {split}; skipping")
                continue
            dom = Domain(eu, load_scored(split, args.cases_dir, cache_dir))
            results[key]["domains"][split] = run_domain(eu, dom, w, c, repeats=args.repeats, seed=args.seed)
            results[key]["novelty"][split] = novelty_analysis(eu, d_train.X1, d_val.X1, d_in, dom, w, c)
            print(f"[transfer] {key} / {split}: zero-shot ECE {results[key]['domains'][split]['zero_shot']['ece']:.3f}")
        results[key]["in_domain_risk"] = run_in_domain_risk(eu, d_val, d_in)
    if not results:
        raise SystemExit("nothing to evaluate")
    out = art / "models"
    md = render(results, SPLIT_NAMES)
    (out / "transfer.md").write_text(md, encoding="utf-8")
    save_json(out / "transfer.json", results)
    figs = figures(results, out / "figs_transfer", SPLIT_NAMES)
    print(md)
    print(f"[transfer] wrote {out / 'transfer.md'} and {len(figs)} figures")


if __name__ == "__main__":
    main()
