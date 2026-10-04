"""Human-readable report rendering: one vocabulary for every table, a caption and glossary under each,
and a plain-English summary generated from the numbers.  Used by eval/evaluate.py and eval/compare.py.
"""
from __future__ import annotations

from common import config as C
from eval.stats import fmt_ci

SPLIT_NAMES = {"test_id": "SciFact test set (in-domain)", "test_ood": "HealthVer test set (out-of-domain, health claims)",
               "test_climate": "Climate-FEVER test set (out-of-domain, climate claims)",
               "test_vitc": "VitaminC test set (out-of-domain, Wikipedia revisions)",
               "test_ladder": "SciFact evidence ladder (rationale-removed variants)",
               "val": "validation set", "train": "training set"}

# ---- one vocabulary ---------------------------------------------------------------------------------
COLS = {
    "utility": ("Utility", "mean reward per case: +1 for a correct verdict, -w for a wrong one, 0 for abstaining, "
                           "minus c per tool call used (w = 1, c = 0.05 unless stated). Higher is better."),
    "accuracy": ("Accuracy", "share of all cases with a correct verdict. Abstaining counts as wrong."),
    "selective_accuracy": ("Accuracy when answering", "accuracy counted only over the cases where the agent gave a verdict."),
    "coverage": ("Answered", "share of cases where the agent gave a verdict instead of abstaining."),
    "harmful": ("Harmful answers", "share of all cases where the agent committed to a wrong verdict."),
    "unnecessary_abstention": ("Needless abstentions", "share of all cases where the agent abstained although answering or "
                                                        "checking would have given the right verdict."),
    "tool_calls": ("Tool calls / case", "average number of search / read / calculate calls per case (0 when not checking)."),
    "fabrication": ("Fabricated citations", "share of answered cases citing a record the agent was never shown and never opened."),
    "llm_calls": ("LLM calls / case", "average number of model calls per case (2 for the two provisional samples, more when checking)."),
    "tokens": ("Tokens / case", "average prompt + completion tokens per case."),
    "gpu_s": ("Model seconds / case", "average wall time of the model calls per case (a GPU-seconds proxy)."),
    "ece_raw": ("Calibration error, stated confidence", "expected calibration error of the model's own verbalised confidence "
                                                        "(0 = the stated confidence matches observed accuracy)."),
    "ece_cal": ("Calibration error, reported credence", "expected calibration error of the number the deployed agent reports: "
                                                        "its credence for the epistemic agents, the fitted calibrator otherwise."),
    "check_rate": ("Checked", "share of cases where the agent ran the checker (search + read tools)."),
    "contested": ("Contested", "among checked cases, the share where the check disagreed with the agent's first answer."),
    "wrong_reason": ("Right for the wrong reason", "among correct supported/refuted verdicts, the share citing no gold rationale "
                                                   "sentence / no gold abstract at all."),
    "grounding_flip": ("Notices removed evidence", "on claims whose supporting abstract was removed from the records, the share "
                                                    "where the agent switched to 'insufficient evidence' or abstained "
                                                    "(counted over removed-evidence twins whose original claim it got right)."),
    "stubborn": ("Stubborn", "share of removed-evidence twins where the agent kept the original verdict with confidence >= 0.7, "
                             "i.e. answered from memory or from an unrelated abstract."),
    "flips": ("Checking fixed / broke", "number of cases the check turned wrong-to-right / right-to-wrong, among checked cases."),
}

POLICY_LABELS = {
    "always_answer": "Always answer (first answer, no checking)",
    "always_verify": "Always run the raw checker",
    "always_abstain": "Always abstain",
    "heuristic": "Hand-tuned thresholds",
    "ours": "Learned controller (answer / check / abstain)",
    "always_check": "Always run the disciplined checker",
    "epistemic_rl": "Epistemic agent, learned (two-stage)",
    "epistemic_eu": "Epistemic agent, credence-based",
    "oracle": "Oracle (knows the right action; upper bound)",
    "shortcut": "Controller trained on the shortcut data",
}
POLICY_DESC = {
    "always_answer": "the frozen model's provisional verdict on the three retrieved abstracts, every time. This is the agent before any controller.",
    "always_verify": "runs the tool loop (search, read, calculate; at most K calls) on every case and returns whatever verdict the loop ends with.",
    "always_abstain": "never gives a verdict. Utility 0 by definition; a sanity floor.",
    "heuristic": "abstain if mean confidence is below a threshold, check if below a second threshold or if the two samples disagree, else answer. Thresholds picked on validation.",
    "ours": "the 3 x 6 softmax policy trained with REINFORCE on the six pre-check signals; argmax at deployment.",
    "always_check": "runs the tool loop on every case, but the verdict must cite a sentence the agent actually read, and where a blind-judge record exists an evidence-only reading replaces the checker's self-assessment.",
    "epistemic_rl": "two softmax heads trained with REINFORCE: before checking (answer / check / abstain) and after checking (commit the checked verdict / keep the first answer / abstain).",
    "epistemic_eu": "decides from calibrated probabilities fit on validation: answers when P(correct) > w/(1+w), checks when the predicted gain from checking exceeds its cost, and after checking commits, keeps the first answer, or abstains by expected reward. The number it reports is that probability.",
    "oracle": "picks, with knowledge of the gold label, the best of answer / check / abstain for each case. Not a competitor.",
    "shortcut": "the same learned controller, trained on raw SciFact where 'no evidence' claims have an empty evidence field. Shows what the dataset shortcut does.",
}


def model_label(llm: dict | None) -> str:
    if not llm:
        return "unknown model"
    try:
        from common import models as M

        for k in M.ORDER:
            if M.get(k)["hf_id"] == llm.get("model"):
                return M.get(k)["label"]
    except Exception:  # noqa: BLE001
        pass
    return str(llm.get("model", "unknown model"))


def pct(d, digits=1):
    return fmt_ci(d, pct=True) if isinstance(d, dict) else ("n/a" if d is None else f"{100 * d:.{digits}f}")


def _p(v):
    return "n/a" if v is None else f"{100 * v:.1f}"


def _sig(d: dict) -> str:
    """Wording for a paired difference with a 95% interval."""
    if d is None or d.get("n", 0) == 0:
        return "not available"
    if d["lo"] > 0:
        return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]: the interval excludes zero, so this is a real gain"
    if d["hi"] < 0:
        return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]: the interval excludes zero, so this is a real loss"
    return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]: the interval includes zero, so the difference could be chance"


def glossary(keys: list[str]) -> str:
    return "\n".join(f"- **{COLS[k][0]}**: {COLS[k][1]}" for k in keys)


# ---- sections ----------------------------------------------------------------------------------
def how_to_read(policies: list[str], w: float, c: float, K: int, split: str, n_cases: int, n_groups: int, llm: dict) -> str:
    L = ["## How to read this report", "",
         f"**Setup.** {n_cases} cases from the {SPLIT_NAMES.get(split, split)}, grouped into {n_groups} claim groups "
         f"(claims that share an abstract, plus each claim's removed-evidence twin). The frozen model is {model_label(llm)}. "
         "For every case the frozen model was run once for each possible action and the outcomes were cached, so every "
         "policy below is scored on exactly the same model outputs; differences between rows are differences in *decisions*, "
         "not in model luck.", "",
         "**What the agent can do with a case.** *Answer*: commit to the provisional verdict (supported / refuted / insufficient "
         "evidence). *Check*: run the tool loop (search the record store, read abstracts, calculate; at most "
         f"{K} calls) and then decide what to believe. *Abstain*: give no verdict.", "",
         f"**How we score it.** A correct verdict earns +1, a wrong one costs w = {w:g}, abstaining earns 0, and every tool call "
         f"costs c = {c:g}. The average over cases is the **utility**. Answering beats abstaining exactly when the chance of being "
         f"right exceeds w/(1+w) = {w / (1 + w):.2f}. **Accuracy** counts an abstention as wrong, so a cautious policy can have "
         "high utility and low accuracy at the same time; read both.", "",
         "**Brackets.** `mean [low, high]` is a 95% confidence interval from resampling whole claim groups 10,000 times. "
         "When two policies' intervals overlap they may still differ, so paired differences are reported separately "
         "(a paired interval that excludes zero is a real difference).", "",
         "**Policies in the tables.**", ""]
    for p in policies:
        L.append(f"- **{POLICY_LABELS[p]}**: {POLICY_DESC[p]}")
    return "\n".join(L)


def table_headline(metrics: dict, policies: list[str], split: str, best_fn) -> str:
    cols = ["utility", "accuracy", "selective_accuracy", "coverage", "harmful", "unnecessary_abstention", "tool_calls", "fabrication"]
    pctcols = {"accuracy", "selective_accuracy", "coverage", "harmful", "unnecessary_abstention", "fabrication"}
    best = {k: best_fn(metrics, k, policies) for k in cols}
    L = ["### Table 1. How good is each policy?", "",
         "*One row per policy, scored on the same cached model outputs. Start with Utility (the single number the reward "
         "defines) and Accuracy, then look at Harmful answers (wrong verdicts delivered) and Needless abstentions (caution "
         "that cost a correct verdict). The best non-oracle value in each column is bold.*", "",
         "| Policy | " + " | ".join(COLS[k][0] + (" %" if k in pctcols else "") for k in cols) + " |", "|---|" + "---|" * len(cols)]
    for p in policies:
        cells = []
        for k in cols:
            s = fmt_ci(metrics[p][k], pct=k in pctcols)
            cells.append(f"**{s}**" if best[k] == p else s)
        L.append(f"| {POLICY_LABELS[p]} | " + " | ".join(cells) + " |")
    L += ["", glossary(cols)]
    return "\n".join(L)


def table_integrity(metrics: dict, policies: list[str], has_twins: bool) -> str:
    L = ["### Table 2. Is the agent cheating, lucky, or answering from memory?", "",
         "*The same policies, judged on how they reached their verdicts rather than on whether they were right. These are the "
         "checks that accuracy alone cannot see: invented citations, correct verdicts with the wrong evidence, and verdicts "
         "that survive the removal of their evidence.*", "",
         "| Policy | Fabricated citations % | Right for the wrong reason % (no gold sentence / no gold abstract) | "
         + ("Notices removed evidence % | Stubborn % | " if has_twins else "")
         + "Checking fixed / broke (n checked) | Needless abstentions, as share of abstentions % |",
         "|---|---|---|" + ("---|---|" if has_twins else "") + "---|---|"]
    for p in policies:
        i = metrics[p]["integrity"]
        row = (f"| {POLICY_LABELS[p]} | {_p(i['fabrication_rate'])} | {_p(i['wrong_reason_rate'])} / {_p(i['wrong_reason_rate_doc'])} | "
               + (f"{_p(i['grounding_flip_rate'])} | {_p(i['stubborn_rate'])} | " if has_twins else "")
               + f"{i['verify_wrong_to_right']} / {i['verify_right_to_wrong']} (n={i['n_verified']}) | {_p(i['unnecessary_abstention_among_abstained'])} |")
        L.append(row)
    keys = ["fabrication", "wrong_reason"] + (["grounding_flip", "stubborn"] if has_twins else []) + ["flips"]
    L += ["", glossary(keys), "- **Needless abstentions, as share of abstentions**: of the cases the policy abstained on, how many it could have got right."]
    if not has_twins:
        L.append("- *Notices removed evidence* and *Stubborn* need removed-evidence twins, which this split does not have.")
    return "\n".join(L)


def table_cost(metrics: dict, policies: list[str]) -> str:
    L = ["### Table 3. What each policy costs, and how honest its confidence is", "",
         "*Cost columns are per case. The two calibration columns compare the model's own stated confidence with the "
         "probability the deployed agent actually reports; both are computed on the cases where a verdict was given, so "
         "they are blank for policies that never answer.*", "",
         "| Policy | " + " | ".join(COLS[k][0] for k in ["llm_calls", "tool_calls", "tokens", "gpu_s", "ece_raw", "ece_cal"]) + " |",
         "|---|---|---|---|---|---|---|"]
    for p in policies:
        m = metrics[p]
        cal = m.get("calibration", {})
        f = lambda k: "n/a" if k not in cal or cal[k] != cal[k] else f"{cal[k]:.3f}"  # noqa: E731
        L.append(f"| {POLICY_LABELS[p]} | {m['llm_calls']['mean']:.2f} | {m['tool_calls']['mean']:.2f} | {m['tokens']['mean']:.0f} | "
                 f"{m['gpu_s']['mean']:.2f} | {f('ece_raw')} | {f('ece_cal')} |")
    L += ["", glossary(["llm_calls", "tool_calls", "tokens", "gpu_s", "ece_raw", "ece_cal"]),
          "- Calibration error (ECE) is the average gap between predicted probability and observed accuracy over 10 probability bins; "
          "0.05 means the reported probabilities are off by five points on average."]
    return "\n".join(L)


def table_epistemic(metrics: dict, stats: dict, behaviour: dict, policies: list[str], judged_share: float | None) -> str:
    pols = [p for p in ("epistemic_eu", "epistemic_rl") if p in policies]
    if not pols:
        return ""
    L = ["### Table 4. What the epistemic agents actually do", "",
         "*How often each epistemic agent checks, how often the check contradicts its first answer, and what it does then. "
         "'Keeps first answer' after a contested check means the agent judged the checker less reliable than its prior.*", ""]
    if judged_share is not None:
        L.append(f"Blind-judge records exist for {100 * judged_share:.0f}% of the cases on this split (every case where the checker "
                 "committed to a verdict and cited a sentence it had read). For those, the checked verdict is the judge's "
                 "evidence-only reading; for the rest it is the checker's own reading, subject to the grounding rule.")
        L.append("")
    L += ["| Agent | Checked % | Contested % | After a contested check: commits checked verdict / keeps first answer / abstains | "
          "Decision mix | Utility | Utility vs always answer | Utility vs learned controller | Calibration error, reported credence |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in pols:
        m, e, b = metrics[p], stats["epistemic"][p], behaviour[p]
        mix = ", ".join(f"{k.replace('_', ' ')} {100 * v:.0f}%" for k, v in m["decision_mix"].items())
        ece = e["credence_ece"]
        L.append(f"| {POLICY_LABELS[p]} | {100 * m['check_rate']:.0f} | {_p(m['contested_rate_among_checked'])} | "
                 f"{b['contested_commit']} / {b['contested_keep']} / {b['contested_abstain']} (of {b['n_contested']}) | {mix} | "
                 f"{fmt_ci(m['utility'])} | {fmt_ci(e['gain_vs_always_answer'])} | {fmt_ci(e['gain_vs_ours'])} | "
                 f"{'n/a' if ece is None else f'{ece:.3f}'} |")
    L += ["", glossary(["check_rate", "contested", "utility", "ece_cal"]),
          "- **Decision mix**: share of cases per final decision. *answer grounded* = first answer, kept only if it cites a shown "
          "abstract; *check commit* = adopted the checked verdict; *check keep prior* = checked, then kept the first answer; "
          "*check abstain* = checked, then abstained (paying the tool cost); *abstain* = abstained without checking.",
          "- **Utility vs ...**: paired difference with its 95% interval; positive favours the epistemic agent."]
    return "\n".join(L)


def plain_summary(metrics: dict, stats: dict, behaviour: dict, policies: list[str], split: str, llm: dict,
                  judged_share: float | None, has_twins: bool, w: float) -> str:
    m = metrics
    aa, orc = m["always_answer"], m["oracle"]
    L = ["## What this report says", ""]
    L.append(f"- **The model's first answer is right {100 * aa['accuracy']['mean']:.0f}% of the time** on the "
             f"{SPLIT_NAMES.get(split, split)} with {model_label(llm)}. If an oracle chose the best action for every case, "
             f"accuracy would be {100 * orc['accuracy']['mean']:.0f}%; the gap is the most any controller could add.")
    if "always_verify" in m:
        iv = m["always_verify"]["integrity"]
        L.append(f"- **Checking as originally built {'hurts' if m['always_verify']['accuracy']['mean'] < aa['accuracy']['mean'] else 'helps'}.** "
                 f"Run on every case, the raw checker is right {100 * m['always_verify']['accuracy']['mean']:.0f}% of the time: it fixed "
                 f"{iv['verify_wrong_to_right']} first answers and broke {iv['verify_right_to_wrong']}.")
    if "always_check" in m:
        ic = m["always_check"]["integrity"]
        jtxt = f" and the blind judge (available for {100 * judged_share:.0f}% of cases)" if judged_share else ""
        L.append(f"- **With the discipline rules{jtxt}, the checked verdict is right {100 * m['always_check']['accuracy']['mean']:.0f}% "
                 f"of the time**, fixing {ic['verify_wrong_to_right']} and breaking {ic['verify_right_to_wrong']}"
                 f"{' - checking now adds information' if m['always_check']['accuracy']['mean'] > aa['accuracy']['mean'] else ' - still below the first answer, so checking everything is not worth it'}.")
    non_oracle = [p for p in policies if p not in ("oracle", "shortcut")]
    best = max(non_oracle, key=lambda p: m[p]["utility"]["mean"])
    L.append(f"- **Highest utility: {POLICY_LABELS[best]}** at {m[best]['utility']['mean']:.3f} (always answering scores "
             f"{aa['utility']['mean']:.3f}; the oracle {orc['utility']['mean']:.3f}).")
    if "ours" in m:
        d = stats["utility_diff_ours_minus_best_baseline"]
        L.append(f"- **Learned controller vs {POLICY_LABELS[stats['best_baseline']]}**: utility difference {_sig(d)}. "
                 f"It gives a verdict on {100 * m['ours']['coverage']['mean']:.0f}% of cases (running the checker first on "
                 f"{100 * m['ours']['check_rate']:.0f}%) and abstains on the other {100 * (1 - m['ours']['coverage']['mean']):.0f}%; "
                 f"harmful answers go from {100 * aa['harmful']['mean']:.0f}% to {100 * m['ours']['harmful']['mean']:.0f}%.")
    for p in [q for q in ("epistemic_eu", "epistemic_rl") if q in m]:
        e, b = stats["epistemic"][p], behaviour[p]
        what = []
        if b["n_contested"]:
            what.append(f"when the check contradicted its first answer ({b['n_contested']} cases) it adopted the checked verdict "
                        f"{b['contested_commit']} times, kept its first answer {b['contested_keep']} times and abstained {b['contested_abstain']} times")
        ece = e["credence_ece"]
        L.append(f"- **{POLICY_LABELS[p]}**: utility {m[p]['utility']['mean']:.3f}, vs always answer {_sig(e['gain_vs_always_answer'])}. "
                 f"It checks {100 * m[p]['check_rate']:.0f}% of cases{'; ' + what[0] if what else ''}. "
                 + (f"Its reported probability of being right has calibration error {ece:.3f} "
                    f"(the model's own stated confidence: {m['always_answer'].get('calibration', {}).get('ece_raw', float('nan')):.3f})." if ece is not None else ""))
    if has_twins and "always_answer" in m:
        ia = m["always_answer"]["integrity"]
        L.append(f"- **Removed-evidence test.** When a claim's supporting abstract is taken away, the first answer switches to "
                 f"'insufficient evidence' {_p(ia['grounding_flip_rate'])}% of the time and keeps the old verdict with high confidence "
                 f"{_p(ia['stubborn_rate'])}% of the time (answering from memory or from an unrelated abstract).")
    fab = [m[p]["integrity"]["fabrication_rate"] for p in non_oracle if m[p]["integrity"]["fabrication_rate"] is not None]
    if fab:
        L.append(f"- **Fabricated citations** stay between {100 * min(fab):.1f}% and {100 * max(fab):.1f}% across policies.")
    return "\n".join(L)


def figure_guide(figs: dict) -> str:
    guide = {
        "risk_coverage": "Risk-coverage curve: accuracy among answered cases (y) against the share of cases answered (x) as the "
                         "willingness to abstain varies. Higher and further right is better; the oracle is the ceiling.",
        "phase_diagram": "Reward-design map for the learned controller: for each wrong-answer penalty w (rows) and tool cost c "
                         "(columns), the colour is the action it mostly takes and the number is its utility. Shows where a reward "
                         "setting collapses the controller into always-abstain or always-check.",
        "phase_diagram_epistemic": "The same map for the credence-based epistemic agent, which needs no retraining per cell.",
        "grounding": "Removed-evidence test per policy: accuracy on the original claims, how often the twin without evidence is "
                     "recognised as 'insufficient', and how often the old verdict is kept (stubborn).",
        "reliability": "Reliability diagram: predicted probability (x) against observed accuracy (y). Points on the diagonal are "
                       "perfectly calibrated; the model's stated confidence vs the reported credence.",
        "cost_frontier": "Accuracy against tool calls per case as the tool cost c varies; the fixed policies are single points.",
        "w_heatmap": "The learned controller's weights: which signal pushes toward which action.",
        "action_mix": "What each policy does on cases where both actions would be right, only checking would be right, only "
                      "answering would be right, or neither.",
    }
    L = ["## Figures", ""]
    for k, path in figs.items():
        L.append(f"- `{path}`: {guide.get(k, k)}")
    return "\n".join(L)
