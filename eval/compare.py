"""Cross-model comparison: does the value of checking shrink as the frozen model gets more capable?

    python -m eval.compare [--models llama3b,llama8b,gemma26b] [--art art]

Reads each model's cache (art/<cache>) and evaluation (art/models/<key>/) and writes
art/models/comparison.md / .json plus figures in art/models/figs_compare/.  Every difference
is a paired cluster bootstrap over claim groups.  With three models from two families, the
trend is descriptive: parameter count is confounded with family and training recipe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import config as C
from common import models as M
from common.io import read_jsonl, save_json
from eval.stats import cluster_bootstrap, fmt_ci, paired_cluster_bootstrap
from scorer.score import load_scored


def _log_rewards(model_dir: Path, split: str, policy: str) -> dict[str, float]:
    return {r["case_id"]: float(r["reward"]) for r in read_jsonl(model_dir / "logs" / f"{split}_{policy}.jsonl")}


def model_summary(key: str, art: Path, cases_dir: Path, split: str, n_boot: int) -> dict | None:
    m = M.get(key)
    mdir = art / "models" / key
    res_path = mdir / f"results_{split}.json"
    cache_dir = art / m["cache"]
    if not res_path.exists() or not (cache_dir / f"{split}.jsonl").exists():
        return None
    res = json.loads(res_path.read_text())
    sc = load_scored(split, cases_dir, cache_dir)
    groups = [s.group_id for s in sc]
    ids = [s.case_id for s in sc]
    prov = np.array([s.prov_correct for s in sc], float)
    ver = np.array([s.ver_correct for s in sc], float)
    cb = lambda v: cluster_bootstrap(v, groups, n_boot)  # noqa: E731
    met = res["metrics"]
    out = {
        "key": key, "label": m["label"], "family": m["family"], "params_total": m["params_total"],
        "params_active": m["params_active"], "n_cases": len(sc), "llm": res.get("llm"),
        "answer_acc": cb(prov), "verify_acc": cb(ver),
        "oracle_acc": met["oracle"]["accuracy"],
        "verify_wrong_to_right": cb((1 - prov) * ver), "verify_right_to_wrong": cb(prov * (1 - ver)),
        "verify_net": cb(ver - prov),
        "utility": {p: met[p]["utility"] for p in ("always_answer", "always_verify", "heuristic", "ours", "oracle") if p in met},
        "ours_mix": met["ours"]["action_mix"],
        "calibration": {"ece_raw_answer": met["always_answer"]["calibration"].get("ece_raw"),
                        "ece_cal_ours": met["ours"]["calibration"].get("ece_cal"),
                        "ece_raw_ours": met["ours"]["calibration"].get("ece_raw")},
        "integrity": {p: {k: met[p]["integrity"].get(k) for k in ("fabrication_rate", "grounding_flip_rate", "stubborn_rate",
                                                                   "wrong_reason_rate_doc")}
                      for p in ("always_answer", "always_verify", "ours")},
        "signals": {f: float(np.std([s.x.get(f) or 0 for s in sc])) for f in ("conf_mean", "agree", "suff_mean", "conf_gap")},
        "agree_rate": float(np.mean([s.x["agree"] for s in sc])),
    }
    rew = {p: _log_rewards(mdir, split, p) for p in ("ours", "always_answer", "heuristic")}
    for base in ("always_answer", "heuristic"):
        out[f"gain_vs_{base}"] = paired_cluster_bootstrap([rew["ours"][i] for i in ids], [rew[base][i] for i in ids], groups, n_boot)
    # epistemic agent (disciplined verifier + two-stage decisions + credences)
    dver = np.array([s.dver_correct for s in sc], float)
    out["disciplined_check_acc"] = cb(dver)
    out["disciplined_fixes"] = cb((1 - prov) * dver)
    out["disciplined_breaks"] = cb(prov * (1 - dver))
    out["judged_share"] = float(np.mean([s.judge is not None for s in sc]))
    # ceiling for OUR pipeline: best of every action it can take (answer, raw/disciplined check, keep prior, abstain), gold known
    from scorer.score import reward as _reward
    acts = [C.ANSWER, C.VERIFY, C.ABSTAIN, C.D_ANSWER, C.CHECK_COMMIT, C.CHECK_KEEP, C.CHECK_ABSTAIN]
    best_r = np.array([max(_reward(s, a) for a in acts) for s in sc])
    out["pipeline_oracle_utility"] = cb(best_r)
    out["epistemic"] = {}
    for pol in ("epistemic_eu", "epistemic_rl"):
        if pol in met and (mdir / "logs" / f"{split}_{pol}.jsonl").exists():
            rp = _log_rewards(mdir, split, pol)
            e = res["stats"].get("epistemic", {}).get(pol, {})
            out["epistemic"][pol] = {
                "utility": met[pol]["utility"], "accuracy": met[pol]["accuracy"], "harmful": met[pol]["harmful"],
                "coverage": met[pol]["coverage"],
                "gain_vs_always_answer": paired_cluster_bootstrap([rp[i] for i in ids], [rew["always_answer"][i] for i in ids], groups, n_boot),
                "gain_vs_ours": paired_cluster_bootstrap([rp[i] for i in ids], [rew["ours"][i] for i in ids], groups, n_boot),
                "check_rate": met[pol].get("check_rate"), "contested_rate": met[pol].get("contested_rate_among_checked"),
                "contested_then_abstained": met[pol].get("contested_then_abstained"),
                "decision_mix": met[pol].get("decision_mix", {}), "credence_ece": e.get("credence_ece"),
                "fabrication": met[pol]["integrity"].get("fabrication_rate"),
                "selective_accuracy": met[pol]["selective_accuracy"],
            }
    out["baseline"] = {"utility": met["always_answer"]["utility"], "harmful": met["always_answer"]["harmful"],
                       "selective_accuracy": met["always_answer"]["selective_accuracy"],
                       "coverage": met["always_answer"]["coverage"],
                       "fabrication_raw_checker": met["always_verify"]["integrity"].get("fabrication_rate")}
    return out


def _fx(d, pct_=True):
    return fmt_ci(d, pct=pct_) if isinstance(d, dict) else ("n/a" if d is None else (f"{100 * d:.1f}" if pct_ else f"{d:.3f}"))


def _real(d) -> str:
    """Short verdict on a paired difference."""
    if not d or d.get("n", 0) == 0:
        return "n/a"
    if d["lo"] > 0:
        return "real gain"
    if d["hi"] < 0:
        return "real loss"
    return "could be chance"


def render(rows: list[dict], split: str) -> str:
    from eval.report import COLS, POLICY_LABELS, SPLIT_NAMES, glossary

    rows = sorted(rows, key=lambda r: r["params_total"])
    by = {r["label"]: r for r in rows}
    labels = [r["label"] for r in rows]
    L = [f"# Three frozen models on one benchmark: {SPLIT_NAMES.get(split, split)}", ""]

    # ---- plain-English summary -------------------------------------------------------------------------
    L += ["## What this comparison says", ""]
    accs = [r["answer_acc"]["mean"] for r in rows]
    rises = all(a < b for a, b in zip(accs, accs[1:]))
    L.append(("- **First-answer accuracy rises with model size**: " if rises else
              "- **First-answer accuracy does not rise with model size here**: ") +
             ", ".join(f"{r['label']} {100 * r['answer_acc']['mean']:.0f}%" for r in rows) +
             ". A perfect per-case choice between answering, checking and abstaining would reach " +
             ", ".join(f"{100 * r['oracle_acc']['mean']:.0f}%" for r in rows) + " respectively.")
    worse = [r for r in rows if r["verify_acc"]["mean"] < r["answer_acc"]["mean"]]
    L.append(f"- **The raw checker hurts {'every model' if len(worse) == len(rows) else str(len(worse)) + ' of ' + str(len(rows)) + ' models'}**: "
             "run on every case it is right " + ", ".join(f"{100 * r['verify_acc']['mean']:.0f}%" for r in rows) +
             " of the time. It breaks more first answers than it fixes (" +
             ", ".join(f"{r['label']}: fixes {100 * r['verify_wrong_to_right']['mean']:.0f}% / breaks {100 * r['verify_right_to_wrong']['mean']:.0f}%" for r in rows) +
             ")" + (", and the smaller the model the worse the damage." if all(a > b for a, b in zip(
                 [r["verify_right_to_wrong"]["mean"] for r in rows], [r["verify_right_to_wrong"]["mean"] for r in rows][1:])) else "."))
    disc = [f"{r['label']} {100 * r['verify_acc']['mean']:.0f}% -> {100 * r['disciplined_check_acc']['mean']:.0f}%"
            f"{' (now above its first answer)' if r['disciplined_check_acc']['mean'] > r['answer_acc']['mean'] else ' (still below its first answer)'}" for r in rows]
    rescued = [r for r in rows if r["disciplined_check_acc"]["mean"] > r["answer_acc"]["mean"]]
    head = ("- **The disciplined checker (grounding rule + blind judge) changes the picture**: " if rescued else
            "- **The disciplined checker (grounding rule + blind judge) does not rescue checking here**: ")
    L.append(head + "checked-verdict accuracy goes " +
             "; ".join(disc) + ". Blind-judge records cover " + ", ".join(f"{100 * r['judged_share']:.0f}%" for r in rows) +
             " of the cases (every case where the checker committed and cited a sentence it had read).")
    gains = []
    for r in rows:
        bits = [f"learned controller {_real(r['gain_vs_always_answer'])} ({r['gain_vs_always_answer']['mean']:+.3f})"]
        for pol, name in (("epistemic_eu", "credence-based epistemic agent"), ("epistemic_rl", "two-stage epistemic agent")):
            e = r["epistemic"].get(pol)
            if e:
                bits.append(f"{name} {_real(e['gain_vs_always_answer'])} ({e['gain_vs_always_answer']['mean']:+.3f})")
        gains.append(f"{r['label']}: " + ", ".join(bits))
    L.append("- **Does deciding beat always answering?** Utility differences vs the first answer, with a verdict on whether the 95% "
             "interval excludes zero. " + "; ".join(gains) + ".")
    eces = []
    for r in rows:
        e = r["epistemic"].get("epistemic_eu")
        if e and e["credence_ece"] is not None:
            eces.append(f"{r['label']} {e['credence_ece']:.3f} (stated confidence {r['calibration']['ece_raw_answer']:.3f})")
    if eces:
        worst = max(r["epistemic"]["epistemic_eu"]["credence_ece"] for r in rows if r["epistemic"].get("epistemic_eu") and r["epistemic"]["epistemic_eu"]["credence_ece"] is not None)
        head = ("- **The credence the epistemic agent reports is well calibrated** on this split: " if worst < 0.1 else
                "- **The credence the epistemic agent reports is poorly calibrated** on this split (it was fit in-domain and does not transfer): ")
        L.append(head + "calibration error " + ", ".join(eces) +
                 ". Lower is better; the model's own stated confidence is given in brackets for comparison.")
    if all(r["integrity"]["always_answer"].get("grounding_flip_rate") is not None for r in rows):
        L.append("- **Removed-evidence test**: when a claim's supporting abstract is taken away, the first answer switches to 'insufficient "
                 "evidence' " + ", ".join(f"{r['label']} {100 * r['integrity']['always_answer']['grounding_flip_rate']:.0f}%" for r in rows) +
                 " of the time, and keeps the old verdict with high confidence " +
                 ", ".join(f"{100 * r['integrity']['always_answer']['stubborn_rate']:.0f}%" for r in rows) + ".")
    L.append("- **Caveat.** Three models from two families cannot separate size from family or training recipe, and Gemma 4 26B-A4B is a "
             "mixture of experts with about 4B parameters active per token. Read the size trend as descriptive.")

    # ---- headline: what our pipeline adds ------------------------------------------------------------
    hl = [r for r in rows if r["epistemic"].get("epistemic_eu")]
    if hl:
        L += ["", "## Headline: what our pipeline adds", "",
              "*Each row compares the frozen model used as-is (it always answers with its first verdict) with the same model wrapped "
              "in our pipeline, on the same cases. \"Our pipeline\" is fixed in advance as the final design, the credence-based "
              "epistemic agent (grounding rule, blind judge, two-stage decision from calibrated credences), so the row is not the "
              "best of several policies picked after seeing these results. Left of each arrow: the model alone; right: with our pipeline.*", "",
              "| Model | Utility | Gain in utility (95% interval) | Share of the possible gain captured | Harmful answers % | "
              "Accuracy when it answers % | Answers given % | Calibration error of the confidence it reports | Fabricated citations % |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in hl:
            e, b = r["epistemic"]["epistemic_eu"], r["baseline"]
            g = e["gain_vs_always_answer"]
            head = r["pipeline_oracle_utility"]["mean"] - b["utility"]["mean"]
            share = f"{100 * g['mean'] / head:.0f}%" if head > 1e-9 else "n/a"
            ece_b, ece_o = r["calibration"]["ece_raw_answer"], e["credence_ece"]
            fab_b = b["fabrication_raw_checker"]
            L.append(f"| {r['label']} | {b['utility']['mean']:.3f} -> {e['utility']['mean']:.3f} | "
                     f"{g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}] ({_real(g)}) | {share} | "
                     f"{100 * b['harmful']['mean']:.1f} -> {100 * e['harmful']['mean']:.1f} | "
                     f"{100 * b['selective_accuracy']['mean']:.1f} -> {100 * e['selective_accuracy']['mean']:.1f} | "
                     f"{100 * b['coverage']['mean']:.0f} -> {100 * e['coverage']['mean']:.0f} | "
                     f"{ece_b:.3f} -> {'n/a' if ece_o is None else f'{ece_o:.3f}'} | "
                     f"{'n/a' if fab_b is None else f'{100 * fab_b:.1f}'} -> {'n/a' if e['fabrication'] is None else f'{100 * e['fabrication']:.1f}'} |")
        L += ["",
              "- **Utility**: mean reward per case (+1 correct verdict, -1 wrong verdict, 0 abstain, -0.05 per tool call). Higher is better.",
              "- **Gain in utility**: paired difference on the same cases with its 95% interval. *real gain* = the interval excludes zero.",
              "- **Share of the possible gain captured**: the gain divided by the gap between the model alone and an oracle that, knowing the "
              "gold label, picks the best of every action our pipeline can take (answer, check with the raw or disciplined checker, keep the "
              "first answer, abstain). 100% would be a perfect decision-maker.",
              "- **Harmful answers**: share of all cases where a wrong verdict was delivered. Lower is better.",
              "- **Accuracy when it answers**: accuracy over the cases where a verdict was given, i.e. how far a given verdict can be trusted.",
              "- **Answers given**: share of cases with a verdict; the rest are abstentions (\"I cannot give a reliable verdict\").",
              "- **Calibration error**: average gap between the confidence reported and the accuracy actually achieved (0 = perfectly honest). "
              "Left: the model's own stated confidence. Right: the probability our pipeline reports.",
              "- **Fabricated citations**: share of verdicts citing a record never shown or opened. Left: the original checker, the only part "
              "of the unwrapped agent that cites records; right: our pipeline, where the grounding rule makes it zero by construction."]

    # ---- how to read ---------------------------------------------------------------------------------
    L += ["", "## How to read these tables", "",
          "Every model was run on the same cases with the same prompts, tools and tool limit; only the frozen model changes. "
          "For each case the model's outputs for every action were cached once, so all policies are scored on identical model "
          "behaviour. A **policy** is a rule for choosing, per case, between *answer* (commit to the first verdict), *check* (run the "
          "search/read tool loop, then decide what to believe) and *abstain* (no verdict). **Utility** is the mean reward: +1 for a "
          "correct verdict, -1 for a wrong one, 0 for abstaining, minus 0.05 per tool call. **Accuracy** counts an abstention as wrong. "
          "`mean [low, high]` is a 95% interval from resampling claim groups; a paired difference whose interval excludes zero is a real "
          "difference. **Parameters** are given as total / active per token.", "",
          "Policies: " + "; ".join(f"**{POLICY_LABELS[k]}** = {d}" for k, d in (
              ("always_answer", "the first verdict, never checked"), ("heuristic", "confidence thresholds tuned on validation"),
              ("ours", "3 x 6 softmax trained with REINFORCE on the six pre-check signals"),
              ("epistemic_eu", "answers when its calibrated P(correct) clears w/(1+w), checks when the predicted gain exceeds the cost, and after checking commits / keeps the prior / abstains by expected reward"),
              ("epistemic_rl", "two REINFORCE heads, before and after checking"), ("oracle", "knows the gold label; upper bound"))) + "."]

    # ---- table 1 -------------------------------------------------------------------------------------
    L += ["", "### Table 1. The model's first answer, and the most any controller could add", "",
          "*How often each model's first verdict is right, and how often a perfect chooser between answering, checking and abstaining "
          "would be right. The gap is the headroom for any controller.*", "",
          "| Model | Parameters (total / active) | First-answer accuracy % | Oracle accuracy % | Headroom (percentage points) |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {r['params_total'] / 1e9:.1f}B / {r['params_active'] / 1e9:.1f}B | {_fx(r['answer_acc'])} | "
                 f"{100 * r['oracle_acc']['mean']:.1f} | {100 * (r['oracle_acc']['mean'] - r['answer_acc']['mean']):.1f} |")
    L += ["", "- **First-answer accuracy**: the *Always answer* policy. **Oracle accuracy**: best of answer / raw check / abstain per case, "
          "with the gold label known. Brackets: 95% interval over claim groups."]

    # ---- table 2 -------------------------------------------------------------------------------------
    L += ["", "### Table 2. Does checking add information? Raw checker vs disciplined checker", "",
          "*Accuracy of the checked verdict when every case is checked, and how often the check turns a wrong first answer right "
          "(fixes) or a right one wrong (breaks). The raw checker returns whatever the tool loop ends with; the disciplined checker "
          "requires a cited sentence the agent actually read and, where a blind-judge record exists, uses an evidence-only reading of "
          "that sentence instead of the checker's self-assessment.*", "",
          "| Model | Raw checker accuracy % | Raw: fixes % | Raw: breaks % | Disciplined checker accuracy % | Disciplined: fixes % | Disciplined: breaks % | Blind judge available for % of cases |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {_fx(r['verify_acc'])} | {_fx(r['verify_wrong_to_right'])} | {_fx(r['verify_right_to_wrong'])} | "
                 f"{_fx(r['disciplined_check_acc'])} | {_fx(r['disciplined_fixes'])} | {_fx(r['disciplined_breaks'])} | {100 * r['judged_share']:.0f} |")
    L += ["", "- **Fixes / breaks**: share of all cases. A checker only adds information when fixes exceed breaks. The oracle's gain "
          "over the first answer equals the fixes."]

    # ---- table 3 -------------------------------------------------------------------------------------
    pol_cols = [("always_answer", "utility"), ("heuristic", "utility"), ("ours", "utility"), ("epistemic_eu", None), ("epistemic_rl", None), ("oracle", "utility")]
    L += ["", "### Table 3. Utility of each policy", "",
          "*Mean reward per case under w = 1, c = 0.05 (higher is better). Use this table to rank policies; Table 4 says which "
          "differences are real.*", "",
          "| Model | " + " | ".join(POLICY_LABELS[k] for k, _ in pol_cols) + " |", "|---|" + "---|" * len(pol_cols)]
    for r in rows:
        cells = []
        for k, kind in pol_cols:
            d = r["utility"].get(k) if kind else r["epistemic"].get(k, {}).get("utility")
            cells.append(fmt_ci(d) if d else "not run")
        L.append(f"| {r['label']} | " + " | ".join(cells) + " |")

    # ---- table 4 -------------------------------------------------------------------------------------
    L += ["", "### Table 4. Are the gains real? Paired utility differences", "",
          "*Each cell is a difference in utility between two policies on the same cases, with a 95% interval and a one-word verdict. "
          "Positive favours the first-named policy.*", "",
          "| Model | Learned controller minus always answer | Credence-based agent minus always answer | Credence-based agent minus learned controller | Two-stage agent minus always answer |",
          "|---|---|---|---|---|"]
    for r in rows:
        eu, rl = r["epistemic"].get("epistemic_eu"), r["epistemic"].get("epistemic_rl")
        cell = lambda d: f"{fmt_ci(d)} ({_real(d)})" if d else "not run"  # noqa: E731
        L.append(f"| {r['label']} | {cell(r['gain_vs_always_answer'])} | {cell(eu and eu['gain_vs_always_answer'])} | "
                 f"{cell(eu and eu['gain_vs_ours'])} | {cell(rl and rl['gain_vs_always_answer'])} |")

    # ---- table 5 -------------------------------------------------------------------------------------
    if any(r["epistemic"] for r in rows):
        L += ["", "### Table 5. What the epistemic agents do", "",
              "*How often each epistemic agent checks, how often the check contradicts its first answer, and the quality and "
              "calibration of what it delivers.*", "",
              "| Model | Agent | Checked % | Contested % (among checked) | Accuracy % | Harmful answers % | Answered % | Calibration error of reported credence |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            for pol in ("epistemic_eu", "epistemic_rl"):
                e = r["epistemic"].get(pol)
                if not e:
                    continue
                L.append(f"| {r['label']} | {POLICY_LABELS[pol]} | {100 * (e['check_rate'] or 0):.0f} | {_fx(e['contested_rate'])} | "
                         f"{_fx(e['accuracy'])} | {_fx(e['harmful'])} | {_fx(e['coverage'])} | "
                         f"{'n/a' if e['credence_ece'] is None else f'{e['credence_ece']:.3f}'} |")
        L += ["", glossary(["check_rate", "contested", "accuracy", "harmful", "coverage", "ece_cal"])]

    # ---- table 6 -------------------------------------------------------------------------------------
    has_tw = all(r["integrity"]["always_answer"].get("grounding_flip_rate") is not None for r in rows)
    L += ["", "### Table 6. Calibration and integrity of the first answer", "",
          "*How honest the model's own confidence is, whether the raw checker invents citations, and (in-domain only) whether the "
          "first answer notices when its evidence is removed.*", "",
          "| Model | " + COLS["ece_raw"][0] + " | Calibration error after the fitted calibrator | " + COLS["fabrication"][0] + " % (raw checker) | "
          + (COLS["grounding_flip"][0] + " % | " + COLS["stubborn"][0] + " % | " if has_tw else "") + "Two samples agree % |",
          "|---|---|---|---|" + ("---|---|" if has_tw else "") + "---|"]
    for r in rows:
        c, i = r["calibration"], r["integrity"]
        L.append(f"| {r['label']} | {c['ece_raw_answer']:.3f} | {c['ece_cal_ours']:.3f} | {_fx(i['always_verify']['fabrication_rate'])} | "
                 + (f"{_fx(i['always_answer']['grounding_flip_rate'])} | {_fx(i['always_answer']['stubborn_rate'])} | " if has_tw else "")
                 + f"{100 * r['agree_rate']:.0f} |")
    L += ["", glossary(["ece_raw", "fabrication"] + (["grounding_flip", "stubborn"] if has_tw else [])),
          "- **Two samples agree**: how often the two provisional samples (same prompt, different seeds) give the same verdict; "
          "when this is near 100% the disagreement signal carries no information."]

    # ---- reading -------------------------------------------------------------------------------------
    def mono(key, sub=None):
        v = [(r[key] if sub is None else r[key][sub])["mean"] for r in rows]
        return "decreases" if all(a > b for a, b in zip(v, v[1:])) else "increases" if all(a < b for a, b in zip(v, v[1:])) else "is not monotonic"
    L += ["", "## Reading the size trend", "",
          f"- Ordered by total parameters ({', '.join(labels)}): first-answer accuracy {mono('answer_acc')}, the share of cases the raw "
          f"checker fixes {mono('verify_wrong_to_right')}, the share it breaks {mono('verify_right_to_wrong')}, and the learned "
          f"controller's gain over always answering {mono('gain_vs_always_answer')}.",
          "- A gain interval that excludes zero is a real effect for that model; overlapping intervals across models mean the size "
          "trend itself is not established."]
    return "\n".join(L) + "\n"


def figures(rows: list[dict], out: Path, split: str) -> list[Path]:
    from eval import figures as F

    plt = F.plt
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["params_total"])
    x = np.array([r["params_total"] / 1e9 for r in rows])
    labels = [r["label"] for r in rows]
    paths = []

    def err(ds):
        m = np.array([d["mean"] for d in ds])
        return m, np.vstack([m - np.array([d["lo"] for d in ds]), np.array([d["hi"] for d in ds]) - m])

    # 1 accuracy vs size
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    F._ax(ax, f"Accuracy by model size ({split})", "Total parameters (billions, log scale)", "Accuracy", grid_axis="both")
    for i, (name, key) in enumerate([("answer", "answer_acc"), ("verify", "verify_acc")]):
        m, e = err([r[key] for r in rows])
        ax.errorbar(x, m, yerr=e, color=F.SERIES[i], linewidth=2, marker="o", markersize=6, markeredgecolor=F.SURFACE,
                    capsize=3, label=name)
    ax.plot(x, [r["oracle_acc"]["mean"] for r in rows], color=F.SERIES[2], linewidth=2, marker="o", markersize=6,
            markeredgecolor=F.SURFACE, label="oracle (best of answer / verify / abstain)")
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{lab}\n{v:.1f}B" for lab, v in zip(labels, x)])
    ax.minorticks_off()
    ax.legend(loc="lower right")
    paths.append(F._save(fig, out / "1_accuracy_by_size.png", "How to read: x = model size (total parameters, log scale), y = accuracy on the same test cases. Blue = the model's first answer, orange = the raw checker run on every case, green = a perfect chooser. Bars are 95% intervals."))

    # 2 value of checking
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    F._ax(ax, "What verification does, by model", "", "Share of cases")
    idx = np.arange(len(rows))
    for k, (name, key) in enumerate([("verify fixes (wrong to right)", "verify_wrong_to_right"),
                                     ("verify breaks (right to wrong)", "verify_right_to_wrong")]):
        m, e = err([r[key] for r in rows])
        ax.bar(idx + (k - 0.5) * 0.34, m, width=0.32, color=F.SERIES[k], yerr=e, capsize=3, edgecolor=F.SURFACE,
               error_kw={"ecolor": F.INK2, "elinewidth": 1}, label=name)
    ax.set_xticks(idx, labels)
    ax.legend(loc="upper right")
    paths.append(F._save(fig, out / "2_value_of_checking.png", "How to read: for each model, the share of cases the raw checker turned from wrong to right (blue) and from right to wrong (orange). Checking only adds information when blue exceeds orange."))

    # 3 controller gain
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    F._ax(ax, "Learned controller vs baselines (utility difference)", "", "Ours minus baseline (mean reward)")
    ax.axhline(0, color=F.AXIS, linewidth=1)
    for k, (name, key) in enumerate([("vs always answer", "gain_vs_always_answer"), ("vs tuned heuristic", "gain_vs_heuristic")]):
        m, e = err([r[key] for r in rows])
        ax.errorbar(idx + (k - 0.5) * 0.2, m, yerr=e, fmt="o", color=F.SERIES[k], markersize=7, markeredgecolor=F.SURFACE,
                    capsize=4, label=name)
    ax.set_xticks(idx, labels)
    ax.legend(loc="best")
    paths.append(F._save(fig, out / "3_controller_gain.png", "How to read: utility of the learned controller minus a baseline, on the same cases, with 95% intervals. Above the zero line = the controller does better; an interval that does not touch zero is a real difference."))

    # 5 epistemic agent vs baselines (utility)
    if any(r["epistemic"] for r in rows):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        F._ax(ax, f"Utility by policy and model ({split})", "", "Mean reward (w=1, c=0.05)")
        series = [("always answer", lambda r: r["utility"]["always_answer"]), ("ours (3-action RL)", lambda r: r["utility"]["ours"]),
                  ("epistemic agent (credence-based)", lambda r: r["epistemic"].get("epistemic_eu", {}).get("utility")),
                  ("oracle", lambda r: r["utility"]["oracle"])]
        wd = 0.2
        for k, (name, get) in enumerate(series):
            ds = [get(r) for r in rows]
            m = np.array([d["mean"] if d else np.nan for d in ds])
            lo = np.array([d["lo"] if d else np.nan for d in ds]); hi = np.array([d["hi"] if d else np.nan for d in ds])
            ax.bar(idx + (k - 1.5) * wd, m, width=wd - 0.02, color=F.SERIES[k], edgecolor=F.SURFACE, label=name,
                   yerr=np.vstack([m - lo, hi - m]), capsize=3, error_kw={"ecolor": F.INK2, "elinewidth": 1})
        ax.axhline(0, color=F.AXIS, linewidth=1)
        ax.set_xticks(idx, labels)
        ax.legend(loc="upper left", fontsize=7.5)
        paths.append(F._save(fig, out / "5_epistemic_utility.png", "How to read: mean reward per case (utility) for four policies per model, with 95% intervals. The oracle knows the gold label and is the ceiling; the question is how close each decision policy gets to it."))

    # 4 calibration
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    F._ax(ax, "Calibration error by model", "", "Expected calibration error (lower is better)")
    for k, (name, key) in enumerate([("raw verbalised confidence", "ece_raw_answer"), ("calibrated P(correct)", "ece_cal_ours")]):
        ax.bar(idx + (k - 0.5) * 0.34, [r["calibration"][key] for r in rows], width=0.32, color=F.SERIES[k],
               edgecolor=F.SURFACE, label=name)
    ax.set_xticks(idx, labels)
    ax.legend(loc="upper right")
    paths.append(F._save(fig, out / "4_calibration_by_model.png", "How to read: expected calibration error, lower is better. Blue = the model's own stated confidence; orange = the probability the deployed agent reports after calibration."))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(M.ORDER))
    ap.add_argument("--art", default=str(C.ART_DIR))
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--split", default="test_id", choices=["test_id"] + C.OOD_SPLITS)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args(argv)
    art = Path(args.art)
    rows, missing = [], []
    for key in args.models.split(","):
        r = model_summary(key, art, Path(args.cases_dir), args.split, args.n_boot)
        (rows.append(r) if r else missing.append(key))
    if not rows:
        raise SystemExit(f"no evaluated models found under {art}/models (missing: {missing})")
    out = art / "models"
    suffix = "" if args.split == "test_id" else ("_ood" if args.split == "test_ood" else "_" + args.split.replace("test_", ""))
    md = render(rows, args.split) + (f"\nNot yet evaluated: {', '.join(missing)}\n" if missing else "")
    (out / f"comparison{suffix}.md").write_text(md, encoding="utf-8")
    save_json(out / f"comparison{suffix}.json", rows)
    figs = figures(rows, out / f"figs_compare{suffix}", args.split)
    print(md)
    print(f"[compare] wrote {out / f'comparison{suffix}.md'} and {len(figs)} figures; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
