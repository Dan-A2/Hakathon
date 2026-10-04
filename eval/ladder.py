"""The evidence ladder: does the agent's confidence track the information actually present, or only the topic?

    python -m eval.ladder [--models ...]

Three levels of label-information for the same claim, with near-constant topical similarity:
  L0  full evidence        the SciFact base case: the gold abstract is retrievable and shown (label supported / refuted)
  L1  rationale removed    the gold abstract is still shown, but its rationale sentences are deleted (test_ladder split):
                           same topic, the decisive sentence is gone -> the justified verdict is insufficient_evidence
  L2  abstract removed     the existing twin: the gold abstract is removed from the record store entirely
plus a contrastive pair test from SciFact itself: pairs of claims that cite the same abstract where one is supported
and the other refuted (VitaminC-style contrastive evidence, but natural).  An agent that reads evidence must give
the two claims different verdicts; one that answers from topic or memory gives the same.

Information-theoretic reading: the mutual information between what the agent sees and the gold label is high at L0,
zero at L1 and L2; lexical overlap (what a shortcut learner uses) barely changes between L0 and L1.
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import config as C
from common import models as M
from common.io import read_jsonl, save_json
from controller.epistemic import EUController
from controller.policy import feature_matrix
from scorer.score import Scored, load_scored, outcome

LEVELS = [("L0", "full evidence (base case)"), ("L1", "rationale removed (topic kept)"), ("L2", "abstract removed (twin)")]


def _eu_decisions(eu: EUController, scored: list[Scored]):
    fill1 = {f: float(m) for f, m in zip(eu.std1.feature_names, eu.std1.mu)}
    fill2 = {f: float(m) for f, m in zip(eu.std2.feature_names, eu.std2.mu)}
    X1 = eu.std1.transform(feature_matrix([s.x for s in scored], eu.std1.feature_names, fill1))
    X2 = eu.std2.transform(feature_matrix([s.z for s in scored], eu.std2.feature_names, fill2))
    dec, cred, _ = eu.decide_batch(X1, X2)
    return dec, cred


def level_stats(scored: list[Scored], dec: np.ndarray, cred: np.ndarray) -> dict:
    """For the three policies: share committing to supported/refuted, accuracy, mean credence when committing."""
    out = {}
    n = len(scored)
    for name, acts in (("first_answer", np.full(n, C.ANSWER)), ("disciplined_check", np.full(n, C.CHECK_COMMIT)), ("credence_agent", dec)):
        outs = [outcome(s, int(a)) for s, a in zip(scored, acts)]
        commit_sr = np.array([o["committed"] and o["verdict"] in C.COMMIT_LABELS for o in outs])
        says_insuff = np.array([o["committed"] and o["verdict"] == "insufficient_evidence" for o in outs])
        correct = np.array([o["correct"] for o in outs], float)
        row = {"n": n, "commit_sr": float(commit_sr.mean()), "says_insufficient": float(says_insuff.mean()),
               "abstain": float(np.mean([not o["committed"] for o in outs])), "accuracy": float(correct.mean())}
        if name == "credence_agent":
            cr = cred[commit_sr]
            row["mean_credence_when_committing"] = float(np.nanmean(cr)) if commit_sr.any() and not np.all(np.isnan(cr)) else float("nan")
        out[name] = row
    return out


def ladder_for_model(key: str, art: Path, cases_dir: Path) -> dict | None:
    m = M.get(key)
    cache_dir = art / m["cache"]
    eu_path = art / "models" / key / "epistemic_eu.npz"
    if not eu_path.exists() or not (cache_dir / "test_id.jsonl").exists():
        return None
    eu = EUController.load(eu_path)
    test = load_scored("test_id", cases_dir, cache_dir)
    by_id = {s.case_id: s for s in test}
    gold = {g["case_id"]: g for g in read_jsonl(cases_dir / "gold" / "test_id.jsonl")}
    base_sr = [s for s in test if s.variant == "base" and s.label in C.COMMIT_LABELS]
    twins = {s.parent_id: s for s in test if s.variant == "ablated"}
    levels = {"L0": base_sr, "L2": [twins[s.case_id] for s in base_sr if s.case_id in twins]}
    ladder_rows = []
    if (cache_dir / "test_ladder.jsonl").exists():
        ladder_rows = load_scored("test_ladder", cases_dir, cache_dir)
        levels["L1"] = ladder_rows
    res = {"levels": {}, "n_claims": len(base_sr), "has_L1": bool(ladder_rows)}
    decs = {}
    for lvl, sc in levels.items():
        if not sc:
            continue
        dec, cred = _eu_decisions(eu, sc)
        decs[lvl] = {s.case_id: (int(d), float(c) if c == c else None) for s, d, c in zip(sc, dec, cred)}
        res["levels"][lvl] = level_stats(sc, dec, cred)
    # per-claim monotonicity of the credence agent's commitment across available levels (needs all three)
    if ladder_rows:
        l1 = {s.parent_id: s for s in ladder_rows}
        mono_commit, mono_cred, total = 0, 0, 0
        for s in base_sr:
            if s.case_id not in twins or s.case_id not in l1:
                continue
            total += 1
            trip = [decs["L0"][s.case_id], decs["L1"][l1[s.case_id].case_id], decs["L2"][twins[s.case_id].case_id]]
            com = [1.0 if (d in (C.D_ANSWER, C.CHECK_COMMIT, C.CHECK_KEEP) and (outcome(x, d)["verdict"] in C.COMMIT_LABELS)) else 0.0
                   for (d, _), x in zip(trip, [s, l1[s.case_id], twins[s.case_id]])]
            if com[0] >= com[1] >= com[2]:
                mono_commit += 1
            cr = [c if (c is not None and k) else 0.0 for (d, c), k in zip(trip, com)]
            if cr[0] >= cr[1] >= cr[2]:
                mono_cred += 1
        res["monotonic_commit_share"] = mono_commit / total if total else float("nan")
        res["monotonic_credence_share"] = mono_cred / total if total else float("nan")
        res["n_triples"] = total
    # contrastive claim pairs: same gold abstract, opposite labels (from test_id; train adds more pairs for the model alone)
    res["pairs"] = {}
    for split in ("test_id", "train"):
        if not (cache_dir / f"{split}.jsonl").exists():
            continue
        sc = {s.case_id: s for s in load_scored(split, cases_dir, cache_dir)}
        g = {x["case_id"]: x for x in read_jsonl(cases_dir / "gold" / f"{split}.jsonl")}
        by_doc = defaultdict(list)
        for cid, gg in g.items():
            if gg["label"] in C.COMMIT_LABELS and cid in sc and not cid.endswith("-abl"):
                for d in gg["gold_doc_ids"]:
                    by_doc[str(d)].append(cid)
        pairs = {tuple(sorted((a, b))) for ids in by_doc.values() for a, b in itertools.combinations(ids, 2) if g[a]["label"] != g[b]["label"]}
        if not pairs:
            continue
        stats = {}
        dec_all = None
        if split == "test_id":
            ids = sorted({i for p in pairs for i in p})
            d, _ = _eu_decisions(eu, [sc[i] for i in ids])
            dec_all = dict(zip(ids, d))
        for name in ("first_answer", "disciplined_check", "credence_agent"):
            if name == "credence_agent" and dec_all is None:
                continue
            diff = both = 0
            for a, b in pairs:
                acts = {i: (dec_all[i] if name == "credence_agent" else (C.ANSWER if name == "first_answer" else C.CHECK_COMMIT)) for i in (a, b)}
                oa, ob = outcome(sc[a], int(acts[a])), outcome(sc[b], int(acts[b]))
                diff += oa["verdict"] != ob["verdict"]
                both += oa["correct"] and ob["correct"]
            stats[name] = {"n_pairs": len(pairs), "different_verdicts": diff / len(pairs), "both_correct": both / len(pairs)}
        res["pairs"][split] = stats
    return res


def render(results: dict) -> str:
    L = ["# The evidence ladder: does confidence track evidence, or topic?", "", "## What this report says", ""]
    for key, r in results.items():
        lab = M.get(key)["label"]
        lv = r["levels"]
        fa = {k: v["first_answer"] for k, v in lv.items()}
        ca = {k: v["credence_agent"] for k, v in lv.items()}
        parts = [f"the model alone commits to supported/refuted on {100 * fa['L0']['commit_sr']:.0f}% of claims with full evidence"]
        if "L1" in fa:
            parts.append(f"{100 * fa['L1']['commit_sr']:.0f}% when the decisive sentence is deleted but the abstract is still shown")
        parts.append(f"{100 * fa['L2']['commit_sr']:.0f}% when the abstract is removed")
        L.append(f"- **{lab}**: " + ", ".join(parts) + f". The credence-based agent: {100 * ca['L0']['commit_sr']:.0f}%"
                 + (f", {100 * ca['L1']['commit_sr']:.0f}%" if "L1" in ca else "") + f", {100 * ca['L2']['commit_sr']:.0f}%"
                 + (f"; its commitment is non-increasing across the three levels for {100 * r['monotonic_commit_share']:.0f}% of claims "
                    f"({r['n_triples']} triples)." if r.get("has_L1") else "."))
        for split, st in r["pairs"].items():
            fa_p = st["first_answer"]
            L.append(f"  - Contrastive pairs on {split} ({fa_p['n_pairs']} pairs of opposite claims over the same abstract): the model alone gives "
                     f"the two claims different verdicts in {100 * fa_p['different_verdicts']:.0f}% of pairs and gets both right in "
                     f"{100 * fa_p['both_correct']:.0f}%" + (f"; the credence-based agent {100 * st['credence_agent']['different_verdicts']:.0f}% / "
                                                              f"{100 * st['credence_agent']['both_correct']:.0f}%." if "credence_agent" in st else "."))
    L += ["", "## How to read this report", "",
          "Each SciFact test claim that is supported or refuted appears at up to three levels. **L0** is the normal case: the gold "
          "abstract is retrievable and shown. **L1** shows the same abstract with its rationale sentences deleted: the topic is "
          "intact, the decisive information is gone, and the only justified verdict is 'insufficient evidence'. **L2** removes the "
          "abstract from the record store entirely. An agent whose confidence tracks evidence should commit to a verdict at L0 and "
          "decline at L1 and L2; an agent fooled by topical similarity still commits at L1. The **contrastive pairs** are two "
          "real SciFact claims citing the same abstract with opposite labels: reading the evidence forces different verdicts.", "",
          "- **Commits to S/R**: share of cases where a supported or refuted verdict was delivered.",
          "- **Says insufficient**: share answering 'insufficient evidence'. **Abstains**: no verdict (epistemic agent only).",
          "- **Accuracy**: correct share (at L1 and L2 the correct answer is 'insufficient evidence').",
          "- **Mean credence when committing**: the probability the credence-based agent attaches to the S/R verdicts it delivers.",
          "- **Monotonic**: share of claims whose commitment (and credence) does not increase from L0 to L1 to L2."]
    L += ["", "### Table 1. Commitment and accuracy at each evidence level", "",
          "*Rows: model x policy. Columns: the three evidence levels. The L1 column is empty until the ladder cache has been built for "
          "that model (`scripts/run_models.sh <model> extra`).*", "",
          "| Model | Policy | L0 commits to S/R % | L0 accuracy % | L1 commits to S/R % | L1 says insufficient % | L1 accuracy % | "
          "L2 commits to S/R % | L2 says insufficient % | L2 accuracy % | Credence when committing L0 / L1 / L2 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    pretty = {"first_answer": "Model alone (first answer)", "disciplined_check": "Disciplined checker, every case", "credence_agent": "Epistemic agent, credence-based"}
    for key, r in results.items():
        lab = M.get(key)["label"]
        for pol in ("first_answer", "disciplined_check", "credence_agent"):
            lv = {k: v[pol] for k, v in r["levels"].items()}
            l1 = lv.get("L1")
            cr = " / ".join(f"{lv[k]['mean_credence_when_committing']:.2f}" if k in lv and "mean_credence_when_committing" in lv[k] and lv[k]["mean_credence_when_committing"] == lv[k]["mean_credence_when_committing"] else "-" for k in ("L0", "L1", "L2")) if pol == "credence_agent" else "n/a"
            L.append(f"| {lab} | {pretty[pol]} | {100 * lv['L0']['commit_sr']:.0f} | {100 * lv['L0']['accuracy']:.0f} | "
                     + (f"{100 * l1['commit_sr']:.0f} | {100 * l1['says_insufficient']:.0f} | {100 * l1['accuracy']:.0f} | " if l1 else "not built | | | ")
                     + f"{100 * lv['L2']['commit_sr']:.0f} | {100 * lv['L2']['says_insufficient']:.0f} | {100 * lv['L2']['accuracy']:.0f} | {cr} |")
    L += ["", "### Table 2. Contrastive pairs: same abstract, opposite claims", "",
          "*Pairs of SciFact claims that cite the same abstract, one supported and one refuted. 'Different verdicts' is the share of "
          "pairs where the agent did not give both claims the same verdict (an agent that ignores the claim-evidence relation gives "
          "the same). 'Both correct' is the share where it got both right. Train-split pairs test the frozen model, not the controller.*", "",
          "| Model | Split | Pairs | Policy | Different verdicts % | Both correct % |", "|---|---|---|---|---|---|"]
    for key, r in results.items():
        lab = M.get(key)["label"]
        for split, st in r["pairs"].items():
            for pol, v in st.items():
                L.append(f"| {lab} | {split} | {v['n_pairs']} | {pretty[pol]} | {100 * v['different_verdicts']:.0f} | {100 * v['both_correct']:.0f} |")
    L += ["", "## Reading", "",
          "- A large drop in commitment from L0 to L1 means the agent reads for the decisive sentence rather than the topic; a small drop "
          "means topical similarity is driving its confidence (the shortcut the ablated twins alone cannot expose, because removing the "
          "whole abstract also removes the topic).",
          "- Credence that falls across the levels is what calibrated uncertainty should look like; flat credence with changing accuracy is "
          "overconfidence.",
          "- Contrastive pairs separate evidence reading from memorisation: the two claims share every surface feature except their relation to the abstract."]
    return "\n".join(L) + "\n"


def figure(results: dict, out: Path) -> Path:
    from eval import figures as F

    plt = F.plt
    keys = list(results)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys) + 1, 4.2), sharey=True, squeeze=False)
    for ax, key in zip(axes[0], keys):
        r = results[key]
        F._ax(ax, M.get(key)["label"], "", "Share committing to supported / refuted" if key == keys[0] else "")
        lv = r["levels"]
        x = np.arange(3)
        for k, (pol, col) in enumerate((("first_answer", F.SERIES[0]), ("disciplined_check", F.SERIES[1]), ("credence_agent", F.SERIES[2]))):
            vals = [lv[l][pol]["commit_sr"] if l in lv else np.nan for l in ("L0", "L1", "L2")]
            ax.bar(x + (k - 1) * 0.27, vals, width=0.25, color=col, edgecolor=F.SURFACE, label={"first_answer": "model alone", "disciplined_check": "disciplined checker", "credence_agent": "credence-based agent"}[pol])
        ax.set_xticks(x, ["L0 full\nevidence", "L1 rationale\nremoved", "L2 abstract\nremoved"], fontsize=8)
        ax.set_ylim(0, 1.05)
    axes[0][-1].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Evidence ladder: commitment should fall as the decisive information is removed", x=0.01, ha="left", fontsize=11, fontweight="bold")
    return F._save(fig, out / "evidence_ladder.png",
                   "How to read: for each model, the share of claims where a supported/refuted verdict was delivered at the three evidence levels. "
                   "Full evidence (L0) warrants a verdict; at L1 the abstract is shown but its decisive sentence is gone; at L2 the abstract is "
                   "absent. Bars that stay high at L1 mean confidence is driven by topic, not evidence.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(M.ORDER))
    ap.add_argument("--art", default=str(C.ART_DIR))
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    args = ap.parse_args(argv)
    art, cases_dir = Path(args.art), Path(args.cases_dir)
    results = {}
    for key in args.models.split(","):
        r = ladder_for_model(key, art, cases_dir)
        if r:
            results[key] = r
    if not results:
        raise SystemExit("no models with a credence-based agent and a test_id cache")
    out = art / "models"
    md = render(results)
    (out / "ladder.md").write_text(md, encoding="utf-8")
    save_json(out / "ladder.json", results)
    fig = figure(results, out / "figs_ladder")
    print(md)
    print(f"[ladder] wrote {out / 'ladder.md'} and {fig}")


if __name__ == "__main__":
    main()
