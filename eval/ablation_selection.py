"""Ablation for idea 2: train the controllers only on decision-relevant cases.

    python -m eval.ablation_selection

A case is decision-relevant when the two committing actions disagree in outcome (the first answer is right and the
disciplined check wrong, or the reverse): only those cases carry information about *which* action to take.  Cases
where both are right or both wrong only inform the abstain decision.  We compare training on all cases (the
default), on the decision-relevant subset plus a 25% sample of the rest, and with decision-relevant cases
up-weighted 3x, for the 3-action controller and the two-stage epistemic controller; test utility on SciFact test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import config as C
from common import models as M
from controller.epistemic import EpiData, argmax_two_stage, decisions_reward, train_two_stage
from controller.train import argmax_reward, train


def variants(relevant: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = len(relevant)
    idx_all = np.arange(n)
    rest = idx_all[~relevant]
    keep_rest = rng.choice(rest, size=max(1, len(rest) // 4), replace=False) if len(rest) else np.array([], int)
    return {"all cases": idx_all,
            "relevant + 25% of the rest": np.sort(np.concatenate([idx_all[relevant], keep_rest])),
            "relevant up-weighted 3x": np.sort(np.concatenate([idx_all, idx_all[relevant], idx_all[relevant]]))}


def run(key: str, art: Path, cases_dir: Path, seeds: list[int], w=1.0, c=0.05) -> dict:
    data = EpiData(cases_dir, art / M.get(key)["cache"], splits=("train", "val", "test_id"))
    tr = data.scored["train"]
    relevant = np.array([s.dprov_correct != s.dver_correct for s in tr])
    Ra, R2 = data.rewards("train", w, c)
    Rav, R2v = data.rewards("val", w, c)
    Rat, R2t = data.rewards("test_id", w, c)
    # 3-action rewards (answer / raw verify / abstain) for the original controller
    from scorer.score import reward
    R3 = {s: np.array([[reward(x, a, w, c) for a in range(3)] for x in data.scored[s]]) for s in ("train", "val", "test_id")}
    out = {"n_train": len(tr), "n_relevant": int(relevant.sum()), "variants": {}}
    for name, idx in variants(relevant, np.random.default_rng(0)).items():
        res3, res2 = [], []
        for seed in seeds:
            r3 = train(data.X1["train"][idx], R3["train"][idx], seed, X_val=data.X1["val"], R_val=R3["val"], epochs=300, patience=50)
            res3.append(argmax_reward(r3["W"], data.X1["test_id"], R3["test_id"]))
            r2 = train_two_stage(data.X1["train"][idx], data.X2["train"][idx], Ra[idx], R2[idx], seed,
                                 val=(data.X1["val"], data.X2["val"], Rav, R2v), epochs=300, patience=50)
            res2.append(decisions_reward(argmax_two_stage(r2["W1"], r2["W2"], data.X1["test_id"], data.X2["test_id"]), Rat, R2t))
        out["variants"][name] = {"n_rows": int(len(idx)), "controller_3action": (float(np.mean(res3)), float(np.std(res3))),
                                 "two_stage": (float(np.mean(res2)), float(np.std(res2)))}
    return out


def render(results: dict) -> str:
    L = ["# Ablation: training only on decision-relevant cases", "",
         "*Idea: cases where answering and checking lead to the same outcome say nothing about which to choose, so training on the "
         "cases where they differ (or weighting those up) might teach the controller faster. Test utility on SciFact test, mean "
         "(standard deviation) over training seeds. 'All cases' is the default used everywhere else.*", "",
         "| Model | Training set | Rows | Decision-relevant cases | Utility, 3-action controller | Utility, two-stage epistemic controller |",
         "|---|---|---|---|---|---|"]
    for key, r in results.items():
        for name, v in r["variants"].items():
            L.append(f"| {M.get(key)['label']} | {name} | {v['n_rows']} | {r['n_relevant']} of {r['n_train']} | "
                     f"{v['controller_3action'][0]:.3f} ({v['controller_3action'][1]:.3f}) | {v['two_stage'][0]:.3f} ({v['two_stage'][1]:.3f}) |")
    L += ["", "- **Decision-relevant**: first answer and disciplined check disagree in correctness.",
          "- Reading: if the subset rows are not better than 'all cases', selection does not help here; re-weighting changes the "
          "effective base rates the controller sees, which can bias it toward checking."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(M.ORDER))
    ap.add_argument("--art", default=str(C.ART_DIR))
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args(argv)
    results = {}
    for key in args.models.split(","):
        if (Path(args.art) / M.get(key)["cache"] / "train.jsonl").exists():
            results[key] = run(key, Path(args.art), Path(args.cases_dir), list(range(args.seeds)))
            print(f"[ablation] {key} done")
    md = render(results)
    (Path(args.art) / "models" / "ablation_selection.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
