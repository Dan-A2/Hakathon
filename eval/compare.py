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
    return out


def render(rows: list[dict], split: str) -> str:
    pct = lambda d: fmt_ci(d, pct=True)  # noqa: E731
    L = [f"# Scaling: does checking pay off more for weaker models? ({split})", "",
         "Each row is one frozen model run through the same cases, prompts, tools and K. Intervals are 95% cluster "
         "bootstraps over claim groups; gains are paired. Gemma 4 26B-A4B is a mixture of experts with about 4B "
         "parameters active per token, so it is listed by both total and active size. Three models from two families "
         "cannot separate size from family or training recipe; read the trend as descriptive.", "",
         "## Capability and the value of checking", "",
         "| Model | Params (total / active) | Answer acc % | Verify acc % | Oracle acc % | Verify fixes % | Verify breaks % |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {r['params_total']/1e9:.1f}B / {r['params_active']/1e9:.1f}B | {pct(r['answer_acc'])} | "
                 f"{pct(r['verify_acc'])} | {100*r['oracle_acc']['mean']:.1f} | "
                 f"{pct(r['verify_wrong_to_right'])} | {pct(r['verify_right_to_wrong'])} |")
    L += ["", "Verify fixes and breaks are the shares of cases that verification turns wrong-to-right and right-to-wrong; "
          "the oracle's gain over answering equals the fixes.", "",
          "## Does the learned controller help?", "",
          "| Model | Utility: always answer | Utility: tuned heuristic | Utility: ours | Ours minus always answer | Ours minus heuristic | Ours action mix (answer / verify / abstain) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        u = r["utility"]
        mix = r["ours_mix"]
        L.append(f"| {r['label']} | {fmt_ci(u['always_answer'])} | {fmt_ci(u['heuristic'])} | {fmt_ci(u['ours'])} | "
                 f"{fmt_ci(r['gain_vs_always_answer'])} | {fmt_ci(r['gain_vs_heuristic'])} | "
                 f"{100*mix['answer']:.0f} / {100*mix['verify']:.0f} / {100*mix['abstain']:.0f} % |")
    L += ["", "## Calibration and integrity", "",
          "| Model | ECE raw (answer) | ECE calibrated (ours) | Fabrication % (always verify) | Grounding flip % (always answer) | Stubborn % (always answer) | No gold abstract cited % (always verify) | Samples agree % |",
          "|---|---|---|---|---|---|---|---|"]
    f = lambda v: "n/a" if v is None else f"{100*v:.1f}"  # noqa: E731
    for r in rows:
        c, i = r["calibration"], r["integrity"]
        L.append(f"| {r['label']} | {c['ece_raw_answer']:.3f} | {c['ece_cal_ours']:.3f} | {f(i['always_verify']['fabrication_rate'])} | "
                 f"{f(i['always_answer']['grounding_flip_rate'])} | {f(i['always_answer']['stubborn_rate'])} | "
                 f"{f(i['always_verify']['wrong_reason_rate_doc'])} | {100*r['agree_rate']:.0f} |")
    # automatic reading of the hypothesis
    by_size = sorted(rows, key=lambda r: r["params_total"])
    if len(by_size) >= 2:
        def mono(key, sub=None):
            v = [(r[key] if sub is None else r[key][sub])["mean"] for r in by_size]
            return "decreases" if all(a > b for a, b in zip(v, v[1:])) else "increases" if all(a < b for a, b in zip(v, v[1:])) else "is not monotonic"
        L += ["", "## Reading", "",
              f"- Ordered by total parameters ({', '.join(r['label'] for r in by_size)}): answer accuracy {mono('answer_acc')}, "
              f"the share verification fixes {mono('verify_wrong_to_right')}, "
              f"and the controller's gain over always answering {mono('gain_vs_always_answer')}.",
              "- A gain interval that excludes zero is a real effect for that model; overlapping intervals across models "
              "mean the size trend itself is not established."]
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
    paths.append(F._save(fig, out / "1_accuracy_by_size.png"))

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
    paths.append(F._save(fig, out / "2_value_of_checking.png"))

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
    paths.append(F._save(fig, out / "3_controller_gain.png"))

    # 4 calibration
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    F._ax(ax, "Calibration error by model", "", "Expected calibration error (lower is better)")
    for k, (name, key) in enumerate([("raw verbalised confidence", "ece_raw_answer"), ("calibrated P(correct)", "ece_cal_ours")]):
        ax.bar(idx + (k - 0.5) * 0.34, [r["calibration"][key] for r in rows], width=0.32, color=F.SERIES[k],
               edgecolor=F.SURFACE, label=name)
    ax.set_xticks(idx, labels)
    ax.legend(loc="upper right")
    paths.append(F._save(fig, out / "4_calibration_by_model.png"))
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(M.ORDER))
    ap.add_argument("--art", default=str(C.ART_DIR))
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--split", default="test_id", choices=["test_id", "test_ood"])
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
    suffix = "" if args.split == "test_id" else "_ood"
    md = render(rows, args.split) + (f"\nNot yet evaluated: {', '.join(missing)}\n" if missing else "")
    (out / f"comparison{suffix}.md").write_text(md, encoding="utf-8")
    save_json(out / f"comparison{suffix}.json", rows)
    figs = figures(rows, out / f"figs_compare{suffix}", args.split)
    print(md)
    print(f"[compare] wrote {out / f'comparison{suffix}.md'} and {len(figs)} figures; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
