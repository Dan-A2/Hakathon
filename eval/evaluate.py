"""Evaluate the seven policies on one split, with cluster-bootstrap CIs, McNemar, figures,
tables, per-case logs and case cards.

    python -m eval.evaluate --split test_id --controller art/controller.npz --figs art/figs

Everything is scored from the same counterfactual cache, so comparisons are paired.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import config as C
from common.io import ensure_dir, read_jsonl, save_json, write_jsonl
from controller.calibrate import Calibrator, brier, ece, fit_calibrator, reliability_bins
from controller.policy import Controller, feature_matrix
from controller.train import Data, argmax_actions, run_sweep
from eval import figures as F
from eval.policies import POLICY_LABELS, POLICY_ORDER, build_policies, per_case_rows
from eval.stats import cluster_bootstrap, fmt_ci, mcnemar, paired_cluster_bootstrap
from scorer.score import Scored, case_type, integrity_report, outcome, reward

BASELINES = ["always_answer", "always_verify", "always_abstain", "heuristic"]
HIGHER_BETTER = {"utility", "accuracy", "selective_accuracy"}
LOWER_BETTER = {"unnecessary_abstention", "harmful", "tool_calls", "fabrication"}


# ----------------------------------------------------------------------------- metrics
def policy_metrics(scored: list[Scored], decisions, w: float, c: float, cal: Calibrator | None,
                   n_boot: int, seed: int) -> tuple[dict, list[dict]]:
    rows = per_case_rows(scored, decisions, w, c)
    groups = [r["group_id"] for r in rows]
    committed = [r for r in rows if r["committed"]]
    gc = [r["group_id"] for r in committed]
    cb = lambda vals, g=groups: cluster_bootstrap(vals, g, n_boot, seed)  # noqa: E731
    m = {
        "utility": cb([r["reward"] for r in rows]),
        "accuracy": cb([float(r["correct"]) for r in rows]),
        "selective_accuracy": cb([float(r["correct"]) for r in committed], gc),
        "coverage": cb([float(r["committed"]) for r in rows]),
        "unnecessary_abstention": cb([float((not r["committed"]) and (r["prov_correct"] or r["ver_correct"])) for r in rows]),
        "harmful": cb([float(r["committed"] and not r["correct"]) for r in rows]),
        "tool_calls": cb([float(r["tool_calls"]) for r in rows]),
        "llm_calls": cb([float(r["llm_calls"]) for r in rows]),
        "tokens": cb([float(r["tokens"]) for r in rows]),
        "gpu_s": cb([float(r["gpu_s"]) for r in rows]),
        "fabrication": cb([float(r["fabricated"]) for r in committed], gc),
        "integrity": integrity_report(scored, {cid: d["action"] for cid, d in decisions.items()}),
        "action_mix": {a: float(np.mean([r["action"] == k for r in rows])) for k, a in enumerate(C.ACTIONS)},
    }
    # calibration on committed cases: raw verbalised confidence vs calibrated P(correct)
    if committed:
        y = np.asarray([float(r["correct"]) for r in committed])
        raw = np.asarray([r["confidence"] for r in committed], dtype=float)
        calib = {"n": len(committed), "ece_raw": ece(raw, y), "brier_raw": brier(raw, y), "bins_raw": reliability_bins(raw, y)}
        if cal is not None:
            p = cal.predict([r["x"] for r in committed], [r["action"] for r in committed])
            calib.update({"ece_cal": ece(p, y), "brier_cal": brier(p, y), "bins_cal": reliability_bins(p, y)})
            for r, pi in zip(committed, p):
                r["p_correct"] = float(pi)
        m["calibration"] = calib
    for r in rows:
        r.setdefault("p_correct", None)
    return m, rows


def fit_val_calibrator(val: list[Scored], ctrl: Controller) -> Calibrator:
    xs, outs = [], []
    for sc in val:
        for a in (C.ANSWER, C.VERIFY):
            xs.append(sc.x)
            outs.append((a, outcome(sc, a)["correct"]))
    return fit_calibrator(xs, outs, ctrl.feature_names, ctrl.std.mu, ctrl.std.sd)


# ----------------------------------------------------------------------------- tables
def _best(metrics: dict, key: str, policies: list[str]) -> str | None:
    cands = [(metrics[p][key]["mean"], p) for p in policies if p != "oracle" and metrics[p][key]["n"]]
    if not cands or key not in HIGHER_BETTER | LOWER_BETTER:
        return None
    return (max if key in HIGHER_BETTER else min)(cands)[1]


def render_headline(metrics: dict, policies: list[str], split: str) -> str:
    cols = [("utility", "Utility", False), ("accuracy", "Accuracy %", True), ("selective_accuracy", "Sel. acc. %", True),
            ("coverage", "Coverage %", True), ("unnecessary_abstention", "Unnec. abstain %", True),
            ("harmful", "Harmful %", True), ("tool_calls", "Tool calls/case", False), ("fabrication", "Fabrication %", True)]
    best = {k: _best(metrics, k, policies) for k, _, _ in cols}
    lines = [f"### Headline results on {split} (mean [95% cluster-bootstrap CI]; best non-oracle in bold)", "",
             "| Policy | " + " | ".join(h for _, h, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for p in policies:
        cells = []
        for k, _, pct in cols:
            s = fmt_ci(metrics[p][k], pct=pct)
            cells.append(f"**{s}**" if best[k] == p else s)
        lines.append(f"| {POLICY_LABELS.get(p, p)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_integrity(metrics: dict, policies: list[str]) -> str:
    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}"
    lines = ["### Integrity checks (per policy)", "",
             "| Policy | Fabrication % | Right-for-wrong-reason % | Grounding flip % | Stubborn % | Verify flips wrong->right / right->wrong | Unnec. abstain (of abstained) % |",
             "|---|---|---|---|---|---|---|"]
    for p in policies:
        i = metrics[p]["integrity"]
        lines.append(f"| {POLICY_LABELS.get(p, p)} | {pct(i['fabrication_rate'])} | {pct(i['wrong_reason_rate'])} | "
                     f"{pct(i['grounding_flip_rate'])} | {pct(i['stubborn_rate'])} | {i['verify_wrong_to_right']} / {i['verify_right_to_wrong']} "
                     f"(n={i['n_verified']}) | {pct(i['unnecessary_abstention_among_abstained'])} |")
    return "\n".join(lines)


def render_cost(metrics: dict, policies: list[str]) -> str:
    lines = ["### Cost and calibration (per case; calibration on committed cases)", "",
             "| Policy | LLM calls | Tool calls | Tokens | GPU-s (latency) | ECE raw | ECE calibrated | Brier raw | Brier calibrated |",
             "|---|---|---|---|---|---|---|---|---|"]
    for p in policies:
        m = metrics[p]
        cal = m.get("calibration", {})
        f = lambda k: "n/a" if k not in cal or cal[k] != cal[k] else f"{cal[k]:.3f}"  # noqa: E731
        lines.append(f"| {POLICY_LABELS.get(p, p)} | {m['llm_calls']['mean']:.2f} | {m['tool_calls']['mean']:.2f} | "
                     f"{m['tokens']['mean']:.0f} | {m['gpu_s']['mean']:.2f} | {f('ece_raw')} | {f('ece_cal')} | {f('brier_raw')} | {f('brier_cal')} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- curves
def risk_coverage_threshold(scored: list[Scored]) -> list[tuple[float, float]]:
    conf = np.asarray([sc.x["conf_mean"] for sc in scored])
    corr = np.asarray([sc.prov_correct for sc in scored], dtype=float)
    pts = []
    for tau in np.linspace(0, 1, 41):
        m = conf >= tau
        if m.sum() == 0:
            continue
        pts.append((float(m.mean()), float(corr[m].mean())))
    return pts


def risk_coverage_oracle(scored: list[Scored]) -> list[tuple[float, float]]:
    n = len(scored)
    capable = sum(1 for sc in scored if sc.prov_correct or sc.ver_correct)
    pts = []
    for k in np.linspace(0.05, 1.0, 20):
        n_cov = max(1, int(round(k * n)))
        pts.append((n_cov / n, min(capable, n_cov) / n_cov))
    return pts


def cov_selacc(scored: list[Scored], actions) -> tuple[float, float]:
    outs = [outcome(sc, int(a)) for sc, a in zip(scored, actions)]
    com = [o for o in outs if o["committed"]]
    return (len(com) / len(outs), float(np.mean([o["correct"] for o in com])) if com else float("nan"))


def area_under(pts: list[tuple[float, float]]) -> float:
    pts = sorted(p for p in pts if p[1] == p[1])
    if len(pts) < 2:
        return float("nan")
    xs, ys = np.asarray([p[0] for p in pts]), np.asarray([p[1] for p in pts])
    return float((getattr(np, "trapezoid", None) or np.trapz)(ys, xs))


def median_rows(sweep_rows: list[dict]) -> dict[tuple[float, float], dict]:
    """Per (w, c): the seed with the median val reward (the shipping rule)."""
    by = defaultdict(list)
    for r in sweep_rows:
        by[(r["w"], r["c"])].append(r)
    out = {}
    for k, rs in by.items():
        rs = sorted(rs, key=lambda r: r["val_reward"])
        out[k] = rs[len(rs) // 2]
    return out


# ----------------------------------------------------------------------------- cards
def case_cards(scored: list[Scored], rows_ours: list[dict], cases: dict[str, dict], w: float, c: float) -> str:
    by_id = {sc.case_id: sc for sc in scored}
    picks = {
        "A good verify (provisional wrong, verified right)": [r for r in rows_ours if r["action"] == C.VERIFY and not r["prov_correct"] and r["correct"]],
        "A good abstain (both actions would have been wrong)": [r for r in rows_ours if r["action"] == C.ABSTAIN and not r["prov_correct"] and not r["ver_correct"]],
        "A failure we own (committed to a wrong verdict)": [r for r in rows_ours if r["committed"] and not r["correct"]],
    }
    out = ["### Case cards (policy: ours)", ""]
    for title, cands in picks.items():
        out.append(f"#### {title}")
        if not cands:
            out.append("_no such case on this split_\n")
            continue
        r = sorted(cands, key=lambda r: -(r["p_correct"] or 0))[0]     # the most confident instance of each kind
        sc, case = by_id[r["case_id"]], cases.get(r["case_id"], {})
        a, b = sc.raw["a"], sc.raw["b"]
        out += [f"- **case** `{r['case_id']}` ({sc.variant}; gold: `{sc.label}`)",
                f"- **claim**: {case.get('question', '')}",
                f"- **initial evidence**: " + "; ".join(f"[{e['doc_id']}] {e['title'][:70]}" for e in case.get("initial_evidence", [])),
                f"- **sample A**: {a['verdict']} (conf {a['confidence']:.2f}, suff {a['evidence_sufficiency']:.2f}); "
                f"**sample B**: {b['verdict']} (conf {b['confidence']:.2f}, suff {b['evidence_sufficiency']:.2f})",
                f"- **signals**: " + ", ".join(f"{k}={v:.2f}" for k, v in sc.x.items() if k != 'bias' and v is not None),
                f"- **action preference**: " + (", ".join(f"{n}={p:.2f}" for n, p in zip(C.ACTIONS, r['action_probs'])) if r['action_probs'] else "n/a")
                + f" -> **{C.ACTIONS[r['action']]}**",
                f"- **delivered**: {r['verdict']} (conf {r['confidence'] if r['confidence'] is None else round(r['confidence'], 2)}); "
                f"cited {r['cited']}; P(correct) {'n/a' if r['p_correct'] is None else round(r['p_correct'], 2)}; reward {r['reward']:+.2f}",
                f"- **tool log**: " + (", ".join(f"{t['tool']}({'ok' if t['ok'] else 'err'})" for t in sc.raw['ver']['tool_log']) if r["action"] == C.VERIFY else "(no tools used)"),
                ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- main
def evaluate(split: str, controller_path: Path, cases_dir: Path, cache_dir: Path, figs_dir: Path | None, out_dir: Path,
             logs_dir: Path, sweep_path: Path | None, n_boot: int = 10000, seed: int = 0,
             shortcut_controller: Path | None = None, w: float | None = None, c: float | None = None) -> dict:
    t0 = time.time()
    ctrl = Controller.load(controller_path, check_prompt=not C_skip_hash())
    w = ctrl.w if w is None else w
    c = ctrl.c if c is None else c
    use_tok = C.TOK_PROB_FEATURE in ctrl.feature_names
    data = Data(cases_dir, cache_dir, use_tok, splits=("train", "val", split))
    scored, val = data.scored[split], data.scored["val"]
    cases = {r["case_id"]: r for r in read_jsonl(cases_dir / f"{split}.jsonl")}
    shortcut = Controller.load(shortcut_controller, check_prompt=False) if shortcut_controller else None

    cal = fit_val_calibrator(val, ctrl)
    cal.save(out_dir / "calibrator.npz")
    pols, heur = build_policies(scored, val, ctrl, w, c, shortcut)
    policies = [p for p in POLICY_ORDER if p in pols]

    metrics, rows = {}, {}
    for p in policies:
        metrics[p], rows[p] = policy_metrics(scored, pols[p], w, c, cal, n_boot, seed)

    # ---- statistics: ours vs best non-oracle baseline; spread of the argmax policy across seeds
    sweep_rows = read_jsonl(sweep_path) if sweep_path and Path(sweep_path).exists() else []
    seed_rows = [r for r in sweep_rows if abs(r["w"] - w) < 1e-9 and abs(r["c"] - c) < 1e-9]
    seeds_summary = None
    if seed_rows:
        Xs_ = data.X[split]
        per_seed = []
        for r in seed_rows:
            acts = argmax_actions(np.asarray(r["W"]), Xs_)
            outs = [outcome(s, int(a)) for s, a in zip(scored, acts)]
            per_seed.append({"seed": r["seed"], "utility": float(np.mean([reward(s, int(a), w, c) for s, a in zip(scored, acts)])),
                             "accuracy": float(np.mean([o["correct"] for o in outs])),
                             "coverage": float(np.mean([o["committed"] for o in outs]))})
        seeds_summary = {"n_seeds": len(per_seed), "per_seed": per_seed,
                         **{f"{k}_mean": float(np.mean([p[k] for p in per_seed])) for k in ("utility", "accuracy", "coverage")},
                         **{f"{k}_std": float(np.std([p[k] for p in per_seed])) for k in ("utility", "accuracy", "coverage")}}
    best_base = max((metrics[p]["utility"]["mean"], p) for p in BASELINES if p in metrics)[1]
    corr_ours = [r["correct"] for r in rows["ours"]]
    corr_base = [r["correct"] for r in rows[best_base]]
    stats = {
        "best_baseline": best_base,
        "mcnemar_ours_vs_best_baseline": mcnemar(corr_ours, corr_base),
        "utility_diff_ours_minus_best_baseline": paired_cluster_bootstrap(
            [r["reward"] for r in rows["ours"]], [r["reward"] for r in rows[best_base]],
            [r["group_id"] for r in rows["ours"]], n_boot, seed),
        "heuristic_thresholds": heur,
        "ours_across_seeds": seeds_summary,
    }

    # ---- per-case logs (spec format)
    ensure_dir(logs_dir)
    for p in policies:
        log_rows = []
        for r in rows[p]:
            log_rows.append({
                "case_id": r["case_id"], "policy": p, "split": split,
                "signals": {k: v for k, v in r["x"].items() if k != "bias"},
                "action_probs": r["action_probs"], "action": C.ACTIONS[r["action"]],
                "verdict": r["verdict"], "cited": r["cited"], "p_correct": r["p_correct"],
                "gold": r["gold"], "correct": r["correct"], "reward": r["reward"],
                "llm_calls": r["llm_calls"], "tool_calls": r["tool_calls"], "tokens": r["tokens"], "gpu_s": r["gpu_s"],
                "integrity": {"fabricated": r["fabricated"], "rationale_hit": r["rationale_hit"]},
            })
        write_jsonl(logs_dir / f"{split}_{p}.jsonl", log_rows)

    # ---- tables + cards
    ensure_dir(out_dir)
    md = [f"# Know-When-To-Check results: {split}", "",
          f"controller `{controller_path}` (w={w}, c={c}, K={ctrl.K}); LLM {ctrl.llm}; {len(scored)} cases in "
          f"{len({sc.group_id for sc in scored})} groups; {n_boot} cluster-bootstrap resamples.", "",
          render_headline(metrics, policies, split), "", render_integrity(metrics, policies), "", render_cost(metrics, policies), "",
          f"**Ours vs best non-oracle baseline ({POLICY_LABELS[best_base]})**: utility difference "
          f"{fmt_ci(stats['utility_diff_ours_minus_best_baseline'])}; McNemar on per-case correctness: "
          f"{stats['mcnemar_ours_vs_best_baseline']['a_right_b_wrong']} ours-only right vs "
          f"{stats['mcnemar_ours_vs_best_baseline']['a_wrong_b_right']} baseline-only right, p = "
          f"{stats['mcnemar_ours_vs_best_baseline']['p_value']:.3g}.", "",
          f"Tuned heuristic thresholds (val): tau_a={heur['tau_a']}, tau_v={heur['tau_v']} (val reward {heur['val_reward']:.3f}).", ""]
    if seeds_summary:
        md.append(f"**Ours across {seeds_summary['n_seeds']} training seeds (argmax policy on {split})**: utility "
                  f"{seeds_summary['utility_mean']:.3f} +/- {seeds_summary['utility_std']:.3f}, accuracy "
                  f"{100 * seeds_summary['accuracy_mean']:.1f} +/- {100 * seeds_summary['accuracy_std']:.1f} %, coverage "
                  f"{100 * seeds_summary['coverage_mean']:.1f} +/- {100 * seeds_summary['coverage_std']:.1f} %. "
                  f"The shipped controller is the median-val-reward seed ({ctrl.meta.get('chosen_seed')}).\n")
    cards = case_cards(scored, rows["ours"], cases, w, c)
    (out_dir / f"results_{split}.md").write_text("\n".join(md) + "\n" + cards, encoding="utf-8")
    (out_dir / f"cards_{split}.md").write_text(cards, encoding="utf-8")

    # ---- figures
    figs = {}
    if figs_dir is not None:
        ensure_dir(figs_dir)
        if not sweep_rows and sweep_path is not None:
            print(f"[eval] no sweep at {sweep_path}; running a 3-seed sweep now (phase diagram + frontier)", file=sys.stderr)
            sweep_rows = run_sweep(data, [0.5, 1, 2, 4], [0, 0.025, 0.05, 0.1, 0.2], [0, 1, 2], Path(sweep_path),
                                   **ctrl.meta.get("train_config", {}))
        med = median_rows(sweep_rows)
        Xs = data.X[split]

        # 1 risk-coverage
        curves = {"confidence threshold (provisional)": risk_coverage_threshold(scored),
                  "oracle": risk_coverage_oracle(scored)}
        ours_pts = [cov_selacc(scored, [d["action"] for d in (pols["ours"][sc.case_id] for sc in scored)])]
        for (sw, sc_), row in sorted(med.items()):
            if abs(sc_ - c) < 1e-9 and abs(sw - w) > 1e-9:
                ours_pts.append(cov_selacc(scored, argmax_actions(np.asarray(row["W"]), Xs)))
        curves["ours (sweeping w)"] = [p for p in ours_pts if p[1] == p[1]]
        areas = {k: area_under(v) for k, v in curves.items()}
        figs["risk_coverage"] = F.fig_risk_coverage(curves, areas, figs_dir / "1_risk_coverage.png")

        # 2 phase diagram (median seed per cell, applied to this split)
        if med:
            ws = sorted({k[0] for k in med})
            cs = sorted({k[1] for k in med})
            maj = [[None] * len(cs) for _ in ws]
            util = np.zeros((len(ws), len(cs)))
            cells = []
            for i, sw in enumerate(ws):
                for j, sc_ in enumerate(cs):
                    row = med[(sw, sc_)]
                    acts = argmax_actions(np.asarray(row["W"]), Xs)
                    util[i, j] = float(np.mean([reward(s, int(a), sw, sc_) for s, a in zip(scored, acts)]))
                    maj[i][j] = C.ACTIONS[int(np.bincount(acts, minlength=3).argmax())]
                    cells.append({"w": sw, "c": sc_, "utility": util[i, j], "majority_action": maj[i][j],
                                  "mix": [float((acts == k).mean()) for k in range(3)], "seed": row["seed"]})
            figs["phase_diagram"] = F.fig_phase_diagram(ws, cs, maj, util, figs_dir / "2_phase_diagram.png",
                                                        title=f"Reward design on {split}: majority action over (w, c)")
            stats["phase_diagram"] = cells
            # 5 cost-accuracy frontier at the deployment w
            ours_front = []
            for sc_ in cs:
                row = med.get((w, sc_))
                if row is None:
                    continue
                acts = argmax_actions(np.asarray(row["W"]), Xs)
                outs = [outcome(s, int(a)) for s, a in zip(scored, acts)]
                ours_front.append((float(np.mean([o["tool_calls"] for o in outs])), float(np.mean([o["correct"] for o in outs])), sc_))
            base_pts = {POLICY_LABELS[p]: (metrics[p]["tool_calls"]["mean"], metrics[p]["accuracy"]["mean"])
                        for p in ("always_answer", "always_verify", "heuristic")}
            figs["cost_frontier"] = F.fig_cost_frontier(ours_front, base_pts, figs_dir / "5_cost_accuracy_frontier.png")

        # 3 grounding test
        gpols = [p for p in ("always_answer", "always_verify", "heuristic", "ours", "oracle") if p in metrics]
        parents_with_twin = {sc.parent_id for sc in scored if sc.variant == "ablated"}
        if parents_with_twin:
            pacc, flip, stub = [], [], []
            for p in gpols:
                prow = [r for r in rows[p] if r["case_id"] in parents_with_twin]
                pacc.append(float(np.mean([r["correct"] for r in prow])) if prow else None)
                flip.append(metrics[p]["integrity"]["grounding_flip_rate"])
                stub.append(metrics[p]["integrity"]["stubborn_rate"])
            figs["grounding"] = F.fig_grounding([POLICY_LABELS[p].split(" (")[0] for p in gpols], pacc, flip, stub,
                                                figs_dir / "3_grounding_test.png")

        # 4 reliability diagram (ours)
        calib = metrics["ours"].get("calibration")
        if calib and "bins_cal" in calib:
            figs["reliability"] = F.fig_reliability(calib["bins_raw"], calib["bins_cal"], calib["ece_raw"], calib["ece_cal"],
                                                    figs_dir / "4_reliability.png")

        # 6 W heatmap with a one-sentence reading
        Wm = ctrl.W.copy()
        bias_i = ctrl.feature_names.index("bias") if "bias" in ctrl.feature_names else -1
        Wn = Wm.copy()
        if bias_i >= 0:
            Wn[:, bias_i] = 0
        ai, fi = np.unravel_index(np.argmax(np.abs(Wn)), Wn.shape)
        direction = "pushes toward" if Wn[ai, fi] > 0 else "pushes away from"
        reading = f"Reading: higher {ctrl.feature_names[fi]} {direction} {ctrl.action_names[ai]} (weight {Wn[ai, fi]:+.2f})."
        figs["w_heatmap"] = F.fig_w_heatmap(ctrl.W, ctrl.feature_names, ctrl.action_names, figs_dir / "6_W_heatmap.png", reading)

        # 7 action mix by case type
        types = ["both_right", "only_verify_right", "only_answer_right", "neither_right"]
        mix = {}
        for p in ("heuristic", "ours"):
            d: dict = {t: Counter() for t in types}
            for sc, r in zip(scored, rows[p]):
                d[case_type(sc)][C.ACTIONS[r["action"]]] += 1
            mix[POLICY_LABELS[p]] = {t: dict(v) for t, v in d.items()}
        figs["action_mix"] = F.fig_action_mix(mix, types, figs_dir / "7_action_mix.png")

    results = {"split": split, "controller": str(controller_path), "w": w, "c": c, "K": ctrl.K, "llm": ctrl.llm,
               "n_cases": len(scored), "n_groups": len({sc.group_id for sc in scored}), "n_boot": n_boot,
               "policies": policies, "metrics": {p: {k: v for k, v in m.items() if k != "calibration"} |
                                                 {"calibration": {k: v for k, v in m.get("calibration", {}).items() if not k.startswith("bins")}}
                                                 for p, m in metrics.items()},
               "stats": stats, "figures": {k: str(v) for k, v in figs.items()}, "seconds": round(time.time() - t0, 1)}
    save_json(out_dir / f"results_{split}.json", results)
    print((out_dir / f"results_{split}.md").read_text())
    print(f"[eval] wrote {out_dir / f'results_{split}.md'}, logs in {logs_dir}, figures: {list(figs)} ({results['seconds']}s)")
    return results


def C_skip_hash() -> bool:
    import os

    return os.environ.get("KWTC_SKIP_HASH_CHECK", "") == "1"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_id", choices=C.SPLITS)
    ap.add_argument("--controller", default=str(C.ART_DIR / "controller.npz"))
    ap.add_argument("--figs", default=None, help="directory for figures (omit to skip figures)")
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--cache-dir", default=str(C.CACHE_DIR))
    ap.add_argument("--out", default=str(C.ART_DIR))
    ap.add_argument("--logs", default=str(C.ART_DIR / "logs"))
    ap.add_argument("--sweep", default=str(C.ART_DIR / "sweep.jsonl"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shortcut-controller", default=None)
    ap.add_argument("--w", type=float, default=None, help="override deployment w (default: the controller's)")
    ap.add_argument("--c", type=float, default=None)
    args = ap.parse_args(argv)
    evaluate(args.split, Path(args.controller), Path(args.cases_dir), Path(args.cache_dir),
             Path(args.figs) if args.figs else None, Path(args.out), Path(args.logs), Path(args.sweep) if args.sweep else None,
             args.n_boot, args.seed, Path(args.shortcut_controller) if args.shortcut_controller else None, args.w, args.c)


if __name__ == "__main__":
    main()
