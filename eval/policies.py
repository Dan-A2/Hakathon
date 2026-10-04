"""The policies compared on test-ID and test-OOD. All are scored from the same cache (paired)."""
from __future__ import annotations

import itertools

import numpy as np

from common import config as C
from controller.policy import Controller
from scorer.score import Scored, oracle_action, outcome, reward

POLICY_ORDER = ["always_answer", "always_verify", "always_abstain", "heuristic", "ours", "always_check",
                "epistemic_rl", "epistemic_eu", "oracle", "shortcut"]
from eval.report import POLICY_LABELS  # one vocabulary for every table and figure
EPISTEMIC = {"epistemic_rl", "epistemic_eu"}


class Decision(dict):
    """action, action_probs (or None), plus the delivered outcome."""


def decide_constant(scored: list[Scored], action: int) -> dict[str, Decision]:
    return {sc.case_id: Decision(action=action, action_probs=None) for sc in scored}


def heuristic_action(x: dict, tau_a: float, tau_v: float) -> int:
    if x["conf_mean"] < tau_a:
        return C.ABSTAIN
    if x["conf_mean"] < tau_v or x["agree"] < 0.5:
        return C.VERIFY
    return C.ANSWER


def tune_heuristic(val: list[Scored], w: float, c: float, grid=None) -> tuple[float, float, float]:
    """Grid-search (tau_a, tau_v) on val mean reward. Returns (tau_a, tau_v, val_reward)."""
    grid = grid if grid is not None else [round(0.05 * i, 2) for i in range(0, 21)]
    conf = np.asarray([sc.x["conf_mean"] for sc in val])
    agree = np.asarray([sc.x["agree"] for sc in val])
    R = np.asarray([[reward(sc, a, w, c) for a in range(3)] for sc in val])
    best = (-np.inf, 0.0, 0.0)
    for ta, tv in itertools.product(grid, grid):
        a = np.where(conf < ta, C.ABSTAIN, np.where((conf < tv) | (agree < 0.5), C.VERIFY, C.ANSWER))
        r = R[np.arange(len(val)), a].mean()
        if r > best[0] + 1e-12:
            best = (float(r), ta, tv)
    return best[1], best[2], best[0]


def decide_heuristic(scored: list[Scored], tau_a: float, tau_v: float) -> dict[str, Decision]:
    return {sc.case_id: Decision(action=heuristic_action(sc.x, tau_a, tau_v), action_probs=None) for sc in scored}


def decide_controller(scored: list[Scored], ctrl: Controller) -> dict[str, Decision]:
    out = {}
    for sc in scored:
        a, p = ctrl.act(sc.x)
        out[sc.case_id] = Decision(action=a, action_probs=[float(v) for v in p])
    return out


def decide_oracle(scored: list[Scored], w: float, c: float) -> dict[str, Decision]:
    return {sc.case_id: Decision(action=oracle_action(sc, w, c), action_probs=None) for sc in scored}


def decide_two_stage(scored: list[Scored], ctrl) -> dict[str, Decision]:
    out = {}
    for sc in scored:
        d, info = ctrl.decide(sc.x, sc.z)
        out[sc.case_id] = Decision(action=d, action_probs=[info["stage1"][n] for n in C.STAGE1_NAMES], stage2=info.get("stage2"))
    return out


def decide_eu(scored: list[Scored], ctrl) -> dict[str, Decision]:
    out = {}
    for sc in scored:
        d, cred, info = ctrl.decide(sc.x, sc.z)
        out[sc.case_id] = Decision(action=d, action_probs=None, credence=cred, ev=info)
    return out


def build_policies(scored: list[Scored], val: list[Scored], ctrl: Controller, w: float, c: float,
                   shortcut_ctrl: Controller | None = None, epi_rl=None, epi_eu=None) -> tuple[dict[str, dict[str, Decision]], dict]:
    tau_a, tau_v, val_r = tune_heuristic(val, w, c)
    pols = {
        "always_answer": decide_constant(scored, C.ANSWER),
        "always_verify": decide_constant(scored, C.VERIFY),
        "always_abstain": decide_constant(scored, C.ABSTAIN),
        "heuristic": decide_heuristic(scored, tau_a, tau_v),
        "ours": decide_controller(scored, ctrl),
        "always_check": decide_constant(scored, C.CHECK_COMMIT),
        "oracle": decide_oracle(scored, w, c),
    }
    if epi_rl is not None:
        pols["epistemic_rl"] = decide_two_stage(scored, epi_rl)
    if epi_eu is not None:
        pols["epistemic_eu"] = decide_eu(scored, epi_eu)
    if shortcut_ctrl is not None:
        pols["shortcut"] = decide_controller(scored, shortcut_ctrl)
    return pols, {"tau_a": tau_a, "tau_v": tau_v, "val_reward": val_r}


def per_case_rows(scored: list[Scored], decisions: dict[str, Decision], w: float, c: float) -> list[dict]:
    """One row per case with the delivered outcome, reward and integrity flags."""
    rows = []
    for sc in scored:
        d = decisions[sc.case_id]
        o = outcome(sc, d["action"])
        rows.append({
            "case_id": sc.case_id, "group_id": sc.group_id, "variant": sc.variant, "parent_id": sc.parent_id,
            "action": d["action"], "action_name": C.DECISIONS[d["action"]], "action_coarse": C.COARSE[d["action"]],
            "checked": d["action"] in C.CHECKED, "contested": sc.contested if d["action"] in C.CHECKED else None,
            "credence": d.get("credence"), "action_probs": d.get("action_probs"), "verdict": o["verdict"],
            "confidence": o["confidence"], "cited": o["cited"], "committed": o["committed"], "correct": o["correct"],
            "reward": reward(sc, d["action"], w, c), "tool_calls": o["tool_calls"], "llm_calls": o["llm_calls"],
            "tokens": o["tokens"], "gpu_s": o["gpu_s"], "fabricated": o["fabricated"], "rationale_hit": o["rationale_hit"],
            "gold": sc.label, "prov_correct": sc.prov_correct, "ver_correct": sc.ver_correct, "x": sc.x,
            "wall_s": sc.wall_s,
        })
    return rows
