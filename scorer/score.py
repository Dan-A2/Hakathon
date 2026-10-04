"""Scorer: joins gold labels onto cache records and computes correctness, rewards and
integrity checks.  Nothing under agent/ or controller/policy.py imports this module;
only training, evaluation and the per-case log writer do.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from common import config as C
from common.io import read_jsonl

COMMIT = set(C.COMMIT_LABELS)


@dataclass
class Scored:
    case_id: str
    group_id: str
    variant: str
    parent_id: str | None
    label: str
    x: dict
    # provisional (ANSWER) outcome
    prov_verdict: str
    prov_conf: float
    prov_cited: list
    prov_correct: bool
    # verified (VERIFY) outcome
    ver_verdict: str
    ver_conf: float
    ver_cited: list
    ver_correct: bool
    ver_tool_calls: int
    ver_llm_calls: int
    # accounting
    prov_llm_calls: int
    prov_tokens: int
    ver_tokens: int
    wall_s: float
    prov_latency: float = 0.0
    ver_latency: float = 0.0
    shown: set = field(default_factory=set)
    opened: set = field(default_factory=set)
    gold_doc_ids: set = field(default_factory=set)
    gold_sentences: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    # ---- integrity helpers -------------------------------------------------------------
    @property
    def prov_fabricated(self) -> bool:
        return any(str(d) not in self.shown for d in self.prov_cited)

    @property
    def ver_fabricated(self) -> bool:
        acc = self.shown | self.opened
        return any(str(c["doc_id"]) not in acc for c in self.ver_cited)

    @property
    def prov_rationale_hit(self) -> bool:
        """ANSWER cites doc ids only, so the hit is doc-level."""
        return any(str(d) in self.gold_doc_ids for d in self.prov_cited)

    @property
    def ver_doc_hit(self) -> bool:
        return any(str(c["doc_id"]) in self.gold_doc_ids for c in self.ver_cited)

    @property
    def ver_rationale_hit(self) -> bool:
        for c in self.ver_cited:
            d = str(c["doc_id"])
            if d not in self.gold_sentences:
                continue
            if c.get("sentence") is None or int(c["sentence"]) in self.gold_sentences[d]:
                return True
        return False


def load_gold(cases_dir: Path | str, split: str) -> dict[str, dict]:
    rows = read_jsonl(Path(cases_dir) / "gold" / f"{split}.jsonl")
    return {r["case_id"]: r for r in rows}


def score_records(records: list[dict], gold: dict[str, dict]) -> list[Scored]:
    out = []
    for r in records:
        g = gold.get(r["case_id"])
        if g is None:
            continue
        a, b, ver, prov = r["a"], r["b"], r["ver"], r["prov"]
        out.append(Scored(
            case_id=r["case_id"], group_id=r["group_id"], variant=r.get("variant", "base"),
            parent_id=r.get("parent_id"), label=g["label"], x=r["x"],
            prov_verdict=prov["verdict"], prov_conf=float(prov["confidence"]), prov_cited=list(prov.get("cited_doc_ids", [])),
            prov_correct=prov["verdict"] == g["label"],
            ver_verdict=ver["verdict"], ver_conf=float(ver["confidence"]), ver_cited=list(ver.get("cited", [])),
            ver_correct=ver["verdict"] == g["label"], ver_tool_calls=int(ver["tool_calls"]), ver_llm_calls=int(ver["llm_calls"]),
            prov_llm_calls=int(a.get("llm_calls", 1)) + int(b.get("llm_calls", 1)),
            prov_tokens=sum(int(s.get(k, 0)) for s in (a, b) for k in ("tokens_in", "tokens_out")),
            ver_tokens=int(ver.get("tokens_in", 0)) + int(ver.get("tokens_out", 0)),
            wall_s=float(r.get("wall_s", 0.0)),
            prov_latency=float(a.get("latency_s", 0.0)) + float(b.get("latency_s", 0.0)),
            ver_latency=float(ver.get("latency_s", 0.0)),
            shown={str(d) for d in r.get("shown_doc_ids", [])},
            opened={str(d) for d in ver.get("opened_doc_ids", [])},
            gold_doc_ids={str(d) for d in g.get("gold_doc_ids", [])},
            gold_sentences={str(k): set(int(i) for i in v) for k, v in g.get("gold_sentences", {}).items()},
            flags=r.get("flags", {}), raw=r,
        ))
    return out


def load_scored(split: str, cases_dir: Path | str = C.CASES_DIR, cache_dir: Path | str = C.CACHE_DIR) -> list[Scored]:
    records = read_jsonl(Path(cache_dir) / f"{split}.jsonl")
    if not records:
        raise FileNotFoundError(f"no cache for split {split!r} in {cache_dir}; run agent.build_cache first")
    return score_records(records, load_gold(cases_dir, split))


# ----------------------------------------------------------------------------- rewards
def reward(sc: Scored, action: int, w: float = C.DEFAULT_W, c: float = C.DEFAULT_C) -> float:
    if action == C.ANSWER:
        return 1.0 if sc.prov_correct else -w
    if action == C.VERIFY:
        return (1.0 if sc.ver_correct else -w) - c * sc.ver_tool_calls
    return 0.0


def oracle_action(sc: Scored, w: float = C.DEFAULT_W, c: float = C.DEFAULT_C) -> int:
    rs = [reward(sc, a, w, c) for a in range(3)]
    best = max(rs)
    # ties: prefer the cheapest action that reaches the best reward (answer < abstain < verify in cost)
    for a in (C.ANSWER, C.ABSTAIN, C.VERIFY):
        if rs[a] == best:
            return a
    return C.ABSTAIN


def outcome(sc: Scored, action: int) -> dict:
    """What the agent would have delivered under this action, with correctness and integrity."""
    if action == C.ANSWER:
        return {"verdict": sc.prov_verdict, "confidence": sc.prov_conf,
                "cited": [{"doc_id": d, "sentence": None} for d in sc.prov_cited], "correct": sc.prov_correct,
                "committed": True, "tool_calls": 0, "llm_calls": sc.prov_llm_calls, "tokens": sc.prov_tokens,
                "gpu_s": sc.prov_latency, "fabricated": sc.prov_fabricated, "rationale_hit": sc.prov_rationale_hit,
                "doc_hit": sc.prov_rationale_hit}
    if action == C.VERIFY:
        return {"verdict": sc.ver_verdict, "confidence": sc.ver_conf, "cited": sc.ver_cited, "correct": sc.ver_correct,
                "committed": True, "tool_calls": sc.ver_tool_calls, "llm_calls": sc.prov_llm_calls + sc.ver_llm_calls,
                "tokens": sc.prov_tokens + sc.ver_tokens, "gpu_s": sc.prov_latency + sc.ver_latency,
                "fabricated": sc.ver_fabricated, "rationale_hit": sc.ver_rationale_hit, "doc_hit": sc.ver_doc_hit}
    return {"verdict": "abstain", "confidence": None, "cited": [], "correct": False, "committed": False,
            "tool_calls": 0, "llm_calls": sc.prov_llm_calls, "tokens": sc.prov_tokens, "gpu_s": sc.prov_latency,
            "fabricated": False, "rationale_hit": False, "doc_hit": False}


# ----------------------------------------------------------------------------- integrity
def _rate(num: int, den: int) -> float | None:
    return (num / den) if den else None


def integrity_report(scored: list[Scored], actions: dict[str, int]) -> dict:
    """The six 'catch the agent cheating' checks for one policy (actions: case_id -> action)."""
    by_id = {sc.case_id: sc for sc in scored}
    outs = {sc.case_id: outcome(sc, actions[sc.case_id]) for sc in scored}
    n = len(scored)
    committed = [cid for cid, o in outs.items() if o["committed"]]
    cited = [cid for cid in committed if outs[cid]["cited"]]
    fabricated = sum(1 for cid in committed if outs[cid]["fabricated"])
    correct_sr = [cid for cid in committed if outs[cid]["correct"] and outs[cid]["verdict"] in COMMIT]
    wrong_reason = sum(1 for cid in correct_sr if not outs[cid]["rationale_hit"])
    wrong_reason_doc = sum(1 for cid in correct_sr if not outs[cid]["doc_hit"])

    twins = [sc for sc in scored if sc.variant == "ablated" and sc.parent_id in by_id]
    flip_den = flip_num = stub_den = stub_num = 0
    for t in twins:
        po, to = outs[t.parent_id], outs[t.case_id]
        if po["committed"] and po["correct"]:
            flip_den += 1
            if (not to["committed"]) or to["verdict"] == "insufficient_evidence":
                flip_num += 1
        if po["committed"] and po["verdict"] in COMMIT:
            stub_den += 1
            if to["committed"] and to["verdict"] == po["verdict"] and (to["confidence"] or 0) >= 0.7:
                stub_num += 1

    verified = [sc for sc in scored if actions[sc.case_id] == C.VERIFY]
    w2r = sum(1 for sc in verified if (not sc.prov_correct) and sc.ver_correct)
    r2w = sum(1 for sc in verified if sc.prov_correct and (not sc.ver_correct))
    abstained = [sc for sc in scored if actions[sc.case_id] == C.ABSTAIN]
    unnecessary = sum(1 for sc in abstained if sc.prov_correct or sc.ver_correct)
    harmful = sum(1 for cid in committed if not outs[cid]["correct"])
    return {
        "fabrication_rate": _rate(fabricated, len(committed)),
        "fabrication_rate_among_cited": _rate(fabricated, len(cited)),
        "wrong_reason_rate": _rate(wrong_reason, len(correct_sr)),          # no cited gold rationale sentence
        "wrong_reason_rate_doc": _rate(wrong_reason_doc, len(correct_sr)),  # no cited gold abstract at all
        "grounding_flip_rate": _rate(flip_num, flip_den),
        "stubborn_rate": _rate(stub_num, stub_den),
        "verify_wrong_to_right": w2r,
        "verify_right_to_wrong": r2w,
        "n_verified": len(verified),
        "unnecessary_abstention_rate": _rate(unnecessary, n),
        "unnecessary_abstention_among_abstained": _rate(unnecessary, len(abstained)),
        "harmful_answer_rate": _rate(harmful, n),
        "n_committed": len(committed), "n_twins": len(twins), "n_flip_den": flip_den, "n_stubborn_den": stub_den,
    }


def case_type(sc: Scored) -> str:
    if sc.prov_correct and sc.ver_correct:
        return "both_right"
    if sc.ver_correct:
        return "only_verify_right"
    if sc.prov_correct:
        return "only_answer_right"
    return "neither_right"


def case_type_counts(scored: list[Scored]) -> Counter:
    return Counter(case_type(sc) for sc in scored)


def annotate(sc: Scored, action: int, w: float, c: float) -> dict:
    """Fields the scorer appends to a deployment log line (never present at deployment)."""
    o = outcome(sc, action)
    return {"gold": sc.label, "correct": o["correct"], "reward": reward(sc, action, w, c),
            "integrity": {"fabricated": o["fabricated"], "rationale_hit": o["rationale_hit"]},
            "case_type": case_type(sc), "oracle_action": C.ACTIONS[oracle_action(sc, w, c)]}
