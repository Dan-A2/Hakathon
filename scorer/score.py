"""Scorer: joins gold labels onto cache records and computes correctness, rewards and
integrity checks.  Nothing under agent/ or controller/policy.py imports this module;
only training, evaluation and the per-case log writer do.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from agent.epistemic import disciplined_provisional, disciplined_verified, post_check_features
from common import config as C
from common.io import read_jsonl
from data.records import StoreRegistry

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
    # epistemic (disciplined) outcomes - see agent/epistemic.py
    dprov_verdict: str = ""
    dprov_correct: bool = False
    dprov_cited: list = field(default_factory=list)
    dver_verdict: str = ""
    dver_correct: bool = False
    dver_cited: list = field(default_factory=list)
    dver_source: str = ""
    z: dict = field(default_factory=dict)
    judge: dict | None = None

    @property
    def contested(self) -> bool:
        """The check disagreed with the prior: the claim is contested for this agent."""
        return self.dver_verdict != self.dprov_verdict

    @property
    def dver_rationale_hit(self) -> bool:
        return any(str(c["doc_id"]) in self.gold_sentences and int(c["sentence"]) in self.gold_sentences[str(c["doc_id"])]
                   for c in self.dver_cited)

    @property
    def dver_doc_hit(self) -> bool:
        return any(str(c["doc_id"]) in self.gold_doc_ids for c in self.dver_cited)

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


class ScoreContext:
    """Record stores and claim texts, needed to apply the epistemic discipline rules."""

    def __init__(self, cases_dir: Path | str, judge: dict[str, dict] | None = None):
        self.registry = StoreRegistry(cases_dir)
        self.cases: dict[str, dict] = {}
        for split in C.ALL_SPLITS:
            for c in read_jsonl(Path(cases_dir) / f"{split}.jsonl"):
                self.cases[c["case_id"]] = c
        self.judge = judge or {}


def score_records(records: list[dict], gold: dict[str, dict], ctx: ScoreContext | None = None) -> list[Scored]:
    out = []
    for r in records:
        g = gold.get(r["case_id"])
        if g is None:
            continue
        a, b, ver, prov = r["a"], r["b"], r["ver"], r["prov"]
        shown = {str(d) for d in r.get("shown_doc_ids", [])}
        opened = {str(d) for d in ver.get("opened_doc_ids", [])}
        dprov = disciplined_provisional(prov, shown)
        if ctx is not None and r["case_id"] in ctx.cases:
            case = ctx.cases[r["case_id"]]
            store, exclude = ctx.registry.for_case(case)
            judge = ctx.judge.get(r["case_id"])
            dver = disciplined_verified(ver, shown, opened, store, exclude, judge)
            z = post_check_features(r["x"], dprov, ver, dver, judge, case["question"], int(r.get("K", C.DEFAULT_K)))
        else:   # no stores available (unit tests): grounding by id only, no judge
            judge = None
            acc = shown | opened
            cites = [c for c in ver.get("cited", []) if str(c.get("doc_id")) in acc and c.get("sentence") is not None]
            if ver["verdict"] in COMMIT and not cites:
                dver = {"verdict": "insufficient_evidence", "confidence": ver["confidence"], "cited": [], "grounded": False, "source": "ungrounded"}
            else:
                dver = {"verdict": ver["verdict"] if ver["verdict"] in COMMIT else "insufficient_evidence", "confidence": ver["confidence"],
                        "cited": cites, "grounded": True, "source": "verifier"}
            z = {}
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
            dprov_verdict=dprov["verdict"], dprov_correct=dprov["verdict"] == g["label"], dprov_cited=list(dprov["cited_doc_ids"]),
            dver_verdict=dver["verdict"], dver_correct=dver["verdict"] == g["label"], dver_cited=list(dver["cited"]),
            dver_source=dver["source"], z=z, judge=judge,
        ))
    return out


def load_judge(cache_dir: Path | str, split: str) -> dict[str, dict]:
    """Blind-judge records for a split (optional; produced by agent/build_judge.py or modal_app.py::judge)."""
    path = Path(cache_dir) / f"judge_{split}.jsonl"
    return {r["case_id"]: r for r in read_jsonl(path)} if path.exists() else {}


def load_scored(split: str, cases_dir: Path | str = C.CASES_DIR, cache_dir: Path | str = C.CACHE_DIR) -> list[Scored]:
    records = read_jsonl(Path(cache_dir) / f"{split}.jsonl")
    if not records:
        raise FileNotFoundError(f"no cache for split {split!r} in {cache_dir}; run agent.build_cache first")
    ctx = ScoreContext(cases_dir, load_judge(cache_dir, split))
    return score_records(records, load_gold(cases_dir, split), ctx)


# ----------------------------------------------------------------------------- rewards
def reward(sc: Scored, action: int, w: float = C.DEFAULT_W, c: float = C.DEFAULT_C) -> float:
    cost = c * sc.ver_tool_calls
    if action == C.ANSWER:
        return 1.0 if sc.prov_correct else -w
    if action == C.VERIFY:
        return (1.0 if sc.ver_correct else -w) - cost
    if action == C.D_ANSWER:
        return 1.0 if sc.dprov_correct else -w
    if action == C.CHECK_COMMIT:
        return (1.0 if sc.dver_correct else -w) - cost
    if action == C.CHECK_KEEP:
        return (1.0 if sc.dprov_correct else -w) - cost
    if action == C.CHECK_ABSTAIN:
        return -cost                                   # checked, learned the claim is contested, said nothing
    return 0.0


def post_check_oracle(sc: Scored, w: float = C.DEFAULT_W, c: float = C.DEFAULT_C) -> int:
    """Best post-check decision with gold (upper bound for the second stage)."""
    rs = {a: reward(sc, a, w, c) for a in C.STAGE2_DECISIONS}
    best = max(rs.values())
    for a in (C.CHECK_KEEP, C.CHECK_COMMIT, C.CHECK_ABSTAIN):
        if rs[a] == best:
            return a
    return C.CHECK_ABSTAIN


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
    if action == C.D_ANSWER:
        committed = sc.dprov_verdict in COMMIT or sc.dprov_verdict == "insufficient_evidence"
        return {"verdict": sc.dprov_verdict, "confidence": sc.prov_conf,
                "cited": [{"doc_id": d, "sentence": None} for d in sc.dprov_cited], "correct": sc.dprov_correct,
                "committed": committed, "tool_calls": 0, "llm_calls": sc.prov_llm_calls, "tokens": sc.prov_tokens,
                "gpu_s": sc.prov_latency, "fabricated": False, "rationale_hit": any(str(d) in sc.gold_doc_ids for d in sc.dprov_cited),
                "doc_hit": any(str(d) in sc.gold_doc_ids for d in sc.dprov_cited)}
    checked = {"tool_calls": sc.ver_tool_calls, "llm_calls": sc.prov_llm_calls + sc.ver_llm_calls + (1 if sc.judge else 0),
               "tokens": sc.prov_tokens + sc.ver_tokens + int((sc.judge or {}).get("tokens", 0)),
               "gpu_s": sc.prov_latency + sc.ver_latency + float((sc.judge or {}).get("latency_s", 0.0))}
    if action == C.CHECK_COMMIT:
        return {"verdict": sc.dver_verdict, "confidence": sc.raw["ver"]["confidence"],
                "cited": [{"doc_id": c["doc_id"], "sentence": c["sentence"]} for c in sc.dver_cited], "correct": sc.dver_correct,
                "committed": True, **checked, "fabricated": False, "rationale_hit": sc.dver_rationale_hit, "doc_hit": sc.dver_doc_hit}
    if action == C.CHECK_KEEP:
        return {"verdict": sc.dprov_verdict, "confidence": sc.prov_conf,
                "cited": [{"doc_id": d, "sentence": None} for d in sc.dprov_cited], "correct": sc.dprov_correct,
                "committed": True, **checked, "fabricated": False, "rationale_hit": any(str(d) in sc.gold_doc_ids for d in sc.dprov_cited),
                "doc_hit": any(str(d) in sc.gold_doc_ids for d in sc.dprov_cited)}
    if action == C.CHECK_ABSTAIN:
        return {"verdict": "abstain", "confidence": None, "cited": [], "correct": False, "committed": False, **checked,
                "fabricated": False, "rationale_hit": False, "doc_hit": False}
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

    verified = [sc for sc in scored if actions[sc.case_id] in C.CHECKED]
    w2r = sum(1 for sc in verified if (not sc.prov_correct) and outs[sc.case_id]["correct"])
    r2w = sum(1 for sc in verified if sc.prov_correct and (not outs[sc.case_id]["correct"]))
    abstained = [sc for sc in scored if not outs[sc.case_id]["committed"]]
    contested = [sc for sc in verified if sc.contested]
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
        "check_rate": _rate(len(verified), n),
        "contested_rate_among_checked": _rate(len(contested), len(verified)),
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


def decision_name(action: int) -> str:
    return C.DECISIONS[action]
