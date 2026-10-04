"""Epistemic discipline applied to the frozen agent's outputs (deterministic, auditable, no LLM calls).

Rule 1  No evidence, no verdict.  A supported/refuted verdict must cite evidence the agent actually
        saw: the provisional verdict must cite an abstract that was shown; the verified verdict must
        cite a sentence of a record it opened (or was shown).  Otherwise the justified output is
        "insufficient_evidence".
Rule 2  Judgement is separated from search.  When a blind-judge record exists (agent/judge.py: an
        LLM call that sees only the claim and the cited sentences), the judge's reading of the
        evidence replaces the verifier's self-assessment.
Rule 3  A check is evidence, not an oracle.  The post-check signals below describe what the check
        revealed (does the new verdict agree with the prior, was it grounded, how confident) so a
        second decision can commit, keep the prior, or abstain - instead of blindly adopting the
        last thing the model said.
"""
from __future__ import annotations

from common import config as C
from data.records import RecordStore, tokenize

JUDGE_TO_VERDICT = {"supports": "supported", "refutes": "refuted", "not_addressed": "insufficient_evidence"}


def accessible_citations(cited: list[dict], shown: set, opened: set, store: RecordStore, exclude: list) -> list[dict]:
    """Citations (doc_id, sentence) that point at a real sentence of a record the agent saw or opened."""
    acc = {str(d) for d in shown} | {str(d) for d in opened}
    out = []
    for c in cited:
        d = str(c.get("doc_id"))
        s = c.get("sentence")
        if d not in acc:
            continue
        rec = store.get(c["doc_id"], exclude=exclude)
        if rec is None or not rec["sentences"]:
            continue
        if len(rec["sentences"]) == 1:
            # a single-sentence record (HealthVer, Climate-FEVER, VitaminC snippets): citing the record IS citing the
            # sentence, whatever index the model wrote (models often number from 1 or copy a listing index)
            s = 0
        if s is None or not (0 <= int(s) < len(rec["sentences"])):
            continue
        out.append({"doc_id": c["doc_id"], "sentence": int(s), "text": rec["sentences"][int(s)]})
    return out


def claim_overlap(claim: str, sentences: list[str]) -> float:
    toks = set(tokenize(claim))
    if not toks:
        return 0.0
    return max((len(toks & set(tokenize(s))) / len(toks) for s in sentences), default=0.0)


def disciplined_provisional(prov: dict, shown: set) -> dict:
    """Rule 1 for the provisional answer: a committed verdict must cite a shown abstract."""
    cited = [d for d in prov.get("cited_doc_ids", []) if str(d) in {str(x) for x in shown}]
    if prov["verdict"] in C.COMMIT_LABELS and not cited:
        return {"verdict": "insufficient_evidence", "confidence": prov["confidence"], "cited_doc_ids": [],
                "grounded": False, "source": "ungrounded"}
    return {"verdict": prov["verdict"], "confidence": prov["confidence"], "cited_doc_ids": cited,
            "grounded": True, "source": "provisional"}


def disciplined_verified(ver: dict, shown: set, opened: set, store: RecordStore, exclude: list,
                         judge: dict | None = None) -> dict:
    """Rules 1 and 2 for the verified answer."""
    cites = accessible_citations(ver.get("cited", []), shown, opened, store, exclude)
    if ver["verdict"] not in C.COMMIT_LABELS:
        return {"verdict": "insufficient_evidence", "confidence": ver["confidence"], "cited": cites,
                "grounded": True, "source": "verifier"}
    if not cites:
        return {"verdict": "insufficient_evidence", "confidence": ver["confidence"], "cited": [],
                "grounded": False, "source": "ungrounded"}
    if judge and judge.get("relation") in JUDGE_TO_VERDICT:
        return {"verdict": JUDGE_TO_VERDICT[judge["relation"]], "confidence": float(judge.get("confidence", ver["confidence"])),
                "cited": cites, "grounded": True, "source": "judge"}
    return {"verdict": ver["verdict"], "confidence": ver["confidence"], "cited": cites, "grounded": True, "source": "verifier"}


def post_check_features(x: dict, dprov: dict, ver: dict, dver: dict, judge: dict | None, claim: str, K: int) -> dict:
    """Rule 3: what the check revealed, as a feature vector for the post-check decision."""
    z = {
        "bias": 1.0,
        "conf_mean": float(x["conf_mean"]),
        "says_insuff": float(x["says_insuff"]),
        "suff_mean": float(x["suff_mean"]),
        "ver_conf": float(ver["confidence"]),
        "agree_prior": 1.0 if dver["verdict"] == dprov["verdict"] else 0.0,
        "check_says_insuff": 1.0 if dver["verdict"] == "insufficient_evidence" else 0.0,
        "grounded": 1.0 if dver["grounded"] else 0.0,
        "opened_any": 1.0 if ver.get("opened_doc_ids") else 0.0,
        "tool_frac": float(ver.get("tool_calls", 0)) / max(1, K),
        "cited_overlap": claim_overlap(claim, [c["text"] for c in dver["cited"]]),
        "judge_available": 1.0 if judge else 0.0,
        "judge_conf": float(judge.get("confidence", 0.0)) if judge else 0.0,
    }
    assert list(z) == C.POST_FEATURE_NAMES
    return z
