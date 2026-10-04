from common import config as C
from scorer.score import integrity_report, oracle_action, outcome, reward, score_records


def _rec(cid, variant="base", parent=None, prov="supported", ver="supported", prov_cited=(10,), ver_cited=((10, 2),),
         opened=(10,), ver_conf=0.9, tool_calls=2):
    return {"case_id": cid, "split": "t", "group_id": "g", "variant": variant, "parent_id": parent, "shown_doc_ids": [10, 11],
            "a": {"verdict": prov, "confidence": 0.8, "evidence_sufficiency": 0.7, "cited_doc_ids": list(prov_cited), "rationale": "", "llm_calls": 1, "tokens_in": 10, "tokens_out": 5, "latency_s": 0.1},
            "b": {"verdict": prov, "confidence": 0.7, "evidence_sufficiency": 0.6, "cited_doc_ids": list(prov_cited), "rationale": "", "llm_calls": 1, "tokens_in": 10, "tokens_out": 5, "latency_s": 0.1},
            "prov": {"verdict": prov, "confidence": 0.8, "cited_doc_ids": list(prov_cited), "rationale": "", "from": "a"},
            "x": {"bias": 1, "conf_mean": 0.75, "agree": 1, "suff_mean": 0.65, "conf_gap": 0.1, "says_insuff": 0},
            "ver": {"verdict": ver, "confidence": ver_conf, "cited": [{"doc_id": d, "sentence": s} for d, s in ver_cited], "rationale": "",
                    "tool_calls": tool_calls, "llm_calls": 3, "tokens_in": 30, "tokens_out": 9, "opened_doc_ids": [str(o) for o in opened], "latency_s": 0.5},
            "flags": {}, "wall_s": 1.0}


GOLD = {"p": {"case_id": "p", "label": "supported", "gold_doc_ids": [10], "gold_sentences": {"10": [2, 3]}},
        "t": {"case_id": "t", "label": "insufficient_evidence", "gold_doc_ids": [], "gold_sentences": {}},
        "q": {"case_id": "q", "label": "refuted", "gold_doc_ids": [12], "gold_sentences": {"12": [0]}}}


def test_correctness_rewards_and_oracle():
    sc = score_records([_rec("p")], GOLD)[0]
    assert sc.prov_correct and sc.ver_correct and not sc.prov_fabricated and sc.ver_rationale_hit
    assert reward(sc, C.ANSWER) == 1.0 and reward(sc, C.VERIFY, c=0.05) == 0.9 and reward(sc, C.ABSTAIN) == 0.0
    assert oracle_action(sc) == C.ANSWER
    wrong = score_records([_rec("q", prov="supported", ver="supported")], GOLD)[0]
    assert oracle_action(wrong) == C.ABSTAIN and outcome(wrong, C.ABSTAIN)["committed"] is False


def test_integrity_checks_on_parent_twin_pair():
    recs = [
        _rec("p"),                                                                     # parent: correct, grounded
        _rec("t", variant="ablated", parent="p", prov="supported", ver="supported", ver_cited=((99, 0),), opened=(), ver_conf=0.95),  # twin: stubborn + fabricated cite
        _rec("q", prov="refuted", ver="refuted", prov_cited=(11,), ver_cited=((12, 5),), opened=(12,)),   # right for the wrong reason
    ]
    scored = score_records(recs, GOLD)
    rep = integrity_report(scored, {"p": C.VERIFY, "t": C.VERIFY, "q": C.VERIFY})
    assert rep["fabrication_rate"] == 1 / 3           # the twin cites doc 99 it never opened
    assert rep["grounding_flip_rate"] == 0.0           # twin kept the parent's S verdict instead of going NEI
    assert rep["stubborn_rate"] == 1.0
    assert rep["wrong_reason_rate"] == 0.5             # q is correct but cites sentence 5, not gold sentence 0
    rep2 = integrity_report(scored, {"p": C.ANSWER, "t": C.ABSTAIN, "q": C.ABSTAIN})
    assert rep2["grounding_flip_rate"] == 1.0 and rep2["unnecessary_abstention_rate"] == 1 / 3
