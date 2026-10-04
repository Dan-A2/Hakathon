import numpy as np

from agent.epistemic import disciplined_provisional, disciplined_verified, post_check_features
from agent.judge import credence_from_logprobs
from common import config as C
from data.records import RecordStore

DOCS = [{"doc_id": 1, "title": "A", "sentences": ["Aspirin reduces myocardial infarction risk.", "N=1000."]},
        {"doc_id": 2, "title": "B", "sentences": ["Vitamin D improves bone density."]}]
STORE = RecordStore(DOCS)
PROV = {"verdict": "supported", "confidence": 0.9, "cited_doc_ids": [1]}
VER = {"verdict": "supported", "confidence": 0.8, "cited": [{"doc_id": 1, "sentence": 0}], "opened_doc_ids": ["1"], "tool_calls": 2}


def test_rule1_no_evidence_no_verdict():
    assert disciplined_provisional(PROV, shown={"1"})["verdict"] == "supported"
    d = disciplined_provisional(PROV, shown={"2"})                      # cites an abstract it was never shown
    assert d["verdict"] == "insufficient_evidence" and d["grounded"] is False
    assert disciplined_verified(VER, set(), {"1"}, STORE, [])["verdict"] == "supported"
    u = disciplined_verified({**VER, "cited": [{"doc_id": 2, "sentence": 0}]}, set(), {"1"}, STORE, [])   # cites an unopened doc
    assert u["verdict"] == "insufficient_evidence" and u["source"] == "ungrounded"
    bad_idx = disciplined_verified({**VER, "cited": [{"doc_id": 1, "sentence": 7}]}, set(), {"1"}, STORE, [])
    assert bad_idx["source"] == "ungrounded"
    assert disciplined_verified({**VER, "verdict": "insufficient_evidence"}, set(), set(), STORE, [])["verdict"] == "insufficient_evidence"


def test_rule1_single_sentence_record_accepts_any_index():
    from agent.epistemic import accessible_citations

    one = RecordStore([{"doc_id": 9, "title": "snippet", "sentences": ["Only sentence."]}])
    for idx in (0, 1, 3, None):
        cites = accessible_citations([{"doc_id": 9, "sentence": idx}], set(), {"9"}, one, [])
        assert cites == [{"doc_id": 9, "sentence": 0, "text": "Only sentence."}]
    assert accessible_citations([{"doc_id": 9, "sentence": 0}], set(), set(), one, []) == []          # never opened -> still ungrounded
    assert accessible_citations([{"doc_id": 1, "sentence": 7}], set(), {"1"}, STORE, []) == []        # multi-sentence record: index must be valid


def test_rule2_judge_overrides_verifier():
    j_not = {"relation": "not_addressed", "confidence": 0.8}
    j_ref = {"relation": "refutes", "confidence": 0.7}
    assert disciplined_verified(VER, set(), {"1"}, STORE, [], j_not)["verdict"] == "insufficient_evidence"
    d = disciplined_verified(VER, set(), {"1"}, STORE, [], j_ref)
    assert d["verdict"] == "refuted" and d["source"] == "judge" and d["confidence"] == 0.7


def test_post_check_features_shape():
    x = {"bias": 1, "conf_mean": 0.8, "agree": 1, "suff_mean": 0.5, "conf_gap": 0.1, "says_insuff": 0}
    dprov = disciplined_provisional(PROV, {"1"})
    dver = disciplined_verified(VER, set(), {"1"}, STORE, [])
    z = post_check_features(x, dprov, VER, dver, None, "Aspirin reduces myocardial infarction risk.", 4)
    assert list(z) == C.POST_FEATURE_NAMES and z["agree_prior"] == 1.0 and z["tool_frac"] == 0.5 and z["cited_overlap"] > 0.9


def test_credence_from_top_logprobs():
    text = '{"relation": "supports", "confidence": 0.9}'
    toks = ['{"relation"', ': ', '"supports"', ', "confidence": 0.9}']
    tops = [[(t, 0.0)] for t in toks]
    tops[2] = [('"supports"', np.log(0.6)), ('"refutes"', np.log(0.1)), ('"not_addressed"', np.log(0.3))]
    cr = credence_from_logprobs(text, tops, "supports", 0.9)
    assert abs(cr["supports"] - 0.6) < 1e-6 and abs(sum(cr.values()) - 1) < 1e-9
    fb = credence_from_logprobs(text, None, "supports", 0.9)
    assert fb["supports"] == 0.9 and abs(fb["refutes"] - 0.05) < 1e-9


def test_two_stage_trainer_and_eu_learn_when_checking_pays():
    """Subgroup f=1: the prior is wrong but the (disciplined) check is right -> check and commit.
    Subgroup f=0: the prior is right and the check is wrong -> answer without checking."""
    from controller.epistemic import EUController, S2_TO_DECISION, argmax_two_stage, decisions_reward, train_two_stage
    from controller.policy import Standardizer

    rng = np.random.default_rng(0)
    n = 500
    f = rng.integers(0, 2, n).astype(float)
    X1 = np.stack([np.ones(n), (f - f.mean()) / f.std()], axis=1)
    agree = 1 - f                                            # when the check is wrong it disagrees with the (right) prior
    X2 = np.stack([np.ones(n), (agree - agree.mean()) / agree.std()], axis=1)
    cost = 0.1
    Ra = np.where(f == 1, -1.0, 1.0)                         # grounded answer
    R2 = np.stack([np.where(f == 1, 1.0, -1.0) - cost,       # check_commit
                   np.where(f == 1, -1.0, 1.0) - cost,       # check_keep_prior
                   np.full(n, -cost)], axis=1)               # check_abstain
    res = train_two_stage(X1, X2, Ra, R2, seed=0, epochs=150, patience=1000)
    dec = argmax_two_stage(res["W1"], res["W2"], X1, X2)
    r = decisions_reward(dec, Ra, R2)
    assert r > 0.85, r                                        # optimum is 1 - cost/2 = 0.95
    assert (dec[f == 1] == C.CHECK_COMMIT).mean() > 0.9 and (dec[f == 0] == C.D_ANSWER).mean() > 0.9

    class Fake:  # minimal Scored stand-in for EUController.fit
        def __init__(self, i):
            self.dprov_correct = bool(f[i] == 0); self.dver_correct = bool(f[i] == 1); self.ver_tool_calls = 2
            self.prov_correct = self.dprov_correct; self.ver_correct = self.dver_correct
    from scorer import score as S
    fakes = [Fake(i) for i in range(n)]
    real_reward = S.reward
    S.reward = lambda sc, a, w=1.0, c=0.05: (1.0 if sc.dprov_correct else -w) if a == C.D_ANSWER else (
        ((1.0 if sc.dver_correct else -w) - c * sc.ver_tool_calls) if a == C.CHECK_COMMIT else
        ((1.0 if sc.dprov_correct else -w) - c * sc.ver_tool_calls) if a == C.CHECK_KEEP else -c * sc.ver_tool_calls if a == C.CHECK_ABSTAIN else 0.0)
    try:
        import controller.epistemic as E
        E.reward = S.reward
        eu = EUController.fit(fakes, X1, X2, Standardizer(["bias", "f"], np.zeros(2), np.ones(2)),
                              Standardizer(["bias", "agree"], np.zeros(2), np.ones(2)), w=1.0, c=0.05)
        dec_eu, cred, _ = eu.decide_batch(X1, X2)
        assert (dec_eu[f == 1] == C.CHECK_COMMIT).mean() > 0.9 and (dec_eu[f == 0] == C.D_ANSWER).mean() > 0.9
        assert np.nanmean(cred) > 0.8
    finally:
        S.reward = real_reward
        E.reward = real_reward
