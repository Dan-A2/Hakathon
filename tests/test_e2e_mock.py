"""End-to-end smoke test on a tiny synthetic corpus: cases -> mock cache -> train -> evaluate -> infer."""
import json
from pathlib import Path

from common.io import write_jsonl


def _make_cases(tmp: Path):
    docs = [
        {"doc_id": 1, "title": "Aspirin and myocardial infarction", "sentences": ["Aspirin reduces myocardial infarction risk in diabetes.", "N=1000."]},
        {"doc_id": 2, "title": "Vitamin D and bones", "sentences": ["Vitamin D does not improve bone density.", "A null trial."]},
        {"doc_id": 3, "title": "Statins and stroke", "sentences": ["Statins lower stroke risk.", "Meta-analysis."]},
        {"doc_id": 4, "title": "Coffee and sleep", "sentences": ["Coffee delays sleep onset."]},
        {"doc_id": 5, "title": "Exercise and mood", "sentences": ["Exercise improves mood in adults."]},
    ]
    write_jsonl(tmp / "records/corpus.jsonl", docs)
    claims = [("Aspirin reduces myocardial infarction risk in diabetes.", "supported", [1], {"1": [0]}),
              ("Vitamin D improves bone density.", "refuted", [2], {"2": [0]}),
              ("Statins lower stroke risk.", "supported", [3], {"3": [0]}),
              ("Coffee improves memory.", "insufficient_evidence", [], {}),
              ("Exercise worsens mood.", "refuted", [5], {"5": [0]}),
              ("Tea cures headaches.", "insufficient_evidence", [], {})]
    from data.records import RecordStore
    store = RecordStore(docs)
    for split, idxs in {"train": [0, 1, 2, 3], "val": [4, 5], "test_id": [0, 1, 3]}.items():
        cases, golds = [], []
        for i in idxs:
            q, label, gold_docs, gold_sents = claims[i]
            cid = f"{split}-{i}"
            cases.append({"case_id": cid, "group_id": f"g{i}", "split": split, "question": q,
                          "initial_evidence": store.initial_evidence(q, k=2), "record_index": "bm25",
                          "record_store": {"path": "records/corpus.jsonl", "exclude": []}, "variant": "base", "parent_id": None})
            golds.append({"case_id": cid, "label": label, "gold_doc_ids": gold_docs, "gold_sentences": gold_sents})
            if gold_docs:
                cases.append({**cases[-1], "case_id": cid + "-abl", "variant": "ablated", "parent_id": cid,
                              "initial_evidence": store.initial_evidence(q, k=2, exclude=gold_docs),
                              "record_store": {"path": "records/corpus.jsonl", "exclude": gold_docs}})
                golds.append({"case_id": cid + "-abl", "label": "insufficient_evidence", "gold_doc_ids": [], "gold_sentences": {}})
        write_jsonl(tmp / f"{split}.jsonl", cases)
        write_jsonl(tmp / "gold" / f"{split}.jsonl", golds)


def test_pipeline_end_to_end(tmp_path):
    cases_dir, cache_dir, art = tmp_path / "cases", tmp_path / "cache", tmp_path / "art"
    _make_cases(cases_dir)

    from agent.build_cache import build_split
    from agent.llm import get_llm
    llm = get_llm("mock")
    for split in ("train", "val", "test_id"):
        out = build_split(split, cases_dir, cache_dir, llm, workers=2, quiet=True)
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert rows and all({"a", "b", "x", "ver", "prov"} <= set(r) for r in rows)
        assert all(r["llm"]["backend"] == "mock" for r in rows)

    from controller.train import Data, fit_and_save
    data = Data(cases_dir, cache_dir, splits=("train", "val", "test_id"))
    ctrl = fit_and_save(data, 1.0, 0.05, [0, 1, 2], art / "controller.npz", epochs=20, patience=100)
    assert (art / "controller.npz").exists() and ctrl.W.shape == (3, 6)

    from eval.evaluate import evaluate
    res = evaluate("test_id", art / "controller.npz", cases_dir, cache_dir, None, art, art / "logs", None, n_boot=200)
    assert set(res["policies"]) >= {"always_answer", "always_verify", "always_abstain", "heuristic", "ours", "oracle"}
    assert (art / "results_test_id.md").exists() and (art / "logs" / "test_id_ours.jsonl").exists()
    assert res["metrics"]["oracle"]["utility"]["mean"] >= res["metrics"]["ours"]["utility"]["mean"] - 1e-9
    assert res["metrics"]["always_abstain"]["utility"]["mean"] == 0.0

    from agent.infer import Runtime, infer_case
    rt = Runtime(art / "controller.npz", art / "calibrator.npz", "mock", cases_dir)
    case = json.loads((cases_dir / "test_id.jsonl").read_text().splitlines()[0])
    r = infer_case(rt, case)
    assert r["action"] in ("answer", "verify", "abstain") and len(r["samples"]) == 2
    assert r["p_correct"] is None or 0.0 <= r["p_correct"] <= 1.0
