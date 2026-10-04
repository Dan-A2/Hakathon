import json

import numpy as np

from eval.transfer import auroc, conformal_threshold, mahalanobis, platt_fit


def test_platt_fit_recovers_scale_and_shift():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2, 2000)
    true_a, true_b = 0.5, -0.7
    y = (rng.uniform(size=2000) < 1 / (1 + np.exp(-(true_a * logits + true_b)))).astype(float)
    a, b = platt_fit(logits, y)
    assert abs(a - true_a) < 0.1 and abs(b - true_b) < 0.15


def test_conformal_threshold_respects_target():
    rng = np.random.default_rng(1)
    cred = rng.uniform(0.3, 1.0, 400)
    correct = (rng.uniform(size=400) < cred).astype(float)     # perfectly calibrated credences
    tau = conformal_threshold(cred, correct, alpha=0.2, delta=0.1)
    assert np.isfinite(tau)
    sel = cred >= tau
    assert np.mean(1 - correct[sel]) <= 0.2 + 1e-9             # the empirical risk on the calibration set is under alpha
    assert conformal_threshold(np.array([0.9] * 30), np.array([0.0] * 30), alpha=0.1) == float("inf")   # nothing passes -> abstain all
    a0, b0 = platt_fit(np.array([5.0, -5.0, 5.0, -5.0] * 10), np.array([0, 1, 0, 1] * 10, float))            # anti-informative signal
    assert a0 == 0.0 and abs(b0) < 1e-6                                                                        # -> constant at base rate 0.5


def test_auroc_and_mahalanobis():
    assert auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0
    assert abs(auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) - 0.5) < 1e-9
    ref = np.random.default_rng(0).normal(size=(500, 3))
    d = mahalanobis(ref, np.array([[0, 0, 0], [5, 5, 5]], float))
    assert d[1] > d[0] and d[0] < 1.0


def test_extra_builders_on_synthetic_inputs(tmp_path, monkeypatch):
    """Climate-FEVER and VitaminC adapters on tiny fake raw files; the ladder on a tiny fake SciFact."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from data import extra

    raw, out = tmp_path / "raw", tmp_path / "cases"
    (raw / "climate_fever").mkdir(parents=True)
    rows = []
    for i in range(9):
        rows.append({"claim_id": str(i), "claim": f"Climate claim number {i} about warming oceans and ice.",
                     "claim_label": i % 3,
                     "evidences": [{"evidence_id": f"e{i}{j}", "evidence_label": (i % 3) if j == 0 else 2, "article": f"Article {j}",
                                    "evidence": f"Sentence {j} about warming oceans ice and claim {i}.", "entropy": 0.0, "votes": []} for j in range(5)]})
    pq.write_table(pa.Table.from_pylist(rows), raw / "climate_fever" / "test.parquet")
    cases, golds = extra.build_climate_fever(raw, out, seed=0, n=6)
    assert len(cases) == 6 and all(len(c["initial_evidence"]) == 3 for c in cases)
    assert all((g["label"] == "insufficient_evidence") == (not g["gold_doc_ids"]) for g in golds)
    (raw / "vitaminc").mkdir(parents=True)
    with (raw / "vitaminc" / "test.jsonl").open("w") as f:
        for i in range(12):
            f.write(json.dumps({"unique_id": f"u{i}", "label": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"][i % 3], "claim": f"Claim {i}.",
                                "evidence": f"Evidence sentence {i} about page {i % 2}.", "page": f"Page {i % 2}", "revision_type": "real"}) + "\n")
    cases, golds = extra.build_vitaminc(raw, out, seed=0, n=6)
    assert len(cases) == 6 and len({c["question"] for c in cases}) == 6
    # ladder on a fake SciFact dev set
    sf = raw / "scifact" / "data"
    sf.mkdir(parents=True)
    with (sf / "corpus.jsonl").open("w") as f:
        f.write(json.dumps({"doc_id": 1, "title": "T", "abstract": ["Topic sentence about aspirin.", "Aspirin reduces infarction risk.", "Methods."], "structured": False}) + "\n")
        f.write(json.dumps({"doc_id": 2, "title": "U", "abstract": ["Vitamin D and bones.", "No effect found."], "structured": False}) + "\n")
    with (sf / "claims_dev.jsonl").open("w") as f:
        f.write(json.dumps({"id": 7, "claim": "Aspirin reduces infarction risk.", "evidence": {"1": [{"sentences": [1], "label": "SUPPORT"}]}, "cited_doc_ids": [1]}) + "\n")
        f.write(json.dumps({"id": 8, "claim": "Vitamin D cures colds.", "evidence": {}, "cited_doc_ids": [2]}) + "\n")
    monkeypatch.setattr(extra, "fetch_scifact", lambda raw_dir: sf)
    cases, golds = extra.build_ladder(raw, out)
    assert len(cases) == 1 and cases[0]["variant"] == "rationale_removed" and golds[0]["label"] == "insufficient_evidence"
    recs = {json.loads(l)["doc_id"]: json.loads(l) for l in (out / "records" / "scifact_dev_rationale_removed.jsonl").open()}
    assert "Aspirin reduces infarction risk." not in recs[1]["sentences"] and len(recs[1]["sentences"]) == 2
