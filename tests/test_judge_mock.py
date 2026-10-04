import json

from agent.build_judge import build_judge_split
from agent.llm import get_llm
from common.io import read_jsonl, write_jsonl


def test_build_judge_on_mock_cache(tmp_path):
    from tests.test_e2e_mock import _make_cases
    from agent.build_cache import build_split

    cases_dir, cache_dir = tmp_path / "cases", tmp_path / "cache"
    _make_cases(cases_dir)
    llm = get_llm("mock")
    build_split("test_id", cases_dir, cache_dir, llm, workers=2, quiet=True)
    out = build_judge_split("test_id", cases_dir, cache_dir, llm, workers=2, quiet=True)
    rows = read_jsonl(out)
    recs = read_jsonl(cache_dir / "test_id.jsonl")
    committed = [r for r in recs if r["ver"]["verdict"] in ("supported", "refuted") and r["ver"]["cited"]]
    assert len(rows) <= len(committed)
    for r in rows:
        assert r["relation"] in ("supports", "refutes", "not_addressed") and abs(sum(r["credence"].values()) - 1) < 1e-6
        assert r["quotes"] and r["judge_prompt_hash"]
    # the scorer joins the judge file and the disciplined verdict follows the judge
    from scorer.score import load_scored
    sc = {s.case_id: s for s in load_scored("test_id", cases_dir, cache_dir)}
    for r in rows:
        assert sc[r["case_id"]].judge is not None
        if sc[r["case_id"]].dver_source == "judge":
            assert sc[r["case_id"]].dver_verdict == {"supports": "supported", "refutes": "refuted", "not_addressed": "insufficient_evidence"}[r["relation"]]
