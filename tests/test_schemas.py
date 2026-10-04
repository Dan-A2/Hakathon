import pytest

from agent.schemas import MalformedOutput, extract_json, normalize_final, normalize_provisional, normalize_verdict, parse_step


def test_extract_json_tolerates_fences_and_prose():
    txt = 'Sure! ```json\n{"verdict": "SUPPORT", "confidence": 0.8, "evidence_sufficiency": 0.7, "cited_doc_ids": ["12"], "rationale": "x"}\n```'
    d = normalize_provisional(extract_json(txt))
    assert d["verdict"] == "supported" and d["cited_doc_ids"] == [12] and d["confidence"] == 0.8


def test_verdict_synonyms_and_clamping():
    assert normalize_verdict("Not Enough Info") == "insufficient_evidence"
    assert normalize_verdict("CONTRADICT") == "refuted"
    assert normalize_verdict("maybe") is None
    d = normalize_provisional({"verdict": "refuted", "confidence": 85, "evidence_sufficiency": -1, "cited_doc_ids": 5})
    assert d["confidence"] == 0.85 and d["evidence_sufficiency"] == 0.0 and d["cited_doc_ids"] == [5]


def test_parse_step_variants():
    assert parse_step('{"tool": "search_records", "args": {"query": "x"}}') == {"tool": "search_records", "args": {"query": "x"}}
    f = parse_step('{"final": {"verdict": "refuted", "confidence": 0.6, "cited": [{"doc_id": 3, "sentence": 2}], "rationale": ""}}')
    assert f["final"]["cited"] == [{"doc_id": 3, "sentence": 2}]
    bare = parse_step('{"verdict": "supported", "confidence": 0.9, "cited": [7]}')
    assert bare["final"]["cited"] == [{"doc_id": 7, "sentence": None}]
    with pytest.raises(MalformedOutput):
        parse_step("let me think about this")
    with pytest.raises(MalformedOutput):
        normalize_final({"verdict": "dunno"})


def test_gemma_native_tool_call_syntax():
    a = parse_step('<|tool_call>call:search_records{query:<|"|>SIDS deaths age<|"|>}<tool_call|>')
    assert a == {"tool": "search_records", "args": {"query": "SIDS deaths age"}}
    b = parse_step('<|tool_call>call:read_record{doc_id: 12345}<tool_call|>')
    assert b == {"tool": "read_record", "args": {"doc_id": 12345}}
    c = parse_step('<|tool_call>call:search_records{query: "0-dimensional biomaterials"}<tool_call|>')
    assert c["args"]["query"] == "0-dimensional biomaterials"
