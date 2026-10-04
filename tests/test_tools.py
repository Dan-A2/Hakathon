import pytest

from agent.tools import SafeCalcError, ToolRunner, safe_calculate
from data.records import RecordStore

DOCS = [
    {"doc_id": 1, "title": "Aspirin and heart attacks", "sentences": ["Aspirin reduces myocardial infarction risk.", "A trial of 1000 patients."]},
    {"doc_id": 2, "title": "Vitamin D and bone density", "sentences": ["Vitamin D improves bone density in older adults."]},
    {"doc_id": 3, "title": "Aspirin dosing", "sentences": ["Low dose aspirin is 81 mg."]},
]


def test_safe_calculate_arithmetic():
    assert safe_calculate("2 + 3 * 4") == "14"
    assert safe_calculate("32 / 100 * 250") == "80"
    assert safe_calculate("sqrt(16) + max(1, 2, 3)") == "7"
    assert safe_calculate("2 ^ 10") == "1024"


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('/etc/passwd')", "2 ** 100000", "x = 1", "lambda: 1", "[1,2][0].__class__"])
def test_safe_calculate_rejects_code(expr):
    with pytest.raises(SafeCalcError):
        safe_calculate(expr)


def test_tool_runner_respects_exclusions_and_logs():
    store = RecordStore(DOCS)
    tools = ToolRunner(store, exclude=[1], shown_doc_ids=[2], calc_backend="local")
    hits = tools.call("search_records", {"query": "aspirin myocardial infarction"})["results"]
    assert all(h["doc_id"] != 1 for h in hits) and hits[0]["doc_id"] == 3
    assert "error" in tools.call("read_record", {"doc_id": 1})          # ablated doc is not accessible
    rec = tools.call("read_record", {"doc_id": 3})
    assert rec["sentences"][0]["i"] == 0 and "3" in tools.opened
    assert tools.call("calculate", {"expression": "81 * 2"})["result"] == "162"
    assert "error" in tools.call("nonsense", {})
    assert [t["tool"] for t in tools.log] == ["search_records", "read_record", "read_record", "calculate", "nonsense"]
    assert tools.accessible_cited == {"2", "3"}


def test_read_record_output_is_bounded():
    from agent.tools import MAX_READ_SENTENCES, MAX_SENTENCE_CHARS

    long_doc = {"doc_id": 7, "title": "Long", "sentences": [f"Sentence {i} " + "w" * 600 for i in range(60)]}
    tools = ToolRunner(RecordStore(DOCS + [long_doc]), exclude=[], shown_doc_ids=[], calc_backend="local")
    rec = tools.call("read_record", {"doc_id": 7})
    assert len(rec["sentences"]) == MAX_READ_SENTENCES and all(len(s["text"]) <= MAX_SENTENCE_CHARS for s in rec["sentences"])
    assert rec["sentences"][-1]["i"] == MAX_READ_SENTENCES - 1 and "more sentences" in rec["note"]
