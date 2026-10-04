"""The verify loop's edge paths: forced final after K calls, malformed JSON retry, provisional standing."""
import json

from agent.llm import BaseLLM, ChatResult
from agent.verify import FORCE_FINAL_MSG, RETRY_MSG, run_verify
from data.records import RecordStore

DOCS = [{"doc_id": 1, "title": "Aspirin", "sentences": ["Aspirin reduces myocardial infarction risk.", "N=1000."]},
        {"doc_id": 2, "title": "Vitamin D", "sentences": ["Vitamin D improves bone density."]}]
CASE = {"case_id": "c", "question": "Aspirin reduces myocardial infarction risk.",
        "initial_evidence": [{"doc_id": 2, "title": "Vitamin D", "text": "Vitamin D improves bone density."}]}
START = {"verdict": "insufficient_evidence", "confidence": 0.4, "cited_doc_ids": [], "rationale": ""}


class Scripted(BaseLLM):
    """Replays a fixed list of replies; records what it was asked."""
    backend, model_id, revision = "scripted", "s", "0"

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, **kw):
        self.seen.append(messages[-1]["content"])
        text = self.replies.pop(0) if self.replies else '{"final": {"verdict": "supported", "confidence": 0.9, "cited": [], "rationale": "fallback"}}'
        return ChatResult(text=text, tokens_in=10, tokens_out=5)


def test_forced_final_after_k_tool_calls():
    search = json.dumps({"tool": "search_records", "args": {"query": "aspirin"}})
    llm = Scripted([search, search, search, search, search,
                    '{"final": {"verdict": "supported", "confidence": 0.8, "cited": [{"doc_id": 1, "sentence": 0}], "rationale": "ok"}}'])
    out = run_verify(llm, CASE, START, RecordStore(DOCS), exclude=[], K=4)
    assert out["tool_calls"] == 4 and out["forced_final"] and out["verdict"] == "supported"
    assert any(FORCE_FINAL_MSG == m for m in llm.seen)
    assert out["llm_calls"] == 6 and out["malformed_turns"] == 1        # the 5th tool call after the budget counts as malformed


def test_malformed_then_recovers():
    llm = Scripted(["sure, let me search", json.dumps({"tool": "read_record", "args": {"doc_id": 1}}),
                    '{"final": {"verdict": "supported", "confidence": 0.7, "cited": [{"doc_id": 1, "sentence": 0}], "rationale": ""}}'])
    out = run_verify(llm, CASE, START, RecordStore(DOCS), exclude=[], K=4)
    assert out["malformed_turns"] == 1 and not out["used_provisional"] and out["opened_doc_ids"] == ["1"]
    assert RETRY_MSG in llm.seen


def test_two_malformed_in_a_row_keeps_provisional():
    llm = Scripted(["garbage", "more garbage"])
    out = run_verify(llm, CASE, START, RecordStore(DOCS), exclude=[], K=4)
    assert out["used_provisional"] and out["verdict"] == "insufficient_evidence" and out["confidence"] == 0.4
    assert out["malformed_turns"] == 2 and out["tool_calls"] == 0


def test_read_record_of_ablated_doc_is_refused_and_logged():
    llm = Scripted([json.dumps({"tool": "read_record", "args": {"doc_id": 1}}),
                    '{"final": {"verdict": "insufficient_evidence", "confidence": 0.6, "cited": [], "rationale": ""}}'])
    out = run_verify(llm, CASE, START, RecordStore(DOCS), exclude=[1], K=4)
    assert out["tool_log"][0]["ok"] is False and out["opened_doc_ids"] == []
