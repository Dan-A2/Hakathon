"""Deterministic offline stand-in for the frozen LLM.

It reads the claim and abstracts out of the prompt, scores lexical overlap, and emits
plausible JSON (with seeded noise, occasional malformed output, and rare fabricated
citations) so every downstream component - features, tool loop, scorer, trainer,
figures - can be exercised without an API key.  Its verdicts are NOT meaningful;
every cache record carries backend="mock" so results cannot be mistaken for real ones.
"""
from __future__ import annotations

import json
import re
import zlib

import numpy as np

from agent.llm import BaseLLM, ChatResult
from data.records import tokenize

_NEG = {"not", "no", "lack", "lacks", "lacking", "without", "reduce", "reduces", "reduced", "decrease",
        "decreases", "decreased", "inhibit", "inhibits", "lower", "lowers", "fail", "fails", "absent", "never"}


def _rng(seed, *parts) -> np.random.Generator:
    key = zlib.crc32(("|".join(str(p) for p in parts)).encode()) ^ (0 if seed is None else int(seed) * 2654435761)
    return np.random.default_rng(key % (2**32))


def _overlap(claim: str, text: str) -> float:
    a, b = set(tokenize(claim)), set(tokenize(text))
    return len(a & b) / max(1, len(a))


def _neg_parity(text: str) -> int:
    return sum(1 for t in tokenize(text) if t in _NEG) % 2


class MockLLM(BaseLLM):
    backend = "mock"
    model_id = "mock-overlap-v1"
    revision = "mock"
    supports_logprobs = True

    def __init__(self, malformed_rate: float = 0.02, fabricate_rate: float = 0.03):
        self.malformed_rate = malformed_rate
        self.fabricate_rate = fabricate_rate

    # ---- helpers ------------------------------------------------------------------------
    @staticmethod
    def _last_user(messages):
        return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    @staticmethod
    def _claim(text: str) -> str:
        m = re.search(r"^Claim: (.*)$", text, re.M)
        return m.group(1).strip() if m else text[:200]

    @staticmethod
    def _docs(text: str) -> list[tuple[int, str]]:
        return [(int(d), t) for d, t in re.findall(r"\[doc_id (\d+)\][^\n]*\n(.*?)(?=\n\n\[doc_id|\n\nReturn|\n\nYour provisional|\Z)", text, re.S)]

    def _result(self, obj, messages, seed) -> ChatResult:
        text = json.dumps(obj)
        n_in = sum(len(m["content"].split()) for m in messages)
        lps = None
        if isinstance(obj, dict) and "verdict" in obj:
            # fake per-token log-probs so the tok_prob feature path is exercised
            r = _rng(seed, text)
            toks = re.findall(r'"[^"]*"|[\{\}\[\]:,]|\S+', text)
            lps = [(t, float(np.log(np.clip(r.uniform(0.55, 0.99), 1e-6, 1.0)))) for t in toks]
        return ChatResult(text=text, tokens_in=n_in, tokens_out=len(text.split()), latency_s=0.0,
                          logprobs=lps, model=self.model_id)

    # ---- tasks --------------------------------------------------------------------------
    def chat(self, messages, *, temperature=0.7, seed=None, max_tokens=700, json_schema=None,
             want_logprobs=False, tag=None) -> ChatResult:
        user = self._last_user(messages)
        first_user = next((m["content"] for m in messages if m["role"] == "user"), user)
        claim = self._claim(first_user)
        if tag == "verify" or any("tool_result" in m["content"] for m in messages if m["role"] == "user") \
                or "provisional verdict" in first_user:
            return self._verify(messages, claim, first_user, seed)
        return self._provisional(messages, claim, first_user, seed, temperature)

    def _provisional(self, messages, claim, user, seed, temperature):
        r = _rng(seed, "prov", claim, user[-300:])
        if r.random() < self.malformed_rate:
            return ChatResult(text="I believe the claim is probably supported by the first abstract.",
                              tokens_in=100, tokens_out=12, model=self.model_id)
        docs = self._docs(user)
        scores = [(_overlap(claim, t), d, t) for d, t in docs]
        best = max(scores, default=(0.0, None, ""))
        s, best_id, best_text = best
        noise = r.normal(0, 0.12) * max(temperature, 0.1) / 0.7
        if best_id is None or s + noise < 0.30:
            verdict = "insufficient_evidence"
            conf = float(np.clip(0.45 + 0.8 * (0.30 - s) + noise, 0.2, 0.95))
            suff = float(np.clip(s + 0.5 * noise, 0.0, 1.0))
            cited = [best_id] if (best_id is not None and r.random() < 0.2) else []
        else:
            verdict = "refuted" if _neg_parity(claim) != _neg_parity(best_text) else "supported"
            if r.random() < 0.25:
                verdict = "supported" if verdict == "refuted" else "refuted"
            conf = float(np.clip(0.40 + s + noise, 0.3, 0.98))
            suff = float(np.clip(0.4 + s + 0.3 * noise, 0.0, 1.0))
            cited = [best_id]
        obj = {"verdict": verdict, "confidence": round(conf, 3), "evidence_sufficiency": round(suff, 3),
               "cited_doc_ids": cited, "rationale": "Mock rationale from lexical overlap."}
        return self._result(obj, messages, seed)

    def _verify(self, messages, claim, first_user, seed):
        tool_results = [m["content"] for m in messages if m["role"] == "user" and "tool_result" in m["content"]]
        forced = any("budget exhausted" in m["content"].lower() for m in messages if m["role"] == "user")
        retry = any("not valid JSON" in m["content"] for m in messages if m["role"] == "user")
        r = _rng(seed, "ver", claim, len(tool_results), forced)
        if not retry and r.random() < self.malformed_rate:
            return ChatResult(text="Let me search for that: search_records(query=...)", tokens_in=100,
                              tokens_out=10, model=self.model_id)
        n = len(tool_results)
        if not forced and n == 0:
            return self._result({"tool": "search_records", "args": {"query": claim}}, messages, seed)
        if not forced and n == 1:
            try:
                res = json.loads(tool_results[-1])["tool_result"]
                hits = res.get("results", [])
            except Exception:
                hits = []
            if hits:
                return self._result({"tool": "read_record", "args": {"doc_id": hits[0]["doc_id"]}}, messages, seed)
            return self._result({"final": {"verdict": "insufficient_evidence", "confidence": 0.6, "cited": [],
                                           "rationale": "Mock: nothing found."}}, messages, seed)
        if not forced and n == 2 and re.search(r"\d", claim) and r.random() < 0.3:
            return self._result({"tool": "calculate", "args": {"expression": "32 / 100 * 250"}}, messages, seed)
        # final: judge from the record we read (if any)
        doc_id, sentences = None, []
        for tr in tool_results:
            try:
                res = json.loads(tr)["tool_result"]
            except Exception:
                continue
            if "sentences" in res:
                doc_id, sentences = res["doc_id"], res["sentences"]
        if doc_id is None or not sentences:
            obj = {"verdict": "insufficient_evidence", "confidence": 0.55, "cited": [], "rationale": "Mock: no record read."}
            return self._result({"final": obj}, messages, seed)
        best_i, best_s = 0, -1.0
        for s in sentences:
            o = _overlap(claim, s["text"])
            if o > best_s:
                best_i, best_s = s["i"], o
        noise = r.normal(0, 0.08)
        if best_s + noise < 0.25:
            obj = {"verdict": "insufficient_evidence", "confidence": round(float(np.clip(0.5 + (0.25 - best_s), 0.3, 0.9)), 3),
                   "cited": [], "rationale": "Mock: record does not settle the claim."}
        else:
            verdict = "refuted" if _neg_parity(claim) != _neg_parity(" ".join(s["text"] for s in sentences)) else "supported"
            if r.random() < 0.2:
                verdict = "supported" if verdict == "refuted" else "refuted"
            cited_doc = doc_id + 1 if r.random() < self.fabricate_rate else doc_id   # rare fabricated citation
            obj = {"verdict": verdict, "confidence": round(float(np.clip(0.5 + best_s + noise, 0.3, 0.98)), 3),
                   "cited": [{"doc_id": cited_doc, "sentence": best_i}], "rationale": "Mock: sentence overlap."}
        return self._result({"final": obj}, messages, seed)
