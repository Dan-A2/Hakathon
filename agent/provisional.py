"""Provisional call: two frozen samples at T=0.7 -> the six pre-action signals."""
from __future__ import annotations

import math
import time

from agent.llm import BaseLLM
from agent.prompts import fill, format_evidence, load_prompt
from agent.schemas import PROVISIONAL_SCHEMA, MalformedOutput, extract_json, normalize_provisional
from common.config import FEATURE_NAMES, PROVISIONAL_TEMPERATURE, TOK_PROB_FEATURE

FALLBACK_SAMPLE = {"verdict": "insufficient_evidence", "confidence": 0.0, "evidence_sufficiency": 0.0,
                   "cited_doc_ids": [], "rationale": ""}


def build_messages(case: dict) -> list[dict]:
    p = load_prompt("provisional.txt")
    return [
        {"role": "system", "content": p["system"]},
        {"role": "user", "content": fill(p["user"], claim=case["question"],
                                         evidence=format_evidence(case.get("initial_evidence", [])))},
    ]


def verdict_token_prob(text: str, logprobs: list[tuple[str, float]] | None, verdict: str) -> float | None:
    """Probability of the first token of the verdict value, from per-token log-probs."""
    if not logprobs:
        return None
    joined = "".join(t for t, _ in logprobs)
    m = joined.find('"verdict"')
    if m < 0:
        m = joined.find("verdict")
    if m < 0:
        return None
    pos = joined.find(verdict, m)
    if pos < 0:
        return None
    acc = 0
    for tok, lp in logprobs:
        if acc + len(tok) > pos:
            return float(math.exp(lp))
        acc += len(tok)
    return None


def run_provisional(llm: BaseLLM, case: dict, seed: int, temperature: float = PROVISIONAL_TEMPERATURE,
                    want_tok_prob: bool | None = None) -> dict:
    """One provisional sample (with a single retry on malformed JSON)."""
    if want_tok_prob is None:
        want_tok_prob = llm.supports_logprobs
    messages = build_messages(case)
    t0 = time.time()
    tokens_in = tokens_out = calls = 0
    last_text, sample, parse_ok = "", None, False
    for attempt in range(2):
        res = llm.chat(messages, temperature=temperature, seed=seed + 100 * attempt, max_tokens=400,
                       json_schema=PROVISIONAL_SCHEMA, want_logprobs=want_tok_prob, tag="provisional")
        calls += 1
        tokens_in += res.tokens_in
        tokens_out += res.tokens_out
        last_text = res.text
        try:
            sample = normalize_provisional(extract_json(res.text))
            parse_ok = True
            tok_prob = verdict_token_prob(res.text, res.logprobs, sample["verdict"]) if want_tok_prob else None
            break
        except MalformedOutput:
            continue
    if sample is None:
        sample, tok_prob = dict(FALLBACK_SAMPLE), None
    sample.update({"seed": seed, "tok_prob": tok_prob, "tokens_in": tokens_in, "tokens_out": tokens_out,
                   "llm_calls": calls, "latency_s": round(time.time() - t0, 3), "parse_ok": parse_ok,
                   "raw_text": last_text[:800]})
    return sample


def provisional_answer(a: dict, b: dict) -> dict:
    """The provisional answer is the higher-confidence sample (ties go to A)."""
    pick, src = (b, "b") if b["confidence"] > a["confidence"] else (a, "a")
    return {"verdict": pick["verdict"], "confidence": pick["confidence"],
            "cited_doc_ids": list(pick["cited_doc_ids"]), "rationale": pick["rationale"], "from": src}


def features_from_samples(a: dict, b: dict) -> dict:
    """The six signals (plus tok_prob when sample A carries one)."""
    x = {
        "bias": 1.0,
        "conf_mean": (a["confidence"] + b["confidence"]) / 2.0,
        "agree": 1.0 if a["verdict"] == b["verdict"] else 0.0,
        "suff_mean": (a["evidence_sufficiency"] + b["evidence_sufficiency"]) / 2.0,
        "conf_gap": abs(a["confidence"] - b["confidence"]),
        "says_insuff": 1.0 if "insufficient_evidence" in (a["verdict"], b["verdict"]) else 0.0,
    }
    assert list(x) == FEATURE_NAMES
    x[TOK_PROB_FEATURE] = a.get("tok_prob")
    return x
