"""The blind judge: an LLM call that sees ONLY the claim and the sentences the verifier cited.

Separating judgement from search removes the verifier's confirmation bias ("I searched, I found
something, so it must be evidence").  The judge's reading of the evidence replaces the verifier's
self-assessed verdict (agent/epistemic.py, rule 2), and its top-k token probabilities over the
three relation labels give a credence that is a real probability distribution, not a verbalised
number.  Its prompt is hashed separately so existing controllers stay valid.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache

from agent.llm import BaseLLM
from agent.prompts import fill, load_prompt
from agent.schemas import MalformedOutput, extract_json
from common.config import PROMPTS_DIR
from common.io import sha256_text

RELATIONS = ["supports", "refutes", "not_addressed"]
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"relation": {"type": "string", "enum": RELATIONS}, "confidence": {"type": "number"},
                   "reason": {"type": "string"}},
    "required": ["relation", "confidence", "reason"],
    "additionalProperties": False,
}
MAX_QUOTES = 5


@lru_cache(maxsize=None)
def judge_prompt_hash() -> str:
    return sha256_text((PROMPTS_DIR / "judge.txt").read_text(encoding="utf-8"))[:16]


def format_quotes(quotes: list[dict]) -> str:
    return "\n".join(f"[doc {q['doc_id']}, sentence {q['sentence']}] {q['text']}" for q in quotes[:MAX_QUOTES])


def build_messages(claim: str, quotes: list[dict]) -> list[dict]:
    p = load_prompt("judge.txt")
    return [{"role": "system", "content": p["system"]},
            {"role": "user", "content": fill(p["user"], claim=claim, quotes=format_quotes(quotes))}]


def _normalize_relation(v) -> str | None:
    if not isinstance(v, str):
        return None
    k = v.strip().lower().replace(" ", "_").replace("-", "_")
    for r in RELATIONS:
        if k == r or k.startswith(r[:3]):
            return r
    return None


def credence_from_logprobs(text: str, top_logprobs, relation: str, confidence: float) -> dict[str, float]:
    """Probability over the three relations from the top-k alternatives at the relation token."""
    fallback = {r: (confidence if r == relation else (1 - confidence) / 2) for r in RELATIONS}
    if not top_logprobs:
        return fallback
    joined = "".join(alts[0][0] for alts in top_logprobs if alts)
    m = joined.find('"relation"')
    pos = joined.find(relation, m if m >= 0 else 0)
    if pos < 0:
        return fallback
    acc, idx = 0, None
    for i, alts in enumerate(top_logprobs):
        tok = alts[0][0] if alts else ""
        if acc + len(tok) > pos:
            idx = i
            break
        acc += len(tok)
    if idx is None or not top_logprobs[idx]:
        return fallback
    mass = {r: 0.0 for r in RELATIONS}
    for tok, lp in top_logprobs[idx]:
        r = _normalize_relation(tok.strip().strip('"').strip())
        if r:
            mass[r] += math.exp(lp)
    total = sum(mass.values())
    if total <= 0:
        return fallback
    return {r: v / total for r, v in mass.items()}


def run_judge(llm: BaseLLM, claim: str, quotes: list[dict], seed: int = 0) -> dict:
    """One evidence-only reading. Malformed output gets one retry, then 'not_addressed' with low confidence."""
    t0 = time.time()
    messages = build_messages(claim, quotes)
    tokens_in = tokens_out = calls = 0
    last = ""
    for attempt in range(2):
        res = llm.chat(messages, temperature=0.0, seed=seed + 100 * attempt, max_tokens=200, json_schema=JUDGE_SCHEMA,
                       want_logprobs=llm.supports_logprobs, top_k_logprobs=5, tag="judge")
        calls += 1
        tokens_in += res.tokens_in
        tokens_out += res.tokens_out
        last = res.text
        try:
            obj = extract_json(res.text)
            rel = _normalize_relation(obj.get("relation"))
            if rel is None:
                raise MalformedOutput("bad relation")
            conf = min(1.0, max(0.0, float(obj.get("confidence", 0.5))))
            return {"relation": rel, "confidence": conf, "reason": str(obj.get("reason", ""))[:300],
                    "credence": credence_from_logprobs(res.text, res.top_logprobs, rel, conf),
                    "n_quotes": len(quotes[:MAX_QUOTES]), "tokens": tokens_in + tokens_out, "llm_calls": calls,
                    "latency_s": round(time.time() - t0, 3), "parse_ok": True, "raw_text": res.text[:400]}
        except (MalformedOutput, ValueError, TypeError):
            continue
    return {"relation": "not_addressed", "confidence": 0.5, "reason": "", "credence": {r: 1 / 3 for r in RELATIONS},
            "n_quotes": len(quotes[:MAX_QUOTES]), "tokens": tokens_in + tokens_out, "llm_calls": calls,
            "latency_s": round(time.time() - t0, 3), "parse_ok": False, "raw_text": last[:400]}
