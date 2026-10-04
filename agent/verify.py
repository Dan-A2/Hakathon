"""Verification: a fixed JSON tool loop of at most K tool calls, every call logged."""
from __future__ import annotations

import json
import time

from agent.llm import BaseLLM
from agent.prompts import fill, format_evidence, load_prompt
from agent.schemas import FINAL_STEP_SCHEMA, VERIFY_STEP_SCHEMA, MalformedOutput, parse_step
from agent.tools import ToolRunner
from common.config import DEFAULT_K, VERIFY_TEMPERATURE
from data.records import RecordStore

FORCE_FINAL_MSG = "Tool budget exhausted. Reply with the final answer JSON only."
MAX_CONTEXT_CHARS = 36_000          # ~9k tokens: force the final answer before the conversation outgrows small context windows
_CONTEXT_ERRORS = ("maximum context length", "context length", "input_tokens", "prompt is too long", "too many tokens",
                   "context_length_exceeded", "reduce the length")


def is_context_error(e: Exception) -> bool:
    return any(k in str(e).lower() for k in _CONTEXT_ERRORS)
RETRY_MSG = "Your reply was not valid JSON for this protocol. Reply with exactly one JSON object: a tool call or a final answer."


def build_messages(case: dict, start: dict, K: int) -> list[dict]:
    p = load_prompt("verify.txt")
    return [
        {"role": "system", "content": fill(p["system"], K=K)},
        {"role": "user", "content": fill(p["user"], claim=case["question"],
                                         evidence=format_evidence(case.get("initial_evidence", [])),
                                         verdict=start["verdict"], confidence=f"{start['confidence']:.2f}")},
    ]


def run_verify(llm: BaseLLM, case: dict, start: dict, store: RecordStore, exclude: list, K: int = DEFAULT_K,
               temperature: float = VERIFY_TEMPERATURE, seed: int = 0, calc_backend: str | None = None) -> dict:
    """Run the tool loop starting from sample A's verdict. Returns the verified outcome + logs."""
    t0 = time.time()
    tools = ToolRunner(store, exclude, [ev["doc_id"] for ev in case.get("initial_evidence", [])], calc_backend)
    messages = build_messages(case, start, K)
    llm_calls = tool_calls = tokens_in = tokens_out = 0
    malformed_turns = 0
    malformed_samples: list[str] = []
    forced = False
    final = None
    retried = False
    context_overflow = False
    while True:
        too_long = sum(len(m["content"]) for m in messages) > MAX_CONTEXT_CHARS
        if (tool_calls >= K or too_long) and not forced:
            forced = True
            messages.append({"role": "user", "content": FORCE_FINAL_MSG})
        try:
            res = llm.chat(messages, temperature=temperature, seed=seed + llm_calls, max_tokens=500, tag="verify",
                           json_schema=FINAL_STEP_SCHEMA if forced else VERIFY_STEP_SCHEMA)
        except Exception as e:  # noqa: BLE001
            if not is_context_error(e):
                raise
            context_overflow = True                      # the conversation outgrew the model: the provisional verdict stands
            break
        llm_calls += 1
        tokens_in += res.tokens_in
        tokens_out += res.tokens_out
        try:
            step = parse_step(res.text)
            if forced and "final" not in step:
                raise MalformedOutput("tool call after budget exhausted")
        except MalformedOutput:
            malformed_turns += 1
            malformed_samples.append(res.text[:300])
            if retried:            # second failure in a row: the provisional verdict stands
                break
            retried = True
            messages.append({"role": "assistant", "content": res.text})
            messages.append({"role": "user", "content": RETRY_MSG})
            continue
        retried = False
        if "final" in step:
            final = step["final"]
            messages.append({"role": "assistant", "content": json.dumps(step)})
            break
        result = tools.call(step["tool"], step["args"])
        tool_calls += 1
        messages.append({"role": "assistant", "content": json.dumps(step)})
        messages.append({"role": "user", "content": json.dumps({"tool_result": result}, ensure_ascii=False)})
        if llm_calls > K + 4:   # hard stop against runaway loops
            break

    used_provisional = final is None
    if used_provisional:
        final = {"verdict": start["verdict"], "confidence": start["confidence"],
                 "cited": [{"doc_id": d, "sentence": None} for d in start.get("cited_doc_ids", [])],
                 "rationale": start.get("rationale", "")}
    return {
        **final,
        "tool_log": tools.log,
        "tool_calls": tool_calls,
        "llm_calls": llm_calls,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "opened_doc_ids": sorted(tools.opened),
        "forced_final": forced,
        "context_overflow": context_overflow,
        "malformed_turns": malformed_turns,
        "malformed_samples": malformed_samples[:3],
        "used_provisional": used_provisional,
        "latency_s": round(time.time() - t0, 3),
    }
