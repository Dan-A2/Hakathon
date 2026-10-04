"""JSON schemas for the model's outputs and tolerant parsing / normalisation."""
from __future__ import annotations

import json
import re
from typing import Any

from common.config import LABELS

PROVISIONAL_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": LABELS},
        "confidence": {"type": "number"},
        "evidence_sufficiency": {"type": "number"},
        "cited_doc_ids": {"type": "array", "items": {"type": "integer"}},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "confidence", "evidence_sufficiency", "cited_doc_ids", "rationale"],
    "additionalProperties": False,
}

_VERDICT_SYNONYMS = {
    "supported": "supported", "support": "supported", "supports": "supported", "true": "supported",
    "refuted": "refuted", "refute": "refuted", "refutes": "refuted", "contradict": "refuted",
    "contradicted": "refuted", "contradicts": "refuted", "false": "refuted",
    "insufficient_evidence": "insufficient_evidence", "insufficient": "insufficient_evidence",
    "insufficient evidence": "insufficient_evidence", "not enough info": "insufficient_evidence",
    "not_enough_info": "insufficient_evidence", "nei": "insufficient_evidence", "noinfo": "insufficient_evidence",
    "neutral": "insufficient_evidence", "unknown": "insufficient_evidence",
}


class MalformedOutput(ValueError):
    """Raised when the model output cannot be parsed into the expected JSON object."""


def extract_json(text: str) -> dict:
    """Return the first JSON object found in text (tolerates code fences and prose)."""
    if text is None:
        raise MalformedOutput("empty output")
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.M).strip()
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", s):
        try:
            obj, _ = dec.raw_decode(s[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise MalformedOutput(f"no JSON object in output: {s[:120]!r}")


def normalize_verdict(v: Any) -> str | None:
    if not isinstance(v, str):
        return None
    key = v.strip().lower().replace("-", "_")
    if key in _VERDICT_SYNONYMS:
        return _VERDICT_SYNONYMS[key]
    key2 = key.replace("_", " ")
    return _VERDICT_SYNONYMS.get(key2)


def _clamp01(x: Any, default: float = 0.5) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    if f > 1.0 and f <= 100.0:   # percentages
        f = f / 100.0
    return min(1.0, max(0.0, f))


def _as_id(x: Any):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        t = x.strip()
        if re.fullmatch(r"-?\d+", t):
            return int(t)
        return t or None
    return None


def normalize_provisional(obj: dict) -> dict:
    verdict = normalize_verdict(obj.get("verdict"))
    if verdict is None:
        raise MalformedOutput(f"bad verdict: {obj.get('verdict')!r}")
    cited = obj.get("cited_doc_ids", [])
    if not isinstance(cited, list):
        cited = [cited]
    ids = [i for i in (_as_id(c) for c in cited) if i is not None]
    return {
        "verdict": verdict,
        "confidence": _clamp01(obj.get("confidence")),
        "evidence_sufficiency": _clamp01(obj.get("evidence_sufficiency")),
        "cited_doc_ids": ids,
        "rationale": str(obj.get("rationale", ""))[:600],
    }


def normalize_final(obj: dict) -> dict:
    verdict = normalize_verdict(obj.get("verdict"))
    if verdict is None:
        raise MalformedOutput(f"bad final verdict: {obj.get('verdict')!r}")
    cited_raw = obj.get("cited", obj.get("cited_doc_ids", []))
    cited = []
    if isinstance(cited_raw, list):
        for c in cited_raw:
            if isinstance(c, dict):
                d = _as_id(c.get("doc_id"))
                s = c.get("sentence", c.get("sentence_index", c.get("idx")))
                try:
                    s = int(s) if s is not None else None
                except (TypeError, ValueError):
                    s = None
                if d is not None:
                    cited.append({"doc_id": d, "sentence": s})
            else:
                d = _as_id(c)
                if d is not None:
                    cited.append({"doc_id": d, "sentence": None})
    return {
        "verdict": verdict,
        "confidence": _clamp01(obj.get("confidence")),
        "cited": cited,
        "rationale": str(obj.get("rationale", ""))[:600],
    }


def parse_step(text: str) -> dict:
    """Parse one verify-loop turn: {'tool':..., 'args':...} or {'final': {...}}."""
    obj = extract_json(text)
    if "final" in obj and isinstance(obj["final"], dict):
        return {"final": normalize_final(obj["final"])}
    if "verdict" in obj and "tool" not in obj:          # model skipped the wrapper
        return {"final": normalize_final(obj)}
    tool = obj.get("tool") or obj.get("name")
    if isinstance(tool, str):
        args = obj.get("args") or obj.get("arguments") or obj.get("input") or {}
        if not isinstance(args, dict):
            args = {"value": args}
        return {"tool": tool.strip(), "args": args}
    raise MalformedOutput(f"neither tool call nor final: {list(obj)[:5]}")
