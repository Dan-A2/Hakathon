"""Verification tools with a complete call log.

search_records / read_record operate on the case's own record store (ablated twins never
see their gold abstracts).  calculate runs LLM-written arithmetic either in a Modal
Sandbox (no network, 10 s timeout) or, locally, through an AST-whitelisted evaluator
that accepts arithmetic and math functions only - never arbitrary Python.
"""
from __future__ import annotations

import ast
import math
import operator as op
import time
from typing import Any

from common import config as C
from data.records import RecordStore

MAX_EXPR_LEN = 300
MAX_READ_SENTENCES = 25        # read_record returns at most this many sentences (indices preserved, rest noted)
MAX_SENTENCE_CHARS = 400
MAX_FIRST_SENTENCE_CHARS = 240
MAX_RESULT_CHARS = 300
_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod, ast.Pow: None}
_UN = {ast.USub: op.neg, ast.UAdd: op.pos}
_CMP = {ast.Lt: op.lt, ast.Gt: op.gt, ast.LtE: op.le, ast.GtE: op.ge, ast.Eq: op.eq, ast.NotEq: op.ne}
_FUNCS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "fabs": math.fabs,
}
_CONSTS = {"pi": math.pi, "e": math.e, "inf": math.inf, "True": True, "False": False}


class SafeCalcError(ValueError):
    pass


def _ev(node):
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node.value, bool):
            return node.value
        raise SafeCalcError("only numbers are allowed")
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise SafeCalcError(f"unknown name {node.id!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_ev(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(b) > 10_000 or (isinstance(a, int) and abs(a) > 1 and abs(b) > 1_000):
                raise SafeCalcError("exponent too large")
            return a ** b
        return _BIN[type(node.op)](a, b)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
        return _CMP[type(node.ops[0])](_ev(node.left), _ev(node.comparators[0]))
    if isinstance(node, ast.BoolOp):
        vals = [_ev(v) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.IfExp):
        return _ev(node.body) if _ev(node.test) else _ev(node.orelse)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_ev(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        args = [_ev(a) for a in node.args]
        if len(args) == 1 and isinstance(args[0], list):
            args = args if node.func.id == "sum" else args[0]
        return _FUNCS[node.func.id](*args)
    raise SafeCalcError(f"unsupported syntax: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """Evaluate an arithmetic expression without executing arbitrary code."""
    expr = (expression or "").strip().replace("^", "**")
    if not expr:
        raise SafeCalcError("empty expression")
    if len(expr) > MAX_EXPR_LEN:
        raise SafeCalcError("expression too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SafeCalcError(f"syntax error: {e.msg}") from e
    val = _ev(tree)
    if isinstance(val, float):
        return f"{val:.10g}"
    return str(val)


def modal_calculate(expression: str) -> str:
    """Run the expression in a network-less Modal Sandbox (used when KWTC_CALC_BACKEND=modal)."""
    from modal_app import run_calc  # lazy: modal is only needed in that mode

    code = "import math\nfrom math import *\nprint(" + expression.strip() + ")"
    return run_calc.remote(code)


class ToolRunner:
    """Executes tool calls for one case and records every call."""

    TOOLS = ("search_records", "read_record", "calculate")

    def __init__(self, store: RecordStore, exclude: list, shown_doc_ids: list, calc_backend: str | None = None,
                 search_k: int = C.SEARCH_K):
        self.store = store
        self.exclude = list(exclude)
        self.shown = {str(d) for d in shown_doc_ids}
        self.opened: set[str] = set()
        self.log: list[dict] = []
        self.calc_backend = calc_backend or C.CALC_BACKEND
        self.search_k = search_k

    @property
    def accessible_cited(self) -> set[str]:
        return self.shown | self.opened

    def call(self, name: str, args: dict) -> dict:
        t0 = time.time()
        try:
            if name == "search_records":
                q = str(args.get("query") or args.get("q") or args.get("value") or "")
                hits = self.store.search(q, k=self.search_k, exclude=self.exclude)
                result = {"results": [{"doc_id": h["doc_id"], "title": h["title"][:MAX_SENTENCE_CHARS],
                                       "first_sentence": h["first_sentence"][:MAX_FIRST_SENTENCE_CHARS]} for h in hits]}
                summary = {"query": q[:80], "doc_ids": [h["doc_id"] for h in hits]}
            elif name == "read_record":
                doc_id = args.get("doc_id", args.get("id", args.get("value")))
                rec = self.store.get(doc_id, exclude=self.exclude) if doc_id is not None else None
                if rec is None:
                    result = {"error": f"record {doc_id} not found in this case's record store"}
                    summary = {"doc_id": doc_id, "found": False}
                else:
                    self.opened.add(str(rec["doc_id"]))
                    sents = rec["sentences"]
                    result = {"doc_id": rec["doc_id"], "title": rec["title"][:MAX_SENTENCE_CHARS],
                              "sentences": [{"i": i, "text": s[:MAX_SENTENCE_CHARS]} for i, s in enumerate(sents[:MAX_READ_SENTENCES])]}
                    if len(sents) > MAX_READ_SENTENCES:
                        result["note"] = f"{len(sents) - MAX_READ_SENTENCES} more sentences not shown"
                    summary = {"doc_id": rec["doc_id"], "found": True, "n_sentences": len(rec["sentences"])}
            elif name == "calculate":
                expr = str(args.get("expression") or args.get("expr") or args.get("code") or args.get("value") or "")
                try:
                    out = modal_calculate(expr) if self.calc_backend == "modal" else safe_calculate(expr)
                    result = {"result": out.strip()[:MAX_RESULT_CHARS]}
                except Exception as e:   # noqa: BLE001 - surface any calculator failure to the model
                    result = {"error": f"calculation failed: {e}"}
                summary = {"expression": expr[:80], "ok": "result" in result}
            else:
                result = {"error": f"unknown tool {name!r}; available: {list(self.TOOLS)}"}
                summary = {}
        except Exception as e:   # noqa: BLE001
            result, summary = {"error": f"tool failure: {e}"}, {}
        self.log.append({"tool": name, "args": args, "ok": "error" not in result, "summary": summary,
                         "t_s": round(time.time() - t0, 4)})
        return result
