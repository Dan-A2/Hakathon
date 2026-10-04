"""Health report for a cache split WITHOUT gold labels (the Gate-1 check):

    python -m agent.cache_report --split train [--cache-dir art/cache]

Shows parse/retry rates, verdict and confidence distributions, tool usage and costs, so you
can tell whether the frozen LLM is producing valid JSON and sensible signals before
spending the full budget.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from common import config as C
from common.io import read_jsonl


def report(rows: list[dict]) -> str:
    n = len(rows)
    if not n:
        return "no records"
    samples = [s for r in rows for s in (r["a"], r["b"])]
    out = [f"records: {n}   llm: {rows[0].get('llm')}   prompt_hash: {rows[0].get('prompt_hash')}   K: {rows[0].get('K')}"]
    out.append(f"provisional JSON: {sum(s['parse_ok'] for s in samples)}/{len(samples)} parsed; "
               f"{sum(s['llm_calls'] > 1 for s in samples)} needed a retry; "
               f"{sum(not s['parse_ok'] for s in samples)} fell back after the retry")
    out.append(f"provisional verdicts: {dict(Counter(r['prov']['verdict'] for r in rows))}")
    conf = np.array([s["confidence"] for s in samples])
    suff = np.array([s["evidence_sufficiency"] for s in samples])
    out.append(f"confidence mean {conf.mean():.2f} (sd {conf.std():.2f}); sufficiency mean {suff.mean():.2f} (sd {suff.std():.2f}); "
               f"samples agree in {np.mean([r['x']['agree'] for r in rows]) * 100:.0f}% of cases; "
               f"says_insuff in {np.mean([r['x']['says_insuff'] for r in rows]) * 100:.0f}%")
    tp = [r["x"].get(C.TOK_PROB_FEATURE) for r in rows]
    out.append(f"tok_prob present in {sum(t is not None for t in tp)}/{n} records")
    ver = [r["ver"] for r in rows]
    out.append(f"verify: verdicts {dict(Counter(v['verdict'] for v in ver))}; tool calls {dict(sorted(Counter(v['tool_calls'] for v in ver).items()))}; "
               f"tools {dict(Counter(t['tool'] for v in ver for t in v['tool_log']))}; tool errors {sum(not t['ok'] for v in ver for t in v['tool_log'])}")
    out.append(f"verify: malformed turns in {sum(v['malformed_turns'] > 0 for v in ver)} cases, provisional stood in "
               f"{sum(v['used_provisional'] for v in ver)}, forced final in {sum(v['forced_final'] for v in ver)}, "
               f"citations in {sum(bool(v['cited']) for v in ver)} cases")
    out.append(f"verify flips vs provisional: {sum(v['verdict'] != r['prov']['verdict'] for v, r in zip(ver, rows))} cases changed verdict")
    tok = np.array([sum(s["tokens_in"] + s["tokens_out"] for s in (r["a"], r["b"])) + r["ver"]["tokens_in"] + r["ver"]["tokens_out"] for r in rows])
    wall = np.array([r["wall_s"] for r in rows])
    out.append(f"cost per case: {tok.mean():.0f} tokens, {wall.mean():.1f} s wall (all three actions); "
               f"LLM calls per case {np.mean([r['a']['llm_calls'] + r['b']['llm_calls'] + r['ver']['llm_calls'] for r in rows]):.1f}")
    flags = Counter(k for r in rows for k, v in r.get("flags", {}).items() if v)
    out.append(f"flags: {dict(flags) or 'none'}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache-dir", default=str(C.CACHE_DIR))
    args = ap.parse_args(argv)
    print(report(read_jsonl(Path(args.cache_dir) / f"{args.split}.jsonl")))


if __name__ == "__main__":
    main()
