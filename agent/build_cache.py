"""Stage 1: the counterfactual cache. All LLM calls happen here.

    python -m agent.build_cache --split train [--limit 20] [--workers 8]

For every case we store both provisional samples, the features, and the verified
outcome, so every policy can later be scored by replay without new LLM calls.
Per-case JSON files make the build idempotent and resumable; the split is merged
into <cache_dir>/<split>.jsonl at the end.  Gold labels are never read here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent.llm import BaseLLM, get_llm
from agent.prompts import prompt_hash
from agent.provisional import features_from_samples, provisional_answer, run_provisional
from agent.verify import run_verify
from common import config as C
from common.io import ensure_dir, load_json, read_jsonl, save_json, write_jsonl
from data.records import StoreRegistry


def build_case(case: dict, llm: BaseLLM, registry: StoreRegistry, K: int = C.DEFAULT_K,
               want_tok_prob: bool | None = None, calc_backend: str | None = None) -> dict:
    t0 = time.time()
    store, exclude = registry.for_case(case)
    a = run_provisional(llm, case, seed=0, want_tok_prob=want_tok_prob)
    b = run_provisional(llm, case, seed=1, want_tok_prob=want_tok_prob)
    x = features_from_samples(a, b)
    ver = run_verify(llm, case, a, store, exclude, K=K, calc_backend=calc_backend)
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "group_id": case["group_id"],
        "variant": case.get("variant", "base"),
        "parent_id": case.get("parent_id"),
        "shown_doc_ids": [ev["doc_id"] for ev in case.get("initial_evidence", [])],
        "a": a,
        "b": b,
        "prov": provisional_answer(a, b),
        "x": x,
        "ver": ver,
        "flags": {"prov_malformed": not (a["parse_ok"] and b["parse_ok"]),
                  "ver_malformed": ver["malformed_turns"] > 0, "ver_used_provisional": ver["used_provisional"]},
        "llm": llm.identity(),
        "prompt_hash": prompt_hash(),
        "K": K,
        "wall_s": round(time.time() - t0, 3),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def case_cache_path(cache_dir: Path, split: str, case_id: str) -> Path:
    return cache_dir / split / f"{case_id}.json"


def merge_split(cache_dir: Path, split: str, case_ids: list[str] | None = None) -> Path:
    d = cache_dir / split
    rows = []
    files = sorted(d.glob("*.json")) if d.exists() else []
    if case_ids is not None:
        wanted = set(case_ids)
        files = [f for f in files if f.stem in wanted]
    for f in files:
        rows.append(load_json(f))
    out = cache_dir / f"{split}.jsonl"
    write_jsonl(out, rows)
    return out


def build_split(split: str, cases_dir: Path, cache_dir: Path, llm: BaseLLM, workers: int = 4,
                limit: int | None = None, K: int = C.DEFAULT_K, want_tok_prob: bool | None = None,
                calc_backend: str | None = None, quiet: bool = False) -> Path:
    cases = read_jsonl(cases_dir / f"{split}.jsonl")
    if limit:
        cases = cases[:limit]
    registry = StoreRegistry(cases_dir)
    ensure_dir(cache_dir / split)
    todo = [c for c in cases if not case_cache_path(cache_dir, split, c["case_id"]).exists()]
    if not quiet:
        print(f"[cache] {split}: {len(cases)} cases, {len(todo)} to build, backend={llm.identity()}", file=sys.stderr)
    errors = 0
    t0 = time.time()

    def work(case):
        rec = build_case(case, llm, registry, K=K, want_tok_prob=want_tok_prob, calc_backend=calc_backend)
        save_json(case_cache_path(cache_dir, split, case["case_id"]), rec, indent=None)
        return rec["case_id"]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(work, c): c["case_id"] for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:   # noqa: BLE001 - keep building the rest
                errors += 1
                print(f"[cache] {futs[fut]} failed: {e!r}", file=sys.stderr)
            if not quiet and (i % 25 == 0 or i == len(todo)):
                print(f"[cache] {split}: {i}/{len(todo)} done ({time.time() - t0:.0f}s)", file=sys.stderr)
    out = merge_split(cache_dir, split, [c["case_id"] for c in cases])
    if not quiet:
        print(f"[cache] wrote {out} ({errors} errors)", file=sys.stderr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=C.ALL_SPLITS + ["all"])
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--cache-dir", default=str(C.CACHE_DIR))
    ap.add_argument("--backend", default=None, help="vllm | claude | mock (default: auto)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--k", type=int, default=C.DEFAULT_K)
    ap.add_argument("--tok-prob", action="store_true", help="request log-probs (vLLM only)")
    args = ap.parse_args(argv)
    llm = get_llm(args.backend)
    splits = C.SPLITS if args.split == "all" else [args.split]
    for s in splits:
        build_split(s, Path(args.cases_dir), Path(args.cache_dir), llm, args.workers, args.limit, args.k,
                    want_tok_prob=True if args.tok_prob else None)


if __name__ == "__main__":
    main()
