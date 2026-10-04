"""Add blind-judge records to an existing cache split (one cheap LLM call per committed, grounded
verified verdict).  Resumable; writes <cache_dir>/judge_<split>.jsonl which the scorer joins by case_id.

    python -m agent.build_judge --split test_id --cache-dir art/cache_llama3b [--backend mock]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent.epistemic import accessible_citations
from agent.judge import judge_prompt_hash, run_judge
from agent.llm import BaseLLM, get_llm
from common import config as C
from common.io import ensure_dir, load_json, read_jsonl, save_json, write_jsonl
from data.records import StoreRegistry


def needs_judge(rec: dict, registry: StoreRegistry, case: dict) -> list[dict]:
    """Quotes to judge: empty when the verifier did not commit or cited nothing it actually saw."""
    ver = rec["ver"]
    if ver["verdict"] not in C.COMMIT_LABELS:
        return []
    store, exclude = registry.for_case(case)
    return accessible_citations(ver.get("cited", []), set(rec.get("shown_doc_ids", [])), set(ver.get("opened_doc_ids", [])),
                                store, exclude)


def judge_record(rec: dict, case: dict, quotes: list[dict], llm: BaseLLM) -> dict:
    out = run_judge(llm, case["question"], quotes)
    return {"case_id": rec["case_id"], "split": rec["split"], **out, "quotes": quotes[:5], "llm": llm.identity(),
            "judge_prompt_hash": judge_prompt_hash(), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


def merge_judge(cache_dir: Path, split: str) -> Path:
    d = cache_dir / "judge" / split
    rows = [load_json(f) for f in sorted(d.glob("*.json"))] if d.exists() else []
    out = cache_dir / f"judge_{split}.jsonl"
    write_jsonl(out, rows)
    return out


def build_judge_split(split: str, cases_dir: Path, cache_dir: Path, llm: BaseLLM, workers: int = 4,
                      limit: int | None = None, quiet: bool = False) -> Path:
    records = read_jsonl(cache_dir / f"{split}.jsonl")
    cases = {c["case_id"]: c for c in read_jsonl(cases_dir / f"{split}.jsonl")}
    registry = StoreRegistry(cases_dir)
    ensure_dir(cache_dir / "judge" / split)
    todo = []
    for rec in records:
        out = cache_dir / "judge" / split / f"{rec['case_id']}.json"
        if out.exists() or rec["case_id"] not in cases:
            continue
        quotes = needs_judge(rec, registry, cases[rec["case_id"]])
        if quotes:
            todo.append((rec, quotes))
    if limit:
        todo = todo[:limit]
    if not quiet:
        print(f"[judge] {split}: {len(records)} records, {len(todo)} committed+grounded verdicts to judge", file=sys.stderr)

    def work(item):
        rec, quotes = item
        row = judge_record(rec, cases[rec["case_id"]], quotes, llm)
        save_json(cache_dir / "judge" / split / f"{rec['case_id']}.json", row, indent=None)
        return rec["case_id"]

    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(work, it) for it in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"[judge] failed: {e!r}", file=sys.stderr)
            if not quiet and (i % 50 == 0 or i == len(todo)):
                print(f"[judge] {split}: {i}/{len(todo)}", file=sys.stderr)
    out = merge_judge(cache_dir, split)
    if not quiet:
        print(f"[judge] wrote {out} ({errors} errors)", file=sys.stderr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=C.SPLITS + ["all"])
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--cache-dir", default=str(C.CACHE_DIR))
    ap.add_argument("--backend", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    llm = get_llm(args.backend)
    for s in (C.SPLITS if args.split == "all" else [args.split]):
        build_judge_split(s, Path(args.cases_dir), Path(args.cache_dir), llm, args.workers, args.limit)


if __name__ == "__main__":
    main()
