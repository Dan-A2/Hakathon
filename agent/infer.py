"""Run the deployed agent on one claim (or one dataset case).

    python -m agent.infer --claim "..." --controller art/controller.npz [--remove-evidence]

Deployment path: two provisional samples -> six signals -> frozen W argmax -> answer /
verify / abstain -> verdict, cited sentences, calibrated P(correct).  No gold labels are
touched.  The controller refuses to run if the prompt hash or the LLM model/revision
differ from the ones the cache was built with.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from agent.llm import BaseLLM, get_llm
from agent.provisional import features_from_samples, provisional_answer, run_provisional
from agent.verify import run_verify
from common import config as C
from controller.calibrate import Calibrator
from controller.policy import Controller
from data.records import RecordStore, StoreRegistry

SCIFACT_RECORDS = "records/scifact_corpus.jsonl"


class Runtime:
    def __init__(self, controller: Path | str, calibrator: Path | str | None = None, backend: str | None = None,
                 cases_dir: Path | str = C.CASES_DIR, allow_mismatch: bool = False):
        skip = allow_mismatch or os.environ.get("KWTC_SKIP_HASH_CHECK") == "1"
        self.ctrl = Controller.load(controller, check_prompt=not skip)
        cal_path = Path(calibrator) if calibrator else Path(controller).parent / "calibrator.npz"
        self.cal = Calibrator.load(cal_path) if cal_path.exists() else None
        self.llm: BaseLLM = get_llm(backend)
        if not skip:
            self.ctrl.check_llm(self.llm.identity())
        self.registry = StoreRegistry(cases_dir)

    @property
    def scifact(self) -> RecordStore:
        return self.registry.get(SCIFACT_RECORDS)


def make_case(claim: str, store_path: str = SCIFACT_RECORDS, registry: StoreRegistry | None = None,
              exclude: list | None = None) -> dict:
    registry = registry or StoreRegistry()
    store = registry.get(store_path)
    exclude = list(exclude or [])
    return {"case_id": "live", "group_id": "live", "split": "live", "question": claim,
            "initial_evidence": store.initial_evidence(claim, k=C.INITIAL_EVIDENCE_K, exclude=exclude),
            "record_index": store_path + (f"_minus_{sorted(exclude)}" if exclude else ""),
            "record_store": {"path": store_path, "exclude": exclude}, "variant": "ablated" if exclude else "base",
            "parent_id": None}


def _resolve_cited(store: RecordStore, exclude: list, cited: list[dict]) -> list[dict]:
    out = []
    for c in cited:
        rec = store.get(c["doc_id"], exclude=exclude)
        text = None
        if rec is not None and c.get("sentence") is not None and 0 <= int(c["sentence"]) < len(rec["sentences"]):
            text = rec["sentences"][int(c["sentence"])]
        out.append({"doc_id": c["doc_id"], "sentence": c.get("sentence"), "title": rec["title"] if rec else None,
                    "text": text, "accessible": rec is not None})
    return out


def infer_case(rt: Runtime, case: dict, force_action: int | None = None) -> dict:
    """The full deployment flow for one case dict (dataset case or a live one)."""
    t0 = time.time()
    store, exclude = rt.registry.for_case(case)
    a = run_provisional(rt.llm, case, seed=0)
    b = run_provisional(rt.llm, case, seed=1)
    x = features_from_samples(a, b)
    prov = provisional_answer(a, b)
    action, probs = rt.ctrl.act(x)
    if force_action is not None:
        action = int(force_action)
    llm_calls = a["llm_calls"] + b["llm_calls"]
    tokens = sum(s[k] for s in (a, b) for k in ("tokens_in", "tokens_out"))
    tool_log, tool_calls, ver = [], 0, None
    if action == C.VERIFY:
        ver = run_verify(rt.llm, case, a, store, exclude, K=rt.ctrl.K)
        final = {"verdict": ver["verdict"], "confidence": ver["confidence"], "cited": ver["cited"], "rationale": ver["rationale"]}
        tool_log, tool_calls = ver["tool_log"], ver["tool_calls"]
        llm_calls += ver["llm_calls"]
        tokens += ver["tokens_in"] + ver["tokens_out"]
    elif action == C.ANSWER:
        final = {"verdict": prov["verdict"], "confidence": prov["confidence"],
                 "cited": [{"doc_id": d, "sentence": None} for d in prov["cited_doc_ids"]], "rationale": prov["rationale"]}
    else:
        final = {"verdict": "abstain", "confidence": None, "cited": [],
                 "rationale": "I cannot give a reliable verdict on this claim from the accessible records.",
                 "provisional_verdicts": [{"verdict": s["verdict"], "confidence": s["confidence"]} for s in (a, b)],
                 "records_seen": [ev["doc_id"] for ev in case["initial_evidence"]]}
    p_correct = None
    if rt.cal is not None and action != C.ABSTAIN:
        p_correct = float(rt.cal.predict([x], [action])[0])
    return {
        "claim": case["question"], "case_id": case.get("case_id"), "variant": case.get("variant"),
        "excluded_doc_ids": list(exclude),
        "initial_evidence": [{"doc_id": ev["doc_id"], "title": ev["title"]} for ev in case["initial_evidence"]],
        "samples": [{k: s[k] for k in ("verdict", "confidence", "evidence_sufficiency", "cited_doc_ids", "rationale", "parse_ok")} for s in (a, b)],
        "signals": {k: v for k, v in x.items() if k != "bias"},
        "action_preference": {n: float(p) for n, p in zip(C.ACTIONS, probs)},   # controller probabilities, not confidence
        "action": C.ACTIONS[action],
        "final": {**final, "cited": _resolve_cited(store, exclude, final["cited"])},
        "p_correct": p_correct,
        "tool_log": tool_log, "tool_calls": tool_calls, "llm_calls": llm_calls, "tokens": tokens,
        "latency_s": round(time.time() - t0, 2), "llm": rt.llm.identity(),
    }


def infer_claim(rt: Runtime, claim: str, remove_evidence: bool = False, exclude: list | None = None) -> dict:
    """Live claim against the SciFact store. remove_evidence re-runs with the cited abstracts removed."""
    case = make_case(claim, SCIFACT_RECORDS, rt.registry, exclude)
    res = infer_case(rt, case)
    if not remove_evidence:
        return res
    cited = {d for s in res["samples"] for d in s["cited_doc_ids"]} | {c["doc_id"] for c in res["final"]["cited"]}
    cited = sorted(set(exclude or []) | cited)
    twin = make_case(claim, SCIFACT_RECORDS, rt.registry, cited)
    res_twin = infer_case(rt, twin)
    return {"with_evidence": res, "evidence_removed": res_twin, "removed_doc_ids": cited}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claim", default=None)
    ap.add_argument("--case-id", default=None, help="run a dataset case by id instead of a free-text claim")
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--controller", default=str(C.ART_DIR / "controller.npz"))
    ap.add_argument("--calibrator", default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--remove-evidence", action="store_true", help="also re-run with the cited evidence removed")
    ap.add_argument("--allow-mismatch", action="store_true", help="skip the prompt-hash / model-revision guard")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args(argv)
    if not args.claim and not args.case_id:
        ap.error("give --claim or --case-id")
    rt = Runtime(args.controller, args.calibrator, args.backend, args.cases_dir, args.allow_mismatch)
    if args.case_id:
        from common.io import read_jsonl

        cases = {r["case_id"]: r for r in read_jsonl(Path(args.cases_dir) / f"{args.split}.jsonl")}
        res = infer_case(rt, cases[args.case_id])
    else:
        res = infer_claim(rt, args.claim, args.remove_evidence)
    if args.json:
        print(json.dumps(res, indent=2))
        return
    runs = [("with evidence", res["with_evidence"]), ("evidence removed", res["evidence_removed"])] if "with_evidence" in res else [("", res)]
    for name, r in runs:
        if name:
            print(f"\n=== {name} (excluded {r['excluded_doc_ids']}) ===")
        print(f"claim: {r['claim']}")
        for i, s in enumerate(r["samples"]):
            print(f"  sample {'AB'[i]}: {s['verdict']} conf={s['confidence']:.2f} suff={s['evidence_sufficiency']:.2f} cited={s['cited_doc_ids']}")
        print("  signals: " + ", ".join(f"{k}={v:.2f}" for k, v in r["signals"].items() if v is not None))
        print("  action preference: " + ", ".join(f"{k}={v:.2f}" for k, v in r["action_preference"].items()) + f"  -> {r['action'].upper()}")
        f = r["final"]
        print(f"  verdict: {f['verdict']}" + (f" (confidence {f['confidence']:.2f})" if f.get("confidence") is not None else ""))
        for c in f["cited"]:
            print(f"    cited [{c['doc_id']}] s{c['sentence']}: {(c['text'] or '(doc-level)')[:120]}")
        print(f"  P(correct) calibrated: {'n/a' if r['p_correct'] is None else round(r['p_correct'], 2)}; "
              f"llm_calls={r['llm_calls']} tool_calls={r['tool_calls']} tokens={r['tokens']} ({r['latency_s']}s)")


if __name__ == "__main__":
    main()
