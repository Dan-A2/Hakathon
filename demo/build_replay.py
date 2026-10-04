"""Build demo/replay.js: curated, real runs of the three frozen models for the presentation demo.

    python -m demo.build_replay            # needs art/ (caches, models) and data/cases/

Every number comes from the cached LLM calls and the shipped credence-based epistemic agent
(art/models/<model>/epistemic_eu.npz); the demo page only replays them, so it works offline.
Gold labels are included for the "reveal" step only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import config as C
from common import models as M
from common.io import read_jsonl
from controller.epistemic import EUController
from data.records import StoreRegistry
from scorer.score import load_scored, outcome

OUT = Path(__file__).parent / "replay.js"
DOMAINS = {"test_id": "Biomedical science (SciFact)", "test_ood": "COVID-19 health (HealthVer)",
           "test_climate": "Climate science (Climate-FEVER)", "test_vitc": "Wikipedia facts (VitaminC)"}
PER_STORY = 4


def _short(s: str | None, n: int = 420) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def _sentence(store, exclude, doc_id, sent):
    rec = store.get(doc_id, exclude=exclude)
    if rec is None:
        return {"doc_id": doc_id, "sentence": sent, "title": None, "text": None}
    text = rec["sentences"][int(sent)] if sent is not None and 0 <= int(sent) < len(rec["sentences"]) else None
    return {"doc_id": doc_id, "sentence": sent, "title": rec["title"], "text": _short(text, 360) if text else None}


def case_payload(sc, case, dec, cred, info, judge, store, exclude, model_key, split) -> dict:
    raw, ver = sc.raw, sc.raw["ver"]
    agent = outcome(sc, dec)
    checked = dec in C.CHECKED
    p = {
        "id": f"{model_key}:{sc.case_id}", "case_id": sc.case_id, "model": model_key, "split": split, "domain": DOMAINS[split],
        "claim": case["question"], "variant": sc.variant, "parent_id": sc.parent_id,
        "evidence": [{"doc_id": ev["doc_id"], "title": _short(ev.get("title"), 140), "text": _short(ev.get("text"), 380)}
                     for ev in case["initial_evidence"]],
        "excluded": list(exclude),
        "samples": [{"verdict": s["verdict"], "confidence": s["confidence"], "sufficiency": s["evidence_sufficiency"],
                     "cited": s["cited_doc_ids"], "rationale": _short(s["rationale"], 360)} for s in (raw["a"], raw["b"])],
        "signals": {k: v for k, v in sc.x.items() if k != "bias" and v is not None},
        "raw": {"verdict": sc.prov_verdict, "confidence": sc.prov_conf, "correct": sc.prov_correct},
        "naive_check": {"verdict": sc.ver_verdict, "confidence": sc.ver_conf, "correct": sc.ver_correct},
        "agent": {"decision": C.DECISIONS[dec], "credence": cred, "verdict": agent["verdict"], "correct": agent["correct"],
                  "cited": [_sentence(store, exclude, c["doc_id"], c.get("sentence")) for c in agent["cited"]],
                  "llm_calls": agent["llm_calls"], "tool_calls": agent["tool_calls"], "tokens": agent["tokens"],
                  "ev": {k: round(v, 4) for k, v in info.items()}},
        "check": None,
        "gold": {"label": sc.label,
                 "sentences": [_sentence(store, [], d, s) for d, ss in sc.gold_sentences.items() for s in ss][:3]},
    }
    if checked:
        steps = []
        for t in ver["tool_log"]:
            s = {"tool": t["tool"], "ok": t.get("ok", True)}
            if t["tool"] == "search_records":
                s["query"] = t["args"].get("query")
                s["hits"] = len(t.get("summary", {}).get("doc_ids", []))
            elif t["tool"] == "read_record":
                d = t["args"].get("doc_id")
                rec = store.get(d, exclude=exclude)
                s["doc_id"], s["title"] = d, _short(rec["title"], 120) if rec else None
            else:
                s["args"] = t.get("args")
            steps.append(s)
        p["check"] = {
            "steps": steps, "verdict": sc.ver_verdict, "confidence": sc.ver_conf, "rationale": _short(ver.get("rationale"), 360),
            "disciplined_verdict": sc.dver_verdict, "source": sc.dver_source,
            "quotes": [_sentence(store, exclude, c["doc_id"], c["sentence"]) for c in sc.dver_cited][:3],
            "judge": None if not judge else {"relation": judge["relation"], "confidence": judge.get("confidence"),
                                             "credence": judge.get("credence"), "reason": _short(judge.get("reason"), 300)},
        }
    return p


def stories(rows: list[dict]) -> dict[str, list[dict]]:
    """Pick the cases that tell the project's story, most striking first."""
    by_id = {r["case_id"]: r for r in rows}
    commit = lambda v: v in C.COMMIT_LABELS  # noqa: E731
    out: dict[str, list] = {k: [] for k in ("caught", "fixed", "kept", "twin", "confident", "miss")}
    for r in rows:
        a, raw, nc = r["agent"], r["raw"], r["naive_check"]
        if commit(raw["verdict"]) and not raw["correct"] and raw["confidence"] >= 0.8 and a["verdict"] == "abstain":
            out["caught"].append((-raw["confidence"], r))
        if a["decision"] == "check_commit" and a["correct"] and not raw["correct"]:
            out["fixed"].append((-(a["credence"] or 0), r))
        if a["decision"] == "check_keep_prior" and a["correct"] and not nc["correct"]:
            out["kept"].append((-(a["credence"] or 0), r))
        if a["decision"] == "answer_grounded" and a["correct"] and commit(a["verdict"]) and (a["credence"] or 0) >= 0.7:
            out["confident"].append((-(a["credence"] or 0), r))
        if commit(a["verdict"]) and not a["correct"] and (a["credence"] or 0) >= 0.6:
            out["miss"].append((-(a["credence"] or 0), r))
        twin = by_id.get(r["case_id"] + "-abl")
        if twin and r["agent"]["correct"] and commit(r["agent"]["verdict"]) and twin["raw"]["verdict"] == r["raw"]["verdict"] \
                and commit(twin["raw"]["verdict"]) and twin["agent"]["verdict"] in ("abstain", "insufficient_evidence"):
            out["twin"].append((-twin["raw"]["confidence"], r))
    picked = {}
    for k, lst in out.items():
        lst.sort(key=lambda t: (t[0], len(t[1]["claim"])))
        chosen, seen = [], set()
        for _, r in lst:                                  # one per domain first, then the next most striking
            if r["split"] not in seen and len(chosen) < PER_STORY:
                chosen.append(r["case_id"])
                seen.add(r["split"])
        for _, r in lst:
            if len(chosen) >= PER_STORY:
                break
            if r["case_id"] not in chosen:
                chosen.append(r["case_id"])
        picked[k] = chosen
    return picked


def summary(model_key: str) -> dict:
    out = {}
    for split, fn in (("test_id", "comparison.json"), ("test_ood", "comparison_ood.json"),
                      ("test_climate", "comparison_climate.json"), ("test_vitc", "comparison_vitc.json")):
        path = C.ART_DIR / "models" / fn
        if not path.exists():
            continue
        row = next((r for r in json.loads(path.read_text()) if r["key"] == model_key), None)
        if not row or not row.get("epistemic", {}).get("epistemic_eu"):
            continue
        e, b = row["epistemic"]["epistemic_eu"], row["baseline"]
        out[split] = {"n": row["n_cases"], "wrong_raw": b["harmful"]["mean"], "wrong_agent": e["harmful"]["mean"],
                      "ece_raw": row["calibration"]["ece_raw_answer"], "ece_agent": e["credence_ece"],
                      "coverage": e["coverage"]["mean"], "sel_acc_raw": b["selective_accuracy"]["mean"],
                      "sel_acc_agent": e["selective_accuracy"]["mean"], "fab_raw": b.get("fabrication_raw_checker"),
                      "fab_agent": e["fabrication"], "check_rate": e["check_rate"],
                      "verify_fixed": row["verify_wrong_to_right"]["mean"], "verify_broke": row["verify_right_to_wrong"]["mean"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-dir", default=str(C.CASES_DIR))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    cases_dir = Path(args.cases_dir)
    reg = StoreRegistry(cases_dir)
    payload = {"models": [], "cases": {}, "stories": {}, "w": None}
    for key in M.ORDER[::-1]:
        m = M.get(key)
        cache_dir = C.ART_DIR / m["cache"]
        eu = EUController.load(C.ART_DIR / "models" / key / "epistemic_eu.npz")
        payload["w"], payload["c"] = eu.w, eu.c
        rows = []
        for split in DOMAINS:
            if not (cache_dir / f"{split}.jsonl").exists():
                continue
            cases = {r["case_id"]: r for r in read_jsonl(cases_dir / f"{split}.jsonl")}
            judges = {r["case_id"]: r for r in read_jsonl(cache_dir / f"judge_{split}.jsonl")} if (cache_dir / f"judge_{split}.jsonl").exists() else {}
            log = {r["case_id"]: r for r in read_jsonl(C.ART_DIR / "models" / key / "logs" / f"{split}_epistemic_eu.jsonl")}
            for sc in load_scored(split, cases_dir, cache_dir):
                dec, cred, info = eu.decide(sc.x, sc.z)
                assert C.DECISIONS[dec] == log[sc.case_id]["action"], (key, sc.case_id)   # same agent as the report
                store, exclude = reg.for_case(cases[sc.case_id])
                rows.append(case_payload(sc, cases[sc.case_id], dec, cred, info, judges.get(sc.case_id), store, exclude, key, split))
        picked = stories(rows)
        by_id = {r["case_id"]: r for r in rows}
        keep = {cid for ids in picked.values() for cid in ids}
        keep |= {cid + "-abl" for cid in keep if cid + "-abl" in by_id}         # every parent ships with its twin
        keep |= {by_id[cid]["parent_id"] for cid in list(keep) if by_id[cid].get("parent_id") in by_id}
        for cid in sorted(keep):
            payload["cases"][f"{key}:{cid}"] = by_id[cid]
        payload["stories"][key] = {k: [f"{key}:{c}" for c in v] for k, v in picked.items()}
        payload["models"].append({"key": key, "label": m["label"], "params_total": m["params_total"],
                                  "params_active": m["params_active"], "summary": summary(key), "n_cases": len(rows)})
        print(f"{key}: {len(rows)} cases scored, {len(keep)} shipped; " + ", ".join(f"{k}={len(v)}" for k, v in picked.items()))
    # a script, not JSON, so demo/index.html also works when opened straight from disk (file://)
    Path(args.out).write_text("window.REPLAY = " + json.dumps(payload, separators=(",", ":"), default=float) + ";\n", encoding="utf-8")
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
