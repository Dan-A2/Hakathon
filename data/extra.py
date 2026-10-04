"""Extra evaluation sets for cross-domain epistemics and the evidence ladder.

    python -m data.extra --out data/cases            # writes ONLY the new splits; existing splits are untouched

Splits written (each with cases, gold and record files, same schema as data/prepare.py):
  test_climate  Climate-FEVER (Diggelmann et al. 2020): 1,535 real-world climate claims, each with 5 Wikipedia
                evidence sentences labelled supports / refutes / not enough info.  A case shows the agent the top-3
                of its 5 sentences; the other 2 are reachable only by checking, so checking has value by
                construction.  NOT_ENOUGH_INFO claims carry evidence that is present but inconclusive (no empty-
                evidence shortcut).  DISPUTED claims (both kinds of evidence) are left out.
  test_vitc     VitaminC (Schuster et al. 2021): contrastive claim-evidence pairs from Wikipedia revisions.  Initial
                evidence = the pair's sentence; accessible records = the other sentences from the same page.
  test_ladder   The evidence ladder, built from SciFact dev: for every supported / refuted claim, a variant whose
                gold abstract is still shown but with the gold rationale sentences deleted ("rationale removed":
                topical evidence that no longer answers the claim; gold label insufficient_evidence).  Together
                with the base case (full evidence) and the existing twin (abstract removed) this gives three
                levels of label-information with near-constant topical similarity.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from common.config import CASES_DIR, INITIAL_EVIDENCE_K, RAW_DIR
from common.io import ensure_dir, read_jsonl, save_json, load_json, write_jsonl
from data.prepare import LABEL_MAP, fetch_scifact, load_scifact_corpus, scifact_gold, SCIFACT_RECORDS
from data.records import RecordStore, split_sentences

CLIMATE_URL = "https://huggingface.co/api/datasets/tdiggelm/climate_fever/parquet/default/test/0.parquet"
VITC_URL = "https://huggingface.co/datasets/tals/vitaminc/resolve/main/test.jsonl"
CF_ID_BASE, VC_ID_BASE = 7_000_000, 8_000_000
CLIMATE_CLAIM_LABEL = {0: "supported", 1: "refuted", 2: "insufficient_evidence"}     # 3 = DISPUTED, excluded
CLIMATE_EVIDENCE_LABEL = {0: "supported", 1: "refuted", 2: "insufficient_evidence"}


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        ensure_dir(dest.parent)
        urllib.request.urlretrieve(url, dest)
    return dest


def _stratified(rows: list[dict], label_of, n: int, seed: int, key_of=None) -> list[dict]:
    """n rows, as balanced over labels as the data allows; at most one row per key (claim) when key_of is given."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    labels = sorted({label_of(r) for r in rows})
    target = {l: n // len(labels) for l in labels}
    for l in labels[: n - sum(target.values())]:
        target[l] += 1
    out, filled, used = [], Counter(), set()
    for r in rows:
        l = label_of(r)
        k = key_of(r) if key_of else id(r)
        if filled[l] < target[l] and k not in used:
            out.append(r)
            filled[l] += 1
            used.add(k)
    return out


# ----------------------------------------------------------------------------- Climate-FEVER
def build_climate_fever(raw_dir: Path, out: Path, seed: int, n: int = 300) -> tuple[list[dict], list[dict]]:
    import pyarrow.parquet as pq

    path = _download(CLIMATE_URL, raw_dir / "climate_fever" / "test.parquet")
    rows = [r for r in pq.read_table(path).to_pylist() if r["claim_label"] in CLIMATE_CLAIM_LABEL]
    chosen = _stratified(rows, lambda r: CLIMATE_CLAIM_LABEL[r["claim_label"]], n, seed)
    ev_ids: dict[str, int] = {}
    cases, golds = [], []
    for r in sorted(chosen, key=lambda r: int(r["claim_id"])):
        recs = []
        for e in r["evidences"]:
            if e["evidence"] not in ev_ids:
                ev_ids[e["evidence"]] = CF_ID_BASE + len(ev_ids)
            recs.append({"doc_id": ev_ids[e["evidence"]], "title": e["article"], "sentences": [e["evidence"]],
                         "label": CLIMATE_EVIDENCE_LABEL.get(e["evidence_label"], "insufficient_evidence")})
        rel = f"records/climate_{int(r['claim_id']):04d}.jsonl"
        write_jsonl(out / rel, [{k: v for k, v in x.items() if k != "label"} for x in recs])
        store = RecordStore([{k: v for k, v in x.items() if k != "label"} for x in recs], name=rel)
        label = CLIMATE_CLAIM_LABEL[r["claim_label"]]
        case_id = f"cf-test-{int(r['claim_id']):04d}"
        cases.append({"case_id": case_id, "group_id": f"cf{int(r['claim_id']):04d}", "split": "test_climate",
                      "question": r["claim"], "initial_evidence": store.initial_evidence(r["claim"], k=INITIAL_EVIDENCE_K),
                      "record_index": f"bm25_{rel}", "record_store": {"path": rel, "exclude": []}, "variant": "base",
                      "parent_id": None, "source": {"dataset": "climate_fever", "claim_id": r["claim_id"]}})
        gold_docs = [x["doc_id"] for x in recs if x["label"] == label] if label != "insufficient_evidence" else []
        golds.append({"case_id": case_id, "label": label, "gold_doc_ids": gold_docs,
                      "gold_sentences": {str(d): [0] for d in gold_docs}})
    return cases, golds


# ----------------------------------------------------------------------------- VitaminC
def build_vitaminc(raw_dir: Path, out: Path, seed: int, n: int = 300) -> tuple[list[dict], list[dict]]:
    path = _download(VITC_URL, raw_dir / "vitaminc" / "test.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    lab = {"SUPPORTS": "supported", "REFUTES": "refuted", "NOT ENOUGH INFO": "insufficient_evidence"}
    by_page: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_page[r["page"]].append(r)
    # keep pages with at least 3 distinct evidence sentences so checking has something to find
    eligible = [r for r in rows if len({x["evidence"] for x in by_page[r["page"]]}) >= 3]
    chosen = _stratified(eligible, lambda r: lab[r["label"]], n, seed, key_of=lambda r: r["claim"])
    ev_ids: dict[str, int] = {}
    page_idx = {p: i for i, p in enumerate(sorted(by_page))}
    written = set()
    cases, golds = [], []
    for r in sorted(chosen, key=lambda r: r["unique_id"]):
        page = r["page"]
        rel = f"records/vitc_p{page_idx[page]:05d}.jsonl"
        recs = []
        for x in by_page[page]:
            if x["evidence"] not in ev_ids:
                ev_ids[x["evidence"]] = VC_ID_BASE + len(ev_ids)
        for ev in dict.fromkeys(x["evidence"] for x in by_page[page]):
            recs.append({"doc_id": ev_ids[ev], "title": page, "sentences": split_sentences(ev) or [ev]})
        if rel not in written:
            write_jsonl(out / rel, recs)
            written.add(rel)
        eid = ev_ids[r["evidence"]]
        label = lab[r["label"]]
        case_id = f"vc-test-{zlib.crc32(r['unique_id'].encode()) % 10**8:08d}"
        sents = split_sentences(r["evidence"]) or [r["evidence"]]
        cases.append({"case_id": case_id, "group_id": f"vcp{page_idx[page]:05d}", "split": "test_vitc", "question": r["claim"],
                      "initial_evidence": [{"doc_id": eid, "title": page, "text": r["evidence"]}],
                      "record_index": f"bm25_{rel}", "record_store": {"path": rel, "exclude": []}, "variant": "base",
                      "parent_id": None, "source": {"dataset": "vitaminc", "unique_id": r["unique_id"], "revision_type": r.get("revision_type")}})
        golds.append({"case_id": case_id, "label": label, "gold_doc_ids": [eid] if label != "insufficient_evidence" else [],
                      "gold_sentences": {str(eid): list(range(len(sents)))} if label != "insufficient_evidence" else {}})
    return cases, golds


# ----------------------------------------------------------------------------- evidence ladder (SciFact dev)
def build_ladder(raw_dir: Path, out: Path) -> tuple[list[dict], list[dict]]:
    data_dir = fetch_scifact(raw_dir)
    recs = load_scifact_corpus(data_dir)
    dev = read_jsonl(data_dir / "claims_dev.jsonl")
    # every rationale sentence of any dev claim, per gold doc
    removed: dict[int, set[int]] = defaultdict(set)
    for c in dev:
        for d, rats in (c.get("evidence") or {}).items():
            for r in rats:
                removed[int(d)].update(int(i) for i in r["sentences"])
    modified = []
    dropped = set()
    for r in recs:
        if r["doc_id"] in removed:
            keep = [s for i, s in enumerate(r["sentences"]) if i not in removed[r["doc_id"]]]
            if len(keep) < 2:                       # nothing topical left to show
                dropped.add(r["doc_id"])
                continue
            modified.append({**r, "sentences": keep})
        else:
            modified.append(r)
    rel = "records/scifact_dev_rationale_removed.jsonl"
    write_jsonl(out / rel, modified)
    store = RecordStore(modified, name="scifact_rr")
    cases, golds = [], []
    for c in dev:
        g = scifact_gold(c)
        if g["label"] == "insufficient_evidence" or any(d in dropped for d in g["gold_doc_ids"]):
            continue
        case_id = f"sf-dev-{c['id']:04d}-rr"
        cases.append({"case_id": case_id, "group_id": f"ladder-{c['id']:04d}", "split": "test_ladder", "question": c["claim"],
                      "initial_evidence": store.initial_evidence(c["claim"], k=INITIAL_EVIDENCE_K),
                      "record_index": "bm25_scifact_dev_rationale_removed", "record_store": {"path": rel, "exclude": []},
                      "variant": "rationale_removed", "parent_id": f"sf-dev-{c['id']:04d}",
                      "source": {"dataset": "scifact", "claim_id": c["id"], "ladder_level": 1,
                                 "gold_shown": any(ev["doc_id"] in g["gold_doc_ids"] for ev in store.initial_evidence(c["claim"], k=INITIAL_EVIDENCE_K))}})
        golds.append({"case_id": case_id, "label": "insufficient_evidence", "gold_doc_ids": [], "gold_sentences": {},
                      "ablated_from": g["gold_doc_ids"], "original_label": g["label"],
                      "removed_sentences": {str(d): sorted(removed[d]) for d in g["gold_doc_ids"]}})
    return cases, golds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CASES_DIR))
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--only", default="climate,vitc,ladder")
    args = ap.parse_args(argv)
    out, raw = Path(args.out), Path(args.raw)
    builders = {"climate": lambda: ("test_climate", build_climate_fever(raw, out, args.seed, args.n)),
                "vitc": lambda: ("test_vitc", build_vitaminc(raw, out, args.seed, args.n)),
                "ladder": lambda: ("test_ladder", build_ladder(raw, out))}
    manifest = load_json(out / "manifest.json", {}) or {}
    manifest.setdefault("extra_splits", {})
    for key in args.only.split(","):
        split, (cases, golds) = builders[key]()
        write_jsonl(out / f"{split}.jsonl", cases)
        write_jsonl(out / "gold" / f"{split}.jsonl", golds)
        summary = {"n": len(cases), "labels": dict(Counter(g["label"] for g in golds)), "groups": len({c["group_id"] for c in cases}),
                   "variants": dict(Counter(c["variant"] for c in cases))}
        manifest["extra_splits"][split] = summary
        print(f"[extra] {split}: {json.dumps(summary)}")
    save_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
