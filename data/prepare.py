"""Build the kwtc case files.

    python -m data.prepare --out data/cases

Steps
 1. Download SciFact (corpus + train/dev claims) and the HealthVer test CSV.
 2. Write a normalised record file per corpus (BM25 is built at load time).
 3. SciFact train -> train/val by connected groups of claims that cite the same
    abstract (union-find on cited_doc_ids); SciFact dev -> test_id.
 4. Every SUPPORT/CONTRADICT claim gets an evidence-ablated twin whose gold
    abstracts are removed from that case's accessible records.
 5. HealthVer test -> test_ood (one claim-evidence pair per claim where possible,
    stratified by label; grouped by question).
 6. Gold labels / gold doc ids / gold rationale sentences are written to
    <out>/gold/<split>.jsonl.  Only scorer/score.py reads that directory.

The optional --shortcut-variant writes <out>_shortcut/ with raw SciFact semantics
(NEI cases have empty evidence, S/R cases see their gold abstract, no twins) for
the "shortcut-trained controller" exhibit.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from common.config import CASES_DIR, RAW_DIR, INITIAL_EVIDENCE_K
from common.io import ensure_dir, read_jsonl, save_json, sha256_files, write_jsonl
from data.records import RecordStore, split_sentences

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
HEALTHVER_URL = "https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_test.csv"

LABEL_MAP = {
    "SUPPORT": "supported", "SUPPORTS": "supported",
    "CONTRADICT": "refuted", "REFUTES": "refuted",
    "NOINFO": "insufficient_evidence", "NEUTRAL": "insufficient_evidence",
}
HV_ID_BASE = 9_000_000   # HealthVer evidence snippets get integer ids above the SciFact range

SCIFACT_RECORDS = "records/scifact_corpus.jsonl"


# ----------------------------------------------------------------------------- download
def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    ensure_dir(dest.parent)
    print(f"[prepare] downloading {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as f:
        f.write(r.read())
    return dest


def fetch_scifact(raw_dir: Path) -> Path:
    d = raw_dir / "scifact"
    data_dir = d / "data"
    if not (data_dir / "corpus.jsonl").exists():
        tgz = download(SCIFACT_URL, d / "data.tar.gz")
        with tarfile.open(tgz) as tf:
            tf.extractall(d)
    return data_dir


def fetch_healthver(raw_dir: Path) -> Path:
    return download(HEALTHVER_URL, raw_dir / "healthver" / "healthver_test.csv")


# ----------------------------------------------------------------------------- scifact
def load_scifact_corpus(data_dir: Path) -> list[dict]:
    recs = []
    for row in read_jsonl(data_dir / "corpus.jsonl"):
        recs.append({"doc_id": int(row["doc_id"]), "title": row["title"], "sentences": list(row["abstract"])})
    return recs


def scifact_gold(claim: dict) -> dict:
    ev = claim.get("evidence") or {}
    if not ev:
        return {"label": "insufficient_evidence", "gold_doc_ids": [], "gold_sentences": {}}
    labels = set()
    sents: dict[str, set[int]] = defaultdict(set)
    for doc_id, rats in ev.items():
        for r in rats:
            labels.add(r["label"])
            sents[str(doc_id)].update(int(s) for s in r["sentences"])
    if len(labels) != 1:   # does not happen in SciFact, but be explicit
        label = "insufficient_evidence"
    else:
        label = LABEL_MAP[labels.pop()]
    return {
        "label": label,
        "gold_doc_ids": [int(d) for d in ev.keys()],
        "gold_sentences": {d: sorted(s) for d, s in sents.items()},
    }


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_claims(claims: list[dict], prefix: str) -> dict[int, str]:
    """Claims citing a common abstract share a group. Returns claim_id -> group_id."""
    uf = UnionFind()
    for c in claims:
        key = ("claim", c["id"])
        uf.find(key)
        for d in c.get("cited_doc_ids", []):
            uf.union(key, ("doc", int(d)))
    roots: dict = {}
    out = {}
    for c in sorted(claims, key=lambda c: c["id"]):
        r = uf.find(("claim", c["id"]))
        if r not in roots:
            roots[r] = f"{prefix}{len(roots):04d}"
        out[c["id"]] = roots[r]
    return out


def make_scifact_case(claim: dict, split: str, group_id: str, store: RecordStore, name_split: str) -> tuple[dict, dict]:
    gold = scifact_gold(claim)
    case_id = f"sf-{name_split}-{claim['id']:04d}"
    case = {
        "case_id": case_id,
        "group_id": group_id,
        "split": split,
        "question": claim["claim"],
        "initial_evidence": store.initial_evidence(claim["claim"], k=INITIAL_EVIDENCE_K),
        "record_index": "bm25_scifact",
        "record_store": {"path": SCIFACT_RECORDS, "exclude": []},
        "variant": "base",
        "parent_id": None,
        "source": {"dataset": "scifact", "claim_id": claim["id"]},
    }
    gold_row = {"case_id": case_id, **gold}
    return case, gold_row


def make_twin(parent: dict, parent_gold: dict, store: RecordStore) -> tuple[dict, dict]:
    exclude = sorted(parent_gold["gold_doc_ids"])
    case_id = parent["case_id"] + "-abl"
    case = {
        **{k: v for k, v in parent.items()},
        "case_id": case_id,
        "initial_evidence": store.initial_evidence(parent["question"], k=INITIAL_EVIDENCE_K, exclude=exclude),
        "record_index": f"bm25_scifact_minus_{exclude}",
        "record_store": {"path": SCIFACT_RECORDS, "exclude": exclude},
        "variant": "ablated",
        "parent_id": parent["case_id"],
    }
    gold = {"case_id": case_id, "label": "insufficient_evidence", "gold_doc_ids": [], "gold_sentences": {},
            "ablated_from": parent_gold["gold_doc_ids"]}
    return case, gold


def build_scifact(data_dir: Path, out: Path, seed: int, val_frac: float = 0.2, twins: bool = True,
                  shortcut: bool = False) -> dict[str, tuple[list[dict], list[dict]]]:
    recs = load_scifact_corpus(data_dir)
    write_jsonl(out / SCIFACT_RECORDS, recs)
    store = RecordStore(recs, name="scifact")

    train_claims = read_jsonl(data_dir / "claims_train.jsonl")
    dev_claims = read_jsonl(data_dir / "claims_dev.jsonl")

    # ---- leak-free split of the official train set into train / val by groups
    groups = group_claims(train_claims, "g")
    group_ids = sorted(set(groups.values()))
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    n_val = int(round(val_frac * len(group_ids)))
    val_groups = set(group_ids[:n_val])
    dev_groups = group_claims(dev_claims, "d")

    splits: dict[str, tuple[list[dict], list[dict]]] = {"train": ([], []), "val": ([], []), "test_id": ([], [])}
    plan = [(c, "val" if groups[c["id"]] in val_groups else "train", groups[c["id"]]) for c in train_claims]
    plan += [(c, "test_id", dev_groups[c["id"]]) for c in dev_claims]

    for claim, split, gid in plan:
        name_split = {"train": "train", "val": "val", "test_id": "dev"}[split]
        case, gold = make_scifact_case(claim, split, gid, store, name_split)
        if shortcut:
            # Raw SciFact semantics: S/R claims see their gold abstract, NEI claims see nothing.
            if gold["label"] == "insufficient_evidence":
                case["initial_evidence"] = []
            else:
                ids = gold["gold_doc_ids"][:INITIAL_EVIDENCE_K]
                case["initial_evidence"] = [
                    {"doc_id": d, "title": store.by_id[str(d)]["title"],
                     "text": " ".join(store.by_id[str(d)]["sentences"])}
                    for d in ids
                ]
            case["record_index"] = "bm25_scifact_raw"
        cases, golds = splits[split]
        cases.append(case)
        golds.append(gold)
        if twins and not shortcut and gold["label"] in ("supported", "refuted"):
            tcase, tgold = make_twin(case, gold, store)
            cases.append(tcase)
            golds.append(tgold)
    return splits


# ----------------------------------------------------------------------------- healthver
def build_healthver(csv_path: Path, out: Path, seed: int, n_target: int = 300) -> tuple[list[dict], list[dict]]:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    for r in rows:
        r["label_norm"] = LABEL_MAP[r["label"].strip().upper()]
        r["evidence"] = r["evidence"].strip()
        r["claim"] = r["claim"].strip()
        r["question"] = r["question"].strip()

    # evidence snippet -> integer id; one record file per question
    ev_ids: dict[str, int] = {}
    for r in rows:
        if r["evidence"] not in ev_ids:
            ev_ids[r["evidence"]] = HV_ID_BASE + len(ev_ids)
    questions = sorted({r["question"] for r in rows})
    q_index = {q: i for i, q in enumerate(questions)}
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["question"]].append(r)
    for q, qrows in by_q.items():
        seen, recs = set(), []
        for r in qrows:
            if r["evidence"] in seen:
                continue
            seen.add(r["evidence"])
            recs.append({"doc_id": ev_ids[r["evidence"]], "title": f"Evidence statement {ev_ids[r['evidence']]}",
                         "sentences": split_sentences(r["evidence"]) or [r["evidence"]]})
        write_jsonl(out / f"records/healthver_q{q_index[q]:03d}.jsonl", recs)

    # ---- selection: stratified by label, one pair per claim first, then fill
    rng = random.Random(seed)
    by_claim: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_claim[r["claim"]].append(r)
    claims = sorted(by_claim)
    rng.shuffle(claims)
    labels = ["supported", "refuted", "insufficient_evidence"]
    target = {l: n_target // 3 for l in labels}
    for l in labels[: n_target - sum(target.values())]:
        target[l] += 1
    filled = Counter()
    chosen: list[dict] = []
    used_ids = set()

    def need(l):   # how far below target a label is (fraction)
        return 1 - filled[l] / max(target[l], 1)

    for _pass in range(2):
        for claim in claims:
            have = sum(1 for c in chosen if c["claim"] == claim)
            if have > _pass:          # pass 0: claims with 0 picks; pass 1: claims with <= 1 pick
                continue
            cands = [r for r in by_claim[claim] if r["id"] not in used_ids and filled[r["label_norm"]] < target[r["label_norm"]]]
            if not cands:
                continue
            cands.sort(key=lambda r: -need(r["label_norm"]))
            pick = cands[0]
            chosen.append(pick)
            used_ids.add(pick["id"])
            filled[pick["label_norm"]] += 1
            if len(chosen) >= n_target:
                break
        if len(chosen) >= n_target:
            break
    if len(chosen) < n_target:
        print(f"[prepare] HealthVer: only {len(chosen)} pairs selected (target {n_target})", file=sys.stderr)
    n_unique = len({c["claim"] for c in chosen})
    print(f"[prepare] HealthVer: {len(chosen)} pairs from {n_unique} unique claims; labels {dict(filled)}")

    cases, golds = [], []
    for r in sorted(chosen, key=lambda r: int(r["id"])):
        qi = q_index[r["question"]]
        eid = ev_ids[r["evidence"]]
        sents = split_sentences(r["evidence"]) or [r["evidence"]]
        case_id = f"hv-test-{int(r['id']):05d}"
        cases.append({
            "case_id": case_id,
            "group_id": f"hvq{qi:03d}",
            "split": "test_ood",
            "question": r["claim"],
            "initial_evidence": [{"doc_id": eid, "title": f"Evidence statement {eid}", "text": r["evidence"]}],
            "record_index": f"bm25_healthver_q{qi:03d}",
            "record_store": {"path": f"records/healthver_q{qi:03d}.jsonl", "exclude": []},
            "variant": "base",
            "parent_id": None,
            "source": {"dataset": "healthver", "pair_id": int(r["id"]), "question": r["question"]},
        })
        label = r["label_norm"]
        golds.append({
            "case_id": case_id,
            "label": label,
            "gold_doc_ids": [eid] if label != "insufficient_evidence" else [],
            "gold_sentences": {str(eid): list(range(len(sents)))} if label != "insufficient_evidence" else {},
        })
    return cases, golds


# ----------------------------------------------------------------------------- main
def summarise(splits: dict[str, tuple[list[dict], list[dict]]]) -> dict:
    summary = {}
    for split, (cases, golds) in splits.items():
        labels = Counter(g["label"] for g in golds)
        variants = Counter(c["variant"] for c in cases)
        summary[split] = {"n": len(cases), "variants": dict(variants), "labels": dict(labels),
                          "groups": len({c["group_id"] for c in cases})}
    return summary


def write_all(out: Path, splits: dict[str, tuple[list[dict], list[dict]]], seed: int, extra: dict | None = None) -> dict:
    for split, (cases, golds) in splits.items():
        write_jsonl(out / f"{split}.jsonl", cases)
        write_jsonl(out / "gold" / f"{split}.jsonl", golds)
    summary = summarise(splits)
    manifest = {"seed": seed, "splits": summary,
                "cases_hash": sha256_files(out / f"{s}.jsonl" for s in splits), **(extra or {})}
    save_json(out / "manifest.json", manifest)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(CASES_DIR))
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--ood-n", type=int, default=300)
    ap.add_argument("--no-twins", action="store_true")
    ap.add_argument("--shortcut-variant", action="store_true",
                    help="also write <out>_shortcut with raw SciFact semantics (empty NEI evidence, no twins)")
    args = ap.parse_args(argv)

    raw, out = Path(args.raw), Path(args.out)
    ensure_dir(out)
    sf_dir = fetch_scifact(raw)
    hv_csv = fetch_healthver(raw)

    splits = build_scifact(sf_dir, out, args.seed, args.val_frac, twins=not args.no_twins)
    splits["test_ood"] = build_healthver(hv_csv, out, args.seed, args.ood_n)
    manifest = write_all(out, splits, args.seed)
    print(json.dumps(manifest["splits"], indent=2))

    if args.shortcut_variant:
        out_s = out.parent / (out.name + "_shortcut")
        ensure_dir(out_s)
        s_splits = build_scifact(sf_dir, out_s, args.seed, args.val_frac, twins=False, shortcut=True)
        # the OOD split is identical; copy the cases so the shortcut controller can be tested on HealthVer
        s_splits["test_ood"] = build_healthver(hv_csv, out_s, args.seed, args.ood_n)
        m = write_all(out_s, s_splits, args.seed, {"variant": "shortcut"})
        print("[prepare] shortcut variant:", json.dumps(m["splits"]))


if __name__ == "__main__":
    main()
