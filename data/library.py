"""Facts about the document libraries (record stores) the agent reads and searches, per domain.

Computed from data/cases/: which record files each split's cases point at, how many records and
sentences they hold, one example record, and where the data came from.  Used by the demo
(demo/build_replay.py) and the report (eval/report_html.py) so both describe the same thing.
"""
from __future__ import annotations

from pathlib import Path

from common import config as C
from common.io import read_jsonl
from data.records import tokenize

SOURCES = {
    "test_id": {"name": "SciFact", "domain": "Biomedical science", "paper": "Wadden et al., EMNLP 2020",
                "url": "https://github.com/allenai/scifact",
                "library": "One shared library: every abstract in the SciFact corpus.",
                "claims": "Claims written by experts from citation sentences in biomedical papers."},
    "test_ood": {"name": "HealthVer", "domain": "COVID-19 health", "paper": "Sarrouti et al., Findings of EMNLP 2021",
                 "url": "https://github.com/sarrouti/HealthVer",
                 "library": "One small library per health question: the evidence snippets annotated for that question.",
                 "claims": "Health claims from web search answers about COVID-19, checked against scientific snippets."},
    "test_climate": {"name": "Climate-FEVER", "domain": "Climate science", "paper": "Diggelmann et al., 2020",
                     "url": "https://huggingface.co/datasets/tdiggelm/climate_fever",
                     "library": "One library per claim: its 5 annotated Wikipedia sentences (the agent is first shown 3).",
                     "claims": "Real climate claims collected from the internet, with Wikipedia evidence."},
    "test_vitc": {"name": "VitaminC", "domain": "Wikipedia facts", "paper": "Schuster et al., NAACL 2021",
                  "url": "https://github.com/TalSchuster/VitaminC",
                  "library": "One library per Wikipedia page: the other sentences of the same page.",
                  "claims": "Contrastive claims from Wikipedia revisions: a small factual edit flips the label."},
}
SEARCH = ("Search is BM25 keyword ranking (rank_bm25), not an LLM or embeddings. The agent first sees the top 3 records "
          "for the claim; if it checks, it calls search_records (top 5: title + first sentence), read_record (the full "
          "record as numbered sentences, at most 25) and calculate, at most 4 calls in total. Evidence-deleted twins "
          "simply have their gold records filtered out of every search and read.")


def _short(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def library_stats(cases_dir: Path | str = C.CASES_DIR, splits=tuple(SOURCES)) -> dict[str, dict]:
    cases_dir = Path(cases_dir)
    out = {}
    for split in splits:
        path = cases_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        cases = read_jsonl(path)
        files = sorted({c["record_store"]["path"] for c in cases})
        n_rec = n_sent = 0
        for f in files:
            for r in read_jsonl(cases_dir / f):
                n_rec += 1
                n_sent += len(r["sentences"])
        # the example record is the top retrieved record for the example claim, so the two read together
        def overlap(c):
            toks = set(tokenize(c["question"]))
            ev = c["initial_evidence"][0]
            return len(toks & set(tokenize(ev["title"] + " " + ev["text"]))) / max(1, len(toks))
        base = [c for c in cases if c.get("variant") == "base" and c["initial_evidence"] and len(c["question"]) <= 110]
        case = max(base, key=lambda c: (overlap(c), -len(c["question"]))) if base else cases[0]
        want = str(case["initial_evidence"][0]["doc_id"]) if case["initial_evidence"] else None
        rec = next((r for r in read_jsonl(cases_dir / case["record_store"]["path"]) if str(r["doc_id"]) == want), None)
        example = None if rec is None else {"doc_id": rec["doc_id"], "title": _short(rec["title"], 140),
                                            "sentences": [_short(x, 220) for x in rec["sentences"][:3]], "n_sentences": len(rec["sentences"])}
        claim = case["question"]
        out[split] = {**SOURCES[split], "split": split, "n_claims": len(cases), "n_files": len(files), "n_records": n_rec,
                      "n_sentences": n_sent, "sent_per_record": n_sent / max(1, n_rec), "example": example,
                      "example_claim": _short(claim, 160)}
    return out
