"""Record stores: BM25 search over abstracts, with per-case exclusions for ablated twins.

A *record* is {"doc_id": int, "title": str, "sentences": [str, ...]}.
Instead of building one BM25 index per ablated case (expensive), we keep one
index per corpus and filter excluded doc_ids at query time.  IDF statistics are
unchanged by removing one or two documents out of thousands, so this is
equivalent for all practical purposes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from rank_bm25 import BM25Okapi

from common.config import CASES_DIR, ABSTRACT_MAX_WORDS
from common.io import read_jsonl

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the of and or in on at to for with by from as is are was were be been being this that these those it its "
    "we our they their than then there which who whom whose what when where why how not no nor but if into over "
    "under between among within without about after before during each both all any few more most other some such "
    "only own same so too very can will just do does did done has have had having may might must shall should "
    "would could also".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text)
    return [p.strip() for p in parts if p.strip()]


def truncate_words(text: str, max_words: int = ABSTRACT_MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def _norm_id(doc_id) -> str:
    return str(doc_id)


class RecordStore:
    """BM25 index over records with query-time exclusion."""

    def __init__(self, records: list[dict], name: str = "records"):
        self.name = name
        self.records = records
        self.by_id = {_norm_id(r["doc_id"]): r for r in records}
        corpus = [tokenize(r["title"] + " " + " ".join(r["sentences"])) for r in records]
        # rank_bm25 divides by the average document length; guard the empty case.
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def from_jsonl(cls, path: Path | str, name: str | None = None) -> "RecordStore":
        rows = read_jsonl(path)
        return cls(rows, name=name or Path(path).stem)

    def __len__(self) -> int:
        return len(self.records)

    # ---- access --------------------------------------------------------------------------
    def get(self, doc_id, exclude: Iterable = ()) -> dict | None:
        """Return the record, or None if it does not exist or is excluded for this case."""
        key = _norm_id(doc_id)
        if key in {_norm_id(x) for x in exclude}:
            return None
        return self.by_id.get(key)

    def search(self, query: str, k: int = 5, exclude: Iterable = ()) -> list[dict]:
        """Top-k records by BM25 for the query, skipping excluded doc_ids."""
        if self._bm25 is None:
            return []
        ex = {_norm_id(x) for x in exclude}
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out = []
        for i in order:
            r = self.records[i]
            if _norm_id(r["doc_id"]) in ex:
                continue
            out.append({
                "doc_id": r["doc_id"],
                "title": r["title"],
                "first_sentence": r["sentences"][0] if r["sentences"] else "",
                "score": float(scores[i]),
            })
            if len(out) >= k:
                break
        return out

    def initial_evidence(self, query: str, k: int = 3, exclude: Iterable = (),
                         max_words: int = ABSTRACT_MAX_WORDS) -> list[dict]:
        """Top-k abstracts formatted as initial evidence for a case."""
        hits = self.search(query, k=k, exclude=exclude)
        out = []
        for h in hits:
            r = self.by_id[_norm_id(h["doc_id"])]
            out.append({
                "doc_id": r["doc_id"],
                "title": r["title"],
                "text": truncate_words(" ".join(r["sentences"]), max_words),
            })
        return out


class StoreRegistry:
    """Lazy cache of RecordStores keyed by their file path (relative to the cases dir)."""

    def __init__(self, cases_dir: Path | str | None = None):
        self.cases_dir = Path(cases_dir) if cases_dir else CASES_DIR
        self._stores: dict[str, RecordStore] = {}

    def get(self, rel_path: str) -> RecordStore:
        if rel_path not in self._stores:
            path = self.cases_dir / rel_path
            self._stores[rel_path] = RecordStore.from_jsonl(path, name=Path(rel_path).stem)
        return self._stores[rel_path]

    def for_case(self, case: dict) -> tuple[RecordStore, list]:
        """Resolve (store, excluded_doc_ids) for a case."""
        rs = case["record_store"]
        return self.get(rs["path"]), list(rs.get("exclude", []))
