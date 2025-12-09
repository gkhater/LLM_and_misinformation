"""Retrieval backends."""

from __future__ import annotations

import csv
import os
import re
import sys
from typing import Iterable, List, Optional

from rank_bm25 import BM25Okapi


class RetrievalBackend:
    def fetch(self, query: str, context: Optional[str] = None, top_k: int = 3) -> List[str]:
        raise NotImplementedError


class ContextOnlyRetrieval(RetrievalBackend):
    """Uses provided context as the only evidence source."""

    def fetch(self, query: str, context: Optional[str] = None, top_k: int = 3) -> List[str]:
        if context and context.strip():
            return [context.strip()]
        return []


class LocalBM25Retrieval(RetrievalBackend):
    """BM25 over a local TSV corpus (id<TAB>text)."""

    def __init__(
        self,
        tsv_path: str,
        text_col: int = 1,
        delimiter: str = "\t",
        lowercase: bool = True,
        min_score: float = 0.0,
    ):
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"BM25 corpus not found at {tsv_path}")
        self.tsv_path = tsv_path
        self.text_col = text_col
        self.delimiter = delimiter
        self.lowercase = lowercase
        self.min_score = min_score
        self._docs: List[str] = []
        self._ids: List[str] = []
        self._tokenized: List[List[str]] = []
        self._load_corpus()
        self._bm25 = BM25Okapi(self._tokenized)

    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def _load_corpus(self) -> None:
        # Allow large text fields.
        try:
            csv.field_size_limit(sys.maxsize)
        except (OverflowError, ValueError):
            csv.field_size_limit(10_000_000)
        with open(self.tsv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            for row in reader:
                if len(row) <= self.text_col:
                    continue
                doc_id = row[0].strip() if row[0].strip() else str(len(self._docs))
                text = row[self.text_col].strip()
                if not text:
                    continue
                self._ids.append(doc_id)
                self._docs.append(text)
                self._tokenized.append(self._tokenize(text))

    def fetch(self, query: str, context: Optional[str] = None, top_k: int = 3) -> List[str]:
        if not query.strip():
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # top_k indices sorted by score desc and filtered by min_score
        candidates = [
            (i, scores[i]) for i in range(len(scores)) if scores[i] >= self.min_score
        ]
        top_idxs = [i for i, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]]
        return [self._docs[i] for i in top_idxs]

    def fetch_with_ids(self, query: str, context: Optional[str] = None, top_k: int = 3) -> List[tuple]:
        """Return (doc_id, text) pairs; falls back to positional ids if missing."""
        if not query.strip():
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        candidates = [
            (i, scores[i]) for i in range(len(scores)) if scores[i] >= self.min_score
        ]
        top_idxs = [i for i, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]]
        return [(self._ids[i] if i < len(self._ids) else str(i), self._docs[i]) for i in top_idxs]


def build_retriever(cfg: dict) -> RetrievalBackend:
    backend = (cfg.get("backend") or "none").lower()
    if backend in ("none", "context", "context_only"):
        return ContextOnlyRetrieval()
    if backend in ("wikipedia_bm25", "bm25_wiki", "local_bm25"):
        path = cfg.get("corpus_path") or cfg.get("path")
        if not path:
            raise RuntimeError("BM25 backend requires 'corpus_path' (TSV with id<TAB>text).")
        min_score = cfg.get("min_score", 0.0)
        return LocalBM25Retrieval(path, min_score=min_score)
    raise ValueError(f"Unknown retrieval backend: {backend}")
