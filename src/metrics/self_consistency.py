from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Callable, List, Optional


def _token_set(text: str) -> set:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return set(tokens)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class SelfConsistencyResult:
    consistency: float  # 0..1, higher is more consistent
    risk: float         # 0..1, higher is more risk
    avg_similarity: float
    contradiction_rate: float


class SelfConsistencyEvaluator:
    """Estimates hallucination risk by comparing multiple generations of the same prompt."""

    def __init__(
        self,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        embedding_model: Optional[str] = None,
        nli_model: Optional[str] = None,
    ):
        self.similarity_fn = similarity_fn or self._lexical_similarity
        self.embedding_model_name = embedding_model
        self.nli_model_name = nli_model
        self._embedder = None
        self._nli = None

    def _lexical_similarity(self, a: str, b: str) -> float:
        return _jaccard(_token_set(a), _token_set(b))

    def _load_embedder(self):
        if self.embedding_model_name and self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _load_nli(self):
        if self.nli_model_name and self._nli is None:
            from transformers import pipeline

            self._nli = pipeline(
                "text-classification",
                model=self.nli_model_name,
                tokenizer=self.nli_model_name,
                return_all_scores=False,
            )
        return self._nli

    def _embed_similarity(self, a: str, b: str) -> Optional[float]:
        embedder = self._load_embedder()
        if not embedder:
            return None
        embs = embedder.encode([a, b], convert_to_tensor=False)
        import numpy as np

        a_vec, b_vec = embs
        denom = (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)) or 1e-9
        return float(np.dot(a_vec, b_vec) / denom)

    def _nli_contradicts(self, a: str, b: str) -> Optional[bool]:
        nli = self._load_nli()
        if not nli:
            return None
        pred = nli({"text": b, "text_pair": a})[0]
        label = pred["label"].upper()
        return "CONTRAD" in label

    def evaluate(self, generations: List[str]) -> SelfConsistencyResult:
        if len(generations) < 2:
            return SelfConsistencyResult(
                consistency=1.0, risk=0.0, avg_similarity=1.0, contradiction_rate=0.0
            )

        sims = []
        contradictions = 0
        embed_scores = []

        for g1, g2 in itertools.combinations(generations, 2):
            lex = self.similarity_fn(g1, g2)
            sims.append(lex)

            emb = self._embed_similarity(g1, g2)
            if emb is not None:
                embed_scores.append(emb)

            nli_contra = self._nli_contradicts(g1, g2)
            if nli_contra:
                contradictions += 1

        avg_lex = sum(sims) / len(sims) if sims else 1.0
        if embed_scores:
            avg_sim = sum(embed_scores) / len(embed_scores)
        else:
            avg_sim = avg_lex

        risk = 1.0 - avg_sim
        total_pairs = len(sims)
        contradiction_rate = contradictions / total_pairs if total_pairs else 0.0

        return SelfConsistencyResult(
            consistency=avg_sim,
            risk=risk,
            avg_similarity=avg_sim,
            contradiction_rate=contradiction_rate,
        )
