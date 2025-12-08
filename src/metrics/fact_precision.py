from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter with basic filtering for non-factual fluff."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    candidates = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        # Filter clearly non-factual/meta sentences.
        lower = s.lower()
        if s.endswith("?"):
            continue
        if "as an ai" in lower or "as a language model" in lower:
            continue
        if len(s.split()) < 4:
            continue
        candidates.append(s)
    return candidates


def _normalize_label(label: str) -> str:
    label = label.lower()
    if "support" in label or "entail" in label:
        return "supported"
    if "refute" in label or "contradict" in label:
        return "refuted"
    return "nei"


def _default_retriever(claim: str, context: Optional[str]) -> List[str]:
    """Fallback retriever: uses provided context only."""
    if context:
        return [context]
    return []


@dataclass
class FactPrecisionResult:
    supported: int
    refuted: int
    nei: int
    unsupported: int
    refute_rate: float
    coverage: float

    @property
    def fact_precision(self) -> float:
        total = self.supported + self.refuted + self.unsupported + self.nei
        if total == 0:
            return 0.0
        return self.supported / total


class FactPrecisionEvaluator:
    """Evaluates factual support of model output statements via retrieval + NLI."""

    def __init__(
        self,
        nli_model_name: str,
        retriever: Optional[Callable[[str, Optional[str]], Iterable[str]]] = None,
        max_evidence: int = 3,
        entail_threshold: float = 0.5,
        contradict_threshold: float = 0.5,
        margin: float = 0.1,
        device: Optional[str] = None,
    ):
        self.nli_model_name = nli_model_name
        self.retriever = retriever or _default_retriever
        self.max_evidence = max_evidence
        self.device = device
        self.entail_threshold = entail_threshold
        self.contradict_threshold = contradict_threshold
        self.margin = margin
        self._clf = None  # lazy-loaded pipeline
        self._cache: dict[Tuple[str, str], str] = {}

    def _load_classifier(self):
        if self._clf is None:
            from transformers import pipeline

            self._clf = pipeline(
                "text-classification",
                model=self.nli_model_name,
                tokenizer=self.nli_model_name,
                device_map=self.device or "auto",
                return_all_scores=False,
            )
        return self._clf

    def _label(self, claim: str, evidence: str) -> str:
        key = (claim, evidence)
        if key in self._cache:
            return self._cache[key]

        pred = self._clf(
            {
                "text": evidence,
                "text_pair": claim,
            }
        )
        # Expect labels like ENTAILMENT / CONTRADICTION / NEUTRAL
        label = pred["label"].upper()
        score = pred.get("score", 0.0)

        if "ENTAIL" in label and score >= self.entail_threshold:
            verdict = "supported"
        elif "CONTRAD" in label and score >= self.contradict_threshold:
            verdict = "refuted"
        else:
            verdict = "nei"

        self._cache[key] = verdict
        return verdict

    def evaluate(
        self,
        model_output_text: str,
        claim_context: Optional[str] = None,
    ) -> FactPrecisionResult:
        sentences = _split_sentences(model_output_text)
        if not sentences:
            return FactPrecisionResult(0, 0, 0, 0, 0.0, 0.0)

        clf = self._load_classifier()

        supported = refuted = nei = unsupported = 0

        for sent in sentences:
            evidence_candidates = list(self.retriever(sent, claim_context))[: self.max_evidence]

            if not evidence_candidates:
                unsupported += 1
                continue

            verdict_for_sent = "nei"
            for ev in evidence_candidates:
                ev = ev.strip()
                if not ev:
                    continue
                verdict = self._label(sent, ev)
                if verdict == "supported":
                    verdict_for_sent = "supported"
                    break
                if verdict == "refuted":
                    verdict_for_sent = "refuted"
            if verdict_for_sent == "supported":
                supported += 1
            elif verdict_for_sent == "refuted":
                refuted += 1
            else:
                nei += 1

        total = supported + refuted + nei + unsupported
        coverage = 1.0 - (unsupported / total) if total else 0.0
        refute_rate = (refuted / total) if total else 0.0

        return FactPrecisionResult(
            supported=supported,
            refuted=refuted,
            nei=nei,
            unsupported=unsupported,
            refute_rate=refute_rate,
            coverage=coverage,
        )
