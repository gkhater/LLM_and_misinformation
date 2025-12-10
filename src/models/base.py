from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple


class BaseClassifier(ABC):
    backend: str
    model_name: str

    @abstractmethod
    def classify(self, claim: str, context: Optional[str] = None) -> Dict:
        """Return parsed model output."""


def parse_json_output(raw: str) -> Dict:
    """Best-effort parsing that never raises; returns structured fields with quality flags."""
    import json
    import re

    safe_raw = "" if raw is None else str(raw)

    def _clamp_conf(val: float) -> float:
        return max(0.0, min(1.0, val))

    def _extract_candidate(text: str) -> Optional[str]:
        """Pull the most likely JSON object out of free-form text."""
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1)
        brace_block = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_block:
            return brace_block.group(0)
        return None

    stripped = safe_raw.strip()
    parse_ok = False

    def _heuristic_parse(text: str) -> Dict:
        lower = text.lower()

        lbl = None
        label_match = re.search(r"label\s*[:=-]\s*([a-zA-Z]+)", text, re.IGNORECASE)
        if label_match:
            lbl = label_match.group(1).lower()
        else:
            for cand in ("accurate", "misleading", "false", "unknown"):
                if re.search(rf"\b{cand}\b", lower):
                    lbl = cand
                    break
        if lbl not in {"accurate", "misleading", "false", "unknown"}:
            lbl = "unknown"

        conf_val = 0.0
        conf_match = re.search(r"confidence\s*[:=-]\s*([01](?:\.\d+)?)", text, re.IGNORECASE)
        if conf_match:
            try:
                conf_val = float(conf_match.group(1))
            except Exception:
                conf_val = 0.0
        else:
            num_match = re.search(r"\b([01](?:\.\d+)?)\b", text)
            if num_match:
                try:
                    conf_val = float(num_match.group(1))
                except Exception:
                    conf_val = 0.0

        rationale = text
        rat_match = re.search(r"rationale\s*[:=-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if rat_match:
            rationale = rat_match.group(1).strip()

        return {
            "label": lbl,
            "confidence": _clamp_conf(conf_val),
            "rationale": rationale if rationale else text,
        }

    candidate = _extract_candidate(stripped)
    if candidate:
        try:
            obj = json.loads(candidate)
            label = str(obj.get("label", "")).lower().strip()
            if label not in {"accurate", "misleading", "false", "unknown"}:
                label = "unknown"
            try:
                conf_val = float(obj.get("confidence", 0.0))
            except Exception:
                conf_val = 0.0
            rationale = str(obj.get("rationale", "")).strip()
            parse_ok = True
            parsed = {
                "label": label,
                "confidence": _clamp_conf(conf_val),
                "rationale": rationale if rationale else stripped,
            }
        except Exception:
            parsed = _heuristic_parse(stripped or safe_raw)
    else:
        parsed = _heuristic_parse(stripped or safe_raw)

    quality = _evaluate_quality(parsed, safe_raw, parse_ok)
    parsed["quality"] = quality
    return parsed


# Lightweight quality heuristics; kept near parser for shared use.
def _evaluate_quality(parsed: Dict, raw_text: str, parse_ok: bool) -> Dict:
    import re

    label_ok = parsed.get("label", "") in {"accurate", "misleading", "false", "unknown"}
    conf_val = parsed.get("confidence", 0.0)
    confidence_ok = isinstance(conf_val, (int, float)) and 0.0 <= float(conf_val) <= 1.0

    rationale_text = (parsed.get("rationale") or "").strip()
    tokens = re.findall(r"[A-Za-z0-9']+", rationale_text.lower())
    rationale_len = len(tokens)

    # Unique 3-gram ratio to penalize heavy repetition.
    trigrams = [" ".join(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))]
    unique_trigrams = len(set(trigrams)) if trigrams else 0
    repetition_score = unique_trigrams / float(len(trigrams)) if trigrams else 1.0

    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "for",
        "of",
        "to",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "it",
        "as",
        "by",
        "at",
        "from",
        "with",
        "about",
        "into",
        "over",
        "after",
        "before",
    }
    stop_count = sum(1 for t in tokens if t in stopwords)
    stopword_ratio = (stop_count / rationale_len) if rationale_len else 1.0

    bad_patterns = (
        "as an ai",
        "language model",
        "cannot provide",
        "lorem ipsum",
        "factors factors",
    )
    has_placeholder = any(pat in rationale_text.lower() for pat in bad_patterns)

    rationale_ok = (
        rationale_len >= 20
        and repetition_score >= 0.4
        and stopword_ratio <= 0.5
        and not has_placeholder
    )

    output_ok = bool(parse_ok and label_ok and confidence_ok and rationale_ok)

    return {
        "parse_ok": bool(parse_ok),
        "label_ok": bool(label_ok),
        "confidence_ok": bool(confidence_ok),
        "rationale_ok": bool(rationale_ok),
        "output_ok": bool(output_ok),
        "repetition_score": float(repetition_score),
        "stopword_ratio": float(stopword_ratio),
        "rationale_len": int(rationale_len),
        "has_placeholder": bool(has_placeholder),
    }
