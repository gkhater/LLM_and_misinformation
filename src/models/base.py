from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseClassifier(ABC):
    backend: str
    model_name: str

    @abstractmethod
    def classify(self, claim: str, context: Optional[str] = None) -> Dict:
        """Return parsed model output."""


def parse_json_output(raw: str) -> Dict:
    """Best-effort parsing that never raises; returns structured fields with fallbacks."""
    import json
    import re

    safe_raw = "" if raw is None else str(raw)
    fallback = {
        "label": "unknown",
        "confidence": 0.0,
        "rationale": "",
    }

    def _clamp_conf(val: float) -> float:
        return max(0.0, min(1.0, val))

    def _extract_candidate(text: str) -> Optional[str]:
        """Pull the most likely JSON object out of free-form text."""
        # ```json ... ``` fences
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1)
        # Any { ... } block
        brace_block = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_block:
            return brace_block.group(0)
        return None

    stripped = safe_raw.strip()
    def _heuristic_parse(text: str) -> Dict:
        lower = text.lower()

        # Label from explicit prefix or by presence of tokens.
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

        # Confidence from explicit prefix or first float in range 0..1.
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

        # Rationale from explicit prefix or the whole text.
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
            return {
                "label": label,
                "confidence": _clamp_conf(conf_val),
                "rationale": rationale if rationale else stripped,
            }
        except Exception:
            # Fall through to heuristic if JSON parsing fails.
            pass

    # Heuristic fallback if JSON not usable.
    parsed = _heuristic_parse(stripped or safe_raw)
    if not parsed.get("rationale"):
        parsed["rationale"] = stripped or safe_raw
    return parsed
