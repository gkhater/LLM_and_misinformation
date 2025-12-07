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
    """Best-effort JSON parsing that never raises; returns a safe fallback."""
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

    candidate = _extract_candidate(safe_raw.strip())
    if not candidate:
        return fallback

    try:
        obj = json.loads(candidate)
    except Exception as exc:  # broad on purpose to never raise to caller
        return fallback

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
        "rationale": rationale,
    }
