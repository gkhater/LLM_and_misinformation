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
    """Best-effort JSON parsing with safeguards."""
    import json

    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        s = raw.find("{")
        e = raw.rfind("}")
        if s == -1 or e == -1 or e <= s:
            raise
        obj = json.loads(raw[s : e + 1])

    label = str(obj.get("label", "")).lower().strip()
    if label not in {"accurate", "misleading", "false", "unknown"}:
        label = "unknown"

    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    rationale = str(obj.get("rationale", "")).strip()

    return {
        "label": label,
        "confidence": conf,
        "rationale": rationale,
    }
