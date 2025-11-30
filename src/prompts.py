"""Prompt templates for misinformation classification."""

MISINFO_SYSTEM_PROMPT = """You are a strict fact-checker.
Given a claim (and optionally some context), you must decide if the claim is:
- "accurate": well-supported by current evidence.
- "misleading": partially true but framed in a way that could mislead.
- "false": contradicted by reliable evidence.
- "unknown": not enough evidence.

Return ONLY a JSON object with keys:
- "label": ["accurate", "misleading", "false", "unknown"]
- "confidence": float in [0,1]
- "rationale": 2-4 sentences.
"""


def build_user_payload(claim: str, context: str | None) -> str:
    """Serialize the user payload as JSON so providers receive the same structure."""
    import json

    payload: dict[str, str] = {"claim": claim}
    if context:
        payload["context"] = context
    return json.dumps(payload, ensure_ascii=False)
