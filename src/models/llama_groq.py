import os
import time

from groq import Groq

from .base import BaseClassifier, parse_json_output
from src.prompts import MISINFO_SYSTEM_PROMPT, build_user_payload


class GroqLlamaClassifier(BaseClassifier):
    backend = "groq"

    def __init__(self, model_name: str, api_key_env: str):
        api = os.getenv(api_key_env)
        if not api:
            raise RuntimeError(f"Missing env var: {api_key_env}")
        self.client = Groq(api_key=api)
        self.model_name = model_name

    def classify(self, claim, context=None):
        payload = build_user_payload(claim, context)

        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": MISINFO_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            temperature=0.0,
            max_tokens=512,
            top_p=1.0,
        )
        latency = (time.time() - t0) * 1000
        raw = resp.choices[0].message.content

        parsed = parse_json_output(raw)
        parsed.update(
            {
                "raw_output": raw,
                "model_name": self.model_name,
                "backend": self.backend,
                "latency_ms": latency,
            }
        )
        return parsed
