import time

import requests

from .base import BaseClassifier, parse_json_output
from src.prompts import MISINFO_SYSTEM_PROMPT, build_user_payload


class VLLMLlamaClassifier(BaseClassifier):
    backend = "vllm"

    def __init__(self, model_name, base_url, api_key="", timeout=60):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def classify(self, claim, context=None):
        payload = build_user_payload(claim, context)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": MISINFO_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
            "top_p": 1.0,
        }

        t0 = time.time()
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        latency = (time.time() - t0) * 1000

        raw = resp.json()["choices"][0]["message"]["content"]

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
