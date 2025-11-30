import time

from transformers import pipeline

from .base import BaseClassifier, parse_json_output
from src.prompts import MISINFO_SYSTEM_PROMPT, build_user_payload


class HFLlamaClassifier(BaseClassifier):
    backend = "hf"

    def __init__(self, model_name, device_map="auto", dtype="auto", max_new_tokens=512):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )

    def classify(self, claim, context=None):
        payload = build_user_payload(claim, context)
        prompt = (
            MISINFO_SYSTEM_PROMPT
            + "\n\nUser JSON input:\n"
            + payload
            + "\n\nAssistant JSON output:\n"
        )

        t0 = time.time()
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )[0]["generated_text"]
        latency = (time.time() - t0) * 1000

        if out.startswith(prompt):
            raw = out[len(prompt) :].strip()
        else:
            raw = out.strip()

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
