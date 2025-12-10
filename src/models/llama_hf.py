import time

from transformers import pipeline

from .base import BaseClassifier, parse_json_output
from src.prompts import MISINFO_SYSTEM_PROMPT, build_user_payload


class HFLlamaClassifier(BaseClassifier):
    backend = "hf"

    def __init__(
        self,
        model_name,
        device_map="auto",
        dtype="auto",
        max_new_tokens=512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        task = "text2text-generation" if "t5" in model_name.lower() else "text-generation"

        self.pipe = pipeline(
            task,
            model=model_name,
            tokenizer=model_name,
            device_map=device_map,
            torch_dtype=dtype,
        )
        self._is_seq2seq = task == "text2text-generation"

    def classify(self, claim, context=None, **gen_kwargs):
        payload = build_user_payload(claim, context)
        prompt = (
            MISINFO_SYSTEM_PROMPT
            + "\n\nUser JSON input:\n"
            + payload
            + "\n\nAssistant JSON output:\n"
        )

        temperature = gen_kwargs.get("temperature", self.temperature)
        top_p = gen_kwargs.get("top_p", self.top_p)
        do_sample = temperature > 0

        t0 = time.time()
        result = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )[0]
        if self._is_seq2seq:
            out_text = result.get("generated_text") or result.get("text") or ""
        else:
            out_text = result.get("generated_text") or result.get("text") or ""
        latency = (time.time() - t0) * 1000

        if out_text.startswith(prompt):
            raw = out_text[len(prompt) :].strip()
        else:
            raw = out_text.strip()

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
