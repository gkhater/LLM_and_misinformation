

# LLM Misinformation Evaluation Pipeline (Multi-Backend)

This document describes how to build a **self-contained, modular evaluation pipeline** for misinformation classification using:

* **Groq Llama-3 70B** (already implemented)
* **Self-hosted small models (7–8B)** using either:

  * **vLLM** (OpenAI-compatible HTTP server)
  * **HuggingFace Transformers** (local inference, no HTTP)

The goal is to keep **identical I/O** across all models for **fair evaluation**.

This document includes:
✔ Repository structure
✔ Required files
✔ Full system architecture
✔ Config formats
✔ Backend implementations
✔ vLLM server instructions
✔ Decision-logging & Git usage
✔ Reproducibility scripts
✔ Everything needed for an agentic system to build the full project end-to-end

---

# 1. Repository Structure

```
llm-misinfo-eval/
├── README.md
├── DECISIONS.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config/
│   ├── base.yaml
│   ├── model_llama70b_groq.yaml
│   ├── model_llama8b_vllm.yaml
│   ├── model_llama8b_hf.yaml
├── data/
│   ├── claims.csv
│   └── README.md
├── outputs/
│   ├── README.md
│   └── (generated JSONL files)
├── src/
│   ├── __init__.py
│   ├── prompts.py
│   ├── dataset.py
│   ├── runner.py
│   ├── cli.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── llama_groq.py
│   │   ├── llama_vllm.py
│   │   └── llama_hf.py
│   └── utils/
│       ├── logging_utils.py
│       └── timing.py
└── scripts/
    ├── run_llama70b_groq.sh
    ├── run_llama8b_vllm.sh
    └── run_llama8b_hf.sh
```

---

# 2. Requirements

`requirements.txt`:

```
python-dotenv
pandas
tqdm
requests
pyyaml

# HF backend
transformers
accelerate
sentencepiece

# If keeping Llama-70B Groq backend:
groq
```

Create the venv:

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

# 3. Configuration Files (YAML)

## `config/base.yaml`

```yaml
dataset:
  csv_path: "data/claims.csv"
  id_column: "id"
  claim_column: "claim"
  context_column: "context"

output:
  dir: "outputs"
  filename_prefix: "results"

generation:
  temperature: 0.0
  max_tokens: 512
  top_p: 1.0
  repetition_penalty: 1.0
```

## `config/model_llama70b_groq.yaml`

```yaml
model:
  name: "llama3-3-70b-versatile"
  backend: "groq"
  groq:
    api_key_env: "GROQ_API_KEY"
```

## `config/model_llama8b_vllm.yaml`

```yaml
model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"
  backend: "vllm"
  vllm:
    base_url: "http://127.0.0.1:8000/v1"
    api_key: ""
```

## `config/model_llama8b_hf.yaml`

```yaml
model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"
  backend: "hf"
  hf:
    device_map: "auto"
    dtype: "auto"
```

---

# 4. Prompts: `src/prompts.py`

```python
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

def build_user_payload(claim: str, context: str | None):
    import json
    payload = {"claim": claim}
    if context:
        payload["context"] = context
    return json.dumps(payload, ensure_ascii=False)
```

---

# 5. Dataset Loading: `src/dataset.py`

```python
import pandas as pd

def load_dataset(config):
    path = config["dataset"]["csv_path"]
    df = pd.read_csv(path)

    id_col = config["dataset"]["id_column"]
    claim_col = config["dataset"]["claim_column"]

    for c in [id_col, claim_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    return df
```

---

# 6. Base Classifier: `src/models/base.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, claim: str, context: Optional[str] = None) -> Dict:
        ...

def parse_json_output(raw: str) -> Dict:
    import json

    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        s = raw.find("{")
        e = raw.rfind("}")
        if s == -1 or e == -1 or e <= s:
            raise
        obj = json.loads(raw[s:e+1])

    label = str(obj.get("label", "")).lower().strip()
    if label not in {"accurate", "misleading", "false", "unknown"}:
        label = "unknown"

    try:
        conf = float(obj.get("confidence", 0.0))
    except:
        conf = 0.0
    conf = max(0, min(1, conf))

    rationale = str(obj.get("rationale", "")).strip()

    return {
        "label": label,
        "confidence": conf,
        "rationale": rationale,
    }
```

---

# 7. Groq Backend: `src/models/llama_groq.py`

```python
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
        )
        latency = (time.time() - t0) * 1000
        raw = resp.choices[0].message.content

        parsed = parse_json_output(raw)
        parsed.update({
            "raw_output": raw,
            "model_name": self.model_name,
            "backend": "groq",
            "latency_ms": latency,
        })
        return parsed
```

---

# 8. vLLM Backend: `src/models/llama_vllm.py`

```python
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
        parsed.update({
            "raw_output": raw,
            "model_name": self.model_name,
            "backend": "vllm",
            "latency_ms": latency,
        })
        return parsed
```

---

# 9. HF Transformers Backend: `src/models/llama_hf.py`

```python
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
            raw = out[len(prompt):].strip()
        else:
            raw = out.strip()

        parsed = parse_json_output(raw)
        parsed.update({
            "raw_output": raw,
            "model_name": self.model_name,
            "backend": "hf",
            "latency_ms": latency,
        })
        return parsed
```

---

# 10. Runner: `src/runner.py`

```python
import json, os
from tqdm import tqdm
from src.dataset import load_dataset
from src.utils.timing import utc_now_iso

def run_model_on_dataset(config, classifier):
    df = load_dataset(config)

    out_dir = config["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    prefix = config["output"]["filename_prefix"]
    model_name = classifier.model_name.replace("/", "_")
    backend = classifier.backend

    out_path = os.path.join(out_dir, f"{prefix}_{backend}_{model_name}.jsonl")

    id_col = config["dataset"]["id_column"]
    claim_col = config["dataset"]["claim_column"]
    context_col = config["dataset"].get("context_column")

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            claim = row[claim_col]
            ctx = row[context_col] if context_col and context_col in df.columns else None
            item_id = row[id_col]

            out = classifier.classify(str(claim), ctx if isinstance(ctx, str) else None)

            record = {
                "id": int(item_id),
                "claim": claim,
                "context": ctx,
                "model_output": out,
                "timestamp": utc_now_iso(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path
```

---

# 11. CLI Entrypoint: `src/cli.py`

```python
import argparse, yaml
from src.runner import run_model_on_dataset
from src.models.llama_groq import GroqLlamaClassifier
from src.models.llama_vllm import VLLMLlamaClassifier
from src.models.llama_hf import HFLlamaClassifier

def load_config(base_path, model_path):
    with open(base_path) as f:
        base = yaml.safe_load(f)
    with open(model_path) as f:
        model = yaml.safe_load(f)
    base["model"] = model["model"]
    return base

def build_classifier(config):
    m = config["model"]
    backend = m["backend"]
    name = m["name"]

    if backend == "groq":
        return GroqLlamaClassifier(name, m["groq"]["api_key_env"])
    elif backend == "vllm":
        v = m["vllm"]
        return VLLMLlamaClassifier(name, v["base_url"], v.get("api_key", ""))
    elif backend == "hf":
        h = m["hf"]
        return HFLlamaClassifier(name, h.get("device_map","auto"), h.get("dtype","auto"))
    else:
        raise ValueError(f"Unknown backend {backend}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default="config/base.yaml")
    p.add_argument("--model-config", required=True)
    args = p.parse_args()

    config = load_config(args.base_config, args.model_config)
    classifier = build_classifier(config)

    out_path = run_model_on_dataset(config, classifier)
    print("Results saved to:", out_path)

if __name__ == "__main__":
    main()
```

---

# 12. Utility Files

## `src/utils/timing.py`

```python
from datetime import datetime, timezone
def utc_now_iso(): return datetime.now(timezone.utc).isoformat()
```

## `src/utils/logging_utils.py`

```python
# Placeholder for decision logging; extend as needed.
```

---

# 13. Shell Scripts (`scripts/`)

### `run_llama70b_groq.sh`

```bash
#!/usr/bin/env bash
source .venv/bin/activate
python -m src.cli --model-config config/model_llama70b_groq.yaml
```

### `run_llama8b_vllm.sh`

```bash
#!/usr/bin/env bash
source .venv/bin/activate
python -m src.cli --model-config config/model_llama8b_vllm.yaml
```

### `run_llama8b_hf.sh`

```bash
#!/usr/bin/env bash
source .venv/bin/activate
python -m src.cli --model-config config/model_llama8b_hf.yaml
```

Make executable:

```bash
chmod +x scripts/*.sh
```

---

# 14. vLLM Server Instructions (Self-Hosted Small Model)

On a machine with ≥16GB VRAM:

```bash
docker run --gpus all --rm \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype float16
```

Endpoint:
`http://YOUR-IP:8000/v1/chat/completions`

---

# 15. DECISIONS.md Template

```
# Design Decisions

## 1. Architecture
- Single BaseClassifier interface for multi-backend consistency.

## 2. Prompt format
- JSON-based user input for robust parsing.

## 3. Output schema
- Unified {label, confidence, rationale} for fair comparison.

## 4. Backends
- groq: hosted 70B
- vllm: self-hosted 7–8B via HTTP
- hf: local transformers

## 5. Deterministic decoding
- temperature=0.0 for all backends.
```

---

# 16. .gitignore

```
.venv/
__pycache__/
*.pyc
.env
outputs/
```

---

# 17. Validation Steps

1. Ensure repository imports compile:

```bash
python -m compileall src
```

2. Smoke-test HF backend (if GPU available):

```bash
python -m src.cli --model-config config/model_llama8b_hf.yaml
```

3. Smoke-test vLLM backend (if server running):

```bash
python -m src.cli --model-config config/model_llama8b_vllm.yaml
```

4. Validate JSONL output correctness.
