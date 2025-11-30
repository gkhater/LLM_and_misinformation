# LLM Misinformation Benchmarking Pipeline

This repo runs the same misinformation-classification prompt across multiple backends:

- Hosted `llama3-70b-8192` via Groq (for a large-model baseline).
- Self-hosted small models via vLLM (OpenAI-compatible API).
- Local HuggingFace Transformers pipeline (no API key; CPU/GPU depending on hardware).

The pipeline keeps identical I/O and decoding parameters across backends for fairness and writes JSONL outputs for easy downstream scoring.

## Quickstart

1) **Create & activate venv**

```powershell
python -m venv .venv
.\\.venv\\Scripts\\activate
```

2) **Install deps**

```powershell
pip install -r requirements.txt
```

3) **Configure secrets**

- Copy `.env.example` to `.env` and set `GROQ_API_KEY` (only required for the Groq backend).

4) **Run a backend**

```powershell
# Groq 70B (hosted)
python -m src.cli --model-config config/model_llama70b_groq.yaml

# Self-hosted vLLM endpoint
python -m src.cli --model-config config/model_llama8b_vllm.yaml

# Local HF pipeline (downloads weights)
python -m src.cli --model-config config/model_llama8b_hf.yaml
```

Outputs land in `outputs/` as JSONL; filename encodes backend and model.

## Repository layout

- `config/` – base dataset/output settings + per-backend model configs.
- `src/` – shared prompt, dataset loader, runner, and backend implementations.
- `data/` – `claims.csv` with `id,split,claim,context`.
- `outputs/` – generated JSONL files (gitignored).
- `docs/sops/` – SOPs for setup/runs/extending.
- `decision.md` – running log of nontrivial choices.

## Notes

- Decoding is deterministic (`temperature=0`, `top_p=1`) for fair comparisons.
- Prompting is JSON-based to simplify downstream parsing and reduce formatting variance across providers.

## Self-hosted small models

- **vLLM server (OpenAI-compatible)**
  ```bash
  docker run --gpus all --rm -p 8000:8000 vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3-8B-Instruct --dtype float16
  ```
  Point `config/model_llama8b_vllm.yaml` `base_url` to the server (Auth optional).

- **Local HF pipeline**
  Running `python -m src.cli --model-config config/model_llama8b_hf.yaml` will download the model locally (CPU/GPU depending on your hardware and `device_map`).
