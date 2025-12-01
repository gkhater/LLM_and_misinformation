# LLM Misinformation Benchmarking Pipeline

Runs a consistent misinformation-classification prompt across multiple backends (hosted Groq 70B, self-hosted vLLM, local HF), keeps decoding aligned for fairness, and writes JSONL outputs for downstream scoring.

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
- Copy `.env.example` to `.env`; set `GROQ_API_KEY` (Groq backend only).

4) **Run a backend**
```powershell
# Groq 70B (hosted)
python -m src.cli --model-config config/model_llama70b_groq.yaml

# Self-hosted vLLM endpoint
python -m src.cli --model-config config/model_llama8b_vllm.yaml

# Local HF pipeline (downloads weights)
python -m src.cli --model-config config/model_llama8b_hf.yaml
```

Outputs land in `outputs/` as JSONL; filenames encode backend/model.

## Repository layout
- `config/` – base dataset/output settings + per-backend model configs.
- `src/` – prompt, dataset loader, runner, backends, and metrics.
- `data/` – `claims.csv` with `id,split,claim,context`.
- `data/wiki_passages.tsv` – BM25 corpus built from curated Wikipedia topics (see `data/wiki_topics_seed.txt`).
- `outputs/` – generated JSONL files (gitignored).
- `docs/sops/` – SOPs for setup/runs/extending.
- `decision.md` – running log of nontrivial choices.
- `logs/tests/` – test run records.
- `metrics/` – source notes for metric design.
- `context/context.md` – high-level context, what to keep/ignore, next steps.

## Notes
- Decoding is deterministic (`temperature=0`, `top_p=1`) for fair comparisons.
- Prompting is JSON-based to reduce formatting drift across providers.
- Metrics (optional, set in `config/base.yaml`):
  - `fact_precision`: retrieval + NLI judge; off by default (needs NLI model + retrieval source).
  - `self_consistency`: multi-sample disagreement risk; off by default (re-queries the model `samples` times with nonzero temperature).
  - Fact Precision fields: `fact_precision = S/(S+R+U)`, `refute_rate = R/(S+R+U)`, `coverage = 1 − U/(S+R+U)`, where S/R/U are supported/refuted/unsupported claims.
  - Self-Consistency fields: similarity (mean pairwise embedding cosine), contradiction_rate (optional NLI), risk = 1 − similarity (configurable).

## Retrieval & corpus (for teammates)
- A small Wikipedia-based BM25 corpus is included: `data/wiki_passages.tsv` built from `data/wiki_topics_seed.txt` using `scripts/build_wiki_corpus.py`.
- To use it, set in `config/base.yaml`:
  ```yaml
  metrics:
    fact_precision:
      enabled: true
      retrieval:
        backend: "local_bm25"
        corpus_path: "data/wiki_passages.tsv"
        top_k: 3
        min_score: 0.0
  ```
- To probe retrieval quality, run:
  ```powershell
  python scripts/probe_retrieval.py --claims path/to/claims.txt --corpus data/wiki_passages.tsv --top-k 5 --min-score 0
  ```
- To regenerate/expand the corpus, edit `data/wiki_topics_seed.txt` and run `scripts/build_wiki_corpus.py` (adjust `--topics-file`, `--pages`, `--chunk-size` as needed).
- Large/generated artifacts:
  - `data/wiki_passages.tsv` is included for convenience but can be regenerated; keep it if teammates need a ready corpus, otherwise add to `.gitignore`.
  - `outputs/` and `logs/` are gitignored by default; keep them untracked.
  - `metrics/*.docx` are user-provided; keep or ignore per repo size policy.

## NLI threshold calibration (optional but recommended)
- Prepare a small CSV dev set with `claim,evidence,label` where label ∈ {support, refute, unknown}.
- Run:
  ```powershell
  python scripts/calibrate_nli_thresholds.py --dev dev_labels.csv --model ynie/roberta-large-snli_mnli_fever_anli_R1
  ```
- Update `config/base.yaml` `entail_threshold` / `contradict_threshold` with the tuned values.

## Self-hosted small models
- **vLLM server (OpenAI-compatible)**
  ```bash
  docker run --gpus all --rm -p 8000:8000 vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3-8B-Instruct --dtype float16
  ```
  Point `config/model_llama8b_vllm.yaml.base_url` to your endpoint.

- **Local HF pipeline**
  Running `python -m src.cli --model-config config/model_llama8b_hf.yaml` will download the model locally (CPU/GPU depending on your hardware and `device_map`).
