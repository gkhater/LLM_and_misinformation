# Design Decisions & Rationale
<!-- Note, these decisions were taken by agent AI systems, this is just for us to be able to understand -->
- Established a multi-backend architecture (Groq hosted 70B, self-hosted vLLM, local HF) with a shared `BaseClassifier` interface to enforce identical I/O and decoding parameters across models for fair comparisons.
- Switched prompting to a JSON-based contract (`MISINFO_SYSTEM_PROMPT` + serialized user payload) to reduce formatting drift across providers and make parsing resilient.
- Added deterministic decoding defaults (`temperature=0.0`, `top_p=1.0`, fixed max tokens) to keep outputs comparable between large and small models.
- Introduced structured configuration (`config/base.yaml` + per-model YAMLs) so runs are reproducible and swappable without code changes.
- Added dataset split handling (`split` column with filter in config) so evaluation/test subsets stay reproducible and clean.
- Output schema standardized to JSONL with metadata (id, claim/context, backend, model, latency, timestamp, raw + parsed output) for downstream scoring and audits.
- Provided SOPs and README to make environment setup, runs, and extension to new models repeatable without tribal knowledge.
- Default CLI target is Groq 70B (for convenience); swap backends by pointing `--model-config` to the desired YAML.
- Installed CPU Torch wheels by default to avoid GPU-specific constraints during setup; adjust the pip index if you want CUDA builds.
