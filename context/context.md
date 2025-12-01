# Project Context

## What’s in the working tree now
- Multi-backend misinformation runner with configurable metrics (Fact Precision + Self-Consistency) wired into `src/`.
- BM25 retrieval backend with `min_score` filtering; retrieval is configurable via `config/base.yaml`.
- Curated Wikipedia seed list (`data/wiki_topics_seed.txt`) and generated BM25 corpus (`data/wiki_passages.tsv`, ~14.6k passages).
- Helper scripts: build corpus, probe retrieval, calibrate NLI thresholds.
- Updated README/decision/config to document metrics and retrieval knobs.
- Logs for corpus build and compile checks.

## What likely belongs in git
- Source code under `src/`, configs (`config/`), docs (`README.md`, `decision.md`, `context/context.md`), scripts (`scripts/`), and seed lists (`data/wiki_topics_seed.txt`).
- Metric design docs (if text/markdown), SOPs, and small test data.

## What likely should be ignored (to discuss)
- Large/generated artifacts: `data/wiki_passages.tsv` (corpus), `outputs/`, caches.
- Logs under `logs/` (unless you want to keep them as provenance).
- Binary docs in `metrics/` (user-provided Word files) if repo should stay lean.

## Current gaps / next steps
- Retrieval quality check: run `scripts/probe_retrieval.py` on known claims to verify top-k relevance; adjust `min_score/top_k` and corpus if needed.
- NLI threshold calibration: create a small labeled dev CSV (claim,evidence,label) and run `scripts/calibrate_nli_thresholds.py`; update `config/base.yaml` with tuned thresholds.
- Claim extraction: currently sentence-split + heuristic filters; consider stronger filtering or a claim extractor for research-grade use.
- Corpus coverage: expand or domain-focus the corpus if evaluation topics go beyond the current seed list.
- Metric docs: finalize formal definitions in README/decision (equations already added in brief).

## Suggested commit message (once scope agreed)
- “Add modular metrics, BM25 retrieval, and wiki corpus tooling for misinformation benchmarking”
