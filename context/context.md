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
- Retrieval/metrics updates: Fact Precision now supports claim-level retrieval (`query_source: claim` in config) to share evidence across rationale sentences; optional `claim_verification` and `label_consistency` metrics check dataset claims vs evidence separately from model rationales.
- LIAR support: added `scripts/build_liar_topics.py` to derive topics from claims; built a larger LIAR-focused BM25 corpus (`data/liar_passages.tsv`, ~3.5k passages) and set it as the default corpus in `config/base.yaml` with claim-level retrieval (top_k=12, min_score=-1, max_evidence=5). Default NLI is now `MoritzLaurer/deberta-v3-base-mnli-fever-anli` (public).
- New workflow: generation-only phase produces JSONL; offline evaluation reuses shared retrieval + NLI cache with consistent thresholds to avoid re-querying Groq for metric tweaks.
- Claim-level metrics now use max-entail vs max-contradict with a small margin to cut down noisy “mixed” verdicts; thresholds/margin live in `config/eval_liar.yaml`.
- Retrieval now post-filters BM25 hits by lexical overlap/keywords and caps max_hits to reduce irrelevant evidence before NLI; fact_precision uses max-entail vs contradict per sentence for stricter rationale scoring.
- Added embedding reranker (MiniLM) before NLI, enabled refiner (large NLI) after filtering, and a two-hit rule for claim_verification. Latest 20-row eval has claim_verification coverage=0 (all NEI) and label_consistency accuracy=0 on labeled rows, pointing to evidence/corpus weakness rather than plumbing.
- Relaxed decision policy to avoid over-NEI: thresholds 0.33, margin 0.07, final_k=5, strong single-hit override, and sparse refiner (top 2 pairs with fast≥0.55). New 20-row eval (wiki-ish corpus) yields claim_verification coverage 9/20 and label_consistency accuracy 3/13; corpus remains main bottleneck.

## Suggested commit message (once scope agreed)
- “Add modular metrics, BM25 retrieval, and wiki corpus tooling for misinformation benchmarking”
