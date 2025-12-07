# Ops Log: Metric Tightening & Integration

## Work done
- Updated metric definitions:
  - Fact Precision: per-sentence claims, top-k evidence aggregation, thresholded NLI (entail/contradict) with unsupported when evidence missing/empty; logs fact_precision, refute_rate, coverage.
  - Self-Consistency: now uses sentence embeddings (optional) + lexical similarity; optional NLI contradictions; returns consistency, risk, avg_similarity, contradiction_rate.
- Added retrieval builder stub:
  - `ContextOnlyRetrieval` default; added local BM25 retriever over TSV corpus (id<TAB>text) using `rank_bm25`. Configurable backend hook; default remains context-only.
- Built a curated Wikipedia seed corpus (~14.6k passages) and added probe/calibration scaffolding scripts.
- Added config knobs:
  - `metrics.fact_precision`: thresholds, top_k, retrieval backend, NLI model.
  - `metrics.self_consistency`: samples, temperature, embedding model, optional NLI.
- Backends now accept generation params from config for metric sampling.
- Installed `sentence-transformers` (with deps).
- Tests: `python -m compileall src` pass (logged separately).

## Files touched (key)
- Config: `config/base.yaml`
- Metrics: `src/metrics/fact_precision.py`, `src/metrics/self_consistency.py`
- Retrieval: `src/retrieval.py`
- Runner: `src/runner.py`
- Backends/CLI: `src/models/*`, `src/cli.py`
- Requirements: `requirements.txt`
- Logs: `logs/tests/2025-12-01_compileall_metrics.md`

## Notes / caveats
- Retrieval backend for Wikipedia BM25 not yet implemented; placeholder raises. Default uses context-only evidence.
- Fact Precision thresholds default to 0.5/0.5 with margin; NLI calibration dataset not yet provided.
- Self-Consistency embedding model defaults to `sentence-transformers/all-MiniLM-L6-v2`; NLI for contradictions is optional and off by default.
