# Ops Log: BM25 threshold & retrieval tweaks

## Work done
- Added `min_score` filtering to local BM25 retrieval (ignore low-scoring passages).
- Exposed `min_score` in `config/base.yaml` under `metrics.fact_precision.retrieval`.
- Retrieval fetch now respects `top_k` and `min_score`; empty results trigger unsupported in Fact Precision.
- Ran `python -m compileall src` (pass).

## Files touched
- Config: `config/base.yaml`
- Retrieval: `src/retrieval.py`
- Tests: `logs/tests/2025-12-01_compileall_after_min_score.md`

## Notes
- Corpus format remains TSV `id<TAB>text`. Add titles/sections inside the text column if desired for interpretability.
- BM25 backend still local-only; default remains `context_only`.
