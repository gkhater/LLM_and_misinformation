# Ops Log: Two-Phase Pipeline & NLI Cache

## Work done
- Added generation-only CLI mode (no metrics) plus offline evaluation mode to reuse outputs without re-querying Groq.
- Introduced shared retrieval + NLI cache (batched, optional two-stage NLI) so fact_precision, claim_verification, and label_consistency consume the same evidence/scores.
- Added eval config (`config/eval_liar.yaml`) and SOP/README updates describing the two-phase flow.
- Extended BM25 retriever with `fetch_with_ids` for evidence provenance in the cache.

## Files touched
- Core: `src/cli.py`, `src/pipeline/generation.py`, `src/pipeline/evaluation.py`, `src/retrieval.py`
- Config/docs: `config/eval_liar.yaml`, `README.md`, `docs/sops/running_benchmarks.md`, `decision.md`, `context/context.md`

## Notes / next steps
- Set `input_jsonl` in `config/eval_liar.yaml` to the actual generation output path before running evaluation.
- NLI cache stored at `outputs/liar_nli_cache.jsonl`; safe to reuse across metric tweaks with the same NLI models.
- Self-consistency metric remains in legacy path; new evaluation path focuses on fact/label checks only.
