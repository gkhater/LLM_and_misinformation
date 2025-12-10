# Ops Log: Two-Phase Pipeline & NLI Cache

## Work done
- Added generation-only CLI mode (no metrics) plus offline evaluation mode to reuse outputs without re-querying Groq.
- Introduced shared retrieval + NLI cache (batched, optional two-stage NLI) so fact_precision, claim_verification, and label_consistency consume the same evidence/scores.
- Added eval config (`config/eval_liar.yaml`) and SOP/README updates describing the two-phase flow.
- Extended BM25 retriever with `fetch_with_ids` for evidence provenance in the cache.
- Tightened aggregation for claim-level metrics: now use max entail vs max contradict with a small margin to reduce spurious “mixed” verdicts; margin configurable in eval config.
- Added retrieval post-filters (lexical overlap + keyword gate + max_hits) and stricter rationale aggregation (max entail vs contradict per sentence) to reduce junk evidence/NLI noise.
- Added MiniLM embedding reranker ahead of NLI, enabled refiner NLI (configurable), and two-hit rule for claim_verification to reduce single-passage noise; cache path now includes refiner model variant.
- Relaxed policy to avoid all-NEI: thresholds 0.33, margin 0.07, final_k=5, strong single-hit override, and sparse refiner (top 2 pairs with fast≥0.55). Latest 20-row eval (wiki-ish corpus) yields claim_verification coverage 9/20 and label_consistency accuracy 3/13; corpus quality is still the limiting factor.

## Files touched
- Core: `src/cli.py`, `src/pipeline/generation.py`, `src/pipeline/evaluation.py`, `src/retrieval.py`
- Config/docs: `config/eval_liar.yaml`, `README.md`, `docs/sops/running_benchmarks.md`, `decision.md`, `context/context.md`

## Notes / next steps
- Set `input_jsonl` in `config/eval_liar.yaml` to the actual generation output path before running evaluation.
- NLI cache stored at `outputs/liar_nli_cache.jsonl`; safe to reuse across metric tweaks with the same NLI models.
- Self-consistency metric remains in legacy path; new evaluation path focuses on fact/label checks only.
- Consider enabling the refiner NLI once downloaded; tune `min_overlap/max_hits/margin` as corpus improves.
- Latest 20-row eval with reranker+refiner shows claim_verification coverage=0 and label_consistency accuracy=0 (labeled rows) — evidence/corpus is now the limiting factor, not plumbing; next step is a better LIAR-aligned corpus.
- With relaxed policy + sparse refiner, coverage improved (9/20) but accuracy remains low due to off-topic corpus; next concrete step is to build/point to a LIAR-aligned evidence set (entity-seeded wiki or fact-check bodies).
