# 2025-12-10 Wiki smoke evaluation (CSV-only)

## Setup
- Config: `config/eval_wiki_smoke.yaml`
- Inputs: `data/claims_liar.csv` (no generation JSONL), `data/wiki_seeded_passages.tsv` (11 passages from 3 claims, corpus_hash=fee05f97ef33a528160515d343ce6349b4050f99)
- Reranker: disabled (offline); BM25 top_k=50; post-filter overlap=0, keyword/numeric gates off.
- NLI: `typeform/distilbert-base-uncased-mnli` (loaded after retry; offline fallback kept process running).
- Cache: `outputs/wiki_smoke_nli_cache_v2.jsonl`

## Commands
```powershell
# Run eval on 5 rows from CSV
.\.venv\Scripts\python -m src.cli --mode evaluation --eval-config config/eval_wiki_smoke.yaml --max-rows 5 --debug-ids 1,2
```

## Outcome
- Evaluation completed, wrote `outputs/wiki_smoke_eval.jsonl` (5 rows).
- Retrieval_stats for all 5 rows: bm25_candidates=50, post_filter_kept=20, rerank_kept=20, final_k=20 (no evidence filtering).
- Verdicts: some claims marked `contradict` due to NLI scores from small corpus; others `nei`.
- HF download attempts still logged (Hugging Face blocked), but pipeline continued with local/partial model.

## Notes
- Max-row handling fixed to honor CLI/config `max_rows`.
- To reduce noisy contradicts, expand corpus beyond 11 passages and/or tighten filter/overlap once retrieval is richer.

---

## 2025-12-10 12:45Z wiring + veto debug
- Config: `config/eval_wiki_smoke.yaml` (min_overlap=0.15, keyword/numeric gates on, reranker disabled, final_k=3, thresholds 0.45/0.45/margin 0.10, strong 0.90/0.20).
- Corpus: `data/wiki_seeded_passages.tsv` (size 853, hash 1dc8c1ffc73c0a7386beb1983015fe8da29bfe04).
- Cache rebuilt: `outputs/wiki_smoke_nli_cache_v3.jsonl` (previous cache removed).
- Changes: `_post_filter_hits` now returns metadata (`query_tokens`, veto flags), retrieval_stats carried through cache entry, JSON-safe defaults enforced; file re-encoded to UTF-8.
- Command: `.\.venv\Scripts\python -m src.cli --mode evaluation --eval-config config/eval_wiki_smoke.yaml --max-rows 5 --debug-ids 0,1`.
- Result: `outputs/wiki_smoke_eval.jsonl` shows populated `query_tokens_all`, entity_veto fields, and corpus metadata. All 5 claims -> `nei` due to entity veto on tiny corpus; no spurious contradictions. Next step: loosen gates or expand corpus to get on-topic evidence.
