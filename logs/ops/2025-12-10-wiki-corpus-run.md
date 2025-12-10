# 2025-12-10 Wikipedia seeded corpus run attempt

## What I tried
- Ran `python scripts/build_wiki_seeded_corpus.py --claims data/claims_liar.csv --out data/wiki_seeded_passages.tsv --max-claims-for-corpus 2000 --seed 13 --resume` (networked Wikipedia fetch).

## Outcome
- Command timed out twice (once default, once with 120s timeout) under restricted network; no corpus built or updated.

## Notes
- Syntax check already passed (`python -m compileall src scripts`).
- Build remains blocked on outbound Wikipedia access; rerun once network is available or provide an offline dump.
