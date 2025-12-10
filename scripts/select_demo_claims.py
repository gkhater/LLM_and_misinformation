"""
Select a small set of claims that are better covered by the frozen demo corpus.

We score each claim by how many passages survive the same post-filter + token
gate used in evaluation, then keep the top N.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict
import sys

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from src.retrieval import build_retriever
from src.pipeline.evaluation import _post_filter_hits, extract_query_tokens, _tokenize


def dedup_hits(hits: List[tuple]) -> List[tuple]:
    seen = set()
    out = []
    for pid, txt in hits:
        if pid in seen:
            continue
        seen.add(pid)
        out.append((pid, txt))
    return out


def token_gate_passages(hits: List[tuple], query_tokens: List[str], min_match: int = 2) -> Tuple[List[tuple], int]:
    """Drop passages failing token match threshold."""
    kept = []
    max_match = 0
    for pid, txt in hits:
        txt_lower = (txt or "").lower()
        title_lower = txt_lower.split("\n", 1)[0] if "\n" in txt_lower else txt_lower
        match_count = sum(1 for tok in query_tokens if tok in txt_lower)
        max_match = max(max_match, match_count)
        matched_title = any(tok in title_lower for tok in query_tokens)
        if match_count >= min_match or matched_title:
            kept.append((pid, txt))
    return kept, max_match


def score_claim(claim: str, retriever, top_k: int, min_overlap: float, max_hits: int, require_keyword: bool, numeric_time_gate: bool, final_k: int) -> Dict:
    evidence = retriever.fetch_with_ids(claim, None, top_k=top_k)
    evidence = dedup_hits(evidence)
    evidence, filter_meta = _post_filter_hits(
        claim,
        evidence,
        min_overlap=min_overlap,
        max_hits=max_hits,
        require_keyword=require_keyword,
        numeric_time_gate=numeric_time_gate,
        final_k=final_k,
    )
    query_tokens = filter_meta.get("query_tokens_all", []) or extract_query_tokens(claim).get("must_match_any", [])
    kept, max_match = token_gate_passages(evidence, query_tokens, min_match=2)
    if not kept:
        kept, max_match = token_gate_passages(evidence, query_tokens, min_match=1)
    return {
        "kept_count": len(kept),
        "raw_hits": len(evidence),
        "max_match": max_match,
        "filter_meta": filter_meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="data/claims_liar.csv", help="Claims CSV path")
    ap.add_argument("--corpus", default="data/wiki_seeded_passages_demo_853.tsv", help="Corpus TSV path")
    ap.add_argument("--out", default="data/demo_claims_covered.csv", help="Output CSV path")
    ap.add_argument("--max-claims", type=int, default=50, help="How many claims to keep")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--min-overlap", type=float, default=0.08)
    ap.add_argument("--max-hits", type=int, default=20)
    ap.add_argument("--final-k", type=int, default=5)
    ap.add_argument("--require-keyword", action="store_true", default=True)
    ap.add_argument("--numeric-time-gate", action="store_true", default=True)
    args = ap.parse_args()

    df = pd.read_csv(args.claims)
    retriever = build_retriever(
        {
            "backend": "local_bm25",
            "corpus_path": args.corpus,
        }
    )

    scored = []
    for _, row in df.iterrows():
        claim = str(row.get("claim", ""))
        res = score_claim(
            claim,
            retriever,
            top_k=args.top_k,
            min_overlap=args.min_overlap,
            max_hits=args.max_hits,
            require_keyword=args.require_keyword,
            numeric_time_gate=args.numeric_time_gate,
            final_k=args.final_k,
        )
        scored.append(
            {
                "id": row.get("id"),
                "claim": claim,
                "context": row.get("context"),
                "label": row.get("label"),
                "split": row.get("split"),
                "kept_count": res["kept_count"],
                "raw_hits": res["raw_hits"],
                "max_match": res["max_match"],
            }
        )

    scored.sort(key=lambda x: (x["kept_count"], x["raw_hits"], x["max_match"]), reverse=True)
    kept = scored[: args.max_claims]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "claim", "context", "label", "split"])
        writer.writeheader()
        for row in kept:
            writer.writerow({k: row.get(k) for k in ["id", "claim", "context", "label", "split"]})
    print(f"Wrote {len(kept)} claims to {out_path}")


if __name__ == "__main__":
    main()
