"""
Probe BM25 retrieval for a set of claims.

Usage:
  python scripts/probe_retrieval.py --claims claims.txt --corpus data/wiki_passages.tsv --top-k 5 --min-score 0

claims.txt: one claim per line.
Outputs top-k passages with scores for manual inspection.
"""

import argparse
import csv
import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi


def load_corpus(tsv_path: str, text_col: int = 1, delimiter: str = "\t"):
    docs = []
    ids = []
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) <= text_col:
                continue
            ids.append(row[0])
            docs.append(row[text_col])
    return ids, docs


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return re.findall(r"\b\w+\b", text)


def probe(claims: List[str], ids: List[str], docs: List[str], top_k: int, min_score: float):
    tokenized = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)

    for claim in claims:
        tokens = tokenize(claim)
        scores = bm25.get_scores(tokens)
        candidates = [(i, scores[i]) for i in range(len(scores)) if scores[i] >= min_score]
        top_idxs = [i for i, _ in sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]]
        print(f"\nCLAIM: {claim}")
        for rank, idx in enumerate(top_idxs, 1):
            print(f"  #{rank} (score={scores[idx]:.2f}) id={ids[idx]} :: {docs[idx][:200]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True, help="File with one claim per line.")
    parser.add_argument("--corpus", required=True, help="TSV corpus (id<TAB>text).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    ids, docs = load_corpus(args.corpus)
    with open(args.claims, "r", encoding="utf-8") as f:
        claims = [line.strip() for line in f if line.strip()]

    probe(claims, ids, docs, args.top_k, args.min_score)


if __name__ == "__main__":
    main()
