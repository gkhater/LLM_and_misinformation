"""
Convert BEIR FEVER corpus.jsonl to a TSV that matches our BM25 pipeline.

Input JSONL rows have fields: _id, title, text.
Output TSV: passage_id \\t title \\t text (chunked).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Tuple
import time


def chunk_text(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    toks = text.split()
    if not toks:
        return []
    chunks = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(toks), step):
        end = start + chunk_tokens
        chunk = " ".join(toks[start:end]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_corpus(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to BEIR FEVER corpus.jsonl")
    ap.add_argument("--out", required=True, help="Output TSV path")
    ap.add_argument("--chunk_tokens", type=int, default=180)
    ap.add_argument("--overlap_tokens", type=int, default=30)
    ap.add_argument("--max_docs", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    doc_count = 0
    chunk_count = 0
    last_log = time.time()

    with out.open("w", encoding="utf-8", newline="") as fh:
        for doc in iter_corpus(inp):
            doc_count += 1
            if args.max_docs and doc_count > args.max_docs:
                break
            doc_id = doc.get("_id") or str(doc_count)
            title = (doc.get("title") or "").replace("\t", " ").strip()
            body = (doc.get("text") or "").replace("\t", " ").strip()
            text = f"{title}\n{body}".strip() if title else body
            for idx, chunk in enumerate(chunk_text(text, args.chunk_tokens, args.overlap_tokens)):
                h = hash_text(chunk)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                pid = f"{doc_id}:{idx}"
                fh.write(f"{pid}\t{title}\t{chunk}\n")
                chunk_count += 1
            if doc_count % 10000 == 0 or (time.time() - last_log) > 30:
                print(f"[progress] docs={doc_count} chunks={chunk_count}")
                last_log = time.time()

    print(f"Done. Docs processed={doc_count}, chunks written={chunk_count}, output={out}")


if __name__ == "__main__":
    main()
