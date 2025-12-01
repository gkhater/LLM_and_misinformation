"""
Build a simple TSV BM25 corpus from Wikipedia pages.

Each line: id<TAB>text
Where text includes title and chunked paragraphs for better context.

Usage:
  python scripts/build_wiki_corpus.py --output data/wiki_passages.tsv --pages 300 --lang en

You can also pass a file with topics (one per line) via --topics-file; otherwise
it samples random pages.
"""

import argparse
import math
import os
from typing import Iterable, List

import wikipediaapi


def chunk_text(text: str, chunk_size: int = 160, overlap: int = 20) -> List[str]:
    """Chunk text into overlapping windows of ~chunk_size words."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def fetch_random_pages(api, n: int) -> List[str]:
    # wikipedia-api does not expose random; fall back to a fixed seed list plus
    # nearby pages from popular categories.
    seed = [
        "United States",
        "United Kingdom",
        "China",
        "India",
        "Russia",
        "France",
        "Germany",
        "Japan",
        "Canada",
        "Australia",
        "Climate change",
        "Artificial intelligence",
        "Machine learning",
        "Deep learning",
        "COVID-19 pandemic",
        "World War II",
        "United Nations",
        "European Union",
        "NASA",
        "SpaceX",
        "International Space Station",
        "Human rights",
        "Economics",
        "Inflation",
        "Quantum mechanics",
        "General relativity",
        "Computer science",
        "Internet",
        "Cybersecurity",
        "Data science",
    ]
    return seed[:n]


def fetch_page(api, title: str):
    return api.page(title)


def build_corpus(
    lang: str,
    pages: int,
    output: str,
    topics_file: str | None,
    chunk_size: int,
    overlap: int,
    user_agent: str,
):
    api = wikipediaapi.Wikipedia(
        language=lang,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent=user_agent,
    )

    if topics_file:
        with open(topics_file, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        topics = fetch_random_pages(api, pages)

    os.makedirs(os.path.dirname(output), exist_ok=True)

    total_chunks = 0
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        for idx, title in enumerate(topics):
            try:
                page = fetch_page(api, title)
                if not page.exists():
                    continue
                full_text = page.text
                if not full_text or len(full_text.split()) < 50:
                    continue
                chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
                for c_idx, chunk in enumerate(chunks):
                    doc_id = f"{page.pageid}:{c_idx}"
                    text = f"{page.title} [SEP] {chunk}"
                    f.write(f"{doc_id}\t{text}\n")
                    total_chunks += 1
            except Exception:
                # Skip pages that fail to fetch/parse
                continue
    return total_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/wiki_passages.tsv")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--pages", type=int, default=300)
    parser.add_argument("--topics-file", default=None, help="Optional file with topics, one per line.")
    parser.add_argument("--chunk-size", type=int, default=160)
    parser.add_argument("--overlap", type=int, default=20)
    parser.add_argument(
        "--user-agent",
        default="llm-misinfo-benchmark/0.1 (contact: example@example.com)",
        help="Identify the client for Wikipedia API.",
    )
    args = parser.parse_args()

    total = build_corpus(
        lang=args.lang,
        pages=args.pages,
        output=args.output,
        topics_file=args.topics_file,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        user_agent=args.user_agent,
    )
    print(f"Wrote {total} passages to {args.output}")


if __name__ == "__main__":
    main()
