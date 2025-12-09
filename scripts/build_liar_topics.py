"""
Derive LIAR-related topics from claims.csv-style input to seed corpus building.

Usage (from repo root):
  python scripts/build_liar_topics.py --input data/claims_liar.csv --out data/liar_topics.txt --text-col claim

Outputs a newline-delimited topic list in --out (default: data/liar_topics.txt).
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


# Light stopword list; extend if needed.
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "in",
    "on",
    "for",
    "of",
    "to",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "it",
    "as",
    "by",
    "at",
    "from",
    "with",
    "about",
    "into",
    "over",
    "after",
    "before",
    "says",
    "say",
    "said",
    "claims",
    "claim",
}


def tokenize(text: str):
    return re.findall(r"[A-Za-z][A-Za-z0-9\\-\\']*", text or "")


def main():
    parser = argparse.ArgumentParser(description="Build LIAR topics from claims CSV.")
    parser.add_argument("--input", default="data/claims_liar.csv")
    parser.add_argument("--out", default="data/liar_topics.txt")
    parser.add_argument("--text-col", default="claim", help="Column to read claim text from.")
    parser.add_argument("--top-n", type=int, default=500, help="How many top tokens to keep.")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.out)
    if not inp.exists():
        raise SystemExit(f"Missing input file: {inp}")

    counter = Counter()
    with inp.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if args.text_col not in reader.fieldnames:
            raise SystemExit(f"Column {args.text_col!r} not found in {inp}")
        for row in reader:
            text = row.get(args.text_col, "") or ""
            for tok in tokenize(text):
                tl = tok.lower()
                if tl in STOPWORDS:
                    continue
                if len(tl) < 4:
                    continue
                if tl.isdigit():
                    continue
                counter[tl] += 1

    topics = [w for w, _ in counter.most_common(args.top_n)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for t in topics:
            fh.write(t + "\n")

    print(f"Wrote {len(topics)} topics to {out}")


if __name__ == "__main__":
    main()
