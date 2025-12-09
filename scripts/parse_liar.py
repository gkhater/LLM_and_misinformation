"""
Convert LIAR TSV splits into the project's CSV schema (id, split, claim, context).

Usage (from repo root):
  python scripts/parse_liar.py --train data/liar_train.tsv --valid data/liar_valid.tsv --test data/liar_test.tsv --out data/claims_liar.csv

Notes:
  - Expects the standard LIAR columns in order; only "statement" and optional "context" are used.
  - Assigns contiguous integer ids across splits.
  - The resulting CSV has columns: id, split, claim, context.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


LIAR_COLUMNS = [
    "label",
    "statement",
    "subject",
    "speaker",
    "job",
    "state",
    "party",
    "barely_true_ct",
    "false_ct",
    "half_true_ct",
    "mostly_true_ct",
    "pants_on_fire_ct",
    "context",
]


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
    out = pd.DataFrame(
        {
            "split": split_name,
            "claim": df["statement"],
            "context": df["context"].fillna(""),
            "label": df["label"].str.lower().str.strip(),
        }
    )
    return out


def convert_liar(train: Path, valid: Path, test: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    if train and train.exists():
        parts.append(load_split(train, "train"))
    if valid and valid.exists():
        parts.append(load_split(valid, "valid"))
    if test and test.exists():
        parts.append(load_split(test, "test"))
    if not parts:
        raise FileNotFoundError("No LIAR splits found to convert.")
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "id", range(len(df)))
    return df


def main():
    parser = argparse.ArgumentParser(description="Convert LIAR TSV files to claims CSV.")
    parser.add_argument("--train", required=False, default="data/liar_train.tsv")
    parser.add_argument("--valid", required=False, default="data/liar_valid.tsv")
    parser.add_argument("--test", required=False, default="data/liar_test.tsv")
    parser.add_argument("--out", required=False, default="data/claims_liar.csv")
    args = parser.parse_args()

    train_path = Path(args.train)
    valid_path = Path(args.valid)
    test_path = Path(args.test)
    out_path = Path(args.out)

    df = convert_liar(train_path, valid_path, test_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path} with splits {df['split'].unique().tolist()}")


if __name__ == "__main__":
    main()
