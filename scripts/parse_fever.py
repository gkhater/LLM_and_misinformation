"""
Convert FEVER JSONL splits into the project's CSV schema (id, split, claim, context).

Usage (from repo root):
  python scripts/parse_fever.py --train data/fever_train.jsonl --dev data/fever_dev.jsonl --test data/fever_test.jsonl --out data/claims_fever.csv

Notes:
  - Expects FEVER JSONL files with keys: id, claim, label, evidence.
  - Assigns a "split" column based on which file it came from: train/dev/test.
  - "context" column will contain a simplified list of evidence IDs.
  - Output CSV has columns: id, split, claim, context.
"""

import argparse
from pathlib import Path
import pandas as pd
import json

def load_split(path: Path, split_name: str) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            claim = item.get("claim", "")
            evidence_list = item.get("evidence", [])
            
            # Flatten evidence IDs into a string; use empty string if no evidence
            context = "; ".join(
                str(evi[0][0]) if evi and evi[0][0] is not None else "" for evi in evidence_list
            )
            rows.append({
                "split": split_name,
                "claim": claim,
                "context": context
            })
    return pd.DataFrame(rows)

def convert_fever(train: Path, dev: Path, test: Path) -> pd.DataFrame:
    parts = []
    if train and train.exists():
        parts.append(load_split(train, "train"))
    if dev and dev.exists():
        parts.append(load_split(dev, "dev"))
    if test and test.exists():
        parts.append(load_split(test, "test"))
    if not parts:
        raise FileNotFoundError("No FEVER splits found to convert.")
    
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "id", range(len(df)))
    return df

def main():
    parser = argparse.ArgumentParser(description="Convert FEVER JSONL files to claims CSV.")
    parser.add_argument("--train", required=False, default="data/fever_train.jsonl")
    parser.add_argument("--dev", required=False, default="data/fever_dev.jsonl")
    parser.add_argument("--test", required=False, default="data/fever_test.jsonl")
    parser.add_argument("--out", required=False, default="data/claims_fever.csv")
    args = parser.parse_args()

    train_path = Path(args.train)
    dev_path = Path(args.dev)
    test_path = Path(args.test)
    out_path = Path(args.out)

    df = convert_fever(train_path, dev_path, test_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path} with splits {df['split'].unique().tolist()}")

if __name__ == "__main__":
    main()
