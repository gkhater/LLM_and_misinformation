"""
Create a fixed slice CSV from a larger claims CSV for reproducible runs.
"""

from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input claims CSV")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--split", default=None, help="Optional split filter")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if args.split and "split" in df.columns:
        df = df[df["split"] == args.split]
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
