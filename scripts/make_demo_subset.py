"""
Select a demo subset of claims with non-NEI verdicts from an eval JSONL,
optionally balancing entail and contradict to keep the demo from feeling
one-sided.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", default="outputs/demo_853_eval.jsonl", help="Eval JSONL with metrics.")
    ap.add_argument("--out-csv", default="data/demo_claims_verified.csv", help="Output CSV with verified claims.")
    ap.add_argument("--target-n", type=int, default=25)
    ap.add_argument("--balance", action="store_true", help="Balance entail and contradict roughly equally.")
    args = ap.parse_args()

    rows = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    entails: List[dict] = []
    contradicts: List[dict] = []
    for row in rows:
        cv = (row.get("metrics") or {}).get("claim_verification", {}) or {}
        verdict = cv.get("verdict")
        if verdict not in {"entail", "contradict"}:
            continue
        dest = entails if verdict == "entail" else contradicts
        dest.append(
            {
                "id": row.get("id"),
                "claim": row.get("claim"),
                "label": row.get("label"),
                "split": row.get("split"),
                "verdict": verdict,
                "max_entail": cv.get("max_entail", 0.0),
                "max_contradict": cv.get("max_contradict", 0.0),
                "max_gap": cv.get("max_gap", 0.0),
                "evidence_count": cv.get("evidence_count", 0),
            }
        )

    entails.sort(key=lambda x: (max(x["max_entail"], x["max_contradict"]), x["max_gap"]), reverse=True)
    contradicts.sort(key=lambda x: (max(x["max_entail"], x["max_contradict"]), x["max_gap"]), reverse=True)

    if args.balance:
        half = args.target_n // 2
        scored = entails[:half] + contradicts[: args.target_n - len(entails[:half])]
        # If not enough contradicts, fill from entails, and vice versa
        if len(scored) < args.target_n:
            scored.extend(entails[len(entails[:half]) : args.target_n - len(scored)])
            if len(scored) < args.target_n:
                scored.extend(contradicts[len(contradicts[: args.target_n - len(scored)]) : args.target_n - len(scored)])
    else:
        scored = entails + contradicts
        scored.sort(key=lambda x: (max(x["max_entail"], x["max_contradict"]), x["max_gap"]), reverse=True)
        scored = scored[: args.target_n]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "claim", "label", "split", "verdict", "max_entail", "max_contradict", "max_gap", "evidence_count"],
        )
        writer.writeheader()
        for row in scored:
            writer.writerow(row)
    print(f"Wrote {len(scored)} verified demo claims to {out_path}")


if __name__ == "__main__":
    main()
