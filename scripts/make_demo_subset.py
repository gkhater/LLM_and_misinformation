"""
Select a demo subset of claims with non-NEI verdicts from an eval JSONL,
optionally balancing entail and contradict to keep the demo from feeling
one-sided and filtering for minimum confidence.
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
    ap.add_argument("--min-max-gap", type=float, default=0.0, help="Filter out low-confidence cases.")
    ap.add_argument(
        "--require-lc-success",
        action="store_true",
        help="Keep only rows where label_consistency.success is true to boost demo accuracy.",
    )
    ap.add_argument("--min-entail", type=int, default=0, help="Minimum entail quota when balancing.")
    ap.add_argument("--min-contradict", type=int, default=0, help="Minimum contradict quota when balancing.")
    ap.add_argument("--min-evidence", type=int, default=0, help="Minimum evidence_count required.")
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
        if cv.get("max_gap", 0.0) < args.min_max_gap:
            continue
        if cv.get("evidence_count", 0) < args.min_evidence:
            continue
        lc = (row.get("metrics") or {}).get("label_consistency", {}) or {}
        lc_success = lc.get("success")
        if args.require_lc_success and not lc_success:
            continue
        max_edge = max(cv.get("max_entail", 0.0), cv.get("max_contradict", 0.0))
        score = max_edge + 0.5 * cv.get("max_gap", 0.0)
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
                "lc_success": lc_success,
                "score": score,
            }
        )

    entails.sort(key=lambda x: x["score"], reverse=True)
    contradicts.sort(key=lambda x: x["score"], reverse=True)

    if args.balance:
        entail_quota = max(args.min_entail, args.target_n // 2)
        contr_quota = max(args.min_contradict, args.target_n - entail_quota)
        take_ent = entails[: entail_quota]
        take_con = contradicts[: contr_quota]
        scored = take_ent + take_con
        # Fill remaining slots if one side is short
        if len(scored) < args.target_n:
            remaining = args.target_n - len(scored)
            # Prefer the side with more remaining
            pool = entails[len(take_ent) :] + contradicts[len(take_con) :]
            pool.sort(key=lambda x: x["score"], reverse=True)
            scored.extend(pool[:remaining])
    else:
        scored = entails + contradicts
        scored.sort(key=lambda x: x["score"], reverse=True)
        scored = scored[: args.target_n]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "claim",
                "label",
                "split",
                "verdict",
                "max_entail",
                "max_contradict",
                "max_gap",
                "evidence_count",
                "lc_success",
                "score",
            ],
        )
        writer.writeheader()
        for row in scored:
            writer.writerow(row)
    print(f"Wrote {len(scored)} verified demo claims to {out_path}")


if __name__ == "__main__":
    main()
