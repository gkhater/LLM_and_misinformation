"""
Print a demo report from an eval JSONL, keeping only top passages per verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", default="outputs/demo_verified_eval.jsonl")
    ap.add_argument("--out-md", default="outputs/demo_report.md")
    ap.add_argument("--max-passages", type=int, default=2)
    args = ap.parse_args()

    rows = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    lines = ["# Demo Report", ""]
    for row in rows:
        claim = row.get("claim", "")
        label = row.get("label", "")
        metrics = row.get("metrics", {})
        cv = metrics.get("claim_verification", {}) or {}
        verdict = cv.get("verdict", "nei")
        rule = cv.get("rule_fired", "")
        lines.append(f"## Claim {row.get('id')}: {claim}")
        lines.append(f"- Label: {label} | Verdict: {verdict} | Rule: {rule}")
        lines.append(f"- max_entail={cv.get('max_entail',0):.2f} | max_contradict={cv.get('max_contradict',0):.2f} | max_gap={cv.get('max_gap',0):.2f}")
        evidence = row.get("evidence", []) or []
        if not evidence:
            lines.append("- Evidence: (none stored)")
        else:
            lines.append("- Evidence:")
            for ev in evidence[: args.max_passages]:
                lines.append(f"  - Title: {ev.get('title','')}")
                lines.append(f"    NLI: entail={ev.get('nli',{}).get('entail',0):.2f} | contradict={ev.get('nli',{}).get('contradict',0):.2f}")
                snippet = (ev.get("text","") or "").replace("\n"," ")[:300]
                lines.append(f"    Snippet: {snippet}")
        lines.append("")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
