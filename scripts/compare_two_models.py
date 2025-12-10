"""
Compare two eval JSONL files (e.g., 8B vs 70B) and emit a markdown summary.

Reports:
- Coverage (non-NEI)
- Verdict distribution
- LC accuracy (if present)
- Binary accuracy (true-ish vs false-ish only; skips mixed/unknown) with n_used
- Examples where model B succeeds on LC and model A fails
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple


TRUEISH = {"true", "mostly-true", "mostly true", "half-true", "half true"}
FALSEISH = {"false", "pants-fire", "pants on fire", "pants-fire!", "pants on-fire", "barely-true", "barely true"}


def load_eval(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize(rows: List[dict]) -> dict:
    verdicts = Counter()
    lc_success = 0
    lc_total = 0
    ev_counts: List[int] = []
    binary_used = 0
    binary_correct = 0
    for row in rows:
        cv = (row.get("metrics") or {}).get("claim_verification", {}) or {}
        verdicts[cv.get("verdict")] += 1
        ev_counts.append(cv.get("evidence_count", 0))
        lc = (row.get("metrics", {}) or {}).get("label_consistency", {}) or {}
        if "success" in lc:
            lc_total += 1
            if lc.get("success"):
                lc_success += 1
        # Binary true/false only; skip mixed/unknown labels.
        label = (row.get("label") or "").lower()
        label_pol = 1 if label in TRUEISH else (-1 if label in FALSEISH else None)
        verdict_pol = None
        if cv.get("verdict") == "entail":
            verdict_pol = 1
        elif cv.get("verdict") == "contradict":
            verdict_pol = -1
        if label_pol is not None and verdict_pol is not None:
            binary_used += 1
            if label_pol == verdict_pol:
                binary_correct += 1
    total = sum(verdicts.values()) or 1
    return {
        "verdicts": verdicts,
        "coverage": 1 - verdicts.get("nei", 0) / total,
        "lc_acc": (lc_success / lc_total) if lc_total else None,
        "avg_ev": sum(ev_counts) / len(ev_counts) if ev_counts else 0.0,
        "binary_acc": (binary_correct / binary_used) if binary_used else None,
        "binary_used": binary_used,
    }


def pick_examples(rows_a: List[dict], rows_b: List[dict], k: int = 5) -> List[Tuple[dict, dict]]:
    """Return pairs where model B LC-success and model A not."""
    pairs = []
    idx = {int(r["id"]): r for r in rows_a if "id" in r}
    for rb in rows_b:
        cid = int(rb.get("id")) if "id" in rb else None
        if cid is None or cid not in idx:
            continue
        ra = idx[cid]
        lca = (ra.get("metrics", {}).get("label_consistency", {}) or {}).get("success")
        lcb = (rb.get("metrics", {}).get("label_consistency", {}) or {}).get("success")
        if lcb and not lca:
            pairs.append((ra, rb))
    return pairs[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval8", required=True, help="Eval JSONL for model A (e.g., 8B)")
    ap.add_argument("--eval70", required=True, help="Eval JSONL for model B (e.g., 70B)")
    ap.add_argument("--out", required=True, help="Output markdown report")
    ap.add_argument("--max-examples", type=int, default=5)
    args = ap.parse_args()

    rows_a = load_eval(Path(args.eval8))
    rows_b = load_eval(Path(args.eval70))
    summary_a = summarize(rows_a)
    summary_b = summarize(rows_b)
    examples = pick_examples(rows_a, rows_b, k=args.max_examples)

    def fmt_verdicts(v: Counter) -> str:
        parts = []
        for key in ["entail", "contradict", "nei"]:
            parts.append(f"{key}: {v.get(key,0)}")
        return ", ".join(parts)

    lines = []
    lines.append(f"# Comparison\n")
    lines.append(f"eval8: `{args.eval8}`")
    lines.append(f"eval70: `{args.eval70}`\n")
    lines.append(f"- Coverage (non-NEI): 8B={summary_a['coverage']:.2f}, 70B={summary_b['coverage']:.2f}")
    lines.append(f"- LC acc: 8B={summary_a['lc_acc']}, 70B={summary_b['lc_acc']}")
    lines.append(f"- Binary acc (true-ish vs false-ish only): 8B={summary_a['binary_acc']} (n={summary_a['binary_used']}), 70B={summary_b['binary_acc']} (n={summary_b['binary_used']})")
    lines.append(f"- Verdicts 8B: {fmt_verdicts(summary_a['verdicts'])}")
    lines.append(f"- Verdicts 70B: {fmt_verdicts(summary_b['verdicts'])}")
    lines.append(f"- Avg evidence_count: 8B={summary_a['avg_ev']:.2f}, 70B={summary_b['avg_ev']:.2f}\n")

    lines.append("## Cases where 70B LC-success and 8B not\n")
    if not examples:
        lines.append("_None found_\n")
    else:
        for ra, rb in examples:
            claim = rb.get("claim", "")
            cvb = rb.get("metrics", {}).get("claim_verification", {}) or {}
            lines.append(f"- id={rb.get('id')} verdict={cvb.get('verdict')} gap={cvb.get('max_gap'):.2f} :: {claim}")
            ev = rb.get("evidence", []) or []
            if ev:
                top = ev[0]
                lines.append(f"  evidence: {top.get('title','')} | ent={top.get('nli',{}).get('entail',0):.2f} contr={top.get('nli',{}).get('contradict',0):.2f}")
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote comparison report to {args.out}")


if __name__ == "__main__":
    main()
