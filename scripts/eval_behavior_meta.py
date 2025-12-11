from __future__ import annotations

import argparse
import json
from typing import Dict, List

from src.metrics.behavior_meta_nli import BehaviorMetaNLI


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _infer_category(row: dict) -> str:
    """
    Try to recover a coarse category from dataset label, if any.
    This is only for analysis; you can tweak mapping later.
    """
    if "category" in row and row["category"]:
        return str(row["category"]).lower()

    label = str(row.get("label", "")).lower()
    if not label:
        return "unknown"

    if label in {"false", "pants-fire", "pants on fire", "pants-fire!", "pants on-fire"}:
        return "false"
    if label in {"half-true", "barely-true", "half true", "barely true"}:
        return "biased"
    # treat true/mostly-true as neutral for behavior analysis
    if label in {"true", "mostly-true", "mostly true"}:
        return "neutral"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach behavior meta-labels (ACCEPT / CORRECT / HEDGE) using NLI."
    )
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Generation JSONL from run_generation.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Where to write JSONL with added behavior_* fields.",
    )
    parser.add_argument(
        "--nli-model",
        required=True,
        help="HF NLI model id, e.g. typeform/distilbert-base-uncased-mnli",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    clf = BehaviorMetaNLI(model_name=args.nli_model)

    out_rows: List[Dict] = []
    for row in rows:
        claim = row.get("claim", "")
        mo = row.get("model_output") or {}
        answer = (
            mo.get("raw_output")
            or mo.get("rationale")
            or " ".join(row.get("rationale_sentences") or [])
        )
