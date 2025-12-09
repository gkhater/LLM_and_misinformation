from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional

from tqdm import tqdm

from src.dataset import load_dataset
from src.utils.timing import utc_now_iso


def _sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter for rationale storage."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def run_generation(config: dict, classifier, max_rows: Optional[int] = None) -> str:
    """Run model inference only; no retrieval or metrics. Writes JSONL."""
    df = load_dataset(config)
    if max_rows:
        df = df.head(max_rows)

    out_dir = config["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    prefix = config["output"]["filename_prefix"]
    model_name = _sanitize_filename(classifier.model_name.replace("/", "_"))
    backend = classifier.backend
    split_tag = config["dataset"].get("split_filter")
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    name_parts = ["gen", prefix, backend, model_name]
    if split_tag:
        name_parts.append(split_tag)
    name_parts.append(stamp)
    out_path = os.path.join(out_dir, "_".join(name_parts) + ".jsonl")

    id_col = config["dataset"].get("id_column", "id")
    claim_col = config["dataset"].get("claim_column", "claim")
    context_col = config["dataset"].get("context_column", "context")
    label_col = config["dataset"].get("label_column")
    split_col = config["dataset"].get("split_column")

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
            claim = row[claim_col]
            ctx = row[context_col] if context_col and context_col in df.columns else None
            item_id = row[id_col]
            split_val = row[split_col] if split_col and split_col in df.columns else None
            label_val = row[label_col] if label_col and label_col in df.columns else None

            out = classifier.classify(str(claim), ctx if isinstance(ctx, str) else None)
            rationale_text = out.get("rationale", "") or out.get("raw_output", "")
            rationale_sents = _split_sentences(rationale_text)

            record: Dict[str, object] = {
                "id": int(item_id),
                "split": split_val,
                "claim": claim,
                "context": ctx,
                "label": label_val,
                "model_output": out,
                "rationale_sentences": rationale_sents,
                "timestamp": utc_now_iso(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path

