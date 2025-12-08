from __future__ import annotations

import json
import os
from datetime import datetime

from tqdm import tqdm

from src.dataset import load_dataset
from src.utils.timing import utc_now_iso
from src.metrics.fact_precision import FactPrecisionEvaluator
from src.metrics.self_consistency import SelfConsistencyEvaluator
from src.retrieval import build_retriever


def _sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def run_model_on_dataset(config: dict, classifier, max_rows: int | None = None) -> str:
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

    name_parts = [prefix, backend, model_name]
    if split_tag:
        name_parts.append(split_tag)
    name_parts.append(stamp)
    out_path = os.path.join(out_dir, "_".join(name_parts) + ".jsonl")

    id_col = config["dataset"].get("id_column", "id")
    claim_col = config["dataset"].get("claim_column", "claim")
    context_col = config["dataset"].get("context_column", "context")

    metrics_cfg = config.get("metrics", {})

    fact_cfg = metrics_cfg.get("fact_precision", {}) if metrics_cfg else {}
    fact_enabled = fact_cfg.get("enabled", False)
    fact_eval = None
    if fact_enabled:
        retrieval_cfg = fact_cfg.get("retrieval", {})
        retriever_backend = build_retriever(retrieval_cfg)
        retrieval_top_k = retrieval_cfg.get("top_k", 3)

        def fetch_fn(q, c):
            return retriever_backend.fetch(q, c, top_k=retrieval_top_k)

        fact_eval = FactPrecisionEvaluator(
            nli_model_name=fact_cfg.get(
                "nli_model_name", "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
            ),
            max_evidence=fact_cfg.get("max_evidence", 3),
            entail_threshold=fact_cfg.get("entail_threshold", 0.5),
            contradict_threshold=fact_cfg.get("contradict_threshold", 0.5),
            margin=fact_cfg.get("margin", 0.1),
            retriever=fetch_fn,
        )

    sc_cfg = metrics_cfg.get("self_consistency", {}) if metrics_cfg else {}
    sc_enabled = sc_cfg.get("enabled", False)
    sc_eval = (
        SelfConsistencyEvaluator(
            embedding_model=sc_cfg.get("embedding_model"),
            nli_model=sc_cfg.get("nli_model"),
        )
        if sc_enabled
        else None
    )
    sc_samples = sc_cfg.get("samples", 3)
    sc_temp = sc_cfg.get("temperature", config.get("generation", {}).get("temperature", 0.3))

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Querying model"):
            claim = row[claim_col]
            ctx = row[context_col] if context_col and context_col in df.columns else None
            item_id = row[id_col]
            split_val = row[config["dataset"]["split_column"]] if config["dataset"].get("split_column") and config["dataset"]["split_column"] in df.columns else None

            out = classifier.classify(str(claim), ctx if isinstance(ctx, str) else None)

            metrics = {}

            if fact_eval:
                fp = fact_eval.evaluate(out.get("raw_output", ""), ctx if isinstance(ctx, str) else None)
                metrics["fact_precision"] = {
                    "fact_precision": fp.fact_precision,
                    "supported": fp.supported,
                    "refuted": fp.refuted,
                    "nei": fp.nei,
                    "unsupported": fp.unsupported,
                    "refute_rate": fp.refute_rate,
                    "coverage": fp.coverage,
                }

            if sc_eval:
                generations = [out.get("raw_output", "")]
                for _ in range(max(0, sc_samples - 1)):
                    alt = classifier.classify(
                        str(claim),
                        ctx if isinstance(ctx, str) else None,
                        temperature=sc_temp,
                    )
                    generations.append(alt.get("raw_output", ""))
                sc_res = sc_eval.evaluate(generations)
                metrics["self_consistency"] = {
                    "consistency": sc_res.consistency,
                    "risk": sc_res.risk,
                    "avg_similarity": sc_res.avg_similarity,
                    "contradiction_rate": sc_res.contradiction_rate,
                }

            record = {
                "id": int(item_id),
                "split": split_val,
                "claim": claim,
                "context": ctx,
                "model_output": out,
                "metrics": metrics,
                "timestamp": utc_now_iso(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path
