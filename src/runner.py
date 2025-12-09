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
    label_col = config["dataset"].get("label_column")

    metrics_cfg = config.get("metrics", {})

    fact_cfg = metrics_cfg.get("fact_precision", {}) if metrics_cfg else {}
    fact_enabled = fact_cfg.get("enabled", False)
    claim_verif_cfg = metrics_cfg.get("claim_verification", {}) if metrics_cfg else {}
    claim_verif_enabled = claim_verif_cfg.get("enabled", False)
    label_consistency_cfg = metrics_cfg.get("label_consistency", {}) if metrics_cfg else {}
    label_consistency_enabled = label_consistency_cfg.get("enabled", False)

    # Build retrievers for fact precision and claim verification (can share configs).
    fact_retrieval_cfg = fact_cfg.get("retrieval", {}) if fact_cfg else {}
    fact_retriever = build_retriever(fact_retrieval_cfg)
    fact_retrieval_top_k = fact_retrieval_cfg.get("top_k", 3)

    claim_retrieval_cfg = claim_verif_cfg.get("retrieval", {}) if claim_verif_cfg else {}
    if not claim_retrieval_cfg:
        claim_retrieval_cfg = fact_retrieval_cfg
    claim_retriever = build_retriever(claim_retrieval_cfg)
    claim_retrieval_top_k = claim_retrieval_cfg.get("top_k", fact_retrieval_top_k)

    def fact_fetch(q, c):
        return fact_retriever.fetch(q, c, top_k=fact_retrieval_top_k)

    def claim_fetch(q, c):
        return claim_retriever.fetch(q, c, top_k=claim_retrieval_top_k)

    fact_eval = None
    if fact_enabled:
        query_source = fact_retrieval_cfg.get("query_source", "rationale")

        fact_eval = FactPrecisionEvaluator(
            nli_model_name=fact_cfg.get(
                "nli_model_name", "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
            ),
            max_evidence=fact_cfg.get("max_evidence", 3),
            entail_threshold=fact_cfg.get("entail_threshold", 0.5),
            contradict_threshold=fact_cfg.get("contradict_threshold", 0.5),
            margin=fact_cfg.get("margin", 0.1),
            retriever=fact_fetch,
            query_source=query_source,
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
            label_val = row[label_col] if label_col and label_col in df.columns else None

            out = classifier.classify(str(claim), ctx if isinstance(ctx, str) else None)

            metrics = {}

            if fact_eval:
                fp = fact_eval.evaluate(
                    out.get("raw_output", ""),
                    ctx if isinstance(ctx, str) else None,
                    claim_text=str(claim),
                )
                metrics["fact_precision"] = {
                    "fact_precision": fp.fact_precision,
                    "supported": fp.supported,
                    "refuted": fp.refuted,
                    "nei": fp.nei,
                    "unsupported": fp.unsupported,
                    "refute_rate": fp.refute_rate,
                    "coverage": fp.coverage,
                }

            if claim_verif_enabled:
                # Lightweight dataset-claim verification (claim vs evidence, independent of LLM rationale).
                nli_model_name = claim_verif_cfg.get(
                    "nli_model_name",
                    fact_cfg.get("nli_model_name", "ynie/roberta-large-snli_mnli_fever_anli_R1"),
                )
                entail_t = claim_verif_cfg.get(
                    "entail_threshold", fact_cfg.get("entail_threshold", 0.5)
                )
                contradict_t = claim_verif_cfg.get(
                    "contradict_threshold", fact_cfg.get("contradict_threshold", 0.5)
                )
                max_ev = claim_verif_cfg.get("max_evidence", fact_cfg.get("max_evidence", 3))

                # Separate evaluator for claim verification to allow distinct retrieval.
                labeler = FactPrecisionEvaluator(
                    nli_model_name=nli_model_name,
                    max_evidence=max_ev,
                    entail_threshold=entail_t,
                    contradict_threshold=contradict_t,
                    retriever=claim_fetch,
                    query_source="claim",
                )

                evidence_candidates = list(
                    claim_fetch(str(claim), ctx if isinstance(ctx, str) else None)
                )[:max_ev]
                verdict_for_claim = "unsupported" if not evidence_candidates else "nei"
                for ev in evidence_candidates:
                    v = labeler._label(str(claim), ev)  # uses cached model if available
                    if v == "supported":
                        verdict_for_claim = "supported"
                        break
                    if v == "refuted":
                        verdict_for_claim = "refuted"
                metrics["claim_verification"] = {
                    "verdict": verdict_for_claim,
                    "evidence_count": len(evidence_candidates),
                }

            # Label-aware consistency metric: uses dataset label if present.
            if label_consistency_enabled and label_val is not None and isinstance(label_val, str):
                lc_true = {l.lower() for l in label_consistency_cfg.get("true_labels", [])}
                lc_false = {l.lower() for l in label_consistency_cfg.get("false_labels", [])}
                lc_mixed = {l.lower() for l in label_consistency_cfg.get("mixed_labels", [])}
                ent_t = label_consistency_cfg.get(
                    "entail_threshold", fact_cfg.get("entail_threshold", 0.4)
                )
                contra_t = label_consistency_cfg.get(
                    "contradict_threshold", fact_cfg.get("contradict_threshold", 0.4)
                )
                max_ev = label_consistency_cfg.get("max_evidence", fact_cfg.get("max_evidence", 5))

                lv = label_val.lower().strip()
                if lv in lc_true:
                    target_polarity = 1
                elif lv in lc_false:
                    target_polarity = -1
                elif lv in lc_mixed:
                    target_polarity = 0
                else:
                    target_polarity = 0

                lc_eval = FactPrecisionEvaluator(
                    nli_model_name=label_consistency_cfg.get(
                        "nli_model_name",
                        fact_cfg.get("nli_model_name", "MoritzLaurer/deberta-v3-base-mnli-fever-anli"),
                    ),
                    max_evidence=max_ev,
                    entail_threshold=ent_t,
                    contradict_threshold=contra_t,
                    retriever=claim_fetch,
                    query_source="claim",
                )

                evidence_candidates = list(
                    claim_fetch(str(claim), ctx if isinstance(ctx, str) else None)
                )[:max_ev]
                has_entail = False
                has_contra = False
                for ev in evidence_candidates:
                    v = lc_eval._label(str(claim), ev)
                    if v == "supported":
                        has_entail = True
                    if v == "refuted":
                        has_contra = True

                if has_entail and not has_contra:
                    predicted_polarity = 1
                    lc_verdict = "entailed"
                elif has_contra and not has_entail:
                    predicted_polarity = -1
                    lc_verdict = "contradicted"
                elif has_entail and has_contra:
                    predicted_polarity = 0
                    lc_verdict = "mixed"
                else:
                    predicted_polarity = 0
                    lc_verdict = "nei"

                success = target_polarity != 0 and predicted_polarity == target_polarity
                coverage = has_entail or has_contra

                metrics["label_consistency"] = {
                    "label": lv,
                    "target_polarity": target_polarity,
                    "predicted_polarity": predicted_polarity,
                    "has_entail": has_entail,
                    "has_contradiction": has_contra,
                    "coverage": coverage,
                    "success": success,
                    "verdict": lc_verdict,
                    "evidence_count": len(evidence_candidates),
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
