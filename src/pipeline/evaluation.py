from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from tqdm import tqdm
from transformers import pipeline

from src.retrieval import build_retriever, ContextOnlyRetrieval, LocalBM25Retrieval
from src.utils.timing import utc_now_iso


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _batched(seq: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def _scores_to_dict(raw_scores: list[dict]) -> Dict[str, float]:
    out = {"entail": 0.0, "contradict": 0.0, "neutral": 0.0}
    for s in raw_scores:
        lbl = s.get("label", "").lower()
        if "entail" in lbl:
            out["entail"] = float(s.get("score", 0.0))
        elif "contradict" in lbl:
            out["contradict"] = float(s.get("score", 0.0))
        elif "neutral" in lbl:
            out["neutral"] = float(s.get("score", 0.0))
    return out


def _classify(scores: Dict[str, float], entail_t: float, contr_t: float) -> str:
    ent = scores.get("entail", 0.0)
    con = scores.get("contradict", 0.0)
    if ent >= entail_t and ent >= con:
        return "entail"
    if con >= contr_t and con >= ent:
        return "contradict"
    return "nei"


def _load_jsonl(path: str, max_rows: Optional[int] = None) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows and idx >= max_rows:
                break
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class NLIService:
    """Two-stage NLI (fast filter + optional refiner) with batching."""

    def __init__(
        self,
        fast_model: str,
        refine_model: Optional[str],
        refine_threshold: float,
        batch_size: int,
    ):
        self.fast_model_name = fast_model
        self.refine_model_name = refine_model
        self.refine_threshold = refine_threshold
        self.batch_size = batch_size
        self._fast = None
        self._refine = None

    def _ensure_fast(self):
        if self._fast is None:
            self._fast = pipeline(
                "text-classification",
                model=self.fast_model_name,
                tokenizer=self.fast_model_name,
                return_all_scores=True,
                truncation=True,
            )
        return self._fast

    def _ensure_refine(self):
        if self.refine_model_name and self._refine is None:
            self._refine = pipeline(
                "text-classification",
                model=self.refine_model_name,
                tokenizer=self.refine_model_name,
                return_all_scores=True,
                truncation=True,
            )
        return self._refine

    def infer(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        fast = self._ensure_fast()

        def _run(pipe, batch_pairs):
            payload = [{"text": p, "text_pair": h} for p, h in batch_pairs]
            return pipe(payload, return_all_scores=True)

        # Stage 1: fast model everywhere.
        fast_outputs: List[Dict[str, float]] = []
        for batch in _batched(pairs, self.batch_size):
            res = _run(fast, batch)
            fast_outputs.extend([_scores_to_dict(r) for r in res])

        # Stage 2: optional refinement on high-signal pairs.
        refine_model = self._ensure_refine()
        if refine_model:
            candidates = []
            for idx, scores in enumerate(fast_outputs):
                max_edge = max(scores["entail"], scores["contradict"])
                if max_edge >= self.refine_threshold:
                    candidates.append(idx)
            if candidates:
                ref_pairs = [pairs[i] for i in candidates]
                refined: List[Dict[str, float]] = []
                for batch in _batched(ref_pairs, self.batch_size):
                    res = _run(refine_model, batch)
                    refined.extend([_scores_to_dict(r) for r in res])
                for loc, scores in zip(candidates, refined):
                    fast_outputs[loc] = scores

        return fast_outputs


def _gather_evidence(retriever, claim_text: str, context: Optional[str], top_k: int) -> List[tuple]:
    if hasattr(retriever, "fetch_with_ids"):
        return retriever.fetch_with_ids(claim_text, context, top_k=top_k)
    # Fallback: fetch plain text and synthesize ids.
    docs = retriever.fetch(claim_text, context, top_k=top_k)
    return [(f"doc_{i}", t) for i, t in enumerate(docs)]


def build_nli_cache(
    gen_rows: List[dict],
    retrieval_cfg: dict,
    nli_cfg: dict,
    cache_path: str,
    max_rows: Optional[int] = None,
) -> Dict[int, dict]:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    existing: Dict[int, dict] = {}
    if os.path.exists(cache_path):
        for row in _load_jsonl(cache_path):
            existing[int(row["id"])] = row

    retriever = build_retriever(retrieval_cfg)
    top_k = retrieval_cfg.get("top_k", 10)

    nli_service = NLIService(
        fast_model=nli_cfg["fast_model_name"],
        refine_model=nli_cfg.get("refine_model_name"),
        refine_threshold=nli_cfg.get("refine_threshold", 0.3),
        batch_size=nli_cfg.get("batch_size", 16),
    )

    with open(cache_path, "a", encoding="utf-8") as cache_file:
        for idx, row in enumerate(tqdm(gen_rows, desc="Building NLI cache")):
            if max_rows and idx >= max_rows:
                break
            claim_id = int(row["id"])
            if claim_id in existing:
                continue

            claim_text = str(row.get("claim", ""))
            context = row.get("context")
            rationale_sents = row.get("rationale_sentences") or _split_sentences(
                row.get("model_output", {}).get("rationale", "")
            )

            evidence = _gather_evidence(retriever, claim_text, context, top_k=top_k)
            if not evidence:
                entry = {
                    "id": claim_id,
                    "claim": claim_text,
                    "label": row.get("label"),
                    "evidence": [],
                    "rationale_sentences": rationale_sents,
                }
                cache_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                existing[claim_id] = entry
                continue

            claim_pairs = [(ev_text, claim_text) for _, ev_text in evidence]
            rationale_pairs = []
            for sent in rationale_sents:
                rationale_pairs.extend([(ev_text, sent) for _, ev_text in evidence])

            claim_scores = nli_service.infer(claim_pairs) if claim_pairs else []
            rationale_scores = nli_service.infer(rationale_pairs) if rationale_pairs else []

            # Group rationale scores by evidence index.
            rationale_matrix: List[List[Dict[str, float]]] = []
            if rationale_scores:
                for i in range(len(evidence)):
                    start = i * len(rationale_sents)
                    end = start + len(rationale_sents)
                    rationale_matrix.append(rationale_scores[start:end])
            else:
                rationale_matrix = [[] for _ in evidence]

            ev_entries = []
            for (ev_id, ev_text), claim_score, rat_scores in zip(
                evidence, claim_scores, rationale_matrix
            ):
                ev_entries.append(
                    {
                        "passage_id": ev_id,
                        "passage_text": ev_text,
                        "nli_claim": claim_score,
                        "nli_rationales": rat_scores,
                    }
                )

            entry = {
                "id": claim_id,
                "claim": claim_text,
                "label": row.get("label"),
                "evidence": ev_entries,
                "rationale_sentences": rationale_sents,
            }
            cache_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing[claim_id] = entry

    return existing


def _fact_precision(cache_entry: dict, entail_t: float, contr_t: float) -> dict:
    rat_sents = cache_entry.get("rationale_sentences", []) or []
    if not rat_sents:
        return {
            "fact_precision": 0.0,
            "supported": 0,
            "refuted": 0,
            "nei": 0,
            "unsupported": 0,
            "refute_rate": 0.0,
            "coverage": 0.0,
        }

    supported = refuted = nei = 0
    for idx, _sent in enumerate(rat_sents):
        verdicts = []
        for ev in cache_entry.get("evidence", []):
            rat_scores = ev.get("nli_rationales", [])
            if idx < len(rat_scores):
                verdicts.append(_classify(rat_scores[idx], entail_t, contr_t))
        if "entail" in verdicts:
            supported += 1
        elif "contradict" in verdicts:
            refuted += 1
        else:
            nei += 1

    total = max(1, supported + refuted + nei)
    fact_precision = supported / total
    return {
        "fact_precision": fact_precision,
        "supported": supported,
        "refuted": refuted,
        "nei": nei,
        "unsupported": 0,
        "refute_rate": refuted / total,
        "coverage": (supported + refuted) / total,
    }


def _claim_verification(cache_entry: dict, entail_t: float, contr_t: float) -> dict:
    verdicts = []
    for ev in cache_entry.get("evidence", []):
        verdicts.append(_classify(ev.get("nli_claim", {}), entail_t, contr_t))

    if "entail" in verdicts:
        verdict = "entail"
    elif "contradict" in verdicts:
        verdict = "contradict"
    else:
        verdict = "nei"

    return {"verdict": verdict, "evidence_count": len(verdicts)}


def _label_consistency(cache_entry: dict, label_cfg: dict, entail_t: float, contr_t: float) -> dict:
    label_val = (cache_entry.get("label") or "").lower().strip()
    true_set = {l.lower() for l in label_cfg.get("true_labels", [])}
    false_set = {l.lower() for l in label_cfg.get("false_labels", [])}
    mixed_set = {l.lower() for l in label_cfg.get("mixed_labels", [])}

    if label_val in true_set:
        target = 1
    elif label_val in false_set:
        target = -1
    elif label_val in mixed_set:
        target = 0
    else:
        target = 0

    has_entail = has_contra = False
    for ev in cache_entry.get("evidence", []):
        verdict = _classify(ev.get("nli_claim", {}), entail_t, contr_t)
        if verdict == "entail":
            has_entail = True
        elif verdict == "contradict":
            has_contra = True

    if has_entail and not has_contra:
        pred = 1
        verdict = "supported"
    elif has_contra and not has_entail:
        pred = -1
        verdict = "refuted"
    elif has_entail and has_contra:
        pred = 0
        verdict = "mixed"
    else:
        pred = 0
        verdict = "nei"

    success = (target == pred) if target != 0 else (pred != 0)

    return {
        "label": label_val,
        "target_polarity": target,
        "predicted_polarity": pred,
        "has_entail": has_entail,
        "has_contradiction": has_contra,
        "coverage": has_entail or has_contra,
        "success": success,
        "verdict": verdict,
        "evidence_count": len(cache_entry.get("evidence", [])),
    }


def run_evaluation(eval_config_path: str, max_rows: Optional[int] = None) -> str:
    with open(eval_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gen_rows = _load_jsonl(cfg["input_jsonl"], max_rows=max_rows)
    cache_path = cfg["nli"]["cache_path"]
    cache = build_nli_cache(
        gen_rows,
        retrieval_cfg=cfg["retrieval"],
        nli_cfg=cfg["nli"],
        cache_path=cache_path,
        max_rows=max_rows,
    )

    entail_t = cfg["nli"].get("entail_threshold", 0.35)
    contr_t = cfg["nli"].get("contradict_threshold", 0.35)

    out_path = cfg["output_jsonl"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for row in tqdm(gen_rows, desc="Evaluating"):
            claim_id = int(row["id"])
            cache_entry = cache.get(claim_id)
            metrics: Dict[str, dict] = {}

            if cfg["metrics"].get("fact_precision", {}).get("enabled", False):
                metrics["fact_precision"] = _fact_precision(cache_entry, entail_t, contr_t)

            if cfg["metrics"].get("claim_verification", {}).get("enabled", False):
                metrics["claim_verification"] = _claim_verification(cache_entry, entail_t, contr_t)

            if cfg["metrics"].get("label_consistency", {}).get("enabled", False):
                metrics["label_consistency"] = _label_consistency(
                    cache_entry,
                    cfg["metrics"]["label_consistency"],
                    entail_t,
                    contr_t,
                )

            out_record = dict(row)
            out_record["metrics"] = metrics
            out_record["timestamp_eval"] = utc_now_iso()
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    return out_path
