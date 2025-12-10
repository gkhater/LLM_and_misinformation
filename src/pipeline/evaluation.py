from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import yaml
from tqdm import tqdm
from transformers import pipeline
import numpy as np

from src.retrieval import build_retriever, ContextOnlyRetrieval, LocalBM25Retrieval
from src.utils.timing import utc_now_iso


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _tokenize(text: str) -> list[str]:
    import re

    return [t.lower() for t in re.findall(r"[A-Za-z0-9']+", text or "")]


def _overlap_score(claim: str, passage: str) -> float:
    c = set(_tokenize(claim))
    p = set(_tokenize(passage))
    if not c or not p:
        return 0.0
    inter = len(c & p)
    return inter / len(c)


def _extract_keywords(claim: str) -> set[str]:
    toks = _tokenize(claim)
    return {t for t in toks if len(t) >= 4}


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
        refine_top_k: int,
        batch_size: int,
    ):
        self.fast_model_name = fast_model
        self.refine_model_name = refine_model
        self.refine_threshold = refine_threshold
        self.refine_top_k = refine_top_k
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
            scored_candidates = []
            for idx, scores in enumerate(fast_outputs):
                max_edge = max(scores["entail"], scores["contradict"])
                if max_edge >= self.refine_threshold:
                    scored_candidates.append((max_edge, idx))
            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                top_idxs = [idx for _, idx in scored_candidates[: self.refine_top_k]]
                ref_pairs = [pairs[i] for i in top_idxs]
                refined: List[Dict[str, float]] = []
                for batch in _batched(ref_pairs, self.batch_size):
                    res = _run(refine_model, batch)
                    refined.extend([_scores_to_dict(r) for r in res])
                for loc, scores in zip(top_idxs, refined):
                    fast_outputs[loc] = scores

        return fast_outputs


def _gather_evidence(retriever, claim_text: str, context: Optional[str], top_k: int) -> List[tuple]:
    if hasattr(retriever, "fetch_with_ids"):
        return retriever.fetch_with_ids(claim_text, context, top_k=top_k)
    # Fallback: fetch plain text and synthesize ids.
    docs = retriever.fetch(claim_text, context, top_k=top_k)
    return [(f"doc_{i}", t) for i, t in enumerate(docs)]


def _post_filter_hits(
    claim_text: str,
    hits: List[tuple],
    min_overlap: float,
    max_hits: int,
    require_keyword: bool,
    numeric_time_gate: bool,
) -> List[tuple]:
    if not hits:
        return []
    keywords = _extract_keywords(claim_text)
    numeric_tokens = {"year", "years", "month", "months", "week", "weeks", "day", "days", "double", "increase", "decrease", "more", "less", "percent", "percentage"}
    claim_has_number = any(ch.isdigit() for ch in claim_text)
    claim_has_numeric_token = claim_has_number or any(tok in _tokenize(claim_text) for tok in numeric_tokens)
    scored = []
    for h in hits:
        doc_id, text = h
        if require_keyword and keywords and not (set(_tokenize(text)) & keywords):
            continue
        if numeric_time_gate and claim_has_numeric_token:
            # Require passage to contain at least one digit or numeric-ish token.
            if not any(ch.isdigit() for ch in text) and not (set(_tokenize(text)) & numeric_tokens):
                continue
        s = _overlap_score(claim_text, text)
        if s >= min_overlap:
            scored.append((s, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:max_hits]] if scored else []


class Reranker:
    """Embedding-based reranker to keep the most relevant passages."""

    def __init__(self, model_name: str, final_k: int):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.final_k = final_k

    def rerank(self, claim_text: str, hits: List[tuple]) -> List[tuple]:
        if not hits:
            return []
        passages = [t for _, t in hits]
        claim_emb = self.model.encode([claim_text], convert_to_numpy=True)[0]
        pass_embs = self.model.encode(passages, convert_to_numpy=True)
        denom = np.linalg.norm(pass_embs, axis=1) * (np.linalg.norm(claim_emb) or 1e-9)
        sims = np.dot(pass_embs, claim_emb) / np.clip(denom, 1e-9, None)
        ranked = sorted(zip(sims, hits), key=lambda x: x[0], reverse=True)
        return [h for _, h in ranked[: self.final_k]]


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
    post_filter_cfg = retrieval_cfg.get("post_filter", {})
    min_overlap = post_filter_cfg.get("min_overlap", 0.0)
    max_hits = post_filter_cfg.get("max_hits", top_k)
    require_keyword = post_filter_cfg.get("require_keyword", False)
    numeric_time_gate = post_filter_cfg.get("numeric_time_gate", False)
    rerank_cfg = retrieval_cfg.get("rerank", {}) or {}
    reranker = None
    if rerank_cfg.get("enabled", False):
        reranker = Reranker(
            model_name=rerank_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            final_k=rerank_cfg.get("final_k", 3),
        )

    nli_service = NLIService(
        fast_model=nli_cfg["fast_model_name"],
        refine_model=nli_cfg.get("refine_model_name"),
        refine_threshold=nli_cfg.get("refine_threshold", 0.3),
        refine_top_k=nli_cfg.get("refine_top_k", 2),
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
            evidence = _post_filter_hits(
                claim_text,
                evidence,
                min_overlap=min_overlap,
                max_hits=max_hits,
                require_keyword=require_keyword,
                numeric_time_gate=numeric_time_gate,
            )
            if reranker:
                evidence = reranker.rerank(claim_text, evidence)
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
        max_ent = 0.0
        max_con = 0.0
        for ev in cache_entry.get("evidence", []):
            rat_scores = ev.get("nli_rationales", [])
            if idx < len(rat_scores):
                max_ent = max(max_ent, rat_scores[idx].get("entail", 0.0))
                max_con = max(max_con, rat_scores[idx].get("contradict", 0.0))
        if max_ent >= entail_t and max_ent >= max_con:
            supported += 1
        elif max_con >= contr_t and max_con > max_ent:
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


def _claim_verification(cache_entry: dict, entail_t: float, contr_t: float, margin: float) -> dict:
    max_entail = 0.0
    max_contra = 0.0
    per_passage = []
    strong_margin = 0.15
    strong_thresh = 0.8
    for ev in cache_entry.get("evidence", []):
        scores = ev.get("nli_claim", {})
        ent = scores.get("entail", 0.0)
        con = scores.get("contradict", 0.0)
        max_entail = max(max_entail, ent)
        max_contra = max(max_contra, con)
        label = "nei"
        if ent >= entail_t and ent >= con + margin:
            label = "entail"
        elif con >= contr_t and con >= ent + margin:
            label = "contradict"
        per_passage.append(label)

    verdict = "nei"
    # strong single-hit override
    if max_contra >= strong_thresh and max_contra >= max_entail + strong_margin:
        verdict = "contradict"
    elif max_entail >= strong_thresh and max_entail >= max_contra + strong_margin:
        verdict = "entail"
    elif per_passage.count("contradict") >= 2:
        verdict = "contradict"
    elif per_passage.count("entail") >= 2:
        verdict = "entail"
    elif max_entail >= entail_t and max_entail >= max_contra + margin:
        verdict = "entail"
    elif max_contra >= contr_t and max_contra >= max_entail + margin:
        verdict = "contradict"

    return {"verdict": verdict, "evidence_count": len(cache_entry.get("evidence", []))}


def _label_consistency(cache_entry: dict, label_cfg: dict, entail_t: float, contr_t: float, margin: float) -> dict:
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

    max_entail = 0.0
    max_contra = 0.0
    per_passage = []
    strong_margin = 0.15
    strong_thresh = 0.8
    for ev in cache_entry.get("evidence", []):
        scores = ev.get("nli_claim", {})
        ent = scores.get("entail", 0.0)
        con = scores.get("contradict", 0.0)
        max_entail = max(max_entail, ent)
        max_contra = max(max_contra, con)
        label = "nei"
        if ent >= entail_t and ent >= con + margin:
            label = "entail"
        elif con >= contr_t and con >= ent + margin:
            label = "contradict"
        per_passage.append(label)

    has_entail = max_entail >= entail_t
    has_contra = max_contra >= contr_t

    if max_contra >= strong_thresh and max_contra >= max_entail + strong_margin:
        pred = -1
        verdict = "refuted"
    elif max_entail >= strong_thresh and max_entail >= max_contra + strong_margin:
        pred = 1
        verdict = "supported"
    elif per_passage.count("contradict") >= 2:
        pred = -1
        verdict = "refuted"
    elif per_passage.count("entail") >= 2:
        pred = 1
        verdict = "supported"
    elif has_entail and (max_entail >= max_contra + margin):
        pred = 1
        verdict = "supported"
    elif has_contra and (max_contra >= max_entail + margin):
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
    margin = cfg["nli"].get("margin", 0.05)

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
                metrics["claim_verification"] = _claim_verification(cache_entry, entail_t, contr_t, margin)

            if cfg["metrics"].get("label_consistency", {}).get("enabled", False):
                metrics["label_consistency"] = _label_consistency(
                    cache_entry,
                    cfg["metrics"]["label_consistency"],
                    entail_t,
                    contr_t,
                    margin,
                )

            out_record = dict(row)
            out_record["metrics"] = metrics
            out_record["timestamp_eval"] = utc_now_iso()
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    return out_path
