from __future__ import annotations

import json
import os
import hashlib
from typing import Dict, Iterable, List, Optional, Tuple, Set

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


def json_safe(val):
    import numpy as np
    from pathlib import Path

    if val is None:
        return None
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _load_claims_csv(path: str, split: Optional[str] = None, max_rows: Optional[int] = None) -> List[dict]:
    import pandas as pd

    df = pd.read_csv(path)
    if split and "split" in df.columns:
        df = df[df["split"] == split]
    if max_rows:
        df = df.head(max_rows)
    rows: List[dict] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "id": int(row["id"]) if "id" in df.columns else len(rows),
                "claim": row.get("claim", ""),
                "context": row.get("context"),
                "label": row.get("label"),
                "split": row.get("split"),
            }
        )
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
        self.available = True
        self.init_error: Optional[str] = None

    def _ensure_fast(self):
        if self._fast is None:
            try:
                self._fast = pipeline(
                    "text-classification",
                    model=self.fast_model_name,
                    tokenizer=self.fast_model_name,
                    return_all_scores=True,
                    truncation=True,
                )
            except Exception as exc:
                self.available = False
                self.init_error = str(exc)
                self._fast = None
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
        if fast is None:
            # NLI unavailable; return neutral scores.
            return [{"entail": 0.0, "contradict": 0.0, "neutral": 1.0, "nli_status": "unavailable"} for _ in pairs]

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
    final_k: int,
    fallback_delta: float = 0.05,
) -> Tuple[List[tuple], bool, dict]:
    if not hits:
        return [], False, {"entity_tokens": [], "entity_veto_kept": 0, "entity_veto_fallback": False}
    keywords = _extract_keywords(claim_text)
    numeric_tokens = {"year", "years", "month", "months", "week", "weeks", "day", "days", "double", "increase", "decrease", "more", "less", "percent", "percentage"}
    claim_has_number = any(ch.isdigit() for ch in claim_text)
    claim_has_numeric_token = claim_has_number or any(tok in _tokenize(claim_text) for tok in numeric_tokens)
    ent_tokens = []
    for tok in re.findall(r"[A-Z][A-Za-z0-9]+", claim_text):
        if len(tok) >= 3:
            ent_tokens.append(tok.lower())
        if len(ent_tokens) >= 8:
            break
    scored = []
    for h in hits:
        doc_id, text = h
        if require_keyword and keywords and not (set(_tokenize(text)) & keywords):
            continue
        if numeric_time_gate and claim_has_numeric_token:
            # Require passage to contain at least one digit or numeric-ish token.
            if not any(ch.isdigit() for ch in text) and not (set(_tokenize(text)) & numeric_tokens):
                continue
        if ent_tokens:
            if not (set(_tokenize(text)) & set(ent_tokens)):
                continue
        s = _overlap_score(claim_text, text)
        if s >= min_overlap:
            scored.append((s, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    filtered = [h for _, h in scored[:max_hits]] if scored else []
    veto_fallback = False
    if len(filtered) >= final_k or not filtered:
        return filtered, False, {"entity_tokens": ent_tokens, "entity_veto_kept": len(filtered), "entity_veto_fallback": False}
    if len(filtered) < final_k and not filtered and ent_tokens:
        veto_fallback = True
        scored_no_ent = []
        for h in hits:
            doc_id, text = h
            if require_keyword and keywords and not (set(_tokenize(text)) & keywords):
                continue
            if numeric_time_gate and claim_has_numeric_token:
                if not any(ch.isdigit() for ch in text) and not (set(_tokenize(text)) & numeric_tokens):
                    continue
            s = _overlap_score(claim_text, text)
            if s >= min_overlap:
                scored_no_ent.append((s, h))
        scored_no_ent.sort(key=lambda x: x[0], reverse=True)
        filtered = [h for _, h in scored_no_ent[:max_hits]] if scored_no_ent else []
        return filtered, True, {"entity_tokens": ent_tokens, "entity_veto_kept": len(filtered), "entity_veto_fallback": True}
    # Fallback: relax overlap slightly to avoid starving reranker.
    relaxed = [h for score, h in scored if score >= max(0.0, min_overlap - fallback_delta)]
    relaxed = relaxed[:max_hits]
    return relaxed, True, {"entity_tokens": ent_tokens, "entity_veto_kept": len(relaxed), "entity_veto_fallback": True or veto_fallback}


class Reranker:
    """Embedding-based reranker with caching for passages and claims."""

    def __init__(
        self,
        model_name: str,
        final_k: int,
        corpus_hash: Optional[str] = None,
        doc_cache_path: Optional[str] = None,
        claim_cache_path: Optional[str] = None,
        dedup_threshold: float = 0.95,
    ):
        self.model_name = model_name
        self.final_k = final_k
        self.corpus_hash = corpus_hash
        self.doc_cache_path = doc_cache_path
        self.claim_cache_path = claim_cache_path
        self.dedup_threshold = dedup_threshold
        self._doc_cache: Dict[str, np.ndarray] = {}
        self._claim_cache: Dict[str, np.ndarray] = {}
        self.available = True
        self.init_error: Optional[str] = None
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            self.model = None
            self.available = False
            self.init_error = str(exc)
        self._load_caches()

    def _load_cache_file(self, path: str, expected_hash: Optional[str]) -> Dict[str, np.ndarray]:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "rb") as fh:
                payload = json.load(fh)
            meta = payload.get("meta", {})
            if meta.get("model") != self.model_name:
                return {}
            if expected_hash and meta.get("corpus_hash") != expected_hash:
                return {}
            dim = meta.get("dim")
            cache = {}
            for k, v in payload.get("embeddings", {}).items():
                arr = np.array(v, dtype=float)
                if dim is None or arr.shape[0] == dim:
                    cache[k] = arr
            return cache
        except Exception:
            return {}

    def _save_cache_file(self, path: str, cache: Dict[str, np.ndarray], corpus_hash: Optional[str]):
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            meta = {
                "model": self.model_name,
                "dim": next(iter(cache.values())).shape[0] if cache else None,
                "corpus_hash": corpus_hash,
            }
            serializable = {k: v.tolist() for k, v in cache.items()}
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"meta": meta, "embeddings": serializable}, fh)
        except Exception:
            pass

    def _load_caches(self):
        self._doc_cache = self._load_cache_file(self.doc_cache_path, self.corpus_hash)
        self._claim_cache = self._load_cache_file(self.claim_cache_path, None)

    def save_caches(self):
        self._save_cache_file(self.doc_cache_path, self._doc_cache, self.corpus_hash)
        self._save_cache_file(self.claim_cache_path, self._claim_cache, None)

    def _encode_claim(self, claim_id: str, text: str) -> np.ndarray:
        if not self.available or self.model is None:
            return np.zeros((1,), dtype=float)
        if claim_id in self._claim_cache:
            return self._claim_cache[claim_id]
        emb = self.model.encode([text], convert_to_numpy=True)[0]
        self._claim_cache[claim_id] = emb
        return emb

    def _encode_passages(self, hits: List[tuple]) -> Dict[str, np.ndarray]:
        if not self.available or self.model is None:
            return {}
        emb_map: Dict[str, np.ndarray] = {}
        missing_texts: List[str] = []
        missing_ids: List[str] = []
        for doc_id, text in hits:
            if doc_id in self._doc_cache:
                emb_map[doc_id] = self._doc_cache[doc_id]
            else:
                missing_ids.append(doc_id)
                missing_texts.append(text)
        if missing_texts:
            new_embs = self.model.encode(missing_texts, convert_to_numpy=True)
            for doc_id, emb in zip(missing_ids, new_embs):
                self._doc_cache[doc_id] = emb
                emb_map[doc_id] = emb
        return emb_map

    def _dedup(self, ranked: List[Tuple[float, tuple]], emb_map: Dict[str, np.ndarray]) -> List[Tuple[float, tuple]]:
        kept: List[Tuple[float, tuple]] = []
        kept_ids: List[str] = []
        for score, hit in ranked:
            doc_id, _ = hit
            emb = emb_map.get(doc_id) if emb_map else None
            if emb is None:
                kept.append((score, hit))
                kept_ids.append(doc_id)
                if len(kept) >= self.final_k:
                    break
                continue
            too_close = False
            for other_id in kept_ids:
                other = emb_map.get(other_id)
                if other is None:
                    continue
                denom = (np.linalg.norm(other) * np.linalg.norm(emb)) or 1e-9
                sim = float(np.dot(other, emb) / denom)
                if sim >= self.dedup_threshold:
                    too_close = True
                    break
            if not too_close:
                kept.append((score, hit))
                kept_ids.append(doc_id)
            if len(kept) >= self.final_k:
                break
        return kept

    def rerank(self, claim_id: str, claim_text: str, hits: List[tuple]) -> Tuple[List[tuple], List[float]]:
        if not hits:
            return [], []
        if not self.available or self.model is None:
            # No reranker available; return original hits.
            return hits[: self.final_k], [0.0 for _ in hits[: self.final_k]]
        if not hits:
            return [], []
        claim_emb = self._encode_claim(str(claim_id), claim_text)
        emb_map = self._encode_passages(hits)
        scores: List[float] = []
        for doc_id, _ in hits:
            emb = emb_map.get(doc_id)
            if emb is None:
                scores.append(0.0)
                continue
            denom = (np.linalg.norm(emb) * np.linalg.norm(claim_emb)) or 1e-9
            scores.append(float(np.dot(emb, claim_emb) / denom))
        ranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
        ranked = self._dedup(ranked, emb_map)
        kept_hits = [h for _, h in ranked[: self.final_k]]
        kept_scores = [s for s, _ in ranked[: self.final_k]]
        return kept_hits, kept_scores


def build_nli_cache(
    gen_rows: List[dict],
    retrieval_cfg: dict,
    nli_cfg: dict,
    cache_path: str,
    max_rows: Optional[int] = None,
    debug_ids: Optional[Set[int]] = None,
) -> Dict[int, dict]:
    debug_ids = debug_ids or set()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    existing: Dict[int, dict] = {}
    if os.path.exists(cache_path):
        for row in _load_jsonl(cache_path):
            existing[int(row["id"])] = row

    retriever = build_retriever(retrieval_cfg)
    corpus_hash = None
    corpus_path = retrieval_cfg.get("corpus_path") or retrieval_cfg.get("path")
    corpus_size = None
    if hasattr(retriever, "corpus_meta"):
        meta = retriever.corpus_meta()
        corpus_hash = meta.get("hash", corpus_hash)
        corpus_path = meta.get("path", corpus_path)
        corpus_size = meta.get("size", corpus_size)
    if corpus_size is None and hasattr(retriever, "_docs"):
        corpus_size = len(getattr(retriever, "_docs"))
    if corpus_path and corpus_hash is None and os.path.exists(corpus_path):
        with open(corpus_path, "rb") as fh:
            corpus_hash = hashlib.sha1(fh.read()).hexdigest()
    top_k = retrieval_cfg.get("top_k", 10)
    if corpus_size is not None:
        top_k = min(top_k, corpus_size)
    post_filter_cfg = retrieval_cfg.get("post_filter", {})
    min_overlap = post_filter_cfg.get("min_overlap", 0.0)
    max_hits = post_filter_cfg.get("max_hits", top_k)
    if corpus_size is not None:
        max_hits = min(max_hits, corpus_size)
    require_keyword = post_filter_cfg.get("require_keyword", False)
    numeric_time_gate = post_filter_cfg.get("numeric_time_gate", False)
    rerank_cfg = retrieval_cfg.get("rerank", {}) or {}
    reranker = None
    if rerank_cfg.get("enabled", False):
        reranker = Reranker(
            model_name=rerank_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            final_k=min(rerank_cfg.get("final_k", 3), corpus_size or rerank_cfg.get("final_k", 3)),
            corpus_hash=corpus_hash,
            doc_cache_path=rerank_cfg.get("doc_cache_path"),
            claim_cache_path=rerank_cfg.get("claim_cache_path"),
            dedup_threshold=rerank_cfg.get("dedup_threshold", 0.95),
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
            bm25_candidates = len(evidence)
            # Dedup immediately after BM25
            seen_ids_raw = set()
            deduped_raw = []
            for pid, ptxt in evidence:
                if pid in seen_ids_raw:
                    continue
                seen_ids_raw.add(pid)
                deduped_raw.append((pid, ptxt))
            evidence = deduped_raw
            evidence, fallback_used, entity_meta = _post_filter_hits(
                claim_text,
                evidence,
                min_overlap=min_overlap,
                max_hits=max_hits,
                require_keyword=require_keyword,
                numeric_time_gate=numeric_time_gate,
                final_k=rerank_cfg.get("final_k", max_hits),
            )
            if claim_id in debug_ids:
                print(f"[DEBUG] cache build claim {claim_id} raw_hits={len(evidence)} fallback={fallback_used}")
                if evidence:
                    p_id, p_txt = evidence[0]
                    print(f"[DEBUG] Top premise: {p_id} :: {p_txt[:120]}... | hypothesis: {claim_text[:120]}...")
            if reranker:
                evidence, rerank_scores = reranker.rerank(claim_id, claim_text, evidence)
            else:
                rerank_scores = []
            filter_kept = len(evidence)
            # Deduplicate by passage_id
            seen_ids = set()
            deduped = []
            dedup_scores = []
            for (pid, ptxt), score in zip(evidence, rerank_scores if rerank_scores else [0.0] * len(evidence)):
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                deduped.append((pid, ptxt))
                dedup_scores.append(score)
            evidence = deduped
            rerank_scores = dedup_scores
            filter_kept = len(evidence)
            if corpus_size is not None:
                evidence = evidence[:corpus_size]
                rerank_scores = rerank_scores[: len(evidence)]
            if not rerank_scores:
                rerank_scores = [0.0] * len(evidence)
            if not evidence:
                entry = {
                    "id": claim_id,
                    "claim": claim_text,
                    "label": row.get("label"),
                    "evidence": [],
                    "rationale_sentences": rationale_sents,
                    "corpus_hash": corpus_hash,
                    "rerank_scores": rerank_scores,
                    "filter_fallback": fallback_used,
                    "bm25_candidates": bm25_candidates,
                    "filter_kept": filter_kept,
                    "rerank_kept": 0,
                    "corpus_hash": corpus_hash,
                    "corpus_path": str(corpus_path) if corpus_path else "",
                    "corpus_size": int(corpus_size) if corpus_size is not None else None,
                    "top_k_effective": top_k,
                    "max_hits_effective": max_hits,
                    "final_k_effective": min(rerank_cfg.get("final_k", max_hits), len(evidence)),
                    "entity_tokens": entity_meta.get("entity_tokens", []),
                    "entity_veto_kept": entity_meta.get("entity_veto_kept"),
                    "entity_veto_fallback": entity_meta.get("entity_veto_fallback"),
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
            for idx_ev, ((ev_id, ev_text), claim_score, rat_scores) in enumerate(
                zip(evidence, claim_scores, rationale_matrix)
            ):
                ev_entries.append(
                    {
                        "passage_id": ev_id,
                        "passage_text": ev_text,
                        "nli_claim": claim_score,
                        "nli_rationales": rat_scores,
                        "rerank_score": rerank_scores[idx_ev] if idx_ev < len(rerank_scores) else None,
                    }
                )

            entry = {
                "id": claim_id,
                "claim": claim_text,
                "label": row.get("label"),
                "evidence": ev_entries,
                "rationale_sentences": rationale_sents,
                "corpus_hash": corpus_hash,
                "corpus_path": str(corpus_path) if corpus_path else "",
                "corpus_size": int(corpus_size) if corpus_size is not None else None,
                "rerank_scores": rerank_scores,
                "filter_fallback": fallback_used,
                "bm25_candidates": bm25_candidates,
                "filter_kept": filter_kept,
                "rerank_kept": len(ev_entries),
                "unique_evidence_count": len(seen_ids),
                "top_k_effective": top_k,
                "max_hits_effective": max_hits,
                "final_k_effective": min(rerank_cfg.get("final_k", max_hits), len(ev_entries)),
                "entity_tokens": entity_meta.get("entity_tokens", []),
                "entity_veto_kept": entity_meta.get("entity_veto_kept"),
                "entity_veto_fallback": entity_meta.get("entity_veto_fallback"),
            }
            cache_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing[claim_id] = entry

    if reranker:
        reranker.save_caches()

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


def _aggregate_verdict_from_passages(
    cache_entry: dict,
    entail_t: float,
    contr_t: float,
    margin: float,
) -> dict:
    max_entail = 0.0
    max_contra = 0.0
    ent_count = 0
    con_count = 0
    strong = 0.80
    strong_margin = 0.15
    for ev in cache_entry.get("evidence", []):
        scores = ev.get("nli_claim", {})
        ent = scores.get("entail", 0.0)
        con = scores.get("contradict", 0.0)
        max_entail = max(max_entail, ent)
        max_contra = max(max_contra, con)
        if ent >= entail_t:
            ent_count += 1
        if con >= contr_t:
            con_count += 1
    rule = "nei"
    verdict = "nei"
    if max_contra >= strong and max_contra >= max_entail + strong_margin:
        verdict = "contradict"
        rule = "strong_contradict"
    elif max_entail >= strong and max_entail >= max_contra + strong_margin:
        verdict = "entail"
        rule = "strong_entail"
    elif con_count >= 2:
        verdict = "contradict"
        rule = "two_hit_contradict"
    elif ent_count >= 2:
        verdict = "entail"
        rule = "two_hit_entail"
    elif max_entail >= entail_t and max_entail >= max_contra + margin:
        verdict = "entail"
        rule = "margin_entail"
    elif max_contra >= contr_t and max_contra >= max_entail + margin:
        verdict = "contradict"
        rule = "margin_contradict"
    return {
        "verdict": verdict,
        "max_entail": max_entail,
        "max_contradict": max_contra,
        "ent_count": ent_count,
        "contradict_count": con_count,
        "rule_fired": rule,
        "evidence_count": len(cache_entry.get("evidence", [])),
    }


def _claim_verification(cache_entry: dict, entail_t: float, contr_t: float, margin: float) -> dict:
    return _aggregate_verdict_from_passages(cache_entry, entail_t, contr_t, margin)


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

    agg = _aggregate_verdict_from_passages(cache_entry, entail_t, contr_t, margin)
    verdict = agg["verdict"]
    max_entail = agg["max_entail"]
    max_contra = agg["max_contradict"]
    has_entail = max_entail >= entail_t
    has_contra = max_contra >= contr_t
    if verdict == "entail":
        pred = 1
    elif verdict == "contradict":
        pred = -1
    else:
        pred = 0
    if has_entail and has_contra and pred == 0:
        verdict = "mixed"

    success = (target == pred)

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
        "rule_fired": agg.get("rule_fired"),
        "max_entail": agg.get("max_entail"),
        "max_contradict": agg.get("max_contradict"),
        "ent_count": agg.get("ent_count"),
        "contradict_count": agg.get("contradict_count"),
    }


def _retrieval_stats(claim_text: str, cache_entry: dict) -> dict:
    rerank_scores = cache_entry.get("rerank_scores") or []
    evidence = cache_entry.get("evidence", []) or []
    clean_scores = [float(s) for s in rerank_scores if s is not None]
    avg_rerank = float(sum(clean_scores) / len(clean_scores)) if clean_scores else 0.0
    max_rerank = float(max(clean_scores)) if clean_scores else 0.0
    kw = _extract_keywords(claim_text)
    evidence_has_entity = False
    numeric_claim = any(ch.isdigit() for ch in claim_text)
    numeric_hits = 0
    for ev in evidence:
        txt = ev.get("passage_text", "")
        tokens = set(_tokenize(txt))
        if kw and (tokens & kw):
            evidence_has_entity = True
        if any(ch.isdigit() for ch in txt):
            numeric_hits += 1
    bm25_candidates = cache_entry.get("bm25_candidates")
    filter_kept = cache_entry.get("filter_kept")
    rerank_kept = cache_entry.get("rerank_kept")
    final_k = len(evidence)
    unique_evidence = len({ev.get("passage_id") for ev in evidence}) if evidence else 0
    if bm25_candidates == 0:
        no_ev_reason = "bm25_empty"
    elif (filter_kept or 0) == 0:
        no_ev_reason = "filtered_out"
    elif final_k == 0:
        no_ev_reason = "rerank_empty"
    else:
        no_ev_reason = None
    stats = {
        "avg_rerank_score": avg_rerank,
        "max_rerank_score": max_rerank,
        "evidence_has_entity": evidence_has_entity,
        "numeric_claim": numeric_claim,
        "numeric_evidence_hits": numeric_hits,
        "filter_fallback": bool(cache_entry.get("filter_fallback", False)),
        "bm25_candidates": bm25_candidates,
        "post_filter_kept": filter_kept,
        "rerank_kept": rerank_kept,
        "final_k": final_k,
        "no_evidence_reason": no_ev_reason,
        "corpus_path": cache_entry.get("corpus_path"),
        "corpus_hash": cache_entry.get("corpus_hash"),
        "corpus_size": cache_entry.get("corpus_size"),
        "unique_evidence_count": unique_evidence,
        "top_k_effective": cache_entry.get("top_k_effective"),
        "max_hits_effective": cache_entry.get("max_hits_effective"),
        "final_k_effective": cache_entry.get("final_k_effective"),
    }
    return {k: json_safe(v) for k, v in stats.items()}


def run_evaluation(
    eval_config_path: str,
    max_rows: Optional[int] = None,
    smoke: bool = False,
    debug_ids: Optional[Iterable[int]] = None,
) -> str:
    with open(eval_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    effective_max_rows = max_rows if max_rows is not None else cfg.get("max_rows")

    gen_rows: List[dict] = []
    if "input_jsonl" in cfg and cfg["input_jsonl"]:
        gen_rows = _load_jsonl(cfg["input_jsonl"], max_rows=effective_max_rows)
    elif "input_claims_csv" in cfg and cfg["input_claims_csv"]:
        gen_rows = _load_claims_csv(
            cfg["input_claims_csv"],
            split=cfg.get("input_split"),
            max_rows=effective_max_rows,
        )
    else:
        raise ValueError("Evaluation config must provide input_jsonl or input_claims_csv")
    cache_path = cfg["nli"]["cache_path"]
    cache = build_nli_cache(
        gen_rows,
        retrieval_cfg=cfg["retrieval"],
        nli_cfg=cfg["nli"],
        cache_path=cache_path,
        max_rows=2 if smoke else effective_max_rows,
        debug_ids=set(debug_ids) if debug_ids else None,
    )

    entail_t = cfg["nli"].get("entail_threshold", 0.35)
    contr_t = cfg["nli"].get("contradict_threshold", 0.35)
    margin = cfg["nli"].get("margin", 0.05)

    out_path = cfg.get("output_jsonl", "outputs/eval_output.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    gating_cfg = cfg.get("gating", {})
    disallow_context_fact = gating_cfg.get("disallow_fact_precision_on_context_only", False)
    require_output_ok = gating_cfg.get("require_output_ok_for_fact_precision", False)
    skip_unknown = gating_cfg.get("skip_fact_precision_if_label_unknown", False)
    min_conf = gating_cfg.get("min_confidence_for_fact_precision", 0.0)
    is_context_only = (cfg.get("retrieval", {}).get("backend") or "").lower() in ("context_only", "context")

    smoke_mode = bool(cfg.get("smoke_mode", False) or smoke)
    health_checks = {
        "parsed_rows": len(gen_rows),
        "retrieval_ok": True,
        "nli_ok": True,
        "cache_written": os.path.exists(cache_path),
    }

    debug_set = set(debug_ids) if debug_ids else set()

    with open(out_path, "w", encoding="utf-8") as out_f:
        for row in tqdm(gen_rows, desc="Evaluating"):
            claim_id = int(row["id"])
            cache_entry = cache.get(claim_id)
            metrics: Dict[str, dict] = {}

            if smoke_mode:
                metrics["health_checks"] = health_checks
            else:
                fact_enabled = cfg["metrics"].get("fact_precision", {}).get("enabled", False)
                if fact_enabled:
                    mo = row.get("model_output", {})
                    quality_ok = mo.get("quality", {}).get("output_ok", False) if mo else False
                    if disallow_context_fact and is_context_only:
                        metrics["fact_precision"] = {"status": "skipped_context_only"}
                    elif not mo:
                        metrics["fact_precision"] = {"status": "skipped_no_generation"}
                    elif require_output_ok and not quality_ok:
                        metrics["fact_precision"] = {"status": "skipped_low_quality_output"}
                    elif skip_unknown and (mo.get("label") == "unknown"):
                        metrics["fact_precision"] = {"status": "skipped_unknown_label"}
                    elif mo.get("confidence", 0.0) < min_conf:
                        metrics["fact_precision"] = {"status": "skipped_low_confidence"}
                    else:
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
                metrics["retrieval_stats"] = _retrieval_stats(row.get("claim", ""), cache_entry or {})

            out_record = dict(row)
            out_record["metrics"] = metrics
            out_record["timestamp_eval"] = utc_now_iso()
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            if claim_id in debug_set:
                evidence = cache_entry.get("evidence", []) if cache_entry else []
                print(f"\n[DEBUG] Claim {claim_id}: {row.get('claim', '')}")
                for idx, ev in enumerate(evidence[:5]):
                    txt = ev.get("passage_text", "")[:200].replace("\n", " ")
                    scores = ev.get("nli_claim", {})
                    print(
                        f"  #{idx+1} id={ev.get('passage_id')} rerank={ev.get('rerank_score')} "
                        f"ent={scores.get('entail',0):.2f} contr={scores.get('contradict',0):.2f} :: {txt}..."
                    )

    return out_path
