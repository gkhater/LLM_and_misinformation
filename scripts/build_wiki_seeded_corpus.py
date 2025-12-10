"""
Build a Wikipedia passage corpus seeded by claim text.

Features:
- Incremental/resumable: caches search and page extracts under data/wiki_cache/.
- Manifest records inputs (claims hash, params) for reproducibility.
- Filters passages to keep claim-relevant text (entity/keyword gate).

Usage (from repo root):
  python scripts/build_wiki_seeded_corpus.py --claims data/claims_liar.csv --out data/wiki_seeded_passages.tsv --max-claims-for-corpus 2000 --seed 13
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests


WIKI_API = "https://en.wikipedia.org/w/api.php"

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "in",
    "on",
    "for",
    "of",
    "to",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "it",
    "as",
    "by",
    "at",
    "from",
    "with",
    "about",
    "into",
    "over",
    "after",
    "before",
    "says",
    "say",
    "said",
    "claims",
    "claim",
}

NUMERIC_KEYWORDS = {
    "year",
    "years",
    "month",
    "months",
    "double",
    "percent",
    "percentage",
    "increase",
    "decrease",
    "pace",
    "layoffs",
    "veterans",
    "border",
    "wall",
    "covid",
    "vaccine",
}


@dataclass
class Manifest:
    claims_hash: str
    params: Dict
    timestamp: float
    pages_fetched: int
    corpus_hash: Optional[str] = None
    titles_by_claim: Optional[Dict[str, List[str]]] = None


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_claims(path: Path, max_claims: Optional[int], seed: Optional[int]) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if seed is not None:
        random.Random(seed).shuffle(rows)
    if max_claims:
        rows = rows[:max_claims]
    return rows


def extract_seeds(text: str) -> Set[str]:
    """Cheap heuristic to pull entity-like strings and keywords."""
    seeds: Set[str] = set()
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-']*", text or "")
    # Capitalized n-grams (2-4)
    caps = [t for t in tokens if t[:1].isupper()]
    for n in range(2, 5):
        for i in range(len(caps) - n + 1):
            span = " ".join(caps[i : i + n])
            if len(span.split()) == n:
                seeds.add(span)
    # Keywords length >=5 excluding stopwords
    for t in tokens:
        tl = t.lower()
        if tl in STOPWORDS or len(tl) < 5:
            continue
        seeds.add(tl)
    # Numeric/time/comparative keywords if present
    if any(ch.isdigit() for ch in text):
        seeds.update(NUMERIC_KEYWORDS)
    else:
        seeds.update(NUMERIC_KEYWORDS & set(tokens))
    return seeds


def _http_request(session: requests.Session, params: dict, timeout: float, retries: int, backoff: float) -> requests.Response:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(WIKI_API, params=params, timeout=(3, timeout))
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(backoff * (2 ** (attempt - 1)))
    if last_exc:
        raise last_exc
    raise RuntimeError("HTTP request failed")


def wiki_search(session: requests.Session, query: str, cache_dir: Path, sleep: float, timeout: float, retries: int, backoff: float, search_limit: int) -> List[str]:
    cache_file = cache_dir / f"search_{sha1_bytes(query.encode('utf-8'))}.json"
    if cache_file.exists():
        try:
            titles = json.loads(cache_file.read_text(encoding="utf-8"))
            if titles:
                return titles[:search_limit]
        except Exception:
            pass
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": search_limit,
        "format": "json",
    }
    resp = _http_request(session, params, timeout=timeout, retries=retries, backoff=backoff)
    data = resp.json()
    titles = [hit["title"] for hit in data.get("query", {}).get("search", [])][:search_limit]
    cache_file.write_text(json.dumps(titles, ensure_ascii=False), encoding="utf-8")
    time.sleep(sleep)
    return titles


def wiki_extract(session: requests.Session, title: str, cache_dir: Path, sleep: float, timeout: float, retries: int, backoff: float) -> str:
    safe_title = sha1_bytes(title.encode("utf-8"))
    cache_file = cache_dir / f"page_{safe_title}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8")).get("extract", "")
        except Exception:
            pass
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "titles": title,
        "format": "json",
    }
    resp = _http_request(session, params, timeout=timeout, retries=retries, backoff=backoff)
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    extract = ""
    for _pid, page in pages.items():
        extract = page.get("extract", "") or ""
        break
    cache_file.write_text(json.dumps({"title": title, "extract": extract}, ensure_ascii=False), encoding="utf-8")
    time.sleep(sleep)
    return extract


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks


def paragraph_gate(paragraph: str, seeds: Set[str]) -> bool:
    lower = paragraph.lower()
    for s in seeds:
        if s.lower() in lower:
            return True
    return False


def build_corpus(
    claims: List[Dict],
    out_path: Path,
    cache_dir: Path,
    chunk_size: int,
    overlap: int,
    sleep: float,
    resume: bool,
    timeout: float,
    retries: int,
    backoff: float,
    user_agent: str,
    max_seeds_per_claim: int,
    search_limit: int,
    max_pages_per_seed: int,
    overwrite: bool = False,
) -> Tuple[int, Set[str], Dict[str, List[str]]]:
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(out_path.parent, exist_ok=True)

    if overwrite and out_path.exists():
        out_path.unlink()

    seen = set()
    if out_path.exists() and resume:
        with out_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 3:
                    seen.add((row[1], sha1_bytes(row[2].encode("utf-8"))))

    total_written = 0
    titles_seen: Set[str] = set()
    titles_by_claim: Dict[str, List[str]] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    with out_path.open("a", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        for claim_row in claims:
            claim_text = claim_row.get("claim", "") or ""
            seeds = list(extract_seeds(claim_text))[:max_seeds_per_claim]
            cid = str(claim_row.get("id", "")) or sha1_bytes(claim_text.encode("utf-8"))
            titles_by_claim[cid] = []
            for seed in seeds:
                try:
                    titles = wiki_search(
                        session,
                        seed,
                        cache_dir,
                        sleep,
                        timeout=timeout,
                        retries=retries,
                        backoff=backoff,
                        search_limit=search_limit,
                    )
                except Exception:
                    continue
                for title in titles[:max_pages_per_seed]:
                    if title in titles_seen and resume:
                        continue
                    try:
                        extract = wiki_extract(
                            session,
                            title,
                            cache_dir,
                            sleep,
                            timeout=timeout,
                            retries=retries,
                            backoff=backoff,
                        )
                    except Exception:
                        continue
                    titles_seen.add(title)
                    titles_by_claim[cid].append(title)
                    if not extract:
                        continue
                    paragraphs = [p.strip() for p in extract.split("\n") if p.strip()]
                    paragraphs = [p for p in paragraphs if paragraph_gate(p, seeds)]
                    for p in paragraphs:
                        for chunk in chunk_text(p, chunk_size, overlap):
                            text = f"{title}\n{chunk}"
                            key = (title, sha1_bytes(chunk.encode("utf-8")))
                            if key in seen:
                                continue
                            seen.add(key)
                            writer.writerow([f"{title}", title, text])
                            total_written += 1
    return total_written, titles_seen, titles_by_claim


def write_manifest(manifest_path: Path, manifest: Manifest):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def compute_corpus_hash(path: Path) -> str:
    return file_sha1(path)


def main():
    parser = argparse.ArgumentParser(description="Build a seeded Wikipedia BM25 corpus.")
    parser.add_argument("--claims", default="data/claims_liar.csv", help="Claims CSV file.")
    parser.add_argument("--out", default="data/wiki_seeded_passages.tsv", help="Output TSV for passages.")
    parser.add_argument("--cache-dir", default="data/wiki_cache", help="Directory for Wikipedia API cache.")
    parser.add_argument("--max-claims-for-corpus", type=int, default=None, help="Limit claims used for seeding the corpus.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for claim sampling.")
    parser.add_argument("--chunk-size", type=int, default=180, help="Words per chunk.")
    parser.add_argument("--overlap", type=int, default=20, help="Word overlap between chunks.")
    parser.add_argument("--sleep", type=float, default=0.15, help="Seconds to sleep between Wikipedia requests.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing TSV/cache without refetching titles already seen.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout per request (seconds).")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retries per request.")
    parser.add_argument("--backoff", type=float, default=0.5, help="Exponential backoff base (seconds).")
    parser.add_argument("--user-agent", default="llm-misinfo-benchmark/0.1 (contact: example@example.com)", help="User-Agent for Wikipedia requests.")
    parser.add_argument("--max-seeds-per-claim", type=int, default=3, help="Cap on seeds per claim to avoid explosion.")
    parser.add_argument("--search-limit", type=int, default=2, help="Max titles to fetch per seed query.")
    parser.add_argument("--max-pages-per-seed", type=int, default=1, help="Max pages fetched per seed (after search).")
    parser.add_argument("--http-test", action="store_true", help="Run a quick HTTP connectivity test and exit.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output TSV instead of appending/resuming.")
    args = parser.parse_args()

    claims_path = Path(args.claims)
    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)

    if not claims_path.exists():
        raise SystemExit(f"Missing claims file: {claims_path}")

    if args.http_test:
        cache_dir.mkdir(parents=True, exist_ok=True)
        session = requests.Session()
        session.headers.update({"User-Agent": args.user_agent})
        try:
            t0 = time.time()
            titles = wiki_search(
                session,
                "Wikipedia",
                cache_dir,
                sleep=0.0,
                timeout=args.timeout,
                retries=args.retries,
                backoff=args.backoff,
                search_limit=1,
            )
            t1 = time.time()
            extract = ""
            if titles:
                extract = wiki_extract(
                    session,
                    titles[0],
                    cache_dir,
                    sleep=0.0,
                    timeout=args.timeout,
                    retries=args.retries,
                    backoff=args.backoff,
                )
            t2 = time.time()
            print(f"HTTP test OK. search_ms={(t1-t0)*1000:.1f}, extract_ms={(t2-t1)*1000:.1f}, title={titles[:1]}")
        except Exception as exc:
            raise SystemExit(f"HTTP test failed: {exc}")
        return

    claims = load_claims(claims_path, args.max_claims_for_corpus, args.seed)
    claims_hash = file_sha1(claims_path)
    manifest_path = cache_dir / "manifest.json"

    written, titles_seen, titles_by_claim = build_corpus(
        claims=claims,
        out_path=out_path,
        cache_dir=cache_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        sleep=args.sleep,
        resume=args.resume,
        timeout=args.timeout,
        retries=args.retries,
        backoff=args.backoff,
        user_agent=args.user_agent,
        max_seeds_per_claim=args.max_seeds_per_claim,
        search_limit=args.search_limit,
        max_pages_per_seed=args.max_pages_per_seed,
        overwrite=args.overwrite,
    )

    corpus_hash = compute_corpus_hash(out_path)
    manifest = Manifest(
        claims_hash=claims_hash,
        params={
            "claims": str(claims_path),
            "out": str(out_path),
            "max_claims_for_corpus": args.max_claims_for_corpus,
            "seed": args.seed,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "sleep": args.sleep,
            "resume": args.resume,
            "timeout": args.timeout,
            "retries": args.retries,
            "backoff": args.backoff,
            "user_agent": args.user_agent,
            "max_seeds_per_claim": args.max_seeds_per_claim,
            "search_limit": args.search_limit,
            "max_pages_per_seed": args.max_pages_per_seed,
        },
        timestamp=time.time(),
        pages_fetched=len(titles_seen),
        corpus_hash=corpus_hash,
        titles_by_claim=titles_by_claim,
    )
    write_manifest(manifest_path, manifest)
    print(f"Wrote {written} passages to {out_path}")
    print(f"Pages fetched: {len(titles_seen)} | corpus_hash={corpus_hash}")


if __name__ == "__main__":
    main()
