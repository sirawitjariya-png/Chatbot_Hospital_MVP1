"""Single-collection RAG with metadata filter, cached query expansion, optional reranker."""
import glob
import logging
import os
from functools import lru_cache
from pathlib import Path

import chromadb
import requests
import trafilatura
from openai import OpenAI
from pypdf import PdfReader

from .config import (
    EMBED_MODEL,
    CHROMA_DIR,
    ROUTER_MODEL,
    OPENAI_TIMEOUT_S,
    RERANKER,
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
)

log = logging.getLogger(__name__)
client = OpenAI()

_db = chromadb.PersistentClient(path=CHROMA_DIR)

COLLECTION_NAME = "hospital-kb"  # why: single collection; source_type metadata distinguishes md vs url


def _load_collection(name: str):
    try:
        return _db.get_or_create_collection(name)
    except Exception:
        log.warning("Collection %s had corrupt config; recreating.", name)
        _db.delete_collection(name)
        return _db.create_collection(name)


_col = _load_collection(COLLECTION_NAME)

# Optional Cohere reranker — imported lazily so the package stays optional
_cohere = None
if RERANKER == "cohere" and COHERE_API_KEY:
    try:
        import cohere
        _cohere = cohere.Client(COHERE_API_KEY)
        log.info("Cohere reranker enabled (%s)", COHERE_RERANK_MODEL)
    except Exception as e:
        log.warning("Cohere reranker requested but not available: %s", e)


def embed(texts):
    """Batch-embed strings via OpenAI."""
    r = client.embeddings.create(model=EMBED_MODEL, input=texts, timeout=OPENAI_TIMEOUT_S)
    return [d.embedding for d in r.data]


def chunk(text, size=800, overlap=120):
    out, step = [], size - overlap
    for i in range(0, max(len(text), 1), step):
        piece = text[i:i + size].strip()
        if piece:
            out.append(piece)
        if i + size >= len(text):
            break
    return out


# --------------------------- loaders -----------------------------------------
def _data_root() -> Path:
    # why: keep ingest cwd-independent — resolve from repo root, not cwd
    from .config import LOGS_DIR  # local import to avoid cycle
    return LOGS_DIR.parent / "data"


def load_files():
    root = _data_root() / "raw"
    for path in glob.glob(str(root / "**" / "*"), recursive=True):
        if os.path.isdir(path):
            continue
        try:
            if path.endswith((".txt", ".md")):
                yield open(path, encoding="utf-8").read(), path
            elif path.endswith(".pdf"):
                text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
                if text.strip():
                    yield text, path
        except Exception as e:
            log.warning("skip file %s: %s", path, e)


def load_urls():
    urls_path = _data_root() / "urls.txt"
    if not urls_path.exists():
        return
    for url in urls_path.read_text().splitlines():
        url = url.strip()
        if not url or url.startswith("#"):
            continue
        try:
            html = requests.get(url, timeout=15).text
            text = trafilatura.extract(html) or ""
            if text:
                yield text, url
        except Exception as e:
            log.warning("skip url %s: %s", url, e)


# --------------------------- ingest ------------------------------------------
def ingest():
    """Re-ingest everything into ONE collection with source_type metadata."""
    docs, metas = [], []

    for text, src in load_files():
        for c in chunk(text):
            docs.append(c)
            metas.append({"source": src, "source_type": "md"})

    for text, src in load_urls():
        for c in chunk(text):
            docs.append(c)
            metas.append({"source": src, "source_type": "url"})

    if not docs:
        log.warning("No documents found. Drop files into data/raw/ or URLs into data/urls.txt.")
        return

    # Clear collection first — why: re-ingest must not duplicate IDs
    try:
        existing = _col.get()
        if existing.get("ids"):
            _col.delete(ids=existing["ids"])
    except Exception:
        pass

    _col.add(
        ids=[str(i) for i in range(len(docs))],
        documents=docs,
        embeddings=embed(docs),
        metadatas=metas,
    )
    n_md  = sum(1 for m in metas if m["source_type"] == "md")
    n_url = sum(1 for m in metas if m["source_type"] == "url")
    n_src = len({m["source"] for m in metas})
    log.info("Indexed %d chunks (%d md, %d url) from %d source(s).", len(docs), n_md, n_url, n_src)
    print(f"Indexed {len(docs)} chunks (md={n_md}, url={n_url}) from {n_src} source(s).")


# --------------------------- query-time --------------------------------------
def _expand_queries_uncached(question: str) -> tuple[str, ...]:
    """Generate 3 paraphrases (Thai + English). Uses the cheap ROUTER_MODEL."""
    prompt = (
        "You are helping retrieve hospital information. "
        "Generate 3 alternative phrasings of the question below — "
        "include both Thai and English if the question is in one language. "
        "Return ONLY the questions, one per line, no numbering or explanation.\n\n"
        f"Question: {question}"
    )
    try:
        out = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=OPENAI_TIMEOUT_S,
        ).choices[0].message.content
        variants = [q.strip() for q in out.strip().splitlines() if q.strip()]
        return tuple([question] + variants[:3])
    except Exception as e:
        log.warning("query expansion failed, falling back to original only: %s", e)
        return (question,)


@lru_cache(maxsize=500)
def _expand_queries_cached(question: str) -> tuple[str, ...]:
    # why: same FAQ asked many times = free after first call. Process-local cache.
    return _expand_queries_uncached(question)


def _rerank(question: str, chunks: list[str], top_n: int = 4) -> list[str]:
    """Cohere reranker if configured, else identity."""
    if not _cohere or not chunks:
        return chunks[:top_n]
    try:
        resp = _cohere.rerank(
            model=COHERE_RERANK_MODEL,
            query=question,
            documents=chunks,
            top_n=top_n,
        )
        return [chunks[r.index] for r in resp.results]
    except Exception as e:
        log.warning("Cohere rerank failed, using vector order: %s", e)
        return chunks[:top_n]


def retrieve(question: str, k: int = 5, source_types: list[str] | None = None) -> list[dict]:
    """Retrieve from the single collection with optional source_type filter.

    Returns a list of {'text': str, 'source': str, 'source_type': str}.
    """
    queries = _expand_queries_cached(question)
    where = {"source_type": {"$in": source_types}} if source_types else None

    seen, hits = set(), []
    for q in queries:
        e = embed([q])[0]
        res = _col.query(query_embeddings=[e], n_results=k, where=where)
        docs  = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for doc, m in zip(docs, metas):
            if doc in seen:
                continue
            seen.add(doc)
            hits.append({
                "text": doc,
                "source": (m or {}).get("source", ""),
                "source_type": (m or {}).get("source_type", ""),
            })

    # Cap before rerank — why: reranker has a 1000-doc soft limit and per-call cost
    candidates = hits[: max(k * 3, 12)]
    texts = [h["text"] for h in candidates]
    reranked = _rerank(question, texts, top_n=k)

    # rebuild as dicts in reranked order
    by_text = {h["text"]: h for h in candidates}
    return [by_text[t] for t in reranked if t in by_text]
