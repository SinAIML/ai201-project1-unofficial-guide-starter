"""Pre-filtered hybrid semantic search over the ChromaDB RAG index.

Strategy (Option 6 — hybrid routing):
  1. Embed the query once with the shared embedding model.
  2. Route it semantically: compare the query embedding to per-tier anchor
     examples (review / summary / curriculum) and pick the most similar tier
     plus a confidence score.
  3. Confident fast-path: if confidence >= ROUTER_CONFIDENCE_THRESHOLD, apply a
     hard pre-filter on that tier (plus any detected professor/course entity)
     and run cosine search within the subset.
  4. Fallback: if confidence is low, or the confident filter matched nothing,
     pool the top candidates from every tier and globally re-rank by cosine
     distance. This guarantees aggregate questions can reach summary chunks.
  5. Last resort: unfiltered search across the whole collection.

Usage:
    python search.py "Is CSCI 5502 Data Mining a hard course?"
    python search.py "Which professor has the highest rating?"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DOCUMENTS_DIR,
    EMBED_MODEL,
    ROUTER_CONFIDENCE_THRESHOLD,
    TIER_EXAMPLES,
    TOP_K,
    normalize_course_code,
)

# ---------------------------------------------------------------------------
# Known entities — loaded once at import from documents/*.json
# ---------------------------------------------------------------------------

def _normalize_ws(text: str) -> str:
    """Collapse internal runs of whitespace to single spaces and strip ends.

    Mirrors the normalization applied in chunk.py so professor names used in
    the hard filter exactly match the values stored in Chroma metadata.
    """
    return re.sub(r"\s+", " ", text or "").strip()


def _load_professor_names() -> list[str]:
    names: list[str] = []
    for f in sorted(DOCUMENTS_DIR.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if "professor" in doc:
                n = _normalize_ws(doc["professor"].get("name", ""))
                if n and n not in names:
                    names.append(n)
                # also store the queried_name variant
                qn = _normalize_ws(doc["professor"].get("queried_name", ""))
                if qn and qn not in names:
                    names.append(qn)
        except Exception:
            pass
    return names


KNOWN_PROFESSORS: list[str] = _load_professor_names()

# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------

# Legacy keyword router. Kept for reference/back-compat; routing now uses the
# semantic router below, which is robust to phrasing (see route_query_semantic).
AGGREGATE_SIGNALS = [
    "which professor", "best", "easiest", "hardest", "most",
    "compare", "recommend", "who should i take", "highest rating",
    "lowest rating", "who is the best", "who is the worst",
]

CURRICULUM_SIGNALS = [
    "course", "curriculum", "required", "credit", "elective",
    "degree", "program", "prerequisite", "bridge", "ms-ds",
    "msds", "syllabus", "requirement",
]


def route_query(query: str) -> str:
    """Legacy keyword router: 'summary', 'curriculum', or 'review'.

    Superseded by route_query_semantic. Retained for reference and so existing
    callers/tests that import it keep working. Its first-match substring logic
    is brittle: "highest overall rating" misses the "highest rating" signal, and
    any query containing "course" is forced to the curriculum tier.
    """
    q = query.lower()
    if any(sig in q for sig in CURRICULUM_SIGNALS):
        return "curriculum"
    if any(sig in q for sig in AGGREGATE_SIGNALS):
        return "summary"
    return "review"


# --- Semantic router -------------------------------------------------------

_tier_prototypes: dict[str, np.ndarray] | None = None


def _get_tier_prototypes() -> dict[str, np.ndarray]:
    """Lazily embed the per-tier anchor examples (normalized) and cache them."""
    global _tier_prototypes
    if _tier_prototypes is None:
        model = _get_model()
        _tier_prototypes = {}
        for tier, examples in TIER_EXAMPLES.items():
            embs = model.encode(examples, normalize_embeddings=True)
            _tier_prototypes[tier] = np.asarray(embs, dtype=np.float32)
    return _tier_prototypes


def route_query_semantic(query_embedding: Any) -> tuple[str, float]:
    """Route a query to a tier by semantic similarity to anchor examples.

    `query_embedding` is a single embedding vector (list or array). Returns
    (tier, confidence), where confidence is the best tier's highest cosine
    similarity against its anchor examples.
    """
    protos = _get_tier_prototypes()
    q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(q))
    if norm > 0:
        q = q / norm

    scores = {tier: float((mat @ q).max()) for tier, mat in protos.items()}
    best_tier = max(scores, key=scores.get)
    return best_tier, scores[best_tier]


def routing_decision(query: str) -> tuple[str, float]:
    """Convenience wrapper: embed `query` and return (tier, confidence)."""
    embedding = _get_model().encode([query]).tolist()[0]
    return route_query_semantic(embedding)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

_COURSE_PATTERN = re.compile(r"\b([A-Za-z]{2,5}[\s-]?\d{4}[A-Za-z]?)\b")


def extract_entities(query: str) -> dict[str, str]:
    """Return detected professor_name and/or course_code from the query."""
    entities: dict[str, str] = {}

    # Professor name — case-insensitive substring match against known names.
    # Normalize the query's whitespace too so "Al  Pisano" matches "Al Pisano".
    q_lower = _normalize_ws(query).lower()
    for name in KNOWN_PROFESSORS:
        if name.lower() in q_lower:
            entities["professor_name"] = name
            break  # take the first (longest) match

    # Course code — regex match, then canonicalize to the same form as metadata
    course_match = _COURSE_PATTERN.search(query)
    if course_match:
        entities["course_code"] = normalize_course_code(course_match.group(1))

    return entities


# ---------------------------------------------------------------------------
# Where-filter builder
# ---------------------------------------------------------------------------

def build_where(tier: str, entities: dict[str, str]) -> dict[str, Any] | None:
    """Build a ChromaDB where-clause dict.

    Returns None when no filtering is needed (full-collection search).
    """
    conditions: list[dict] = [{"type": {"$eq": tier}}]

    if "professor_name" in entities:
        conditions.append({"professor_name": {"$eq": entities["professor_name"]}})

    if "course_code" in entities and tier == "review":
        conditions.append({"course_code": {"$eq": entities["course_code"]}})

    if len(conditions) == 1:
        # Single condition: no need for $and
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_collection: Any = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection() -> Any:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _query(
    collection: Any,
    query_embedding: list,
    top_k: int,
    where: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Run a single Chroma query and return result dicts (empty on no match)."""
    kwargs: dict[str, Any] = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    try:
        results = collection.query(**kwargs)
    except Exception as exc:  # Chroma can raise if a filter matches zero docs
        print(f"  Search error: {exc}")
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [
        {"text": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(docs, metas, dists)
    ]


def _multi_tier_query(
    collection: Any, query_embedding: list, top_k: int
) -> list[dict[str, Any]]:
    """Pool top candidates from every tier and globally re-rank by distance.

    Querying each tier separately guarantees every tier (notably the summary
    tier) is represented in the candidate pool; the global sort then keeps the
    closest overall. Distances are comparable because the whole collection uses
    a single cosine space.
    """
    pooled: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for tier in TIER_EXAMPLES:
        for hit in _query(collection, query_embedding, top_k, {"type": {"$eq": tier}}):
            key = (hit["metadata"].get("type"), hit["text"])
            if key in seen:
                continue
            seen.add(key)
            pooled.append(hit)

    pooled.sort(key=lambda r: r["distance"])
    return pooled[:top_k]


def search(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Run hybrid-routed semantic search and return top-k result dicts.

    Option 6 routing:
      1. Semantic router picks a tier + confidence.
      2. Confident fast-path (confidence >= ROUTER_CONFIDENCE_THRESHOLD):
         tier + entity filter, then tier-only filter.
      3. Fallback (low confidence, or the confident path matched nothing):
         pool all tiers and globally re-rank.
      4. Last resort: unfiltered full-collection search.
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()
    tier, confidence = route_query_semantic(query_embedding[0])
    entities = extract_entities(query)

    # Confident fast-path: hard pre-filter on the routed tier.
    if confidence >= ROUTER_CONFIDENCE_THRESHOLD:
        where_full = build_where(tier, entities)
        where_type_only = {"type": {"$eq": tier}}
        for where in (where_full, where_type_only):
            hits = _query(collection, query_embedding, top_k, where)
            if hits:
                return hits
        print(f"  (fallback to multi-tier search; tier={tier} matched nothing)")
    else:
        print(
            f"  (low routing confidence {confidence:.3f} < "
            f"{ROUTER_CONFIDENCE_THRESHOLD}; using multi-tier search)"
        )

    # Robust fallback: pool every tier and re-rank globally.
    hits = _multi_tier_query(collection, query_embedding, top_k)
    if hits:
        return hits

    # Last resort: unfiltered.
    return _query(collection, query_embedding, top_k, None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_results(query: str, results: list[dict]) -> None:
    tier, confidence = routing_decision(query)
    entities = extract_entities(query)
    print(f"\nQuery : {query}")
    print(f"Tier  : {tier}  (confidence={confidence:.3f})")
    if entities:
        print(f"Entities detected: {entities}")
    print(f"{'─' * 60}")
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        score = 1 - r["distance"]  # cosine similarity from cosine distance
        chunk_type = meta.get("type", "?")

        # Attribution line
        if chunk_type in ("review", "summary"):
            attribution = (
                f"[{chunk_type}] {meta.get('professor_name', '?')} "
                f"— {meta.get('source', '')} {meta.get('rmp_url', '')}"
            )
        else:
            attribution = (
                f"[{chunk_type}] {meta.get('source', meta.get('doc_id', '?'))} "
                f"— {meta.get('url', '')}"
            )

        print(f"\n#{i}  score={score:.3f}  {attribution}")
        print(r["text"][:400] + ("..." if len(r["text"]) > 400 else ""))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search.py \"<query>\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    hits = search(query)
    _print_results(query, hits)
    sys.exit(0)
