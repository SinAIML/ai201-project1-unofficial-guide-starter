"""Embed all chunks and write them to a persistent ChromaDB collection.

Idempotent: deletes and recreates the collection on every run so re-indexing
after document updates is safe.

Usage:
    python index.py
"""

from __future__ import annotations

import sys

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # noqa: BLE001 - we want a clean, actionable message
    print(
        f"ERROR: missing dependency: {exc.name}. "
        "Install requirements first with `pip install -r requirements.txt`.",
        file=sys.stderr,
    )
    sys.exit(1)

from chunk import build_chunks
from config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL

BATCH_SIZE = 128  # tune down if memory is tight


def _validate_chunks(chunks: list[dict]) -> list[str]:
    """Return a list of validation error messages (empty when chunks are valid).

    Guards against the two most common data-quality failures before they reach
    ChromaDB: missing required fields and duplicate IDs.
    """
    errors: list[str] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for i, chunk in enumerate(chunks):
        for field in ("id", "text", "metadata"):
            if field not in chunk:
                errors.append(f"chunk #{i} is missing required field '{field}'")

        cid = chunk.get("id")
        if cid is not None:
            if cid in seen_ids:
                duplicate_ids.add(cid)
            seen_ids.add(cid)

        text = chunk.get("text")
        if isinstance(text, str) and not text.strip():
            errors.append(f"chunk '{cid}' has empty text")

    if duplicate_ids:
        sample = ", ".join(sorted(duplicate_ids)[:5])
        errors.append(
            f"{len(duplicate_ids)} duplicate chunk id(s) detected (e.g. {sample})"
        )

    return errors


def build_index() -> int:
    """Build the index. Returns a process exit code (0 = success)."""
    print("Building chunks...")
    try:
        chunks = build_chunks()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to build chunks: {exc}", file=sys.stderr)
        return 1
    print(f"  {len(chunks)} total chunks")

    if not chunks:
        print(
            "ERROR: no chunks were produced. Check that the documents/ directory "
            "contains valid source files.",
            file=sys.stderr,
        )
        return 1

    errors = _validate_chunks(chunks)
    if errors:
        print("ERROR: chunk validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"Loading embedding model: {EMBED_MODEL}")
    try:
        model = SentenceTransformer(EMBED_MODEL)
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: failed to load embedding model '{EMBED_MODEL}': {exc}. "
            "Check the model name and your network connection (the model is "
            "downloaded on first use).",
            file=sys.stderr,
        )
        return 1

    print(f"Connecting to ChromaDB at: {CHROMA_DIR}")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: failed to open ChromaDB at '{CHROMA_DIR}': {exc}. "
            "Check that the path is writable.",
            file=sys.stderr,
        )
        return 1

    # Delete existing collection so re-runs are idempotent. Distinguish "not
    # found" (expected on a first run) from real failures so a corrupt DB does
    # not get masked and then crash create_collection below.
    try:
        existing = {c.name for c in client.list_collections()}
        if COLLECTION_NAME in existing:
            client.delete_collection(COLLECTION_NAME)
            print(f"  Dropped existing collection '{COLLECTION_NAME}'")
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: failed to reset collection '{COLLECTION_NAME}': {exc}.",
            file=sys.stderr,
        )
        return 1

    try:
        collection = client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"ERROR: failed to create collection '{COLLECTION_NAME}': {exc}.",
            file=sys.stderr,
        )
        return 1
    print(f"  Created collection '{COLLECTION_NAME}'")

    # Embed and upsert in batches
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print("Embedding and indexing...")
    indexed = 0
    for start in range(0, len(chunks), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        batch_ids = ids[start : start + BATCH_SIZE]
        batch_meta = metadatas[start : start + BATCH_SIZE]

        try:
            embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_meta,
            )
        except MemoryError:
            print(
                f"ERROR: ran out of memory while embedding batch starting at "
                f"chunk {start + 1}. Lower BATCH_SIZE (currently {BATCH_SIZE}) "
                "and re-run.",
                file=sys.stderr,
            )
            return 1
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: failed to index batch starting at chunk {start + 1}: "
                f"{exc}. The collection may be partially populated; re-run to "
                "rebuild it cleanly.",
                file=sys.stderr,
            )
            return 1

        indexed += len(batch_texts)
        end = min(start + BATCH_SIZE, len(chunks))
        print(f"  Indexed chunks {start + 1}–{end} / {len(chunks)}")

    print(f"\nDone. {indexed} chunks indexed into '{COLLECTION_NAME}'.")
    return 0


if __name__ == "__main__":
    sys.exit(build_index())
