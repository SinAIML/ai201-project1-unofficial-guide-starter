"""Retrieval-augmented generation over the ChromaDB index using Groq.

Pipeline:
  1. Retrieve top-k chunks for the query via search.search().
  2. Format them into numbered, attributed context blocks.
  3. Ask a Groq-hosted LLM to answer using ONLY that context, with [n] citations.

Usage:
    python generate.py "Is CSCI 5502 Data Mining a hard course?"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

from config import GROQ_MODEL, TOP_K
from search import search

# The .env lives one directory above the project folder; load it explicitly and
# also fall back to the default search so either layout works.
load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv()

# Number of prior turns (user+assistant pairs) to pass to the model for context.
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = """You are The Unofficial Guide, an assistant that answers questions \
about University of Colorado Boulder Data Science professors and the MS-DS curriculum.

Rules:
- Answer ONLY using the information in the provided context blocks.
- Cite the sources you use with their bracketed number, e.g. [1] or [2].
- If the context does not contain enough information to answer, say "I don't have \
enough information to answer that based on the available reviews and curriculum data." \
Do not invent professors, courses, ratings, or facts.
- Be concise and specific. When summarizing student opinions, make clear they are \
student reviews, not objective fact."""


def _attribution(meta: dict[str, Any]) -> str:
    """Build a short, human-readable source label from chunk metadata."""
    ctype = meta.get("type", "?")
    if ctype in ("review", "summary"):
        who = meta.get("professor_name", "Unknown professor")
        src = meta.get("source", "RateMyProfessors")
        url = meta.get("rmp_url", "")
        course = meta.get("course_code")
        label = f"({ctype}) {who}"
        if course:
            label += f" — {course}"
        label += f" — {src}"
        if url:
            label += f" {url}"
        return label

    # curriculum / other
    src = meta.get("source") or meta.get("doc_id", "curriculum")
    url = meta.get("url", "")
    label = f"({ctype}) {src}"
    if url:
        label += f" {url}"
    return label


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as numbered, attributed context blocks."""
    if not chunks:
        return "(no relevant context was retrieved)"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] {_attribution(c['metadata'])}\n{c['text'].strip()}"
        )
    return "\n\n".join(blocks)


def build_messages(
    query: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Assemble the chat messages: system + recent history + grounded user turn."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        # Keep only the most recent turns to bound the prompt size.
        for msg in history[-MAX_HISTORY_MESSAGES:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    context = format_context(chunks)
    user_content = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def _get_client():
    """Create a Groq client, with a clear error if the key is missing."""
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to the .env file (see .env at the "
            "repository root) or export it in your environment."
        )
    # Imported lazily so a missing dependency produces a clean message.
    from groq import Groq

    return Groq()


def retrieve(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Retrieve the top-k chunks for a query (thin wrapper around search.search)."""
    return search(query, top_k=top_k)


def generate_answer_stream(
    query: str,
    history: list[dict[str, str]] | None = None,
    top_k: int = TOP_K,
    chunks: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """Yield answer text deltas for a query, grounded in retrieved chunks.

    If `chunks` is provided it is used directly (so the caller can reuse and
    display the same retrieval); otherwise retrieval runs here.
    """
    if chunks is None:
        chunks = retrieve(query, top_k=top_k)

    messages = build_messages(query, chunks, history)
    client = _get_client()

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def generate_answer(
    query: str,
    history: list[dict[str, str]] | None = None,
    top_k: int = TOP_K,
) -> tuple[str, list[dict[str, Any]]]:
    """Non-streaming convenience wrapper. Returns (answer_text, retrieved_chunks)."""
    chunks = retrieve(query, top_k=top_k)
    answer = "".join(
        generate_answer_stream(query, history=history, top_k=top_k, chunks=chunks)
    )
    return answer, chunks


def _print_sources(chunks: list[dict[str, Any]]) -> None:
    print(f"\n{'─' * 60}\nSources:")
    if not chunks:
        print("  (none)")
        return
    for i, c in enumerate(chunks, start=1):
        score = 1 - c["distance"]
        print(f"  [{i}] score={score:.3f}  {_attribution(c['metadata'])}")


if __name__ == "__main__":
    # Ensure Unicode (em-dashes, box chars) prints on legacy Windows consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if len(sys.argv) < 2:
        print('Usage: python generate.py "<question>"')
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])
    try:
        retrieved = retrieve(user_query)
        print(f"Query: {user_query}\n{'─' * 60}")
        for piece in generate_answer_stream(user_query, chunks=retrieved):
            print(piece, end="", flush=True)
        print()
        _print_sources(retrieved)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
