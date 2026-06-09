"""Build all chunks for the RAG pipeline from documents/*.json.

Three chunk types are produced:
  - Tier 1 (type=review):      one chunk per RMP review, metadata-injected text
  - Tier 2 (type=summary):     one synthesized chunk per professor
  - Curriculum (type=curriculum): fixed-size chunks from curriculum JSON pages

Usage:
    python chunk.py          # prints chunk counts per type
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCUMENTS_DIR, normalize_course_code

# ---------------------------------------------------------------------------
# Curriculum chunking helpers
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

_curriculum_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # Prefer sentence boundaries before falling back to word/character splits
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Navigation boilerplate found at the bottom of every CU curriculum page.
# These chunks contain only site navigation links and no course content.
_NAV_SIGNALS = [
    "Admissions\nStudent Events\nCurriculum\n",
    "admissions\nstudent events\ncurriculum\n",
    "Student Events\nCurriculum\nBridge Courses\n",
    "student events\ncurriculum\nbridge courses\n",
]
_NAV_PHRASES = [
    "looking for something else?",
    "additional courses can be found in our",
    "apply now\nrequest more info",
    "join a webinar",
    # Elective request form template — field labels only, not a course listing
    "course name\ncourse number\ncourse description",
]


def _is_nav_chunk(text: str) -> bool:
    """Return True if the chunk is mostly site-navigation boilerplate.

    Navigation footer chunks crowd out real course-content chunks in retrieval
    because they mention "courses", "curriculum", and "MS-DS" in link text.
    """
    lower = text.lower()
    if any(sig in lower for sig in _NAV_PHRASES):
        return True
    # Count navigation-menu lines vs content lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    nav_keywords = {"admissions", "student events", "curriculum", "bridge courses",
                    "core curriculum", "data science electives", "faculty", "finances",
                    "faqs", "current students", "graduate certificate", "student stories",
                    "apply now", "request more info", "join a webinar", "electives list"}
    nav_line_count = sum(1 for ln in lines if ln.lower() in nav_keywords)
    return nav_line_count / len(lines) > 0.5


def chunk_text(text: str) -> list[str]:
    """Split curriculum text into overlapping chunks with sentence-boundary preference.

    Falls back to returning the whole text as a single chunk if the splitter
    produces nothing (e.g. text shorter than the overlap window).
    Navigation-only chunks are filtered out before returning.
    """
    raw = _curriculum_splitter.split_text(text)
    chunks = [c.strip() for c in raw if c.strip() and not _is_nav_chunk(c)]
    return chunks or [text.strip()]


# ---------------------------------------------------------------------------
# Generic safety helpers
# ---------------------------------------------------------------------------


def _warn(message: str) -> None:
    """Emit a non-fatal warning to stderr."""
    print(f"WARNING: {message}", file=sys.stderr)


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float, returning default on None or non-numeric input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    """Coerce a value to int, returning default on None or non-numeric input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_ws(text: str) -> str:
    """Collapse internal runs of whitespace to single spaces and strip ends."""
    return re.sub(r"\s+", " ", text or "").strip()


def _unique_id(candidate: str, seen: set[str]) -> str:
    """Return an id guaranteed unique within `seen`, suffixing duplicates."""
    if candidate not in seen:
        seen.add(candidate)
        return candidate
    i = 2
    while f"{candidate}-{i}" in seen:
        i += 1
    new_id = f"{candidate}-{i}"
    seen.add(new_id)
    _warn(f"duplicate chunk id '{candidate}' renamed to '{new_id}'")
    return new_id


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_chunks() -> list[dict[str, Any]]:
    """Return all chunks as a list of dicts: {id, text, metadata}.

    Each document is processed in isolation: a malformed or unreadable file is
    skipped with a warning instead of aborting the whole run. Chunk ids are
    de-duplicated across the entire corpus.
    """
    all_chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for json_file in sorted(DOCUMENTS_DIR.glob("*.json")):
        try:
            doc = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _warn(f"skipping {json_file.name}: could not read/parse ({exc})")
            continue

        fallback_id = json_file.stem
        try:
            if "professor" in doc:
                chunks = _chunks_from_professor_doc(doc, fallback_id)
            else:
                chunks = _chunks_from_curriculum_doc(doc, fallback_id)
        except Exception as exc:  # noqa: BLE001 - isolate per-document failures
            _warn(f"skipping {json_file.name}: failed to chunk ({exc})")
            continue

        for chunk in chunks:
            chunk["id"] = _unique_id(chunk["id"], seen_ids)
            all_chunks.append(chunk)

    return all_chunks


def _chunks_from_professor_doc(doc: dict, fallback_id: str) -> list[dict[str, Any]]:
    """Produce Tier-1 review chunks + one Tier-2 summary chunk for a professor."""
    prof = doc.get("professor") or {}
    name = _normalize_ws(prof.get("name") or prof.get("queried_name") or "") or "Unknown"
    dept = _normalize_ws(prof.get("department") or "") or "Unknown"
    slug = doc.get("doc_id") or fallback_id
    rmp_url = prof.get("rmp_url", "")
    summary_id = f"{slug}_summary"

    chunks: list[dict[str, Any]] = []
    tag_counter: Counter = Counter()
    top_excerpts: list[tuple[int, str]] = []  # (thumbs_up, first_sentence)

    # --- Tier 1: one chunk per review ---
    for idx, review in enumerate(doc.get("reviews") or [], start=1):
        comment = (review.get("comment") or "").strip()
        if not comment:
            continue

        course = normalize_course_code(review.get("course") or "") or "N/A"
        quality = _to_float(review.get("quality"))
        difficulty = _to_float(review.get("difficulty"))
        tags: list[str] = review.get("tags") or []
        thumbs_up = _to_int(review.get("thumbs_up"))
        thumbs_down = _to_int(review.get("thumbs_down"))
        date: str = review.get("date") or ""
        # Synthesize a stable id if the source omitted review_id
        review_id = review.get("review_id") or f"{slug}::r{idx}"

        # Minimal metadata injection: Professor | Department | Course only
        text = f"Professor: {name} | Department: {dept} | Course: {course}\n{comment}"

        chunks.append({
            "id": review_id,
            "text": text,
            "metadata": {
                "type": "review",
                "professor_name": name,
                "department": dept,
                "course_code": course,
                "quality": quality,
                "difficulty": difficulty,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "date": date,
                "source": "RateMyProfessors",
                "rmp_url": rmp_url,
                "parent_id": summary_id,
            },
        })

        # Accumulate for Tier-2 summary
        tag_counter.update(tags)
        first_sentence = re.split(r"(?<=[.!?])\s", comment)[0]
        top_excerpts.append((thumbs_up, first_sentence))

    if not chunks:
        _warn(f"{slug}: no usable reviews; emitting summary chunk only")

    # --- Tier 2: one summary chunk per professor ---
    avg_q = _to_float(prof.get("overall_rating"))
    avg_d = _to_float(prof.get("level_of_difficulty"))
    pct = _to_float(prof.get("percent_take_again"))
    n = _to_int(prof.get("num_ratings"))

    top_tags_str = ", ".join(t for t, _ in tag_counter.most_common(5)) or "none"

    # Prefer the highest-thumbs_up excerpts; ties keep original (recency) order
    top_excerpts.sort(key=lambda x: x[0], reverse=True)
    excerpts_str = " | ".join(ex for _, ex in top_excerpts[:3]) or "No notable excerpts."

    summary_text = (
        f"Professor {name} — average quality {avg_q}/5, average difficulty {avg_d}/5, "
        f"based on {n} ratings. Department: {dept}. "
        f"{pct:.1f}% would take again. "
        f"Common themes: {top_tags_str}. "
        f"Sample feedback: {excerpts_str}"
    )

    chunks.append({
        "id": summary_id,
        "text": summary_text,
        "metadata": {
            "type": "summary",
            "professor_name": name,
            "department": dept,
            "review_count": n,
            "avg_quality": avg_q,
            "avg_difficulty": avg_d,
            "percent_take_again": pct,
            "rmp_url": rmp_url,
            "source": "RateMyProfessors",
        },
    })

    return chunks


_COURSE_LINE_RE = re.compile(
    r"([A-Z]{2,5}\s*\d{4}[A-Za-z]?)"   # course code
    r"[:\s\-–]+([^\n]{3,80})",          # separator + course name (3–80 chars)
)
# Prerequisite noise: names that start with conjunctions/prepositions
_PREREQ_NOISE_RE = re.compile(r"^(and|or|with|of|a |the |min|all)\b", re.I)


def _extract_course_index_lines(text: str, source: str) -> list[str]:
    """Return compact 'CODE: Name' lines for each distinct course found in text.

    Used to build a per-document course-index chunk that embeds well for
    'list all MSDS courses' queries.  Course codes below 5000 are skipped since
    they appear as prerequisites, not MS-DS program offerings.
    """
    seen: set[str] = set()
    lines: list[str] = []
    if source:
        lines.append(f"MS-DS courses in {source}:")
    for m in _COURSE_LINE_RE.finditer(text):
        code_raw = re.sub(r"\s+", " ", m.group(1)).strip()
        # Extract numeric part to filter out sub-5000 prerequisite courses
        num_match = re.search(r"\d+", code_raw)
        if num_match and int(num_match.group()) < 5000:
            continue
        name = re.sub(r"\s+", " ", m.group(2)).strip().rstrip("(").strip()
        if code_raw not in seen and len(name) > 4 and not _PREREQ_NOISE_RE.match(name) and not name.startswith("("):
            seen.add(code_raw)
            lines.append(f"{code_raw}: {name}")
    return lines if len(lines) > 1 else []   # skip if no courses found


def _chunks_from_curriculum_doc(doc: dict, fallback_id: str) -> list[dict[str, Any]]:
    """Produce fixed-size chunks from a curriculum page document."""
    doc_id = doc.get("doc_id") or fallback_id
    text = (doc.get("text") or "").strip()
    if not text:
        _warn(f"{doc_id}: empty 'text' field; document skipped")
        return []

    source = doc.get("source", "")
    url = doc.get("url", "")

    raw_chunks = chunk_text(text)
    result: list[dict[str, Any]] = []

    for i, chunk in enumerate(raw_chunks):
        result.append({
            "id": f"{doc_id}_c{i}",
            "text": chunk,
            "metadata": {
                "type": "curriculum",
                "source": source,
                "url": url,
                "doc_id": doc_id,
                "chunk_index": i,
            },
        })

    # Build a compact course-index chunk for this document: one line per course
    # found in the text.  This chunk embeds well for "list all MSDS courses" queries
    # where course-code-dense chunks would otherwise rank below structural text.
    course_lines = _extract_course_index_lines(text, source)
    if course_lines:
        index_text = "\n".join(course_lines)
        result.append({
            "id": f"{doc_id}_index",
            "text": index_text,
            "metadata": {
                "type": "curriculum",
                "source": source,
                "url": url,
                "doc_id": doc_id,
                "chunk_index": -1,  # sentinel: this is a synthesized index chunk
            },
        })

    return result


# ---------------------------------------------------------------------------
# CLI verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chunks = build_chunks()
    counts: Counter = Counter(c["metadata"]["type"] for c in chunks)
    print(f"Total chunks: {len(chunks)}")
    for chunk_type, count in sorted(counts.items()):
        print(f"  {chunk_type}: {count}")
    sys.exit(0)
