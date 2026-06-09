"""Shared configuration constants and helpers for the RAG pipeline."""

import re
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHROMA_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "cu_boulder_guide"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10

# Groq-hosted generation model. Swap for any model id available on your Groq
# account (e.g. "llama-3.1-8b-instant" for faster/cheaper responses).
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Semantic query routing (search.py) -----------------------------------
# The router embeds the query and compares it to the anchor examples below.
# If the best tier's confidence is at least this threshold, that tier is used
# as a hard pre-filter; otherwise the search falls back to pooling all tiers.
# Tuned empirically against the evaluation questions (see README).
ROUTER_CONFIDENCE_THRESHOLD = 0.35

# Anchor/prototype queries per tier. The router routes a query to the tier
# whose anchors it is most semantically similar to — this generalizes to
# paraphrases instead of relying on exact keyword substrings.
TIER_EXAMPLES = {
    "review": [
        "Is this professor good at teaching?",
        "What do students say about grading and exams?",
        "Is this course hard or easy?",
        "What are the negatives of this professor's teaching style?",
        "How is the workload and difficulty for this class?",
    ],
    "summary": [
        "Which professor has the highest overall rating?",
        "Who is the best professor to take?",
        "Compare professors by difficulty and quality.",
        "Which instructor should I take and who should I avoid?",
        "Recommend the easiest or hardest professor.",
    ],
    "curriculum": [
        "What courses are required for the MS-DS degree?",
        "What elective courses are available in the program?",
        "What are the degree credit and prerequisite requirements?",
        "What does the curriculum or syllabus cover?",
        "What are the bridge courses for the program?",
    ],
}

_COURSE_CODE_RE = re.compile(r"^\s*([A-Za-z]{2,5})[\s-]?(\d{4})([A-Za-z]?)\s*$")


def normalize_course_code(raw: str) -> str:
    """Canonicalize a course code to 'DEPT NNNN' form.

    The source corpus mixes formats — 'CSCI4308', 'CSCI-1000', 'CSCI 5502'.
    Normalizing both the stored metadata (chunk.py) and the query-extracted
    code (search.py) to a single canonical form keeps the hard filter aligned.

    Examples:
        'CSCI-1000' -> 'CSCI 1000'
        'csci4308'  -> 'CSCI 4308'
        'CSCI 5502' -> 'CSCI 5502'

    Input that doesn't match the pattern is returned stripped (uppercased).
    """
    if not raw:
        return ""
    m = _COURSE_CODE_RE.match(raw)
    if not m:
        return raw.strip().upper()
    dept, number, suffix = m.groups()
    return f"{dept.upper()} {number}{suffix.upper()}"
