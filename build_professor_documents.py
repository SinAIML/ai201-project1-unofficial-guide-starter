"""Build the RAG source corpus: one JSON document per CU Boulder professor.

For each professor listed in PROFESSORS, this script:
  1. Looks up the professor on RateMyProfessors (scoped to CU Boulder).
  2. Pulls every review (rating) for that professor.
  3. Writes a single citable JSON document to documents/<slug>.json.

Each review carries a stable ``review_id`` so the downstream RAG pipeline can
attribute generated answers back to a specific review.

Usage:
    python build_documents.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from rmp_client import RMPClient
from rmp_client.errors import RMPError
from rmp_client.models import Professor

# CU Boulder on RateMyProfessors.
SCHOOL_QUERY = "University of Colorado Boulder"

# Hard-coded target professors (CU Boulder Data Science program).
PROFESSORS = [
    "Alfonso Bastias",
    "Jem Corcoran",
    "Brian Zaharatos",
    "Geena Kim",
    "Ioana Fleming",
    "Al Pisano",
    "Sriram Sankarnarayanan",
    "Qin Lv",
    "William Kuskin",
    "Osita Onyejekwe",
    "Alan Paradise",
    "Christopher Vargo",
]

OUTPUT_DIR = Path(__file__).parent / "documents"


def slugify(name: str) -> str:
    """Turn a professor name into a filesystem- and citation-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _normalize_school(name: str) -> str:
    """Collapse a school name to lowercase alphanumerics for exact matching.

    This distinguishes the real "University of Colorado - Boulder" from the
    typo'd stub "Univerity of Colorado Boulder" that has no professors.
    """
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def find_school_id(client: RMPClient) -> str:
    """Resolve the CU Boulder school ID via search."""
    result = client.search_schools(SCHOOL_QUERY)
    target = _normalize_school(SCHOOL_QUERY)
    for school in result.schools:
        if _normalize_school(school.name) == target:
            print(f"  school: {school.name} (id={school.id})")
            return school.id
    raise SystemExit(f"Could not find a school matching {SCHOOL_QUERY!r}")


def fetch_professor(client: RMPClient, name: str, school_id: str) -> Optional[Professor]:
    """Search for a professor at CU Boulder and return the first result."""
    result = client.search_professors(name, school_id=school_id)
    if not result.professors:
        return None
    return result.professors[0]


def build_document(client: RMPClient, name: str, prof: Professor) -> dict:
    """Assemble one citable JSON document for a professor and their reviews."""
    slug = slugify(name)
    reviews = []
    text_blocks = []

    for i, rating in enumerate(client.iter_professor_ratings(prof.id), start=1):
        comment = (rating.comment or "").strip()
        if not comment:
            continue
        review_id = f"{slug}::r{i}"
        reviews.append(
            {
                "review_id": review_id,
                "date": rating.date.isoformat() if rating.date else None,
                "course": rating.course_raw,
                "quality": rating.quality,
                "difficulty": rating.difficulty,
                "tags": rating.tags,
                "thumbs_up": rating.thumbs_up,
                "thumbs_down": rating.thumbs_down,
                "comment": comment,
            }
        )
        course = f" [{rating.course_raw}]" if rating.course_raw else ""
        text_blocks.append(
            f"Review {i}{course} (quality={rating.quality}, "
            f"difficulty={rating.difficulty}): {comment}"
        )

    school_name = prof.school.name if prof.school else SCHOOL_QUERY
    header = (
        f"Professor {prof.name} teaches in the {prof.department or 'Unknown'} "
        f"department at {school_name}. Overall rating {prof.overall_rating}/5 "
        f"from {prof.num_ratings} ratings; average difficulty "
        f"{prof.level_of_difficulty}/5; {prof.percent_take_again}% would take again."
    )

    return {
        "doc_id": slug,
        "source": "RateMyProfessors",
        "retrieved_at": date.today().isoformat(),
        "professor": {
            "queried_name": name,
            "name": prof.name,
            "rmp_id": prof.id,
            "department": prof.department,
            "school": school_name,
            "overall_rating": prof.overall_rating,
            "num_ratings": prof.num_ratings,
            "percent_take_again": prof.percent_take_again,
            "level_of_difficulty": prof.level_of_difficulty,
            "rmp_url": f"https://www.ratemyprofessors.com/professor/{prof.id}",
        },
        "num_reviews": len(reviews),
        "text": header + "\n\n" + "\n\n".join(text_blocks),
        "reviews": reviews,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    missing: list[str] = []

    with RMPClient() as client:
        print("Resolving school...")
        school_id = find_school_id(client)

        for name in PROFESSORS:
            print(f"\nProcessing: {name}")
            try:
                prof = fetch_professor(client, name, school_id)
            except RMPError as exc:
                print(f"  ERROR searching: {exc}")
                missing.append(name)
                continue

            if prof is None:
                print("  not found at CU Boulder, skipping")
                missing.append(name)
                continue

            print(f"  matched: {prof.name} (id={prof.id}, {prof.num_ratings} ratings)")
            try:
                doc = build_document(client, name, prof)
            except RMPError as exc:
                print(f"  ERROR fetching ratings: {exc}")
                missing.append(name)
                continue

            out_path = OUTPUT_DIR / f"{doc['doc_id']}.json"
            out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  wrote {out_path.name} ({doc['num_reviews']} reviews)")
            written += 1

    print(f"\nDone. Wrote {written}/{len(PROFESSORS)} documents to {OUTPUT_DIR}")
    if missing:
        print("Missing / unresolved: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
