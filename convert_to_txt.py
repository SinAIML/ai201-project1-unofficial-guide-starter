"""Convert all JSON documents in documents/ to structured TXT files.

Detects two document types:
  - RMP professor reviews (has a "professor" key)
  - Curriculum pages (no "professor" key)

Each TXT file preserves every metadata field from the JSON in a
human-readable format suitable for the RAG pipeline.

Usage:
    python convert_to_txt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"


def format_professor_doc(doc: dict) -> str:
    prof = doc["professor"]
    lines = [
        f"Source: {doc['source']}",
        f"Doc ID: {doc['doc_id']}",
        f"Retrieved: {doc['retrieved_at']}",
        "",
        f"Professor: {prof['name']}",
        f"Department: {prof['department']}",
        f"School: {prof['school']}",
        f"Overall Rating: {prof['overall_rating']}/5.0",
        f"Difficulty: {prof['level_of_difficulty']}/5.0",
        f"Would Take Again: {prof['percent_take_again']}%",
        f"Number of Ratings: {prof['num_ratings']}",
        f"RMP URL: {prof['rmp_url']}",
        "",
        "--- Reviews ---",
    ]

    for review in doc.get("reviews", []):
        date = review.get("date") or "Unknown"
        course = review.get("course") or "N/A"
        lines.append("")
        lines.append(f"[{date} | {course}] (Review ID: {review['review_id']})")
        lines.append(f"Quality: {review['quality']}, Difficulty: {review['difficulty']}")
        lines.append(f"Tags: {review.get('tags', [])}")
        lines.append(f"Thumbs Up: {review.get('thumbs_up', 0)}, Thumbs Down: {review.get('thumbs_down', 0)}")
        lines.append(review.get("comment", ""))

    return "\n".join(lines) + "\n"


def format_curriculum_doc(doc: dict) -> str:
    lines = [
        f"Source: {doc['source']}",
        f"Doc ID: {doc['doc_id']}",
        f"URL: {doc['url']}",
        f"Retrieved: {doc['retrieved_at']}",
        "",
        "--- Content ---",
        "",
        doc.get("text", ""),
    ]

    return "\n".join(lines) + "\n"


def main() -> int:
    json_files = sorted(DOCUMENTS_DIR.glob("*.json"))
    if not json_files:
        print("No JSON files found in documents/")
        return 1

    converted = 0
    for json_file in json_files:
        doc = json.loads(json_file.read_text(encoding="utf-8"))

        if "professor" in doc:
            txt = format_professor_doc(doc)
        else:
            txt = format_curriculum_doc(doc)

        out_path = json_file.with_suffix(".txt")
        out_path.write_text(txt, encoding="utf-8")
        print(f"  wrote {out_path.name}")
        converted += 1

    print(f"\nDone. Converted {converted} JSON files to TXT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
