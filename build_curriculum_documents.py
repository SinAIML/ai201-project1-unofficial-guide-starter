"""Scrape CU Boulder MS-DS curriculum pages and write one JSON document per page.

Each output file follows the same schema used by build_documents.py so the
documents slot directly into the existing RAG pipeline.

Usage:
    python build_web_documents.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).parent / "documents"

BASE = "https://www.colorado.edu/program/data-science/campus/curriculum"

PAGES = [
    ("curriculum_msds",             "CU Boulder MS-DS Curriculum Overview",      BASE),
    ("curriculum_bridge_courses",   "CU Boulder MS-DS Curriculum — Bridge Courses",    BASE + "/Bridge-Courses"),
    ("curriculum_statistics",       "CU Boulder MS-DS Curriculum — Statistics",         BASE + "/statistics"),
    ("curriculum_computer_science", "CU Boulder MS-DS Curriculum — Computer Science",   BASE + "/computer-science"),
    ("curriculum_general_ds",       "CU Boulder MS-DS Curriculum — General Data Science", BASE + "/general-data-science"),
    ("curriculum_other_core",       "CU Boulder MS-DS Curriculum — Other Core Courses", BASE + "/other-core-courses"),
    ("curriculum_ds_electives",     "CU Boulder MS-DS Curriculum — Data Science Electives", BASE + "/data-science-electives"),
]


def scrape_page(doc_id: str, source: str, url: str) -> dict | None:
    """Fetch one page and return a citable document dict, or None on failure."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ERROR fetching {url}: {exc}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    main = soup.find("main") or soup.body
    if main is None:
        print(f"  ERROR: no <main> or <body> found for {url}")
        return None

    text = main.get_text(separator="\n", strip=True)

    return {
        "doc_id": doc_id,
        "source": source,
        "url": url,
        "retrieved_at": date.today().isoformat(),
        "text": text,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    failed: list[str] = []

    for doc_id, source, url in PAGES:
        print(f"Scraping: {source}")
        doc = scrape_page(doc_id, source, url)
        if doc is None:
            failed.append(doc_id)
            continue

        out_path = OUTPUT_DIR / f"{doc_id}.json"
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        char_count = len(doc["text"])
        print(f"  wrote {out_path.name} ({char_count:,} chars)")
        written += 1

    print(f"\nDone. Wrote {written}/{len(PAGES)} documents to {OUTPUT_DIR}")
    if failed:
        print("Failed: " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
