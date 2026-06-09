# Presentation Script — The Unofficial Guide (3-min version)

> Slides 1–8: ~90 sec | Demo + eval walkthrough: ~90 sec

---

## Slides 1–2 — Title + Problem *(~20 sec)*

"This is The Unofficial Guide — a RAG system for CU Boulder's MS-DS program. Official catalogs tell you *what* a course covers but not *how* it's taught. This system pulls student reviews from RateMyProfessors and official curriculum pages and lets you ask plain-language questions and get cited, grounded answers."

---

## Slides 3–6 — Architecture, Sources, Chunking, Retrieval *(~30 sec)*

"The pipeline: 19 JSON documents get chunked three ways — one chunk per review with a metadata prefix injected, one synthesized summary chunk per professor for aggregate queries, and 500-char overlapping chunks for curriculum pages. These get embedded with `all-MiniLM-L6-v2` locally — no API cost — stored in ChromaDB, and retrieved top-k=5 by cosine similarity with metadata pre-filtering so you don't get reviews from the wrong professor."

---

## Slide 7 — Query Routing *(~15 sec)*

"The original keyword router mis-routed three of my five eval queries. I replaced it with a hybrid semantic router — cosine similarity against tier anchor examples. If confidence is above 0.35, fast-path. Otherwise, pool all tiers and re-rank. All five queries now route correctly."

---

## Slide 8 — Generation *(~10 sec)*

"Generation is Groq LLaMA 3.3 at temperature 0.2. The system prompt has one key rule: answer only from the context, cite with [n], and if the data isn't there, say so."

---

## LIVE DEMO *(~60 sec total)*

*[Open Streamlit or terminal. Keep Sources expander visible.]*

---

**Query 1 — Curriculum** *(15 sec)*
```
python generate.py "What courses are required for the MS-DS degree?"
```
"Curriculum query, routes with 1.0 confidence. Returns the 30-credit structure with [1][2] citations to the actual curriculum pages."

---

**Query 2 — Retrieval works well** *(20 sec)*
```
python generate.py "What do students say about grading fairness in statistics courses?"
```
"This is my best case. No professor name in the query, so the retrieval runs across all reviews and surfaces the most semantically relevant ones — grading reviews for Bastias and Qin Lv. Qin Lv has a review that says 'playing darts for your grade.' The model assembles that into a cited answer. No official source could give you this."

> *Sources: [1][3] Alfonso Bastias — RateMyProfessors | [2][4][5] Qin Lv — RateMyProfessors*

---

**Query 3 — System struggles** *(15 sec)*
```
python generate.py "Is CSCI 5502 Data Mining considered a hard course?"
```
"This is the honest failure. Routes correctly to the review tier — confidence 0.56 — but the corpus only has CSCI 4502 and 1000 reviews, not 5502. The grounding rule kicks in and it says 'I don't have enough information.' That's a corpus gap, not a routing bug. The fix is adding 5502 reviews; the pipeline indexes them automatically."

> *Sources: [1] CSCI 4502 review — RateMyProfessors | [2] CSCI 1000 review — RateMyProfessors*

---

## Eval Repo Walkthrough *(~10 sec)*

*[Quick scroll through repo in VS Code.]*

"`chunk.py` — three chunk types. `search.py` — semantic router + pre-filter. `generate.py` — grounding prompt + Groq. `README.md` evaluation section has the before/after routing table — Q2, Q3, Q4 all went from mis-routed non-answers to correct-tier responses after the router change."

---

## Slide 11 — Takeaways *(~5 sec)*

"Three-tier chunking, semantic routing, and strict grounding are what make this work. Thanks."
