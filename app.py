"""Streamlit chat interface for The Unofficial Guide RAG assistant.

Run with:
    streamlit run app.py

Requires the index to be built first:
    python index.py
"""

from __future__ import annotations

import streamlit as st

from config import GROQ_MODEL, TOP_K
from generate import _attribution, generate_answer_stream, retrieve

st.set_page_config(
    page_title="The Unofficial Guide",
    page_icon="🎓",
    layout="centered",
)

st.title("The Unofficial Guide")
st.caption("Student reviews & curriculum Q&A for CU Boulder Data Science")


def _render_sources(chunks: list[dict]) -> None:
    """Render a collapsible list of the retrieved source chunks."""
    if not chunks:
        return
    with st.expander(f"Sources ({len(chunks)})"):
        for i, c in enumerate(chunks, start=1):
            score = 1 - c["distance"]
            st.markdown(f"**[{i}]** `score={score:.3f}` — {_attribution(c['metadata'])}")
            preview = c["text"].strip()
            if len(preview) > 400:
                preview = preview[:400] + "..."
            st.caption(preview)


with st.sidebar:
    st.header("Settings")
    st.markdown(f"**Model:** `{GROQ_MODEL}`")
    st.markdown(f"**Top-k chunks:** `{TOP_K}`")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("chunks"):
            _render_sources(msg["chunks"])

prompt = st.chat_input("Ask about a professor, course, or the MS-DS curriculum...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # History passed to the model excludes the message we just appended.
        history = st.session_state.messages[:-1]
        try:
            chunks = retrieve(prompt, top_k=TOP_K)
            answer = st.write_stream(
                generate_answer_stream(prompt, history=history, chunks=chunks)
            )
            _render_sources(chunks)
        except Exception as exc:  # noqa: BLE001 - show a friendly error in the UI
            answer = (
                f"Sorry, something went wrong: {exc}\n\n"
                "Make sure the index is built (`python index.py`) and that "
                "`GROQ_API_KEY` is set in your `.env`."
            )
            chunks = []
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks}
    )
