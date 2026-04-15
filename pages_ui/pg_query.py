"""
pages_ui/pg_query.py
--------------------
Live query pipeline page.
Type a question → see retrieved chunks → see generated answer.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.getcwd())


def render():
    st.markdown("""
    <div style='margin-bottom:24px;'>
      <div style='font-family:"DM Mono",monospace;font-size:20px;font-weight:500;color:#d0d0f4;'>
        Query Pipeline</div>
      <div style='font-size:13px;color:#50508a;margin-top:4px;'>
        Ask a question. Watch the full RAG pipeline execute live.
      </div>
    </div>""", unsafe_allow_html=True)

    if not os.path.exists("chroma_store"):
        st.error("Vector store not found. Run `python src/ingest.py` first.")
        return

    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not set. Set it before running the app.")
        return

    # ── Query input ────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Your question</div>', unsafe_allow_html=True)

    example_queries = [
        "How many sick leave days do employees get per year?",
        "What encryption standard is used for confidential data?",
        "Can remote employees use public Wi-Fi?",
        "How many casual leave days are employees entitled to?",
        "What must vendors sign before accessing organizational data?",
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "query",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
    with col2:
        k = st.selectbox("Top-K chunks", [3, 5, 7], index=1, label_visibility="visible")

    st.markdown("<div style='margin-bottom:6px;font-size:11px;color:#36367a;'>Try an example:</div>",
                unsafe_allow_html=True)
    ex_cols = st.columns(len(example_queries))
    for col, eq in zip(ex_cols, example_queries):
        if col.button(eq[:35] + "...", key=f"ex_{eq[:10]}"):
            query = eq

    if not query:
        st.markdown("""
        <div style='background:#0d0d1c;border:1px dashed #1a1a30;border-radius:10px;
                    padding:32px;text-align:center;margin-top:20px;'>
          <div style='font-size:13px;color:#303060;'>Enter a question above to run the pipeline</div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Run pipeline ───────────────────────────────────────────────────────
    with st.spinner("Running RAG pipeline..."):
        try:
            from src.retrieve import retrieve
            chunks = retrieve(query, k=k)
        except Exception as e:
            st.error(f"Retrieval error: {e}")
            return

    # ── Show retrieved chunks ──────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-hdr">Retrieved chunks (top {k})</div>', unsafe_allow_html=True)

    for i, chunk in enumerate(chunks):
        similarity = 1 - chunk["distance"]
        bar_width = int(similarity * 100)
        color = "#1d9e75" if similarity > 0.6 else "#9e7a1d" if similarity > 0.4 else "#9e3d1d"
        st.markdown(f"""
        <div class="chunk-card">
          <div class="chunk-meta">
            RANK {i+1} &nbsp;·&nbsp; {chunk['source']} &nbsp;·&nbsp;
            similarity: {similarity:.4f}
            <span style='display:inline-block;width:80px;height:4px;
                         background:#14142a;border-radius:2px;margin-left:8px;
                         vertical-align:middle;'>
              <span style='display:block;width:{bar_width}%;height:4px;
                           background:{color};border-radius:2px;'></span>
            </span>
          </div>
          {chunk['text'][:400]}{'...' if len(chunk['text']) > 400 else ''}
        </div>""", unsafe_allow_html=True)

    # ── Generate answer ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Generated answer</div>', unsafe_allow_html=True)

    with st.spinner("Generating answer with Gemini..."):
        try:
            from src.generate import generate
            result = generate(query, chunks)
            answer = result["answer"]
        except Exception as e:
            st.error(f"Generation error: {e}")
            return

    is_oos = "don't have information" in answer.lower()
    border_color = "#9e3d1d" if is_oos else "#1d9e75"
    text_color = "#c07060" if is_oos else "#90c8b0"

    st.markdown(f"""
    <div style='background:#070d0b;border:1px solid #16302a;
                border-left:3px solid {border_color};border-radius:8px;
                padding:18px 22px;color:{text_color};
                font-size:14px;line-height:1.8;'>
      {'⚠️ Out of scope — ' if is_oos else ''}
      {answer}
    </div>""", unsafe_allow_html=True)

    # ── Stats ──────────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card">
        <span class="val" style="color:#3d3d9e;">{k}</span>
        <span class="lbl">chunks retrieved</span></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card">
        <span class="val" style="color:#1d9e75;">{result['chunks_used']}</span>
        <span class="lbl">chunks in context</span></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card">
        <span class="val" style="color:#9e7a1d;">{len(answer.split())}</span>
        <span class="lbl">words in answer</span></div>""", unsafe_allow_html=True)
