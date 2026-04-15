"""
pages_ui/pg_overview.py
-----------------------
Overview / dashboard page — shows project summary + eval scores if available.
"""

import streamlit as st
import json
import os


def _load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _metric_card(val, label, sub="", color="#3ddc84"):
    return f"""
    <div class="metric-card">
        <span class="val" style="color:{color}">{val}</span>
        <span class="lbl">{label}</span>
        <span class="sublbl">{sub}</span>
    </div>"""


def render():
    st.markdown("""
    <div style='margin-bottom:28px;'>
      <div style='font-family:"DM Mono",monospace;font-size:26px;font-weight:500;
                  color:#d0d0f4;letter-spacing:-1px;'>DocuEval</div>
      <div style='font-size:13px;color:#50508a;margin-top:6px;line-height:1.6;max-width:600px;'>
        A RAG pipeline over company policy documents with a complete evaluation suite.
        Measures retrieval quality, generation faithfulness, and hallucination behavior.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline overview ──────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Pipeline stages</div>', unsafe_allow_html=True)

    stages = [
        ("01", "Ingest", "Chunk + embed docs", "#1d9e75"),
        ("02", "Retrieve", "Top-K from ChromaDB", "#3d3d9e"),
        ("03", "Generate", "Gemini 2.0 Flash", "#9e3d7a"),
        ("04", "Evaluate", "Score + report", "#9e7a1d"),
    ]
    cols = st.columns(4)
    for col, (num, title, sub, clr) in zip(cols, stages):
        col.markdown(f"""
        <div class="metric-card" style="border-left:3px solid {clr}20;text-align:left;padding:16px 18px;">
          <div style='font-family:"DM Mono",monospace;font-size:10px;color:{clr};
                      text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>STEP {num}</div>
          <div style='font-size:15px;font-weight:500;color:#d0d0f0;'>{title}</div>
          <div style='font-size:12px;color:#50508a;margin-top:3px;'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Live scores if available ───────────────────────────────────────────
    ret = _load_json("results/retrieval_scores.json")
    gen = _load_json("results/generation_scores.json")
    oos = _load_json("results/oos_results.json")

    if ret or gen or oos:
        st.markdown('<div class="section-hdr">Latest eval scores</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        if ret:
            r5 = ret["summary"].get("k=5", {})
            c1.markdown(_metric_card(
                f"{r5.get('avg_recall',0):.2f}", "Recall@5", "retrieval", "#1d9e75"
            ), unsafe_allow_html=True)
            c2.markdown(_metric_card(
                f"{r5.get('avg_mrr',0):.2f}", "MRR@5", "retrieval", "#3d9e7a"
            ), unsafe_allow_html=True)
            c3.markdown(_metric_card(
                f"{r5.get('avg_hit_rate',0):.2f}", "Hit Rate@5", "retrieval", "#5d9e5a"
            ), unsafe_allow_html=True)

        if gen:
            g = gen["summary"]
            c4.markdown(_metric_card(
                f"{g.get('avg_faithfulness',0):.2f}", "Faithfulness", "generation", "#9e3d9e"
            ), unsafe_allow_html=True)
            c5.markdown(_metric_card(
                f"{g.get('avg_answer_relevance',0):.1f}/5", "Relevance", "generation", "#7a3d9e"
            ), unsafe_allow_html=True)

        if oos:
            o = oos["summary"]
            c6.markdown(_metric_card(
                f"{o.get('refusal_rate',0):.0%}", "Refusal Rate", "OOS testing", "#9e7a1d"
            ), unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#0d0d1c;border:1px solid #1a1a30;border-radius:10px;
                    padding:24px;color:#404070;font-size:13px;text-align:center;margin:16px 0;'>
            No evaluation results yet.<br>
            <span style='font-family:"DM Mono",monospace;font-size:11px;color:#303060;'>
            Run eval scripts first, then refresh this page.
            </span>
        </div>""", unsafe_allow_html=True)

    # ── Documents in corpus ────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Document corpus</div>', unsafe_allow_html=True)
    docs_path = "data/raw"
    if os.path.exists(docs_path):
        files = [f for f in os.listdir(docs_path) if f.endswith(".txt")]
        if files:
            cols = st.columns(len(files))
            for col, fname in zip(cols, files):
                fpath = os.path.join(docs_path, fname)
                size = os.path.getsize(fpath)
                with open(fpath) as f:
                    lines = f.read().split("\n")
                col.markdown(f"""
                <div class="metric-card" style="text-align:left;padding:14px 16px;">
                  <div style='font-family:"DM Mono",monospace;font-size:11px;
                              color:#3d3d9e;margin-bottom:6px;'>📄</div>
                  <div style='font-size:12px;color:#b0b0d0;font-weight:500;
                              word-break:break-all;'>{fname}</div>
                  <div style='font-size:10px;color:#303060;margin-top:4px;'>
                    {size} bytes · {len(lines)} lines</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("data/raw/ folder not found. Make sure you run this from your docueval/ project root.")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Ground truth stats ────────────────────────────────────────────────
    gt = _load_json("data/ground_truth.json")
    if gt:
        st.markdown('<div class="section-hdr">Ground truth dataset</div>', unsafe_allow_html=True)
        in_scope = [q for q in gt if not q.get("out_of_scope")]
        oos_q = [q for q in gt if q.get("out_of_scope")]
        validated = [q for q in gt if q.get("validated")]

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(_metric_card(len(gt), "Total queries", "all types", "#5050c0"), unsafe_allow_html=True)
        c2.markdown(_metric_card(len(in_scope), "In-scope", "used for eval", "#1d9e75"), unsafe_allow_html=True)
        c3.markdown(_metric_card(len(oos_q), "Out-of-scope", "hallucination test", "#9e3d1d"), unsafe_allow_html=True)
        c4.markdown(_metric_card(len(validated), "Validated", "human-reviewed", "#9e9e1d"), unsafe_allow_html=True)
