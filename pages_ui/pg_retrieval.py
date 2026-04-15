"""
pages_ui/pg_retrieval.py
------------------------
Retrieval evaluation results page.
Shows Recall@K, Precision@K, MRR, Hit Rate with per-query breakdown.
"""

import streamlit as st
import json
import os
import pandas as pd


def _load():
    path = "results/retrieval_scores.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _bar(val, max_val=1.0, color="#1d9e75"):
    pct = int((val / max_val) * 100)
    return f"""
    <div style='display:flex;align-items:center;gap:10px;'>
      <div style='flex:1;background:#14142a;border-radius:3px;height:5px;'>
        <div style='width:{pct}%;height:5px;background:{color};border-radius:3px;'></div>
      </div>
      <div style='font-family:"DM Mono",monospace;font-size:12px;
                  color:#d0d0f0;min-width:40px;text-align:right;'>{val:.4f}</div>
    </div>"""


def render():
    st.markdown("""
    <div style='margin-bottom:24px;'>
      <div style='font-family:"DM Mono",monospace;font-size:20px;font-weight:500;color:#d0d0f4;'>
        Retrieval Evaluation</div>
      <div style='font-size:13px;color:#50508a;margin-top:4px;'>
        Did the right documents come back? Recall@K · Precision@K · MRR · Hit Rate
      </div>
    </div>""", unsafe_allow_html=True)

    data = _load()

    if not data:
        st.warning("No retrieval results found. Run `python src/eval_retrieval.py` first.")
        if st.button("Run Retrieval Evaluation Now"):
            with st.spinner("Running eval_retrieval.py..."):
                import subprocess
                result = subprocess.run(
                    ["python", "src/eval_retrieval.py"],
                    capture_output=True, text=True, cwd=os.getcwd()
                )
                if result.returncode == 0:
                    st.success("Done! Refresh the page.")
                    st.code(result.stdout[-1500:])
                else:
                    st.error("Error:")
                    st.code(result.stderr[-1500:])
        return

    summary = data["summary"]

    # ── Summary metric cards ───────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Summary scores</div>', unsafe_allow_html=True)

    tabs = st.tabs(["K = 3", "K = 5"])
    for tab, k_key, k_label in zip(tabs, ["k=3", "k=5"], ["K = 3", "K = 5"]):
        with tab:
            s = summary.get(k_key, {})
            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                (c1, s.get("avg_recall", 0), f"Recall@{k_key[-1]}", "#1d9e75", 0.65),
                (c2, s.get("avg_precision", 0), f"Precision@{k_key[-1]}", "#3d9e7a", 0.30),
                (c3, s.get("avg_mrr", 0), f"MRR@{k_key[-1]}", "#7a9e3d", 0.55),
                (c4, s.get("avg_hit_rate", 0), f"Hit Rate@{k_key[-1]}", "#9e7a1d", 0.70),
            ]
            for col, val, label, color, threshold in metrics:
                status = "✓" if val >= threshold else "✗"
                status_color = "#3ddc84" if val >= threshold else "#ff6b6b"
                col.markdown(f"""
                <div class="metric-card">
                  <span class="val" style="color:{color};">{val:.4f}</span>
                  <span class="lbl">{label}</span>
                  <span class="sublbl" style="color:{status_color};">
                    {status} threshold: {threshold}
                  </span>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Score breakdown bars ───────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Score breakdown — K=5</div>', unsafe_allow_html=True)

    s5 = summary.get("k=5", {})
    metrics_bar = [
        ("Recall@5", s5.get("avg_recall", 0), "#1d9e75"),
        ("Precision@5", s5.get("avg_precision", 0), "#3d9e7a"),
        ("MRR@5", s5.get("avg_mrr", 0), "#7a9e3d"),
        ("Hit Rate@5", s5.get("avg_hit_rate", 0), "#9e7a1d"),
    ]
    for label, val, color in metrics_bar:
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:14px;padding:10px 0;
                    border-bottom:1px solid #14142a;'>
          <div style='font-size:12px;color:#60608a;width:120px;flex-shrink:0;'>{label}</div>
          <div style='flex:1;background:#12122a;border-radius:4px;height:7px;'>
            <div style='width:{int(val*100)}%;height:7px;background:{color};
                        border-radius:4px;transition:width 0.5s;'></div>
          </div>
          <div style='font-family:"DM Mono",monospace;font-size:13px;
                      color:#d0d0f0;min-width:52px;text-align:right;'>{val:.4f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Per-query table ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Per-query breakdown</div>', unsafe_allow_html=True)

    per_query = data.get("per_query", [])
    if per_query:
        rows = []
        for q in per_query:
            s5q = q["scores"].get("k=5", {})
            rows.append({
                "ID": q["query_id"],
                "Query": q["query"][:60] + "..." if len(q["query"]) > 60 else q["query"],
                "Expected source": q["expected_source"],
                "Recall@5": s5q.get("recall", 0),
                "Precision@5": round(s5q.get("precision", 0), 3),
                "MRR@5": round(s5q.get("reciprocal_rank", 0), 3),
                "Hit@5": s5q.get("hit_rate", 0),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Recall@5": st.column_config.ProgressColumn("Recall@5", min_value=0, max_value=1, format="%.0f"),
                "Precision@5": st.column_config.NumberColumn("Precision@5", format="%.3f"),
                "MRR@5": st.column_config.NumberColumn("MRR@5", format="%.3f"),
                "Hit@5": st.column_config.ProgressColumn("Hit@5", min_value=0, max_value=1, format="%.0f"),
            }
        )

        # Failed queries
        failed = [q for q in per_query if q["scores"].get("k=5", {}).get("hit_rate", 0) == 0]
        if failed:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            with st.expander(f"⚠️ {len(failed)} queries with 0 Hit Rate — investigate these"):
                for q in failed:
                    st.markdown(f"""
                    <div style='padding:10px;border-bottom:1px solid #14142a;'>
                      <div style='font-size:12px;color:#ff6b6b;font-family:"DM Mono",monospace;
                                  margin-bottom:4px;'>{q['query_id']}</div>
                      <div style='font-size:13px;color:#9090b0;'>{q['query']}</div>
                      <div style='font-size:11px;color:#404060;margin-top:3px;'>
                        Expected: {q['expected_source']} |
                        Got: {', '.join(set(q.get('retrieved_sources', [])))}</div>
                    </div>""", unsafe_allow_html=True)
