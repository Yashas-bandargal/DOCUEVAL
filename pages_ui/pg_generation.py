"""
pages_ui/pg_generation.py
--------------------------
Generation evaluation results page.
Shows Faithfulness, Answer Relevance, ROUGE-L per query.
"""

import streamlit as st
import json
import os
import pandas as pd


def _load():
    path = "results/generation_scores.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def render():
    st.markdown("""
    <div style='margin-bottom:24px;'>
      <div style='font-family:"DM Mono",monospace;font-size:20px;font-weight:500;color:#d0d0f4;'>
        Generation Evaluation</div>
      <div style='font-size:13px;color:#50508a;margin-top:4px;'>
        Was the answer faithful, relevant, and accurate? LLM-as-judge + ROUGE-L.
      </div>
    </div>""", unsafe_allow_html=True)

    data = _load()

    if not data:
        st.warning("No generation results found. Run `python src/eval_generation.py` first.")
        st.info("⏱ This takes ~10 minutes due to API rate limits. The script runs 3 API calls per query.")
        if st.button("Run Generation Evaluation Now"):
            with st.spinner("Running eval_generation.py... this will take ~10 minutes."):
                import subprocess
                result = subprocess.run(
                    ["python", "src/eval_generation.py"],
                    capture_output=True, text=True, cwd=os.getcwd()
                )
                if result.returncode == 0:
                    st.success("Done! Refresh the page.")
                else:
                    st.error("Error occurred:")
                    st.code(result.stderr[-2000:])
        return

    summary = data["summary"]
    per_query = data.get("per_query", [])

    # ── Summary cards ──────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Summary scores</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    faith = summary.get("avg_faithfulness", 0)
    relev = summary.get("avg_answer_relevance", 0)
    rouge = summary.get("avg_rouge_l", 0)
    pass_rate = summary.get("faithfulness_pass_rate", 0)

    c1.markdown(f"""
    <div class="metric-card">
      <span class="val" style="color:{'#3ddc84' if faith>=0.8 else '#ff6b6b'};">{faith:.4f}</span>
      <span class="lbl">Faithfulness</span>
      <span class="sublbl">{'✓ PASS' if faith>=0.8 else '✗ BELOW 0.80'}</span>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card">
      <span class="val" style="color:{'#3ddc84' if relev>=3.5 else '#ffa040'};">{relev:.2f}/5</span>
      <span class="lbl">Answer Relevance</span>
      <span class="sublbl">{'✓ PASS' if relev>=3.5 else '✗ BELOW 3.5'}</span>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-card">
      <span class="val" style="color:{'#3ddc84' if rouge>=0.25 else '#ffa040'};">{rouge:.4f}</span>
      <span class="lbl">ROUGE-L</span>
      <span class="sublbl">{'✓ PASS' if rouge>=0.25 else '✗ BELOW 0.25'}</span>
    </div>""", unsafe_allow_html=True)

    c4.markdown(f"""
    <div class="metric-card">
      <span class="val" style="color:#9e5dc0;">{pass_rate*100:.0f}%</span>
      <span class="lbl">Faithfulness pass %</span>
      <span class="sublbl">{summary.get('num_queries',0)} queries evaluated</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Per-query detail ───────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Per-query results</div>', unsafe_allow_html=True)

    if per_query:
        # Table
        rows = []
        for q in per_query:
            sc = q["scores"]
            rows.append({
                "ID": q["query_id"],
                "Query": q["query"][:55] + "..." if len(q["query"]) > 55 else q["query"],
                "Faithful": sc.get("faithfulness", 0),
                "Relevance": sc.get("answer_relevance", 0),
                "ROUGE-L": round(sc.get("rouge_l", 0), 4),
            })
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Faithful": st.column_config.ProgressColumn("Faithful", min_value=0, max_value=1, format="%.0f"),
                "Relevance": st.column_config.ProgressColumn("Relevance", min_value=0, max_value=5, format="%.0f"),
                "ROUGE-L": st.column_config.NumberColumn("ROUGE-L", format="%.4f"),
            }
        )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Query inspector
        st.markdown('<div class="section-hdr">Query inspector</div>', unsafe_allow_html=True)
        query_ids = [q["query_id"] for q in per_query]
        selected = st.selectbox("Select a query to inspect", query_ids)

        q_data = next((q for q in per_query if q["query_id"] == selected), None)
        if q_data:
            sc = q_data["scores"]
            faith_val = sc.get("faithfulness", 0)
            rel_val = sc.get("answer_relevance", 0)
            rouge_val = sc.get("rouge_l", 0)

            col1, col2, col3 = st.columns(3)
            col1.markdown(f"""
            <div class="metric-card">
              <span class="val" style="color:{'#3ddc84' if faith_val==1 else '#ff6b6b'};">
                {'Faithful ✓' if faith_val==1 else 'Hallucinated ✗'}</span>
              <span class="lbl">Faithfulness</span>
            </div>""", unsafe_allow_html=True)
            col2.markdown(f"""
            <div class="metric-card">
              <span class="val" style="color:#9e5dc0;">{rel_val}/5</span>
              <span class="lbl">Answer Relevance</span>
            </div>""", unsafe_allow_html=True)
            col3.markdown(f"""
            <div class="metric-card">
              <span class="val" style="color:#9e7a1d;">{rouge_val:.4f}</span>
              <span class="lbl">ROUGE-L</span>
            </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            c_l, c_r = st.columns(2)
            with c_l:
                st.markdown("<div style='font-size:11px;color:#36367a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Expected answer</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background:#0a0a16;border:1px solid #1a1a30;border-left:3px solid #3d3d9e;
                            border-radius:8px;padding:14px 16px;color:#9090c0;font-size:13px;
                            line-height:1.7;min-height:80px;'>
                  {q_data.get('expected_answer','—')}
                </div>""", unsafe_allow_html=True)
            with c_r:
                st.markdown("<div style='font-size:11px;color:#36367a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Generated answer</div>", unsafe_allow_html=True)
                border = "#3ddc84" if faith_val == 1 else "#ff6b6b"
                st.markdown(f"""
                <div style='background:#070d0b;border:1px solid #16302a;border-left:3px solid {border};
                            border-radius:8px;padding:14px 16px;color:#90c0a0;font-size:13px;
                            line-height:1.7;min-height:80px;'>
                  {q_data.get('generated_answer','—')}
                </div>""", unsafe_allow_html=True)
