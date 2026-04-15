"""
pages_ui/pg_oos.py
------------------
Out-of-scope testing page.
Shows hallucination detection results on unanswerable queries.
"""

import streamlit as st
import json
import os


def _load():
    path = "results/oos_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def render():
    st.markdown("""
    <div style='margin-bottom:24px;'>
      <div style='font-family:"DM Mono",monospace;font-size:20px;font-weight:500;color:#d0d0f4;'>
        Out-of-Scope Testing</div>
      <div style='font-size:13px;color:#50508a;margin-top:4px;'>
        Does the system hallucinate when it has no answer, or does it correctly refuse?
      </div>
    </div>""", unsafe_allow_html=True)

    # Concept explanation
    with st.expander("What is out-of-scope testing?"):
        st.markdown("""
        <div style='font-size:13px;color:#7070a0;line-height:1.8;padding:8px 0;'>
        Out-of-scope queries are questions whose answers do NOT exist in the knowledge base.
        A well-built RAG system should say <code>"I don't have information about this"</code>
        instead of confidently making something up (hallucination).<br><br>
        This test reveals a critical failure mode. If a user asks about dress code policy
        and your system invents an answer — that's a trust-destroying bug.
        </div>""", unsafe_allow_html=True)

    data = _load()

    if not data:
        st.warning("No OOS results found. Run `python src/eval_oos.py` first.")
        if st.button("Run OOS Evaluation Now"):
            with st.spinner("Running eval_oos.py..."):
                import subprocess
                result = subprocess.run(
                    ["python", "src/eval_oos.py"],
                    capture_output=True, text=True, cwd=os.getcwd()
                )
                if result.returncode == 0:
                    st.success("Done! Refresh the page.")
                    st.code(result.stdout)
                else:
                    st.error("Error:")
                    st.code(result.stderr[-1500:])
        return

    summary = data["summary"]
    per_query = data.get("per_query", [])

    # ── Summary ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Summary</div>', unsafe_allow_html=True)

    total = summary.get("total_oos_queries", 0)
    correct = summary.get("correct_refusals", 0)
    hallucinations = summary.get("hallucinations", 0)
    rate = summary.get("refusal_rate", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
    <div class="metric-card">
      <span class="val" style="color:#5050c0;">{total}</span>
      <span class="lbl">Total OOS queries</span>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""
    <div class="metric-card">
      <span class="val" style="color:#3ddc84;">{correct}</span>
      <span class="lbl">Correct refusals</span>
      <span class="sublbl">system said "I don't know"</span>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""
    <div class="metric-card">
      <span class="val" style="color:{'#ff6b6b' if hallucinations>0 else '#3ddc84'};">{hallucinations}</span>
      <span class="lbl">Hallucinations</span>
      <span class="sublbl">invented an answer</span>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""
    <div class="metric-card">
      <span class="val" style="color:{'#3ddc84' if rate>=0.6 else '#ff6b6b'};">{rate*100:.0f}%</span>
      <span class="lbl">Refusal rate</span>
      <span class="sublbl">{'✓ PASS' if rate>=0.6 else '✗ NEEDS IMPROVEMENT'}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Per-query results ──────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Per-query results</div>', unsafe_allow_html=True)

    for q in per_query:
        refused = q.get("correctly_refused", False)
        icon = "✓" if refused else "✗"
        bg = "#070d07" if refused else "#0d0707"
        border = "#1d9e75" if refused else "#9e1d1d"
        text_color = "#60c080" if refused else "#c06060"
        status_text = "Correctly refused" if refused else "HALLUCINATED"

        st.markdown(f"""
        <div style='background:{bg};border:1px solid {border}20;border-left:3px solid {border};
                    border-radius:8px;padding:14px 18px;margin-bottom:10px;'>
          <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;'>
            <div style='font-family:"DM Mono",monospace;font-size:11px;color:#404060;'>
              {q['query_id']}
            </div>
            <div style='font-size:11px;color:{text_color};font-weight:500;'>
              {icon} {status_text}
            </div>
          </div>
          <div style='font-size:13px;color:#9090b0;margin-bottom:8px;font-weight:500;'>
            Q: {q['query']}
          </div>
          <div style='font-size:12px;color:{text_color};font-family:"DM Mono",monospace;
                      line-height:1.6;opacity:0.85;'>
            A: {q.get('generated_answer','—')[:300]}
          </div>
        </div>""", unsafe_allow_html=True)

    # ── What to do if refusal rate is low ─────────────────────────────────
    if rate < 0.6:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#1a0a08;border:1px solid #40200a;border-radius:8px;padding:16px 20px;'>
          <div style='font-size:12px;color:#ffa040;font-weight:500;margin-bottom:8px;'>
            ⚠️ Refusal rate is too low — system is hallucinating on out-of-scope queries
          </div>
          <div style='font-size:12px;color:#705030;line-height:1.7;'>
            Fix: Strengthen the prompt in <code>src/generate.py</code>. Make the instruction
            more explicit — add "Under no circumstances should you answer from outside knowledge."
            Then re-run eval_oos.py.
          </div>
        </div>""", unsafe_allow_html=True)
