"""
pages_ui/pg_regression.py
--------------------------
Regression test suite page.
Runs pytest and shows pass/fail per test with score thresholds.
"""

import streamlit as st
import json
import os
import subprocess


def _load_scores():
    ret = gen = oos = None
    if os.path.exists("results/retrieval_scores.json"):
        with open("results/retrieval_scores.json") as f:
            ret = json.load(f)
    if os.path.exists("results/generation_scores.json"):
        with open("results/generation_scores.json") as f:
            gen = json.load(f)
    if os.path.exists("results/oos_results.json"):
        with open("results/oos_results.json") as f:
            oos = json.load(f)
    return ret, gen, oos


THRESHOLDS = {
    "Recall@5": ("retrieval", 0.65),
    "MRR@5": ("retrieval", 0.55),
    "Hit Rate@5": ("retrieval", 0.70),
    "Faithfulness": ("generation", 0.75),
    "Answer Relevance": ("generation", 3.0),
    "ROUGE-L": ("generation", 0.20),
    "OOS Refusal Rate": ("oos", 0.60),
}


def render():
    st.markdown("""
    <div style='margin-bottom:24px;'>
      <div style='font-family:"DM Mono",monospace;font-size:20px;font-weight:500;color:#d0d0f4;'>
        Regression Tests</div>
      <div style='font-size:13px;color:#50508a;margin-top:4px;'>
        Quality gates that fail automatically if metrics drop below thresholds.
      </div>
    </div>""", unsafe_allow_html=True)

    with st.expander("What are regression tests?"):
        st.markdown("""
        <div style='font-size:13px;color:#7070a0;line-height:1.8;padding:8px 0;'>
        Regression tests assert that evaluation scores stay above defined thresholds.
        If you change your chunking strategy, prompt, or embedding model — run these tests
        to verify that quality has NOT degraded.<br><br>
        The tests read from <code>results/*.json</code> and check each metric against
        a minimum threshold. If a metric drops below threshold, the test fails with a
        clear message explaining what dropped and where to look.<br><br>
        These same tests run automatically on every GitHub push via the CI/CD workflow.
        </div>""", unsafe_allow_html=True)

    ret, gen, oos = _load_scores()

    # ── Manual threshold check ─────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Threshold checks</div>', unsafe_allow_html=True)

    checks = []
    if ret:
        r5 = ret["summary"].get("k=5", {})
        checks.append(("Recall@5", r5.get("avg_recall", 0), 0.65, "retrieval"))
        checks.append(("MRR@5", r5.get("avg_mrr", 0), 0.55, "retrieval"))
        checks.append(("Hit Rate@5", r5.get("avg_hit_rate", 0), 0.70, "retrieval"))
    if gen:
        g = gen["summary"]
        checks.append(("Faithfulness", g.get("avg_faithfulness", 0), 0.75, "generation"))
        checks.append(("Answer Relevance", g.get("avg_answer_relevance", 0), 3.0, "generation"))
        checks.append(("ROUGE-L", g.get("avg_rouge_l", 0), 0.20, "generation"))
    if oos:
        o = oos["summary"]
        checks.append(("OOS Refusal Rate", o.get("refusal_rate", 0), 0.60, "oos"))

    if not checks:
        st.info("No evaluation results found. Run all eval scripts first, then come back here.")
        return

    passed = sum(1 for _, val, thresh, _ in checks if val >= thresh)
    total = len(checks)

    # Overall status banner
    all_pass = passed == total
    banner_bg = "#070d07" if all_pass else "#0d0707"
    banner_border = "#1d9e75" if all_pass else "#9e1d1d"
    banner_color = "#3ddc84" if all_pass else "#ff6b6b"
    banner_text = f"ALL {total} TESTS PASSING ✓" if all_pass else f"{passed}/{total} TESTS PASSING — {total-passed} FAILURES"

    st.markdown(f"""
    <div style='background:{banner_bg};border:1px solid {banner_border};
                border-radius:10px;padding:16px 22px;margin-bottom:18px;
                font-family:"DM Mono",monospace;font-size:14px;color:{banner_color};
                letter-spacing:0.5px;'>
      {banner_text}
    </div>""", unsafe_allow_html=True)

    # Individual checks
    category_colors = {
        "retrieval": "#1d9e75",
        "generation": "#9e3d9e",
        "oos": "#9e7a1d"
    }
    for metric, val, thresh, category in checks:
        passed_check = val >= thresh
        icon = "✓" if passed_check else "✗"
        color = "#3ddc84" if passed_check else "#ff6b6b"
        cat_color = category_colors.get(category, "#5050c0")

        st.markdown(f"""
        <div style='display:flex;align-items:center;padding:12px 16px;
                    background:#0d0d1c;border:1px solid #1a1a30;border-radius:8px;
                    margin-bottom:6px;gap:14px;'>
          <div style='font-size:14px;color:{color};font-family:"DM Mono",monospace;
                      min-width:20px;'>{icon}</div>
          <div style='font-size:12px;color:#7070a0;min-width:160px;'>{metric}</div>
          <div style='flex:1;background:#12122a;border-radius:3px;height:5px;'>
            <div style='width:{min(int(val/max(thresh,0.01)*100*0.8),100)}%;height:5px;
                        background:{color};border-radius:3px;opacity:0.7;'></div>
          </div>
          <div style='font-family:"DM Mono",monospace;font-size:13px;color:#d0d0f0;
                      min-width:58px;text-align:right;'>{val:.4f}</div>
          <div style='font-size:11px;color:#303050;min-width:80px;text-align:right;'>
            min: {thresh}</div>
          <div style='font-size:10px;color:{cat_color};min-width:70px;text-align:right;
                      text-transform:uppercase;letter-spacing:0.5px;'>{category}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Run pytest ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Run pytest suite</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:12px;color:#404070;margin-bottom:12px;line-height:1.7;'>
    This runs <code>pytest src/test_regression.py -v</code> — the actual test file
    with assertions. Same tests that run in GitHub Actions CI/CD.
    </div>""", unsafe_allow_html=True)

    if st.button("▶  Run pytest now"):
        with st.spinner("Running pytest..."):
            result = subprocess.run(
                ["python", "-m", "pytest", "src/test_regression.py", "-v", "--tb=short"],
                capture_output=True, text=True, cwd=os.getcwd()
            )

        output = result.stdout + result.stderr
        lines = output.split("\n")

        formatted = []
        for line in lines:
            if "PASSED" in line:
                formatted.append(f'<div style="color:#3ddc84;font-family:\'DM Mono\',monospace;font-size:11px;padding:1px 0;">{line}</div>')
            elif "FAILED" in line or "ERROR" in line:
                formatted.append(f'<div style="color:#ff6b6b;font-family:\'DM Mono\',monospace;font-size:11px;padding:1px 0;">{line}</div>')
            elif line.startswith("="):
                formatted.append(f'<div style="color:#5050c0;font-family:\'DM Mono\',monospace;font-size:11px;padding:3px 0;">{line}</div>')
            else:
                formatted.append(f'<div style="color:#505070;font-family:\'DM Mono\',monospace;font-size:11px;padding:1px 0;">{line}</div>')

        st.markdown(f"""
        <div style='background:#050508;border:1px solid #14142a;border-radius:8px;
                    padding:16px 18px;max-height:400px;overflow-y:auto;'>
          {''.join(formatted)}
        </div>""", unsafe_allow_html=True)

        if result.returncode == 0:
            st.success("All tests passed ✓")
        else:
            st.error("Some tests failed. Fix the issues above and re-run eval scripts.")

    # ── CI/CD info ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0a080d;border:1px solid #1e1430;border-left:3px solid #5050c0;
                border-radius:8px;padding:14px 18px;'>
      <div style='font-size:11px;color:#5050c0;text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:6px;'>GitHub Actions CI/CD</div>
      <div style='font-size:12px;color:#504060;line-height:1.7;'>
        Every push to <code>main</code> automatically runs the full eval pipeline
        and these pytest tests via <code>.github/workflows/eval.yml</code>.
        A green checkmark on your GitHub repo = all quality gates passing.
      </div>
    </div>""", unsafe_allow_html=True)
