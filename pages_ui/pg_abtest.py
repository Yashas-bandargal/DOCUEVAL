"""
pages_ui/pg_abtest.py
---------------------
A/B test results page.
Compares chunk_size=300 vs chunk_size=500 configurations.
"""

import streamlit as st
import json
import os


def _load():
    path = "results/ab_test_results.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def render():
    st.markdown("""
    <div style='margin-bottom:24px;'>
      <div style='font-family:"DM Mono",monospace;font-size:20px;font-weight:500;color:#d0d0f4;'>
        A/B Test — Chunk Size Comparison</div>
      <div style='font-size:13px;color:#50508a;margin-top:4px;'>
        Does splitting documents into smaller or larger chunks give better retrieval?
      </div>
    </div>""", unsafe_allow_html=True)

    with st.expander("Why A/B test chunk sizes?"):
        st.markdown("""
        <div style='font-size:13px;color:#7070a0;line-height:1.8;padding:8px 0;'>
        <b style='color:#9090c0;'>Smaller chunks (300 chars)</b> → Higher precision.
        Each chunk is more focused on one topic, so fewer irrelevant chunks come back.<br><br>
        <b style='color:#9090c0;'>Larger chunks (500 chars)</b> → Higher recall.
        More context per chunk means less chance of an answer being split across two chunks
        that both get missed.<br><br>
        The right choice depends on your specific corpus. This A/B test measures it objectively
        instead of guessing.
        </div>""", unsafe_allow_html=True)

    data = _load()

    if not data:
        st.warning("No A/B test results found. Run `python src/ab_test.py` first.")
        st.info("⏱ This re-ingests the corpus twice — takes 2-3 minutes.")
        if st.button("Run A/B Test Now"):
            with st.spinner("Running ab_test.py... re-ingesting twice..."):
                import subprocess
                result = subprocess.run(
                    ["python", "src/ab_test.py"],
                    capture_output=True, text=True, cwd=os.getcwd()
                )
                if result.returncode == 0:
                    st.success("Done! Refresh the page.")
                    st.code(result.stdout)
                else:
                    st.error("Error:")
                    st.code(result.stderr[-1500:])
        return

    cfg_a = data.get("config_A", {})
    cfg_b = data.get("config_B", {})
    scores_a = cfg_a.get("scores", {})
    scores_b = cfg_b.get("scores", {})

    # ── Config overview ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Configurations compared</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid #1d9e75;text-align:left;padding:16px 20px;">
          <div style='font-size:10px;color:#1d9e75;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:8px;'>Config A</div>
          <div style='font-size:15px;color:#d0d0f0;font-weight:500;margin-bottom:4px;'>
            {cfg_a.get('label','Config A')}</div>
          <div style='font-size:12px;color:#50508a;'>
            chunk_size = {cfg_a.get('chunk_size', '—')} &nbsp;·&nbsp;
            K = {cfg_a.get('k', '—')}
          </div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid #3d3d9e;text-align:left;padding:16px 20px;">
          <div style='font-size:10px;color:#3d3d9e;text-transform:uppercase;
                      letter-spacing:1px;margin-bottom:8px;'>Config B</div>
          <div style='font-size:15px;color:#d0d0f0;font-weight:500;margin-bottom:4px;'>
            {cfg_b.get('label','Config B')}</div>
          <div style='font-size:12px;color:#50508a;'>
            chunk_size = {cfg_b.get('chunk_size', '—')} &nbsp;·&nbsp;
            K = {cfg_b.get('k', '—')}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Head to head comparison ────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Head-to-head comparison</div>', unsafe_allow_html=True)

    metrics = [
        ("Recall", "avg_recall", "#1d9e75"),
        ("MRR", "avg_mrr", "#7a9e3d"),
        ("Hit Rate", "avg_hit_rate", "#9e7a1d"),
    ]

    winners = []
    for label, key, _ in metrics:
        a_val = scores_a.get(key, 0)
        b_val = scores_b.get(key, 0)
        winners.append("A" if a_val >= b_val else "B")

    for label, key, color in metrics:
        a_val = scores_a.get(key, 0)
        b_val = scores_b.get(key, 0)
        winner = "A" if a_val >= b_val else "B"
        w_color = "#1d9e75" if winner == "A" else "#3d3d9e"

        st.markdown(f"""
        <div style='background:#0d0d1c;border:1px solid #1a1a30;border-radius:10px;
                    padding:16px 20px;margin-bottom:10px;'>
          <div style='display:flex;align-items:center;justify-content:space-between;
                      margin-bottom:12px;'>
            <div style='font-size:13px;color:#9090c0;font-weight:500;'>{label}</div>
            <div style='font-size:11px;color:{w_color};font-family:"DM Mono",monospace;'>
              Config {winner} wins
            </div>
          </div>
          <div style='display:flex;align-items:center;gap:12px;'>
            <div style='font-size:11px;color:#1d9e75;width:60px;'>Config A</div>
            <div style='flex:1;background:#12122a;border-radius:4px;height:8px;'>
              <div style='width:{int(a_val*100)}%;height:8px;background:#1d9e75;
                          border-radius:4px;'></div>
            </div>
            <div style='font-family:"DM Mono",monospace;font-size:13px;
                        color:#d0d0f0;min-width:50px;text-align:right;'>{a_val:.4f}</div>
          </div>
          <div style='display:flex;align-items:center;gap:12px;margin-top:8px;'>
            <div style='font-size:11px;color:#3d3d9e;width:60px;'>Config B</div>
            <div style='flex:1;background:#12122a;border-radius:4px;height:8px;'>
              <div style='width:{int(b_val*100)}%;height:8px;background:#3d3d9e;
                          border-radius:4px;'></div>
            </div>
            <div style='font-family:"DM Mono",monospace;font-size:13px;
                        color:#d0d0f0;min-width:50px;text-align:right;'>{b_val:.4f}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Verdict ────────────────────────────────────────────────────────────
    a_wins = winners.count("A")
    b_wins = winners.count("B")
    overall_winner = "Config A" if a_wins >= b_wins else "Config B"
    winner_label = cfg_a.get("label") if overall_winner == "Config A" else cfg_b.get("label")
    winner_color = "#1d9e75" if overall_winner == "Config A" else "#3d3d9e"

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:#070d0b;border:1px solid {winner_color}30;
                border-left:4px solid {winner_color};border-radius:10px;
                padding:18px 22px;'>
      <div style='font-size:12px;color:{winner_color};text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:6px;'>Recommended configuration</div>
      <div style='font-size:16px;color:#d0d0f0;font-weight:500;'>{overall_winner}: {winner_label}</div>
      <div style='font-size:12px;color:#506050;margin-top:6px;'>
        Won {max(a_wins, b_wins)}/3 metrics. Use this configuration in production.
      </div>
    </div>""", unsafe_allow_html=True)
