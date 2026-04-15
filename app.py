"""
app.py
------
DocuEval — RAG Pipeline + Evaluation Suite
Streamlit UI entry point.

Run from your docueval/ project root:
    streamlit run app.py
"""

import streamlit as st
import os

st.set_page_config(
    page_title="DocuEval",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

[data-testid="stSidebar"] { background: #07070f; border-right: 1px solid #16162a; }
[data-testid="stSidebar"] * { color: #b0b0cc !important; }

.main .block-container { background: #07070f; padding-top: 1.5rem; max-width: 1100px; }

.metric-card {
    background: #0d0d1c; border: 1px solid #1a1a30;
    border-radius: 12px; padding: 22px 16px; text-align: center;
}
.metric-card .val {
    font-family: 'DM Mono', monospace; font-size: 30px;
    font-weight: 500; color: #e0e0f4; display: block;
}
.metric-card .lbl {
    font-size: 11px; color: #50508a; text-transform: uppercase;
    letter-spacing: 1.2px; margin-top: 6px; display: block;
}
.metric-card .sublbl {
    font-size: 10px; color: #30304a; margin-top: 2px; display: block;
}

.section-hdr {
    font-size: 10px; font-weight: 600; color: #3a3a7a;
    text-transform: uppercase; letter-spacing: 2px;
    padding-bottom: 8px; border-bottom: 1px solid #14142a;
    margin-bottom: 14px;
}

.chunk-card {
    background: #0a0a16; border: 1px solid #18183a;
    border-left: 3px solid #3a3a9e; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 8px;
    font-family: 'DM Mono', monospace; font-size: 12px;
    color: #9090b8; line-height: 1.65;
}
.chunk-meta {
    font-size: 10px; color: #363660; margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.8px;
}

.answer-box {
    background: #070d0b; border: 1px solid #163026;
    border-left: 3px solid #1d9e75; border-radius: 8px;
    padding: 16px 20px; color: #90c8b0;
    font-size: 14px; line-height: 1.75; margin-top: 10px;
}

.badge-pass {
    background: #0a2016; color: #3ddc84;
    border: 1px solid #183828; border-radius: 20px;
    padding: 3px 10px; font-size: 10px;
    font-family: 'DM Mono', monospace; display: inline-block;
    margin: 2px 0;
}
.badge-fail {
    background: #200a0a; color: #ff6b6b;
    border: 1px solid #381818; border-radius: 20px;
    padding: 3px 10px; font-size: 10px;
    font-family: 'DM Mono', monospace; display: inline-block;
    margin: 2px 0;
}
.badge-warn {
    background: #201808; color: #ffa040;
    border: 1px solid #382808; border-radius: 20px;
    padding: 3px 10px; font-size: 10px;
    font-family: 'DM Mono', monospace; display: inline-block;
    margin: 2px 0;
}

.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 0; border-bottom: 1px solid #16162a; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #50508a; padding: 10px 20px; font-size: 13px; border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"] { color: #c0c0f0; border-bottom: 2px solid #5050c0; background: transparent; }

.stTextInput input, .stTextArea textarea {
    background: #0d0d1c !important; border: 1px solid #22224a !important;
    color: #d0d0f0 !important; border-radius: 8px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #4040a0 !important;
    box-shadow: 0 0 0 2px rgba(64,64,160,0.15) !important;
}

.stButton > button {
    background: #14142a !important; color: #9090d0 !important;
    border: 1px solid #22224a !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 13px !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: #20204a !important; color: #d0d0ff !important;
    border-color: #4040a0 !important;
}
.stSelectbox > div > div {
    background: #0d0d1c !important; border: 1px solid #22224a !important;
    color: #d0d0f0 !important; border-radius: 8px !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #07070f; }
::-webkit-scrollbar-thumb { background: #20204a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 28px 0;'>
      <div style='font-family:"DM Mono",monospace;font-size:19px;font-weight:500;
                  color:#d0d0f0;letter-spacing:-0.5px;'>DocuEval</div>
      <div style='font-size:10px;color:#36367a;text-transform:uppercase;
                  letter-spacing:2px;margin-top:5px;'>RAG Evaluation Suite</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠  Overview",
         "💬  Query Pipeline",
         "📊  Retrieval Eval",
         "🤖  Generation Eval",
         "🚫  OOS Testing",
         "⚖️  A/B Test",
         "🧪  Regression Tests"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#28285a;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>System status</div>", unsafe_allow_html=True)

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    st.markdown(
        f'<div>{"<span class=badge-pass>● API KEY SET</span>" if api_key else "<span class=badge-fail>● NO API KEY</span>"}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="margin-top:5px">{"<span class=badge-pass>● VECTOR STORE READY</span>" if os.path.exists("chroma_store") else "<span class=badge-fail>● RUN ingest.py FIRST</span>"}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="margin-top:5px">{"<span class=badge-pass>● EVAL RESULTS FOUND</span>" if os.path.exists("results/retrieval_scores.json") else "<span class=badge-warn>● NO EVAL RESULTS YET</span>"}</div>',
        unsafe_allow_html=True
    )

# ── Route ─────────────────────────────────────────────────────────────────────
if "Overview" in page:
    from pages_ui import pg_overview as pg
elif "Query" in page:
    from pages_ui import pg_query as pg
elif "Retrieval" in page:
    from pages_ui import pg_retrieval as pg
elif "Generation" in page:
    from pages_ui import pg_generation as pg
elif "OOS" in page:
    from pages_ui import pg_oos as pg
elif "A/B" in page:
    from pages_ui import pg_abtest as pg
elif "Regression" in page:
    from pages_ui import pg_regression as pg

pg.render()
