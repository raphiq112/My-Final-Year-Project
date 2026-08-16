"""
Landing Page — Pre-Release Movie Success Prediction
========================================================
Entry point of the Streamlit app. Run with: streamlit run app.py
"""

import base64
from pathlib import Path
import streamlit as st
from utils.styling import inject_theme

st.set_page_config(
    page_title="Pre-Release Movie Success Prediction",
    page_icon="🎬",
    layout="centered",
)
inject_theme()

# ── Background image ──────────────────────────────────────────────────────────
# Place your image at movie_app/assets/landing_bg.jpg (or change the filename
# below to match whatever you saved). Base64-encoding it into the CSS is the
# only reliable way to get a true full-bleed background in Streamlit — a
# plain st.image() call can't sit behind other elements like this.
BG_PATH = Path(__file__).parent / "assets" / "landing_bg.png"

if BG_PATH.exists():
    bg_bytes = BG_PATH.read_bytes()
    bg_b64 = base64.b64encode(bg_bytes).decode()
    ext = BG_PATH.suffix.lstrip(".")
    st.markdown(f"""
        <style>
        .stApp {{
            background-image:  url("data:image/{ext};base64,{bg_b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)
else:
    st.caption(
        f"No background image found at {BG_PATH.relative_to(Path(__file__).parent)} — "
        f"add one to enable the landing page background."
    )

st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem;">
        <h1 style="font-size:48px; font-weight:600; color:#f4f4fa; letter-spacing:0.02em;">
            PRE-RELEASE MOVIE SUCCESS<br>PREDICTION
        </h1>
        <p style="color:#8888a0; font-size:24px; margin-top:8px;">
            See the future of film, before it's even finished.
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Let's predict", use_container_width=True, type="primary"):
        st.switch_page("pages/1_homepage.py")

st.markdown("""
    <div style="text-align:center; margin-top:2rem; color:#6b6b80; font-size:12px;">
        Powered by TMDB metadata · Reddit sentiment · YouTube trailer engagement
    </div>
""", unsafe_allow_html=True)
