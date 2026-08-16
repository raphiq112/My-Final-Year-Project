"""
Top Navigation Bar
====================
Shared nav component rendered at the top of every "main" page (Homepage,
Movies, Analysis Overview, Prediction History, Custom Movie). Movie Details
deliberately does NOT include this — per the user's request it has no nav,
since it's reached by clicking through from somewhere else, not browsed to
directly.
"""

import streamlit as st
from pathlib import Path
import base64

LOGO_PATH = Path(__file__).parent.parent / "assets" / "logo.png"


def _logo_html():
    """Render the user's own logo if present at assets/logo.png, otherwise
    fall back to a plain text wordmark so the nav still works before they
    add one."""
    if LOGO_PATH.exists():
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        ext = LOGO_PATH.suffix.lstrip(".")
        return f'<img src="data:image/{ext};base64,{b64}" style="height:28px;">'
    return '<span style="font-weight:700;font-size:18px;color:#f4f4fa;letter-spacing:0.04em;">PRP</span>'


def render_navbar(active="Home"):
    """
    Renders the nav bar visually (logo + link labels as styled HTML, matching
    the TMDB reference look) and a row of real Streamlit buttons underneath
    that actually perform navigation — Streamlit can't make arbitrary HTML
    <a> tags trigger st.switch_page, so the buttons are the functional layer
    while the styled bar above gives the TMDB-like visual.
    """
    st.markdown(f"""<style>
.pmsp-navbar {{
    display:flex; align-items:center; gap:28px;
    background:#0d0e1a; padding:14px 24px; border-bottom:1px solid rgba(255,255,255,0.08);
    margin: -1rem -1rem 1rem -1rem;
}}
.pmsp-navlink {{
    color:#b0b0c0; font-size:14px; font-weight:500; cursor:pointer;
}}
.pmsp-navlink.active {{ color:#f4f4fa; border-bottom:2px solid #7f77dd; padding-bottom:4px; }}
</style>""", unsafe_allow_html=True)

    st.markdown(
        f'<div class="pmsp-navbar">{_logo_html()}'
        f'<span class="pmsp-navlink {"active" if active=="Home" else ""}">Homepage</span>'
        f'<span class="pmsp-navlink {"active" if active=="Movies" else ""}">Movies</span>'
        f'<span class="pmsp-navlink {"active" if active=="Analysis" else ""}">Analysis Overview</span>'
        f'<span class="pmsp-navlink {"active" if active=="History" else ""}">Prediction History</span>'
        f'<span class="pmsp-navlink {"active" if active=="Custom" else ""}">Add Custom Movie</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns([1, 1, 1.3, 1, 1.3, 1, 4])
    with cols[0]:
        if st.button("Home", key="nav_home", use_container_width=True,
                      type="primary" if active == "Home" else "secondary"):
            st.switch_page("pages/1_homepage.py")
    with cols[1]:
        if st.button("Popular", key="nav_popular", use_container_width=True,
                      type="primary" if active == "Movies_popular" else "secondary"):
            st.switch_page("pages/2a_popular_movies.py")
    with cols[2]:
        if st.button("Upcoming", key="nav_upcoming", use_container_width=True,
                      type="primary" if active == "Movies_upcoming" else "secondary"):
            st.switch_page("pages/2b_upcoming_movies.py")
    with cols[3]:
        if st.button("Analysis", key="nav_analysis", use_container_width=True,
                      type="primary" if active == "Analysis" else "secondary"):
            st.switch_page("pages/3_analysis_overview.py")
    with cols[4]:
        if st.button("History", key="nav_history", use_container_width=True,
                      type="primary" if active == "History" else "secondary"):
            st.switch_page("pages/4_prediction_history.py")
    with cols[5]:
        if st.button("Custom", key="nav_custom", use_container_width=True,
                      type="primary" if active == "Custom" else "secondary"):
            st.switch_page("pages/5_custom_movie.py")
