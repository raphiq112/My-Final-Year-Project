"""
Shared Styling
===============
Injects the dark theme CSS (purple/teal/amber accents on near-black
background) used across all pages, matching the UI mockups in Chapter 3
Table 3.3 / Figures 3.5-3.9.
"""

import streamlit as st

PALETTE = {
    "bg": "#0d0e1a",
    "card": "#16172a",
    "border": "rgba(255,255,255,0.08)",
    "text_primary": "#f4f4fa",
    "text_secondary": "#8888a0",
    "text_muted": "#6b6b80",
    "purple": "#7f77dd",
    "teal": "#5dcaa5",
    "amber": "#facb7d",
    "coral": "#f0997b",
    "pink": "#d4537e",
}


def inject_theme():
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {PALETTE['bg']};
        }}
        .movie-card {{
            background: {PALETTE['card']};
            border-radius: 10px;
            padding: 12px 14px;
            border: 0.5px solid {PALETTE['border']};
        }}
        .badge-success {{
            display: inline-block;
            background: rgba(239,159,39,0.18);
            color: {PALETTE['amber']};
            font-size: 11px;
            font-weight: 500;
            padding: 5px 11px;
            border-radius: 20px;
        }}
        .metric-label {{
            font-size: 11px;
            color: {PALETTE['text_secondary']};
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: 500;
            color: {PALETTE['text_primary']};
        }}
        </style>
    """, unsafe_allow_html=True)
