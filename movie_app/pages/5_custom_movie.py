"""
Custom Movie — What-If Simulator
=====================================
Lets the user simulate a hypothetical movie by directly setting feature
values, rather than scraping real data (see Add New Movie for that).
No API calls — every control here maps to a real trained feature so the
prediction is defensible in Chapter 4, unlike a generic "social buzz" slider.
"""

import streamlit as st
import numpy as np
from utils.styling import inject_theme
from utils.navbar import render_navbar
from utils import feature_engineering as fe
from utils.model_loader import load_all_artifacts, predict_one

st.set_page_config(page_title="Custom movie", page_icon="🎬", layout="wide")
inject_theme()
render_navbar(active="Custom")


@st.cache_resource
def get_artifacts():
    return load_all_artifacts()


st.markdown("""
    <h2 style="color:#f4f4fa; font-weight:500;">Insert your movie</h2>
    <p style="color:#8888a0; font-size:13px;">
        Simulate a hypothetical movie by setting feature values directly —
        no live scraping involved. Every control below maps to a feature the
        model was actually trained on.
    </p>
""", unsafe_allow_html=True)

artifacts = get_artifacts()
genre_options = [c.replace("genre_", "").replace("_", " ") for c in artifacts["genre_cols"]]

budget_m = st.number_input("Budget (USD, millions)", min_value=0.0, max_value=500.0,
                            value=50.0, step=5.0)
genre_choice = st.selectbox("Primary genre", genre_options)
runtime = st.slider("Runtime (minutes)", 60, 220, 115)
release_month = st.selectbox("Release month", list(range(1, 13)),
                              format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                       "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
release_year = st.number_input("Release year", min_value=2024, max_value=2030, value=2026, step=1)
is_major_studio = st.checkbox("Major studio production", value=True)
cast_size = st.slider("Number of billed cast members", 1, 15, 6)

st.markdown("**Expected social sentiment**")
sentiment_score = st.slider(
    "Reddit sentiment (negative ↔ positive)", -1.0, 1.0, 0.2, step=0.05,
    help="Maps directly to avg_compound — the VADER compound sentiment score "
         "your model was trained on, not an arbitrary buzz metric."
)
trailer_percentile = st.slider(
    "Expected trailer reception (percentile vs. dataset)", 0, 100, 50, step=5,
    help="Maps to trailer_score — your trailer-quality ranking feature, "
         "expressed as a percentile for intuitive input."
)

if st.button("Predict", type="primary"):
    budget = budget_m * 1_000_000

    tmdb_data = {
        "budget": budget,
        "runtime": runtime,
        "vote_count": 500,  # neutral placeholder — a custom movie has no real vote history
        "year": release_year,
        "production_companies": "Major Studio Co." if is_major_studio else "Independent Films",
        "cast": ", ".join([f"Actor {i+1}" for i in range(cast_size)]),
        "genre": genre_choice,
    }

    # Sentiment slider (-1 to 1) maps directly to avg_compound.
    # Positive/negative ratios derived proportionally so they stay internally consistent.
    pos_ratio = max(0.0, sentiment_score) * 0.7 + 0.15
    neg_ratio = max(0.0, -sentiment_score) * 0.7 + 0.05
    reddit_data = {
        "post_count": 30,  # assume moderate pre-release buzz exists
        "avg_compound": sentiment_score,
        "avg_positive": max(sentiment_score, 0),
        "avg_negative": max(-sentiment_score, 0),
        "avg_neutral": 1 - abs(sentiment_score),
        "positive_ratio": round(pos_ratio, 3),
        "negative_ratio": round(neg_ratio, 3),
        "total_score": 500,
        "avg_num_comments": 40,
    }

    # Trailer percentile (0-100) maps to trailer_score directly.
    yt_data = {
        "video_view_count": 5_000_000,
        "likes": 200_000,
        "total_likes": 200_000,
        "trailer_score": trailer_percentile,
        "comment_count": 300,
        "avg_compound": sentiment_score,
        "avg_positive": max(sentiment_score, 0),
        "avg_negative": max(-sentiment_score, 0),
        "avg_neutral": 1 - abs(sentiment_score),
        "positive_ratio": round(pos_ratio, 3),
        "negative_ratio": round(neg_ratio, 3),
        "weighted_compound": sentiment_score,
    }

    row_df = fe.engineer_row(
        tmdb_data, reddit_data, yt_data,
        artifacts["mlb"], artifacts["genre_cols"], artifacts["fill_values"],
    )
    prediction = predict_one(row_df, artifacts, budget=budget)

    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Predicted rating", f"{prediction['pred_rating']:.1f} / 10")
    col2.metric("Predicted revenue", f"${prediction['pred_revenue']:,.0f}")
    st.markdown(f"**Success category:** {prediction['success_label']}")
    st.caption(
        "This is a simulated, hypothetical movie — no real scraping occurred. "
        "Vote count, view count, and audience size are set to neutral "
        "placeholders since a custom movie has no real popularity history yet."
    )
