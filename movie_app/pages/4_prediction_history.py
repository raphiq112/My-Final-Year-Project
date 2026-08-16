"""
Prediction History
====================
Displays every movie this app has predicted so far (the CSV cache from
Add New Movie / Movie Details), in a systematic table view, with a download
button. Replaces the old standalone Prediction Result page in the nav.
"""

import streamlit as st
import pandas as pd
from utils.styling import inject_theme
from utils.navbar import render_navbar
from utils import cache_store

st.set_page_config(page_title="Prediction History", page_icon="📋", layout="wide")
inject_theme()
render_navbar(active="History")

st.markdown("### Prediction History")
st.caption("All movies predicted so far, most recent first.")

df = cache_store.load_cache()

if df.empty:
    st.info("No predictions yet — go to Homepage or Add Custom Movie to make your first prediction.")
    st.stop()

# Most recent first
if "scraped_at" in df.columns:
    df = df.sort_values("scraped_at", ascending=False)

# ── Filters ──────────────────────────────────────────────────────────────────────
filt1, filt2 = st.columns(2)
with filt1:
    label_options = ["All"] + sorted(df["success_label"].dropna().unique().tolist())
    label_filter = st.selectbox("Filter by success category", label_options)
with filt2:
    search_title = st.text_input("Search by title", placeholder="e.g. Avatar")

view = df.copy()
if label_filter != "All":
    view = view[view["success_label"] == label_filter]
if search_title:
    view = view[view["Title"].str.contains(search_title, case=False, na=False)]

st.markdown(f"**{len(view)}** of **{len(df)}** total predictions shown")

# ── Systematic table view ───────────────────────────────────────────────────────
display_cols = ["Title", "Release_Date", "Budget", "pred_revenue", "pred_rating",
                 "success_label", "scraped_at"]
display_cols = [c for c in display_cols if c in view.columns]

formatted = view[display_cols].copy()
if "Budget" in formatted.columns:
    formatted["Budget"] = formatted["Budget"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
if "pred_revenue" in formatted.columns:
    formatted["pred_revenue"] = formatted["pred_revenue"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
if "pred_rating" in formatted.columns:
    formatted["pred_rating"] = formatted["pred_rating"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

st.dataframe(formatted, use_container_width=True, hide_index=True)

# ── Download ─────────────────────────────────────────────────────────────────────
st.download_button(
    "⬇ Download full prediction history (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="movie_predictions.csv",
    mime="text/csv",
)
