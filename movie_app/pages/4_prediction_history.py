"""
Prediction History
====================
Displays every movie this app has predicted so far, with filters, a
download button, and — new in this revision — a way to delete a saved
prediction so that title gets re-scraped and re-predicted next time it's
opened from Movie Details (since Movie Details always shows a cached
result immediately if one exists).

Deletion goes through a new `cache_store.delete_movie(titles)` function —
see the top of utils/cache_store.py.

Also new: a "⚠️ Notes only" filter and a ⚠️ column, surfacing predictions
whose classification relied on imputed/proxy data (missing budget, no
Reddit posts, no trailer/comments found) — see
utils/feature_engineering.compute_data_notes().
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
filt1, filt2, filt3 = st.columns([2, 2, 1.4])
with filt1:
    label_options = ["All"] + sorted(df["success_label"].dropna().unique().tolist())
    label_filter = st.selectbox("Filter by success category", label_options)
with filt2:
    search_title = st.text_input("Search by title", placeholder="e.g. Avatar")
with filt3:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    notes_only = st.checkbox("⚠️ Notes only", help="Show only predictions where some "
                              "metadata was missing/imputed (e.g. unknown budget, "
                              "no sentiment fetched).")

view = df.copy()
if "data_notes" not in view.columns:
    view["data_notes"] = ""
view["data_notes"] = view["data_notes"].fillna("")

if label_filter != "All":
    view = view[view["success_label"] == label_filter]
if search_title:
    view = view[view["Title"].str.contains(search_title, case=False, na=False)]
if notes_only:
    view = view[view["data_notes"].str.strip() != ""]

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

formatted.insert(1, "Notes ⚠️", view["data_notes"].apply(lambda n: "⚠️" if str(n).strip() else ""))

st.dataframe(formatted, use_container_width=True, hide_index=True)

flagged = view[view["data_notes"].str.strip() != ""]
if not flagged.empty:
    with st.expander(f"⚠️ Data-quality notes ({len(flagged)} prediction(s) affected)"):
        for _, r in flagged.iterrows():
            st.markdown(f"**{r['Title']}**")
            for n in str(r["data_notes"]).split("|"):
                if n.strip():
                    st.caption(f"• {n.strip()}")

# ── Download ─────────────────────────────────────────────────────────────────────
st.download_button(
    "⬇ Download full prediction history (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="movie_predictions.csv",
    mime="text/csv",
)

# ── Manage predictions: delete to allow re-prediction ────────────────────────────
st.divider()
st.markdown("#### Manage predictions")
st.caption("Deleting a title removes its cached result — next time you open it from "
           "Movie Details, it will be re-scraped and re-predicted from scratch.")

titles_available = sorted(view["Title"].dropna().unique().tolist())
titles_to_delete = st.multiselect("Select title(s) to delete", titles_available)

confirm = st.checkbox("I understand this permanently removes the selected cached prediction(s).",
                       disabled=not titles_to_delete)

if st.button("🗑 Delete selected", type="secondary", disabled=not (titles_to_delete and confirm)):
    n_deleted = cache_store.delete_movie(titles_to_delete)
    st.success(f"Deleted {n_deleted} prediction(s): {', '.join(titles_to_delete)}")
    st.rerun()
