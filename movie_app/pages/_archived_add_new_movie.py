"""
Add New Movie — Live Scrape + Predict
==========================================
Implements the Chapter 3 system flowchart: search a movie not yet cached →
pull live from TMDB + Reddit (Arctic Shift) + YouTube → engineer features
identically to training → run trained models (inference only) → append to
the CSV cache so future lookups are instant.
"""

import streamlit as st
from utils.styling import inject_theme
from utils import tmdb_client, reddit_client, youtube_client, cache_store
from utils import feature_engineering as fe
from utils.model_loader import load_all_artifacts, predict_one

st.set_page_config(page_title="Add new movie", page_icon="🎬", layout="centered")
inject_theme()


@st.cache_resource
def get_artifacts():
    return load_all_artifacts()


st.markdown("""
    <h2 style="color:#f4f4fa; font-weight:500;">Predict an upcoming release</h2>
    <p style="color:#8888a0; font-size:13px;">
        Search TMDB for a title not yet in the database. Data is pulled live
        from TMDB, Reddit, and YouTube.
    </p>
""", unsafe_allow_html=True)

try:
    tmdb_api_key = st.secrets["TMDB_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error(
        "TMDB_API_KEY not found. Add it to .streamlit/secrets.toml — see the "
        "comment at the top of that file for the format."
    )
    st.stop()

query = st.text_input("Search movie title", placeholder="e.g. Avatar: Fire and Ash")

if query:
    # ── Cache check first (matches Chapter 3 flowchart) ────────────────────────
    cached = cache_store.find_cached(query)
    if cached:
        st.info(f"'{cached['Title']}' is already in the database — showing cached prediction.")
        col1, col2 = st.columns(2)
        col1.metric("Predicted rating", f"{cached['pred_rating']:.1f} / 10")
        col2.metric("Predicted revenue", f"${cached['pred_revenue']:,.0f}")
        st.markdown(f"**Success category:** {cached['success_label']}")
        st.stop()

    candidates = tmdb_client.search_movies(query, tmdb_api_key)
    if not candidates:
        st.warning("No matching movies found on TMDB. Try a different spelling.")
        st.stop()

    options = {f"{c['title']} ({c['release_date'][:4] if c['release_date'] else 'TBA'})": c
               for c in candidates}
    choice_label = st.selectbox("Select the correct movie", list(options.keys()))
    chosen = options[choice_label]

    if st.button("Fetch and predict", type="primary"):
        status_box = st.status("Fetching from 3 sources", expanded=True)

        with status_box:
            st.write("TMDB metadata...")
            tmdb_data = tmdb_client.get_movie_full(chosen["id"], tmdb_api_key)
            st.write(f"✓ TMDB metadata done — budget ${tmdb_data['budget']:,.0f}")

            st.write("Reddit sentiment (Arctic Shift)...")
            reddit_data = reddit_client.get_reddit_sentiment(
                tmdb_data["title"], tmdb_data["release_date"]
            )
            n_posts = reddit_data.get("raw_post_count_fetched", 0)
            st.write(f"✓ Reddit done — {n_posts} posts found")

            st.write("YouTube trailer engagement...")
            yt_data = youtube_client.get_youtube_sentiment(
                tmdb_data["title"], tmdb_data["release_date"], str(tmdb_data["year"])
            )
            if yt_data.get("video_id"):
                st.write(f"✓ YouTube done — trailer found, {yt_data.get('comment_count', 0)} comments")
            else:
                st.write("✓ YouTube done — no trailer found, using neutral defaults")

            status_box.update(label="Done", state="complete")

        if not reddit_data.get("post_count"):
            st.caption("No Reddit posts found yet — sentiment features filled with neutral defaults.")
        if not yt_data.get("video_id"):
            st.caption("No trailer found yet — YouTube features filled with neutral defaults.")

        # ── Feature engineering + prediction (inference only) ──────────────────
        artifacts = get_artifacts()
        row_df = fe.engineer_row(
            tmdb_data, reddit_data, yt_data,
            artifacts["mlb"], artifacts["genre_cols"], artifacts["fill_values"],
        )
        prediction = predict_one(row_df, artifacts, budget=tmdb_data["budget"])

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Predicted rating", f"{prediction['pred_rating']:.1f} / 10")
        col2.metric("Predicted revenue", f"${prediction['pred_revenue']:,.0f}")
        st.markdown(f"**Success category:** {prediction['success_label']}")

        # ── Cache for next time ──────────────────────────────────────────────────
        cache_store.append_movie(tmdb_data, prediction)
        st.success("Saved to database — future searches for this title will be instant.")
