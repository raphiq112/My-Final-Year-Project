"""
Movie Details — Metadata + Social Signals
==============================================
Matches Figure 3.7 (UI Design of Movie Details): poster + metadata on the
left, Reddit/YouTube side-by-side panels on the right, "Let's Predict"
button to proceed to Prediction Result. Reuses the same scrapers and
feature-engineering pipeline as Add New Movie — this page doesn't duplicate
that logic, it just reads/displays the data on the way to prediction.
"""

import streamlit as st
from utils.styling import inject_theme
from utils import tmdb_client, reddit_client, youtube_client, cache_store
from utils import feature_engineering as fe
from utils.model_loader import load_all_artifacts, predict_one

st.set_page_config(page_title="Movie details", page_icon="🎬", layout="wide")
inject_theme()


@st.cache_resource
def get_artifacts():
    return load_all_artifacts()


try:
    tmdb_api_key = st.secrets["TMDB_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("TMDB_API_KEY not found. Add it to .streamlit/secrets.toml.")
    st.stop()

# No nav bar on this page by design — it's reached only by clicking through
# from Homepage, Movies, or search, never browsed to directly. A simple
# back-to-homepage link is enough.
if st.button("← Back"):
    st.switch_page("pages/1_homepage.py")

movie_id = st.session_state.get("selected_movie_id")
movie_title = st.session_state.get("selected_movie_title")

if not movie_title:
    st.warning("No movie selected. Go back to the homepage and pick one.")
    st.stop()

# ── Load data: cached row, or fresh TMDB lookup ──────────────────────────────────
cached = cache_store.find_cached(movie_title)

if cached:
    tmdb_data = {
        "title": cached["Title"], "release_date": cached["Release_Date"],
        "budget": cached["Budget"], "director": cached.get("director", ""),
        "genre": cached.get("genre", ""), "poster_path": cached.get("poster_path"),
        "vote_count": 0, "runtime": 0, "year": "", "production_companies": "", "cast": "",
    }
    has_prediction_cached = True
elif movie_id:
    with st.spinner("Loading movie details..."):
        tmdb_data = tmdb_client.get_movie_full(movie_id, tmdb_api_key)
    has_prediction_cached = False
else:
    st.error("Couldn't load this movie's details.")
    st.stop()

# ── Header: poster + core metadata ───────────────────────────────────────────────
col1, col2 = st.columns([1, 2])
with col1:
    poster = tmdb_client.poster_url(tmdb_data.get("poster_path"))
    if poster:
        st.image(poster, use_container_width=True)
    else:
        st.markdown(
            "<div style='background:#16172a;border-radius:8px;aspect-ratio:2/3;"
            "display:flex;align-items:center;justify-content:center;color:#5a5a72;'>"
            "No poster</div>", unsafe_allow_html=True,
        )
with col2:
    st.markdown(f"### {tmdb_data['title']}")
    budget = tmdb_data.get("budget", 0) or 0
    st.markdown(
        f'<p style="color:#8888a0;font-size:13px;line-height:1.7;">'
        f'Release date: {tmdb_data.get("release_date", "TBA")}<br>'
        f'Director: {tmdb_data.get("director", "Unknown")}<br>'
        f'Genre: {tmdb_data.get("genre", "Unknown")}<br>'
        f'Budget: ${budget:,.0f}</p>',
        unsafe_allow_html=True,
    )
    predict_clicked = st.button("Let's predict →", type="primary")

st.divider()

# ── Reddit / YouTube side-by-side panels ─────────────────────────────────────────
st.markdown("**Social signals**")
panel1, panel2 = st.columns(2)

# These panels only show LIVE data for un-cached movies — for already-cached
# movies we skip re-scraping here and just confirm a prediction exists.
reddit_data, yt_data = {}, {}

if not cached:
    with st.spinner("Checking Reddit and YouTube..."):
        reddit_data = reddit_client.get_reddit_sentiment(
            tmdb_data["title"], tmdb_data.get("release_date", "")
        )
        yt_data = youtube_client.get_youtube_sentiment(
            tmdb_data["title"], tmdb_data.get("release_date", ""), str(tmdb_data.get("year", ""))
        )

with panel1:
    st.markdown("**🔴 Reddit**")
    if cached:
        st.caption("Sentiment already captured in cached prediction.")
    elif reddit_data.get("post_count"):
        st.metric("Posts found", reddit_data["post_count"])
        st.progress(max(0.0, min(1.0, (reddit_data["avg_compound"] + 1) / 2)),
                    text=f"Avg sentiment: {reddit_data['avg_compound']:+.2f}")
    else:
        st.caption("No Reddit posts found yet for this title.")

with panel2:
    st.markdown("**▶️ YouTube**")
    if cached:
        st.caption("Trailer engagement already captured in cached prediction.")
    elif yt_data.get("video_id"):
        st.metric("Trailer views", f"{yt_data['video_view_count']:,}")
        st.metric("Likes", f"{yt_data['likes']:,}")
    else:
        st.caption("No trailer found yet for this title.")

# ── Predict + inline result ────────────────────────────────────────────────────────
# No separate Prediction Result page anymore (replaced by Prediction History,
# per request) — the result is shown right here, and is also saved to the
# cache so it shows up in History afterward.
if predict_clicked:
    if cached:
        result = {
            "title": cached["Title"], "pred_revenue": cached["pred_revenue"],
            "pred_rating": cached["pred_rating"], "success_label": cached["success_label"],
            "budget": cached["Budget"],
        }
    else:
        artifacts = get_artifacts()
        row_df = fe.engineer_row(
            tmdb_data, reddit_data, yt_data,
            artifacts["mlb"], artifacts["genre_cols"], artifacts["fill_values"],
        )
        prediction = predict_one(row_df, artifacts, budget=tmdb_data["budget"])
        cache_store.append_movie(tmdb_data, prediction)
        result = {
            "title": tmdb_data["title"], "pred_revenue": prediction["pred_revenue"],
            "pred_rating": prediction["pred_rating"], "success_label": prediction["success_label"],
            "budget": tmdb_data["budget"],
        }

    st.divider()
    st.markdown("### Prediction result")
    rcol1, rcol2 = st.columns(2)
    rcol1.metric("Predicted rating", f"{result['pred_rating']:.1f} / 10")
    rcol2.metric("Predicted revenue", f"${result['pred_revenue']:,.0f}")
    st.markdown(f"<span class='badge-success'>{result['success_label']}</span>",
                unsafe_allow_html=True)
    st.caption("Saved to Prediction History.")
    if st.button("View full prediction history →"):
        st.switch_page("pages/4_prediction_history.py")
