"""
Movie Details — Metadata + Social Signals + Prediction
=============================================================
Three-column header: poster | metadata | prediction result. Reuses the
same scrapers and feature-engineering pipeline as before.

Changes in this revision:
  - Poster is smaller (fixed width instead of use_container_width) and
    metadata font is bumped up — the old layout made the poster huge and
    the text tiny.
  - Prediction result now lives in its own right-hand column instead of
    appearing full-width below a divider after clicking predict. If the
    movie is already cached, the result shows immediately without needing
    to click anything.
  - The fetched YouTube trailer now plays inline via st.video.
  - Reddit average sentiment renders as a colour-graded (red→green hue)
    bar instead of the flat default st.progress bar.
  - Social signal panels (Reddit/YouTube) now persist across reruns even
    after a prediction has been cached — they're stashed in
    st.session_state the first time they're fetched live, and a "Refresh
    social signals" button re-fetches them for movies that were cached in
    an earlier session (before this page kept them around).
  - A "Sample sentiment fetched" expander shows a few raw Reddit
    posts / YouTube comments with their compound scores, if the
    reddit_client / youtube_client functions return a `samples` list.
    (See note near the bottom if yours don't yet.)
  - The page now also accepts ?movie_id=<id> in the URL as a fallback to
    session_state, so posters linked from the homepage can navigate here
    directly without going through st.switch_page.
  - New: a "☁️ Sentiment word cloud" expander, built from the raw fetched
    Reddit/YouTube text (utils/wordcloud_viz.py) — descriptive only, not
    a model input.
  - New: a "⚠️ Notes on this prediction" box flags when the classification
    relied on imputed/proxy data — no budget (100M revenue proxy used),
    no Reddit posts found, no trailer matched, or no pre-release comments
    (utils/feature_engineering.compute_data_notes()). Persisted to the
    cache so it still shows up for cached predictions later.
"""

import streamlit as st
from utils.styling import inject_theme
from utils import tmdb_client, reddit_client, youtube_client, cache_store
from utils import feature_engineering as fe
from utils import wordcloud_viz
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

# Fallback: came in via a clickable poster link (?movie_id=123) rather than
# session_state set by a button click elsewhere in the app.
if not movie_id:
    qp_movie_id = st.query_params.get("movie_id")
    if qp_movie_id:
        try:
            movie_id = int(qp_movie_id)
            st.session_state["selected_movie_id"] = movie_id
        except (TypeError, ValueError):
            movie_id = None

if not movie_title and not movie_id:
    st.warning("No movie selected. Go back to the homepage and pick one.")
    st.stop()

# ── Load data: cached row, or fresh TMDB lookup ──────────────────────────────────
# If we only have a movie_id (no title yet — e.g. arrived via poster link),
# fetch TMDB details first so we have a title to check the cache with.
tmdb_data = None
if not movie_title and movie_id:
    with st.spinner("Loading movie details..."):
        tmdb_data = tmdb_client.get_movie_full(movie_id, tmdb_api_key)
    movie_title = tmdb_data.get("title", "")
    st.session_state["selected_movie_title"] = movie_title

cached = cache_store.find_cached(movie_title)

if cached:
    tmdb_data = {
        "title": cached["Title"], "release_date": cached["Release_Date"],
        "budget": cached["Budget"], "director": cached.get("director", ""),
        "genre": cached.get("genre", ""), "poster_path": cached.get("poster_path"),
        "vote_count": 0, "runtime": 0, "year": "", "production_companies": "", "cast": "",
    }
elif tmdb_data is None:
    if movie_id:
        with st.spinner("Loading movie details..."):
            tmdb_data = tmdb_client.get_movie_full(movie_id, tmdb_api_key)
    else:
        st.error("Couldn't load this movie's details.")
        st.stop()

if not tmdb_data:
    st.error("Couldn't load this movie's details.")
    st.stop()

# ── Header: poster | metadata | prediction result ────────────────────────────────
col_poster, col_meta, col_pred = st.columns([1, 1.3, 1.4])

with col_poster:
    poster = tmdb_client.poster_url(tmdb_data.get("poster_path"))
    if poster:
        st.image(poster, width=220)
    else:
        st.markdown(
            "<div style='background:#16172a;border-radius:8px;width:220px;aspect-ratio:2/3;"
            "display:flex;align-items:center;justify-content:center;color:#5a5a72;'>"
            "No poster</div>", unsafe_allow_html=True,
        )

with col_meta:
    st.markdown(f"### {tmdb_data['title']}")
    budget = tmdb_data.get("budget", 0) or 0
    st.markdown(
        f'<p style="color:#c4c4d8;font-size:15px;line-height:2.0;">'
        f'<b>Release date:</b> {tmdb_data.get("release_date", "TBA")}<br>'
        f'<b>Director:</b> {tmdb_data.get("director", "Unknown")}<br>'
        f'<b>Genre:</b> {tmdb_data.get("genre", "Unknown")}<br>'
        f'<b>Budget:</b> ${budget:,.0f}</p>',
        unsafe_allow_html=True,
    )
    predict_clicked = st.button("Let's predict →", type="primary")

st.divider()

# ── Reddit / YouTube side-by-side panels ─────────────────────────────────────────
st.markdown("**Social signals**")
panel1, panel2 = st.columns(2)

social_key = f"social_{movie_title}"
reddit_data, yt_data = {}, {}

if not cached:
    # Fetch once and stash in session_state so it survives future reruns
    # even after this movie becomes cached (e.g. right after predicting).
    if social_key not in st.session_state:
        with st.spinner("Checking Reddit and YouTube..."):
            reddit_data = reddit_client.get_reddit_sentiment(
                tmdb_data["title"], tmdb_data.get("release_date", "")
            )
            yt_data = youtube_client.get_youtube_sentiment(
                tmdb_data["title"], tmdb_data.get("release_date", ""), str(tmdb_data.get("year", ""))
            )
        st.session_state[social_key] = {"reddit": reddit_data, "youtube": yt_data}
    else:
        reddit_data = st.session_state[social_key]["reddit"]
        yt_data = st.session_state[social_key]["youtube"]
else:
    # Already predicted in an earlier run. Use the live data we fetched
    # this session if we have it; otherwise offer to re-fetch for display.
    if social_key in st.session_state:
        reddit_data = st.session_state[social_key]["reddit"]
        yt_data = st.session_state[social_key]["youtube"]
    else:
        st.caption("This title was already predicted in an earlier session, "
                    "so its raw sentiment fetch isn't in memory anymore.")
        if st.button("🔄 Refresh social signals"):
            with st.spinner("Re-checking Reddit and YouTube..."):
                reddit_data = reddit_client.get_reddit_sentiment(
                    tmdb_data["title"], tmdb_data.get("release_date", "")
                )
                yt_data = youtube_client.get_youtube_sentiment(
                    tmdb_data["title"], tmdb_data.get("release_date", ""), str(tmdb_data.get("year", ""))
                )
            st.session_state[social_key] = {"reddit": reddit_data, "youtube": yt_data}
            st.rerun()


def _hue_sentiment_bar(score):
    """Custom red→green hue bar (st.progress can't be recoloured)."""
    pct = max(0.0, min(1.0, (score + 1) / 2))
    hue = int(pct * 120)  # 0 = red, 120 = green
    st.markdown(
        f'<div style="background:#20202f;border-radius:6px;height:10px;width:100%;'
        f'overflow:hidden;margin:8px 0 4px;">'
        f'<div style="background:hsl({hue},70%,50%);height:100%;width:{pct*100:.0f}%;"></div>'
        f'</div>'
        f'<p style="color:#8888a0;font-size:12px;margin:0;">Avg sentiment: {score:+.2f}</p>',
        unsafe_allow_html=True,
    )


with panel1:
    st.markdown("**🔴 Reddit**")
    if reddit_data.get("post_count"):
        st.metric("Posts found", reddit_data["post_count"])
        _hue_sentiment_bar(reddit_data.get("avg_compound", 0.0))
    elif cached and social_key not in st.session_state:
        pass  # refresh prompt already shown above
    else:
        st.caption("No Reddit posts found yet for this title.")

with panel2:
    st.markdown("**▶️ YouTube**")
    if yt_data.get("video_id"):
        st.metric("Trailer views", f"{yt_data['video_view_count']:,}")
        st.metric("Likes", f"{yt_data['likes']:,}")
        st.video(f"https://www.youtube.com/watch?v={yt_data['video_id']}")
    elif cached and social_key not in st.session_state:
        pass
    else:
        st.caption("No trailer found yet for this title.")

# ── Sample sentiment preview ──────────────────────────────────────────────────────
with st.expander("📝 Sample sentiment fetched"):
    r_samples = reddit_data.get("samples")
    y_samples = yt_data.get("samples")

    def _hue_chip(compound):
        pct = max(0.0, min(1.0, (compound + 1) / 2))
        hue = int(pct * 120)
        return f'<span style="color:hsl({hue},70%,60%); font-weight:600;">{compound:+.2f}</span>'

    st.markdown("**Reddit**")
    if r_samples:
        for s in r_samples:
            st.markdown(
                f'<p style="font-size:13px; color:#c4c4d8; margin:4px 0;">'
                f'“{str(s.get("text", ""))[:180]}” — {_hue_chip(s.get("compound", 0))}</p>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No sample posts to show (either none were fetched, or this "
                    "movie was loaded from an earlier cached session).")

    st.markdown("**YouTube**")
    if y_samples:
        for s in y_samples:
            st.markdown(
                f'<p style="font-size:13px; color:#c4c4d8; margin:4px 0;">'
                f'“{str(s.get("text", ""))[:180]}” — {_hue_chip(s.get("compound", 0))}</p>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No sample comments to show (either none were fetched, or this "
                    "movie was loaded from an earlier cached session).")

# ── Sentiment word cloud ──────────────────────────────────────────────────────────
# Descriptive/exploratory only — built from the raw fetched text, not from
# the feature vectors the models were trained on. Suggested by the project
# supervisor as a qualitative complement to the VADER aggregate scores.
with st.expander("☁️ Sentiment word cloud"):
    wc_col1, wc_col2 = st.columns(2)
    r_texts = reddit_data.get("all_texts") or []
    y_texts = yt_data.get("all_texts") or []

    with wc_col1:
        st.markdown("**Reddit**")
        r_img = wordcloud_viz.build_wordcloud_image(r_texts, movie_title=movie_title)
        if r_img is not None:
            st.image(r_img, use_container_width=True)
        elif r_texts:
            st.caption("Not enough distinct text to build a cloud — showing raw word counts instead.")
            for word, count in wordcloud_viz.top_words(r_texts, movie_title=movie_title, n=10):
                st.caption(f"{word} — {count}")
        else:
            st.caption("No Reddit text available for this title.")

    with wc_col2:
        st.markdown("**YouTube**")
        y_img = wordcloud_viz.build_wordcloud_image(y_texts, movie_title=movie_title)
        if y_img is not None:
            st.image(y_img, use_container_width=True)
        elif y_texts:
            st.caption("Not enough distinct text to build a cloud — showing raw word counts instead.")
            for word, count in wordcloud_viz.top_words(y_texts, movie_title=movie_title, n=10):
                st.caption(f"{word} — {count}")
        else:
            st.caption("No YouTube comment text available for this title.")

    st.caption("Word cloud is descriptive only — it visualises what was fetched, "
                "it isn't an input to the prediction models.")

# ── Prediction result panel ───────────────────────────────────────────────────────
with col_pred:
    st.markdown("**Prediction**")
    result = None
    data_notes = []

    if cached:
        result = {
            "title": cached["Title"], "pred_revenue": cached["pred_revenue"],
            "pred_rating": cached["pred_rating"], "success_label": cached["success_label"],
        }
        raw_notes = cached.get("data_notes", "")
        if isinstance(raw_notes, str) and raw_notes.strip():
            data_notes = [n.strip() for n in raw_notes.split("|") if n.strip()]
    elif predict_clicked:
        artifacts = get_artifacts()
        row_df = fe.engineer_row(
            tmdb_data, reddit_data, yt_data,
            artifacts["mlb"], artifacts["genre_cols"], artifacts["fill_values"],
        )
        prediction = predict_one(row_df, artifacts, budget=tmdb_data["budget"])
        data_notes = fe.compute_data_notes(tmdb_data, reddit_data, yt_data, artifacts.get("fill_values"))
        cache_store.append_movie(tmdb_data, prediction, notes=data_notes)
        result = {
            "title": tmdb_data["title"], "pred_revenue": prediction["pred_revenue"],
            "pred_rating": prediction["pred_rating"], "success_label": prediction["success_label"],
        }

    if result:
        st.metric("Predicted rating", f"{result['pred_rating']:.1f} / 10")
        st.metric("Predicted revenue", f"${result['pred_revenue']:,.0f}")
        st.markdown(f"<span class='badge-success'>{result['success_label']}</span>",
                    unsafe_allow_html=True)
        st.caption("Saved to Prediction History." if not cached else "Loaded from Prediction History.")

        if data_notes:
            notes_html = "".join(f"<li>{n}</li>" for n in data_notes)
            st.markdown(
                f'<div style="background:rgba(250,203,125,0.12); border:1px solid rgba(250,203,125,0.4); '
                f'border-radius:8px; padding:10px 14px; margin-top:10px;">'
                f'<p style="color:#facb7d; font-size:12px; font-weight:600; margin:0 0 4px;">'
                f'⚠️ Notes on this prediction</p>'
                f'<ul style="color:#d8c9a8; font-size:12px; margin:0; padding-left:18px;">{notes_html}</ul>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if st.button("View full prediction history →"):
            st.switch_page("pages/4_prediction_history.py")
    else:
        st.caption("Click “Let's predict →” to generate a forecast for this title.")
