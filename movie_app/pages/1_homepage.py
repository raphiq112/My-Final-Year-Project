"""
Homepage — Hero + Trending + Popular Trailers
==================================================
Wide TMDB-style layout: nav bar, hero welcome container with search,
horizontally-scrolling Trending row, and a Popular Trailers row sourced
from youtube_sentiment_results.csv (top 10 by views).

Changes in this revision:
  - Hero background now reads assets/homepage_bg.(jpg|png) if present,
    base64-encoded inline (same trick app.py uses for the landing page),
    instead of the old broken `url('https://www.polygon.com/...')` (that
    was a webpage link, not an image, so it never actually rendered).
  - Posters (Trending + search results) are now directly clickable —
    wrapped in <a href="/movie_details?movie_id=..."> — no separate
    "open" button/expander anymore. movie_details.py reads ?movie_id=
    from the URL as a fallback to session_state.
  - Popular Trailers thumbnails are directly clickable to play inline —
    wrapped in <a href="/homepage?play_video=..."> which updates the
    query string on the *same* page, no separate ▶ buttons anymore.
"""

import base64
import streamlit as st
import pandas as pd
from pathlib import Path
from utils.styling import inject_theme
from utils.navbar import render_navbar
from utils import tmdb_client

st.set_page_config(page_title="Homepage", page_icon="🎬", layout="wide")
inject_theme()
render_navbar(active="Home")

try:
    tmdb_api_key = st.secrets["TMDB_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("TMDB_API_KEY not found. Add it to .streamlit/secrets.toml.")
    st.stop()

DATA_DIR = Path(__file__).parent.parent / "data"
ASSETS_DIR = Path(__file__).parent.parent / "assets"


# ── Custom hero background image (base64 data URI) ────────────────────────────────
def _bg_data_uri():
    """Looks for assets/homepage_bg.jpg or .png. Drop your own image there
    to replace the hero background — no code changes needed."""
    for name in ("spiderman_post.avif", "homepage_bg1.jpg", "homepage_bg2.jpg"):
        path = ASSETS_DIR / name
        if path.exists():
            ext = path.suffix.lstrip(".").lower()
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            data = base64.b64encode(path.read_bytes()).decode()
            return f"data:image/{mime};base64,{data}"
    return None


_hero_img = _bg_data_uri()
_hero_bg_layer = f", url('{_hero_img}')" if _hero_img else ""

# ── Hero / welcome container ──────────────────────────────────────────────────────
st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(127,119,221,0.55), rgba(13,14,26,0.80)){_hero_bg_layer};
            background-size: cover;
            background-position: center;
            border-radius: 16px; padding: 48px 32px; text-align: center; margin-bottom: 28px;
            border: 0.5px solid rgba(255,255,255,0.08);">
    <h1 style="font-size: 32px; color: #f4f4fa; font-weight: 600; margin-bottom: 8px;">Welcome.</h1>
    <p style="color: #e0e0f0; font-size: 15px; margin-bottom: 0;">
        See a movie's chance of success before it ever hits theaters.
    </p>
</div>
""", unsafe_allow_html=True)

search_col1, search_col2, search_col3 = st.columns([1, 3, 1])
with search_col2:
    query = st.text_input("", placeholder="🔍  Search for a movie...", label_visibility="collapsed")
    if query:
        results = tmdb_client.search_movies(query, tmdb_api_key)
        if results:
            rows_html = ""
            for movie in results[:5]:
                poster = tmdb_client.poster_url(movie["poster_path"]) or ""
                title = movie["title"]
                year = movie["release_date"][:4] if movie.get("release_date") else "TBA"
                rows_html += (
                    f'<a href="/movie_details?movie_id={movie["id"]}" target="_self" '
                    f'style="text-decoration:none; display:flex; align-items:center; gap:14px; '
                    f'padding:8px 6px; border-radius:8px; cursor:pointer;">'
                    f'<img src="{poster}" style="width:44px;height:64px;object-fit:cover;'
                    f'border-radius:6px;background:#16172a;flex-shrink:0;">'
                    f'<span style="color:#f4f4fa; font-size:14px; font-weight:500;">{title} ({year})</span>'
                    f'</a>'
                )
            st.markdown(f'<div style="max-width:640px;margin:6px auto 0;">{rows_html}</div>',
                        unsafe_allow_html=True)
        else:
            st.caption("No matches found.")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


def _poster_scroller(items, subtitle_fn):
    """
    Renders a horizontally-scrolling row of poster cards. Each poster is a
    clickable <a> pointing at /movie_details?movie_id=..., so clicking the
    poster itself navigates — no separate button/expander needed.
    """
    cards_html = ""
    for item in items:
        poster = item.get("poster_url") or ""
        title = item.get("title", "")
        sub = subtitle_fn(item)
        mid = item.get("id", "")
        cards_html += (
            f'<a href="/movie_details?movie_id={mid}" target="_self" '
            f'style="text-decoration:none; cursor:pointer;">'
            f'<div style="flex:0 0 140px; scroll-snap-align:start;">'
            f'<img src="{poster}" style="width:140px; height:198px; object-fit:cover; '
            f'border-radius:8px; background:#16172a;">'
            f'<p style="color:#f4f4fa; font-size:12px; margin:6px 0 0; font-weight:500; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis; width:140px;">{title}</p>'
            f'<p style="color:#6b6b80; font-size:11px; margin:2px 0 0;">{sub}</p>'
            f'</div></a>'
        )
    st.markdown(
        f'<div style="display:flex; gap:14px; overflow-x:auto; scroll-snap-type:x mandatory; '
        f'padding-bottom:10px;">{cards_html}</div>',
        unsafe_allow_html=True,
    )


# ── Trending section ────────────────────────────────────────────────────────────
st.markdown("### Trending")
tab_today, tab_week = st.tabs(["Today", "This Week"])

with tab_today:
    trending = tmdb_client.get_trending(tmdb_api_key, window="day", max_results=15)
    if trending:
        items = [{"id": m["id"], "title": m["title"],
                   "poster_url": tmdb_client.poster_url(m["poster_path"]) or "",
                   "release_date": m.get("release_date", "")} for m in trending]
        _poster_scroller(items, subtitle_fn=lambda it: it.get("release_date", "TBA") or "TBA")
    else:
        st.caption("Trending data unavailable right now.")

with tab_week:
    trending_wk = tmdb_client.get_trending(tmdb_api_key, window="week", max_results=15)
    if trending_wk:
        items = [{"id": m["id"], "title": m["title"],
                   "poster_url": tmdb_client.poster_url(m["poster_path"]) or "",
                   "release_date": m.get("release_date", "")} for m in trending_wk]
        _poster_scroller(items, subtitle_fn=lambda it: it.get("release_date", "TBA") or "TBA")
    else:
        st.caption("Trending data unavailable right now.")

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── Popular Trailers section (from youtube_sentiment_results.csv) ─────────────────
st.markdown("### Popular Trailers")

yt_csv_path = DATA_DIR / "youtube_sentiment_results.csv"
if not yt_csv_path.exists():
    st.caption(
        "youtube_sentiment_results.csv not found in /data — copy it from your "
        "notebook environment to enable this section."
    )
else:
    yt_df = pd.read_csv(yt_csv_path)

    # A click on a thumbnail sets ?play_video=<id> on this same page (no page
    # change), which we read back here to decide what's "now playing".
    played_qp = st.query_params.get("play_video")
    if played_qp:
        st.session_state["now_playing_video_id"] = played_qp
        if "video_id" in yt_df.columns:
            match = yt_df[yt_df["video_id"].astype(str) == str(played_qp)]
            if not match.empty:
                st.session_state["now_playing_title"] = match.iloc[0].get("Title", "")

    if "video_view_count" in yt_df.columns:
        top10 = yt_df.dropna(subset=["video_view_count"]).sort_values(
            "video_view_count", ascending=False).head(10)

        trailer_html = ""
        for _, row in top10.iterrows():
            title = row.get("Title", "Unknown")
            views = row.get("video_view_count", 0)
            video_id = row.get("video_id", "")
            views_fmt = f"{views/1e6:.1f}M" if views >= 1e6 else f"{views:,.0f}"
            thumb = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg" if video_id else ""
            href = f"/homepage?play_video={video_id}" if video_id else "#"
            trailer_html += (
                f'<a href="{href}" target="_self" style="text-decoration:none; cursor:pointer;">'
                f'<div style="flex:0 0 200px; scroll-snap-align:start; position:relative;">'
                f'<img src="{thumb}" style="width:200px; height:113px; object-fit:cover; '
                f'border-radius:8px; background:#16172a;">'
                f'<div style="position:absolute; top:42px; left:88px; width:26px; height:26px; '
                f'border-radius:50%; background:rgba(13,14,26,0.75); display:flex; '
                f'align-items:center; justify-content:center; color:#fff; font-size:12px;">▶</div>'
                f'<p style="color:#f4f4fa; font-size:12px; margin:6px 0 0; font-weight:500; '
                f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis; width:200px;">{title}</p>'
                f'<p style="color:#6b6b80; font-size:11px; margin:2px 0 0;">👁 {views_fmt} views</p>'
                f'</div></a>'
            )
        st.markdown(
            f'<div style="display:flex; gap:14px; overflow-x:auto; scroll-snap-type:x mandatory; '
            f'padding-bottom:10px;">{trailer_html}</div>',
            unsafe_allow_html=True,
        )

        if st.session_state.get("now_playing_video_id"):
            st.markdown(f"**Now playing: {st.session_state.get('now_playing_title', '')}**")
            st.video(f"https://www.youtube.com/watch?v={st.session_state['now_playing_video_id']}")
    else:
        st.caption("youtube_sentiment_results.csv is missing a 'video_id' / 'video_view_count' column.")
