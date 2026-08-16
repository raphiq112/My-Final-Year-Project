"""
Homepage — Hero + Trending + Popular Trailers
==================================================
Wide TMDB-style layout: nav bar, hero welcome container with search,
horizontally-scrolling Trending row (15+ movies), and a Popular Trailers
row sourced from youtube_sentiment_results.csv (top 10 by views).
"""

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

# ── Hero / welcome container ──────────────────────────────────────────────────────
st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(127,119,221,0.15), rgba(93,202,165,0.1)), url('https://www.polygon.com/spider-man-brand-new-day-trailer-2-hulk-sadie-sink/');
            background-size: cover;
            background-position: center;
            border-radius: 16px; padding: 48px 32px; text-align: center; margin-bottom: 28px;
            border: 0.5px solid rgba(255,255,255,0.08);">
    <h1 style="font-size: 32px; color: #f4f4fa; font-weight: 600; margin-bottom: 8px;">Welcome.</h1>
    <p style="color: #a0a0b8; font-size: 15px; margin-bottom: 0;">
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
            for movie in results[:5]:
                c1, c2 = st.columns([1, 6])
                with c1:
                    poster = tmdb_client.poster_url(movie["poster_path"])
                    if poster:
                        st.image(poster, width=60)
                with c2:
                    if st.button(f"{movie['title']} ({movie['release_date'][:4] if movie['release_date'] else 'TBA'})",
                                 key=f"hero_search_{movie['id']}", use_container_width=True):
                        st.session_state["selected_movie_id"] = movie["id"]
                        st.session_state["selected_movie_title"] = movie["title"]
                        st.switch_page("pages/2_movie_details.py")
        else:
            st.caption("No matches found.")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


def _poster_scroller(items, key_prefix, subtitle_fn, link_fn=None):
    """
    Renders a horizontally-scrolling row of poster cards. Streamlit has no
    native horizontal scroll container, so this uses raw flex CSS with
    overflow-x:auto for the VISUAL row. HTML/CSS can't call st.switch_page,
    so an expander with real buttons underneath provides the actual click
    behaviour — same constraint as the nav bar.
    """
    cards_html = ""
    for item in items:
        poster = item.get("poster_url") or ""
        title = item.get("title", "")
        sub = subtitle_fn(item)
        cards_html += (
            f'<div style="flex:0 0 140px; scroll-snap-align:start;">'
            f'<img src="{poster}" style="width:140px; height:198px; object-fit:cover; '
            f'border-radius:8px; background:#16172a;">'
            f'<p style="color:#f4f4fa; font-size:12px; margin:6px 0 0; font-weight:500; '
            f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis; width:140px;">{title}</p>'
            f'<p style="color:#6b6b80; font-size:11px; margin:2px 0 0;">{sub}</p>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:flex; gap:14px; overflow-x:auto; scroll-snap-type:x mandatory; '
        f'padding-bottom:10px;">{cards_html}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Open a title from this row"):
        n_cols = 5
        for row_start in range(0, len(items), n_cols):
            cols = st.columns(n_cols)
            for j, item in enumerate(items[row_start:row_start + n_cols]):
                with cols[j]:
                    if link_fn:
                        st.link_button(item["title"][:18], link_fn(item), use_container_width=True)
                    else:
                        if st.button(item["title"][:18], key=f"{key_prefix}_{row_start}_{j}",
                                     use_container_width=True):
                            st.session_state["selected_movie_id"] = item.get("id")
                            st.session_state["selected_movie_title"] = item["title"]
                            st.switch_page("pages/2_movie_details.py")


# ── Trending section ────────────────────────────────────────────────────────────
st.markdown("### Trending")
tab_today, tab_week = st.tabs(["Today", "This Week"])

with tab_today:
    trending = tmdb_client.get_trending(tmdb_api_key, window="day", max_results=15)
    if trending:
        items = [{"id": m["id"], "title": m["title"],
                   "poster_url": tmdb_client.poster_url(m["poster_path"]) or "",
                   "release_date": m.get("release_date", "")} for m in trending]
        _poster_scroller(items, "trend_day",
                          subtitle_fn=lambda it: it.get("release_date", "TBA") or "TBA")
    else:
        st.caption("Trending data unavailable right now.")

with tab_week:
    trending_wk = tmdb_client.get_trending(tmdb_api_key, window="week", max_results=15)
    if trending_wk:
        items = [{"id": m["id"], "title": m["title"],
                   "poster_url": tmdb_client.poster_url(m["poster_path"]) or "",
                   "release_date": m.get("release_date", "")} for m in trending_wk]
        _poster_scroller(items, "trend_week",
                          subtitle_fn=lambda it: it.get("release_date", "TBA") or "TBA")
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
            trailer_html += (
                f'<div style="flex:0 0 200px; scroll-snap-align:start;">'
                f'<img src="{thumb}" style="width:200px; height:113px; object-fit:cover; '
                f'border-radius:8px; background:#16172a;">'
                f'<p style="color:#f4f4fa; font-size:12px; margin:6px 0 0; font-weight:500; '
                f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis; width:200px;">{title}</p>'
                f'<p style="color:#6b6b80; font-size:11px; margin:2px 0 0;">👁 {views_fmt} views</p>'
                f'</div>'
            )
        st.markdown(
            f'<div style="display:flex; gap:14px; overflow-x:auto; scroll-snap-type:x mandatory; '
            f'padding-bottom:10px;">{trailer_html}</div>',
            unsafe_allow_html=True,
        )

        st.caption("Select a trailer to play it here, or open directly on YouTube.")
        n_cols = 5
        rows = top10.reset_index(drop=True)
        for row_start in range(0, len(rows), n_cols):
            cols = st.columns(n_cols)
            for j in range(n_cols):
                idx = row_start + j
                if idx >= len(rows):
                    continue
                r = rows.iloc[idx]
                vid = r.get("video_id", "")
                with cols[j]:
                    if vid and st.button(f"▶ {str(r.get('Title',''))[:16]}",
                                          key=f"play_{idx}", use_container_width=True):
                        st.session_state["now_playing_video_id"] = vid
                        st.session_state["now_playing_title"] = r.get("Title", "")

        if st.session_state.get("now_playing_video_id"):
            st.markdown(f"**Now playing: {st.session_state.get('now_playing_title','')}**")
            st.video(f"https://www.youtube.com/watch?v={st.session_state['now_playing_video_id']}")
    else:
        st.caption("youtube_sentiment_results.csv is missing a 'video_id' / 'video_view_count' column.")
