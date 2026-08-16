"""
Upcoming Movies
=================
Identical layout to Popular Movies (utils/movie_grid.py) — only the fetch
function differs, restricting results to a forward-looking release window.
"""

import streamlit as st
from utils.styling import inject_theme
from utils.navbar import render_navbar
from utils import tmdb_client
from utils.movie_grid import render_movie_grid

st.set_page_config(page_title="Upcoming Movies", page_icon="🎬", layout="wide")
inject_theme()
render_navbar(active="Movies_upcoming")

try:
    tmdb_api_key = st.secrets["TMDB_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("TMDB_API_KEY not found. Add it to .streamlit/secrets.toml.")
    st.stop()


def _fetch(api_key, sort_by, genre_id, page):
    return tmdb_client.discover_movies(api_key, sort_by=sort_by, genre_id=genre_id,
                                        page=page, upcoming_only=True)


render_movie_grid("Upcoming Movies", _fetch, tmdb_api_key)
