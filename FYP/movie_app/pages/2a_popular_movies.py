"""
Popular Movies
================
Matches the TMDB reference screenshot layout — sort + genre filter, poster
grid, pagination. Same layout as Upcoming Movies (utils/movie_grid.py).
"""

import streamlit as st
from utils.styling import inject_theme
from utils.navbar import render_navbar
from utils import tmdb_client
from utils.movie_grid import render_movie_grid

st.set_page_config(page_title="Popular Movies", page_icon="🎬", layout="wide")
inject_theme()
render_navbar(active="Movies_popular")

try:
    tmdb_api_key = st.secrets["TMDB_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("TMDB_API_KEY not found. Add it to .streamlit/secrets.toml.")
    st.stop()


def _fetch(api_key, sort_by, genre_id, page):
    return tmdb_client.discover_movies(api_key, sort_by=sort_by, genre_id=genre_id,
                                        page=page, upcoming_only=False)


render_movie_grid("Popular Movies", _fetch, tmdb_api_key)
