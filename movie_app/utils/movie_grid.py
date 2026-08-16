"""
Shared Movie Grid Layout
==========================
Used by both pages/2a_popular_movies.py and pages/2b_upcoming_movies.py so
the two stay visually identical (per the user's request) without duplicating
the grid/filter/sort code twice.
"""

import streamlit as st
from utils import tmdb_client

GENRE_OPTIONS = ["All", "Action", "Adventure", "Animation", "Comedy", "Crime",
                  "Documentary", "Drama", "Family", "Fantasy", "History",
                  "Horror", "Music", "Mystery", "Romance", "Science Fiction",
                  "Thriller", "War", "Western"]

SORT_OPTIONS = {
    "Most popular": "popularity.desc",
    "Release date (newest)": "release_date.desc",
    "Release date (oldest)": "release_date.asc",
    "Rating (highest)": "vote_average.desc",
}


def render_movie_grid(page_title, fetch_fn, api_key, page_size=15):
    """
    fetch_fn(api_key, sort_by, genre_id, page) -> list of movie dicts.
    Renders sort + genre filter controls, a responsive poster grid capped at
    page_size, and pagination.
    """
    st.markdown(f"### {page_title}")

    filt1, filt2, filt3 = st.columns([2, 2, 5])
    with filt1:
        sort_label = st.selectbox("Sort", list(SORT_OPTIONS.keys()), key=f"{page_title}_sort")
    with filt2:
        genre_label = st.selectbox("Genre", GENRE_OPTIONS, key=f"{page_title}_genre")

    page_num = st.session_state.get(f"{page_title}_page", 1)

    genre_id = tmdb_client.GENRE_NAME_TO_ID.get(genre_label) if genre_label != "All" else None
    movies = fetch_fn(api_key, SORT_OPTIONS[sort_label], genre_id, page_num)
    movies = movies[:page_size]

    if not movies:
        st.caption("No movies match this filter.")
        return

    n_cols = 5
    for row_start in range(0, len(movies), n_cols):
        cols = st.columns(n_cols)
        for j, m in enumerate(movies[row_start:row_start + n_cols]):
            with cols[j]:
                poster = tmdb_client.poster_url(m.get("poster_path"))
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='background:#16172a;border-radius:8px;aspect-ratio:2/3;"
                        "display:flex;align-items:center;justify-content:center;color:#5a5a72;'>"
                        "No poster</div>", unsafe_allow_html=True,
                    )
                st.caption(f"**{m['title']}**  \n{m.get('release_date', 'TBA')}")
                if st.button("View", key=f"{page_title}_{row_start}_{j}", use_container_width=True):
                    st.session_state["selected_movie_id"] = m["id"]
                    st.session_state["selected_movie_title"] = m["title"]
                    st.switch_page("pages/2_movie_details.py")

    nav1, nav2, nav3 = st.columns([1, 1, 6])
    with nav1:
        if page_num > 1 and st.button("← Previous", key=f"{page_title}_prev"):
            st.session_state[f"{page_title}_page"] = page_num - 1
            st.rerun()
    with nav2:
        if st.button("Next →", key=f"{page_title}_next"):
            st.session_state[f"{page_title}_page"] = page_num + 1
            st.rerun()
