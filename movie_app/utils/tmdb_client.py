"""
TMDB Single-Movie Fetcher
============================
Adapted from tmdb_scraperOLD.py — that script bulk-discovers movies; this
module fetches ONE movie on demand for the Add New Movie page (search +
detail lookup), reusing the same genre/director/cast extraction logic.
"""

import requests
import streamlit as st

BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"

# Standard TMDB genre IDs (stable, documented by TMDB) — used by the
# Popular/Upcoming Movies pages' genre filter without needing an extra API
# call just to resolve names to ids.
GENRE_NAME_TO_ID = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80,
    "Documentary": 99, "Drama": 18, "Family": 10751, "Fantasy": 14,
    "History": 36, "Horror": 27, "Music": 10402, "Mystery": 9648,
    "Romance": 10749, "Science Fiction": 878, "Thriller": 53, "War": 10752,
    "Western": 37,
}


def poster_url(poster_path):
    """Build a full TMDB poster image URL, or None if no poster exists."""
    if not poster_path:
        return None
    return f"{POSTER_BASE}{poster_path}"


def _genres_map(api_key):
    resp = requests.get(f"{BASE_URL}/genre/movie/list",
                         params={"api_key": api_key, "language": "en-US"}, timeout=15)
    resp.raise_for_status()
    return {g["id"]: g["name"] for g in resp.json().get("genres", [])}


@st.cache_data(ttl=1800, show_spinner=False)
def get_trending(api_key, window="day", max_results=15):
    """Fetch TMDB's /trending/movie/{window} feed for the homepage Trending
    section. window is 'day' or 'week', matching TMDB's own toggle."""
    if not api_key:
        return []
    resp = requests.get(f"{BASE_URL}/trending/movie/{window}",
                         params={"api_key": api_key, "language": "en-US"},
                         timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])[:max_results]
    return [
        {
            "id": r["id"],
            "title": r.get("title", ""),
            "release_date": r.get("release_date", ""),
            "poster_path": r.get("poster_path"),
        }
        for r in results
    ]


@st.cache_data(ttl=1800, show_spinner=False)
def discover_movies(api_key, sort_by="popularity.desc", genre_id=None, page=1,
                     upcoming_only=False):
    """
    Powers both the Popular and Upcoming Movies pages via TMDB's /discover/movie,
    which supports sort_by + genre filtering natively (unlike /movie/popular or
    /movie/upcoming, which don't accept these params) — so both pages share one
    function and stay visually/behaviourally identical, per the user's request.
    """
    if not api_key:
        return []
    params = {
        "api_key": api_key, "language": "en-US", "page": page,
        "sort_by": sort_by, "include_adult": "false",
    }
    if genre_id:
        params["with_genres"] = genre_id
    if upcoming_only:
        from datetime import datetime, timedelta
        today = datetime.utcnow().date()
        params["primary_release_date.gte"] = today.strftime("%Y-%m-%d")
        params["primary_release_date.lte"] = (today + timedelta(days=180)).strftime("%Y-%m-%d")
    else:
        from datetime import datetime
        params["primary_release_date.lte"] = datetime.utcnow().strftime("%Y-%m-%d")

    resp = requests.get(f"{BASE_URL}/discover/movie", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("results", [])


@st.cache_data(ttl=3600, show_spinner=False)
def get_upcoming(api_key, max_results=10):
    """Fetch currently-upcoming releases from TMDB for the homepage poster
    grid's 'Upcoming Movies' row. Resolves genre_ids to names via one extra
    cached call to _genres_map (not one call per movie), so genre filtering
    on the homepage works for live results too, not just cached predictions."""
    if not api_key:
        return []
    genres_map = _genres_map(api_key)
    resp = requests.get(f"{BASE_URL}/movie/upcoming",
                         params={"api_key": api_key, "language": "en-US", "page": 1,
                                 "region": "US"},
                         timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])[:max_results]
    return [
        {
            "id": r["id"],
            "title": r.get("title", ""),
            "release_date": r.get("release_date", ""),
            "poster_path": r.get("poster_path"),
            "overview": r.get("overview", ""),
            "genre": ", ".join(filter(None, [genres_map.get(gid, "") for gid in r.get("genre_ids", [])])),
        }
        for r in results
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def search_movies(query, api_key, max_results=6):
    """Search TMDB by title. Returns a list of lightweight candidate dicts
    for a selectbox — full details are fetched separately once one is chosen."""
    if not query or not api_key:
        return []
    resp = requests.get(f"{BASE_URL}/search/movie",
                         params={"api_key": api_key, "query": query, "language": "en-US"},
                         timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])[:max_results]
    return [
        {
            "id": r["id"],
            "title": r.get("title", ""),
            "release_date": r.get("release_date", ""),
            "poster_path": r.get("poster_path"),
            "overview": r.get("overview", ""),
        }
        for r in results
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def get_movie_full(movie_id, api_key):
    """
    Fetch full detail for one movie — same fields as tmdb_scraperOLD.py's
    get_movie_details(), plus genre names and year, ready for
    feature_engineering.engineer_row()'s tmdb_data argument.
    """
    genres_map = _genres_map(api_key)

    resp = requests.get(
        f"{BASE_URL}/movie/{movie_id}",
        params={"api_key": api_key, "append_to_response": "credits", "language": "en-US"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    director = ""
    cast_members = []
    credits = data.get("credits", {})
    for crew_member in credits.get("crew", []):
        if crew_member.get("job") == "Director":
            director = crew_member.get("name", "")
            break
    for cast_member in credits.get("cast", [])[:5]:
        cast_members.append(cast_member.get("name", ""))

    production_companies = [c.get("name", "") for c in data.get("production_companies", [])]
    genre_names = [genres_map.get(g["id"], "") for g in data.get("genres", [])]
    release_date = data.get("release_date", "")
    year = int(release_date.split("-")[0]) if release_date else 0

    return {
        "title": data.get("title", ""),
        "year": year,
        "release_date": release_date,
        "budget": data.get("budget", 0),
        "revenue": data.get("revenue", 0),
        "vote_average": data.get("vote_average", 0),
        "vote_count": data.get("vote_count", 0),
        "runtime": data.get("runtime", 0),
        "production_companies": ", ".join(production_companies),
        "director": director,
        "cast": ", ".join(cast_members),
        "genre": ", ".join(filter(None, genre_names)),
        "poster_path": data.get("poster_path"),
        "overview": data.get("overview", ""),
    }
