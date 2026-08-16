"""
Movie Cache (CSV)
==================
Appends newly scraped + predicted movies to a CSV cache, so the next time
someone searches the same title, the app skips scraping entirely and loads
the saved row instantly — matching the Chapter 3 system flowchart's
cache-check-first behaviour.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

CACHE_PATH = Path(__file__).parent.parent / "data" / "movie_predictions.csv"

CACHE_COLUMNS = [
    "Title", "Release_Date", "Budget", "Revenue", "Rating",
    "pred_revenue", "pred_rating", "success_label",
    "poster_path", "director", "genre", "scraped_at",
]


def _ensure_cache_exists():
    if not CACHE_PATH.exists():
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=CACHE_COLUMNS).to_csv(CACHE_PATH, index=False)


def load_cache() -> pd.DataFrame:
    _ensure_cache_exists()
    return pd.read_csv(CACHE_PATH)


def find_cached(title: str):
    """Case-insensitive exact-title lookup. Returns a dict row or None."""
    df = load_cache()
    if df.empty:
        return None
    match = df[df["Title"].str.lower() == title.strip().lower()]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def append_movie(tmdb_data: dict, prediction: dict):
    """Append one newly predicted movie to the cache CSV."""
    _ensure_cache_exists()
    row = {
        "Title": tmdb_data.get("title", ""),
        "Release_Date": tmdb_data.get("release_date", ""),
        "Budget": tmdb_data.get("budget", 0),
        "Revenue": tmdb_data.get("revenue", 0),  # 0 for unreleased movies
        "Rating": tmdb_data.get("vote_average", 0),
        "pred_revenue": prediction.get("pred_revenue"),
        "pred_rating": prediction.get("pred_rating"),
        "success_label": prediction.get("success_label"),
        "poster_path": tmdb_data.get("poster_path", ""),
        "director": tmdb_data.get("director", ""),
        "genre": tmdb_data.get("genre", ""),
        "scraped_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    df = load_cache()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CACHE_PATH, index=False)
    return row
