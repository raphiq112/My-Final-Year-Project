"""
Reddit Single-Movie Sentiment Fetcher (Arctic Shift)
========================================================
Adapted from reddit_sentiment_scraper.py — same Arctic Shift query +
VADER sentiment logic, but fetches ONE movie synchronously for the
Add New Movie page instead of looping over a whole CSV.
"""

import time
from datetime import datetime, timedelta
import requests
import streamlit as st

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

ARCTIC_SHIFT_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
LOOKBACK_DAYS = 90
POST_LIMIT = 100
MAX_RETRIES = 3


def clean_title(title: str) -> str:
    """Same simplification as reddit_sentiment_scraper.py — broadens
    substring match recall on Arctic Shift's title search."""
    for char in [":", "-", "–", "&"]:
        title = title.replace(char, " ")
    return " ".join(title.split())


def _fetch_posts(movie_title, after_ts, before_ts, limit=POST_LIMIT):
    params = {
        "subreddit": "movies",
        "title": movie_title,
        "after": after_ts,
        "before": before_ts,
        "limit": limit,
        "sort": "desc",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(ARCTIC_SHIFT_URL, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except requests.exceptions.Timeout:
            time.sleep(5)
        except Exception:
            pass
        if attempt < MAX_RETRIES:
            time.sleep(2.0 * attempt)
    return []


def _compute_sentiment(posts, analyzer):
    """Identical aggregation logic to reddit_sentiment_scraper.py's
    compute_sentiment(), so avg_compound/positive_ratio/etc. mean the
    same thing here as they do in the training CSV."""
    if not posts or not analyzer:
        return {
            "post_count": 0, "avg_compound": None, "avg_positive": None,
            "avg_negative": None, "avg_neutral": None, "positive_ratio": None,
            "negative_ratio": None, "total_score": 0, "avg_num_comments": 0,
        }

    compounds, positives, negatives, neutrals = [], [], [], []
    total_score, total_comments = 0, 0

    for post in posts:
        text = ((post.get("title") or "") + " " + (post.get("selftext") or "")).strip()
        if not text:
            continue
        scores = analyzer.polarity_scores(text)
        compounds.append(scores["compound"])
        positives.append(scores["pos"])
        negatives.append(scores["neg"])
        neutrals.append(scores["neu"])
        total_score += int(post.get("score", 0))
        total_comments += int(post.get("num_comments", 0))

    n = len(compounds)
    if n == 0:
        return {
            "post_count": len(posts), "avg_compound": 0, "avg_positive": 0,
            "avg_negative": 0, "avg_neutral": 0, "positive_ratio": 0,
            "negative_ratio": 0, "total_score": total_score, "avg_num_comments": 0,
        }

    pos_count = sum(1 for c in compounds if c >= 0.05)
    neg_count = sum(1 for c in compounds if c <= -0.05)

    return {
        "post_count": len(posts),
        "avg_compound": round(sum(compounds) / n, 4),
        "avg_positive": round(sum(positives) / n, 4),
        "avg_negative": round(sum(negatives) / n, 4),
        "avg_neutral": round(sum(neutrals) / n, 4),
        "positive_ratio": round(pos_count / n, 4),
        "negative_ratio": round(neg_count / n, 4),
        "total_score": total_score,
        "avg_num_comments": round(total_comments / len(posts), 2),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_reddit_sentiment(movie_title: str, release_date: str) -> dict:
    """
    Fetch r/movies pre-release sentiment for one movie.
    release_date: 'YYYY-MM-DD' string (as returned by TMDB).
    Returns the same field shape as reddit_sentiment_results.csv rows.
    """
    analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

    try:
        release_dt = datetime.strptime(release_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        # Upcoming movies sometimes have provisional/missing dates on TMDB —
        # fall back to "today" as the window anchor so the lookback still works.
        release_dt = datetime.utcnow()

    after_dt = release_dt - timedelta(days=LOOKBACK_DAYS)
    before_dt = release_dt

    search_title = clean_title(movie_title)
    posts = _fetch_posts(search_title, int(after_dt.timestamp()), int(before_dt.timestamp()))
    sentiment = _compute_sentiment(posts, analyzer)
    sentiment["raw_post_count_fetched"] = len(posts)
    return sentiment
