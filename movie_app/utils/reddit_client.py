"""
Reddit Single-Movie Sentiment Fetcher (Arctic Shift)
========================================================
Adapted from reddit_sentiment_scraper.py — same Arctic Shift query +
VADER sentiment logic, but fetches ONE movie synchronously for the
Add New Movie page instead of looping over a whole CSV.

Fix (see get_reddit_sentiment docstring): Arctic Shift's own API docs
state the `title` keyword-search parameter is "not supported with very
active users or subreddits" — and r/movies is about as active as
subreddits get. That restriction returning zero results (rather than an
error) is the most likely reason title-filtered search against r/movies
started coming back empty for every title, including ones that used to
work fine (e.g. Barbie) during the original data collection. Server-side
title filtering is now tried first (cheap, one request) and, if it comes
back empty, falls back to paging through r/movies by date only and
filtering for the movie client-side.
"""

import re
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
MAX_FALLBACK_PAGES = 8  # ~800 posts max scanned client-side per movie

# A descriptive User-Agent is good practice for a free public API like this
# one and cheap insurance against any UA-based filtering on their end.
USER_AGENT = "MoviePredictionFYP/1.0 (academic research; contact via GitHub issue)"


def clean_title(title: str) -> str:
    """Same simplification as reddit_sentiment_scraper.py — broadens
    substring match recall on Arctic Shift's title search."""
    for char in [":", "-", "–", "&"]:
        title = title.replace(char, " ")
    return " ".join(title.split())


def _significant_words(title: str) -> list:
    """Words worth matching on — drops very short tokens (a, of, the...)
    that would make the client-side filter too loose."""
    return [w.lower() for w in re.findall(r"[a-zA-Z0-9']+", title) if len(w) > 2]


def _mentions_movie(text: str, title_words: list) -> bool:
    """Client-side stand-in for Arctic Shift's server-side title keyword
    search (which ANDs all words together) — requires every significant
    title word to appear somewhere in the post's title/selftext."""
    if not title_words:
        return False
    text_lower = text.lower()
    return all(w in text_lower for w in title_words)


def _request_page(params):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(ARCTIC_SHIFT_URL, params=params, timeout=30,
                                 headers={"User-Agent": USER_AGENT})
            if resp.status_code == 200:
                return resp.json().get("data", [])
            if resp.status_code == 429:
                time.sleep(5 * attempt)
                continue
        except requests.exceptions.Timeout:
            time.sleep(5)
        except Exception:
            pass
        if attempt < MAX_RETRIES:
            time.sleep(2.0 * attempt)
    return []


def _fetch_posts(movie_title, after_ts, before_ts, limit=POST_LIMIT):
    """Fast path: ask Arctic Shift to filter by title server-side. Cheap
    (one request) but documented as unsupported for very active subreddits
    — r/movies likely qualifies, so this may legitimately come back empty
    even for movies with plenty of real discussion."""
    params = {
        "subreddit": "movies",
        "title": movie_title,
        "after": after_ts,
        "before": before_ts,
        "limit": limit,
        "sort": "desc",
    }
    return _request_page(params)


def _fetch_posts_broad(movie_title, after_ts, before_ts, target_count=POST_LIMIT,
                        max_pages=MAX_FALLBACK_PAGES):
    """
    Fallback used when the title-filtered fast path returns nothing. Pages
    through ALL r/movies posts in the date window (no title filter, so the
    "very active subreddit" restriction doesn't apply) and keeps only the
    ones that actually mention the movie, matched client-side.
    """
    title_words = _significant_words(movie_title)
    if not title_words:
        return []

    matched = []
    cursor_before = before_ts

    for _ in range(max_pages):
        page = _request_page({
            "subreddit": "movies",
            "after": after_ts,
            "before": cursor_before,
            "limit": 100,
            "sort": "desc",
        })
        if not page:
            break

        for post in page:
            text = f"{post.get('title') or ''} {post.get('selftext') or ''}"
            if _mentions_movie(text, title_words):
                matched.append(post)

        oldest = min((p.get("created_utc", cursor_before) for p in page), default=cursor_before)
        if len(page) < 100 or oldest >= cursor_before:
            break  # reached the start of the window
        cursor_before = oldest

        if len(matched) >= target_count:
            break

    return matched[:target_count]


def _compute_sentiment(posts, analyzer):
    """Identical aggregation logic to reddit_sentiment_scraper.py's
    compute_sentiment(), so avg_compound/positive_ratio/etc. mean the
    same thing here as they do in the training CSV.

    Also computes weighted_compound, an upvote-weighted compound score,
    for parity with YouTube's like-weighted compound score. Reddit's
    `score` field can be zero or negative (net upvotes minus downvotes,
    also subject to Reddit's anti-spam vote fuzzing), unlike YouTube's
    like count which is never negative, so the weight here floors at 1
    with max(score, 0) + 1 rather than YouTube's max(likes, 1). This
    keeps a downvoted post's sentiment in the average instead of giving
    it a negative or zero weight, while still letting highly-upvoted
    posts pull the average toward their sentiment more strongly.
    """
    if not posts or not analyzer:
        return {
            "post_count": 0, "avg_compound": None, "avg_positive": None,
            "avg_negative": None, "avg_neutral": None, "positive_ratio": None,
            "negative_ratio": None, "total_score": 0, "avg_num_comments": 0,
            "weighted_compound": None, "samples": [], "all_texts": [],
        }

    compounds, positives, negatives, neutrals = [], [], [], []
    total_score, total_comments = 0, 0
    weighted_sum, weight_total = 0.0, 0.0
    scored_texts = []  # (text, compound) — used for sample posts + word cloud

    for post in posts:
        text = ((post.get("title") or "") + " " + (post.get("selftext") or "")).strip()
        if not text:
            continue
        scores = analyzer.polarity_scores(text)
        compounds.append(scores["compound"])
        positives.append(scores["pos"])
        negatives.append(scores["neg"])
        neutrals.append(scores["neu"])
        score = int(post.get("score", 0))
        total_score += score
        total_comments += int(post.get("num_comments", 0))
        scored_texts.append((text, scores["compound"]))
        w = max(score, 0) + 1
        weighted_sum += scores["compound"] * w
        weight_total += w

    n = len(compounds)
    if n == 0:
        return {
            "post_count": len(posts), "avg_compound": 0, "avg_positive": 0,
            "avg_negative": 0, "avg_neutral": 0, "positive_ratio": 0,
            "negative_ratio": 0, "total_score": total_score, "avg_num_comments": 0,
            "weighted_compound": None, "samples": [], "all_texts": [],
        }

    pos_count = sum(1 for c in compounds if c >= 0.05)
    neg_count = sum(1 for c in compounds if c <= -0.05)

    # Surface the 3 most sentiment-extreme posts (by |compound|) as a quick
    # preview of what was actually fetched — shown in the "Sample sentiment
    # fetched" expander on the Movie Details page.
    top_samples = sorted(scored_texts, key=lambda t: abs(t[1]), reverse=True)[:3]
    samples = [{"text": text, "compound": compound} for text, compound in top_samples]

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
        "weighted_compound": round(weighted_sum / weight_total, 4) if weight_total else 0,
        "samples": samples,
        "all_texts": [t for t, _ in scored_texts],
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_reddit_sentiment(movie_title: str, release_date: str) -> dict:
    """
    Fetch r/movies pre-release sentiment for one movie.
    release_date: 'YYYY-MM-DD' string (as returned by TMDB).
    Returns the same field shape as reddit_sentiment_results.csv rows.

    Tries the cheap server-side title-filtered search first; if that comes
    back with zero posts, falls back to scanning r/movies by date and
    matching client-side, since Arctic Shift documents `title` keyword
    search as unsupported on very active subreddits (r/movies included) —
    which otherwise silently looks identical to "no discussion found".
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
    after_ts, before_ts = int(after_dt.timestamp()), int(before_dt.timestamp())

    search_title = clean_title(movie_title)
    posts = _fetch_posts(search_title, after_ts, before_ts)
    fetch_method = "title_filter"

    if not posts:
        posts = _fetch_posts_broad(search_title, after_ts, before_ts)
        fetch_method = "broad_scan"

    sentiment = _compute_sentiment(posts, analyzer)
    sentiment["raw_post_count_fetched"] = len(posts)
    sentiment["fetch_method"] = fetch_method
    return sentiment