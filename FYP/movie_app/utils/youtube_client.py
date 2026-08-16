"""
YouTube Single-Movie Trailer Fetcher
========================================
Adapted from youtube_sentiment_scraper.py's YouTubeCollector class — same
trailer search/scoring + comment sentiment logic, stripped of the batch
quota tracker and checkpoint/resume machinery since this runs once per
on-demand lookup rather than over a whole CSV.
"""

import time
from datetime import datetime, timedelta, timezone
import requests
import streamlit as st

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

BASE_URL = "https://www.googleapis.com/youtube/v3"
MAX_COMMENTS = 100
LOOKBACK_DAYS = 180
MAX_RETRIES = 3

OFFICIAL_KEYWORDS = [
    "official", "movies", "pictures", "studios", "entertainment",
    "films", "warner", "disney", "universal", "paramount", "sony",
    "marvel", "netflix", "amazon", "apple", "lionsgate", "a24", "pixar",
    "mgm", "20th century", "miramax", "legendary", "neon",
]
SPAM_KEYWORDS = [
    "reaction", "review", "breakdown", "explained", "analysis",
    "fan made", "fan trailer", "reupload", "re-upload",
    "hindi", "dubbed", "tamil", "telugu", "best moments", "all scenes",
]


def _get(endpoint, params, api_key, retries=MAX_RETRIES):
    """Same retry + 429 back-off as YouTubeCollector._get(), minus the
    daily quota ledger (a single on-demand lookup costs ~101-201 units,
    trivial against the 10,000/day free tier — no need to track it here)."""
    params = {**params, "key": api_key}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(10 * (2 ** (attempt - 1)))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response else 0
            if code == 403:
                return None  # quota exceeded or comments disabled — caller handles gracefully
            if code == 404:
                return None
            time.sleep(2 * attempt)
        except Exception:
            time.sleep(2 * attempt)
    return None


def _score_video(video, release_dt):
    """Identical scoring rubric to youtube_sentiment_scraper.py's _score_video."""
    score = 0.0
    snippet = video.get("snippet", {})
    stats = video.get("statistics", {})
    title = snippet.get("title", "").lower()
    channel = snippet.get("channelTitle", "").lower()
    pub_str = snippet.get("publishedAt", "")

    if any(kw in channel for kw in OFFICIAL_KEYWORDS):
        score += 50
    if "official" in title:
        score += 30
    if "trailer" in title:
        score += 20
    if any(kw in title for kw in SPAM_KEYWORDS):
        score -= 50

    if pub_str:
        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            release_utc = release_dt.replace(tzinfo=timezone.utc)
            earliest = release_utc - timedelta(days=LOOKBACK_DAYS)
            if earliest <= pub_dt <= release_utc:
                score += 10
            elif pub_dt > release_utc:
                score -= 20
        except Exception:
            pass

    try:
        views = int(stats.get("viewCount", 0))
        score += min(views / 1_000_000, 10)
    except Exception:
        pass

    return score


def _search_trailer(movie_title, release_dt, api_key, year=""):
    """Same two-query (specific then broad) trailer search as the original
    search_trailer() — tries the high-confidence query first to save quota."""
    queries = [
        f'{movie_title} official trailer{" " + year if year else ""}',
        f'{movie_title} trailer',
    ]
    release_utc = release_dt.replace(tzinfo=timezone.utc)
    earliest = release_utc - timedelta(days=LOOKBACK_DAYS + 30)
    after_str = earliest.strftime("%Y-%m-%dT%H:%M:%SZ")
    before_str = (release_utc + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

    best_video, best_score = None, -999

    for query in queries:
        data = _get("search", {
            "part": "snippet", "q": query, "type": "video", "order": "relevance",
            "publishedAfter": after_str, "publishedBefore": before_str, "maxResults": 5,
        }, api_key)
        if not data:
            continue

        video_ids = [item["id"]["videoId"] for item in data.get("items", [])
                     if item.get("id", {}).get("videoId")]
        if not video_ids:
            continue

        stats_data = _get("videos", {"part": "snippet,statistics", "id": ",".join(video_ids)}, api_key)
        if not stats_data:
            continue

        for video in stats_data.get("items", []):
            s = _score_video(video, release_dt)
            if s > best_score:
                best_score, best_video = s, video

        if best_video and best_score >= 50:
            break
        time.sleep(0.5)

    return best_video, round(best_score, 2)


def _get_comments(video_id, release_dt, api_key, max_comments=MAX_COMMENTS):
    """Same pre-release-only comment pagination as the original get_comments()."""
    comments = []
    page_token = None
    release_utc = release_dt.replace(tzinfo=timezone.utc)

    while len(comments) < max_comments:
        params = {"part": "snippet", "videoId": video_id, "order": "relevance",
                   "maxResults": 100, "textFormat": "plainText"}
        if page_token:
            params["pageToken"] = page_token

        data = _get("commentThreads", params, api_key)
        if not data:
            break

        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            pub_str = top.get("publishedAt", "")
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                if pub_dt > release_utc:
                    continue
            except Exception:
                continue
            comments.append({
                "text": top.get("textDisplay", ""),
                "like_count": int(top.get("likeCount", 0)),
            })
            if len(comments) >= max_comments:
                break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return comments


def _compute_sentiment(comments, analyzer):
    """Identical aggregation to compute_sentiment() in the original script,
    including the like-weighted compound score."""
    empty = {
        "comment_count": 0, "avg_compound": None, "avg_positive": None,
        "avg_negative": None, "avg_neutral": None, "positive_ratio": None,
        "negative_ratio": None, "total_likes": 0, "weighted_compound": None,
    }
    if not comments or not analyzer:
        return empty

    compounds, positives, negatives, neutrals = [], [], [], []
    total_likes, weighted_sum, weight_total = 0, 0.0, 0.0

    for c in comments:
        text = c.get("text", "").strip()
        if not text:
            continue
        s = analyzer.polarity_scores(text)
        compounds.append(s["compound"])
        positives.append(s["pos"])
        negatives.append(s["neg"])
        neutrals.append(s["neu"])
        likes = c.get("like_count", 0)
        total_likes += likes
        w = max(likes, 1)
        weighted_sum += s["compound"] * w
        weight_total += w

    n = len(compounds)
    if n == 0:
        return empty

    pos_count = sum(1 for c in compounds if c >= 0.05)
    neg_count = sum(1 for c in compounds if c <= -0.05)

    return {
        "comment_count": n,
        "avg_compound": round(sum(compounds) / n, 4),
        "avg_positive": round(sum(positives) / n, 4),
        "avg_negative": round(sum(negatives) / n, 4),
        "avg_neutral": round(sum(neutrals) / n, 4),
        "positive_ratio": round(pos_count / n, 4),
        "negative_ratio": round(neg_count / n, 4),
        "total_likes": total_likes,
        "weighted_compound": round(weighted_sum / weight_total, 4) if weight_total else 0,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_youtube_sentiment(movie_title: str, release_date: str, year: str = "") -> dict:
    """
    Find the trailer for one movie and return view/like stats + pre-release
    comment sentiment. release_date: 'YYYY-MM-DD' string. Returns the same
    field shape as a row in youtube_sentiment_results.csv (renamed at the
    feature-engineering step to avoid clashing with Reddit's identical names,
    same as the notebook's YOUTUBE_RENAME does).
    """
    analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
    api_key = st.secrets.get("YOUTUBE_API_KEY", "")

    try:
        release_dt = datetime.strptime(release_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        release_dt = datetime.utcnow()

    video, score = _search_trailer(movie_title, release_dt, api_key, year)

    if not video:
        return {
            "video_id": "", "video_title": "", "video_view_count": 0,
            "likes": 0, "like_ratio": 0, "trailer_score": 0,
            "notes": "no trailer found",
            **_compute_sentiment([], analyzer),
        }

    vid_id = video["id"]
    snippet = video["snippet"]
    stats = video.get("statistics", {})
    views = int(stats.get("viewCount", 0))
    likes = int(stats.get("likeCount", 0))
    like_ratio = likes / likes if likes else 0  # public dislike counts removed; matches notebook's constant-1.0 note

    comments = _get_comments(vid_id, release_dt, api_key)
    sentiment = _compute_sentiment(comments, analyzer)

    return {
        "video_id": vid_id,
        "video_title": snippet.get("title", ""),
        "channel_name": snippet.get("channelTitle", ""),
        "video_view_count": views,
        "likes": likes,
        "like_ratio": round(like_ratio, 4),
        "trailer_score": score,
        "notes": "" if sentiment["comment_count"] > 0 else "comments disabled or all post-release",
        **sentiment,
    }
