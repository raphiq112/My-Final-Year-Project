"""
Feature Engineering — Single-Movie Inference Path
====================================================
Mirrors the feature engineering logic in movie_prediction_phases2_4.ipynb
(Phase 2) exactly, but refactored to transform ONE new movie row instead of
a full batch dataframe.

This module is the single source of truth for feature definitions, used by
both the Add New Movie and Custom Movie pages so predictions stay consistent
with how the models were trained.
"""

import json
import numpy as np
import pandas as pd
import joblib

# ── Feature lists (must match notebook ALL_FEATS exactly) ─────────────────────

MAJOR_STUDIOS = [
    'Warner Bros. Pictures', 'Universal Pictures', 'Paramount Pictures',
    'Marvel Studios', 'Walt Disney Pictures', 'Columbia Pictures',
    '20th Century Studios', 'Lionsgate', 'Netflix', 'Amazon Studios',
    'Apple Original Films', 'Sony Pictures'
]

NUMERIC_FEATS = ['log_budget', 'budget_known', 'Runtime', 'runtime_bucket',
                  'Vote_Count', 'is_major_studio', 'cast_size', 'Year']

REDDIT_FEATS = ['has_reddit', 'post_count', 'avg_compound', 'avg_positive',
                'avg_negative', 'avg_neutral', 'positive_ratio',
                'negative_ratio', 'total_score', 'avg_num_comments']

YOUTUBE_FEATS = ['log_view_count', 'log_likes', 'log_total_likes', 'trailer_score',
                  'has_yt_comments', 'comment_count', 'yt_avg_compound', 'yt_avg_positive',
                  'yt_avg_negative', 'yt_avg_neutral', 'yt_positive_ratio',
                  'yt_negative_ratio', 'weighted_compound']

REVENUE_MULTIPLIER = 2.5
RATING_THRESHOLD = 7.0


def load_genre_columns(mlb_path="models/genre_mlb.pkl"):
    """Load the MultiLabelBinarizer fitted on the full training dataset,
    so genre dummy columns line up exactly with what the models expect."""
    mlb = joblib.load(mlb_path)
    genre_cols = [f'genre_{g.replace(" ", "_")}' for g in mlb.classes_]
    return mlb, genre_cols


def load_fill_values(path="models/fill_values.json"):
    """Load training-set scalar fill values (e.g. trailer_score median) that
    were saved by export_artifacts_FROM_NOTEBOOK.py. Required because a
    single new movie has no batch to compute its own median from — we must
    reuse the exact value the models were trained against."""
    with open(path) as f:
        return json.load(f)


def get_all_feats(genre_cols):
    """Reconstruct ALL_FEATS in the exact order the models were trained on."""
    return NUMERIC_FEATS + genre_cols + REDDIT_FEATS + YOUTUBE_FEATS


def engineer_row(tmdb_data: dict, reddit_data: dict, yt_data: dict,
                  mlb, genre_cols: list, fill_values: dict) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from raw scraped data, applying the
    exact same transforms as Phase 2 of the notebook (Section 2.3).

    Parameters
    ----------
    tmdb_data : dict with keys matching TMDB scraper output —
        title, year, runtime, vote_count, budget, production_companies,
        cast (comma-joined string), director
    reddit_data : dict — output of compute_sentiment() from the Reddit scraper,
        or {} / all-None if no posts found
    yt_data : dict — merged output of get_movie_sentiment() + compute_sentiment()
        from the YouTube scraper, or {} / all-None if no trailer found
    mlb, genre_cols : from load_genre_columns()
    fill_values : from load_fill_values() — training-set medians for fields
        that can't be computed from a single row (currently: trailer_score)

    Returns
    -------
    pd.DataFrame with exactly one row and columns = NUMERIC_FEATS + genre_cols
    + REDDIT_FEATS + YOUTUBE_FEATS, ready to feed into the trained models.
    """
    row = {}

    # ── Numeric / studio / budget (mirrors notebook Section 2.3) ──────────────
    budget = float(tmdb_data.get('budget', 0) or 0)
    row['budget_known'] = int(budget > 0)
    row['log_budget'] = np.log1p(budget)

    row['Runtime'] = float(tmdb_data.get('runtime', 0) or 0)
    runtime = row['Runtime']
    if runtime <= 90:
        row['runtime_bucket'] = 0
    elif runtime <= 120:
        row['runtime_bucket'] = 1
    elif runtime <= 150:
        row['runtime_bucket'] = 2
    else:
        row['runtime_bucket'] = 3

    row['Vote_Count'] = float(tmdb_data.get('vote_count', 0) or 0)
    row['Year'] = int(tmdb_data.get('year', 0) or 0)

    production_companies = str(tmdb_data.get('production_companies', '') or '')
    row['is_major_studio'] = int(any(s in production_companies for s in MAJOR_STUDIOS))

    cast_str = str(tmdb_data.get('cast', '') or '')
    row['cast_size'] = len([c for c in cast_str.split(',') if c.strip()])

    # ── Genre multi-label encoding (uses the FITTED mlb — never re-fit) ───────
    genre_str = str(tmdb_data.get('genre', '') or '')
    genre_list = [g.strip() for g in genre_str.split(',') if g.strip()]
    genre_vec = mlb.transform([genre_list])[0]   # 1 x n_genres
    for col, val in zip(genre_cols, genre_vec):
        row[col] = int(val)

    # ── Reddit features (notebook Section 2.3 — has_reddit + neutral fill) ───
    post_count = reddit_data.get('post_count') or 0
    row['has_reddit'] = int(post_count > 0)
    row['post_count'] = post_count or 0
    row['avg_compound'] = reddit_data.get('avg_compound') if reddit_data.get('avg_compound') is not None else 0.0
    row['avg_positive'] = reddit_data.get('avg_positive') if reddit_data.get('avg_positive') is not None else 0.0
    row['avg_negative'] = reddit_data.get('avg_negative') if reddit_data.get('avg_negative') is not None else 0.0
    row['avg_neutral'] = reddit_data.get('avg_neutral') if reddit_data.get('avg_neutral') is not None else 0.5
    row['positive_ratio'] = reddit_data.get('positive_ratio') if reddit_data.get('positive_ratio') is not None else 0.0
    row['negative_ratio'] = reddit_data.get('negative_ratio') if reddit_data.get('negative_ratio') is not None else 0.0
    row['total_score'] = reddit_data.get('total_score') or 0
    row['avg_num_comments'] = reddit_data.get('avg_num_comments') or 0.0

    # ── YouTube features (notebook Section 2.3 — log-transform + neutral fill)
    view_count = yt_data.get('video_view_count') or yt_data.get('views') or 0
    likes = yt_data.get('likes') or 0
    total_likes = yt_data.get('total_likes') or 0
    row['log_view_count'] = np.log1p(view_count)
    row['log_likes'] = np.log1p(likes)
    row['log_total_likes'] = np.log1p(total_likes)
    row['trailer_score'] = yt_data.get('trailer_score')
    if row['trailer_score'] is None or row['trailer_score'] == 0:
        row['trailer_score'] = fill_values.get('trailer_score_median', 0.0)

    comment_count = yt_data.get('comment_count') or 0
    row['has_yt_comments'] = int(comment_count > 0)
    row['comment_count'] = comment_count
    row['yt_avg_compound'] = yt_data.get('avg_compound') if yt_data.get('avg_compound') is not None else 0.0
    row['yt_avg_positive'] = yt_data.get('avg_positive') if yt_data.get('avg_positive') is not None else 0.0
    row['yt_avg_negative'] = yt_data.get('avg_negative') if yt_data.get('avg_negative') is not None else 0.0
    row['yt_avg_neutral'] = yt_data.get('avg_neutral') if yt_data.get('avg_neutral') is not None else 0.5
    row['yt_positive_ratio'] = yt_data.get('positive_ratio') if yt_data.get('positive_ratio') is not None else 0.0
    row['yt_negative_ratio'] = yt_data.get('negative_ratio') if yt_data.get('negative_ratio') is not None else 0.0
    row['weighted_compound'] = yt_data.get('weighted_compound') if yt_data.get('weighted_compound') is not None else 0.0

    all_feats = get_all_feats(genre_cols)
    return pd.DataFrame([row], columns=all_feats)


def classify_success(pred_rev, pred_rat, budget):
    """Identical thresholds to the notebook's Two-Stage Success Classification."""
    if budget and budget > 0:
        fin_success = pred_rev >= REVENUE_MULTIPLIER * budget
    else:
        fin_success = pred_rev >= 100_000_000
    crit_success = pred_rat >= RATING_THRESHOLD

    if fin_success and crit_success:
        return 'Blockbuster Hit'
    elif fin_success:
        return 'Commercial Success'
    elif crit_success:
        return 'Critical Darling'
    else:
        return 'Flop'
