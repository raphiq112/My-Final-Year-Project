"""
Analysis Overview — Live Phase 2-4 Charts
==============================================
Regenerates the same EDA, feature importance, ablation study, and
actual-vs-predicted charts as movie_prediction_phases2_4.ipynb, computed
live from the CSVs in /data each time this page loads — so it always
reflects whatever's currently in your dataset, matching Chapter 4/5.

Requires: data/tmdb_movies_cleaned.csv, data/reddit_sentiment_results.csv,
data/youtube_sentiment_results.csv, plus the trained models in /models
(from export_artifacts_FROM_NOTEBOOK.py) for the evaluation sections.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from utils.styling import inject_theme
from utils.navbar import render_navbar
from utils.model_loader import load_all_artifacts
from utils import feature_engineering as fe

matplotlib.use("Agg")

st.set_page_config(page_title="Analysis Overview", page_icon="📊", layout="wide")
inject_theme()
render_navbar(active="Analysis")

DATA_DIR = Path(__file__).parent.parent / "data"

# Dark-theme matplotlib styling so charts match the app's palette rather than
# rendering with a jarring white background.
plt.rcParams.update({
    "figure.facecolor": "#0d0e1a", "axes.facecolor": "#16172a",
    "axes.edgecolor": "#3a3a55", "axes.labelcolor": "#d0d0dc",
    "xtick.color": "#9b9bb0", "ytick.color": "#9b9bb0",
    "text.color": "#f4f4fa", "grid.color": "#26273d",
    "axes.titlecolor": "#f4f4fa",
})

st.markdown("### Analysis Overview")
st.caption("Live-generated from the data and trained models in /data and /models. "
           "Mirrors Phases 2–4 of the training notebook.")

# ── Load CSVs ──────────────────────────────────────────────────────────────────
required_files = ["tmdb_movies_cleaned.csv", "reddit_sentiment_results.csv",
                   "youtube_sentiment_results.csv"]
missing = [f for f in required_files if not (DATA_DIR / f).exists()]
if missing:
    st.warning(
        f"Missing data files in /data: {', '.join(missing)}. "
        f"Copy them from your notebook environment to enable this page."
    )
    st.stop()

tmdb = pd.read_csv(DATA_DIR / "tmdb_movies_cleaned.csv")
reddit = pd.read_csv(DATA_DIR / "reddit_sentiment_results.csv")
youtube = pd.read_csv(DATA_DIR / "youtube_sentiment_results.csv")

tab_eda, tab_eval, tab_ablation = st.tabs(["Exploratory Data Analysis", "Model Evaluation", "Ablation Study"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA (mirrors notebook Section 2.5)
# ════════════════════════════════════════════════════════════════════════════
with tab_eda:
    REDDIT_COLS = ['Title', 'post_count', 'avg_compound', 'avg_positive', 'avg_negative',
                   'avg_neutral', 'positive_ratio', 'negative_ratio', 'total_score', 'avg_num_comments']
    YOUTUBE_RENAME = {
        'avg_compound': 'yt_avg_compound', 'avg_positive': 'yt_avg_positive',
        'avg_negative': 'yt_avg_negative', 'avg_neutral': 'yt_avg_neutral',
        'positive_ratio': 'yt_positive_ratio', 'negative_ratio': 'yt_negative_ratio',
    }
    yt = youtube.rename(columns=YOUTUBE_RENAME)
    YOUTUBE_COLS = ['Title', 'video_view_count', 'likes', 'trailer_score', 'comment_count',
                     'yt_avg_compound', 'yt_avg_positive', 'yt_avg_negative', 'yt_avg_neutral',
                     'yt_positive_ratio', 'yt_negative_ratio', 'total_likes', 'weighted_compound']
    yt_cols_present = [c for c in YOUTUBE_COLS if c in yt.columns]

    df = tmdb.merge(reddit[[c for c in REDDIT_COLS if c in reddit.columns]], on='Title', how='left')
    df = df.merge(yt[yt_cols_present], on='Title', how='left')

    for col in ['Budget', 'Revenue', 'Rating', 'Vote_Count', 'Runtime']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['log_view_count'] = np.log1p(df.get('video_view_count', 0).fillna(0))
    df['has_reddit'] = (df.get('post_count', 0).fillna(0) > 0).astype(int)
    df['is_major_studio'] = df.get('Production_Companies', '').fillna('').apply(
        lambda x: int(any(s in str(x) for s in fe.MAJOR_STUDIOS)))

    df_rev = df[df['Revenue'] > 0].copy() if 'Revenue' in df.columns else pd.DataFrame()

    st.markdown(f"**Dataset size:** {len(df)} movies · "
                f"**Revenue subset:** {len(df_rev)} · "
                f"**Reddit coverage:** {(df['has_reddit']==1).sum()}/{len(df)}")

    col1, col2 = st.columns(2)

    with col1:
        if not df_rev.empty:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.hist(np.log1p(df_rev['Revenue']), bins=24, color='#7f77dd', edgecolor='#0d0e1a', alpha=0.9)
            ax.set_title('Revenue Distribution (log scale)')
            ax.set_xlabel('log(Revenue + 1)')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if 'Rating' in df.columns:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.hist(df['Rating'].dropna(), bins=20, color='#5dcaa5', edgecolor='#0d0e1a', alpha=0.9)
            ax.axvline(7.0, color='#facb7d', linestyle='--', linewidth=1.5, label='Critical threshold (7.0)')
            ax.set_title('Rating Distribution')
            ax.legend(fontsize=8)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col2:
        if not df_rev.empty and 'Budget' in df_rev.columns:
            mask = df_rev['Budget'] > 0
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.scatter(np.log1p(df_rev.loc[mask, 'Budget']), np.log1p(df_rev.loc[mask, 'Revenue']),
                       alpha=0.5, color='#d4537e', s=30, edgecolors='none')
            ax.set_title('Budget vs Revenue (log scale)')
            ax.set_xlabel('log(Budget + 1)')
            ax.set_ylabel('log(Revenue + 1)')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if not df_rev.empty:
            sent = df_rev.loc[df_rev['has_reddit'] == 1, 'avg_compound'] if 'avg_compound' in df_rev.columns else pd.Series([])
            if len(sent) > 0:
                fig, ax = plt.subplots(figsize=(5.5, 3.8))
                ax.hist(sent, bins=18, color='#f0997b', edgecolor='#0d0e1a', alpha=0.9)
                ax.axvline(0, color='#d0d0dc', linestyle='--', linewidth=1)
                ax.set_title(f'Reddit Sentiment (n={len(sent)} movies)')
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Evaluation (mirrors notebook Section 4.1, 4.4)
# ════════════════════════════════════════════════════════════════════════════
with tab_eval:
    try:
        artifacts = load_all_artifacts()
    except FileNotFoundError as e:
        st.warning(str(e))
        st.stop()

    confidence = artifacts.get("confidence", {})
    rev_rmse = confidence.get("revenue_rmse")
    rat_rmse = confidence.get("rating_rmse")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Revenue Model**")
        if rev_rmse:
            st.metric("Test RMSE", f"${rev_rmse:,.0f}")
        else:
            st.caption("RMSE not available — re-run export_artifacts_FROM_NOTEBOOK.py")
    with col2:
        st.markdown("**Rating Model**")
        if rat_rmse:
            st.metric("Test RMSE", f"{rat_rmse:.4f}")
        else:
            st.caption("RMSE not available — re-run export_artifacts_FROM_NOTEBOOK.py")

    st.markdown("**Feature Importance**")
    feat_tab1, feat_tab2 = st.columns(2)

    def _feat_imp_chart(model, title, ax):
        imp = pd.Series(model.feature_importances_, index=artifacts["all_feats"])
        top = imp.sort_values(ascending=False).head(12)
        colors = ['#d4537e' if f in fe.REDDIT_FEATS else
                  '#5dcaa5' if f in fe.YOUTUBE_FEATS else
                  '#7f77dd' if f.startswith('genre_') else
                  '#facb7d' for f in top.index]
        top[::-1].plot(kind='barh', ax=ax, color=colors[::-1])
        ax.set_title(title, fontsize=10)

    with feat_tab1:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        _feat_imp_chart(artifacts["best_rev"], "Revenue Model — Top Features", ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with feat_tab2:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        _feat_imp_chart(artifacts["best_rat"], "Rating Model — Top Features", ax)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Ablation Study (mirrors notebook Section 4.5)
# ════════════════════════════════════════════════════════════════════════════
with tab_ablation:
    ablation_path = DATA_DIR / "results_summary.json"
    if not ablation_path.exists():
        st.caption(
            "results_summary.json not found in /data — this is saved by the "
            "notebook's Section 4.6 ('Save All Outputs') cell. Copy it over "
            "to show the ablation chart here."
        )
    else:
        import json
        with open(ablation_path) as f:
            summary = json.load(f)

        ab_rev = summary.get("ablation_revenue", {})
        ab_rat = summary.get("ablation_rating", {})

        col1, col2 = st.columns(2)
        with col1:
            if ab_rev:
                names = list(ab_rev.keys())
                vals = [ab_rev[n]["R2"] for n in names]
                fig, ax = plt.subplots(figsize=(5.5, 4))
                ax.bar(names, vals, color=['#9b9bb0', '#d4537e', '#5dcaa5', '#7f77dd'][:len(names)])
                ax.set_title("Revenue Model — R² by Feature Set")
                ax.tick_params(axis='x', rotation=20)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        with col2:
            if ab_rat:
                names = list(ab_rat.keys())
                vals = [ab_rat[n]["R2"] for n in names]
                fig, ax = plt.subplots(figsize=(5.5, 4))
                ax.bar(names, vals, color=['#9b9bb0', '#d4537e', '#5dcaa5', '#7f77dd'][:len(names)])
                ax.set_title("Rating Model — R² by Feature Set")
                ax.tick_params(axis='x', rotation=20)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        st.caption(
            "Note: social sentiment features did not consistently outperform "
            "the metadata-only baseline at this sample size — see Chapter 5 discussion."
        )
