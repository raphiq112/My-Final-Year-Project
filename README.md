# 🎬 Pre-Release Movie Revenue and Rating Prediction

A machine learning system that predicts a movie's **box office revenue (USD)** and **audience rating (1–10)** *before it is released*, using only pre-release signals: TMDB production metadata, Reddit pre-release sentiment, and YouTube trailer engagement/sentiment. A rule-based two-stage classifier then translates the two continuous predictions into one of four outcome categories — **Blockbuster Hit**, **Commercial Success**, **Critical Darling**, or **Flop**.

Final Year Project — Bachelor of Computer Science (Hons.), College of Computing, Informatics and Mathematics, Universiti Teknologi MARA.
Supervised by Mohammad Bakri bin Che Haron.

> 📄 Full thesis writeup: [`docs/Final_Year_Project_RafiqHakeemiRoslan.pdf`](docs/Final_Year_Project_RafiqHakeemiRoslan.pdf)

---

## Table of Contents

- [Overview](#overview)
- [Why pre-release only?](#why-pre-release-only)
- [System Architecture](#system-architecture)
- [Data Sources](#data-sources)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Two-Stage Success Classification](#two-stage-success-classification)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Notebook (Training Pipeline)](#running-the-notebook-training-pipeline)
- [Running the Streamlit App (Inference)](#running-the-streamlit-app-inference)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Most movie-industry investment decisions are still made on intuition rather than evidence, and most existing success-prediction models rely on **post-release** signals (e.g. opening-weekend gross), which makes them useless for early production planning. This project builds a system that predicts success using **only information available before release**, so it can support decisions during production and marketing planning rather than after the fact.

A dataset of **197 movies (2020–2025)** was collected and merged from three sources, engineered into **37 features**, and used to train Random Forest and Gradient Boosting regressors for both revenue and rating. The full pipeline — from raw scraping to a final prediction — is reproduced in an interactive **Streamlit web app** that performs live inference on new/unreleased movies.

## Why pre-release only?

Unreleased movies face a **cold-start problem**: no user ratings, no review history, no collaborative-filtering signal exists yet. This project deliberately restricts itself to information that exists *before* a movie hits theatres:

- Production metadata (budget, genre, cast, director, runtime, studio) — from TMDB
- Community anticipation — from Reddit posts in the 90 days before release
- Trailer reception — from YouTube view/like counts and pre-release comment sentiment

## System Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   TMDB API   │   │ Arctic Shift │   │  YouTube     │
│  (metadata)  │   │ (Reddit data)│   │  Data API v3 │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                           ▼
              ┌─────────────────────────┐
              │   Merge by movie title   │
              │   + Feature Engineering  │
              │   (37 engineered feats)  │
              └────────────┬─────────────┘
                            ▼
          ┌─────────────────────────────────┐
          │  Random Forest / Gradient        │
          │  Boosting  (Revenue · Rating)    │
          └────────────────┬─────────────────┘
                            ▼
          ┌─────────────────────────────────┐
          │  Two-Stage Rule-Based Classifier  │
          │  → Blockbuster / Commercial /     │
          │    Critical Darling / Flop        │
          └────────────────┬─────────────────┘
                            ▼
              ┌─────────────────────────┐
              │   Streamlit Web App      │
              │  (cache-first inference) │
              └─────────────────────────┘
```

The Streamlit app checks a local CSV cache before scraping anything — if a movie has already been predicted, the cached row loads instantly instead of re-hitting three external APIs.

## Data Sources

| Source | Role | Auth Required | Notes |
|---|---|---|---|
| [TMDB API](https://www.themoviedb.org/documentation/api) | Movie metadata (budget, genre, cast, director, runtime, production companies, vote count) | Yes (free) | Secondary/reference source |
| [Arctic Shift API](https://arctic-shift.photon-reddit.com/) | Reddit r/movies posts, 90-day pre-release window | No | Used because official Reddit API access was rejected for this project — see [Known Limitations](#known-limitations) |
| [YouTube Data API v3](https://developers.google.com/youtube/v3) | Official trailer identification (custom scoring heuristic) + up to 100 pre-release comments | Yes (free tier) | Comment sentiment scored with VADER |

All three sources are merged by movie title into a single dataframe of 197 movies released 2020–2025 (after cleaning an initial pull of 1,000 TMDB movies down by release-window and vote-count filters).

## Feature Engineering

37 total features (`ALL_FEATS`), split into four groups:

| Group | Count | Examples |
|---|---|---|
| Numeric / Studio | 8 | `log_budget`, `budget_known`, `Runtime`, `runtime_bucket`, `Vote_Count`, `is_major_studio`, `cast_size`, `Year` |
| Genre (multi-label) | 19 | `genre_Action`, `genre_Science_Fiction`, ... |
| Reddit sentiment | 10 | `has_reddit`, `post_count`, `avg_compound`, `positive_ratio`, `total_score`, ... |
| YouTube trailer | 13 | `log_view_count`, `log_likes`, `trailer_score`, `has_yt_comments`, `yt_avg_compound`, ... |

Key decisions:
- Genre `MultiLabelBinarizer` is fit on the **full** dataset so the revenue and rating subsets always share identical columns.
- `budget`, `view_count`, and `likes` are log-transformed (`log1p`) to reduce heavy right-skew.
- Missing Reddit/YouTube data is **not** dropped — it's neutral-filled with a `has_reddit`/`has_yt_comments` flag so the model can distinguish "no social presence" from "negative sentiment."
- `trailer_score` is filled with the **training-set median**, not zero — a zero would imply the worst possible trailer match rather than a genuine absence of data.
- `like_ratio` was **excluded** — YouTube removed public dislike counts, so this feature was a constant `1.0` across all 197 rows and carried no signal.

## Models

Two model families, trained separately for each target (4 models total):

- **Random Forest** (`RandomForestRegressor`)
- **Gradient Boosting** (`GradientBoostingRegressor` — used as a drop-in stand-in for XGBoost since `xgboost` wasn't available in the training environment; the code path is a one-line swap back to `xgb.XGBRegressor` if needed)

Revenue is trained in **log space** (`log1p`) and back-transformed (`expm1`) for reporting and inference. Rating is trained directly on the 1–10 scale. Evaluation uses MAE, RMSE, and R² on a held-out 20% test split, plus 5-fold cross-validation for a more stable generalisation estimate.

## Two-Stage Success Classification

Raw regression outputs are converted into one of four labels using fixed, transparent thresholds — not a separately trained classifier:

| Label | Revenue condition | Rating condition |
|---|---|---|
| 🏆 Blockbuster Hit | Revenue ≥ 2.5× budget | Rating ≥ 7.0 |
| 💰 Commercial Success | Revenue ≥ 2.5× budget | Rating < 7.0 |
| 🎭 Critical Darling | Revenue < 2.5× budget | Rating ≥ 7.0 |
| 📉 Flop | Revenue < 2.5× budget | Rating < 7.0 |

When budget is unknown, a **$100M** revenue proxy threshold is used instead of the 2.5× rule.

## Results

5-fold cross-validation (more honest generalisation estimate than the single train/test split):

**Revenue model** (log space)

| Model | R² (CV) | MAE (log) | RMSE (log) |
|---|---|---|---|
| Random Forest | 0.4455 | 1.1343 | 1.5833 |
| Gradient Boosting | 0.4001 | 1.1879 | 1.6405 |

**Rating model** (1–10 scale)

| Model | R² (CV) | MAE | RMSE |
|---|---|---|---|
| Random Forest | 0.1028 | 0.5096 | 0.6804 |
| Gradient Boosting | 0.1463 | 0.4793 | 0.6494 |

**Ablation study** (Random Forest, fixed config, isolating feature-source effects): adding Reddit and/or YouTube sentiment features did **not** consistently outperform the metadata-only baseline at this dataset size (197 movies). The gains seen from the full boosted model appear driven more by model tuning than by the social features themselves. This is discussed candidly in Chapter 5 of the thesis rather than glossed over.

## Repository Structure

```
.
├── notebooks/
│   └── movie_prediction_phases2_4.ipynb   # Phases 2–4: feature engineering → training → evaluation
├── movie_app/
│   ├── app.py                             # Landing page entry point
│   ├── pages/
│   │   ├── 1_homepage.py
│   │   ├── 2_movie_details.py
│   │   ├── 2a_popular_movies.py
│   │   ├── 2b_upcoming_movies.py
│   │   ├── 3_analysis_overview.py
│   │   ├── 4_prediction_history.py
│   │   └── 5_custom_movie.py
│   ├── utils/
│   │   ├── cache_store.py                 # CSV cache read/write (cache-first inference)
│   │   ├── feature_engineering.py         # Single-row inference-time feature pipeline
│   │   ├── model_loader.py                # Loads trained artefacts, runs predict_one()
│   │   ├── movie_grid.py                  # Shared poster grid for Popular/Upcoming pages
│   │   ├── navbar.py                      # Shared nav bar component
│   │   ├── reddit_client.py               # Arctic Shift single-movie fetcher
│   │   ├── styling.py                     # Shared dark theme CSS
│   │   ├── tmdb_client.py                 # TMDB single-movie fetcher
│   │   └── youtube_client.py              # YouTube single-movie trailer/comment fetcher
│   ├── models/                            # Trained artefacts (see below) — not committed, see .gitignore
│   ├── data/                              # CSV cache + training CSVs — not committed, see .gitignore
│   └── .streamlit/
│       └── secrets.toml                   # API keys — NEVER commit this file
├── docs/
│   └── Final_Year_Project_RafiqHakeemiRoslan.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** adjust the `notebooks/` and `movie_app/` paths above to match how you actually organise the repo — the important thing on upload is keeping the notebook, the Streamlit `pages/`+`utils/` split, and the exported model artefacts each in their own clearly named folder.

### Trained artefacts (exported by the notebook, loaded by the app)

| File | Purpose |
|---|---|
| `best_rev.pkl` | Best-performing revenue model (by test R²) |
| `best_rat.pkl` | Best-performing rating model (by test R²) |
| `genre_mlb.pkl` | Fitted `MultiLabelBinarizer` — never re-fit at inference |
| `imputer_rev.pkl` / `imputer_rat.pkl` | Fitted `SimpleImputer` per target — never re-fit at inference |
| `fill_values.json` | Training-set median of `trailer_score`, used to fill missing values at inference |
| `confidence.json` | Test-set RMSE per target, shown in the UI as the model's confidence range |
| `all_feats_reference.json` | Canonical feature order the models expect |

These are generated by the last cell of the notebook and must be copied into `movie_app/models/` before the app will run predictions.

## Getting Started

### Prerequisites

- Python 3.10+
- API keys: [TMDB](https://www.themoviedb.org/settings/api) and [YouTube Data API v3](https://console.cloud.google.com/apis/library/youtube.googleapis.com) (Arctic Shift needs no key)

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Notebook (Training Pipeline)

1. Place `tmdb_movies_cleaned.csv`, `reddit_sentiment_results.csv`, and `youtube_sentiment_results.csv` in the same directory as the notebook (or update the `pd.read_csv` paths).
2. Open `notebooks/movie_prediction_phases2_4.ipynb` in Jupyter.
3. Run all cells top to bottom. This regenerates all EDA/evaluation charts, trains all 4 models, runs the ablation study and cross-validation, and saves `movie_predictions.csv` + `results_summary.json`.
4. Export trained artefacts (`best_rev.pkl`, `best_rat.pkl`, `genre_mlb.pkl`, imputers, `fill_values.json`, `confidence.json`) and copy them into `movie_app/models/`.

## Running the Streamlit App (Inference)

1. Create `movie_app/.streamlit/secrets.toml`:

   ```toml
   TMDB_API_KEY = "your_tmdb_key_here"
   YOUTUBE_API_KEY = "your_youtube_key_here"
   ```

2. Make sure `movie_app/models/` contains the exported artefacts above.
3. From the `movie_app/` directory:

   ```bash
   streamlit run app.py
   ```

4. The app checks `data/movie_predictions.csv` for a cache hit before scraping anything, so previously-searched movies load instantly.

## Known Limitations

Documented candidly here (and in Chapter 5/6 of the thesis) rather than hidden, since they matter for interpreting results honestly:

- **"XGBoost" in docs, `GradientBoostingRegressor` in code** — `xgboost` wasn't available in the training environment, so `GradientBoostingRegressor` was substituted as a near-equivalent gradient-boosted tree model. The thesis documents this explicitly.
- **Official Reddit API access was rejected** — Reddit data is collected via Arctic Shift, a community-maintained archival mirror, instead of the official API.
- **Small dataset (197 movies)** — limits how confidently the ablation study can isolate the marginal value of social sentiment features; results should be read as suggestive, not definitive.
- **Rating R² is low (0.10–0.15, cross-validated)** and roughly flat across all feature configurations — audience rating is genuinely hard to predict from pre-release signals alone.
- **`is_major_studio` is a manually curated 12-studio allowlist** with a US/Hollywood bias; it won't generalise well to independent or non-US productions.
- **`like_ratio` excluded** — constant at 1.0 across the dataset because YouTube removed public dislike counts.
- **High proportion of "Blockbuster Hit" classifications (~48%)** — a byproduct of the 2.5× budget threshold combined with dataset composition.
- **Single train/test split metrics (used on the poster) vs. cross-validated metrics (used in the thesis) differ** — the CV figures are the more honest generalisation estimate and are what's reported above.

## Future Work

- Compare Arctic Shift coverage against official Reddit API access, if granted.
- Replace/augment VADER with transformer-based sentiment (BERT/RoBERTa) for better handling of sarcasm and context.
- Expand the training dataset for a larger, more internationally diverse sample.
- Incorporate external economic indicators (seasonal box office trends, competing releases in the same window).
- Extend the success classifier to an ROI-based measure rather than a fixed revenue multiplier.
- Scheduled re-scraping of Reddit/YouTube data as a movie's release date approaches, to track sentiment drift over time rather than a single snapshot.

## Tech Stack

- **Language:** Python
- **ML:** scikit-learn (Random Forest, Gradient Boosting, `MultiLabelBinarizer`, `SimpleImputer`), VADER (`vaderSentiment`)
- **App:** Streamlit
- **Data:** pandas, NumPy
- **Viz:** matplotlib, seaborn
- **APIs:** TMDB API, YouTube Data API v3, Arctic Shift API

## License

This project is submitted as an academic Final Year Project. If you'd like to reuse the code, consider adding an [MIT License](https://choosealicense.com/licenses/mit/) — check with your institution's academic integrity policy before doing so, since some universities restrict public redistribution of thesis-linked code until after grading/exhibition.

## Acknowledgements

- Supervisor: **Mohammad Bakri bin Che Haron**
- Dr. Raseeda Binti Hamzah
- College of Computing, Informatics and Mathematics, Universiti Teknologi MARA
