# notebooks/

This folder contains the full offline training pipeline for the project — everything that happens **before** the Streamlit app takes over for live inference. It's meant to be run once (or whenever the underlying data changes) to produce the trained model artefacts the app depends on.

## Contents

| File | Description |
|---|---|
| `movie_prediction_phases2_4.ipynb` | Phases 2–4: Feature Engineering → Model Training → Evaluation |

There's no "Phase 1" notebook — Phase 1 is the raw data collection (the TMDB/Reddit/YouTube scrapers), which are separate standalone scripts, not notebook cells. This notebook picks up **after** scraping, starting from the three cleaned CSVs.

## What this notebook does

The notebook is organised into four sections, matching its own markdown headers:

### 1. Setup & Imports
Loads pandas, scikit-learn, matplotlib/seaborn. Notes that `GradientBoostingRegressor` is used in place of `xgboost.XGBRegressor` because `xgboost` wasn't available in the environment this was built in — the swap is a one-line change if you have it installed (see the comment in the first code cell).

### 2. Phase 2 — Feature Engineering
- **2.1** Merges `tmdb_movies_cleaned.csv` + `reddit_sentiment_results.csv` + `youtube_sentiment_results.csv` on `Title`.
- **2.2** Genre multi-label encoding via `MultiLabelBinarizer`, fit on the **full** dataset (not just the revenue subset) so the revenue and rating feature matrices always share identical columns.
- **2.3** Builds numeric/studio/budget features, Reddit features (neutral-filled if missing, with a `has_reddit` flag), and YouTube features (log-transformed views/likes, median-filled `trailer_score`, `has_yt_comments` flag).
- **2.4** Defines `ALL_FEATS` (37 columns total) and splits into `df_rev` (171 movies with known revenue) and `df_rat` (all 197 movies).
- **2.5** Exploratory data analysis — 8-panel chart covering revenue/rating distributions, budget-vs-revenue, trailer views vs revenue, Reddit/YouTube sentiment histograms, top genres, and studio-type revenue comparison. Saved as `eda_overview.png`.

### 3. Phase 3 — Model Training
- **3.1** Builds imputed feature matrices (`SimpleImputer`, median strategy, fit on training data only) and does an 80/20 train/test split (`random_state=42`). Revenue target is log-transformed (`log1p`) before training.
- **3.2** Trains 4 models total: Random Forest + Gradient Boosting, each for Revenue and Rating.

### 4. Phase 4 — Evaluation
- **4.1** MAE / RMSE / R² on the test set for all 4 models (revenue metrics back-transformed to dollar scale).
- **4.2** Actual-vs-predicted scatter plots (`actual_vs_predicted.png`).
- **4.3** Residual plots (`residual_plots.png`).
- **4.4** Feature importance bar charts, colour-coded by feature group (`feature_importance.png`).
- **4.5** Ablation study — compares metadata-only baseline vs. +Reddit vs. +YouTube vs. +both, using a fixed Random Forest so the comparison isolates feature-set effects (`ablation_reddit.png`).
- **4.6** 5-fold cross-validation for both model families and both targets — the more reliable generalisation estimate, since a single 20%-test split on ~170–200 rows is fairly noisy.
- **Two-Stage Success Classification** — applies the revenue-multiplier/rating-threshold rules to the best model's predictions across the full revenue dataset, producing the four category labels and a distribution chart (`success_categories.png`).
- **Save All Outputs** — writes `movie_predictions.csv` and `results_summary.json`.

## Inputs required

Place these three CSVs in the same directory as the notebook (or edit the `pd.read_csv(...)` calls in the "Load Data" cell to point elsewhere):

- `tmdb_movies_cleaned.csv`
- `reddit_sentiment_results.csv`
- `youtube_sentiment_results.csv`

These are **not included in this repo** — see the root `README.md` / `movie_app/data/README.md` for what each one contains. If you don't have them, you'll need to run the TMDB/Reddit/YouTube scrapers first (not part of this notebook).

## Outputs produced

Running the notebook end-to-end writes the following into the same directory:

**Charts**
- `eda_overview.png`
- `actual_vs_predicted.png`
- `residual_plots.png`
- `feature_importance.png`
- `ablation_reddit.png`
- `success_categories.png`

**Data**
- `movie_predictions.csv` — every revenue-dataset movie with its predicted revenue, predicted rating, and success label (this becomes the Streamlit app's initial cache — copy it into `movie_app/data/`)
- `results_summary.json` — all evaluation metrics, ablation results, and success-label distribution in one JSON file (powers the Ablation Study tab on the app's Analysis Overview page — copy it into `movie_app/data/`)

**What this notebook does NOT export by itself**

The notebook as written prints and evaluates models in-memory but doesn't include a `joblib.dump(...)` export cell for the deployed app's artefacts. Before the Streamlit app can run predictions, you need to add an export step (referred to elsewhere in the docs as `export_artifacts_FROM_NOTEBOOK.py`, either as a standalone script or an extra cell at the end of this notebook) that saves:

```python
import joblib, json

joblib.dump(best_rev, 'best_rev.pkl')
joblib.dump(best_rat, 'best_rat.pkl')
joblib.dump(mlb, 'genre_mlb.pkl')
joblib.dump(imp_rev, 'imputer_rev.pkl')
joblib.dump(imp_rat, 'imputer_rat.pkl')

with open('fill_values.json', 'w') as f:
    json.dump({'trailer_score_median': df['trailer_score'].median()}, f)

with open('confidence.json', 'w') as f:
    json.dump({
        'revenue_rmse': res['rf_rev']['RMSE'] if res['rf_rev']['R2'] >= res['xgb_rev']['R2'] else res['xgb_rev']['RMSE'],
        'rating_rmse':  res['rf_rat']['RMSE'] if res['rf_rat']['R2'] >= res['xgb_rat']['R2'] else res['xgb_rat']['RMSE'],
    }, f)

with open('all_feats_reference.json', 'w') as f:
    json.dump(ALL_FEATS, f)
```

Then copy all of the resulting files into `movie_app/models/`.

## Running it

```bash
pip install -r ../requirements.txt
jupyter notebook movie_prediction_phases2_4.ipynb
```

Run all cells top to bottom (`Kernel → Restart & Run All`). Total runtime is a few minutes on a typical laptop — the slowest steps are the 5-fold CV cell and the ablation study, since both retrain models internally.

## Notes on reproducibility

- Every model and split uses `random_state=42`, so re-running the notebook on the same input CSVs will reproduce identical results.
- If you regenerate the underlying CSVs (e.g. by re-scraping), `MultiLabelBinarizer` and `SimpleImputer` will pick up whatever new genres/values appear in that data. If the deployed app is using an older set of trained artefacts, make sure to re-export and replace **all** artefact files together — mixing an old `genre_mlb.pkl` with a newly retrained model will misalign feature columns and produce silently wrong predictions.
