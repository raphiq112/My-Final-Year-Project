"""
Model Loader
=============
Loads trained models + fitted preprocessing artifacts once and caches them
across Streamlit reruns (via st.cache_resource at the call site in app.py).
"""

import joblib
from pathlib import Path
from utils import feature_engineering as fe

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_all_artifacts():
    """Load everything needed for inference. Raises a clear error if the
    notebook export step (export_artifacts_FROM_NOTEBOOK.py) hasn't been run."""
    required = ['best_rev.pkl', 'best_rat.pkl', 'genre_mlb.pkl', 'fill_values.json']
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {missing}. "
            f"Run export_artifacts_FROM_NOTEBOOK.py inside your notebook "
            f"environment first, then copy the models/ folder here."
        )

    best_rev = joblib.load(MODELS_DIR / 'best_rev.pkl')
    best_rat = joblib.load(MODELS_DIR / 'best_rat.pkl')
    mlb, genre_cols = fe.load_genre_columns(MODELS_DIR / 'genre_mlb.pkl')
    fill_values = fe.load_fill_values(MODELS_DIR / 'fill_values.json')

    confidence_path = MODELS_DIR / 'confidence.json'
    if confidence_path.exists():
        import json
        with open(confidence_path) as f:
            confidence = json.load(f)
    else:
        confidence = {'revenue_rmse': None, 'rating_rmse': None}

    return {
        'best_rev': best_rev,
        'best_rat': best_rat,
        'mlb': mlb,
        'genre_cols': genre_cols,
        'fill_values': fill_values,
        'all_feats': fe.get_all_feats(genre_cols),
        'confidence': confidence,
    }


def predict_one(row_df, artifacts, budget):
    """
    Run a single engineered row through both models and classify success.
    `row_df` must have columns == artifacts['all_feats'] in that exact order
    (this is guaranteed if it came from feature_engineering.engineer_row()).
    """
    import numpy as np

    row_df = row_df[artifacts['all_feats']]  # enforce column order defensively

    pred_rev = float(np.expm1(artifacts['best_rev'].predict(row_df)[0]))
    pred_rat = float(artifacts['best_rat'].predict(row_df)[0])
    label = fe.classify_success(pred_rev, pred_rat, budget)

    return {
        'pred_revenue': pred_rev,
        'pred_rating': pred_rat,
        'success_label': label,
    }
