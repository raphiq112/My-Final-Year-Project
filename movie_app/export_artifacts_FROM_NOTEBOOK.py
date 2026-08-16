"""
Export Trained Artifacts for Deployment
=========================================
Run this ONCE inside your notebook environment (after Phase 3/4 cells have
executed, so rf_rev/xgb_rev/rf_rat/xgb_rat, imp_rev/imp_rat, mlb, df_rev/df_rat
all exist in memory) to save everything the Streamlit app needs for
inference-only prediction.

Add this as a new cell at the END of movie_prediction_phases2_4.ipynb,
then copy the resulting models/ folder into your Streamlit project.
"""

import joblib
import json
import numpy as np

# ── 1. Trained models ──────────────────────────────────────────────────────────
# Per your memory notes: best_rev / best_rat already picks whichever of
# rf_*/xgb_* scored higher R² on the test set — saving those directly.
joblib.dump(best_rev, 'models/best_rev.pkl')
joblib.dump(best_rat, 'models/best_rat.pkl')

# Also save both individually in case you want to compare/ensemble later
joblib.dump(rf_rev,  'models/rf_rev.pkl')
joblib.dump(xgb_rev, 'models/xgb_rev.pkl')
joblib.dump(rf_rat,  'models/rf_rat.pkl')
joblib.dump(xgb_rat, 'models/xgb_rat.pkl')

# ── 2. Fitted preprocessing — CRITICAL for correct single-row inference ──────
# The SimpleImputer median is fitted on the full training set. Re-fitting it
# on a single new movie would have nothing to compute a median FROM, so the
# exact fitted imputer must be reused at inference time.
joblib.dump(imp_rev, 'models/imputer_rev.pkl')
joblib.dump(imp_rat, 'models/imputer_rat.pkl')

# The genre MultiLabelBinarizer must also be reused (never re-fit), so that
# a new movie's genre columns line up with the columns the model was trained on.
joblib.dump(mlb, 'models/genre_mlb.pkl')

# ── 3. Saved scalar fill values (training-set medians) ────────────────────────
# trailer_score is median-filled at the BATCH level in the notebook
# (df['trailer_score'].fillna(df['trailer_score'].median())). For single-row
# inference we need that same median saved as a constant.
fill_values = {
    'trailer_score_median': float(df['trailer_score'].median()),
}
with open('models/fill_values.json', 'w') as f:
    json.dump(fill_values, f, indent=2)

# ── 3b. Test-set RMSE for the confidence-range card ───────────────────────────
# The Prediction Result page shows "predicted revenue ± range" — that range
# should be the actual test-set RMSE (in dollar scale), not a guessed
# percentage. res['rf_rev']/res['xgb_rev'] already hold this from Phase 4.
best_rev_key = 'rf_rev' if res['rf_rev']['R2'] >= res['xgb_rev']['R2'] else 'xgb_rev'
best_rat_key = 'rf_rat' if res['rf_rat']['R2'] >= res['xgb_rat']['R2'] else 'xgb_rat'
confidence = {
    'revenue_rmse': float(res[best_rev_key]['RMSE']),
    'rating_rmse': float(res[best_rat_key]['RMSE']),
}
with open('models/confidence.json', 'w') as f:
    json.dump(confidence, f, indent=2)

# ── 4. Feature order sanity check ──────────────────────────────────────────────
# Confirms ALL_FEATS order so utils/feature_engineering.py can be checked against it.
with open('models/all_feats_reference.json', 'w') as f:
    json.dump(list(ALL_FEATS), f, indent=2)

print("Saved to models/:")
print("  best_rev.pkl, best_rat.pkl, rf_rev.pkl, xgb_rev.pkl, rf_rat.pkl, xgb_rat.pkl")
print("  imputer_rev.pkl, imputer_rat.pkl, genre_mlb.pkl")
print("  fill_values.json, all_feats_reference.json, confidence.json")
print()
print(f"trailer_score training median: {fill_values['trailer_score_median']:.4f}")
print(f"Revenue RMSE (for confidence range): ${confidence['revenue_rmse']:,.0f}")
print(f"Rating RMSE (for confidence range): {confidence['rating_rmse']:.4f}")
print(f"Total features: {len(ALL_FEATS)}")
