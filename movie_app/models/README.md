# models/

Trained model artefacts go here, exported from the notebook's final cell:

- `best_rev.pkl`
- `best_rat.pkl`
- `genre_mlb.pkl`
- `imputer_rev.pkl`
- `imputer_rat.pkl`
- `fill_values.json`
- `confidence.json`
- `all_feats_reference.json`

The app will raise a clear `FileNotFoundError` on startup if any required file is missing (see `utils/model_loader.py`).
