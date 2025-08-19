# fastapi/deps/model.py
from functools import lru_cache
from pathlib import Path
import json
import joblib

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "heart_rf_final.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
DEFAULT_THRESHOLD = 0.5

@lru_cache
def get_threshold() -> float:
    try:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            return float(json.load(f).get("threshold", DEFAULT_THRESHOLD))
    except FileNotFoundError:
        return DEFAULT_THRESHOLD

@lru_cache
def get_pipeline():
    model = joblib.load(MODEL_PATH)
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            return model
    except Exception:
        pass
    preproc = joblib.load(PREP_PATH)
    from sklearn.pipeline import Pipeline
    return Pipeline(steps=[("prep", preproc), ("clf", model)])
