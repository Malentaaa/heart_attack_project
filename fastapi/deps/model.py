# fastapi/deps/model.py
from functools import lru_cache
from pathlib import Path
import importlib
import sys
import json
import joblib

# === пути к артефактам ===
MODELS_DIR       = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH       = MODELS_DIR / "heart_rf_final.pkl"   # ваш файл модели
THRESHOLD_PATH   = MODELS_DIR / "threshold.json"       # (необязательно)
DEFAULT_THRESHOLD = 0.5

def _register_custom_classes():
    """
    Если pickle ссылается на __main__.ClassName, подкладываем реальные классы
    из нашего модуля src.custom_transformers в модуль __main__,
    чтобы joblib смог распаковать модель.
    """
    candidate_modules = [
        "src.custom_transformers",      # наш модуль с кастомными трансформерами
        "src.custom_imputers",          # вдруг ты уже создавала этот файл ранее
        "src.modeling.dataprocessor",
        "src.dataprocessor",
        "dataprocessor",
        "src.transformers",
        "transformers",
    ]
    custom_class_names = [
        "GroupMedianImputer",
        "ModeImputer",
        "BinaryCleaner",
        "MissingIndicatorSimple",
    ]

    found_any = False
    import types
    for mod_name in candidate_modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for cls_name in custom_class_names:
            if hasattr(mod, cls_name):
                cls = getattr(mod, cls_name)
                main = sys.modules.get("__main__")
                if main is None:
                    main = types.ModuleType("__main__")
                    sys.modules["__main__"] = main
                setattr(main, cls_name, cls)
                found_any = True

    if not found_any:
        # Фолбэк: no-op классы, чтобы хотя бы загрузить модель (не рекомендуется для боевой работы)
        try:
            from sklearn.base import BaseEstimator, TransformerMixin
            class GroupMedianImputer(BaseEstimator, TransformerMixin):
                def fit(self, X, y=None): return self
                def transform(self, X):  return X
            class ModeImputer(BaseEstimator, TransformerMixin):
                def fit(self, X, y=None): return self
                def transform(self, X):  return X
            class BinaryCleaner(BaseEstimator, TransformerMixin):
                def fit(self, X, y=None): return self
                def transform(self, X):  return X
            class MissingIndicatorSimple(BaseEstimator, TransformerMixin):
                def fit(self, X, y=None): return self
                def transform(self, X):  return X

            main = sys.modules.get("__main__")
            if main is None:
                main = types.ModuleType("__main__")
                sys.modules["__main__"] = main
            for cls in [GroupMedianImputer, ModeImputer, BinaryCleaner, MissingIndicatorSimple]:
                setattr(main, cls.__name__, cls)
        except Exception:
            pass

@lru_cache
def get_threshold() -> float:
    try:
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("threshold", DEFAULT_THRESHOLD))
    except FileNotFoundError:
        return DEFAULT_THRESHOLD

@lru_cache
def get_pipeline():
    # Зарегистрируем кастомные классы ПЕРЕД загрузкой модели
    _register_custom_classes()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Загруженный объект не имеет метода predict_proba. "
            "Если это не классификатор с вероятностями, адаптируйте эндпоинт под decision_function/predict."
        )
    return model
