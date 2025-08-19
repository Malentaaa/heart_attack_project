# fastapi/deps/model.py
from functools import lru_cache
from pathlib import Path
import importlib
import sys
import json
import joblib

# === пути к артефактам ===
MODELS_DIR      = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH      = MODELS_DIR / "heart_rf_final.pkl"   # ваш файл модели
THRESHOLD_PATH  = MODELS_DIR / "threshold.json"       # (необязательно)
DEFAULT_THRESHOLD = 0.5

def _register_custom_classes():
    """
    Если pickle ссылается на __main__.GroupMedianImputer,
    делаем алиас: подкладываем реальный класс в модуль __main__.
    """
    candidate_modules = [
        "src.custom_imputers",           # <== наш новый модуль
        "src.modeling.dataprocessor",    # на случай, если у тебя уже есть такой
        "src.dataprocessor",
        "dataprocessor",
        "src.transformers",
        "transformers",
    ]
    custom_class_names = ["GroupMedianImputer"]

    found_any = False
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
                    import types
                    main = types.ModuleType("__main__")
                    sys.modules["__main__"] = main
                setattr(main, cls_name, cls)
                found_any = True

    if not found_any:
        # Фолбэк: подложим no-op версию, чтобы хотя бы загрузить pickle.
        # (Лучше, конечно, иметь реальный класс — см. src/custom_imputers.py)
        try:
            from sklearn.base import BaseEstimator, TransformerMixin
            class GroupMedianImputer(BaseEstimator, TransformerMixin):
                def __init__(self, *args, **kwargs): pass
                def fit(self, X, y=None): return self
                def transform(self, X):  return X
            main = sys.modules.get("__main__")
            if main is None:
                import types
                main = types.ModuleType("__main__")
                sys.modules["__main__"] = main
            setattr(main, "GroupMedianImputer", GroupMedianImputer)
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
    # зарегистрируем кастомные классы ПЕРЕД загрузкой
    _register_custom_classes()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # проверим, что есть вероятности
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Загруженный объект не имеет метода predict_proba. "
            "Если это не классификатор с вероятностями, адаптируйте код под decision_function/predict."
        )
    return model
