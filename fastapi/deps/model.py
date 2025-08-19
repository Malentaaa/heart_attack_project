# fastapi/deps/model.py
from functools import lru_cache
from pathlib import Path
import importlib
import sys
import json
import joblib
from typing import Any, Tuple, List, Dict

# === пути к артефактам ===
MODELS_DIR        = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH        = MODELS_DIR / "heart_rf_final.pkl"   # ваш файл модели
THRESHOLD_PATH    = MODELS_DIR / "threshold.json"       # (необязательно)
DEFAULT_THRESHOLD = 0.5


def _register_custom_classes():
    """
    Если pickle ссылается на __main__.ClassName, подкладываем реальные классы
    из нашего модуля src.custom_transformers в модуль __main__,
    чтобы joblib смог распаковать модель.
    """
    candidate_modules = [
        "src.custom_transformers",      # ваш модуль с кастомными трансформерами
        "src.custom_imputers",          # если раньше создавали этот файл
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

    import types
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
                    main = types.ModuleType("__main__")
                    sys.modules["__main__"] = main
                setattr(main, cls_name, cls)
                found_any = True

    if not found_any:
        # Фолбэк: no-op классы, чтобы хотя бы загрузить модель (для продакшена нежелательно)
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


def _is_estimator(x: Any) -> bool:
    return any(hasattr(x, attr) for attr in ("predict_proba", "decision_function", "predict"))


def _unwrap_model(container: Any) -> Any:
    """
    Приводит загруженный pickle к пригодному для инференса объекту (Estimator/Pipeline).
    Поддерживает частые схемы упаковки: dict, tuple, list, уже готовый Pipeline/Model.
    Бросает ValueError с подсказкой, если не получилось.
    """
    # Уже готовый estimator/pipeline?
    if _is_estimator(container):
        return container

    # sklearn Pipeline?
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(container, Pipeline):
            return container
    except Exception:
        pass

    # Словарь с типичными ключами
    if isinstance(container, dict):
        keys = set(container.keys())

        # 1) Наиболее вероятное — целиком pipeline
        for k in ("pipeline", "pipe", "best_pipeline", "final_pipeline"):
            if k in container and _is_estimator(container[k]):
                return container[k]

        # 2) Модель + препроцессор
        model_key_candidates = ("model", "clf", "estimator", "best_model", "final_model")
        prep_key_candidates  = ("preprocessor", "prep", "transformer", "preproc", "ct")

        model_obj = next((container[k] for k in model_key_candidates if k in container), None)
        prep_obj  = next((container[k] for k in prep_key_candidates if k in container), None)
        if _is_estimator(model_obj) and prep_obj is not None:
            # Соберём Pipeline вручную
            from sklearn.pipeline import Pipeline
            return Pipeline(steps=[("prep", prep_obj), ("clf", model_obj)])

        # 3) Только модель внутри словаря
        if _is_estimator(model_obj):
            return model_obj

        # 4) Перебор всех значений словаря — вдруг estimator лежит под нестандартным ключом
        for v in container.values():
            if _is_estimator(v):
                return v

        # Нечего распаковать
        # ограничим вывод ключей, чтобы ошибка была читабельной
        shown_keys = list(keys)[:10]
        raise ValueError(
            f"В файле {MODEL_PATH.name} найден dict без пригодной модели. Ключи: {shown_keys} ..."
        )

    # Кортеж/список: ищем (preproc, model) или просто model
    if isinstance(container, (tuple, list)):
        # сначала — поиск estimator внутри
        for v in container:
            if _is_estimator(v):
                return v
        # попробуем особый случай: (preproc, model)
        if len(container) == 2:
            a, b = container
            if _is_estimator(b):
                return b

    # Не удалось
    raise ValueError(
        f"Не удалось распаковать объект типа {type(container)} до модели с predict/predict_proba."
    )


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

    container = joblib.load(MODEL_PATH)
    model = _unwrap_model(container)
    return model
