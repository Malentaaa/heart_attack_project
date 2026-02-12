# app_backend/deps/model.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List

import importlib
import sys
import json
import joblib
import pickle

# === пути к артефактам (корень проекта → models) ===
MODELS_DIR        = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH        = MODELS_DIR / "heart_rf_final.pkl"
THRESHOLD_PATH    = MODELS_DIR / "threshold.json"
EXPECTED_JSON     = MODELS_DIR / "expected_features.json"
DEFAULT_THRESHOLD = 0.5

# буфер для threshold, считанного из pkl
_THRESHOLD_FROM_MODEL: float | None = None


# ---------------------------------------------------------------------
# Кастомные классы из твоего src — подложим их в __main__, чтобы
# распаковка joblib/pickle прошла, даже если в артефакте ссылки на __main__.
# ---------------------------------------------------------------------
def _register_custom_classes() -> None:
    candidate_modules = [
        "src.custom_transformers",
        "src.custom_imputers",
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
    main = sys.modules.get("__main__")
    if main is None:
        main = types.ModuleType("__main__")
        sys.modules["__main__"] = main

    found_any = False
    for mod_name in candidate_modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for cls_name in custom_class_names:
            if hasattr(mod, cls_name):
                setattr(main, cls_name, getattr(mod, cls_name))
                found_any = True

    if not found_any:
        # мягкий фолбэк — no-op трансформеры (годится для инференса, если препроцессор отсутствует).
        try:
            from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
            class _NoOp(BaseEstimator, TransformerMixin):
                def fit(self, X, y=None): return self
                def transform(self, X):  return X
            for cls_name in custom_class_names:
                setattr(main, cls_name, type(cls_name, (_NoOp,), {}))
        except Exception:
            pass


# ---------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------
def _is_estimator_like(obj: Any) -> bool:
    return any(hasattr(obj, a) for a in ("predict_proba", "decision_function", "predict"))


def _reorder_pipeline(p) -> Any:
    """Переставляем шаги в правильный порядок: prep -> select? -> clf."""
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer  # type: ignore
        try:
            from sklearn.feature_selection._base import SelectorMixin  # sklearn>=1.2
        except Exception:  # pragma: no cover
            from sklearn.feature_selection import SelectorMixin  # sklearn<1.2

        if not isinstance(p, Pipeline):
            return p

        steps = list(p.steps)

        def role(name: str, obj: Any) -> str:
            n = (name or "").lower()
            if isinstance(obj, ColumnTransformer) or n in {"prep", "preproc", "preprocessor", "ct"}:
                return "prep"
            if isinstance(obj, SelectorMixin) or any(k in n for k in ("select", "kbest", "selector", "fs")):
                return "sel"
            if any(k in n for k in ("clf", "model", "estimator", "rf", "xgb", "lgb", "svc", "logreg")):
                return "clf"
            if _is_estimator_like(obj):
                return "clf"
            return "other"

        roles = [(role(n, o), n, o) for (n, o) in steps]
        prep = next(((n, o) for r, n, o in roles if r == "prep"), None)
        sel  = next(((n, o) for r, n, o in roles if r == "sel"),  None)
        clf  = next(((n, o) for r, n, o in roles if r == "clf"),  None)

        if clf is None:
            names = [n for _, n, _ in roles]
            raise ValueError(f"В Pipeline не найден классификатор. Шаги: {names}")

        new_steps: list[tuple[str, Any]] = []
        if prep: new_steps.append(prep)
        if sel:  new_steps.append(sel)
        new_steps.append(clf)

        used = {n for n, _ in new_steps}
        for n, o in steps:
            if n not in used:
                new_steps.append((n, o))

        return Pipeline(new_steps)
    except Exception:
        return p


def _extract_from_container(container: Any) -> Any:
    """
    Достаём модель/пайплайн из контейнера, подхватываем threshold.
    Контейнером может быть Pipeline, Estimator, dict, tuple/list.
    """
    global _THRESHOLD_FROM_MODEL

    obj = container

    # dict: самый частый кейс — {"pipeline": ..., "threshold": ...}
    if isinstance(container, dict):
        if "threshold" in container:
            try:
                _THRESHOLD_FROM_MODEL = float(container["threshold"])
            except Exception:
                _THRESHOLD_FROM_MODEL = None

        for key in ("pipeline", "pipe", "final_pipeline", "best_pipeline"):
            if key in container:
                obj = container[key]
                break
        else:
            # найдём любой estimator/pipeline в значениях
            for v in container.values():
                if _is_estimator_like(v):
                    obj = v
                    break

    # список/кортеж — возьмём первый estimator
    if isinstance(obj, (list, tuple)):
        for v in obj:
            if _is_estimator_like(v):
                obj = v
                break

    # если это Pipeline — сразу починим порядок
    obj = _reorder_pipeline(obj)

    # финальная валидация
    if hasattr(obj, "steps"):  # sklearn.Pipeline
        last = obj.steps[-1][1]
        if not _is_estimator_like(last):
            names = [n for n, _ in obj.steps]
            raise ValueError(f"Последний шаг не классификатор ({type(last).__name__}). Шаги: {names}")
    elif not _is_estimator_like(obj):
        raise ValueError(f"Загруженный объект не похож на sklearn-модель/пайплайн: {type(obj)}")

    return obj


# ---------------------------------------------------------------------
# Публичные функции для роутов
# ---------------------------------------------------------------------
@lru_cache
def get_pipeline():
    """Читаем pkl (joblib → pickle fallback), регистрируем кастомные классы, чиним порядок шагов."""
    _register_custom_classes()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

    try:
        container = joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            container = pickle.load(f)

    model = _extract_from_container(container)
    return model


@lru_cache
def get_threshold() -> float:
    """Порог: threshold.json → значение из pkl → DEFAULT_THRESHOLD."""
    # 1) JSON имеет приоритет
    try:
        if THRESHOLD_PATH.exists():
            data = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
            return float(data.get("threshold", DEFAULT_THRESHOLD))
    except Exception:
        pass
    # 2) если порог был в pkl — используем его
    if _THRESHOLD_FROM_MODEL is not None:
        try:
            return float(_THRESHOLD_FROM_MODEL)
        except Exception:
            pass
    # 3) запасной вариант
    return DEFAULT_THRESHOLD


@lru_cache
def get_expected_features() -> List[str]:
    """
    Список СЫРЫХ признаков в порядке, который ожидает препроцессор.
    Источник приоритетов:
      1) feature_names_in_ у препроцессора/пайплайна,
      2) models/expected_features.json,
      3) пустой список (мягкая нормализация в роутере).
    """
    pipe = get_pipeline()

    # 1) На самом объекте
    for obj in (pipe, getattr(pipe, "preprocessor", None)):
        if obj is not None and hasattr(obj, "feature_names_in_"):
            try:
                return list(getattr(obj, "feature_names_in_"))
            except Exception:
                pass

    # 1.1) Если это sklearn.Pipeline — попробуем по именованным шагам
    try:
        steps = getattr(pipe, "named_steps", {})
        for name in ("prep", "preprocessor", "transformer", "ct"):
            if name in steps and hasattr(steps[name], "feature_names_in_"):
                return list(getattr(steps[name], "feature_names_in_"))
    except Exception:
        pass

    # 2) Файл с фичами
    try:
        if EXPECTED_JSON.exists():
            data = json.loads(EXPECTED_JSON.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            if isinstance(data, list):
                return list(data)
    except Exception:
        pass

    # 3) Не нашли — вернём пустой
    return []
