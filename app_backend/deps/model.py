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
EXPECTED_JSON     = MODELS_DIR / "expected_features.json"


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
    Возвращаем пригодный для инференса sklearn.Pipeline / Estimator.
    Если это Pipeline — гарантируем порядок prep -> selector? -> clf.
    Если clf не найден, бросаем понятную ошибку со списком шагов.
    """
    from sklearn.pipeline import Pipeline

    def _is_estimator_like(obj: Any) -> bool:
        return any(hasattr(obj, a) for a in ("predict_proba", "decision_function", "predict"))

    def _role(name: str, obj: Any) -> str:
        n = (name or "").lower()
        # явный препроцессор по типу/названию
        try:
            from sklearn.compose import ColumnTransformer
            if isinstance(obj, ColumnTransformer):
                return "prep"
        except Exception:
            pass
        if any(k in n for k in ("prep", "preproc", "preprocessor", "ct", "transformer")):
            return "prep"
        # селектор
        try:
            from sklearn.feature_selection._base import SelectorMixin
            if isinstance(obj, SelectorMixin):
                return "sel"
        except Exception:
            pass
        if any(k in n for k in ("select", "kbest", "selector", "fs", "feature_select")):
            return "sel"
        # модель
        if any(k in n for k in ("clf", "model", "estimator", "rf", "xgb", "lgb", "svc", "logreg")):
            return "clf"
        if _is_estimator_like(obj):
            return "clf"
        return "other"

    def _reorder_pipeline(p: Pipeline) -> Pipeline:
        steps = list(p.steps)
        roles = [(_role(n, o), n, o) for (n, o) in steps]
        prep = next(((i, n, o) for i, (r, n, o) in enumerate(roles) if r == "prep"), None)
        sel  = next(((i, n, o) for i, (r, n, o) in enumerate(roles) if r == "sel"), None)
        clf  = next(((i, n, o) for i, (r, n, o) in enumerate(roles) if r == "clf"), None)

        if clf is None:
            names = [n for _, n, _ in roles]
            raise ValueError(f"В Pipeline не найден классификатор (последний шаг). Шаги: {names}")

        new_steps = []
        if prep: new_steps.append((prep[1], prep[2]))
        if sel:  new_steps.append((sel[1],  sel[2]))
        new_steps.append((clf[1], clf[2]))  # clf строго последним

        # добавим прочие, не дублируя
        core = {n for n, _ in new_steps}
        for n, o in steps:
            if n not in core:
                new_steps.append((n, o))

        return Pipeline(new_steps)

    # --- распаковка контейнера ---
    # 1) Уже Pipeline?
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(container, Pipeline):
            return _reorder_pipeline(container)
    except Exception:
        pass

    # 2) Просто estimator?
    if _is_estimator(container):
        return container

    # 3) dict
    if isinstance(container, dict):
        # готовый pipeline по ключу
        for k in ("pipeline", "pipe", "final_pipeline", "best_pipeline"):
            if k in container:
                obj = container[k]
                if isinstance(obj, Pipeline):
                    return _reorder_pipeline(obj)
                if _is_estimator(obj):
                    return obj
        # собрать из частей: preprocessor -> selector? -> model
        model_obj = next((container.get(k) for k in ("model","clf","estimator","best_model","final_model") if k in container), None)
        prep_obj  = next((container.get(k) for k in ("preprocessor","prep","transformer","preproc","ct") if k in container), None)
        sel_obj   = next((container.get(k) for k in ("selector","kbest","feature_selector","feature_selection","fs","select") if k in container), None)
        if prep_obj is not None and model_obj is not None:
            return Pipeline([("prep", prep_obj), *([("select", sel_obj)] if sel_obj is not None else []), ("clf", model_obj)])

        # любой estimator среди значений
        for v in container.values():
            if _is_estimator(v):
                return v

        raise ValueError(f"dict без пригодной модели/препроцессора. Ключи: {list(container.keys())[:12]} ...")

    # 4) tuple/list — ищем estimator
    if isinstance(container, (tuple, list)):
        for v in container:
            if _is_estimator(v):
                return v
        if len(container) == 2 and _is_estimator(container[1]):
            return container[1]

    raise ValueError(f"Не удалось распаковать объект типа {type(container)} до модели с predict/predict_proba.")


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

@lru_cache
def get_expected_features() -> List[str]:
    """
    Возвращает список СЫРЫХ признаков, которые ожидает пайплайн.
    Порядок важен (ставим как в feature_names_in_).
    Источники:
      1) feature_names_in_ у препроцессора/модели (если есть)
      2) models/expected_features.json (если есть)
      3) иначе — пустой список (тогда нормализация в роутере мягкая)
    """
    pipe = get_pipeline()

    # 1) Прямо на объекте
    for obj in (pipe, getattr(pipe, "preprocessor", None)):
        if obj is not None and hasattr(obj, "feature_names_in_"):
            try:
                return list(getattr(obj, "feature_names_in_"))
            except Exception:
                pass

    # 1.1) Если это sklearn.Pipeline — попробуем по шагам
    try:
        steps = getattr(pipe, "named_steps", {})
        for name in ("prep", "preprocessor", "transformer", "ct"):
            if name in steps and hasattr(steps[name], "feature_names_in_"):
                return list(getattr(steps[name], "feature_names_in_"))
    except Exception:
        pass

    # 2) Файл-список (опционально)
    try:
        if EXPECTED_JSON.exists():
            import json
            with open(EXPECTED_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            if isinstance(data, list):
                return list(data)
    except Exception:
        pass

    # 3) Не нашли — вернём пустой список
    return []
