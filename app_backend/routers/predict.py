# fastapi/routers/predict.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO
import re
import numpy as np
import pandas as pd

from ..schemas import Features, PredictResponse, PredictCSVResponse, PredictItem
from ..deps.model import get_pipeline, get_threshold, get_expected_features
router = APIRouter()

# -------------------- УТИЛИТЫ --------------------

def _safe_get_pipeline():
    try:
        return get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

def _infer_labels_and_probs(pipe, X: pd.DataFrame, threshold: float):
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X)[:, 1]
        labels = (probs >= threshold).astype(int)
        return labels, probs
    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-scores))
        labels = (probs >= threshold).astype(int)
        return labels, probs
    preds = pipe.predict(X)
    probs = preds.astype(float)
    labels = preds.astype(int)
    return labels, probs

def _read_csv_smart(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(content))
    if df.shape[1] == 1:
        df = pd.read_csv(BytesIO(content), sep=";")
    return df

# === твоя функция: приводим имена к snake_case ===
def to_snake_case(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r'[\s\-]+', '_', name)                # пробелы и дефисы → _
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name) # camelCase → camel_case
    name = re.sub(r'[^\w_]', '', name)                  # убрать лишние символы (и CK-MB → ck_mb)
    return name.lower()

# === твоя логика нормализации gender ===
def normalize_gender(series: pd.Series) -> pd.Series:
    """
    male/female, '1'/'0', '1.0'/'0.0' -> {1.0, 0.0}, NaN сохраняем
    """
    s = series.astype(str).str.strip().str.lower()
    s = (s.replace({"male": "1", "female": "0"})
           .str.replace(".0", "", regex=False)
           .replace({"nan": np.nan}))
    return pd.to_numeric(s, errors="coerce")

# синонимы (ключи должны быть КАНОНИЧЕСКИМИ именами, по которым обучалась модель)
# я предположил канон-имена в snake_case, исходя из твоего списка
KNOWN_ALIASES: Dict[str, List[str]] = {
    "age": ["Age"],
    "cholesterol": ["Cholesterol", "chol", "chol_mgdl", "cholesterol_mg_dl"],
    "heart_rate": ["Heart rate", "thalach", "max_heart_rate", "maxhr", "hr"],
    "diabetes": ["Diabetes"],
    "family_history": ["Family History"],
    "smoking": ["Smoking"],
    "obesity": ["Obesity"],
    "alcohol_consumption": ["Alcohol Consumption"],
    "exercise_hours_per_week": ["Exercise Hours Per Week", "exercise_hours", "exercise_per_week", "exercise"],
    "diet": ["Diet"],
    "previous_heart_problems": ["Previous Heart Problems"],
    "medication_use": ["Medication Use"],
    "stress_level": ["Stress Level"],
    "sedentary_hours_per_day": ["Sedentary Hours Per Day", "sedentary_hours", "sedentary_time", "sitting_hours"],
    "income": ["Income", "annual_income", "salary", "income_usd"],
    "bmi": ["BMI", "body_mass_index", "imc", "индекс массы тела", "imt"],
    "triglycerides": ["Triglycerides", "trig", "triglyceride", "triglycerides_mgdl"],
    "physical_activity_days_per_week": ["Physical Activity Days Per Week"],
    "sleep_hours_per_day": ["Sleep Hours Per Day", "sleep_hours", "sleep_duration", "sleep"],
    "blood_sugar": ["Blood sugar", "fbs", "fasting_blood_sugar", "blood_sugar_mgdl"],
    "ck_mb": ["CK-MB", "ckmb"],
    "troponin": ["Troponin"],
    "gender": ["Gender", "sex"],  # + твоя нормализация male/female → 1/0
    "systolic_blood_pressure": ["Systolic blood pressure", "trestbps", "sbp", "ap_hi", "systolic_bp"],
    "diastolic_blood_pressure": ["Diastolic blood pressure", "dbp", "ap_lo", "diastolic_bp"],
}

def _build_alias_map(expected: List[str]) -> Dict[str, str]:
    """
    Строим карту: snake-ключ -> каноническое имя из expected.
    (и только для тех канонических, которые действительно ожидаются моделью)
    """
    amap: Dict[str, str] = {}
    # канонические имена
    for canon in expected:
        amap[to_snake_case(canon)] = canon
    # синонимы для ожидаемых канонов
    for canon, syns in KNOWN_ALIASES.items():
        if canon in expected:
            amap[to_snake_case(canon)] = canon
            for s in syns:
                amap[to_snake_case(s)] = canon
    return amap

def _normalize_df_columns(df: pd.DataFrame, expected: List[str]) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    1) приводим названия колонок df к snake_case,
    2) маппим синонимы → канон,
    3) добавляем недостающие (NaN),
    4) упорядочиваем: expected → остальные,
    5) нормализуем gender, если есть.
    """
    if not expected:
        return df, [], {}

    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]

    amap = _build_alias_map(expected)
    renames: Dict[str, str] = {}
    for c in list(df.columns):
        if c in amap:
            canon = amap[c]
            if canon not in df.columns or canon == c:
                renames[c] = canon
    df = df.rename(columns=renames)

    # gender → {0,1}
    if "gender" in df.columns:
        df["gender"] = normalize_gender(df["gender"])

    # добавим отсутствующие фичи
    missing = [c for c in expected if c not in df.columns]
    for c in missing:
        df[c] = np.nan

    # порядок: expected → другие
    ordered = expected + [c for c in df.columns if c not in expected]
    df = df[ordered]

    return df, missing, renames

def _normalize_payload_dict(d: Dict[str, Any], expected: List[str]) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    """
    То же самое для JSON: ключи → snake_case, синонимы → канон, добавим отсутствующие, нормализуем gender.
    """
    if not expected:
        return d, [], {}

    amap = _build_alias_map(expected)
    out: Dict[str, Any] = {}
    renames: Dict[str, str] = {}

    for k, v in d.items():
        k_snake = to_snake_case(k)
        if k_snake in amap:
            canon = amap[k_snake]
            out[canon] = v
            if canon != k:
                renames[k] = canon
        else:
            out[k_snake] = v

    # gender → {0,1}
    if "gender" in out and out["gender"] is not None:
        out["gender"] = pd.Series([out["gender"]])
        out["gender"] = normalize_gender(out["gender"]).iloc[0]

    missing = [c for c in expected if c not in out]
    for c in missing:
        out[c] = None

    return out, missing, renames

# -------------------- СЕРВИС --------------------

@router.get("/ping")
def ping():
    return {"msg": "pong"}

@router.get("/model_status", summary="Проверка, что модель загрузилась")
def model_status():
    try:
        pipe = _safe_get_pipeline()
        feats = get_expected_features()

        # соберём список шагов пайплайна (имя шага + класс)
        pipeline_steps = []
        try:
            for name, obj in getattr(pipe, "steps", []):
                pipeline_steps.append({"name": name, "type": type(obj).__name__})
        except Exception:
            pass

        return {
            "ok": True,
            "has_predict_proba": bool(hasattr(pipe, "predict_proba")),
            "has_decision_function": bool(hasattr(pipe, "decision_function")),
            "has_predict": bool(hasattr(pipe, "predict")),
            "threshold": get_threshold(),
            "expected_features": feats,
            "pipeline_steps": pipeline_steps,   # <-- теперь видно порядок шагов
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# -------------------- ИНФЕРЕНС --------------------

@router.post("/predict", response_model=PredictResponse, summary="Предсказание по одному объекту")
def predict_one(payload: Features):
    pipe = _safe_get_pipeline()
    try:
        raw = payload.model_dump(exclude_none=True)
        if not raw:
            raise ValueError("Тело запроса пустое — отправьте хотя бы один признак.")

        expected = get_expected_features()
        data, missing, renames = _normalize_payload_dict(raw, expected)
        X = pd.DataFrame([data])

        labels, probs = _infer_labels_and_probs(pipe, X, get_threshold())
        return PredictResponse(label=int(labels[0]), proba=float(probs[0]))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

@router.post(
    "/predict_csv",
    response_model=PredictCSVResponse,
    summary="Предсказания для CSV-файла (каждая строка — объект)"
)
async def predict_csv(file: UploadFile = File(...)):
    pipe = _safe_get_pipeline()

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (*.csv)")

    try:
        content = await file.read()
        df = _read_csv_smart(content)
        if df.empty:
            raise ValueError("CSV пустой")

        expected = get_expected_features()
        df_norm, missing, renames = _normalize_df_columns(df, expected)

        labels, probs = _infer_labels_and_probs(pipe, df_norm, get_threshold())

        items: List[PredictItem] = [
            PredictItem(row_id=int(i), label=int(l), proba=float(p))
            for i, (l, p) in enumerate(zip(labels, probs), start=1)
        ]
        pos = int(labels.sum())
        neg = int(len(labels) - pos)

        return PredictCSVResponse(total=len(df_norm), positive=pos, negative=neg, items=items)

    except HTTPException:
        raise
    except Exception as e:
        cols = list(df.columns) if 'df' in locals() else []
        raise HTTPException(
            status_code=400,
            detail=f"CSV prediction error: {e} | incoming_cols={cols}"
        )
