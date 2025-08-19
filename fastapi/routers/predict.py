# fastapi/routers/predict.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, List
from io import BytesIO
import numpy as np
import pandas as pd

from schemas import Features, PredictResponse, PredictCSVResponse, PredictItem
from deps.model import get_pipeline, get_threshold

router = APIRouter()

@router.get("/ping")
def ping():
    return {"msg": "pong"}

def _safe_get_pipeline():
    """Аккуратно загружаем модель, чтобы отдавать понятные ошибки."""
    try:
        return get_pipeline()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

def _infer_labels_and_probs(pipe, X: pd.DataFrame, threshold: float):
    """
    Универсальный инференс:
      - если есть predict_proba -> берём proba[:,1]
      - elif есть decision_function -> применяем сигмоиду как эвристику и порог
      - else -> берём predict(), а "вероятность" = {0.0, 1.0}
    Возвращает: (labels: np.ndarray[int], probs: np.ndarray[float])
    """
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X)[:, 1]
        labels = (probs >= threshold).astype(int)
        return labels, probs

    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        # Эвристика: приводим к [0,1] через сигмоиду.
        # Это не калиброванные вероятности, но даёт понятную шкалу.
        probs = 1.0 / (1.0 + np.exp(-scores))
        labels = (probs >= threshold).astype(int)
        return labels, probs

    # Фолбэк: только predict()
    preds = pipe.predict(X)
    # делаем вид, что это "жёсткая" вероятность
    probs = preds.astype(float)
    labels = preds.astype(int)
    return labels, probs

@router.post("/predict", response_model=PredictResponse, summary="Предсказание по одному объекту")
def predict_one(payload: Features):
    pipe = _safe_get_pipeline()
    try:
        data: Dict[str, Any] = payload.model_dump(exclude_none=True)
        if not data:
            raise ValueError("Тело запроса пустое — отправьте хотя бы один признак.")
        X = pd.DataFrame([data])
        thr = get_threshold()
        labels, probs = _infer_labels_and_probs(pipe, X, thr)
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
        # 1-я попытка — обычный CSV (запятая)
        df = pd.read_csv(BytesIO(content))
        # если получилась одна колонка — попробуем ';'
        if df.shape[1] == 1:
            df = pd.read_csv(BytesIO(content), sep=";")

        if df.empty:
            raise ValueError("CSV пустой")

        thr = get_threshold()
        labels, probs = _infer_labels_and_probs(pipe, df, thr)

        items: List[PredictItem] = [
            PredictItem(row_id=int(i), label=int(l), proba=float(p))
            for i, (l, p) in enumerate(zip(labels, probs), start=1)
        ]
        pos = int(labels.sum())
        neg = int(len(labels) - pos)

        return PredictCSVResponse(total=len(df), positive=pos, negative=neg, items=items)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV prediction error: {e}")

@router.get("/model_status", summary="Проверка, что модель загрузилась")
def model_status():
    try:
        pipe = _safe_get_pipeline()
        status = {
            "ok": True,
            "has_predict_proba": bool(hasattr(pipe, "predict_proba")),
            "has_decision_function": bool(hasattr(pipe, "decision_function")),
            "has_predict": bool(hasattr(pipe, "predict")),
            "threshold": get_threshold(),
        }
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
