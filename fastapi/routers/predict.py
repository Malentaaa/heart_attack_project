# fastapi/routers/predict.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, List
from io import BytesIO
import pandas as pd

from schemas import Features, PredictResponse, PredictCSVResponse, PredictItem
from deps.model import get_pipeline, get_threshold

router = APIRouter()

@router.get("/ping")
def ping():
    return {"msg": "pong"}

def _safe_get_pipeline():
    """Отдельная функция, чтобы красиво ловить ошибки загрузки модели."""
    try:
        return get_pipeline()
    except FileNotFoundError as e:
        # Чётко подсветим, что не найден файл модели/препроцессора
        raise HTTPException(status_code=500, detail=f"Model file not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

@router.post("/predict", response_model=PredictResponse, summary="Предсказание по одному объекту")
def predict_one(payload: Features):
    pipe = _safe_get_pipeline()
    try:
        X = pd.DataFrame([payload.model_dump(exclude_none=True)])
        proba = float(pipe.predict_proba(X)[:, 1][0])
        label = int(proba >= get_threshold())
        return PredictResponse(label=label, proba=proba)
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

        # 1-я попытка: стандартный CSV с запятой
        df = pd.read_csv(BytesIO(content))

        # Если прочлась одна большая колонка, попробуем ; как разделитель
        if df.shape[1] == 1:
            df = pd.read_csv(BytesIO(content), sep=";")

        if df.empty:
            raise ValueError("CSV пустой")

        probs = pipe.predict_proba(df)[:, 1]
        thr = get_threshold()
        labels = (probs >= thr).astype(int)

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
        _ = get_threshold()
        _ = get_pipeline()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))