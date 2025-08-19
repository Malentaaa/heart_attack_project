# fastapi/routers/predict.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Dict, Any, List
import pandas as pd
from io import BytesIO

from schemas import Features, PredictResponse, PredictCSVResponse, PredictItem
from deps.model import get_pipeline, get_threshold

router = APIRouter()

@router.get("/ping")
def ping():
    return {"msg": "pong"}

@router.post("/predict", response_model=PredictResponse, summary="Предсказание по одному объекту")
def predict_one(payload: Features, pipe = Depends(get_pipeline)):
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
async def predict_csv(file: UploadFile = File(...), pipe = Depends(get_pipeline)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл")

    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))

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

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV prediction error: {e}")