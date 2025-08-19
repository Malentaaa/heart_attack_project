# fastapi/schemas.py
from typing import Optional, List
from pydantic import BaseModel, ConfigDict, Field

# Разрешим присылать любые фичи (полезно, если их много)
class Features(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: Optional[float] = Field(None, description="Возраст")
    sex: Optional[int] = Field(None, description="Пол (0/1)")
    chol: Optional[float] = None
    trestbps: Optional[float] = None
    thalach: Optional[float] = None

class PredictResponse(BaseModel):
    label: int
    proba: float

class PredictItem(BaseModel):
    row_id: int
    label: int
    proba: float

class PredictCSVResponse(BaseModel):
    total: int
    positive: int
    negative: int
    items: List[PredictItem]
