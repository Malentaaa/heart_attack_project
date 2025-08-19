from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from routers import predict

app = FastAPI(title="Heart Attack Risk API", version="0.1.0")
app.include_router(predict.router, prefix="/api", tags=["predict"])

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 1) статика по адресу /static/...
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 2) корень "/" — отдать index.html
@app.get("/", include_in_schema=False)
def root():
    return FileResponse(STATIC_DIR / "index.html")

# health теперь не перекрывается статикой
@app.get("/health", tags=["service"])
def health():
    return {"status": "ok"}