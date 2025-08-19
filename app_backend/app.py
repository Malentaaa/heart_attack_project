from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from .routers import predict

app = FastAPI(title="Heart Attack API", version="1.0.0")

# API
app.include_router(predict.router, prefix="/api")

# Статика (/static/*)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Корень -> index.html (или fallback HTML)
@app.get("/", include_in_schema=False)
def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        try:
            html = index_path.read_text(encoding="utf-8", errors="ignore")
            return HTMLResponse(html)
        except Exception as e:
            return HTMLResponse(f"<!doctype html><h1>Ошибка чтения index.html</h1><pre>{e}</pre>", status_code=500)
    # fallback — поможет понять, что роут точно работает
    return HTMLResponse("<!doctype html><h1>It works ✅</h1><p>Положи index.html в app_backend/static/</p>")

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

@app.get("/__debug", include_in_schema=False)
def debug():
    files = []
    try:
        files = [p.name for p in static_dir.glob("*")]
    except Exception:
        pass
    return {
        "cwd": str(Path.cwd()),
        "static_dir": str(static_dir.resolve()),
        "index_exists": (static_dir / "index.html").exists(),
        "files": files,
    }
