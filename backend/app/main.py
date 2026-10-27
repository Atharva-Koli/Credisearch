from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.api.routes import search

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Credisearch API",
    description="Price search across sources, with ML product matching and deal scoring.",
)

app.include_router(search.router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health_check():
    return {"status": "ok"}
