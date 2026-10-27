from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from app.api.routes import history, search

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Credisearch API",
    description="Price search across sources, with ML product matching, deal scoring and price history.",
)

app.include_router(search.router)
app.include_router(history.router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def _png(name: str) -> Response:
    return Response(
        (STATIC_DIR / name).read_bytes(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon-32.png", include_in_schema=False)
def favicon():
    return _png("favicon-32.png")


@app.get("/apple-touch-icon.png", include_in_schema=False)
def apple_touch_icon():
    return _png("apple-touch-icon.png")


@app.get("/health")
def health_check():
    return {"status": "ok"}
