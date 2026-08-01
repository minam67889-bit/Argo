"""FastAPI application entrypoint."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import router as api_router
from app.core import storage
from app.core.config import settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    storage.init_db()

    app = FastAPI(
        title="Argo",
        description="A real LLM agent toolkit — chat + coding agent in one UI.",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    # Static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Templates
    templates_dir = Path(__file__).parent / "templates"
    index_html = templates_dir / "index.html"

    @app.get("/", response_class=HTMLResponse)
    async def index():
        if index_html.exists():
            return HTMLResponse(index_html.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Argo</h1><p>UI not built yet.</p>")

    @app.get("/favicon.ico")
    async def favicon():
        fav = static_dir / "favicon.ico"
        if fav.exists():
            return FileResponse(str(fav))
        return HTMLResponse("", status_code=204)

    return app


app = create_app()


def main() -> None:
    """Run with uvicorn when called as `python -m app.main`."""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.ARGO_HOST,
        port=settings.ARGO_PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
