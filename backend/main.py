"""FastAPI application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `models` and `database` packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles

from backend.routers import models, query, processing, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    from backend.deps import get_active_processor
    proc = get_active_processor()
    if proc and proc.is_running:
        proc.stop()


app = FastAPI(title="Semantic Video Search", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(query.router)
app.include_router(processing.router)
app.include_router(ws.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


_build_dir = Path(__file__).resolve().parent.parent / "frontend" / "build"
if _build_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_build_dir), html=True), name="static")
