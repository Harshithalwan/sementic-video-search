"""FastAPI application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `models` and `database` packages are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import settings
from backend.routers import models, query, processing, ws, video


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(query.router)
app.include_router(processing.router)
app.include_router(ws.router)
app.include_router(video.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


def main() -> None:
    """CLI entry point — run the backend server via ``python -m backend.main``.

    Equivalent to ``uvicorn backend.main:app`` but also accepts
    ``--enable-latency-logging`` so per-component latency logging can be
    enabled for sessions started through the web UI.
    """
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="backend.main",
        description="Semantic Video Search backend server.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001).")
    parser.add_argument(
        "--enable-latency-logging",
        action="store_true",
        help="Enable per-component latency logging to latency_logs/ directory (JSONL format).",
    )
    args = parser.parse_args()

    settings.LATENCY_LOGGING_ENABLED = args.enable_latency_logging

    uvicorn.run("backend.main:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()

