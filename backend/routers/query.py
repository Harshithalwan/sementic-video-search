"""Query/search endpoint."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.search import search_captions

router = APIRouter(prefix="/api", tags=["query"])


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    collection: str = "captions_lfm2.5"
    video_name: Optional[str] = None
    video_id: Optional[str] = None
    ts_from: Optional[float] = None
    ts_to: Optional[float] = None
    time_from: Optional[float] = None
    time_to: Optional[float] = None


@router.post("/query")
def query_captions(req: QueryRequest):
    if not req.query.strip():
        return {"error": "Query cannot be empty", "results": []}

    try:
        results = search_captions(
            query_text=req.query,
            top_k=req.top_k,
            collection=req.collection,
            video_name=req.video_name,
            video_id=req.video_id,
            ts_from=req.ts_from,
            ts_to=req.ts_to,
            time_from=req.time_from,
            time_to=req.time_to,
        )
        return {"results": results}
    except Exception as e:
        return {"error": str(e), "results": []}
