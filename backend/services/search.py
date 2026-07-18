"""Query service — search and filter logic extracted from Streamlit UI."""

from __future__ import annotations

from typing import Any, Optional

from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from database import VectorStore


def build_filter(
    video_name: Optional[str] = None,
    video_id: Optional[str] = None,
    ts_from: Optional[float] = None,
    ts_to: Optional[float] = None,
    time_from: Optional[float] = None,
    time_to: Optional[float] = None,
) -> Optional[Filter]:
    conditions: list[FieldCondition] = []

    if video_name:
        conditions.append(
            FieldCondition(key="video_name", match=MatchValue(value=video_name))
        )
    if video_id:
        conditions.append(
            FieldCondition(key="video_id", match=MatchValue(value=video_id))
        )
    if ts_from is not None or ts_to is not None:
        conditions.append(
            FieldCondition(
                key="video_timestamp_ms",
                range=Range(gte=ts_from, lte=ts_to),
            )
        )
    if time_from is not None or time_to is not None:
        conditions.append(
            FieldCondition(
                key="current_time_secs",
                range=Range(gte=time_from, lte=time_to),
            )
        )

    if not conditions:
        return None
    return Filter(must=conditions)


def search_captions(
    query_text: str,
    top_k: int,
    collection: str,
    video_name: Optional[str] = None,
    video_id: Optional[str] = None,
    ts_from: Optional[float] = None,
    ts_to: Optional[float] = None,
    time_from: Optional[float] = None,
    time_to: Optional[float] = None,
) -> list[dict[str, Any]]:
    store = VectorStore(collection_name=collection)
    store.ensure_collection_exists()

    qfilter = build_filter(
        video_name=video_name,
        video_id=video_id,
        ts_from=ts_from,
        ts_to=ts_to,
        time_from=time_from,
        time_to=time_to,
    )

    results = store.query(query_text, top_k=top_k, query_filter=qfilter)

    return [
        {
            "score": r.score,
            "document": r.document,
            "metadata": r.metadata or {},
        }
        for r in results
    ]


def list_collections() -> list[str]:
    from qdrant_client import QdrantClient
    from pathlib import Path

    client = QdrantClient(path=str(Path("./video_captions_db").resolve()))
    collections = client.get_collections()
    return [c.name for c in collections.collections]


def list_video_names(collection: str) -> list[str]:
    store = VectorStore(collection_name=collection)
    store.ensure_collection_exists()
    return store.list_video_names()
