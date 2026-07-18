"""Model and collection information endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from models import MODEL_REGISTRY, DEFAULT_MODEL_IDS, DEFAULT_SYSTEM_PROMPTS, COLLECTION_NAMES
from backend.services.search import list_collections, list_video_names

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
def get_models():
    models = []
    for model_type in MODEL_REGISTRY:
        models.append({
            "type": model_type,
            "default_id": DEFAULT_MODEL_IDS.get(model_type, ""),
            "default_prompt": DEFAULT_SYSTEM_PROMPTS.get(model_type, None),
        })
    return {"models": models, "collections": COLLECTION_NAMES}


@router.get("/collections")
def get_collections():
    try:
        return {"collections": list_collections()}
    except Exception:
        return {"collections": list(COLLECTION_NAMES.values())}


@router.get("/collections/{name}/videos")
def get_collection_videos(name: str):
    try:
        return {"video_names": list_video_names(name)}
    except Exception:
        return {"video_names": []}
