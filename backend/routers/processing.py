"""Video processing REST endpoints — start, stop, status."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from models import DEFAULT_MODEL_IDS, COLLECTION_NAMES
from backend.deps import get_active_processor, set_active_processor

router = APIRouter(prefix="/api", tags=["processing"])


class ProcessStartRequest(BaseModel):
    video_path: str
    model_type: str = "lfm2.5"
    system_prompt: Optional[str] = None
    caption_interval: float = 1.0
    max_new_tokens: int = 500
    max_frames: Optional[int] = None
    collection_name: Optional[str] = None
    fps: int = 20
    clip_duration: float = 4.0


class ProcessStopRequest(BaseModel):
    pass


@router.post("/process/start")
def start_processing(req: ProcessStartRequest):
    existing = get_active_processor()
    if existing and existing.is_running:
        raise HTTPException(status_code=409, detail="Processing already active")

    model_id = DEFAULT_MODEL_IDS.get(req.model_type, req.model_type)
    collection_name = req.collection_name or COLLECTION_NAMES.get(
        req.model_type, f"captions_{req.model_type}"
    )

    from backend.services.processor import VideoProcessor

    processor = VideoProcessor(
        video_path=req.video_path,
        model_type=req.model_type,
        model_id=model_id,
        system_prompt=req.system_prompt,
        caption_interval=req.caption_interval,
        max_new_tokens=req.max_new_tokens,
        max_frames=req.max_frames if req.max_frames and req.max_frames > 0 else None,
        collection_name=collection_name,
        vector_db_path="./video_captions_db",
        fps=req.fps,
        clip_duration=req.clip_duration,
        on_caption=lambda _: None,
        on_status=lambda _: None,
        on_error=lambda _: None,
        on_done=lambda _: None,
    )

    set_active_processor(processor)
    return {"video_id": processor.video_id, "status": "started"}


@router.post("/process/stop")
def stop_processing(_req: ProcessStopRequest):
    processor = get_active_processor()
    if processor is None or not processor.is_running:
        raise HTTPException(status_code=404, detail="No active processing session")

    processor.stop()
    return {"status": "stopped"}


@router.get("/process/status")
def process_status():
    processor = get_active_processor()
    if processor is None or not processor.is_running:
        return {"active": False}

    return {
        "active": True,
        "video_id": processor.video_id,
        "captions_count": processor.caption_count,
    }
