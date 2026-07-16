"""WebSocket endpoint for streaming captions in real time."""

from __future__ import annotations

import json
import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from models import DEFAULT_MODEL_IDS, COLLECTION_NAMES
from backend.deps import get_active_processor, set_active_processor
from backend.services.processor import VideoProcessor

router = APIRouter(tags=["ws"])


@router.websocket("/ws/captions")
async def captions_ws(ws: WebSocket):
    await ws.accept()

    processor: VideoProcessor | None = None
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "start":
                existing = get_active_processor()
                if existing and existing.is_running:
                    await ws.send_json({"type": "error", "message": "Processing already active"})
                    continue

                config = msg.get("config", {})
                model_type = config.get("model_type", "lfm2.5")
                model_id = DEFAULT_MODEL_IDS.get(model_type, model_type)
                collection_name = config.get("collection_name") or COLLECTION_NAMES.get(
                    model_type, f"captions_{model_type}"
                )

                async def send_caption(data: dict[str, Any]) -> None:
                    await ws.send_json({"type": "caption", "data": data})

                async def send_status(message: str) -> None:
                    await ws.send_json({"type": "status", "message": message})

                async def send_error(message: str) -> None:
                    await ws.send_json({"type": "error", "message": message})

                async def send_done() -> None:
                    await ws.send_json({"type": "done"})

                def _on_caption(data: dict[str, Any]) -> None:
                    asyncio.run_coroutine_threadsafe(send_caption(data), loop)

                def _on_status(message: str) -> None:
                    asyncio.run_coroutine_threadsafe(send_status(message), loop)

                def _on_error(message: str) -> None:
                    asyncio.run_coroutine_threadsafe(send_error(message), loop)

                def _on_done() -> None:
                    asyncio.run_coroutine_threadsafe(send_done(), loop)

                processor = VideoProcessor(
                    video_path=config.get("video_path", ""),
                    model_type=model_type,
                    model_id=model_id,
                    system_prompt=config.get("system_prompt"),
                    caption_interval=config.get("caption_interval", 1.0),
                    max_new_tokens=config.get("max_new_tokens", 500),
                    max_frames=config.get("max_frames") if config.get("max_frames", 0) > 0 else None,
                    collection_name=collection_name,
                    vector_db_path=config.get("vector_db_path", "./video_captions_db"),
                    fps=config.get("fps", 20),
                    clip_duration=config.get("clip_duration", 4.0),
                    on_caption=_on_caption,
                    on_status=_on_status,
                    on_error=_on_error,
                    on_done=_on_done,
                )

                set_active_processor(processor)
                processor.start()
                await ws.send_json({"type": "status", "message": f"Processing started (video_id: {processor.video_id})"})

            elif msg_type == "stop":
                if processor and processor.is_running:
                    processor.stop()
                    await ws.send_json({"type": "status", "message": "Processing stopped"})
                else:
                    await ws.send_json({"type": "error", "message": "No active processing session"})

            else:
                await ws.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        if processor and processor.is_running:
            processor.stop()
            set_active_processor(None)
