"""Video frame extraction endpoint.

Given a video file path (as stored in caption metadata ``source``) and a
timestamp in milliseconds, extracts the closest frame and returns it as a
JPEG image.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/api", tags=["video"])

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
JPEG_QUALITY = 85
SEEK_RETRIES = 5


@router.get("/frame")
def get_frame(path: str, timestamp_ms: float = 0):
    if not path:
        raise HTTPException(status_code=400, detail="Missing 'path' query parameter")

    video_path = Path(path)

    if not video_path.is_absolute():
        raise HTTPException(
            status_code=400,
            detail=f"Path must be absolute: {path}",
        )

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video file not found: {path}")

    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {video_path.suffix}",
        )

    if timestamp_ms < 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timestamp (must be >= 0): {timestamp_ms}",
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise HTTPException(
            status_code=404,
            detail=f"Could not open video file: {path}",
        )

    try:
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        if frame_count > 0 and fps > 0:
            duration_ms = frame_count / fps * 1000.0
            if timestamp_ms > duration_ms + 10:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Timestamp {timestamp_ms}ms exceeds video duration "
                        f"({duration_ms:.0f}ms)"
                    ),
                )

        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        frame = None
        for _ in range(SEEK_RETRIES):
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
                break

        if frame is None:
            raise HTTPException(
                status_code=404,
                detail="Could not read a frame at the requested timestamp",
            )

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            raise HTTPException(
                status_code=500,
                detail="Failed to encode frame as JPEG",
            )

        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache",
                "X-Frame-Timestamp-Ms": str(timestamp_ms),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {e}")
    finally:
        capture.release()
