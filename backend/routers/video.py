"""Video file streaming endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api", tags=["video"])

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}


@router.get("/video")
def stream_video(request: Request, path: str):
    if not path:
        raise HTTPException(status_code=400, detail="Missing 'path' query parameter")

    video_path = Path(path)

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {path}")

    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {video_path.suffix}")

    file_size = video_path.stat().st_size
    content_type = _get_content_type(video_path.suffix)

    range_header = request.headers.get("range")

    if range_header:
        range_start, range_end = _parse_range(range_header, file_size)
        content_length = range_end - range_start + 1

        def iter_range():
            with open(video_path, "rb") as f:
                f.seek(range_start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(1024 * 1024, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {range_start}-{range_end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )

    def iter_file():
        with open(video_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(
        iter_file(),
        status_code=200,
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )


def _parse_range(range_header: str, file_size: int) -> tuple[int, int]:
    try:
        range_spec = range_header.split("=")[1]
        parts = range_spec.split("-")
        range_start = int(parts[0])
        range_end = int(parts[1]) if parts[1] else file_size - 1
    except (IndexError, ValueError):
        raise HTTPException(status_code=416, detail="Invalid Range header")

    if range_start >= file_size or range_end >= file_size:
        raise HTTPException(status_code=416, detail="Range not satisfiable")

    return range_start, range_end


def _get_content_type(suffix: str) -> str:
    return {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
        ".flv": "video/x-flv",
        ".wmv": "video/x-ms-wmv",
    }.get(suffix.lower(), "video/mp4")
