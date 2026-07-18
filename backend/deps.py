"""Shared dependencies for FastAPI routes."""

from __future__ import annotations

import threading
from typing import Optional

from backend.services.processor import VideoProcessor

# Global state for active processing session
_active_processor: Optional[VideoProcessor] = None
_processor_lock = threading.Lock()


def get_active_processor() -> Optional[VideoProcessor]:
    with _processor_lock:
        return _active_processor


def set_active_processor(processor: Optional[VideoProcessor]) -> None:
    global _active_processor
    with _processor_lock:
        _active_processor = processor
