"""Video processing service — runs in a background thread, pushes captions via callback."""

from __future__ import annotations

import time
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import cv2

from models import create_captioner
from database import VectorStore


class VideoProcessor:
    """Manages a single video processing session."""

    def __init__(
        self,
        video_path: str,
        model_type: str,
        model_id: str,
        system_prompt: Optional[str],
        caption_interval: float,
        max_new_tokens: int,
        max_frames: Optional[int],
        collection_name: str,
        vector_db_path: str,
        fps: int,
        clip_duration: float,
        on_caption: Callable[[dict[str, Any]], None],
        on_status: Callable[[str], None],
        on_error: Callable[[str], None],
        on_done: Callable[[], None],
    ) -> None:
        self.video_path = video_path
        self.model_type = model_type
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.caption_interval = caption_interval
        self.max_new_tokens = max_new_tokens
        self.max_frames = max_frames
        self.collection_name = collection_name
        self.vector_db_path = vector_db_path
        self.fps = fps
        self.clip_duration = clip_duration
        self.on_caption = on_caption
        self.on_status = on_status
        self.on_error = on_error
        self.on_done = on_done

        self.video_id = str(uuid.uuid4())
        self.video_name = Path(video_path).stem
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.caption_count = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        try:
            captioner = create_captioner(
                self.model_type,
                self.model_id,
                fps=self.fps,
                clip_duration=self.clip_duration,
                system_prompt=self.system_prompt,
            )
        except Exception as e:
            self.on_error(f"Failed to load model: {e}")
            self.on_done()
            return

        store = None
        try:
            store = VectorStore(
                collection_name=self.collection_name,
                db_path=self.vector_db_path,
            )
        except Exception as e:
            self.on_status(f"Warning: Vector DB init failed: {e}")

        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            self.on_error("Could not open video file.")
            self.on_done()
            return

        previous_caption = ""
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = capture.get(cv2.CAP_PROP_FPS) or 30.0

        self.on_status(f"Processing started — {total_frames} frames, {fps_video:.1f} fps")

        next_caption_at = 0.0

        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break

                now = time.monotonic()

                if not captioner.needs_all_frames:
                    if now < next_caption_at:
                        time.sleep(min(0.02, next_caption_at - now))
                        continue

                caption = captioner.caption_frame(frame, previous_caption)
                caption = caption.replace("\n", " ").strip()

                if caption:
                    current_time_str = time.strftime("%H:%M:%S")
                    now_secs = time.time()
                    current_time_secs = now_secs - (now_secs // 86400 * 86400)

                    video_timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                    video_timestamp_str = (
                        f"{int(video_timestamp_ms // 60000):02d}:"
                        f"{int((video_timestamp_ms % 60000) // 1000):02d}:"
                        f"{int(video_timestamp_ms % 1000):03d}"
                    )

                    self.on_caption({
                        "time": current_time_str,
                        "caption": caption,
                        "frame": self.caption_count,
                        "video_ts": video_timestamp_str,
                    })

                    if store is not None:
                        try:
                            store.save_caption(
                                caption,
                                metadata={
                                    "video_id": self.video_id,
                                    "video_name": self.video_name,
                                    "model_type": self.model_type,
                                    "current_time": current_time_str,
                                    "current_time_secs": current_time_secs,
                                    "video_timestamp": video_timestamp_str,
                                    "video_timestamp_ms": video_timestamp_ms,
                                    "frame_index": self.caption_count,
                                    "source": self.video_path,
                                    "caption": caption,
                                },
                            )
                        except Exception:
                            pass

                    previous_caption = caption
                    self.caption_count += 1

                    if self.max_frames is not None and self.caption_count >= self.max_frames:
                        break

                if not captioner.needs_all_frames:
                    next_caption_at = time.monotonic() + self.caption_interval
        finally:
            capture.release()
            self.on_done()
