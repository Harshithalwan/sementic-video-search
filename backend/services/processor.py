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
from config import ActivityDetectionConfig, YOLOConfig
from detectors import ActivityDetector, ObjectDetector
from loggers.latency_logger import LatencyLogger


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
        on_frame: Optional[Callable[[bytes], None]] = None,
        activity_detection_enabled: bool = False,
        activity_detection_threshold: float = 0.85,
        yolo_enabled: bool = False,
        yolo_model: str = "yolov8n.pt",
        yolo_confidence: float = 0.5,
        yolo_tracking: bool = True,
        stream_width: int = 960,
        jpeg_quality: int = 80,
        latency_logging_enabled: bool = False,
    ) -> None:
        self.video_path = str(Path(video_path).resolve())
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
        self.on_frame = on_frame or (lambda _: None)
        self.activity_detection_enabled = activity_detection_enabled
        self.activity_detection_threshold = activity_detection_threshold
        self.yolo_enabled = yolo_enabled
        self.yolo_model = yolo_model
        self.yolo_confidence = yolo_confidence
        self.yolo_tracking = yolo_tracking
        self.stream_width = stream_width
        self.jpeg_quality = jpeg_quality
        self.latency_logging_enabled = latency_logging_enabled

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
                max_new_tokens=self.max_new_tokens,
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

        # Initialise detectors
        activity_detector = None
        if self.activity_detection_enabled:
            activity_detector = ActivityDetector(
                ActivityDetectionConfig(
                    enabled=True,
                    threshold=self.activity_detection_threshold,
                )
            )
            self.on_status(f"Activity detection enabled (threshold: {self.activity_detection_threshold})")

        object_detector = None
        if self.yolo_enabled:
            object_detector = ObjectDetector(
                YOLOConfig(
                    enabled=True,
                    model_path=self.yolo_model,
                    confidence_threshold=self.yolo_confidence,
                    tracking_enabled=self.yolo_tracking,
                )
            )
            tracking_note = " + tracking" if self.yolo_tracking else ""
            self.on_status(f"YOLO detection enabled (model: {self.yolo_model}{tracking_note})")

        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            self.on_error("Could not open video file.")
            self.on_done()
            return

        previous_caption = ""
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = capture.get(cv2.CAP_PROP_FPS) or 30.0

        self.on_status(f"Processing started — {total_frames} frames, {fps_video:.1f} fps")

        logger = LatencyLogger(
            model_type=self.model_type,
            model_id=self.model_id,
        ) if self.latency_logging_enabled else None

        next_caption_at = 0.0

        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break

                now = time.monotonic()

                # Stream every frame to the client, annotating with YOLO boxes
                # (and tracking trails/arrows when tracking is enabled). The feed
                # is paced by this loop: when caption generation takes longer,
                # frames wait.
                yolo_classes = []
                yolo_tracks = []
                frame_to_stream = frame
                if object_detector:
                    if self.yolo_tracking:
                        frame_to_stream, yolo_tracks, yolo_ms = object_detector.track(frame)
                        yolo_classes = [t["class"] for t in yolo_tracks]
                    else:
                        frame_to_stream, yolo_classes, yolo_ms = object_detector.annotate(frame)
                    if logger:
                        logger.log_yolo(yolo_ms, yolo_classes, self.caption_count)
                self._send_frame(frame_to_stream)

                # Caption gating: frame-by-frame models only caption every
                # caption_interval seconds. The video feed itself is unaffected.
                if not captioner.needs_all_frames:
                    if now < next_caption_at:
                        time.sleep(min(0.02, next_caption_at - now))
                        continue

                # Activity detection gate (gates captioning only)
                if activity_detector:
                    ssim_active, ssim_score, ssim_ms = activity_detector.is_active(frame)
                    if logger:
                        logger.log_ssim(ssim_ms, ssim_score, ssim_active, self.caption_count, self.model_type)
                    if not ssim_active:
                        if not captioner.needs_all_frames:
                            next_caption_at = time.monotonic() + self.caption_interval
                        continue

                t_cap = time.perf_counter()
                caption = captioner.caption_frame(frame, previous_caption)
                caption_ms = (time.perf_counter() - t_cap) * 1000
                caption = caption.replace("\n", " ").strip()

                if caption:
                    if logger:
                        logger.log_caption(caption_ms, caption, self.model_type, self.model_id, self.caption_count)
                    current_time_str = time.strftime("%H:%M:%S")
                    now_secs = time.time()
                    current_time_secs = now_secs - (now_secs // 86400 * 86400)

                    video_timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                    video_timestamp_str = (
                        f"{int(video_timestamp_ms // 60000):02d}:"
                        f"{int((video_timestamp_ms % 60000) // 1000):02d}:"
                        f"{int(video_timestamp_ms % 1000):03d}"
                    )

                    movement_summary = self._movement_summary(yolo_tracks)
                    enriched_caption = caption
                    if movement_summary:
                        enriched_caption = f"{caption} | {movement_summary}"

                    self.on_caption({
                        "time": current_time_str,
                        "caption": caption,
                        "frame": self.caption_count,
                        "video_ts": video_timestamp_str,
                        "yolo_objects": yolo_classes,
                        "yolo_tracks": yolo_tracks,
                        "movement_summary": movement_summary,
                    })

                    if store is not None:
                        try:
                            store.save_caption(
                                enriched_caption,
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
                                    "enriched_caption": enriched_caption,
                                    "yolo_objects": yolo_classes,
                                    "yolo_tracks": yolo_tracks,
                                    "movement_summary": movement_summary,
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
            if logger:
                logger.close()
            if store is not None:
                store.close()
            self.on_done()

    @staticmethod
    def _movement_summary(tracks: list[dict]) -> str:
        """Build a compact human-readable summary of tracked-object movement."""
        parts = []
        for t in tracks:
            tid = t.get("track_id")
            if tid is None:
                continue
            name = t.get("class", "object")
            direction = t.get("direction", "unknown")
            speed_pct = t.get("speed", 0.0) * 100
            parts.append(f"{name}#{tid} {direction} ({speed_pct:.1f}%/s)")
        return "; ".join(parts)

    def _send_frame(self, frame) -> None:
        """Downscale and JPEG-encode a frame, then push it via the callback."""
        try:
            if frame.shape[1] > self.stream_width:
                scale = self.stream_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (self.stream_width, int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buf = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if ok:
                self.on_frame(buf.tobytes())
        except Exception:
            pass
