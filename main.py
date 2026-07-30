"""Orchestrator — CLI entry-point for stream captioning and database querying."""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from models import create_captioner, DEFAULT_MODEL_IDS, MODEL_REGISTRY, COLLECTION_NAMES
from database import VectorStore
from config import pipeline_config
from detectors import ActivityDetector, ObjectDetector


# ── Configuration ────────────────────────────────────────────────────

@dataclass(frozen=True)
class StreamConfig:
    mode: str
    source: str
    model_type: str
    model_id: str
    caption_interval: float
    max_new_tokens: int
    max_frames: Optional[int]
    show_preview: bool
    vector_db_path: str
    qdrant_url: Optional[str]
    collection_name: str
    disable_vector_db: bool
    # video-native model options
    fps: int = 20
    clip_duration: float = 3.0
    # query-mode fields
    query: Optional[str] = None
    top_k: int = 5
    filter_video_id: Optional[str] = None
    filter_video_name: Optional[str] = None
    filter_ts_from: Optional[float] = None
    filter_ts_to: Optional[float] = None
    filter_time_from: Optional[float] = None
    filter_time_to: Optional[float] = None
    # detection options
    enable_activity_detection: bool = False
    activity_threshold: float = 0.85
    enable_yolo: bool = False
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.5


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption a live video stream with a Vision-Language Model, "
        "or query previously stored captions.",
    )

    # Mode ----------------------------------------------------------
    parser.add_argument(
        "--mode",
        choices=["stream", "query"],
        default="stream",
        help="Operating mode: 'stream' for live captioning, 'query' to search stored captions.",
    )

    # Stream options ------------------------------------------------
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index, video file path, RTSP URL, or camera device string.",
    )
    parser.add_argument(
        "--model-type",
        choices=list(MODEL_REGISTRY.keys()),
        default="lfm2.5",
        help="The family/type of Vision-Language Model to use.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model id to load. If omitted, defaults based on --model-type.",
    )
    parser.add_argument(
        "--caption-interval",
        type=float,
        default=1.0,
        help="Seconds to wait between model calls so the stream does not overload the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Maximum tokens generated per caption.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many model-generated captions.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show a live preview window while captions are generated.",
    )

    # Video-native model options -----------------------------------
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Target sampling FPS for video-native models (e.g. llava-video).",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=3.0,
        help="Seconds of video per clip for video-native models.",
    )

    # Database options (shared) -------------------------------------
    parser.add_argument(
        "--vector-db-path",
        default="./video_captions_db",
        help="Local directory to persist the Qdrant database.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=None,
        help="Optional URL of a running Qdrant server (e.g. http://localhost:6333).",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Qdrant collection name. Defaults to a model-specific name (e.g. captions_lfm2.5).",
    )
    parser.add_argument(
        "--disable-vector-db",
        action="store_true",
        help="Disable storing generated captions in the vector database (stream mode only).",
    )

    # Query options -------------------------------------------------
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="The search query (required in query mode).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to return (query mode).",
    )

    # Query filters (all optional) ----------------------------------
    parser.add_argument(
        "--filter-video-id",
        type=str,
        default=None,
        help="Only return captions with this video ID.",
    )
    parser.add_argument(
        "--filter-video-name",
        type=str,
        default=None,
        help="Only return captions with this video name.",
    )
    parser.add_argument(
        "--filter-ts-from",
        type=float,
        default=None,
        help="Only return captions at or after this video timestamp (milliseconds).",
    )
    parser.add_argument(
        "--filter-ts-to",
        type=float,
        default=None,
        help="Only return captions at or before this video timestamp (milliseconds).",
    )
    parser.add_argument(
        "--filter-time-from",
        type=float,
        default=None,
        help="Only return captions at or after this wall-clock time (seconds since midnight).",
    )
    parser.add_argument(
        "--filter-time-to",
        type=float,
        default=None,
        help="Only return captions at or before this wall-clock time (seconds since midnight).",
    )

    # Detection options ------------------------------------------------
    parser.add_argument(
        "--enable-activity-detection",
        action="store_true",
        help="Enable SSIM-based activity detection. Frames with no visual change are skipped.",
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=0.85,
        help="SSIM threshold for activity detection (default 0.85). Lower = more sensitive.",
    )
    parser.add_argument(
        "--enable-yolo",
        action="store_true",
        help="Enable YOLO object detection. Detected objects are saved alongside captions.",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.5,
        help="YOLO confidence threshold (default: 0.5).",
    )

    return parser.parse_args()


def resolve_source(source: str):
    """Convert a source string to an int (webcam) or validated path."""
    if source.isdigit():
        return int(source)
    path = Path(source)
    if path.exists():
        return str(path.resolve())
    return source


def build_config(args: argparse.Namespace) -> StreamConfig:
    model_id = args.model_id or DEFAULT_MODEL_IDS.get(args.model_type, args.model_type)
    collection_name = args.collection_name or COLLECTION_NAMES.get(
        args.model_type, f"captions_{args.model_type}"
    )

    return StreamConfig(
        mode=args.mode,
        source=args.source,
        model_type=args.model_type,
        model_id=model_id,
        caption_interval=max(0.25, args.caption_interval),
        max_new_tokens=max(8, args.max_new_tokens),
        max_frames=args.max_frames,
        show_preview=args.show_preview,
        vector_db_path=args.vector_db_path,
        qdrant_url=args.qdrant_url,
        collection_name=collection_name,
        disable_vector_db=args.disable_vector_db,
        fps=args.fps,
        clip_duration=args.clip_duration,
        query=args.query,
        top_k=args.top_k,
        filter_video_id=args.filter_video_id,
        filter_video_name=args.filter_video_name,
        filter_ts_from=args.filter_ts_from,
        filter_ts_to=args.filter_ts_to,
        filter_time_from=args.filter_time_from,
        filter_time_to=args.filter_time_to,
        enable_activity_detection=args.enable_activity_detection,
        activity_threshold=args.activity_threshold,
        enable_yolo=args.enable_yolo,
        yolo_model=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
    )


# ── Stream mode ──────────────────────────────────────────────────────

def _derive_video_name(source: str) -> str:
    """Return a human-readable video name from the source string."""
    if source.isdigit():
        return f"webcam_{source}"
    path = Path(source)
    if path.exists():
        return path.name
    return source.replace("://", "_").replace("/", "_").replace(":", "_")


def run_stream(config: StreamConfig) -> None:
    """Open a video source, caption frames, and optionally persist to the vector DB."""
    capture = cv2.VideoCapture(resolve_source(config.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {config.source}")

    video_id = str(uuid.uuid4())
    video_name = _derive_video_name(config.source)

    captioner = create_captioner(
        config.model_type,
        config.model_id,
        fps=config.fps,
        clip_duration=config.clip_duration,
        max_new_tokens=config.max_new_tokens,
    )
    previous_caption = ""
    caption_count = 0
    next_caption_at = 0.0

    # Initialise detectors
    activity_detector = None
    if config.enable_activity_detection:
        pipeline_config.activity_detection.enabled = True
        pipeline_config.activity_detection.threshold = config.activity_threshold
        activity_detector = ActivityDetector(pipeline_config.activity_detection)
        print(f"Activity detection enabled (SSIM threshold: {config.activity_threshold})")

    object_detector = None
    if config.enable_yolo:
        pipeline_config.yolo.enabled = True
        pipeline_config.yolo.model_path = config.yolo_model
        pipeline_config.yolo.confidence_threshold = config.yolo_confidence
        object_detector = ObjectDetector(pipeline_config.yolo)
        print(f"YOLO detection enabled (model: {config.yolo_model})")

    # Initialise vector store (if enabled)
    store: Optional[VectorStore] = None
    if not config.disable_vector_db:
        try:
            store = VectorStore(
                collection_name=config.collection_name,
                db_path=config.vector_db_path,
                qdrant_url=config.qdrant_url,
            )
        except Exception as e:
            print(
                f"Warning: Failed to initialize vector store ({e}). "
                "Proceeding without database storage."
            )

    print(f"Loaded {config.model_id} ({config.model_type})")
    print(f"Capturing from {config.source}")
    print(f"Video ID: {video_id}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if config.show_preview:
                cv2.imshow("Video Stream Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.monotonic()

            # Frame-by-frame models: skip frames based on caption_interval.
            # Video-native models: pass every frame to the captioner for buffering.
            if not captioner.needs_all_frames:
                if now < next_caption_at:
                    time.sleep(min(0.02, next_caption_at - now))
                    continue

            # Activity detection gate
            if activity_detector and not activity_detector.is_active(frame):
                if not captioner.needs_all_frames:
                    next_caption_at = time.monotonic() + config.caption_interval
                continue

            # YOLO detection (runs on all active frames)
            yolo_classes = []
            if object_detector:
                yolo_classes, yolo_ms = object_detector.detect(frame)

            caption = captioner.caption_frame(frame, previous_caption)
            caption = caption.replace("\n", " ").strip()
            if caption:
                current_time_str = time.strftime("%H:%M:%S")
                now_secs = time.time()
                current_time_secs = now_secs - (now_secs // 86400 * 86400)

                video_timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                video_timestamp_str = f"{int(video_timestamp_ms // 60000):02d}:{int((video_timestamp_ms % 60000) // 1000):02d}:{int(video_timestamp_ms % 1000):03d}"
                yolo_str = f" | Objects: {', '.join(yolo_classes)}" if yolo_classes else ""
                print(f"[{current_time_str}] {caption}{yolo_str}", flush=True)

                if store is not None:
                    try:
                        store.save_caption(
                            caption,
                            metadata={
                                "video_id": video_id,
                                "video_name": video_name,
                                "model_type": config.model_type,
                                "current_time": current_time_str,
                                "current_time_secs": current_time_secs,
                                "video_timestamp": video_timestamp_str,
                                "video_timestamp_ms": video_timestamp_ms,
                                "frame_index": caption_count,
                                "source": str(Path(config.source).resolve()) if not str(config.source).isdigit() else str(config.source),
                                "caption": caption,
                                "yolo_objects": yolo_classes,
                            },
                        )
                    except Exception as e:
                        print(f"Warning: Failed to save caption to vector database: {e}")

                previous_caption = caption
                caption_count += 1
                if config.max_frames is not None and caption_count >= config.max_frames:
                    break

            if not captioner.needs_all_frames:
                next_caption_at = time.monotonic() + config.caption_interval
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if config.show_preview:
            cv2.destroyAllWindows()


# ── Query mode ───────────────────────────────────────────────────────

def _build_query_filter(config: StreamConfig) -> Optional[Filter]:
    """Construct a Qdrant Filter from optional CLI filter flags."""
    conditions: list[FieldCondition] = []

    if config.filter_video_id is not None:
        conditions.append(
            FieldCondition(key="video_id", match=MatchValue(value=config.filter_video_id))
        )
    if config.filter_video_name is not None:
        conditions.append(
            FieldCondition(key="video_name", match=MatchValue(value=config.filter_video_name))
        )

    ts_from = config.filter_ts_from
    ts_to = config.filter_ts_to
    if ts_from is not None or ts_to is not None:
        conditions.append(
            FieldCondition(
                key="video_timestamp_ms",
                range=Range(
                    gte=ts_from,
                    lte=ts_to,
                ),
            )
        )

    time_from = config.filter_time_from
    time_to = config.filter_time_to
    if time_from is not None or time_to is not None:
        conditions.append(
            FieldCondition(
                key="current_time_secs",
                range=Range(
                    gte=time_from,
                    lte=time_to,
                ),
            )
        )

    if not conditions:
        return None
    return Filter(must=conditions)


def run_query(config: StreamConfig) -> None:
    """Search the vector database for matching captions."""
    if not config.query:
        print("Error: --query is required in query mode.")
        sys.exit(1)

    try:
        store = VectorStore(
            collection_name=config.collection_name,
            db_path=config.vector_db_path,
            qdrant_url=config.qdrant_url,
        )
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        sys.exit(1)

    store.ensure_collection_exists()

    query_filter = _build_query_filter(config)
    if query_filter:
        parts = []
        for c in query_filter.must:
            if hasattr(c, "match"):
                parts.append(f"{c.key}={c.match.value}")
            elif hasattr(c, "range"):
                r = c.range
                lo = f">={r.gte}" if r.gte is not None else ""
                hi = f"<={r.lte}" if r.lte is not None else ""
                parts.append(f"{c.key} [{lo} {hi}]")
        print(f"Filters: {', '.join(parts)}")

    print(f"Searching for: '{config.query}' (top {config.top_k} matches)...")

    try:
        results = store.query(config.query, top_k=config.top_k, query_filter=query_filter)
    except Exception as e:
        print(f"Error executing search query: {e}")
        sys.exit(1)

    if not results:
        print("No matching captions found.")
        return

    print("\n" + "=" * 180)
    print(f"{'Score':<8} | {'Model':<12} | {'Video ID':<36} | {'Video Name':<20} | {'Vid TS':<14} | {'Time':<10} | {'Frame':<7} | {'Objects':<20} | {'Caption'}")
    print("-" * 180)
    for r in results:
        payload = r.metadata or {}
        model_type = payload.get("model_type", "N/A")
        video_id = payload.get("video_id", "N/A")
        video_name = payload.get("video_name", "N/A")
        video_ts = payload.get("video_timestamp", "N/A")
        current_time = payload.get("current_time", "N/A")
        frame_idx = payload.get("frame_index", "N/A")
        yolo_objects = payload.get("yolo_objects", [])
        objects_str = ", ".join(yolo_objects) if yolo_objects else "-"
        caption = r.document or "No text"
        score = f"{r.score:.4f}"
        print(f"{score:<8} | {model_type:<12} | {video_id:<36} | {video_name:<20} | {video_ts:<14} | {current_time:<10} | {frame_idx:<7} | {objects_str:<20} | {caption}")
    print("=" * 180 + "\n")


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config = build_config(args)

    if config.mode == "query":
        run_query(config)
    else:
        run_stream(config)


if __name__ == "__main__":
    main()
