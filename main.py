"""Orchestrator — CLI entry-point for stream captioning and database querying."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from models import create_captioner, DEFAULT_MODEL_IDS, MODEL_REGISTRY
from database import VectorStore


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
    # query-mode fields
    query: Optional[str] = None
    top_k: int = 5


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
        default=2.0,
        help="Seconds to wait between model calls so the stream does not overload the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
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
        default="captions",
        help="Name of the Qdrant collection to store/search captions.",
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

    return parser.parse_args()


def resolve_source(source: str):
    """Convert a source string to an int (webcam) or validated path."""
    if source.isdigit():
        return int(source)
    path = Path(source)
    if path.exists():
        return str(path)
    return source


def build_config(args: argparse.Namespace) -> StreamConfig:
    model_id = args.model_id or DEFAULT_MODEL_IDS.get(args.model_type, args.model_type)

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
        collection_name=args.collection_name,
        disable_vector_db=args.disable_vector_db,
        query=args.query,
        top_k=args.top_k,
    )


# ── Stream mode ──────────────────────────────────────────────────────

def run_stream(config: StreamConfig) -> None:
    """Open a video source, caption frames, and optionally persist to the vector DB."""
    capture = cv2.VideoCapture(resolve_source(config.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {config.source}")

    captioner = create_captioner(config.model_type, config.model_id)
    previous_caption = ""
    caption_count = 0
    next_caption_at = 0.0

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
            if now < next_caption_at:
                time.sleep(min(0.02, next_caption_at - now))
                continue

            caption = captioner.caption_frame(frame, previous_caption)
            caption = caption.replace("\n", " ").strip()
            if caption:
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] {caption}", flush=True)

                if store is not None:
                    try:
                        store.save_caption(
                            caption,
                            metadata={
                                "timestamp": timestamp,
                                "frame_index": caption_count,
                                "source": str(config.source),
                            },
                        )
                    except Exception as e:
                        print(f"Warning: Failed to save caption to vector database: {e}")

                previous_caption = caption
                caption_count += 1
                if config.max_frames is not None and caption_count >= config.max_frames:
                    break

            next_caption_at = time.monotonic() + config.caption_interval
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if config.show_preview:
            cv2.destroyAllWindows()


# ── Query mode ───────────────────────────────────────────────────────

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

    print(f"Searching for: '{config.query}' (top {config.top_k} matches)...")

    try:
        results = store.query(config.query, top_k=config.top_k)
    except Exception as e:
        print(f"Error executing search query: {e}")
        sys.exit(1)

    if not results:
        print("No matching captions found.")
        return

    print("\n" + "=" * 80)
    print(f"{'Score':<8} | {'Timestamp':<10} | {'Frame':<7} | {'Caption'}")
    print("-" * 80)
    for r in results:
        payload = r.metadata or {}
        timestamp = payload.get("timestamp", "N/A")
        frame_idx = payload.get("frame_index", "N/A")
        caption = r.document or "No text"
        score = f"{r.score:.4f}"
        print(f"{score:<8} | {timestamp:<10} | {frame_idx:<7} | {caption}")
    print("=" * 80 + "\n")


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
