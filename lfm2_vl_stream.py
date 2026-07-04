from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import uuid

import cv2
import torch
from PIL import Image
from qdrant_client import QdrantClient
from transformers import AutoProcessor, AutoModelForImageTextToText


def ensure_torchvision() -> None:
    try:
        import torchvision  # noqa: F401
    except Exception as exc:  # pragma: no cover - helpful runtime error
        raise ImportError(
            "LFM2.5-VL requires the 'torchvision' package.\n"
            "Install it with a command matching your PyTorch install, for example:\n"
            "  pip install torchvision\n"
            "Or follow the official guide: https://pytorch.org/get-started/locally/\n"
        ) from exc


import abc

DEFAULT_LFM_MODEL_ID = "LiquidAI/LFM2.5-VL-450M"
DEFAULT_MINICPM_MODEL_ID = "openbmb/MiniCPM-V-4.6"


@dataclass(frozen=True)
class StreamConfig:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption a local video stream with a Vision-Language Model."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index, video file path, RTSP URL, or camera device string.",
    )
    parser.add_argument(
        "--model-type",
        choices=["lfm2.5", "minicpm-v"],
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
        help="Name of the Qdrant collection to store captions.",
    )
    parser.add_argument(
        "--disable-vector-db",
        action="store_true",
        help="Disable storing generated captions in the vector database.",
    )
    return parser.parse_args()


def resolve_source(source: str):
    if source.isdigit():
        return int(source)
    path = Path(source)
    if path.exists():
        return str(path)
    return source


def build_config(args: argparse.Namespace) -> StreamConfig:
    model_id = args.model_id
    if not model_id:
        if args.model_type == "minicpm-v":
            model_id = DEFAULT_MINICPM_MODEL_ID
        else:
            model_id = DEFAULT_LFM_MODEL_ID

    return StreamConfig(
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
    )


def select_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


class BaseVideoCaptioner(abc.ABC):
    @abc.abstractmethod
    def caption_frame(self, frame_bgr, previous_caption: str) -> str:
        """Generates a caption for a single frame.

        Args:
            frame_bgr: The video frame in BGR format (from OpenCV).
            previous_caption: The caption generated for the previous frame.

        Returns:
            The generated caption string.
        """
        pass


class LFM25VideoCaptioner(BaseVideoCaptioner):
    def __init__(self, model_id: str) -> None:
        ensure_torchvision()
        dtype = select_torch_dtype()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def caption_frame(self, frame_bgr, previous_caption: str) -> str:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        prompt = (
            "You are captioning a live video stream.\n"
            f"Previous caption: {previous_caption or 'none'}.\n"
            # "If the scene is unchanged, reply exactly: No significant change."
        )

        conversation = [
            {
                "role": "system",
                "content": "You are a helpful multimodal assistant by Liquid AI.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.1,
                top_k=50,
                top_p=0.1,
                repetition_penalty=1.05,
            )

        generated_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        caption = self.processor.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0].strip()
        return caption


class MiniCPMVVideoCaptioner(BaseVideoCaptioner):
    def __init__(self, model_id: str) -> None:
        dtype = select_torch_dtype()
        from transformers import AutoModelForImageTextToText
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def caption_frame(self, frame_bgr, previous_caption: str) -> str:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        prompt = (
            "You are captioning a live video stream.\n"
            f"Previous caption: {previous_caption or 'none'}.\n"
            # "If the scene is unchanged, reply exactly: No significant change."
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            downsample_mode="16x"
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=48,
                temperature=0.1,
                top_k=50,
                top_p=0.1,
                repetition_penalty=1.05,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[:, input_len:]
        caption = self.processor.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0].strip()
        return caption


def create_captioner(model_type: str, model_id: str) -> BaseVideoCaptioner:
    if model_type == "minicpm-v" or "minicpm" in model_id.lower():
        return MiniCPMVVideoCaptioner(model_id)
    return LFM25VideoCaptioner(model_id)


def run_stream(config: StreamConfig) -> None:
    capture = cv2.VideoCapture(resolve_source(config.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {config.source}")

    captioner = create_captioner(config.model_type, config.model_id)
    previous_caption = ""
    caption_count = 0
    next_caption_at = 0.0

    qdrant_client = None
    if not config.disable_vector_db:
        try:
            if config.qdrant_url:
                print(f"Connecting to Qdrant server at {config.qdrant_url}...")
                qdrant_client = QdrantClient(url=config.qdrant_url)
            else:
                db_path = Path(config.vector_db_path).resolve()
                print(f"Initializing local Qdrant database at {db_path}...")
                qdrant_client = QdrantClient(path=str(db_path))
            print(f"Using vector database collection: '{config.collection_name}'")
        except Exception as e:
            print(f"Warning: Failed to initialize Qdrant client ({e}). Proceeding without vector database storage.")
            qdrant_client = None

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
                
                if qdrant_client is not None:
                    try:
                        caption_id = str(uuid.uuid4())
                        qdrant_client.add(
                            collection_name=config.collection_name,
                            documents=[caption],
                            metadata=[{
                                "timestamp": timestamp,
                                "frame_index": caption_count,
                                "source": str(config.source)
                            }],
                            ids=[caption_id]
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


def main() -> None:
    args = parse_args()
    config = build_config(args)
    run_stream(config)


if __name__ == "__main__":
    main()