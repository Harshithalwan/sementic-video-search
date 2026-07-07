"""Vision-Language Model captioners.

Usage::

    from models import create_captioner

    captioner = create_captioner("lfm2.5", "LiquidAI/LFM2.5-VL-450M")
    caption = captioner.caption_frame(frame_bgr, previous_caption="")
"""

from .base import BaseVideoCaptioner
from .lfm import LFM25VideoCaptioner
from .minicpm import MiniCPMVVideoCaptioner
from .llavavideo import LLaVAVideoCaptioner

DEFAULT_LFM_MODEL_ID = "LiquidAI/LFM2.5-VL-450M"
LFM_MODEL_LARGER = "LiquidAI/LFM2.5-VL-1.8B"
DEFAULT_MINICPM_MODEL_ID = "openbmb/MiniCPM-V-4.6"
DEFAULT_LLAVA_VIDEO_MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"

# Registry mapping model-type names to their classes.
MODEL_REGISTRY: dict[str, type[BaseVideoCaptioner]] = {
    "lfm2.5": LFM25VideoCaptioner,
    "minicpm-v": MiniCPMVVideoCaptioner,
    "lfm2.5_larger": LFM25VideoCaptioner,
    "llava-video": LLaVAVideoCaptioner,
}

# Default model IDs keyed by model-type name.
DEFAULT_MODEL_IDS: dict[str, str] = {
    "lfm2.5": DEFAULT_LFM_MODEL_ID,
    "minicpm-v": DEFAULT_MINICPM_MODEL_ID,
    "lfm2.5_larger": LFM_MODEL_LARGER,
    "llava-video": DEFAULT_LLAVA_VIDEO_MODEL_ID,
}

# Collection name per model type.
# When adding a new model, add its entry here so captions are stored in a separate collection.
COLLECTION_NAMES: dict[str, str] = {
    "lfm2.5": "captions_lfm2.5",
    "minicpm-v": "captions_minicpm_v",
    "lfm2.5_larger": "captions_lfm2.5_larger",
    "llava-video": "captions_llava_video",
}

def create_captioner(model_type: str, model_id: str, **kwargs) -> BaseVideoCaptioner:
    """Instantiate the appropriate captioner for *model_type*.

    When adding a new model, register it in ``MODEL_REGISTRY`` and
    ``DEFAULT_MODEL_IDS`` above.

    Additional keyword arguments (e.g. ``fps``, ``clip_duration``) are
    forwarded only to captioners whose constructor accepts them.
    """
    cls = MODEL_REGISTRY.get(model_type)
    if cls is None:
        # Fallback: try matching on model_id substring.
        if "minicpm" in model_id.lower():
            cls = MiniCPMVVideoCaptioner
        elif "llava" in model_id.lower():
            cls = LLaVAVideoCaptioner
        elif "lfm2.5_larger" in model_id.lower():
            cls = LFM25VideoCaptioner
        else:
            cls = LFM25VideoCaptioner
    if cls is LLaVAVideoCaptioner:
        return cls(model_id, **kwargs)
    return cls(model_id)


__all__ = [
    "BaseVideoCaptioner",
    "LFM25VideoCaptioner",
    "MiniCPMVVideoCaptioner",
    "LLaVAVideoCaptioner",
    "create_captioner",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL_IDS",
    "COLLECTION_NAMES",
    "DEFAULT_LFM_MODEL_ID",
    "DEFAULT_MINICPM_MODEL_ID",
    "LFM_MODEL_LARGER",
    "DEFAULT_LLAVA_VIDEO_MODEL_ID",
]
