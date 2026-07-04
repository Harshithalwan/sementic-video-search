"""Vision-Language Model captioners.

Usage::

    from models import create_captioner

    captioner = create_captioner("lfm2.5", "LiquidAI/LFM2.5-VL-450M")
    caption = captioner.caption_frame(frame_bgr, previous_caption="")
"""

from .base import BaseVideoCaptioner
from .lfm import LFM25VideoCaptioner
from .minicpm import MiniCPMVVideoCaptioner

DEFAULT_LFM_MODEL_ID = "LiquidAI/LFM2.5-VL-450M"
DEFAULT_MINICPM_MODEL_ID = "openbmb/MiniCPM-V-4.6"

# Registry mapping model-type names to their classes.
MODEL_REGISTRY: dict[str, type[BaseVideoCaptioner]] = {
    "lfm2.5": LFM25VideoCaptioner,
    "minicpm-v": MiniCPMVVideoCaptioner,
}

# Default model IDs keyed by model-type name.
DEFAULT_MODEL_IDS: dict[str, str] = {
    "lfm2.5": DEFAULT_LFM_MODEL_ID,
    "minicpm-v": DEFAULT_MINICPM_MODEL_ID,
}


def create_captioner(model_type: str, model_id: str) -> BaseVideoCaptioner:
    """Instantiate the appropriate captioner for *model_type*.

    When adding a new model, register it in ``MODEL_REGISTRY`` and
    ``DEFAULT_MODEL_IDS`` above.
    """
    cls = MODEL_REGISTRY.get(model_type)
    if cls is None:
        # Fallback: try matching on model_id substring.
        if "minicpm" in model_id.lower():
            cls = MiniCPMVVideoCaptioner
        else:
            cls = LFM25VideoCaptioner
    return cls(model_id)


__all__ = [
    "BaseVideoCaptioner",
    "LFM25VideoCaptioner",
    "MiniCPMVVideoCaptioner",
    "create_captioner",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL_IDS",
    "DEFAULT_LFM_MODEL_ID",
    "DEFAULT_MINICPM_MODEL_ID",
]
