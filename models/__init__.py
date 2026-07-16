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

# Default system prompts keyed by model-type name.
# These are used when no custom system prompt is provided.
DEFAULT_SYSTEM_PROMPTS: dict[str, str | None] = {
    "lfm2.5": "You are a helpful multimodal assistant by Liquid AI. You are brief and concise.",
    "lfm2.5_larger": "You are a helpful multimodal assistant by Liquid AI. You are brief and concise.",
    "minicpm-v": (
        'This is a frame from a CCTV footage. Identify any animals or humans in the frame. '
        'If any human is present, describe their appearance and action. '
        'If there\'s no animal or human, say "No Activity"'
    ),
    "llava-video": (
        "You are captioning a CCTV video. Camera is installed on the top, looking down. "
        "There's house door on the left side, which isn't visible due to camera angle. "
        "There are cars parked on the other end of the road."
        "Describe if there's any change in the scene, any humans or animals present, "
        "their actions and attire, what they are carrying or holding, etc."
        "and prominent background features."
    ),
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

    Additional keyword arguments (e.g. ``fps``, ``clip_duration``,
    ``system_prompt``) are forwarded to the captioner constructor.
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

    # Frame-by-frame models only accept system_prompt, not fps/clip_duration.
    filtered = {k: v for k, v in kwargs.items() if k == "system_prompt"}
    return cls(model_id, **filtered)


__all__ = [
    "BaseVideoCaptioner",
    "LFM25VideoCaptioner",
    "MiniCPMVVideoCaptioner",
    "LLaVAVideoCaptioner",
    "create_captioner",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL_IDS",
    "DEFAULT_SYSTEM_PROMPTS",
    "COLLECTION_NAMES",
    "DEFAULT_LFM_MODEL_ID",
    "DEFAULT_MINICPM_MODEL_ID",
    "LFM_MODEL_LARGER",
    "DEFAULT_LLAVA_VIDEO_MODEL_ID",
]
