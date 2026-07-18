"""Shared base class and utilities for Vision-Language Model captioners."""

from __future__ import annotations

import abc

import torch


def ensure_torchvision() -> None:
    """Raise a helpful error if ``torchvision`` is not installed."""
    try:
        import torchvision  # noqa: F401
    except Exception as exc:  # pragma: no cover - helpful runtime error
        raise ImportError(
            "LFM2.5-VL requires the 'torchvision' package.\n"
            "Install it with a command matching your PyTorch install, for example:\n"
            "  pip install torchvision\n"
            "Or follow the official guide: https://pytorch.org/get-started/locally/\n"
        ) from exc


def select_torch_dtype() -> torch.dtype:
    """Pick the best available dtype for the current hardware."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


class BaseVideoCaptioner(abc.ABC):
    """Abstract base class that every VLM captioner must implement."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self._custom_system_prompt = system_prompt

    @property
    def needs_all_frames(self) -> bool:
        """Whether the captioner requires every video frame (not just sampled ones).

        Frame-by-frame models return ``False`` — the orchestrator skips frames
        based on ``caption_interval``.  Video-native models that buffer clips
        internally return ``True`` so they receive every frame from the source.
        """
        return False

    @abc.abstractmethod
    def caption_frame(self, frame_bgr, previous_caption: str) -> str:
        """Generate a caption for a single video frame.

        Args:
            frame_bgr: The video frame in BGR format (from OpenCV).
            previous_caption: The caption generated for the previous frame.

        Returns:
            The generated caption string.
        """
