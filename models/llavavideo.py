"""LLaVA-NeXT-Video captioner with native video clip processing."""

from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

from .base import BaseVideoCaptioner, select_torch_dtype


_DEFAULT_PROMPT = (
    "You are captioning a CCTV video. Camera is installed on the top, looking down. "
    "There's house door on the left side, which isn't visible due to camera angle. "
    "There are cars parked on the other end of the road."
    "Describe if there's any change in the scene, any humans or animals present, "
    "their actions and attire, what they are carrying or holding, etc."
    "and prominent background features.\n"
)


class LLaVAVideoCaptioner(BaseVideoCaptioner):
    """Vision-Language captioner backed by ``llava-hf/LLaVA-NeXT-Video-7B-hf``.

    Buffers frames into short video clips at a configurable FPS and clip
    duration, then passes the entire clip to the model for a single caption.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier.
    fps:
        Target frame rate for sampling frames from the incoming stream.
    clip_duration:
        Seconds of video to accumulate before generating a caption.
    max_new_tokens:
        Maximum tokens to generate per caption.
    """

    def __init__(
        self,
        model_id: str,
        fps: int = 2,
        clip_duration: float = 4.0,
        max_new_tokens: int = 128,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(system_prompt=system_prompt)
        self._default_prompt = _DEFAULT_PROMPT
        self.fps = fps
        self.clip_duration = clip_duration
        self.target_frames = max(1, int(round(fps * clip_duration)))
        self.max_new_tokens = max_new_tokens

        self._buffer: list = []
        self._last_sample_time: float = 0.0

        dtype = select_torch_dtype()
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            dtype=dtype,
        )
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_id)

    @property
    def needs_all_frames(self) -> bool:
        return True

    def caption_frame(self, frame_bgr, previous_caption: str) -> str:
        """Buffer the frame; return a caption only when a full clip is ready."""
        now = time.monotonic()

        if self._last_sample_time == 0.0:
            self._last_sample_time = now

        frame_interval = 1.0 / self.fps
        if now - self._last_sample_time >= frame_interval:
            self._buffer.append(frame_bgr.copy())
            self._last_sample_time = now

        if len(self._buffer) < self.target_frames:
            return ""

        caption = self._caption_clip(previous_caption)
        self._buffer.clear()
        return caption

    def _caption_clip(self, previous_caption: str) -> str:
        """Generate a single caption from the accumulated clip frames."""
        frame_count = len(self._buffer)

        # Convert buffered BGR frames to RGB numpy array (num_frames, H, W, 3)
        frames_rgb = np.stack([
            cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in self._buffer
        ])

        prompt = (self._custom_system_prompt or self._default_prompt) + (
            f"This clip contains {frame_count} frames sampled at {self.fps} fps.\n"
        )

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video"},
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt_text,
            videos=frames_rgb,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                top_k=50,
                top_p=0.1,
                repetition_penalty=1.05,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        caption = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
        return caption
