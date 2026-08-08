"""MiniCPM-V captioner implementation."""

from __future__ import annotations

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base import BaseVideoCaptioner, select_torch_dtype, select_torch_device


_DEFAULT_PROMPT = (
    'This is a frame from a CCTV footage. Identify any animals or humans in the frame. '
    'If any human is present, describe their appearance and action. '
    'If there\'s no animal or human, say "No Activity"'
)


class MiniCPMVVideoCaptioner(BaseVideoCaptioner):
    """Vision-Language captioner backed by OpenBMB MiniCPM-V."""

    def __init__(self, model_id: str, system_prompt: str | None = None, max_new_tokens: int = 128) -> None:
        super().__init__(system_prompt=system_prompt)
        self.max_new_tokens = max_new_tokens
        self._default_prompt = _DEFAULT_PROMPT
        dtype = select_torch_dtype()
        device = select_torch_device()
        if torch.cuda.is_available():
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=dtype,
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def caption_frame(self, frame_bgr, previous_caption: str) -> str:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        prompt = self._custom_system_prompt or self._default_prompt

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
            downsample_mode="16x",
        )
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

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
        generated_tokens = output_ids[:, input_len:]
        caption = self.processor.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0].strip()
        return caption
