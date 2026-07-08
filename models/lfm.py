"""LFM2.5-VL captioner implementation."""

from __future__ import annotations

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base import BaseVideoCaptioner, select_torch_dtype, ensure_torchvision


class LFM25VideoCaptioner(BaseVideoCaptioner):
    """Vision-Language captioner backed by Liquid AI's LFM2.5-VL."""

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
            "This is a frame from a CCTV footage. Identify any animals or humans in the frame. If any human is present, describe their appearance and action. If there's no animal or human, say \"No Activity\"\n" 
            # "Describe the scene."
            # f"Previous caption: {previous_caption or 'none'}.\n"
        )

        conversation = [
            {
                "role": "system",
                "content": "You are a helpful multimodal assistant by Liquid AI. You are brief and concise.",
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
                max_new_tokens=96,
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
