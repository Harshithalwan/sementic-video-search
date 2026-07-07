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
            "You are captioning a documentry video. If there are any different human or animal in the frame, identify them and describe their actions. For human mention what they are wearing and what color is it. Also describe the prominent background features to give a hint of the environment.\n"
            f"Previous caption: {previous_caption or 'none'}.\n"
            "Be concise. Do not repeat the previous caption."
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
