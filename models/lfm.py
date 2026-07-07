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
            "Footage clips from CCTV will be shared. Camera is installed on the top, looking down. There's house door on the left side, which isn't visible due to camera angle. There are cars parked on the other end of the road. If there are any activity, describe that in details. Any animal entering or leaving the frame, mention that specifically, include the type of animal as well.\n" 
            "Any human entering or leaving the frame, mention their color of clothes and what they're carrying or holding, or what they are doing. If no activity is observed, just say \"no activity\". Previous frame captions are shared below for temporal context.\n"
            f"Previous caption: {previous_caption or 'none'}.\n"
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
                max_new_tokens=128,
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
