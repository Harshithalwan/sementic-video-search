"""MiniCPM-V captioner implementation."""

from __future__ import annotations

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base import BaseVideoCaptioner, select_torch_dtype


class MiniCPMVVideoCaptioner(BaseVideoCaptioner):
    """Vision-Language captioner backed by OpenBMB MiniCPM-V."""

    def __init__(self, model_id: str) -> None:
        dtype = select_torch_dtype()
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
            "This is a frame from a CCTV footage. Identify any animals or humans in the frame. If any human is present, describe their appearance and action. If there's no animal or human, say \"No Activity\"\n" 
            # "Any human entering or leaving the frame, mention their color of clothes and what they're carrying or holding, or what they are doing. If no activity is observed, just say \"no activity\"." 
            # "Previous frame captions are shared below for temporal context.\n"
            # f"Previous caption: {previous_caption or 'none'}.\n"
            # "Be concise. Do not repeat the previous caption."
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
            downsample_mode="16x",
        )
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
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
