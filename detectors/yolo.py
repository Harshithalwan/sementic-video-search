import time
from typing import List

import cv2
import numpy as np

from config import YOLOConfig


class ObjectDetector:
    def __init__(self, config: YOLOConfig):
        self.model_path = config.model_path
        self.confidence = config.confidence_threshold
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> tuple[List[str], float]:
        t0 = time.perf_counter()
        self._load()

        results = self.model(frame, verbose=False, conf=self.confidence)
        classes: List[str] = []
        for r in results:
            if r.boxes is not None:
                for cls_id in r.boxes.cls.cpu().numpy():
                    classes.append(self.model.names[int(cls_id)])

        ms = (time.perf_counter() - t0) * 1000
        return classes, ms
