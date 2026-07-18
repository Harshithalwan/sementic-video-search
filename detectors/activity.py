import time

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as skimage_ssim

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from config import ActivityDetectionConfig


class ActivityDetector:
    def __init__(self, config: ActivityDetectionConfig):
        self.threshold = config.threshold
        self.compare_size = config.compare_size
        self.prev_gray = None

    def is_active(self, frame: np.ndarray) -> tuple[bool, float, float]:
        t0 = time.perf_counter()
        resized = cv2.resize(frame, self.compare_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return True, 1.0, (time.perf_counter() - t0) * 1000

        if HAS_SKIMAGE:
            score = skimage_ssim(self.prev_gray, gray)
        else:
            result = cv2.matchTemplate(
                self.prev_gray, gray, cv2.TM_CCOEFF_NORMED
            )
            score = float(result[0, 0])

        active = score < self.threshold
        if active:
            self.prev_gray = gray
        return active, float(score), (time.perf_counter() - t0) * 1000
