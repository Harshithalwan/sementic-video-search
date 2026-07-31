import math
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import YOLOConfig


class ObjectDetector:
    def __init__(self, config: YOLOConfig):
        self.model_path = config.model_path
        self.confidence = config.confidence_threshold
        self.tracking_enabled = config.tracking_enabled
        self.tracker = config.tracker
        self.direction_window_seconds = config.direction_window_seconds
        self.stationary_threshold = config.stationary_threshold
        self.model = None

        # Per-track centroid history: {track_id: deque[(t, cx, cy)]}
        self._track_history: dict[int, deque] = {}
        self._last_seen: dict[int, float] = {}
        self._prune_after_seconds = max(2.0, self.direction_window_seconds * 4.0)

    def _load(self):
        if self.model is not None:
            return
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    def _infer(self, frame: np.ndarray):
        """Run inference and return the raw results plus elapsed time."""
        t0 = time.perf_counter()
        self._load()

        results = self.model(frame, verbose=False, conf=self.confidence)
        ms = (time.perf_counter() - t0) * 1000
        return results, ms

    @staticmethod
    def _extract(results):
        """Flatten detection results into (boxes, class ids, confidences, class names)."""
        boxes = []
        class_ids = []
        confidences = []
        class_names = []
        for r in results:
            if r.boxes is None:
                continue
            names = getattr(r, "names", None) or getattr(results[0], "names", {})
            for box in r.boxes.xyxy.cpu().numpy().astype(int):
                boxes.append(box)
            for cls_id in r.boxes.cls.cpu().numpy().astype(int):
                class_ids.append(int(cls_id))
                class_names.append(names[int(cls_id)])
            for conf in r.boxes.conf.cpu().numpy():
                confidences.append(float(conf))
        return boxes, class_ids, confidences, class_names

    def detect(self, frame: np.ndarray) -> tuple[List[str], float]:
        """Run detection and return (class names, elapsed ms)."""
        results, ms = self._infer(frame)
        _, _, _, class_names = self._extract(results)
        return class_names, ms

    def annotate(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str], float]:
        """Run detection and draw bounding boxes + labels onto a copy of the frame.

        Returns:
            (annotated_frame, class names, elapsed ms)
        """
        results, ms = self._infer(frame)
        boxes, class_ids, confidences, class_names = self._extract(results)

        annotated = frame.copy() if boxes else frame
        rng = np.random.RandomState(42)
        colors = {
            cls_id: tuple(int(c) for c in rng.randint(0, 255, size=3))
            for cls_id in set(class_ids)
        }

        for box, cls_id, conf, name in zip(boxes, class_ids, confidences, class_names):
            color = colors.get(cls_id, (0, 255, 0))
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated, class_names, ms

    # ------------------------------------------------------------------
    # Tracking (ByteTrack/BoT-SORT via Ultralytics) + movement analysis
    # ------------------------------------------------------------------

    def track(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict], float]:
        """Run object tracking and annotate the frame with trails + direction arrows.

        Returns:
            (annotated_frame, tracks, elapsed ms)
            where each track dict contains track_id, class, confidence, bbox,
            centroid, movement direction, displacement and speed.
        """
        t0 = time.perf_counter()
        self._load()

        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            verbose=False,
        )
        ms = (time.perf_counter() - t0) * 1000

        now = time.monotonic()
        frame_h, frame_w = frame.shape[:2]

        boxes = []
        class_ids = []
        confidences = []
        track_ids: List[Optional[int]] = []

        for r in results:
            if r.boxes is None:
                continue
            names = getattr(r, "names", None) or getattr(results[0], "names", {})
            ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else np.full(len(r.boxes), -1)
            for box, cls_id, conf, tid in zip(
                r.boxes.xyxy.cpu().numpy().astype(int),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
                ids,
            ):
                boxes.append(box)
                class_ids.append(int(cls_id))
                confidences.append(float(conf))
                track_ids.append(int(tid) if tid >= 0 else None)

        self._prune_tracks(now)

        tracks: List[dict] = []
        annotated = frame.copy() if boxes else frame
        rng = np.random.RandomState(42)
        colors = {
            cls_id: tuple(int(c) for c in rng.randint(0, 255, size=3))
            for cls_id in set(class_ids)
        }

        for box, cls_id, conf, tid, name in zip(
            boxes, class_ids, confidences, track_ids, self._names_of(results, class_ids)
        ):
            color = colors.get(cls_id, (0, 255, 0))
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            track_info = {
                "track_id": tid,
                "class": name,
                "confidence": float(conf),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "cx": cx,
                "cy": cy,
                "direction": "unknown",
                "dx": 0.0,
                "dy": 0.0,
                "nx": 0.0,
                "ny": 0.0,
                "speed": 0.0,
                "speed_px_per_sec": 0.0,
                "window_seconds": 0.0,
            }

            if tid is not None:
                history = self._track_history.setdefault(tid, deque(maxlen=1200))
                history.append((now, cx, cy))
                self._last_seen[tid] = now
                track_info.update(self._movement(history, now, frame_w, frame_h))
                newest_t = history[-1][0]
                window_start = newest_t - self.direction_window_seconds
                window_pts = [
                    (int(px), int(py)) for t, px, py in history if t >= window_start
                ]
                self._draw_trail(annotated, window_pts, color)
                if track_info["direction"] != "stationary" and len(window_pts) >= 2:
                    self._draw_arrow(annotated, window_pts, color)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}" + (f" #{tid}" if tid is not None else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            tracks.append(track_info)

        return annotated, tracks, ms

    def reset(self) -> None:
        """Clear tracker state (called when starting a new video session)."""
        self._track_history.clear()
        self._last_seen.clear()

    @staticmethod
    def _names_of(results, class_ids: List[int]) -> List[str]:
        names = getattr(results[0], "names", {})
        return [names.get(cid, "object") for cid in class_ids]

    def _movement(self, history, now: float, frame_w: int, frame_h: int) -> dict:
        """Compute displacement, direction and speed over the configured window.

        Direction convention (image coordinates, +y is down):
            angle = atan2(ny, nx) -> 0=right, 90=down, 180=left, 270=up.
        """
        newest_t, newest_cx, newest_cy = history[-1]
        oldest_t, oldest_cx, oldest_cy = history[0]
        if len(history) < 2 or newest_t - oldest_t <= 0:
            return {
                "direction": "stationary",
                "dx": 0.0, "dy": 0.0, "nx": 0.0, "ny": 0.0,
                "speed": 0.0, "speed_px_per_sec": 0.0, "window_seconds": 0.0,
            }

        # Pick the oldest sample inside the direction window.
        cutoff = newest_t - self.direction_window_seconds
        for t, cx, cy in history:
            if t >= cutoff:
                oldest_t, oldest_cx, oldest_cy = t, cx, cy
                break

        dx = newest_cx - oldest_cx
        dy = newest_cy - oldest_cy
        dt = newest_t - oldest_t

        nx = dx / frame_w
        ny = dy / frame_h
        speed_norm = math.hypot(nx, ny) / dt if dt > 0 else 0.0
        speed_px = math.hypot(dx, dy) / dt if dt > 0 else 0.0

        if speed_norm < self.stationary_threshold:
            direction = "stationary"
        else:
            ang = math.degrees(math.atan2(ny, nx)) % 360
            labels = [
                "right", "down-right", "down", "down-left",
                "left", "up-left", "up", "up-right",
            ]
            direction = labels[int(round(ang / 45)) % 8]

        return {
            "direction": direction,
            "dx": dx,
            "dy": dy,
            "nx": nx,
            "ny": ny,
            "speed": speed_norm,
            "speed_px_per_sec": speed_px,
            "window_seconds": dt,
        }

    def _prune_tracks(self, now: float) -> None:
        for tid in list(self._last_seen):
            if now - self._last_seen[tid] > self._prune_after_seconds:
                self._last_seen.pop(tid, None)
                self._track_history.pop(tid, None)

    @staticmethod
    def _draw_trail(annotated: np.ndarray, pts: list, color) -> None:
        if len(pts) < 2:
            return
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(annotated, p1, p2, color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_arrow(annotated: np.ndarray, pts: list, color) -> None:
        cv2.arrowedLine(annotated, pts[0], pts[-1], color, 2, cv2.LINE_AA, tipLength=0.25)
