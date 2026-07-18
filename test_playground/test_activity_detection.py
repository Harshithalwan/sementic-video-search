import argparse
import csv
import os
import time

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as skimage_ssim

    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class PixelChangeDetector:
    name = "Pixel Change Ratio"

    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.compare_size = (64, 64)
        self.prev_gray = None

    def is_active(self, frame):
        t0 = time.perf_counter()
        resized = cv2.resize(frame, self.compare_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.prev_gray is None:
            self.prev_gray = gray
            return True, 1.0, (time.perf_counter() - t0) * 1000

        diff = np.mean(np.abs(gray - self.prev_gray)) / 255.0
        active = diff >= self.threshold
        if active:
            self.prev_gray = gray
        return active, float(diff), (time.perf_counter() - t0) * 1000


class HistogramDetector:
    name = "Histogram Comparison"

    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.prev_hist = None

    def is_active(self, frame):
        t0 = time.perf_counter()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()

        if self.prev_hist is None:
            self.prev_hist = hist
            return True, 1.0, (time.perf_counter() - t0) * 1000

        dist = cv2.compareHist(
            self.prev_hist.reshape(-1, 1).astype(np.float32),
            hist.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        score = 1.0 - dist
        active = dist >= self.threshold
        if active:
            self.prev_hist = hist
        return active, float(score), (time.perf_counter() - t0) * 1000


class SSIMDetector:
    name = "Structural Similarity"

    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.compare_size = (128, 128)
        self.prev_gray = None

    def is_active(self, frame):
        t0 = time.perf_counter()
        resized = cv2.resize(frame, self.compare_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return True, 1.0, (time.perf_counter() - t0) * 1000

        if HAS_SKIMAGE:
            score = skimage_ssim(self.prev_gray, gray)
        else:
            result = cv2.matchTemplate(self.prev_gray, gray, cv2.TM_CCOEFF_NORMED)
            score = float(result[0, 0])

        active = score < self.threshold
        if active:
            self.prev_gray = gray
        return active, float(score), (time.perf_counter() - t0) * 1000


class BGSubtractorDetector:
    name = "BG Subtractor (MOG2)"

    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

    def is_active(self, frame):
        t0 = time.perf_counter()
        mask = self.bg.apply(frame)
        ratio = np.count_nonzero(mask) / mask.size
        active = ratio >= self.threshold
        return active, float(ratio), (time.perf_counter() - t0) * 1000


class OpticalFlowDetector:
    name = "Optical Flow Magnitude"

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.compare_size = (160, 120)
        self.prev_gray = None

    def is_active(self, frame):
        t0 = time.perf_counter()
        resized = cv2.resize(frame, self.compare_size)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return True, 1.0, (time.perf_counter() - t0) * 1000

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(np.mean(mag))
        active = mean_mag >= self.threshold
        if active:
            self.prev_gray = gray
        return active, mean_mag, (time.perf_counter() - t0) * 1000


class CLIPDetector:
    name = "CLIP ViT-B/32"

    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.prev_embedding = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.eval()

    def _embed(self, frame):
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            pixel_values = inputs["pixel_values"]
            vision_out = self.model.vision_model(pixel_values)
            pooled = vision_out.pooler_output
            features = self.model.visual_projection(pooled)
        features = features.float()
        return features / features.norm(dim=-1, keepdim=True)

    def is_active(self, frame):
        import torch
        t0 = time.perf_counter()
        self._load()
        emb = self._embed(frame)

        if self.prev_embedding is None:
            self.prev_embedding = emb
            return True, 1.0, (time.perf_counter() - t0) * 1000

        sim = float(
            torch.nn.functional.cosine_similarity(
                emb, self.prev_embedding, dim=-1
            ).item()
        )
        active = sim < self.threshold
        if active:
            self.prev_embedding = emb
        return active, sim, (time.perf_counter() - t0) * 1000


class SigLIPDetector:
    name = "SigLIP Base"

    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.model = None
        self.processor = None
        self.prev_embedding = None

    def _load(self):
        if self.model is not None:
            return
        from transformers import SiglipModel, SiglipProcessor

        model_id = "google/siglip-base-patch16-224"
        self.processor = SiglipProcessor.from_pretrained(model_id)
        self.model = SiglipModel.from_pretrained(model_id)
        self.model.eval()

    def _embed(self, frame):
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            pixel_values = inputs["pixel_values"]
            vision_out = self.model.vision_model(pixel_values)
            features = vision_out.pooler_output
        features = features.float()
        return features / features.norm(dim=-1, keepdim=True)

    def is_active(self, frame):
        import torch
        t0 = time.perf_counter()
        self._load()
        emb = self._embed(frame)

        if self.prev_embedding is None:
            self.prev_embedding = emb
            return True, 1.0, (time.perf_counter() - t0) * 1000

        sim = float(
            torch.nn.functional.cosine_similarity(
                emb, self.prev_embedding, dim=-1
            ).item()
        )
        active = sim < self.threshold
        if active:
            self.prev_embedding = emb
        return active, sim, (time.perf_counter() - t0) * 1000


ALL_DETECTORS = [
    PixelChangeDetector,
    HistogramDetector,
    SSIMDetector,
    BGSubtractorDetector,
    OpticalFlowDetector,
    CLIPDetector,
    SigLIPDetector,
]


class ActivityBenchmark:
    def __init__(self, source, threshold, skip_interval, output_dir):
        self.source = source
        self.threshold = threshold
        self.skip_interval = skip_interval
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.detectors = [cls(threshold=threshold) for cls in ALL_DETECTORS]
        self.results = {d.name: [] for d in self.detectors}
        self.cap = cv2.VideoCapture(source)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def run(self):
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.skip_interval != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / self.fps
            for det in self.detectors:
                active, score, ms = det.is_active(frame)
                self.results[det.name].append(
                    {
                        "frame": frame_idx,
                        "timestamp": round(timestamp, 2),
                        "active": active,
                        "score": round(score, 4),
                        "ms": round(ms, 2),
                    }
                )
            frame_idx += 1

        self.cap.release()
        self._write_report()
        self._write_video()
        self._print_summary()

    def _write_report(self):
        path = os.path.join(self.output_dir, "benchmark_report.csv")
        headers = ["frame_num", "timestamp_sec"]
        for det in self.detectors:
            name_slug = det.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            headers.extend(
                [f"{name_slug}_active", f"{name_slug}_score", f"{name_slug}_ms"]
            )

        max_len = max(len(v) for v in self.results.values())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i in range(max_len):
                row = []
                if i < len(list(self.results.values())[0]):
                    entry = list(self.results.values())[0][i]
                    row.extend([entry["frame"], entry["timestamp"]])
                else:
                    row.extend(["", ""])
                for det in self.detectors:
                    if i < len(self.results[det.name]):
                        e = self.results[det.name][i]
                        row.extend([e["active"], e["score"], e["ms"]])
                    else:
                        row.extend(["", "", ""])
                writer.writerow(row)
        print(f"CSV report saved: {path}")

    def _write_video(self):
        self.cap = cv2.VideoCapture(self.source)
        orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        display_w = min(640, orig_w)
        display_h = int(orig_h * display_w / orig_w)
        overlay_h = 28 * len(self.detectors) + 10
        canvas_w = display_w
        canvas_h = display_h + overlay_h

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(self.output_dir, "benchmark_video.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, self.fps, (canvas_w, canvas_h))

        method_colors = [
            (0, 200, 0),
            (0, 165, 255),
            (255, 100, 0),
            (0, 200, 200),
            (200, 0, 200),
            (255, 255, 0),
            (0, 128, 255),
        ]

        frame_idx = 0
        result_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % self.skip_interval != 0:
                frame_idx += 1
                continue

            display = cv2.resize(frame, (display_w, display_h))
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[:display_h, :display_w] = display

            y_off = display_h + 5
            for di, det in enumerate(self.detectors):
                color = method_colors[di % len(method_colors)]
                if result_idx < len(self.results[det.name]):
                    entry = self.results[det.name][result_idx]
                    status = "ACTIVE" if entry["active"] else "  SKIP"
                    bar_color = (0, 200, 0) if entry["active"] else (0, 0, 200)
                    text = f"{det.name:<28s} {status}  score={entry['score']:.4f}  {entry['ms']:.1f}ms"
                    cv2.putText(
                        canvas, text, (5, y_off + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bar_color, 1, cv2.LINE_AA
                    )
                y_off += 28

            writer.write(canvas)
            result_idx += 1
            frame_idx += 1

        self.cap.release()
        writer.release()
        print(f"Annotated video saved: {out_path}")

    def _print_summary(self):
        print("\n" + "=" * 80)
        print("ACTIVITY DETECTION BENCHMARK SUMMARY")
        print(f"Source: {self.source}")
        print(f"Threshold: {self.threshold} | Skip interval: every {self.skip_interval} frame(s)")
        print("=" * 80)
        print(
            f"{'Method':<30s} | {'Active':>10s} | {'Total':>10s} | {'Avg ms/frame':>14s} | {'Total Time':>12s}"
        )
        print("-" * 80)

        for det in self.detectors:
            entries = self.results[det.name]
            active_count = sum(1 for e in entries if e["active"])
            total_count = len(entries)
            total_ms = sum(e["ms"] for e in entries)
            avg_ms = total_ms / total_count if total_count else 0
            total_s = total_ms / 1000
            print(
                f"{det.name:<30s} | {active_count:>5d}/{total_count:<4d} | {total_count:>10d} | {avg_ms:>12.2f}ms | {total_s:>10.2f}s"
            )

        print("=" * 80)
        print(
            "Higher active count = more sensitive. "
            "Lower ms/frame = faster. Adjust --threshold to tune sensitivity."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark activity detection methods on a video"
    )
    parser.add_argument("--source", required=True, help="Path to input video file")
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Detection threshold (default: 0.05)"
    )
    parser.add_argument(
        "--skip-interval",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)",
    )
    parser.add_argument(
        "--output-dir",
        default="activity_test_results",
        help="Output directory (default: activity_test_results/)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        print(f"Error: video file not found: {args.source}")
        return

    print(f"Loading detectors...")
    print(
        f"  scikit-image SSIM: {'available' if HAS_SKIMAGE else 'NOT FOUND (using OpenCV matchTemplate fallback)'}"
    )
    print(f"  CLIP/SigLIP models: will be lazy-loaded on first use")

    bench = ActivityBenchmark(
        source=args.source,
        threshold=args.threshold,
        skip_interval=args.skip_interval,
        output_dir=args.output_dir,
    )
    print(
        f"Video: {args.source} ({bench.total_frames} frames, {bench.fps:.1f} FPS)"
    )
    print(f"Processing {bench.total_frames // args.skip_interval} frames...\n")

    bench.run()


if __name__ == "__main__":
    main()
