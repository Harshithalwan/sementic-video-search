import argparse
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description="Test YOLO object detection on a video")
    parser.add_argument("--source", required=True, help="Path to input video file or webcam index")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model path (default: yolo26n.pt)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS (default: 0.5)")
    parser.add_argument("--skip-interval", type=int, default=1, help="Process every Nth frame (default: 1)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    from ultralytics import YOLO

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"Classes: {len(model.names)} — {', '.join(model.names.values())}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source: {args.source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {args.source} ({width}x{height}, {fps:.1f} FPS, {total_frames} frames)")
    print("Press 'q' to quit, SPACE to pause/resume\n")

    import numpy as np

    COLORS = np.random.randint(0, 255, size=(len(model.names), 3), dtype=np.uint8)

    frame_idx = 0
    processed = 0
    total_ms = 0.0
    paused = False
    all_class_counts = {}

    window_name = f"YOLO Detection — {args.model}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % args.skip_interval == 0:
                    t0 = time.perf_counter()
                    results = model(frame, verbose=False, conf=args.confidence, iou=args.iou)
                    ms = (time.perf_counter() - t0) * 1000
                    total_ms += ms
                    processed += 1

                    for r in results:
                        if r.boxes is None:
                            continue
                        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                        confs = r.boxes.conf.cpu().numpy()
                        clss = r.boxes.cls.cpu().numpy().astype(int)

                        for box, conf, cls_id in zip(boxes, confs, clss):
                            class_name = model.names[cls_id]
                            all_class_counts[class_name] = all_class_counts.get(class_name, 0) + 1
                            color = tuple(int(c) for c in COLORS[cls_id])

                            x1, y1, x2, y2 = box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            label = f"{class_name} {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    avg_ms = total_ms / processed
            else:
                ms = 0

            overlay = frame.copy()
            info_lines = [
                f"Model: {args.model} | Conf: {args.confidence}",
                f"Frame: {frame_idx} | Processed: {processed}",
                f"Inference: {ms:.1f}ms" if not paused else "PAUSED",
            ]
            y_pos = 25
            for line in info_lines:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(overlay, (5, y_pos - th - 5), (15 + tw, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(overlay, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                y_pos += th + 15

            cv2.imshow(window_name, overlay)

            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused

            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("YOLO DETECTION SUMMARY")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Confidence: {args.confidence} | IoU: {args.iou}")
    print(f"Frames processed: {processed}/{frame_idx}")
    print(f"Average inference: {total_ms / processed:.1f}ms" if processed else "No frames processed")
    print(f"Total time: {total_ms / 1000:.2f}s")
    if all_class_counts:
        print(f"\nDetected objects ({sum(all_class_counts.values())} total):")
        for cls_name, count in sorted(all_class_counts.items(), key=lambda x: -x[1]):
            print(f"  {cls_name:<20s} {count:>5d}")
    print("=" * 50)


if __name__ == "__main__":
    main()
