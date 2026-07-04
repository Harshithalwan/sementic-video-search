# LFM2.5-VL stream captioning

This workspace contains a small Python implementation that opens a local video source and feeds sampled frames to Liquid AI's LFM2.5-VL model for continuous captioning.

## What it does

- Opens a webcam, video file, or stream URL through `cv2.VideoCapture`.
- Sends only one frame every few seconds to the model so inference stays throttled.
- Carries the previous caption into the next prompt so the output reads like a live stream instead of disconnected frame descriptions.
- Prints timestamped captions to the terminal.

## Install

```bash
pip install -r requirements.txt
```

## Run

Webcam:

```bash
python lfm2_vl_stream.py --source 0
```

Video file:

```bash
python lfm2_vl_stream.py --source path\to\video.mp4
```

RTSP or other stream URL:

```bash
python lfm2_vl_stream.py --source rtsp://user:pass@host:554/stream
```

## Useful flags

- `--caption-interval 2.0` controls how long the script waits between model calls.
- `--max-new-tokens 48` keeps each caption short.
- `--show-preview` opens a small preview window.
- `--model-type lfm2.5` selects the model family. Supported families are `lfm2.5` (default) and `minicpm-v`.
- `--model-id` specifies the exact Hugging Face model id to load (e.g. `openbmb/MiniCPM-V-4.6`). If omitted, a default model is selected based on the `--model-type`.

## Notes

The model is used on single frames, not native video tensors. The continuous effect comes from sampling the stream over time and feeding the prior caption back into the prompt.
