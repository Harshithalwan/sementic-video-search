# Semantic Video Search

Captions live video streams using Vision-Language Models and stores captions in a Qdrant vector database for semantic search.

## Prerequisites

- Python 3.10+
- Node.js 18+

## Install

```bash
pip install -r requirements.txt
cd frontend && npm install
```

## Running

### Backend (FastAPI)

```bash
uvicorn backend.main:app --port 8001
```

### Frontend (SvelteKit)

```bash
cd frontend
npm run dev          # dev server at http://localhost:5173
npm run build        # production build
npm run preview      # preview production build
npm run check        # type-check
```

### CLI (no frontend needed)

**Stream mode** — caption a video source:

```bash
python main.py --mode stream --source 0                         # webcam
python main.py --mode stream --source path\to\video.mp4          # video file
python main.py --mode stream --source rtsp://host:554/stream     # RTSP stream
python main.py --mode stream --model-type minicpm-v              # different model
python main.py --mode stream --source video.mp4 --show-preview   # show preview window
```

**Query mode** — search stored captions:

```bash
python main.py --mode query --query "a person waving"
python main.py --mode query --query "empty room" --top-k 10
python main.py --mode query --query "dog" --filter-video-name video.mp4 --filter-ts-from 5000 --filter-ts-to 60000
```

## Parameters

### Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {stream,query}` | `stream` | `stream` for live captioning, `query` to search stored captions. |

### Model Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type {lfm2.5,lfm2.5_larger,minicpm-v,llava-video}` | `lfm2.5` | Vision-Language Model family. |
| `--model-id MODEL_ID` | *(auto)* | Exact Hugging Face model id. If omitted, a default is chosen based on `--model-type`. |

### Stream Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source SOURCE` | `0` | Video source: webcam index, file path, RTSP URL. |
| `--caption-interval SECONDS` | `2.0` | Seconds between model calls. |
| `--max-new-tokens N` | `48` | Maximum tokens generated per caption. |
| `--max-frames N` | *(unlimited)* | Stop after this many captions. |
| `--show-preview` | off | Show a live OpenCV preview window (`q` to quit). |

### Detection Options

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-activity-detection` | off | Skip frames with low SSIM change. |
| `--activity-threshold` | `0.85` | SSIM threshold for activity detection. |
| `--enable-yolo` | off | Enable YOLO object detection. |
| `--yolo-model PATH` | `yolo26n.pt` | YOLO model path. |
| `--yolo-confidence` | `0.5` | YOLO confidence threshold. |

### Latency Logging

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-latency-logging` | off | Log per-component latency (SSIM, YOLO, caption model) to `latency_logs/` as JSON Lines. |

When enabled, a `latency_logs/{hostname}_{session}_{model_type}.jsonl` file is created with one JSON object per line. Each line includes `event` type (`system_info`, `ssim`, `yolo`, or `caption`), `elapsed_ms`, `hostname`, and component-specific metadata — making it easy to compare latency across machines.

**Usage:**
```bash
python main.py --mode stream --source video.mp4 --enable-yolo --enable-activity-detection --enable-latency-logging
```

### Database Options

| Flag | Default | Description |
|------|---------|-------------|
| `--vector-db-path PATH` | `./video_captions_db` | Local directory for the Qdrant database. |
| `--qdrant-url URL` | *(none)* | URL of a running Qdrant server (e.g. `http://localhost:6333`). Overrides local storage. |
| `--collection-name NAME` | *(model-specific)* | Qdrant collection name. Defaults vary by `--model-type`. |
| `--disable-vector-db` | off | Skip saving captions to the database (stream mode only). |

### Query Options

| Flag | Default | Description |
|------|---------|-------------|
| `--query TEXT` | *(required)* | The semantic search query. |
| `--top-k N` | `5` | Number of search results to return. |
| `--filter-video-id TEXT` | — | Exact match on video ID (UUID). |
| `--filter-video-name TEXT` | — | Exact match on video file name. |
| `--filter-ts-from MS` | — | Video timestamp lower bound (milliseconds). |
| `--filter-ts-to MS` | — | Video timestamp upper bound (milliseconds). |
| `--filter-time-from SECS` | — | Wall-clock time lower bound (seconds since midnight). |
| `--filter-time-to SECS` | — | Wall-clock time upper bound (seconds since midnight). |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check. |
| GET | `/api/models` | List available models and collections. |
| GET | `/api/collections` | List Qdrant collections. |
| GET | `/api/collections/{name}/videos` | List video names in a collection. |
| POST | `/api/query` | Semantic search (`query`, `top_k`, `collection`, `filters`). |
| POST | `/api/process/start` | Start video processing. |
| POST | `/api/process/stop` | Stop active processing. |
| GET | `/api/process/status` | Get processing status. |
| WS | `/ws/captions` | Real-time caption stream. |

## Resetting the Database

```bash
python reset_db.py                                    # list collections
python reset_db.py --collection-name captions_lfm2.5   # delete a collection
python reset_db.py --full                              # wipe everything
python reset_db.py -y --collection-name captions_minicpm_v  # skip confirmation
```

### Reset Options

| Flag | Default | Description |
|------|---------|-------------|
| `--vector-db-path PATH` | `./video_captions_db` | Local database directory. |
| `--qdrant-url URL` | *(none)* | Remote Qdrant server URL. |
| `--collection-name NAME` | *(list mode)* | Collection to delete. |
| `--full` | off | Delete ALL collections and database directory. |
| `-y, --yes` | off | Skip confirmation prompt. |

## Docker Deployment

### Build

```bash
docker build -t dessertation .
```

### Run

```bash
docker run -p 8001:8001 -v hf_cache:/app/hf_cache dessertation
```

The app is available at **http://localhost:8001**.

### How it works

- The SvelteKit frontend is built and served as static files from FastAPI — no separate frontend server needed.
- HuggingFace models are **not** baked into the image. On the first request that triggers a model, it will be downloaded automatically from HuggingFace Hub. Subsequent runs use the cached download.
- The `-v hf_cache:/app/hf_cache` flag persists the model cache across container restarts so you only download once. Omit it if you don't mind re-downloading each time.

### Selecting a model

Models are selected via the web UI or API. The default model is `lfm2.5` (`LiquidAI/LFM2.5-VL-450M`), which is the smallest and fastest. Other available models:

| Model Type | Model ID | Size |
|---|---|---|
| `lfm2.5` | `LiquidAI/LFM2.5-VL-450M` | ~450M |
| `lfm2.5_larger` | `LiquidAI/LFM2.5-VL-1.8B` | ~1.8B |
| `minicpm-v` | `openbmb/MiniCPM-V-4.6` | ~4B |
| `llava-video` | `llava-hf/LLaVA-NeXT-Video-7B-hf` | ~7B |

### With a remote Qdrant server

If you run Qdrant separately (e.g. via Docker), override the connection at runtime:

```bash
docker run -p 8001:8001 \
  -v hf_cache:/app/hf_cache \
  dessertation \
  uvicorn backend.main:app --host 0.0.0.0 --port 8001
```

Then pass `--qdrant-url http://host.docker.internal:6333` via the API if needed.

## Project Structure

```
├── backend/            # FastAPI app (routers, services)
├── database/           # Qdrant vector store abstraction
├── detectors/          # SSIM activity detection, YOLO object detection
├── loggers/            # Latency logging (JSONL per-component timing)
├── frontend/           # SvelteKit UI
├── models/             # Vision-Language Model implementations
│   ├── base.py         # BaseVideoCaptioner ABC + shared helpers
│   ├── lfm.py          # LFM2.5-VL captioner
│   ├── minicpm.py      # MiniCPM-V captioner
│   └── llavavideo.py   # LLaVA-NeXT-Video captioner (native video clips)
├── main.py             # CLI entry-point (stream + query)
├── config.py           # Pipeline configuration dataclasses
├── reset_db.py         # Database utility
├── Dockerfile          # Multi-stage Docker build (frontend + backend)
├── .dockerignore
└── requirements.txt
```
