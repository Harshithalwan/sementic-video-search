# Semantic Video Search

A Python pipeline that captions a live video stream using Vision-Language Models and stores the captions in a vector database for semantic search.

## What it does

- Opens a webcam, video file, or stream URL through `cv2.VideoCapture`.
- Feeds sampled frames to a Vision-Language Model (LFM2.5-VL or MiniCPM-V) for continuous captioning.
- Carries the previous caption into the next prompt so the output reads like a live narration rather than disconnected frame descriptions.
- Stores timestamped captions in a Qdrant vector database.
- Lets you semantically query stored captions to find matching scenes.

## Install

```bash
pip install -r requirements.txt
```

## Streaming — Generate Captions

The `stream` mode opens a video source, generates captions, and optionally saves them to the vector database.

Webcam (default source):

```bash
python main.py --mode stream --source 0
```

Video file:

```bash
python main.py --mode stream --source path\to\video.mp4
```

RTSP or other stream URL:

```bash
python main.py --mode stream --source rtsp://user:pass@host:554/stream
```

Use a different model:

```bash
python main.py --mode stream --model-type minicpm-v --source path\to\video.mp4
```

## Querying — Search Stored Captions

The `query` mode performs a semantic search over previously stored captions.

```bash
python main.py --mode query --query "a person waving"
```

Return more results:

```bash
python main.py --mode query --query "empty room" --top-k 10
```

## Parameters

### Mode

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {stream,query}` | `stream` | `stream` for live captioning, `query` to search stored captions. |

### Model Options (stream mode)

| Flag | Default | Description |
|------|---------|-------------|
| `--model-type {lfm2.5,minicpm-v}` | `lfm2.5` | The Vision-Language Model family to use. |
| `--model-id MODEL_ID` | *(auto)* | Exact Hugging Face model id (e.g. `openbmb/MiniCPM-V-4.6`). If omitted, a default is chosen based on `--model-type`. |

### Stream Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source SOURCE` | `0` | Video source: webcam index, file path, RTSP URL, or camera device string. |
| `--caption-interval SECONDS` | `2.0` | Seconds to wait between model calls to throttle inference. |
| `--max-new-tokens N` | `48` | Maximum tokens generated per caption. |
| `--max-frames N` | *(unlimited)* | Stop after this many captions. |
| `--show-preview` | off | Show a live OpenCV preview window (`q` to quit). |

### Database Options (shared)

| Flag | Default | Description |
|------|---------|-------------|
| `--vector-db-path PATH` | `./video_captions_db` | Local directory for the Qdrant database. |
| `--qdrant-url URL` | *(none)* | URL of a running Qdrant server (e.g. `http://localhost:6333`). Overrides local storage. |
| `--collection-name NAME` | *(model-specific)* | Qdrant collection name. Defaults to `captions_lfm2.5` or `captions_minicpm_v` based on `--model-type`. |
| `--disable-vector-db` | off | Skip saving captions to the database (stream mode only). |

### Query Options

| Flag | Default | Description |
|------|---------|-------------|
| `--query TEXT` | *(required)* | The semantic search query (e.g. `"a person walking"`). |
| `--top-k N` | `5` | Number of search results to return. |

### Query Filters (all optional)

Filter results by metadata fields. Multiple filters are combined with AND logic.

| Flag | Type | Description |
|------|------|-------------|
| `--filter-video-id TEXT` | `str` | Exact match on video ID (UUID assigned per stream session). |
| `--filter-video-name TEXT` | `str` | Exact match on video file name. |
| `--filter-ts-from MS` | `float` | Video timestamp lower bound (milliseconds). |
| `--filter-ts-to MS` | `float` | Video timestamp upper bound (milliseconds). |
| `--filter-time-from SECS` | `float` | Wall-clock time lower bound (seconds since midnight). |
| `--filter-time-to SECS` | `float` | Wall-clock time upper bound (seconds since midnight). |

Examples:

```bash
# Captions from a specific video session
python main.py --mode query --query "person walking" --filter-video-id <uuid>

# Captions for a specific video file
python main.py --mode query --query "car" --filter-video-name video.mp4

# Captions within the first 30 seconds of the video (0–30000 ms)
python main.py --mode query --query "hello" --filter-ts-from 0 --filter-ts-to 30000

# Captions generated between 10:00 and 10:30 (36000–37800 seconds since midnight)
python main.py --mode query --query "meeting" --filter-time-from 36000 --filter-time-to 37800

# Combining multiple filters
python main.py --mode query --query "dog" --filter-video-name video.mp4 --filter-ts-from 5000 --filter-ts-to 60000
```

## Resetting the Database

The `reset_db.py` script lets you list collections, delete a specific one, or wipe the entire database.

List all available collections:

```bash
python reset_db.py
```

Delete a specific collection:

```bash
python reset_db.py --collection-name captions_lfm2.5
```

Wipe all collections and the database directory:

```bash
python reset_db.py --full
```

Skip the confirmation prompt:

```bash
python reset_db.py -y --collection-name captions_minicpm_v
```

Use a custom database path or a remote Qdrant server:

```bash
python reset_db.py --vector-db-path ./my_db_dir --qdrant-url http://localhost:6333
```

### Reset Options

| Flag | Default | Description |
|------|---------|-------------|
| `--vector-db-path PATH` | `./video_captions_db` | Local directory for the Qdrant database. |
| `--qdrant-url URL` | *(none)* | URL of a running Qdrant server. |
| `--collection-name NAME` | *(list mode)* | Collection to delete. If omitted, lists available collections. |
| `--full` | off | Delete ALL collections and remove the database directory. |
| `-y, --yes` | off | Skip the confirmation prompt. |

## Project Structure

```
├── models/              # One file per VLM
│   ├── base.py          # BaseVideoCaptioner ABC + shared helpers
│   ├── lfm.py           # LFM2.5-VL captioner
│   └── minicpm.py       # MiniCPM-V captioner
├── database/            # Vector database layer
│   └── vector_store.py  # VectorStore (save + query)
├── ui/                  # Placeholder for future UI
├── main.py              # CLI entry-point (stream + query)
└── requirements.txt
```

## Notes

The models operate on single frames, not native video tensors. The continuous narration effect comes from sampling the stream over time and feeding the prior caption back into the prompt.
