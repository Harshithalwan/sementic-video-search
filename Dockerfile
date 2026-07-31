# ── Stage 1: Build SvelteKit frontend ─────────────────────────────
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

RUN npm install -D @sveltejs/adapter-static

COPY frontend/ ./

RUN sed -i "s|@sveltejs/adapter-auto|@sveltejs/adapter-static|g" svelte.config.js

RUN npm run build


# ── Stage 2: Python runtime ──────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY --from=frontend-build /app/frontend/build ./frontend/build

ENV HF_HOME=/app/hf_cache
ENV HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8001

CMD ["python", "-m", "backend.main", "--host", "0.0.0.0", "--port", "8001"]
