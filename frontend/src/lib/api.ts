const BASE = '';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options
    });
  } catch (e) {
    throw new Error(
      `Cannot reach backend at ${window.location.origin}. Is the FastAPI server running? (uvicorn backend.main:app --port 8000)`
    );
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export interface ModelInfo {
  type: string;
  default_id: string;
  default_prompt: string | null;
}

export interface ModelsResponse {
  models: ModelInfo[];
  collections: Record<string, string>;
}

export interface CollectionsResponse {
  collections: string[];
}

export interface VideosResponse {
  video_names: string[];
}

export interface CaptionMetadata {
  video_id: string;
  video_name: string;
  model_type: string;
  current_time: string;
  current_time_secs: number;
  video_timestamp: string;
  video_timestamp_ms: number;
  frame_index: number;
  source: string;
  caption: string;
  enriched_caption?: string;
  yolo_objects: string[];
  movement_summary?: string;
}

export interface QueryResult {
  score: number;
  document: string;
  metadata: CaptionMetadata;
}

export interface QueryResponse {
  results: QueryResult[];
  error?: string;
}

export interface ProcessStartResponse {
  video_id: string;
  status: string;
}

export interface ProcessStatusResponse {
  active: boolean;
  video_id?: string;
  captions_count?: number;
}

export function getModels(): Promise<ModelsResponse> {
  return request('/api/models');
}

export function getCollections(): Promise<CollectionsResponse> {
  return request('/api/collections');
}

export function getCollectionVideos(name: string): Promise<VideosResponse> {
  return request(`/api/collections/${encodeURIComponent(name)}/videos`);
}

export function searchCaptions(params: {
  query: string;
  top_k: number;
  collection: string;
  video_name?: string;
  video_id?: string;
  ts_from?: number;
  ts_to?: number;
  time_from?: number;
  time_to?: number;
}): Promise<QueryResponse> {
  return request('/api/query', {
    method: 'POST',
    body: JSON.stringify(params)
  });
}

export function startProcessing(config: {
  video_path: string;
  model_type: string;
  system_prompt?: string;
  caption_interval: number;
  max_new_tokens: number;
  max_frames: number;
  collection_name?: string;
  fps: number;
  clip_duration: number;
  activity_detection_enabled?: boolean;
  activity_detection_threshold?: number;
  yolo_enabled?: boolean;
  yolo_model?: string;
  yolo_confidence?: number;
  yolo_tracking?: boolean;
}): Promise<ProcessStartResponse> {
  return request('/api/process/start', {
    method: 'POST',
    body: JSON.stringify(config)
  });
}

export function stopProcessing(): Promise<{ status: string }> {
  return request('/api/process/stop', { method: 'POST', body: '{}' });
}

export function getProcessStatus(): Promise<ProcessStatusResponse> {
  return request('/api/process/status');
}
