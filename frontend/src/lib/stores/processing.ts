import { writable } from 'svelte/store';
import type { WsMessage } from '$lib/ws';
import { CaptionWebSocket } from '$lib/ws';

export interface YoloTrack {
  track_id: number | null;
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
  cx: number;
  cy: number;
  direction: string;
  dx: number;
  dy: number;
  nx: number;
  ny: number;
  speed: number;
  speed_px_per_sec: number;
  window_seconds: number;
}

export interface Caption {
  time: string;
  caption: string;
  frame: number;
  video_ts: string;
  yolo_objects: string[];
  yolo_tracks?: YoloTrack[];
  movement_summary?: string;
}

function createProcessingStore() {
  const isProcessing = writable(false);
  const captions = writable<Caption[]>([]);
  const statusMessage = writable('');
  const errorMessage = writable('');
  const videoId = writable('');
  const videoName = writable('');
  const modelName = writable('');
  const frame = writable<ArrayBuffer | null>(null);
  let ws: CaptionWebSocket | null = null;

  function connect() {
    if (ws) return;
    ws = new CaptionWebSocket(
      (msg: WsMessage) => {
        switch (msg.type) {
          case 'caption':
            captions.update((c) => [...c, msg.data]);
            break;
          case 'status':
            statusMessage.set(msg.message);
            break;
          case 'error':
            errorMessage.set(msg.message);
            break;
          case 'done':
            isProcessing.set(false);
            statusMessage.set('Processing complete');
            break;
        }
      },
      (jpeg: ArrayBuffer) => {
        frame.set(jpeg);
      }
    );
    ws.connect();
  }

  function startProcessing(config: any) {
    captions.set([]);
    statusMessage.set('');
    errorMessage.set('');
    frame.set(null);
    isProcessing.set(true);
    const path = (config.video_path ?? '') as string;
    videoName.set(path.split(/[\\/]/).pop() || path);
    modelName.set((config.model_type ?? '') as string);
    if (!ws) connect();
    ws!.start(config);
  }

  function stopProcessing() {
    ws?.stop();
    isProcessing.set(false);
    statusMessage.set('Stopped');
  }

  function reset() {
    captions.set([]);
    statusMessage.set('');
    errorMessage.set('');
    frame.set(null);
    videoName.set('');
    modelName.set('');
    isProcessing.set(false);
  }

  function disconnect() {
    ws?.disconnect();
    ws = null;
  }

  return {
    isProcessing,
    captions,
    statusMessage,
    errorMessage,
    videoId,
    videoName,
    modelName,
    frame,
    connect,
    startProcessing,
    stopProcessing,
    reset,
    disconnect
  };
}

export const processing = createProcessingStore();
