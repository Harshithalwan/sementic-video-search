import { writable, derived } from 'svelte/store';
import type { WsMessage } from '$lib/ws';
import { CaptionWebSocket } from '$lib/ws';

export interface Caption {
  time: string;
  caption: string;
  frame: number;
  video_ts: string;
}

function createProcessingStore() {
  const isProcessing = writable(false);
  const captions = writable<Caption[]>([]);
  const statusMessage = writable('');
  const errorMessage = writable('');
  const videoId = writable('');
  let ws: CaptionWebSocket | null = null;

  function connect() {
    ws = new CaptionWebSocket((msg: WsMessage) => {
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
    });
    ws.connect();
  }

  function startProcessing(config: object) {
    captions.set([]);
    statusMessage.set('');
    errorMessage.set('');
    isProcessing.set(true);
    if (!ws) connect();
    ws!.start(config);
  }

  function stopProcessing() {
    ws?.stop();
    isProcessing.set(false);
    statusMessage.set('Stopped');
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
    connect,
    startProcessing,
    stopProcessing,
    disconnect
  };
}

export const processing = createProcessingStore();
