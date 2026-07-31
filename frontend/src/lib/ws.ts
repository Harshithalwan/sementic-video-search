export type WsMessage =
  | { type: 'caption'; data: { time: string; caption: string; frame: number; video_ts: string; yolo_objects: string[] } }
  | { type: 'status'; message: string }
  | { type: 'error'; message: string }
  | { type: 'done' };

export type WsHandler = (msg: WsMessage) => void;
export type WsFrameHandler = (jpeg: ArrayBuffer) => void;

export class CaptionWebSocket {
  private ws: WebSocket | null = null;
  private handler: WsHandler;
  private frameHandler: WsFrameHandler;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private shouldReconnect = true;

  constructor(handler: WsHandler, frameHandler: WsFrameHandler) {
    this.handler = handler;
    this.frameHandler = frameHandler;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/captions`;

    try {
      this.ws = new WebSocket(url);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      console.log('[ws] connected');
    };

    this.ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        this.frameHandler(event.data);
        return;
      }
      try {
        const msg: WsMessage = JSON.parse(event.data);
        this.handler(msg);
      } catch {
        // ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      // onclose will fire after onerror
    };
  }

  private scheduleReconnect() {
    if (!this.shouldReconnect) return;
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 3000);
  }

  send(msg: object) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  start(config: object) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.handler({ type: 'error', message: 'WebSocket not connected. Is the backend running?' });
      return;
    }
    this.send({ type: 'start', config });
  }

  stop() {
    this.send({ type: 'stop' });
  }

  disconnect() {
    this.shouldReconnect = false;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }
}
