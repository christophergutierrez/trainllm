/** WebSocket client for real-time training events and agent communication. */

type Listener = (data: any) => void;

class WsChannel {
  private ws: WebSocket | null = null;
  private listeners: Set<Listener> = new Set();
  private reconnectTimer: number | null = null;
  private url: string;

  constructor(path: string) {
    const loc = typeof window !== 'undefined' ? window.location : { host: 'localhost:5173', protocol: 'http:' };
    const wsProtocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';
    const base = import.meta.env.VITE_WS_URL || `${wsProtocol}//${loc.host}`;
    this.url = `${base}${path}`;
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    this.ws = new WebSocket(this.url);

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.listeners.forEach(fn => fn(data));
      } catch {}
    };

    this.ws.onclose = () => {
      this.reconnectTimer = window.setTimeout(() => this.connect(), 3000);
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.connect();
    }
    return () => this.listeners.delete(fn);
  }

  send(data: object) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }
}

export const trainingWs = new WsChannel('/ws/training');
export const agentWs = new WsChannel('/ws/agent');
