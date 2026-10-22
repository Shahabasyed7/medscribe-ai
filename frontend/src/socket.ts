/**
 * Native WebSocket manager for real-time PCM16 streaming and ASR messages.
 */

import { ServerMessage } from './types';
import { store } from './state';

export interface WebSocketCallbacks {
  onTranscript?: (text: string, chunk: number) => void;
  onVad?: (speaking: boolean) => void;
  onDone?: () => void;
  onError?: (message: string) => void;
}

export class AudioWebSocket {
  private ws: WebSocket | null = null;
  private callbacks: WebSocketCallbacks;
  private isConnecting = false;

  constructor(callbacks: WebSocketCallbacks = {}) {
    this.callbacks = callbacks;
  }

  connect(): Promise<void> {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return Promise.resolve();
    }

    this.isConnecting = true;
    return new Promise((resolve, reject) => {
      let wsUrl: string;
      const configuredWsBase = (import.meta.env.VITE_WS_BASE_URL || '').trim();

      if (configuredWsBase) {
        const cleanBase = configuredWsBase.replace(/\/+$/, '');
        wsUrl = `${cleanBase}/ws/audio`;
      } else {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsUrl = `${protocol}//${window.location.host}/ws/audio`;
      }

      try {
        this.ws = new WebSocket(wsUrl);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
          this.isConnecting = false;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            if (typeof event.data === 'string') {
              const msg: ServerMessage = JSON.parse(event.data);
              this.handleServerMessage(msg);
            }
          } catch (err) {
            console.error('Failed to parse WebSocket message:', err);
          }
        };

        this.ws.onerror = (err) => {
          this.isConnecting = false;
          console.error('WebSocket connection error:', err);
          store.setError('WebSocket error connecting to transcription server.');
          if (this.callbacks.onError) {
            this.callbacks.onError('WebSocket connection error');
          }
          reject(err);
        };

        this.ws.onclose = () => {
          this.isConnecting = false;
          store.setSpeaking(false);
          this.ws = null;
        };
      } catch (err) {
        this.isConnecting = false;
        reject(err);
      }
    });
  }

  private handleServerMessage(msg: ServerMessage): void {
    switch (msg.type) {
      case 'vad':
        store.setSpeaking(msg.speaking);
        if (msg.speaking) {
          store.setProvisionalText('Listening and transcribing speech...');
        } else {
          store.setProvisionalText('');
        }
        if (this.callbacks.onVad) this.callbacks.onVad(msg.speaking);
        break;

      case 'transcript':
        store.addTranscriptChunk({ text: msg.text, chunk: msg.chunk });
        if (this.callbacks.onTranscript) this.callbacks.onTranscript(msg.text, msg.chunk);
        break;

      case 'done':
        store.setSpeaking(false);
        if (this.callbacks.onDone) this.callbacks.onDone();
        break;

      case 'error':
        store.setError(msg.message);
        if (this.callbacks.onError) this.callbacks.onError(msg.message);
        break;
    }
  }

  sendAudioChunk(chunk: ArrayBuffer): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(chunk);
    }
  }

  sendStop(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send('stop');
    }
  }

  close(): void {
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // Ignore close errors
      }
      this.ws = null;
    }
  }

  isOpen(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  isConnectingState(): boolean {
    return this.isConnecting;
  }
}
