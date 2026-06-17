import StreamWorker from './worker?worker';
import type { TopK } from './proto';

export type SocketChannel = 'audio' | 'infer';
export type SocketState = 'connecting' | 'open' | 'closed' | 'error';

export interface AudioMsg {
  seq: number;
  t_us_capture: number | null;
  pcm: Float32Array;
}

export interface InferenceMsg {
  seq: number;
  t_us_capture: number | null;
  top_k: TopK[];
  head_id: string | null;
  head_version: number | null;
}

export interface StatusMsg {
  channel: SocketChannel;
  state: SocketState;
}

type WorkerOut =
  | ({ type: 'audio' } & AudioMsg)
  | ({ type: 'inference' } & InferenceMsg)
  | ({ type: 'status' } & StatusMsg)
  | { type: 'unsupported'; reason: string };

type Listener<T> = (data: T) => void;

class Topic<T> {
  private readonly listeners = new Set<Listener<T>>();
  on(fn: Listener<T>): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }
  emit(data: T): void {
    for (const fn of this.listeners) fn(data);
  }
}

export interface StreamClient {
  start(): void;
  stop(): void;
  readonly audio: Topic<AudioMsg>;
  readonly inference: Topic<InferenceMsg>;
  readonly status: Topic<StatusMsg>;
  readonly unsupported: Topic<string>;
}

export function createStreamClient(): StreamClient {
  const audio = new Topic<AudioMsg>();
  const inference = new Topic<InferenceMsg>();
  const status = new Topic<StatusMsg>();
  const unsupported = new Topic<string>();
  let worker: Worker | null = null;
  let started = false;

  return {
    audio,
    inference,
    status,
    unsupported,
    start() {
      if (started) return;
      started = true;
      worker = new StreamWorker();
      worker.onmessage = (e: MessageEvent<WorkerOut>) => {
        const m = e.data;
        switch (m.type) {
          case 'audio':
            audio.emit(m);
            return;
          case 'inference':
            inference.emit(m);
            return;
          case 'status':
            status.emit(m);
            return;
          case 'unsupported':
            unsupported.emit(m.reason);
            return;
        }
      };
      worker.postMessage({ type: 'start' });
    },
    stop() {
      if (!started) return;
      // terminate() (not a posted 'stop', which the abort would discard) reclaims the
      // worker-owned WebSockets and AudioDecoder, but closes the WS abruptly with no close frame.
      worker?.terminate();
      worker = null;
      started = false;
    }
  };
}
