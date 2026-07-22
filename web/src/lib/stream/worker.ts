/// <reference lib="webworker" />

import { wsUrl } from '$lib/api/base';
import { decodeEnvelope, type TopK } from './proto';

// Owns both WS streams for the page lifetime, decodes envelopes, runs Opus through WebCodecs, and
// posts transferable PCM windows + inference frames. Frames drain regardless of UI visibility:
// the daemon disconnects clients that lag > 64 frames.

// Must match the daemon's WS_SUBPROTOCOL exactly (strict admission + echo).
const SUBPROTOCOL = 'acousticslab.v1';
// Doubles per failure up to MAX, resets to MIN on onopen. 1 s (not sub-second) because each retry
// posts a `status` that re-renders the main-thread pill, so faster retries flood it under steady
// reject; daemon restarts take >= 1 s anyway.
const RECONNECT_MIN_MS = 1_000;
const RECONNECT_MAX_MS = 5_000;
const OPUS_SAMPLE_RATE = 48_000;

type Channel = 'audio' | 'infer';
type SocketState = 'connecting' | 'open' | 'closed' | 'error';

type InMsg = { type: 'start' } | { type: 'stop' };

type OutMsg =
  | { type: 'audio'; seq: number; t_us_capture: number | null; pcm: Float32Array }
  | {
      type: 'inference';
      seq: number;
      t_us_capture: number | null;
      top_k: TopK[];
      head_id: string | null;
      head_version: number | null;
    }
  | { type: 'status'; channel: Channel; state: SocketState }
  | { type: 'unsupported'; reason: string };

interface ChannelState {
  ws: WebSocket | null;
  backoff: number;
  retryTimer: ReturnType<typeof setTimeout> | null;
}

const channels: Record<Channel, ChannelState> = {
  audio: { ws: null, backoff: RECONNECT_MIN_MS, retryTimer: null },
  infer: { ws: null, backoff: RECONNECT_MIN_MS, retryTimer: null }
};

let running = false;
let opusDecoder: AudioDecoder | null = null;

// Keyed by the synthetic chunk timestamp WebCodecs preserves input->output, so seq/t_us matching
// survives decode errors (zero outputs for one input) and late/reordered outputs that a positional
// FIFO would permanently desync. A counter (not t_us, null for unstamped frames so nulls collide).
const pendingAudioMeta = new Map<number, { seq: number; t_us: number | null }>();
let pendingTsSource = 0;
// Soft cap: a decoder stall (malformed payload, stops emitting `output`) would grow the Map
// unboundedly. 256 ~= 5 s at ~50 Hz; oldest (insertion order) evicted past the cap.
const PENDING_META_CAP = 256;

self.onmessage = (e: MessageEvent<InMsg>) => {
  switch (e.data.type) {
    case 'start':
      start();
      break;
    case 'stop':
      stop();
      break;
  }
};

function start(): void {
  if (running) return;
  running = true;

  if (typeof AudioDecoder === 'undefined') {
    post({
      type: 'unsupported',
      reason:
        'WebCodecs AudioDecoder is unavailable in this browser. Live audio playback will not work.'
    });
    // Audio channel never opens; post `closed` so the main thread's optimistic 'connecting' flips
    // to "won't try" instead of hanging forever. Inference still works without the decoder.
    post({ type: 'status', channel: 'audio', state: 'closed' });
    openChannel('infer');
    return;
  }

  opusDecoder = new AudioDecoder({
    output: onDecodedAudio,
    error: (e) => {
      console.warn('opus decode error', e);
    }
  });
  opusDecoder.configure({ codec: 'opus', sampleRate: OPUS_SAMPLE_RATE, numberOfChannels: 1 });

  openChannel('audio');
  openChannel('infer');
}

function stop(): void {
  running = false;
  for (const ch of ['audio', 'infer'] as const) {
    const c = channels[ch];
    if (c.retryTimer) clearTimeout(c.retryTimer);
    c.retryTimer = null;
    c.ws?.close();
    c.ws = null;
  }
  pendingAudioMeta.clear();
  pendingTsSource = 0;
  opusDecoder?.close();
  opusDecoder = null;
}

function openChannel(ch: Channel): void {
  if (!running) return;
  const c = channels[ch];

  // Single-socket invariant: cancel a retry armed by a prior onclose, else a fast stop->start
  // leaves the timer armed and the c.ws overwrite below yields two concurrent sockets per channel.
  if (c.retryTimer) {
    clearTimeout(c.retryTimer);
    c.retryTimer = null;
  }
  // Detach listeners and close any lingering socket before overwriting; an orphan whose onclose
  // hasn't fired would keep posting status for a dead channel.
  if (c.ws) {
    c.ws.onopen = null;
    c.ws.onmessage = null;
    c.ws.onerror = null;
    c.ws.onclose = null;
    c.ws.close();
    c.ws = null;
  }

  post({ type: 'status', channel: ch, state: 'connecting' });

  const url = wsUrl(`/stream/${ch}`);
  let ws: WebSocket;
  try {
    ws = new WebSocket(url, SUBPROTOCOL);
  } catch (e) {
    // The WebSocket ctor throws synchronously on a malformed URL (a misconfigured base); without
    // this catch the channel breaks permanently (c.ws keeps its prior value, no retry scheduled).
    // Retry on the same backoff curve as a runtime close.
    console.warn('WebSocket construction failed', e);
    post({ type: 'status', channel: ch, state: 'error' });
    post({ type: 'status', channel: ch, state: 'closed' });
    c.retryTimer = setTimeout(() => {
      openChannel(ch);
    }, c.backoff);
    c.backoff = Math.min(RECONNECT_MAX_MS, c.backoff * 2);
    return;
  }
  ws.binaryType = 'arraybuffer';
  c.ws = ws;

  ws.onopen = () => {
    c.backoff = RECONNECT_MIN_MS;
    post({ type: 'status', channel: ch, state: 'open' });
  };
  ws.onmessage = (e) => {
    handleFrame(new Uint8Array(e.data as ArrayBuffer));
  };
  ws.onerror = () => {
    post({ type: 'status', channel: ch, state: 'error' });
  };
  ws.onclose = () => {
    post({ type: 'status', channel: ch, state: 'closed' });
    if (!running) return;
    c.retryTimer = setTimeout(() => {
      openChannel(ch);
    }, c.backoff);
    c.backoff = Math.min(RECONNECT_MAX_MS, c.backoff * 2);
  };
}

function handleFrame(bytes: Uint8Array): void {
  let env;
  try {
    env = decodeEnvelope(bytes);
  } catch (e) {
    console.warn('envelope decode failed', e);
    return;
  }
  switch (env.kind) {
    case 'audio':
      dispatchAudio(env.audio);
      return;
    case 'inference':
      dispatchInference(env.inference);
      return;
    case 'unknown':
      return;
  }
}

function dispatchAudio(frame: import('./proto').AudioFrame): void {
  if (frame.codec !== 'opus' || !frame.payload || !opusDecoder) return;
  // One synthetic value as both EncodedAudioChunk.timestamp and Map key (see pendingAudioMeta).
  const ts = pendingTsSource++;
  pendingAudioMeta.set(ts, { seq: frame.seq, t_us: frame.t_us_capture_monotonic });
  // keys().next() yields the oldest entry (insertion == iteration order).
  while (pendingAudioMeta.size > PENDING_META_CAP) {
    const oldest = pendingAudioMeta.keys().next().value;
    if (oldest === undefined) break;
    pendingAudioMeta.delete(oldest);
  }
  try {
    opusDecoder.decode(
      new EncodedAudioChunk({
        type: 'key',
        timestamp: ts,
        data: frame.payload
      })
    );
  } catch (e) {
    // Drop the entry on synchronous failure; async errors flow through the error callback and are
    // reclaimed by the size cap instead.
    pendingAudioMeta.delete(ts);
    console.warn('opus decode dispatch failed', e);
  }
}

function onDecodedAudio(audio: AudioData): void {
  const meta = pendingAudioMeta.get(audio.timestamp);
  if (meta !== undefined) pendingAudioMeta.delete(audio.timestamp);
  const pcm = new Float32Array(audio.numberOfFrames);
  try {
    audio.copyTo(pcm, { planeIndex: 0, format: 'f32-planar' });
  } catch {
    try {
      audio.copyTo(pcm, { planeIndex: 0, format: 'f32' });
    } catch (e) {
      console.warn('AudioData copy failed', e);
      audio.close();
      return;
    }
  }
  audio.close();
  const msg: OutMsg = {
    type: 'audio',
    seq: meta?.seq ?? 0,
    t_us_capture: meta?.t_us ?? null,
    pcm
  };
  (self as unknown as Worker).postMessage(msg, [pcm.buffer]);
}

function dispatchInference(frame: import('./proto').InferenceFrame): void {
  post({
    type: 'inference',
    seq: frame.seq,
    t_us_capture: frame.t_us_capture_monotonic,
    top_k: frame.top_k,
    head_id: frame.head_id,
    head_version: frame.head_version
  });
}

function post(msg: OutMsg): void {
  (self as unknown as Worker).postMessage(msg);
}

export {};
