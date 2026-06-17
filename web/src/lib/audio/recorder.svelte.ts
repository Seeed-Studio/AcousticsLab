import { WAV_SAMPLE_RATE } from './wav';
import { encodeWavFromChunks } from './resample';
import type { PcmSource } from './pcm-source';
import { CursorSmoother } from './cursor-smoother';
import { envelopeFromRing, pushToRing } from './ring-buffer';
import { m } from '$lib/i18n';

// Mic recorder -> 44.1 kHz mono PCM-16 WAV. AudioWorklet (not MediaRecorder) avoids a lossy
// opus/webm round-trip and Safari's patchy MediaRecorder audio. Consumer MUST call dispose() in
// onDestroy (an implicit Svelte onDestroy works only inside a component context).

export type RecorderState = 'idle' | 'requesting' | 'recording' | 'finalizing' | 'error';

export interface RecorderResult {
  blob: Blob;
  durationMs: number;
  sampleRate: number; // always WAV_SAMPLE_RATE regardless of capture rate
}

export interface RecorderOptions {
  // Soft cap (default 50 min): WAV must fit the 256 MiB import ceiling and stay clear of finalize peak RAM (~1.07 GiB at 50 min, >1.3 GiB at 1 hour).
  maxDurationMs?: number;
  onMaxDurationReached?: () => void;
  // Required or an auto-stopped recording is silently lost; null = too short / finalize errored.
  onAutoStop?: (result: RecorderResult | null) => void;
}

export interface StartOptions {
  // Applied via { ideal } not { exact } so a stale id (unplugged) degrades to default, not reject.
  deviceId?: string;
}

const DEFAULT_MAX_DURATION_MS = 3_000_000;

// Inline worklet (no Vite bundling): copies each mono frame out of the reused input plane, then posts it as a transferable so the hand-off pays no memcpy.
const WORKLET_SOURCE = `
class PcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch && ch.length > 0) {
      const copy = new Float32Array(ch);
      this.port.postMessage(copy, [copy.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
`;

// Re-added per recording (fresh AudioContext each start); finally() revokes the URL even on addModule throw.
async function ensureCaptureModule(ctx: AudioContext): Promise<void> {
  const blob = new Blob([WORKLET_SOURCE], { type: 'application/javascript' });
  const url = URL.createObjectURL(blob);
  await ctx.audioWorklet.addModule(url).finally(() => URL.revokeObjectURL(url));
}

// Tighter depth bounds than network streams: worklet render-quantum cadence (~375 Hz) jitters less than network packets.
const RECORDER_SMOOTHING = {
  latencyMs: 30,
  minDepthMs: 10,
  maxDepthMs: 200,
  resetAfterMs: 250,
  maxFrameMs: 100,
  slewGain: 0.1,
  maxSlew: 0.025
} as const;

export class Recorder implements PcmSource {
  state = $state<RecorderState>('idle');
  level = $state(0); // smoothed RMS, 0..1
  durationMs = $state(0);
  error = $state<string | null>(null);

  private readonly maxDurationMs: number;
  private readonly onMaxDurationReached: (() => void) | undefined;
  private readonly onAutoStop: ((result: RecorderResult | null) => void) | undefined;

  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private analyser: AnalyserNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  // Transferred worklet buffers (no receipt copy); finalize hands ownership to encodeWavFromChunks.
  private chunks: Float32Array[] = [];
  private capturedSamples = 0;
  private captureRate = 48_000;
  private startedAtMs = 0;
  private levelRaf = 0;
  private autoStopTimer: ReturnType<typeof setTimeout> | null = null;
  private rmsBuf: Float32Array | null = null;

  // Rolling PCM ring the live-waveform canvas envelopes at RAF; sized to capture rate, reused.
  private liveRing: Float32Array = new Float32Array(0);
  private liveWriteIdx = 0;
  private liveTotalWritten = 0;
  private static readonly LIVE_WINDOW_SECONDS = 3; // matches the waveform canvas window

  private readonly smoother = new CursorSmoother(RECORDER_SMOOTHING);

  constructor(options: RecorderOptions = {}) {
    this.maxDurationMs = options.maxDurationMs ?? DEFAULT_MAX_DURATION_MS;
    this.onMaxDurationReached = options.onMaxDurationReached;
    this.onAutoStop = options.onAutoStop;
  }

  async start(options: StartOptions = {}): Promise<void> {
    if (this.state === 'recording' || this.state === 'requesting') return;
    this.error = null;
    this.state = 'requesting';
    try {
      // Auto-gain / echo / noise-suppression off: they colour the signal away from classifier input.
      const audio: MediaTrackConstraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1
      };
      if (options.deviceId) {
        audio.deviceId = { ideal: options.deviceId };
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio,
        video: false
      });
      // cancel()/dispose() during the prompt mutates state: stop the stream, bail before wiring a stale lifecycle (TS narrows state across the await, hence the suppression).
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
      if (this.state !== 'requesting') {
        for (const track of stream.getTracks()) track.stop();
        return;
      }
      this.stream = stream;

      // Record at native device rate (waveform shows raw mic); resample to WAV_SAMPLE_RATE at finalize.
      this.ctx = new AudioContext();
      this.captureRate = this.ctx.sampleRate;
      const ringCapacity = Math.ceil(this.captureRate * Recorder.LIVE_WINDOW_SECONDS);
      if (this.liveRing.length !== ringCapacity) {
        this.liveRing = new Float32Array(ringCapacity);
      } else {
        this.liveRing.fill(0);
      }
      this.liveWriteIdx = 0;
      this.liveTotalWritten = 0;
      this.smoother.reset();
      await ensureCaptureModule(this.ctx);
      // cancel()/dispose() during addModule runs teardownGraph; bail before touching a closed context.
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
      if (this.state !== 'requesting') return;

      this.source = this.ctx.createMediaStreamSource(stream);
      this.analyser = this.ctx.createAnalyser();
      // 1024 @ 48 kHz = 21 ms window, below the 33 ms RAF tick (bigger = smoother but laggier RMS).
      this.analyser.fftSize = 1024;
      this.analyser.smoothingTimeConstant = 0; // raw frame; smoothed in JS

      this.workletNode = new AudioWorkletNode(this.ctx, 'pcm-capture');
      this.workletNode.port.onmessage = (e: MessageEvent<Float32Array>) => {
        if (this.state !== 'recording') return;
        const frame = e.data;
        this.chunks.push(frame);
        this.capturedSamples += frame.length;
        this.pushLiveRing(frame);
      };

      this.source.connect(this.analyser);
      this.source.connect(this.workletNode);
      // Chrome only schedules the worklet if it reaches a sink; muted gain -> destination is silent.
      const sink = this.ctx.createGain();
      sink.gain.value = 0;
      this.workletNode.connect(sink).connect(this.ctx.destination);

      this.chunks = [];
      this.capturedSamples = 0;
      this.durationMs = 0;
      this.level = 0;
      this.startedAtMs = performance.now();
      this.state = 'recording';
      this.startLevelLoop();

      this.autoStopTimer = setTimeout(() => {
        this.autoStopTimer = null;
        this.onMaxDurationReached?.();
        // finalizeAutoStop (not bare stop()) so onAutoStop receives the encoded WAV.
        void this.finalizeAutoStop();
      }, this.maxDurationMs);
    } catch (e) {
      this.teardownGraph();
      this.error = friendlyMicError(e);
      this.state = 'error';
      throw e;
    }
  }

  // Null if no samples or called outside 'recording' (first call owns the finalize); throws on encode error.
  async stop(): Promise<RecorderResult | null> {
    if (this.state !== 'recording') return null;
    this.state = 'finalizing';
    if (this.autoStopTimer !== null) {
      clearTimeout(this.autoStopTimer);
      this.autoStopTimer = null;
    }
    this.cancelLevelLoop();
    // Snapshot before teardown clears them; move chunks out of `this` (encoder takes ownership, nulls slots).
    const totalSamples = this.capturedSamples;
    const inputRate = this.captureRate;
    const chunks = this.chunks;
    this.chunks = [];
    this.teardownGraph();

    try {
      if (totalSamples === 0) {
        this.state = 'idle';
        return null;
      }
      const { blob, outputSamples } = await encodeWavFromChunks(
        chunks,
        totalSamples,
        inputRate,
        WAV_SAMPLE_RATE
      );
      const durationMs = Math.round((outputSamples / WAV_SAMPLE_RATE) * 1000);
      this.durationMs = durationMs;
      this.state = 'idle';
      return {
        blob,
        durationMs,
        sampleRate: WAV_SAMPLE_RATE
      };
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      this.state = 'error';
      throw e;
    }
  }

  // Split out so the cap timer can await the encode; on error stop() set error+state and we deliver null so onAutoStop always settles.
  private async finalizeAutoStop(): Promise<void> {
    let result: RecorderResult | null = null;
    try {
      result = await this.stop();
    } catch {
      // stop() already set state/error
    }
    this.onAutoStop?.(result);
  }

  cancel(): void {
    if (this.state === 'idle') return;
    if (this.autoStopTimer !== null) {
      clearTimeout(this.autoStopTimer);
      this.autoStopTimer = null;
    }
    this.cancelLevelLoop();
    this.chunks = [];
    this.capturedSamples = 0;
    this.durationMs = 0;
    this.teardownGraph();
    this.state = 'idle';
    this.error = null;
  }

  reset(): void {
    if (this.state === 'recording' || this.state === 'requesting' || this.state === 'finalizing') {
      this.cancel();
    }
    this.state = 'idle';
    this.error = null;
    this.level = 0;
    this.durationMs = 0;
  }

  dispose(): void {
    this.cancel();
  }

  // PcmSource: waveform canvas reads mic or network-stream data through identical shapes into renderer-owned scratch buffers (zero per-RAF allocation).

  // 0 unless recording, so callers can guard.
  get sampleRate(): number {
    return this.state === 'recording' ? this.captureRate : 0;
  }

  // Worklet posts at ~375 Hz vs 60 Hz RAF; raw liveTotalWritten would alias and step, so CursorSmoother jitter-buffers it.
  renderCursor(nowMs: number = performance.now()): number {
    return this.smoother.step(this.liveTotalWritten, this.liveRing.length, this.captureRate, nowMs);
  }

  // Bins before the oldest available sample read zero (canvas paints a flat baseline).
  envelopeAt(
    endSample: number,
    samples: number,
    bins: number,
    lo: Float32Array,
    hi: Float32Array
  ): void {
    envelopeFromRing(this.liveRing, this.liveTotalWritten, endSample, samples, bins, lo, hi);
  }

  private pushLiveRing(frame: Float32Array): void {
    this.liveWriteIdx = pushToRing(this.liveRing, this.liveWriteIdx, frame);
    this.liveTotalWritten += frame.length;
  }

  private startLevelLoop(): void {
    const tick = (): void => {
      if (this.state !== 'recording') {
        this.levelRaf = 0;
        return;
      }
      const a = this.analyser;
      if (a) {
        const n = a.fftSize;
        if (this.rmsBuf?.length !== n) {
          this.rmsBuf = new Float32Array(n);
        }
        // TS 5.7 wants Float32Array<ArrayBuffer>; our buffer is always ArrayBuffer-backed.
        a.getFloatTimeDomainData(this.rmsBuf as Float32Array<ArrayBuffer>);
        let sumSq = 0;
        for (let i = 0; i < n; i++) {
          const v = this.rmsBuf[i];
          sumSq += v * v;
        }
        const rms = Math.sqrt(sumSq / n);
        this.level = this.level * 0.5 + rms * 0.5; // EMA for steadier visuals
      }
      this.durationMs = Math.round(performance.now() - this.startedAtMs);
      this.levelRaf = requestAnimationFrame(tick);
    };
    this.levelRaf = requestAnimationFrame(tick);
  }

  private cancelLevelLoop(): void {
    if (this.levelRaf !== 0) {
      cancelAnimationFrame(this.levelRaf);
      this.levelRaf = 0;
    }
  }

  // disconnect() can throw on an already-detached node (caught via try/catch); ctx.close() may reject on an already-closed context (handled below via .catch).
  private teardownGraph(): void {
    if (this.workletNode) {
      try {
        this.workletNode.port.onmessage = null;
        this.workletNode.disconnect();
      } catch {
        /* empty */
      }
      this.workletNode = null;
    }
    if (this.analyser) {
      try {
        this.analyser.disconnect();
      } catch {
        /* empty */
      }
      this.analyser = null;
    }
    if (this.source) {
      try {
        this.source.disconnect();
      } catch {
        /* empty */
      }
      this.source = null;
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (this.ctx) {
      // Don't await: some Safari builds reject close() on already-closed contexts.
      this.ctx.close().catch(() => undefined);
      this.ctx = null;
    }
  }
}

function friendlyMicError(err: unknown): string {
  if (err && typeof err === 'object' && 'name' in err) {
    const name = (err as { name: string }).name;
    switch (name) {
      case 'NotAllowedError':
      case 'SecurityError':
        return m.recorder.mic_error_denied;
      case 'NotFoundError':
      case 'OverconstrainedError':
        return m.recorder.mic_error_not_found;
      case 'NotReadableError':
        return m.recorder.mic_error_in_use;
      case 'AbortError':
        return m.recorder.mic_error_interrupted;
    }
  }
  if (err instanceof Error) return finishCopy(err.message);
  return m.recorder.mic_error_generic;
}

function finishCopy(s: string): string {
  const t = s.trim();
  if (!t) return m.recorder.mic_error_generic;
  const head = t[0].toUpperCase() + t.slice(1);
  return /[.!?…]$/.test(head) ? head : `${head}.`;
}
