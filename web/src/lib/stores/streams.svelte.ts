import { createStreamClient, type SocketState, type StreamClient } from '$lib/stream/client';
import type { TopK } from '$lib/stream/proto';
import type { PcmSource } from '$lib/audio/pcm-source';
import { CursorSmoother } from '$lib/audio/cursor-smoother';
import { envelopeFromRing, pushToRing } from '$lib/audio/ring-buffer';

// Streaming-worker singleton. The PCM ring is deliberately non-reactive ($state on 50 Hz frames would
// thrash consumers); renderers poll snapshot() at RAF. Worker + its two daemon WS subs are refcount-gated
// (not always-on) so the ~50 Hz Opus decode loop and live WS stop burning CPU/bandwidth when no one watches.

const PCM_SAMPLE_RATE = 48_000;
const PCM_BUFFER_SECONDS = 10;
const DEFAULT_RENDER_LATENCY_MS = 120;

export interface HeadInfo {
  head_id: string | null;
  head_version: number | null;
}

class StreamsStore implements PcmSource {
  audioStatus = $state<SocketState>('closed');
  inferStatus = $state<SocketState>('closed');
  latestTopK = $state<TopK[]>([]);
  head = $state<HeadInfo>({ head_id: null, head_version: null });
  unsupportedReason = $state<string | null>(null);
  inferenceFps = $state(0);

  readonly sampleRate = PCM_SAMPLE_RATE;
  readonly renderLatencyMs = DEFAULT_RENDER_LATENCY_MS;
  // Sliding visualizer lookback, NOT long-form capture (use tap() to accumulate beyond rollover).
  readonly ringSeconds = PCM_BUFFER_SECONDS;
  private readonly ring = new Float32Array(PCM_SAMPLE_RATE * PCM_BUFFER_SECONDS);
  private writeIdx = 0;
  private totalSamplesWritten = 0;

  // Exclusive index of the latest PCM write: shared anchor so panels reading the ring at different moments align on one audio-time window.
  get latestSample(): number {
    return this.totalSamplesWritten;
  }

  // Caps cursor speed within ±2.5% so packet jitter doesn't surface as visible jumps; looser depth bounds than the recorder's since network jitter exceeds worklet jitter.
  private readonly smoother = new CursorSmoother({
    latencyMs: DEFAULT_RENDER_LATENCY_MS,
    minDepthMs: 32,
    maxDepthMs: 500,
    resetAfterMs: 250,
    maxFrameMs: 100,
    slewGain: 0.1,
    maxSlew: 0.025
  });

  private client: StreamClient | null = null;
  // Live acquire() holders; worker runs while > 0. Plain field, not $state (nothing renders off it).
  private refcount = 0;
  private inferenceTimes: number[] = [];
  // Decays inferenceFps / clears latestTopK when the socket stays open but frames stop (device loss, engine fault -- never silence, which classifies normally), so the panel doesn't freeze on a stale prediction.
  private inferenceWatchdog: ReturnType<typeof setInterval> | null = null;
  private readonly pcmTaps = new Set<(pcm: Float32Array) => void>();

  // Opens the worker on the 0->1 refcount edge; returns an idempotent dispose stopping it on 1->0 (usable directly as a Svelte $effect cleanup).
  acquire(): () => void {
    this.refcount += 1;
    if (this.refcount === 1) this.connectClient();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.refcount -= 1;
      if (this.refcount === 0) this.disconnectClient();
    };
  }

  private connectClient(): void {
    if (this.client) return;
    // Set 'connecting' synchronously so the acquiring frame doesn't flash 'closed' before the worker's
    // first status; WebCodecs-unsupported still posts audio 'closed' after 'unsupported' so it can't hang.
    this.audioStatus = 'connecting';
    this.inferStatus = 'connecting';
    // Every listener gates on this.client === captured client: Worker.terminate() drops only pending
    // tasks, so already-posted messages still fire; the guard no-ops stale listeners across a rapid
    // disconnect->reconnect instead of clobbering reset state.
    const client = createStreamClient();
    this.client = client;
    client.audio.on(({ pcm }) => {
      if (this.client !== client) return;
      this.pushPcm(pcm);
    });
    client.inference.on(({ top_k, head_id, head_version }) => {
      if (this.client !== client) return;
      this.latestTopK = top_k;
      if (head_id !== this.head.head_id || head_version !== this.head.head_version) {
        this.head = { head_id, head_version };
      }
      this.trackInferenceFps();
    });
    client.status.on(({ channel, state }) => {
      if (this.client !== client) return;
      if (channel === 'audio') this.audioStatus = state;
      else this.inferStatus = state;
    });
    client.unsupported.on((reason) => {
      if (this.client !== client) return;
      this.unsupportedReason = reason;
    });
    client.start();
    // Decay FPS / Top-K when frames stop (no frames => no trackInferenceFps call); 1 Hz << the 2 s window.
    this.inferenceWatchdog ??= setInterval(() => this.recomputeInferenceFps(), 1_000);
  }

  private disconnectClient(): void {
    // Tear down worker only if present, but ALWAYS run the reset below: gating it on `client` would strand status at 'connecting' if connectClient threw before assigning this.client. Idempotent.
    if (this.client) {
      this.client.stop();
      this.client = null;
    }
    if (this.inferenceWatchdog !== null) {
      clearInterval(this.inferenceWatchdog);
      this.inferenceWatchdog = null;
    }
    // Snap reactive state to the construction-time sentinel so a re-acquire doesn't ghost pre-disconnect numbers on its first frame.
    this.audioStatus = 'closed';
    this.inferStatus = 'closed';
    this.latestTopK = [];
    this.inferenceFps = 0;
    this.head = { head_id: null, head_version: null };
    this.inferenceTimes.length = 0;
    // Zero ring + counters so a re-acquired visualizer doesn't paint the pre-disconnect tail before the first new packet (smoother self-resets after resetAfterMs of no samples).
    this.ring.fill(0);
    this.writeIdx = 0;
    this.totalSamplesWritten = 0;
    // unsupportedReason NOT reset: WebCodecs availability is a fixed browser fact, so the banner surfaces on next mount without waiting for the worker to re-announce.
  }

  // Most-recent `samples` values; pass `out` to write in-place (clamps to its length) and skip the per-frame allocation.
  snapshot(samples: number, out?: Float32Array): Float32Array {
    return this.snapshotAt(this.totalSamplesWritten, samples, out);
  }

  // Monotonic playhead: bursty 20 ms opus packets are jitter-buffered so the cursor trails the live edge by latencyMs and advances from RAF time at nominal rate, not write jitter. PcmSource signature so visualizers paint stream and mic through one path.
  renderCursor(nowMs: number = performance.now()): number {
    return this.smoother.step(this.totalSamplesWritten, this.ring.length, this.sampleRate, nowMs);
  }

  envelopeAt(
    endSample: number,
    samples: number,
    bins: number,
    lo: Float32Array,
    hi: Float32Array
  ): void {
    envelopeFromRing(this.ring, this.totalSamplesWritten, endSample, samples, bins, lo, hi);
  }

  // Samples ending at an absolute exclusive sample index: the sync primitive letting every panel sample one shared render cursor instead of "latest" at slightly different moments.
  snapshotAt(endSample: number, samples: number, out?: Float32Array): Float32Array {
    const r = this.ring.length;
    const n = Math.min(samples, r, out?.length ?? Infinity);
    const buf = out ?? new Float32Array(n);
    if (n === 0) return buf;

    buf.fill(0);
    const latest = this.totalSamplesWritten;
    const oldestAvailable = Math.max(0, latest - r);
    const clampedEnd = Math.max(oldestAvailable, Math.min(Math.floor(endSample), latest));
    const requestedStart = clampedEnd - n;
    const copyStart = Math.max(requestedStart, oldestAvailable);
    const copyLen = clampedEnd - copyStart;
    if (copyLen <= 0) return buf;

    const dst = copyStart - requestedStart;
    const start = copyStart % r;
    if (start + copyLen <= r) {
      buf.set(this.ring.subarray(start, start + copyLen), dst);
    } else {
      const head = r - start;
      buf.set(this.ring.subarray(start), dst);
      buf.set(this.ring.subarray(0, copyLen - head), dst + head);
    }
    return buf;
  }

  private pushPcm(pcm: Float32Array): void {
    this.writeIdx = pushToRing(this.ring, this.writeIdx, pcm);
    this.totalSamplesWritten += pcm.length;
    // Fan out AFTER the ring write so a tap reading latestSample sees the just-pushed packet.
    if (this.pcmTaps.size > 0) {
      for (const tap of this.pcmTaps) tap(pcm);
    }
  }

  // Subscribe to each PCM packet (worker-transferred Float32Array, may be retained without memcpy); taps fire in insertion order. Returns an idempotent dispose.
  tap(cb: (pcm: Float32Array) => void): () => void {
    this.pcmTaps.add(cb);
    return () => {
      this.pcmTaps.delete(cb);
    };
  }

  private trackInferenceFps(): void {
    this.inferenceTimes.push(performance.now());
    this.recomputeInferenceFps();
  }

  // Driven by both arriving frames and the watchdog, so a stalled stream decays to 0 Hz and clears stale Top-K.
  private recomputeInferenceFps(): void {
    const cutoff = performance.now() - 2_000;
    while (this.inferenceTimes.length > 0 && this.inferenceTimes[0] < cutoff) {
      this.inferenceTimes.shift();
    }
    this.inferenceFps = this.inferenceTimes.length / 2;
    if (this.inferenceTimes.length === 0 && this.latestTopK.length > 0) {
      this.latestTopK = [];
    }
  }
}

export const streams = new StreamsStore();
