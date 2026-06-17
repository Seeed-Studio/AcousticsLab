// Source owns data shape and jitter-smoothing, renderer owns canvas math, so one renderer paints any PCM source.
export interface PcmSource {
  // 0 signals idle (draw loop short-circuits on 0, no stale buffers) so no separate `isActive` field is needed.
  readonly sampleRate: number;

  // Playhead sample index at nominal rate from RAF time `nowMs`, decoupling visual flow from bursty arrival (~375 Hz worklet quantum vs 60 Hz RAF); may clamp/reset on stalls.
  renderCursor(nowMs: number): number;

  // Fill caller-owned `lo`/`hi` (no per-RAF alloc) with per-bin min/max across `bins` slots over `[endSample - samples, endSample)`; out-of-range bins read 0 for a flat baseline, not stale ring contents.
  envelopeAt(
    endSample: number,
    samples: number,
    bins: number,
    lo: Float32Array,
    hi: Float32Array
  ): void;
}
