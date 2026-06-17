// Smooths a playhead across bursty PCM clocks (opus packets, mic quanta): start `latencyMs` behind
// the live edge, advance from RAF time at the sample rate with a small proportional slew, and snap
// back to the delayed edge on large discontinuities (tab throttle, reconnect, stall). Tune network
// with a deeper jitter buffer; mic runs tighter since jitter is render-quantum-bounded.

export interface SmoothingConfig {
  readonly latencyMs: number;
  readonly minDepthMs: number; // below: hold, don't rewind
  readonly maxDepthMs: number; // above: snap to live edge
  readonly resetAfterMs: number; // RAF gap above this (tab throttle) forces a snap
  readonly maxFrameMs: number; // advance cap so a recovery tick can't fast-forward
  readonly slewGain: number;
  readonly maxSlew: number; // slew cap so normal jitter never visibly alters playhead speed
}

export class CursorSmoother {
  private sample = 0;
  private timeMs = 0;
  private initialized = false;

  constructor(private readonly config: SmoothingConfig) {}

  reset(): void {
    this.sample = 0;
    this.timeMs = 0;
    this.initialized = false;
  }

  // `latest` is the producer's exclusive monotonic write index; `ringLength` bounds lookback.
  step(latest: number, ringLength: number, sampleRate: number, nowMs: number): number {
    if (latest === 0) {
      this.sample = 0;
      this.timeMs = nowMs;
      this.initialized = false;
      return 0;
    }
    const cfg = this.config;
    const oldestAvailable = Math.max(0, latest - ringLength);
    const samplesPerMs = sampleRate / 1000;
    const latencySamples = Math.max(0, cfg.latencyMs * samplesPerMs);
    const minDepthSamples = Math.min(latencySamples, cfg.minDepthMs * samplesPerMs);
    const maxDepthSamples = Math.max(latencySamples + 1, cfg.maxDepthMs * samplesPerMs);

    if (!this.initialized || this.sample < oldestAvailable || this.sample > latest) {
      return this.snapToEdge(latest, oldestAvailable, latencySamples, nowMs);
    }

    const rawDtMs = nowMs - this.timeMs;
    if (rawDtMs <= 0) return Math.floor(this.sample);
    this.timeMs = nowMs;

    const depthSamples = latest - this.sample;
    if (rawDtMs > cfg.resetAfterMs || depthSamples > maxDepthSamples) {
      return this.snapToEdge(latest, oldestAvailable, latencySamples, nowMs);
    }
    if (depthSamples < minDepthSamples) {
      return Math.floor(this.sample); // underrun: hold, don't rewind; next producer tick refills depth
    }

    const dtMs = Math.min(rawDtMs, cfg.maxFrameMs);
    const normalizedError =
      latencySamples > 0
        ? (depthSamples - latencySamples) / latencySamples
        : depthSamples > 0
          ? 1
          : 0;
    let slew = normalizedError * cfg.slewGain;
    if (slew > cfg.maxSlew) slew = cfg.maxSlew;
    else if (slew < -cfg.maxSlew) slew = -cfg.maxSlew;
    const next = this.sample + dtMs * samplesPerMs * (1 + slew);

    const monotonicFloor = Math.max(this.sample, oldestAvailable);
    this.sample = Math.max(monotonicFloor, Math.min(next, latest));
    return Math.floor(this.sample);
  }

  private snapToEdge(
    latest: number,
    oldestAvailable: number,
    latencySamples: number,
    nowMs: number
  ): number {
    const cursor = Math.max(oldestAvailable, Math.min(latest, latest - latencySamples));
    this.sample = cursor;
    this.timeMs = nowMs;
    this.initialized = true;
    return Math.floor(cursor);
  }
}
