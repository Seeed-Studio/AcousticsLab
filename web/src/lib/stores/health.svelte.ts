import { status as statusApi } from '$lib/api/endpoints';
import type { StatusSnapshot } from '$lib/api/types';

const POLL_INTERVAL_MS = 2_000;

class HealthStore {
  snapshot = $state<StatusSnapshot | null>(null);
  lastError = $state<string | null>(null);
  lastUpdated = $state<number | null>(null);

  private timer: ReturnType<typeof setInterval> | null = null;
  private inflight = false;
  // Stable identity so removeEventListener detaches it (an anonymous closure wouldn't).
  private onVisibilityChange = (): void => {
    if (typeof document === 'undefined' || document.hidden) return;
    // Tick on re-show so a long background pause doesn't keep stale data for up to
    // POLL_INTERVAL_MS; tick's inflight gate makes the interval+visibility race harmless.
    void this.tick();
  };

  start(): void {
    if (this.timer !== null) return;
    void this.tick();
    this.timer = setInterval(() => void this.tick(), POLL_INTERVAL_MS);
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this.onVisibilityChange);
    }
  }

  stop(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this.onVisibilityChange);
    }
  }

  // `unreachable` = transport failure (poll threw); `unhealthy` = daemon reachable but a
  // subsystem self-reports a fault. Distinct so one failed subsystem isn't shown as offline.
  get level(): 'unknown' | 'ok' | 'degraded' | 'unhealthy' | 'unreachable' {
    if (this.lastError) return 'unreachable';
    if (!this.snapshot) return 'unknown';
    let degraded = this.snapshot.metrics_stale;
    for (const sub of Object.values(this.snapshot.subsystems)) {
      if (!sub.healthy) return 'unhealthy';
      if (sub.stale || sub.degraded_reason) degraded = true;
    }
    return degraded ? 'degraded' : 'ok';
  }

  private async tick(): Promise<void> {
    if (this.inflight) return;
    if (typeof document !== 'undefined' && document.hidden) return;
    this.inflight = true;
    try {
      this.snapshot = await statusApi.get();
      this.lastError = null;
      this.lastUpdated = Date.now();
    } catch (e) {
      this.lastError = e instanceof Error ? e.message : String(e);
    } finally {
      this.inflight = false;
    }
  }
}

export const health = new HealthStore();
