// JSONL-paged tail for the `event_gap` (409) recovery path: live progress normally flows over SSE, but when
// the daemon's per-job event ring overflows past the consumer's `after_seq` this reads the durable append-only
// `<job_id>.jsonl` mirror instead. Dormant on typical loads.

import { training as trainingApi } from './endpoints';
import { isNotFound } from '$lib/utils/error-copy';
import type { LogEvent, TrainLogLine, Uuid } from './types';

export interface TrainingLogTailOptions {
  intervalMs?: number;
  pageLimit?: number;
  // Fired once per event in monotonic `seq` order; consumers must skip unknown forward-compat `kind` values, not throw.
  onEvent?: (event: TrainLogLine) => void;
  // Non-404 transport/parse errors only; paging continues and the next tick recovers.
  onError?: (err: unknown) => void;
}

const DEFAULT_INTERVAL_MS = 1_000;
// Covers a 1000-epoch run's worst case with headroom; daemon caps page size at 1000.
const DEFAULT_PAGE_LIMIT = 500;

function sleep(ms: number): Promise<void> {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

export class TrainingLogTail {
  private workspaceId: Uuid | null = null;
  private jobId: Uuid | null = null;
  // Exclusive cursor (`seq > afterSeq`); daemon echoes `next_after_seq` unchanged on an empty page, so no-new-events == `next_after_seq === afterSeq`.
  private afterSeq = 0;
  private intervalMs = DEFAULT_INTERVAL_MS;
  private pageLimit = DEFAULT_PAGE_LIMIT;
  private opts: TrainingLogTailOptions = {};
  private timer: ReturnType<typeof setTimeout> | null = null;
  private fetching = false;
  private visibilityHandler: (() => void) | null = null;

  start(workspaceId: Uuid, jobId: Uuid, opts: TrainingLogTailOptions = {}): void {
    this.stop();
    this.workspaceId = workspaceId;
    this.jobId = jobId;
    this.afterSeq = 0;
    this.intervalMs = opts.intervalMs ?? DEFAULT_INTERVAL_MS;
    this.pageLimit = opts.pageLimit ?? DEFAULT_PAGE_LIMIT;
    this.opts = opts;

    if (typeof document !== 'undefined') {
      this.visibilityHandler = (): void => {
        if (document.hidden) return;
        // Tick immediately on re-show so a long background pause doesn't leave stale logs for a full interval.
        this.cancelTimer();
        void this.tick();
      };
      document.addEventListener('visibilitychange', this.visibilityHandler);
    }

    // Immediate first tick so a `start()` right after `POST /train` surfaces initial events within one round-trip.
    void this.tick();
  }

  stop(): void {
    this.workspaceId = null;
    this.jobId = null;
    this.cancelTimer();
    if (this.visibilityHandler && typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this.visibilityHandler);
    }
    this.visibilityHandler = null;
  }

  get running(): boolean {
    return this.workspaceId !== null && this.jobId !== null;
  }

  // Force-drain to the daemon tail so resolution guarantees the terminal JSONL line reached `onEvent`. The
  // daemon's polled `state` flips terminal BEFORE its buffered JSONL writer flushes, so stopping on the first
  // empty tick would drop the tail; instead retry empty ticks for `stableThreshold` attempts spaced by
  // `tickDelayMs`, bounded by `maxWaitMs`. No-op if not running; re-entrant with the periodic timer via `fetching`.
  async drain(
    opts: {
      maxWaitMs?: number;
      stableThreshold?: number;
      tickDelayMs?: number;
    } = {}
  ): Promise<void> {
    const maxWait = opts.maxWaitMs ?? 3_000;
    const stableTarget = opts.stableThreshold ?? 4;
    const tickDelay = opts.tickDelayMs ?? 150;
    const startedAt = Date.now();
    let stable = 0;
    // Hard cap against a zero-delay zero-progress spin.
    const MAX_ITERS = 200;
    for (let i = 0; i < MAX_ITERS; i++) {
      const wsId = this.workspaceId;
      const jobId = this.jobId;
      if (wsId === null || jobId === null) return;
      if (Date.now() - startedAt > maxWait) return;
      const before = this.afterSeq;
      const ran = await this.tickOnce();
      if (this.workspaceId !== wsId || this.jobId !== jobId) return;
      if (!ran) {
        // Skipped (fetch in flight), NOT an empty page; retry without advancing `stable`, else drain could resolve before the terminal event lands.
        await sleep(tickDelay);
        continue;
      }
      if (this.afterSeq > before) {
        // Progress: keep paging with no delay to catch a fast-emitting run.
        stable = 0;
        continue;
      }
      stable += 1;
      if (stable >= stableTarget) return;
      await sleep(tickDelay);
    }
  }

  private cancelTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private scheduleNextTick(): void {
    this.cancelTimer();
    if (this.workspaceId === null || this.jobId === null) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.tick();
    }, this.intervalMs);
  }

  // Skips the fetch while the tab is hidden, then reschedules; `drain()` bypasses this gate by calling `tickOnce` directly.
  private async tick(): Promise<void> {
    if (typeof document !== 'undefined' && document.hidden) {
      this.scheduleNextTick();
      return;
    }
    await this.tickOnce();
    if (this.workspaceId !== null && this.jobId !== null) {
      this.scheduleNextTick();
    }
  }

  // `true` if a fetch ran (even empty), `false` if skipped (no job bound, or fetch in flight); `drain` relies on this so a skip isn't miscounted as a stable empty page.
  private async tickOnce(): Promise<boolean> {
    const wsId = this.workspaceId;
    const jobId = this.jobId;
    if (wsId === null || jobId === null) return false;
    if (this.fetching) return false;
    this.fetching = true;
    try {
      const page = await trainingApi.readLogPage(wsId, jobId, {
        afterSeq: this.afterSeq,
        limit: this.pageLimit
      });
      // Job swapped mid-GET: drop the response so old-job events don't leak into the new consumer.
      if (this.workspaceId !== wsId || this.jobId !== jobId) return true;
      for (const evt of page.events) {
        // Trusts the backend schema: the daemon silently drops malformed JSONL lines, and a bad response envelope would already have thrown in `readLogPage`; consumers narrow on `kind`.
        this.opts.onEvent?.(evt as unknown as TrainLogLine);
      }
      this.afterSeq = page.next_after_seq;
    } catch (e) {
      if (this.workspaceId !== wsId || this.jobId !== jobId) return true;
      if (isNotFound(e)) {
        // JSONL not created yet (early admission): retry next tick.
        return true;
      }
      this.opts.onError?.(e);
    } finally {
      this.fetching = false;
    }
    return true;
  }
}

// Re-exported so consumers can narrow tail output without a second import.
export type { LogEvent };
