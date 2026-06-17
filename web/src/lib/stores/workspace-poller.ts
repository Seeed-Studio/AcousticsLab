import { workspaces as wsApi } from '$lib/api/endpoints';
import { categories } from '$lib/stores/categories.svelte';
import { slices } from '$lib/stores/slices.svelte';
import { isNotFound } from '$lib/utils/error-copy';
import type { Uuid, WorkspaceDetail } from '$lib/api/types';

// Per-workspace revision poller: invalidation only, never writes lists/entries directly. A revision
// advance flips per-category + workspace-wide stale bits (lazy refresh-on-expand reads them) and kicks
// a background reconcile that does the actual list/entry writes. Single-flight via a `fetching` flag
// (wsApi.get takes no AbortSignal; stale results drop via the workspaceId guard).

export interface WorkspacePollerOptions {
  // Daemon GET is cache-only (no asset walk), so the default 2s cadence is trivial load.
  intervalMs?: number;
  onDetail?: (detail: WorkspaceDetail) => void;
  // Poller self-stops on 404 so a forgotten teardown doesn't spam the daemon.
  onGone?: () => void;
  // Non-404 errors keep polling; a transient blip resolves next interval.
  onError?: (err: unknown) => void;
}

const DEFAULT_INTERVAL_MS = 2_000;

export class WorkspacePoller {
  private workspaceId: Uuid | null = null;
  private intervalMs = DEFAULT_INTERVAL_MS;
  private opts: WorkspacePollerOptions = {};
  private timer: ReturnType<typeof setTimeout> | null = null;
  private fetching = false;
  private visibilityHandler: (() => void) | null = null;

  // A second `start` tears down the prior poller so one instance can swap workspaces.
  start(detail: WorkspaceDetail, opts: WorkspacePollerOptions = {}): void {
    this.stop();
    this.workspaceId = detail.id;
    this.intervalMs = opts.intervalMs ?? DEFAULT_INTERVAL_MS;
    this.opts = opts;
    // Seed the loaded revision as baseline, else no-prior-receipts reads -1 and the first tick
    // always false-marks stale.
    slices.setRevisionAtLeast(detail.id, detail.workspace_revision.id);

    if (typeof document !== 'undefined') {
      this.visibilityHandler = (): void => {
        if (document.hidden) return;
        // Tick immediately on regaining visibility so a long background pause doesn't show stale
        // data for a full interval; the pending timer yields its slot.
        this.cancelTimer();
        void this.tick();
      };
      document.addEventListener('visibilitychange', this.visibilityHandler);
    }

    this.scheduleNextTick();
  }

  stop(): void {
    this.workspaceId = null;
    this.cancelTimer();
    // Reset single-flight: the old GET's finally reschedules only for its own workspace, so a swap
    // mid-flight would leave `fetching` stuck true and stall the new poller until next visibilitychange.
    this.fetching = false;
    if (this.visibilityHandler && typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this.visibilityHandler);
    }
    this.visibilityHandler = null;
  }

  private cancelTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private scheduleNextTick(): void {
    this.cancelTimer();
    if (this.workspaceId === null) return;
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.tick();
    }, this.intervalMs);
  }

  private async tick(): Promise<void> {
    const wsId = this.workspaceId;
    if (wsId === null) return;
    if (this.fetching) {
      // The outstanding fetch's finally reschedules.
      return;
    }
    if (typeof document !== 'undefined' && document.hidden) {
      this.scheduleNextTick();
      return;
    }
    if (slices.mutationsInFlightFor(wsId) > 0) {
      this.scheduleNextTick();
      return;
    }

    this.fetching = true;
    try {
      const detail = await wsApi.get(wsId);
      if (this.workspaceId !== wsId) return; // swapped/stopped mid-flight; drop stale response

      this.opts.onDetail?.(detail);

      // Re-check after the await: a write begun during the round-trip advances the daemon revision
      // before its receipt lands, so comparing now would false-positive against our own work.
      if (slices.mutationsInFlightFor(wsId) > 0) return;

      const incoming = detail.workspace_revision.id;
      // Feeds the live chip + slices' recursion-on-newer hook.
      slices.setRevisionAtLeast(wsId, incoming);
      // Compare against the last FULLY-RECONCILED revision, not the bound we just bumped: that would
      // swallow our own advance and mask a failed/blocked reconcile that must keep re-firing.
      const synced = slices.lastSyncedRevisionFor(wsId) ?? -1;
      if (incoming > synced) {
        // External advance: marking stale re-fires expanded panes' tracked reads and kicks a
        // background reconcile so the persisted sync record catches up to `incoming`.
        slices.markStaleForWorkspace(wsId, incoming);
        categories.markStale(wsId);
      }
    } catch (e) {
      if (this.workspaceId !== wsId) return;
      if (isNotFound(e)) {
        // Tear down before notifying: an in-place onGone that doesn't route away would otherwise
        // leave the visibility listener lingering.
        this.stop();
        this.opts.onGone?.();
        return;
      }
      this.opts.onError?.(e);
    } finally {
      // Clear the flag + reschedule only while still bound to this workspace: a stale fetch that
      // resolved after a stop()/swap must not clobber the new poller's in-flight `fetching` (stop()
      // already cleared it directly), which would let a concurrent tick defeat single-flight.
      if (this.workspaceId === wsId) {
        this.fetching = false;
        this.scheduleNextTick();
      }
    }
  }
}
