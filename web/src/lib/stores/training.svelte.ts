// Reactive training-job tracker: at most one active job (daemon enforces `max_train_jobs=1`; a
// second submit 409s `another_train_running`) plus a rolling per-workspace history of terminal jobs
// replayed from the durable `<ws>/training_logs/<job_id>.jsonl` backstop (eager + older disclosure).

import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import { training as trainingApi } from '$lib/api/endpoints';
import { isApiError } from '$lib/api/http';
import { enqueueDelete } from '$lib/api/delete-queue';
import { awaitJobTerminal } from '$lib/api/jobs';
import { TrainingSubscriber } from '$lib/api/training-subscriber';
import { TrainingLogTail } from '$lib/api/training-log-tail';
import { capFirst, errorCopy } from '$lib/utils/error-copy';
import { stageLabel, TERMINAL_TRAINING_STATES } from '$lib/components/training/labels';
import { m } from '$lib/i18n';
import { formatLabelsList } from '$lib/components/category/labels';
import { formatBytes } from '$lib/utils/format';
import type {
  EpochMetrics,
  JobProgress,
  JobState,
  LogEvent,
  Rfc3339,
  Stage,
  TrainLogLine,
  TrainingCfg,
  TrainingJobView,
  Uuid
} from '$lib/api/types';

export interface TrainingLogLine {
  at: Rfc3339;
  phase: TrainingJobView['progress']['phase'];
  message: string;
  // Monotonic daemon seq; dedups a replayed tail.
  seq: number;
  // Absent on synthetic seed lines.
  event?: TrainLogLine;
}

// Bounds re-render cost on a worst-case 1000-epoch run.
const MAX_LOG_LINES = 500;

// Pinned to the daemon's keep-last-N log retention so the in-memory surface matches on-disk; exported
// so retention-hint copy interpolates the same number. Bump in lockstep.
export const TRAINING_HISTORY_MAX_PER_WS = 10;
const MAX_HISTORY_PER_WS = TRAINING_HISTORY_MAX_PER_WS;

// Eager-card count; the component caps `[active, ...eagerHistory]` counting `active` as one slot (not
// additive), else the row count blinks N->N+1 across a Train run.
export const TRAINING_INITIAL_VISIBLE = 2;
const INITIAL_VISIBLE = TRAINING_INITIAL_VISIBLE;

// Per-click cap for "Load N more"/first-expand auto-load, bounding `Promise.all` parallelism so a
// slow eMMC backend doesn't stall the UI.
export const TRAINING_HISTORY_PAGE_SIZE = 5;
const PAGE_SIZE = TRAINING_HISTORY_PAGE_SIZE;

// Cumulative (not consecutive) cap on `event_gap` recoveries per job, stopping a pathological backfill
// loop (ring evicting faster than we catch up); the durable JSONL + final hydrate still reconcile.
const MAX_GAP_RETRIES = 3;

// Max JSONL events per hydration round-trip (one page, no pagination); longer runs truncate.
const HYDRATION_LOG_LIMIT = 1024;

interface DiscoveredRun {
  jobId: Uuid;
  // Approximates finish time (terminal events are the last writes); the sort key for newest-first.
  mtime: Rfc3339;
  sizeBytes: number;
}

// Inert localStorage namespace for a since-removed soft-hide list; cleared on hydration.
const LEGACY_HIDDEN_STORAGE_PREFIX = 'acoustics-lab:training-hidden:';

function clearLegacyHiddenStorage(workspaceId: Uuid): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.removeItem(`${LEGACY_HIDDEN_STORAGE_PREFIX}${workspaceId}`);
  } catch {
    /* best-effort */
  }
}

// An active job or a pinned terminal entry; same shape, `cancelling` meaningful only on the active slot.
export interface TrackedTrainingJob {
  workspaceId: Uuid;
  jobId: Uuid;
  // Daemon-preallocated at submit, pinned for the job's life; appears in detail's `heads[]` only on
  // a successful publish.
  headId: Uuid;
  view: TrainingJobView | null;
  epochs: EpochMetrics[];
  // Ordered by daemon `seq`.
  logLines: TrainingLogLine[];
  // True from `cancel()` until the `cancelled` terminal.
  cancelling: boolean;
}

class TrainingStore {
  // Single in-flight tracked job owned by this tab; null when none here. Another tab's train still
  // runs at the daemon (a submit 409 reveals it) but isn't owned here.
  active = $state<TrackedTrainingJob | null>(null);
  // Newest-first, capped at `MAX_HISTORY_PER_WS`. Survives `active` freeing so detail surfaces history
  // + heads without polling. Dropped on `forget`.
  private historyByWs = new SvelteMap<Uuid, TrackedTrainingJob[]>();
  // Counter (not the terminal job, so it re-fires on back-to-back terminals): detail's `$effect`
  // re-fetches on each bump so the heads list / revision pick up the published head.
  terminalSeq = $state(0);
  startError = $state<string | null>(null);
  starting = $state(false);

  // Run-time source of truth: per-job SSE stream updating `active` via three disjoint callbacks
  // (onEvent/onProgress/onStateTransition; onGap/onError don't assign it).
  private subscriber = new TrainingSubscriber();

  // Dormant JSONL paging substrate, bound only on `event_gap` recovery.
  private logTail = new TrainingLogTail();

  // Presence is the hydration idempotence guard; a failed listing leaves it unset to retry.
  private discoveredByWs = new SvelteMap<Uuid, DiscoveredRun[]>();

  // True from the listing call's start to the eager batch's settlement; the empty state shows only
  // after hydration completes with zero discovered.
  private hydratingByWs = new SvelteMap<Uuid, boolean>();

  private loadingMoreByWs = new SvelteMap<Uuid, boolean>();

  // In the store so it survives a TrainPane remount within the session. Reset on `forget`.
  private olderExpandedByWs = new SvelteMap<Uuid, boolean>();

  private deletingHistoryByWs = new SvelteMap<Uuid, SvelteSet<Uuid>>();

  // Per-workspace so workspace A's error doesn't show on B.
  private historyDeleteErrorByWs = new SvelteMap<Uuid, string | null>();

  // Refcount (not a boolean) so two overlapping rapid-delete refills both report `true` until BOTH
  // settle, keeping the placeholder visible across the whole shrink->backfill window.
  private autoRefillingByWs = new SvelteMap<Uuid, number>();

  // Older-tier batch size snapshotted at click-time as `min(loadable, PAGE_SIZE)` so
  // badge/skeleton/landed rows agree. A no-op load leaves 0.
  private olderLoadingPendingByWs = new SvelteMap<Uuid, number>();

  // In-flight `recover` Promises coalescing near-simultaneous callers (detail `load()` +
  // `TrainPane.onMount`) onto one round-trip; plain `Map` since no UI tracks it. The post-await
  // `active !== null` guard only catches the second caller if the first finished.
  private recoveringByWs = new Map<Uuid, Promise<void>>();

  historyFor(workspaceId: Uuid): readonly TrackedTrainingJob[] {
    return this.historyByWs.get(workspaceId) ?? [];
  }

  // History is newest-first, so index 0 is the most-recent terminal.
  terminalFor(workspaceId: Uuid): TrackedTrainingJob | null {
    const hist = this.historyByWs.get(workspaceId);
    return hist && hist.length > 0 ? hist[0] : null;
  }

  // Active slot iff bound to `workspaceId`; a sibling workspace's job is invisible.
  activeFor(workspaceId: Uuid): TrackedTrainingJob | null {
    if (this.active?.workspaceId === workspaceId) return this.active;
    return null;
  }

  // Re-throws on failure for an inline banner. Returns the preallocated head id (== eventual publish).
  async start(workspaceId: Uuid, cfg: TrainingCfg): Promise<Uuid> {
    if (this.active !== null) {
      // Defence-in-depth for callsites bypassing the form's `canSubmit` gate.
      throw new Error(m.error.another_train_running);
    }
    this.starting = true;
    this.startError = null;
    try {
      const resp = await trainingApi.start(workspaceId, cfg);
      this.active = {
        workspaceId,
        jobId: resp.job_id,
        headId: resp.head_id,
        view: null,
        epochs: [],
        // Synthetic seed for immediate feedback; `seq: -1` can't collide with a daemon seq (from 1).
        logLines: [
          {
            seq: -1,
            at: new Date().toISOString(),
            phase: 'prepare',
            message: m.training.store_log.seed_submitted
          }
        ],
        cancelling: false
      };
      // History intentionally not cleared on submit so the operator can scroll prior runs in flight.
      this.bindSubscriber(resp.job_id);
      return resp.head_id;
    } catch (e) {
      this.startError = errorCopy(e);
      throw e;
    } finally {
      this.starting = false;
    }
  }

  // Recover an in-flight job after a page reload: list jobs, find the (at most one) `running` entry,
  // bind SSE. No-op when the active slot is already bound; idempotent and coalesced.
  async recover(workspaceId: Uuid): Promise<void> {
    if (this.active !== null) return;
    const inflight = this.recoveringByWs.get(workspaceId);
    if (inflight) return inflight;
    const p = this.doRecover(workspaceId).finally(() => {
      this.recoveringByWs.delete(workspaceId);
    });
    this.recoveringByWs.set(workspaceId, p);
    return p;
  }

  private async doRecover(workspaceId: Uuid): Promise<void> {
    let jobs: TrainingJobView[] = [];
    try {
      jobs = await trainingApi.list(workspaceId);
    } catch {
      // Best-effort and silent: a transient failure may cost live progress until the next submit,
      // but the worker keeps going.
      return;
    }
    // Re-check after the await: a concurrent `start()` set `this.active`; overwriting would orphan
    // its freshly-bound subscriber.
    if (this.active !== null) return;
    // Most-recent running entry; the daemon sorts ascending by `started_at`, so newest is last.
    const running = jobs.filter((j) => j.state === 'running');
    if (running.length === 0) return;
    const view = running[running.length - 1];

    // `headId` empty (not on the JobView wire; active-run UI reads `view.result?.head_id`). `view:
    // null` not the polled snapshot - SSE replays from `seq=0`, so seeding would let `applyEventToView`
    // downgrade the phase (train->prepare->train) whereas null rebuilds flicker-free in tens of ms.
    // `at` is now (not `view.started_at`) so the seed line doesn't read as a backwards log ahead of SSE.
    this.active = {
      workspaceId,
      jobId: view.job_id,
      headId: '',
      view: null,
      epochs: [],
      logLines: [
        {
          seq: -1,
          at: new Date().toISOString(),
          phase: 'prepare',
          message: m.training.store_log.seed_recovered
        }
      ],
      cancelling: false
    };
    this.bindSubscriber(view.job_id);
  }

  // Cancel the tracked job, if any. Returns after the DELETE ack; the worker exit and `cancelled`
  // terminal land later via SSE. Subsequent clicks no-op via `cancelling`.
  async cancel(): Promise<void> {
    const job = this.active;
    if (!job || job.cancelling) return;
    // Spread-with-override fires reactivity once (new identity); mutate-then-reassign would fire
    // twice with a transient torn state.
    this.active = { ...job, cancelling: true };
    try {
      await trainingApi.cancel(job.workspaceId, job.jobId);
    } catch (e) {
      // Un-set so the operator can retry; no banner (cancel is a quiet escape hatch). Re-read and
      // patch only when still ours (the job may have gone terminal during the await). The runtime
      // null check is load-bearing: TS narrows `active` non-null pre-await and ignores re-assignment.
      // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
      if (this.active !== null && this.active.jobId === job.jobId) {
        this.active = { ...this.active, cancelling: false };
      }
      console.warn('[training] cancel failed', e);
      throw e;
    }
  }

  // History-row deletion is per-entry only (a batch fan-out collides with `max_delete_jobs=1`,
  // 409'ing all but one). The daemon also `JobConflict`s a log delete while a Train job is active;
  // the menu's `activeFor === null` gate races a producer start, so failures surface inline.

  historyDeletingForJob(workspaceId: Uuid, jobId: Uuid): boolean {
    return this.deletingHistoryByWs.get(workspaceId)?.has(jobId) ?? false;
  }

  historyDeleteErrorFor(workspaceId: Uuid): string | null {
    return this.historyDeleteErrorByWs.get(workspaceId) ?? null;
  }

  dismissHistoryDeleteError(workspaceId: Uuid): void {
    this.historyDeleteErrorByWs.set(workspaceId, null);
  }

  // Delete one terminal row's JSONL backstop, serialised through the global delete-queue so it can't
  // 409 against a concurrent dataset/converter/workspace delete. On failure local state is untouched
  // for retry and the banner shows the error. No-ops on the live active slot (the daemon would 409).
  async deleteHistoryEntry(workspaceId: Uuid, jobId: Uuid): Promise<void> {
    if (this.historyDeletingForJob(workspaceId, jobId)) return;
    if (this.active?.workspaceId === workspaceId && this.active.jobId === jobId) return;
    let set = this.deletingHistoryByWs.get(workspaceId);
    if (!set) {
      set = new SvelteSet<Uuid>();
      this.deletingHistoryByWs.set(workspaceId, set);
    }
    set.add(jobId);
    // Clear the previous error so the banner doesn't shadow a pending retry.
    this.historyDeleteErrorByWs.set(workspaceId, null);
    try {
      await enqueueDelete(async () => {
        const ack = await trainingApi.deleteLog(workspaceId, jobId);
        await awaitJobTerminal(ack.job_id, 'training-log delete');
      });
      this.removeFromHistory(workspaceId, jobId);
      // Pull the next discovered entry into the eager tier. Fire-and-forget; a failed refill leaves
      // the tier short, recovered next mount/older-expand.
      void this.refillEagerAfterDelete(workspaceId);
    } catch (e) {
      const message =
        e instanceof Error && e.message ? capFirst(e.message, 'Delete failed.') : errorCopy(e);
      this.historyDeleteErrorByWs.set(workspaceId, message);
      console.warn('[training] history delete failed', e);
      throw e;
    } finally {
      const cur = this.deletingHistoryByWs.get(workspaceId);
      if (cur) {
        cur.delete(jobId);
        if (cur.size === 0) this.deletingHistoryByWs.delete(workspaceId);
      }
    }
  }

  // Drop a jobId from both `historyByWs` and `discoveredByWs`; they must agree or the "Show N older
  // runs" badge re-counts the deleted entry.
  private removeFromHistory(workspaceId: Uuid, jobId: Uuid): void {
    const hist = this.historyByWs.get(workspaceId);
    if (hist) {
      const filtered = hist.filter((j) => j.jobId !== jobId);
      if (filtered.length !== hist.length) {
        if (filtered.length === 0) this.historyByWs.delete(workspaceId);
        else this.historyByWs.set(workspaceId, filtered);
      }
    }
    this.dropFromDiscovered(workspaceId, jobId);
  }

  // Refill the eager tier after a delete left it under budget. `autoRefillingByWs` brackets the
  // placeholder to the post-removal-through-push window (avoids a shrink->re-grow judder); gating on
  // it (not `loadingMoreByWs`) lets a delete + immediate "Load N more" run in parallel, the rare
  // double-fetch made benign by dedup.
  private async refillEagerAfterDelete(workspaceId: Uuid): Promise<void> {
    const hist = this.historyByWs.get(workspaceId) ?? [];
    if (hist.length >= INITIAL_VISIBLE) return;
    const gap = INITIAL_VISIBLE - hist.length;
    if (this.loadableOlderCountFor(workspaceId) === 0) return;
    const prev = this.autoRefillingByWs.get(workspaceId) ?? 0;
    this.autoRefillingByWs.set(workspaceId, prev + 1);
    try {
      await this.loadBatch(workspaceId, gap);
    } catch (e) {
      console.warn('[training] eager-tier auto-refill after delete failed', e);
    } finally {
      const cur = this.autoRefillingByWs.get(workspaceId) ?? 0;
      if (cur <= 1) this.autoRefillingByWs.delete(workspaceId);
      else this.autoRefillingByWs.set(workspaceId, cur - 1);
    }
  }

  // Set the "older runs" disclosure state. On expand, fire-and-forget refresh + load so it operates
  // on the backend's current keep-last-N (mount-time discovery drifts as the producer prunes;
  // cross-tab/external prunes are unobservable otherwise).
  setOlderExpanded(workspaceId: Uuid, expanded: boolean): void {
    this.olderExpandedByWs.set(workspaceId, expanded);
    if (expanded) {
      void this.handleOlderExpand(workspaceId);
    }
  }

  // Older-runs expand: refresh the listing (honest badge), then auto-load the first PAGE_SIZE iff
  // the older TIER (not total history.length, so it fires on fresh mount AND after a train pushed an
  // eager card down) holds fewer than one batch. `loadingMoreByWs` is held across BOTH phases so the
  // pending affordance covers the re-list (else a slow listing expands to empty then flips loading);
  // `loadBatch` is called directly to dodge the re-entry guard on the flag just set.
  private async handleOlderExpand(workspaceId: Uuid): Promise<void> {
    if (this.loadingMoreByWs.get(workspaceId)) return;
    this.loadingMoreByWs.set(workspaceId, true);
    // Optimistic pre-refresh skeleton count so the disclosure mounts with placeholders in the click
    // frame, not empty space for the refresh round-trip; gated on the same tier-vs-budget check the
    // post-refresh branch uses, which then wins if retention pruned meanwhile.
    if (this.olderHistoryFor(workspaceId).length < PAGE_SIZE) {
      const initial = Math.min(this.loadableOlderCountFor(workspaceId), PAGE_SIZE);
      if (initial > 0) this.olderLoadingPendingByWs.set(workspaceId, initial);
    }
    try {
      await this.refreshDiscovery(workspaceId);
      const older = this.olderHistoryFor(workspaceId);
      if (older.length < PAGE_SIZE) {
        // Authoritative count `loadBatch` will surface; `delete` on zero collapses the skeleton tier
        // same-tick if discovery is exhausted.
        const pending = Math.min(this.loadableOlderCountFor(workspaceId), PAGE_SIZE);
        if (pending > 0) this.olderLoadingPendingByWs.set(workspaceId, pending);
        else this.olderLoadingPendingByWs.delete(workspaceId);
        await this.loadBatch(workspaceId, PAGE_SIZE);
      }
    } catch (e) {
      // `setOlderExpanded` voids this Promise, so a rejection would be unhandled. `loadBatch` can't
      // reject today (fetches swallow and return null), but guard a future error-surfacing refactor.
      console.warn('[training] older-expand load failed', e);
    } finally {
      this.olderLoadingPendingByWs.delete(workspaceId);
      // Re-write only if the entry still exists: a racing `forget` deletes it, and an unconditional
      // `.set` would re-insert a stale `{ws -> false}` nothing reaps.
      if (this.loadingMoreByWs.has(workspaceId)) this.loadingMoreByWs.set(workspaceId, false);
    }
  }

  discoveredFor(workspaceId: Uuid): readonly DiscoveredRun[] {
    return this.discoveredByWs.get(workspaceId) ?? [];
  }

  hydratingFor(workspaceId: Uuid): boolean {
    return this.hydratingByWs.get(workspaceId) ?? false;
  }

  // With `eagerSkeletonCountFor`, drives the placeholder reserving the emptied slot from
  // post-removal through push (a single in-place card swap vs a shrink->re-grow judder).
  autoRefillingFor(workspaceId: Uuid): boolean {
    return (this.autoRefillingByWs.get(workspaceId) ?? 0) > 0;
  }

  // Eager `<ul>` skeleton count: the gap to `INITIAL_VISIBLE` whenever a load is incoming (initial
  // hydration OR a post-delete refill), else 0. Unified so the renderer stays one `{#each}` (the
  // empty-then-2-skeletons case falls out as history.length===0).
  eagerSkeletonCountFor(workspaceId: Uuid): number {
    if (!this.hydratingFor(workspaceId) && !this.autoRefillingFor(workspaceId)) return 0;
    const hist = this.historyByWs.get(workspaceId) ?? [];
    return Math.max(0, INITIAL_VISIBLE - hist.length);
  }

  // Older-tier `<ul>` skeleton count, snapshotted at click-time to exactly the rows the in-flight
  // `loadBatch` will surface; zero when no load is pending.
  olderSkeletonCountFor(workspaceId: Uuid): number {
    return this.olderLoadingPendingByWs.get(workspaceId) ?? 0;
  }

  loadingMoreFor(workspaceId: Uuid): boolean {
    return this.loadingMoreByWs.get(workspaceId) ?? false;
  }

  olderExpandedFor(workspaceId: Uuid): boolean {
    return this.olderExpandedByWs.get(workspaceId) ?? false;
  }

  eagerHistoryFor(workspaceId: Uuid): readonly TrackedTrainingJob[] {
    const hist = this.historyByWs.get(workspaceId) ?? [];
    if (hist.length <= INITIAL_VISIBLE) return hist;
    return hist.slice(0, INITIAL_VISIBLE);
  }

  olderHistoryFor(workspaceId: Uuid): readonly TrackedTrainingJob[] {
    const hist = this.historyByWs.get(workspaceId) ?? [];
    if (hist.length <= INITIAL_VISIBLE) return [];
    return hist.slice(INITIAL_VISIBLE);
  }

  // Older-tier entries still revealable (discovered but not yet loaded); drives "Load N more".
  // Clamped at remaining capacity so a stale mount snapshot can't inflate the badge with pruned ids
  // that never load (`fetchAndReplay`'s 404 handler also prunes them, converging).
  loadableOlderCountFor(workspaceId: Uuid): number {
    const discovered = this.discoveredByWs.get(workspaceId);
    if (!discovered) return 0;
    const history = this.historyByWs.get(workspaceId) ?? [];
    const loadedIds = new Set<Uuid>(history.map((j) => j.jobId));
    let n = 0;
    for (const r of discovered) {
      if (loadedIds.has(r.jobId)) continue;
      n++;
    }
    const remainingCapacity = Math.max(0, MAX_HISTORY_PER_WS - history.length);
    return Math.min(n, remainingCapacity);
  }

  // Idempotent (returns once discovery is populated; a failed listing leaves it unset to retry).
  // Mount fires this alongside the independent `recover()`; `fetchAndReplay`'s active-slot guard keeps
  // the active jobId out of history if it appears in both.
  async hydrateHistory(workspaceId: Uuid): Promise<void> {
    if (this.discoveredByWs.has(workspaceId)) return;
    if (this.hydratingByWs.get(workspaceId)) return;
    this.hydratingByWs.set(workspaceId, true);
    clearLegacyHiddenStorage(workspaceId);
    try {
      const discovered = await this.fetchDiscoveryListing(workspaceId);
      this.discoveredByWs.set(workspaceId, discovered);
      await this.loadBatch(workspaceId, INITIAL_VISIBLE);
    } catch (e) {
      // Best-effort; next mount retries. No banner: no-past-runs and a transient hiccup look
      // identical, so a banner would be noisy on cold start.
      console.warn('[training] hydrate failed', e);
    } finally {
      this.hydratingByWs.set(workspaceId, false);
    }
  }

  // List `training_logs/` into a sorted, capped `DiscoveredRun[]`; throws on listing error. `limit:
  // 100` over-fetches keep-last-N because the server sorts by jobId not mtime, so over-fetch
  // guarantees the mtime-newest after the client sort and covers a workspace straddling an upgrade.
  private async fetchDiscoveryListing(workspaceId: Uuid): Promise<DiscoveredRun[]> {
    const listing = await trainingApi.listLogs(workspaceId, { limit: 100 });
    const discovered: DiscoveredRun[] = [];
    for (const entry of listing.entries) {
      if (entry.kind !== 'file') continue;
      if (!entry.name.endsWith('.jsonl')) continue;
      const jobId = entry.name.slice(0, -'.jsonl'.length);
      if (jobId.length === 0) continue;
      // Trust boundary: the stem flows untouched into the row key and delete target, so only
      // canonical-UUID stems become jobIds (else a stray `.jsonl` renders a phantom row whose Delete
      // points at an arbitrary path component).
      if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(jobId)) continue;
      discovered.push({
        jobId,
        mtime: entry.mtime,
        sizeBytes: entry.size_bytes ?? 0
      });
    }
    // Newest-first by mtime, tie-broken by jobId (filesystems give sub-second-equal mtimes on fast
    // back-to-back runs); return-0 unreachable (paths unique) but required by `Array.sort`'s contract.
    discovered.sort((a, b) => {
      if (a.mtime > b.mtime) return -1;
      if (a.mtime < b.mtime) return 1;
      if (a.jobId < b.jobId) return -1;
      if (a.jobId > b.jobId) return 1;
      return 0;
    });
    // Clamp so discovery never overstates the loadable count (an upgrade-straddle or pre-sweep race
    // can briefly list more than the backend keeps), stopping `loadBatch` fetching soon-evicted entries.
    return discovered.slice(0, MAX_HISTORY_PER_WS);
  }

  // Re-list and replace discovery so the disclosure operates on the backend's current keep-last-N
  // (including pruning since mount). Best-effort: a failure leaves discovery in place.
  private async refreshDiscovery(workspaceId: Uuid): Promise<void> {
    try {
      const discovered = await this.fetchDiscoveryListing(workspaceId);
      this.discoveredByWs.set(workspaceId, discovered);
    } catch (e) {
      console.warn('[training] refresh discovery failed', { workspaceId, error: e });
    }
  }

  // Reveal up to PAGE_SIZE older-tier runs (explicit "Load N more"). Bounded per-click so
  // `Promise.all` never fans out wider than PAGE_SIZE (a wider fan-out was laggy on eMMC);
  // serialised via `loadingMoreByWs` so a double-click can't fire two batches.
  async loadMoreHistory(workspaceId: Uuid): Promise<void> {
    if (!this.discoveredByWs.has(workspaceId)) return;
    if (this.loadingMoreByWs.get(workspaceId)) return;
    this.loadingMoreByWs.set(workspaceId, true);
    // Snapshot batch size BEFORE the fetch - the same `min(loadable, PAGE_SIZE)` cap `loadBatch`
    // applies, so placeholders line up one-to-one with the rows that land. Zero skips.
    const pending = Math.min(this.loadableOlderCountFor(workspaceId), PAGE_SIZE);
    if (pending > 0) this.olderLoadingPendingByWs.set(workspaceId, pending);
    try {
      await this.loadBatch(workspaceId, PAGE_SIZE);
    } finally {
      // Forget-safe: skip the re-write if a racing `forget` deleted the entry.
      if (this.loadingMoreByWs.has(workspaceId)) this.loadingMoreByWs.set(workspaceId, false);
      this.olderLoadingPendingByWs.delete(workspaceId);
    }
  }

  // Walk discovery, skip already-loaded entries, fetch up to `targetAdd` JSONLs in parallel, push
  // into history. Shared by the eager and lazy paths.
  private async loadBatch(workspaceId: Uuid, targetAdd: number): Promise<void> {
    if (targetAdd <= 0) return;
    const discovered = this.discoveredByWs.get(workspaceId);
    if (!discovered || discovered.length === 0) return;
    const loadedIds = new Set<Uuid>((this.historyByWs.get(workspaceId) ?? []).map((j) => j.jobId));
    const toFetch: DiscoveredRun[] = [];
    for (const run of discovered) {
      if (toFetch.length >= targetAdd) break;
      if (loadedIds.has(run.jobId)) continue;
      // Skip the active slot (recover() owns it); `active` may bind between here and resolution, so
      // `fetchAndReplay` re-checks at push time too.
      if (this.active?.jobId === run.jobId) continue;
      toFetch.push(run);
    }
    if (toFetch.length === 0) return;
    const fetched = await Promise.all(
      toFetch.map((run) => this.fetchAndReplay(workspaceId, run.jobId))
    );
    const terminals = fetched.filter((j): j is TrackedTrainingJob => j !== null);
    if (terminals.length > 0) this.pushHistoryBatch(workspaceId, terminals);
  }

  // Fetch one JSONL page and replay it into a `TrackedTrainingJob`. Returns `null` when: unreadable;
  // 404 (retention swept it - pruned here so the loadable-count and retry loop converge); replay
  // produced no view; still in-flight (recover() owns it); or the jobId is the active slot
  // (double-tracking renders two cards).
  private async fetchAndReplay(workspaceId: Uuid, jobId: Uuid): Promise<TrackedTrainingJob | null> {
    if (this.active?.jobId === jobId) return null;
    let page;
    try {
      page = await trainingApi.readLogPage(workspaceId, jobId, {
        afterSeq: 0,
        limit: HYDRATION_LOG_LIMIT
      });
    } catch (e) {
      if (isApiError(e) && e.status === 404) {
        // 404 = retention swept it; prune. Transient 5xx/network errors leave it for retry.
        this.dropFromDiscovered(workspaceId, jobId);
      } else {
        console.warn('[training] replay fetch failed', { workspaceId, jobId, error: e });
      }
      return null;
    }
    // Re-check active: a fast-finishing run could have terminated and re-pushed via SSE during the
    // round-trip, making the session-pushed entry authoritative.
    if (this.active?.jobId === jobId) return null;
    if (page.events.length === 0) return null;
    return replayJsonl(workspaceId, jobId, page.events);
  }

  private dropFromDiscovered(workspaceId: Uuid, jobId: Uuid): void {
    const prev = this.discoveredByWs.get(workspaceId);
    if (!prev) return;
    const next = prev.filter((r) => r.jobId !== jobId);
    if (next.length === prev.length) return;
    this.discoveredByWs.set(workspaceId, next);
  }

  // Clear all local state for `workspaceId` (e.g. workspace deleted): stop the subscriber + any
  // in-flight gap backfill, free the slot if ours, then clear bookkeeping so a re-mount re-hydrates
  // fresh. No daemon mutation - the workspace delete flow already cancels the train job.
  forget(workspaceId: Uuid): void {
    if (this.active?.workspaceId === workspaceId) {
      this.subscriber.stop();
      this.logTail.stop();
      this.active = null;
    }
    this.historyByWs.delete(workspaceId);
    this.discoveredByWs.delete(workspaceId);
    this.hydratingByWs.delete(workspaceId);
    this.loadingMoreByWs.delete(workspaceId);
    this.olderExpandedByWs.delete(workspaceId);
    // Safe to drop: the in-flight delete pipeline's `finally` short-circuits on undefined `.get`, and
    // the failure banner has no surface once the workspace is gone.
    this.deletingHistoryByWs.delete(workspaceId);
    this.historyDeleteErrorByWs.delete(workspaceId);
    // In-flight loads can't be aborted (they resolve into no-ops/404s); clearing these just stops
    // stale skeletons on a same-id remount.
    this.autoRefillingByWs.delete(workspaceId);
    this.olderLoadingPendingByWs.delete(workspaceId);
    // Drop the coalesced recover Promise so a same-id remount fires fresh instead of coalescing onto it.
    this.recoveringByWs.delete(workspaceId);
    clearLegacyHiddenStorage(workspaceId);
  }

  // Bind SSE from `afterSeq` (0 fresh; post-backfill cursor on a `recoverFromGap` rebind);
  // idempotent since `start()` calls `stop()` first. `onEvent` also fires `handleTerminal` as a
  // belt-and-braces so a stream dropping the state transition still routes the job into history.
  private bindSubscriber(jobId: Uuid, afterSeq = 0, gapRetries = 0): void {
    this.subscriber.start(jobId, {
      afterSeq,
      onEvent: (event) => {
        this.ingestLogEvent(jobId, event);
      },
      onProgress: (progress, at, seq) => {
        const cur = this.active;
        if (cur?.jobId !== jobId) return;
        // Drop progress ticks at/below the high-water seq: an EventSource reconnect replays from
        // `after_seq=0` and would otherwise walk the bar backward (N->1->N). The `>= 0` floor exempts
        // negative synthetic seeds from gating real daemon ticks.
        const maxSeq = cur.logLines.reduce((m, l) => (l.seq > m ? l.seq : m), -1);
        if (seq >= 0 && seq <= maxSeq) return;
        this.active = {
          ...cur,
          view: applyProgressToView(cur.view, progress, at, jobId, cur.workspaceId)
        };
      },
      onStateTransition: (state, at) => {
        const cur = this.active;
        if (cur?.jobId !== jobId) return;
        const nextView = applyStateToView(cur.view, state, at, jobId, cur.workspaceId);
        this.active = { ...cur, view: nextView };
        if (nextView.state !== 'running') {
          this.handleTerminal(cur.workspaceId, jobId, nextView);
        }
      },
      onGap: ({ latest_seq }) => {
        if (gapRetries >= MAX_GAP_RETRIES) {
          console.warn(
            '[training-subscriber] gap-recovery retries exhausted; UI may be incomplete'
          );
          return;
        }
        void this.recoverFromGap(jobId, latest_seq, gapRetries + 1);
      },
      onError: (reason) => {
        // Transient errors auto-reconnect (seq dedup catches the replay); this fires only for
        // unexpected permanent closes not classifiable as a gap or 404.
        console.warn('[training-subscriber]', reason);
      }
    });
  }

  // Called from `onGap` when the ring evicted events older than `after_seq`: stop the failed
  // subscriber so its retries don't compete with paging, page JSONL from seq=0 (`ingestLogEvent`
  // dedups), then re-bind at the highest seq we now hold. No-op if the active slot shifted - the
  // backfill can itself deliver the terminal, nulling `active` so the rebind guard catches it.
  private async recoverFromGap(jobId: Uuid, latest: number, gapRetries: number): Promise<void> {
    const cur = this.active;
    if (cur?.jobId !== jobId) return;
    this.subscriber.stop();
    this.logTail.start(cur.workspaceId, jobId, {
      onEvent: (event) => {
        this.ingestLogEvent(jobId, event);
      },
      onError: (err) => {
        console.warn('[training-log-tail] gap backfill error', err);
      }
    });
    try {
      await this.logTail.drain();
    } catch (e) {
      console.warn('[training-log-tail] gap backfill drain failed', e);
    }
    this.logTail.stop();
    const still = this.active;
    if (still?.jobId !== jobId) return;
    const lastSeq = still.logLines.reduce((m, l) => (l.seq > m ? l.seq : m), Math.max(0, latest));
    this.bindSubscriber(jobId, lastSeq, gapRetries);
  }

  // Apply one `TrainEvent` to the active slot: render a log line (unknown kinds drop), merge an
  // epoch on `epoch_completed`, update view fields, and fire the belt-and-braces terminal trigger.
  // Deduped on `seq`; no-op when the active slot shifted mid-await.
  private ingestLogEvent(jobId: Uuid, event: TrainLogLine): void {
    const cur = this.active;
    if (cur?.jobId !== jobId) return;
    // Dedup against ANY absorbed seq, not just the last: out-of-order arrival (an SSE rebind
    // interleaving an undrained backfill page) needs the full walk. The `>= 0` gate exempts the
    // negative synthetic seeds (never collide with daemon seqs from 1).
    if (event.seq >= 0 && cur.logLines.some((l) => l.seq === event.seq)) return;
    const rendered = renderEvent(event);
    const nextEpochs =
      event.kind === 'epoch_completed' ? mergeEpochFromEvent(cur.epochs, event) : cur.epochs;
    const nextLogLines =
      rendered === null
        ? cur.logLines
        : capLog([
            ...cur.logLines,
            {
              seq: event.seq,
              at: event.at,
              phase: rendered.phase,
              message: rendered.message,
              event
            }
          ]);
    const nextView = applyEventToView(cur.view, event, jobId, cur.workspaceId);
    // Capture head_id off the wire because `recover()` seeds `''`: `job_submitted` populates it,
    // `head_published` re-asserts at publish (replay path captures identically).
    const nextHeadId =
      event.kind === 'job_submitted' || event.kind === 'head_published'
        ? event.head_id
        : cur.headId;
    this.active = {
      ...cur,
      headId: nextHeadId,
      view: nextView,
      epochs: nextEpochs,
      logLines: nextLogLines
    };
    // Belt-and-braces terminal trigger; idempotent with `onStateTransition` via the jobId guard.
    if (
      event.kind === 'job_completed' ||
      event.kind === 'job_failed' ||
      event.kind === 'job_cancelled'
    ) {
      this.handleTerminal(cur.workspaceId, jobId, nextView);
    }
  }

  // Freeze the active slot into history. The jobId guard no-ops every call after the first (the
  // terminal trigger fires from both the typed-event and state-transition paths) and runs BEFORE the
  // subscriber/tail teardown so a stale-jobId call can't rip live transports from a newly-bound slot.
  private handleTerminal(workspaceId: Uuid, jobId: Uuid, view: TrainingJobView): void {
    const cur = this.active;
    if (cur?.jobId !== jobId) return;
    this.subscriber.stop();
    this.logTail.stop();
    const terminal: TrackedTrainingJob = {
      ...cur,
      view,
      cancelling: false
    };
    this.pushHistory(workspaceId, terminal);
    this.active = null;
    this.terminalSeq = this.terminalSeq + 1;
  }

  private pushHistory(workspaceId: Uuid, terminal: TrackedTrainingJob): void {
    this.pushHistoryBatch(workspaceId, [terminal]);
  }

  // Merge terminal jobs into history, deduped on jobId (incoming wins) and capped at
  // `MAX_HISTORY_PER_WS`. Sorting newest-first by `view.started_at` (vs prepend-only) is what makes
  // hydration correct - a hydrated older entry lands at its chronological position while a fresh
  // session terminal sorts to index 0.
  private pushHistoryBatch(workspaceId: Uuid, terminals: readonly TrackedTrainingJob[]): void {
    if (terminals.length === 0) return;
    const prev = this.historyByWs.get(workspaceId) ?? [];
    const incomingIds = new Set(terminals.map((t) => t.jobId));
    const filtered = prev.filter((j) => !incomingIds.has(j.jobId));
    const merged = [...filtered, ...terminals];
    // return-0 path unreachable after dedup but required by `Array.sort`'s non-zero-on-equal contract.
    merged.sort((a, b) => {
      const aT = a.view?.started_at ?? '';
      const bT = b.view?.started_at ?? '';
      if (aT > bT) return -1;
      if (aT < bT) return 1;
      if (a.jobId < b.jobId) return -1;
      if (a.jobId > b.jobId) return 1;
      return 0;
    });
    this.historyByWs.set(workspaceId, merged.slice(0, MAX_HISTORY_PER_WS));
  }
}

// Seed view when `view = null`; `started_at` is the first event's timestamp (typically
// `job_submitted`, closest to a true start); `result`/`error`/`finished_at` absent until terminal.
function initialView(jobId: Uuid, workspaceId: Uuid, startedAt: Rfc3339): TrainingJobView {
  return {
    job_id: jobId,
    workspace_id: workspaceId,
    state: 'running',
    progress: { phase: 'prepare', current: 0, total: 0, message: '' },
    started_at: startedAt
  };
}

// Apply one `TrainEvent` to the view - the only path surfacing phase + per-epoch metrics, since the
// cross-cutting progress field carries only flat `{done, total}`.
function applyEventToView(
  view: TrainingJobView | null,
  event: TrainLogLine,
  jobId: Uuid,
  workspaceId: Uuid
): TrainingJobView {
  const base: TrainingJobView = view ?? initialView(jobId, workspaceId, event.at);
  switch (event.kind) {
    case 'job_submitted':
      return {
        ...base,
        state: 'running',
        started_at: event.at,
        progress: { ...base.progress, phase: 'prepare' }
      };
    case 'phase_started':
      return { ...base, progress: { ...base.progress, phase: event.phase } };
    case 'epoch_completed': {
      const metrics: EpochMetrics = {
        epoch: event.epoch,
        epochs: event.epochs,
        train_loss: event.train_loss,
        train_acc: event.train_acc,
        val_acc: event.val_acc,
        best_val_acc: event.best_val_acc,
        val_loss: event.val_loss,
        best_val_loss: event.best_val_loss
      };
      return {
        ...base,
        progress: {
          ...base.progress,
          current: event.epoch,
          total: event.epochs,
          metrics
        }
      };
    }
    case 'job_completed':
      return {
        ...base,
        state: 'completed',
        finished_at: event.at,
        result: event.result
      };
    case 'job_failed':
      return {
        ...base,
        state: 'failed',
        finished_at: event.at,
        error: event.error
      };
    case 'job_cancelled':
      return {
        ...base,
        state: 'cancelled',
        finished_at: event.at
      };
    // Informational kinds (job_running, dataset_scanned, etc.) have no view-field effect.
    default:
      return base;
  }
}

// Apply a rate-limited progress tick to `view.progress.current/total` (phase + metrics flow only via
// typed events); `total` falls back to the prior value before the producer learns the work-set size.
function applyProgressToView(
  view: TrainingJobView | null,
  progress: JobProgress,
  at: Rfc3339,
  jobId: Uuid,
  workspaceId: Uuid
): TrainingJobView {
  const base: TrainingJobView = view ?? initialView(jobId, workspaceId, at);
  return {
    ...base,
    progress: {
      ...base.progress,
      current: progress.done,
      total: progress.total ?? base.progress.total
    }
  };
}

// Apply a `JobState` transition (fires only on terminal; pre-terminal is implicitly 'running'),
// mapping the cross-cutting enum to the training enum: `succeeded`->`completed`, `queued|running`->
// `running` (training has no queue).
function applyStateToView(
  view: TrainingJobView | null,
  state: JobState,
  at: Rfc3339,
  jobId: Uuid,
  workspaceId: Uuid
): TrainingJobView {
  const base: TrainingJobView = view ?? initialView(jobId, workspaceId, at);
  let trainState: TrainingJobView['state'];
  switch (state) {
    case 'succeeded':
      trainState = 'completed';
      break;
    case 'failed':
      trainState = 'failed';
      break;
    case 'cancelled':
      trainState = 'cancelled';
      break;
    case 'queued':
    case 'running':
    default:
      trainState = 'running';
      break;
  }
  const isTerminal =
    trainState === 'completed' || trainState === 'failed' || trainState === 'cancelled';
  return {
    ...base,
    state: trainState,
    finished_at: isTerminal ? at : base.finished_at
  };
}

// Cap in the store (not the renderer) so the footprint stays bounded even if a consumer never paginates.
function capLog(lines: TrainingLogLine[]): TrainingLogLine[] {
  if (lines.length <= MAX_LOG_LINES) return lines;
  return lines.slice(lines.length - MAX_LOG_LINES);
}

// Merge one `epoch_completed` into the per-epoch list: same `epoch` -> keep newest; out-of-order ->
// replace slot; unobserved -> append. Returns a new array only when content changed, so reactivity
// fires only on a real update.
function mergeEpochFromEvent(
  prev: EpochMetrics[],
  event: Extract<TrainLogLine, { kind: 'epoch_completed' }>
): EpochMetrics[] {
  const m: EpochMetrics = {
    epoch: event.epoch,
    epochs: event.epochs,
    train_loss: event.train_loss,
    train_acc: event.train_acc,
    val_acc: event.val_acc,
    best_val_acc: event.best_val_acc,
    val_loss: event.val_loss,
    best_val_loss: event.best_val_loss
  };
  if (m.epoch === 0) return prev; // epochs are 1-indexed
  const last = prev.length > 0 ? prev[prev.length - 1] : null;
  if (last?.epoch === m.epoch) {
    // Replace iff `best_val_acc` advanced, else return the same reference so reactivity doesn't fire.
    const cur = m.best_val_acc;
    const ref = last.best_val_acc;
    if (cur !== null && ref !== null && cur > ref) {
      return [...prev.slice(0, -1), m];
    }
    return prev;
  }
  if (last !== null && last.epoch > m.epoch) {
    const idx = prev.findIndex((e) => e.epoch === m.epoch);
    if (idx >= 0) {
      const next = prev.slice();
      next[idx] = m;
      return next;
    }
    return prev;
  }
  return [...prev, m];
}

// Render one `TrainEvent` to an operator-facing log line; `null` drops it from scrollback. The phase
// column hard-codes each kind's natural phase mirroring the daemon's emission order, so a producer
// emitting a kind in a different phase must update it here.
function renderEvent(event: TrainLogLine): { phase: Stage; message: string } | null {
  switch (event.kind) {
    case 'job_submitted':
      return {
        phase: 'prepare',
        message: m.training.store_log.job_submitted(event.backbone)
      };
    case 'job_running':
      return { phase: 'prepare', message: m.training.store_log.job_running };
    case 'phase_started':
      return {
        phase: event.phase,
        message: m.training.store_log.phase_prefix(stageLabel(event.phase))
      };
    case 'dataset_scanned': {
      // Class-count headline only; a truncated per-class breakdown would mislead.
      return {
        phase: 'dataset_scan',
        message: m.training.store_log.scanned_dataset(event.n_classes, event.n_examples_total)
      };
    }
    case 'feature_extract_completed': {
      const dropped = event.dropped_nan + event.dropped_io;
      return {
        phase: 'feature_extract',
        message: m.training.store_log.features_extracted(
          event.kept,
          dropped,
          (event.elapsed_ms / 1000).toFixed(2)
        )
      };
    }
    case 'train_split':
      return {
        phase: 'train',
        message: m.training.store_log.train_split(event.train_n, event.val_n)
      };
    case 'epoch_completed': {
      const lossStr = Number.isFinite(event.train_loss) ? event.train_loss.toFixed(4) : '-';
      const trainAccStr = Number.isFinite(event.train_acc)
        ? `${(event.train_acc * 100).toFixed(1)}%`
        : '-';
      const valAccLabel =
        event.val_acc !== null && Number.isFinite(event.val_acc)
          ? `${(event.val_acc * 100).toFixed(1)}%`
          : null;
      return {
        phase: 'train',
        message: m.training.store_log.epoch_completed(
          event.epoch,
          event.epochs,
          lossStr,
          trainAccStr,
          valAccLabel
        )
      };
    }
    case 'train_completed': {
      // Both args non-null (and val_acc finite) for the `· best val …` suffix to render, else the
      // catalog hides the clause; `?? null` collapses the wire's `undefined` to the typed null.
      const bestValAccLabel =
        event.best_val_acc !== undefined &&
        event.best_val_acc !== null &&
        Number.isFinite(event.best_val_acc)
          ? `${(event.best_val_acc * 100).toFixed(1)}%`
          : null;
      const bestEpoch = event.best_val_epoch ?? null;
      return {
        phase: 'train',
        message: m.training.store_log.train_loop_done(
          event.epochs_run,
          (event.total_elapsed_ms / 1000).toFixed(2),
          bestValAccLabel,
          bestEpoch
        )
      };
    }
    case 'head_published':
      // Full head id inline (not a styled sub-row) so it's selectable as one text run for copy; the
      // log holds the full id vs the 8-char tag in the row header.
      return {
        phase: 'publish',
        message: m.training.store_log.head_published(
          event.head_id,
          formatBytes(event.size_bytes),
          event.n_classes,
          event.workspace_revision.id
        )
      };
    case 'job_completed': {
      // Full (untruncated) labels list - scrollback is the authoritative archive; `formatLabelsList`
      // prettifies reserved synthetics (`_unknown_`) to match the vocabulary.
      const labelsList =
        event.result.classes.length > 0 ? formatLabelsList(event.result.classes) : '';
      return { phase: 'publish', message: m.training.store_log.job_completed(labelsList) };
    }
    case 'job_failed':
      return {
        phase: event.stage,
        message: m.training.store_log.job_failed(stageLabel(event.stage), event.error)
      };
    case 'job_cancelled':
      return {
        phase: event.stage,
        message:
          event.reason === 'shutdown'
            ? m.training.store_log.job_cancelled_shutdown(stageLabel(event.stage))
            : m.training.store_log.job_cancelled(stageLabel(event.stage))
      };
    default: {
      // Forward-compat: a newer daemon's unrecognised `kind` returns null so live + replay skip it
      // silently. The `never` assertion errors at compile time when a new `TrainEvent` variant lands.
      const _exhaustive: never = event;
      void _exhaustive;
      return null;
    }
  }
}

// Synthesise a `TrackedTrainingJob` from a JSONL page via the same helpers as the live SSE path (so
// a hydrated card is indistinguishable from a session terminal). Returns `null` on empty `events`,
// no view, or a still-`running` final state (an abandoned run, omitted). Caller filters nulls.
function replayJsonl(
  workspaceId: Uuid,
  jobId: Uuid,
  events: readonly LogEvent[]
): TrackedTrainingJob | null {
  if (events.length === 0) return null;
  let view: TrainingJobView | null = null;
  let epochs: EpochMetrics[] = [];
  let logLines: TrainingLogLine[] = [];
  // Capture the latest head_id (`job_submitted`/`head_published`) so a failed run keeps the
  // preallocated id.
  let headId: Uuid = '';
  for (const raw of events) {
    // Cast the loose wire `LogEvent` to the discriminated `TrainLogLine` for per-kind narrowing;
    // unknown kinds drop through the helpers' default arms.
    const event = raw as unknown as TrainLogLine;
    if (event.kind === 'job_submitted' || event.kind === 'head_published') {
      headId = event.head_id;
    }
    view = applyEventToView(view, event, jobId, workspaceId);
    if (event.kind === 'epoch_completed') {
      epochs = mergeEpochFromEvent(epochs, event);
    }
    const rendered = renderEvent(event);
    if (rendered !== null) {
      logLines.push({
        seq: event.seq,
        at: event.at,
        phase: rendered.phase,
        message: rendered.message,
        event
      });
    }
  }
  if (view === null) return null;
  // Skip in-flight runs (the SSE path owns those; a hydrated duplicate renders two cards).
  if (view.state === 'running') return null;
  // Same per-job cap the live path enforces, so a long abandoned-run JSONL doesn't blow the memory
  // budget via hydration alone.
  if (logLines.length > MAX_LOG_LINES) {
    logLines = logLines.slice(logLines.length - MAX_LOG_LINES);
  }
  return {
    workspaceId,
    jobId,
    headId,
    view,
    epochs,
    logLines,
    cancelling: false
  };
}

export function isTerminalTrainingState(state: TrainingJobView['state'] | undefined): boolean {
  return state !== undefined && TERMINAL_TRAINING_STATES.has(state);
}

// Falls back to a generic so the operator never reads "undefined".
export function describeTerminalFailure(view: TrainingJobView): string {
  if (view.state !== 'failed') return '';
  // `||` not `??`: an empty-string `error` is as useless as null, so fall back to last progress.
  const errorMsg = view.error?.trim() ?? '';
  const progressMsg = view.progress.message.trim();
  const raw = errorMsg || progressMsg;
  return raw ? capFirst(raw, m.training.summary.failed_default) : m.training.summary.failed_default;
}

export const training = new TrainingStore();
