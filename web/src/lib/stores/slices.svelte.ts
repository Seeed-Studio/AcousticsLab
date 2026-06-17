import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import {
  bulkDeleteSlices,
  bulkPutSlices,
  deleteSlice as idbDeleteSlice,
  deleteSlicesForCategory,
  listSlicesForCategory,
  putSlice,
  sliceKey
} from '$lib/idb/slices';
import { getDB, sliceFilename, sliceIdFromFilename, STORE_SLICES } from '$lib/idb/db';
import type { SliceKey, SliceRecord } from '$lib/idb/db';
import { deleteWorkspaceSync, getWorkspaceSync, putWorkspaceSync } from '$lib/idb/workspace-sync';
import { assets } from '$lib/api/endpoints';
import { enqueueDelete } from '$lib/api/delete-queue';
import { awaitJobTerminal } from '$lib/api/jobs';
import { isTransientUploadError, sleepAbortable, UploadPool, xhrPut } from '$lib/api/upload';
import { isApiError } from '$lib/api/http';
import { capFirst, errorCopy, isNotFound } from '$lib/utils/error-copy';
import type { AssetReceipt, Uuid } from '$lib/api/types';

export interface BulkSliceDeleteFailure {
  id: string;
  filename: string;
  error: string;
}

export interface BulkSliceDeleteOutcome {
  succeeded: number;
  failed: BulkSliceDeleteFailure[];
}

// Reactive cache over the IDB `slices` store, keyed per `(workspace_id, category_name)`; slices
// flow `local -> uploading -> committed | failed`. Identity is content-addressed (id = sha256 hex
// of WAV bytes, filename `<id>.wav`): same content across categories shares cache rows but has
// independent slices-store rows (composite key `[workspace_id, category_name, id]`), and re-slicing
// identical audio within one category dedups by IDB-overwrite.
//
// Revision-keyed sync. Mount short-circuit: skip index GETs when persisted
// `last_synced_revision_id >= workspaceRevision` (unforced). Per-category reconcile = filename
// set-difference: daemon-only -> synthesise committed; local-committed absent on daemon -> drop
// ONLY if the listing succeeded; local non-committed -> always preserve.
//
// SvelteMap is reactive on `.set`/`.delete` but values are NOT deeply reactive, so every mutation
// replaces the slice with a fresh object. Callers wrap `refresh()` in `untrack` so a read inside
// it can't register the list as an `$effect` dep and self-retrigger.

function key(workspaceId: Uuid, categoryName: string): string {
  return `${workspaceId} ${categoryName}`;
}

function flightKey(workspaceId: Uuid, categoryName: string, id: string): string {
  return `${workspaceId}/${categoryName}/${id}`;
}

function byCreatedAsc(a: SliceRecord, b: SliceRecord): number {
  if (a.created_at === b.created_at) return 0;
  return a.created_at < b.created_at ? -1 : 1;
}

interface SliceList {
  entries: SliceRecord[];
  loading: boolean;
  loaded: boolean;
  error: string | null;
}

const EMPTY_LIST: Readonly<SliceList> = Object.freeze({
  entries: [] as SliceRecord[],
  loading: false,
  loaded: false,
  error: null as string | null
});

const PENDING_STATES: ReadonlySet<SliceRecord['state']> = new Set(['local', 'uploading', 'failed']);

export type CategorySyncStatus = 'empty' | 'synced' | 'pending' | 'uploading' | 'failed';

const MAX_CONCURRENT_UPLOADS = 3;
const UPLOAD_RETRY_ATTEMPTS = 4;
const UPLOAD_RETRY_BASE_MS = 500;
const UPLOAD_RETRY_MAX_MS = 8000;

const MAX_CONCURRENT_INDEX_FETCHES = 3;

async function withConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T) => Promise<R>
): Promise<R[]> {
  if (items.length === 0) return [];
  const results = Array.from<R | undefined>({ length: items.length });
  let cursor = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    for (;;) {
      const idx = cursor;
      cursor += 1;
      if (idx >= items.length) return;
      results[idx] = await fn(items[idx]);
    }
  });
  await Promise.all(workers);
  return results as R[];
}

// Null for filenames outside the strict `<sha256hex>.wav` shape: foreign-named bytes fail the
// content-addressed integrity check on download anyway.
function synthesiseServerSlice(
  workspaceId: Uuid,
  categoryName: string,
  filename: string,
  mtime: string
): SliceRecord | null {
  const id = sliceIdFromFilename(filename);
  if (id === null) return null;
  return {
    id,
    workspace_id: workspaceId,
    category_name: categoryName,
    blob: null,
    state: 'committed',
    created_at: mtime
  };
}

type DatasetListingEntries = Awaited<ReturnType<typeof assets.listCategory>>['entries'];

// Page past the daemon's 1000-entry cap to list EVERY entry: the reconcile treats the listing as
// authoritative, so a capped page would destroy committed IDB rows beyond the first 1000 as
// remote-deleted orphans. `complete` is false when a concurrent mutation shifts offsets and
// truncates the walk; the caller MUST suppress the orphan sweep then.
async function listAllCategoryEntries(
  workspaceId: Uuid,
  categoryName: string
): Promise<{ entries: DatasetListingEntries; complete: boolean }> {
  const PAGE = 1000;
  const all: DatasetListingEntries = [];
  let offset = 0;
  for (;;) {
    const page = await assets.listCategory(workspaceId, categoryName, { limit: PAGE, offset });
    all.push(...page.entries);
    if (all.length >= page.total) return { entries: all, complete: true };
    // Empty page while `total` claims more (concurrent delete shrank the dir mid-walk) would loop
    // forever: bail incomplete to suppress the orphan sweep.
    if (page.entries.length === 0) return { entries: all, complete: false };
    offset += page.entries.length;
  }
}

class SlicesStore {
  private lists = new SvelteMap<string, SliceList>();
  private workspacesLoaded = new SvelteSet<Uuid>();

  private uploadPool = new UploadPool(MAX_CONCURRENT_UPLOADS);
  // Keyed by `flightKey`: the same hash in different (ws, cat) tuples is a distinct upload.
  private inflightUploads = new SvelteMap<string, AbortController>();
  // `flightKey`s mid-`delete()`. In-memory only; tab close mid-batch self-heals on next mount.
  deletingIds = new SvelteSet<string>();
  // Highest daemon revision from any source; bumps on every advance regardless of reconcile.
  private latestRevisions = new SvelteMap<Uuid, number>();
  // Mirror of `workspace_sync.last_synced_revision_id`. The poller's reconcile gate compares THIS
  // (not latestRevisions, which would mask failures), so a failed/blocked reconcile leaves it
  // behind and the next poll retries.
  private lastSyncedRevisions = new SvelteMap<Uuid, number>();
  private mutationsInFlight = new SvelteMap<Uuid, number>();
  // Poller stamps keys on revision advance; the per-category refresh clears on reconcile.
  private staleKeys = new SvelteSet<string>();
  // Re-entry guard: workspaces with an in-flight workspace-wide reconcile.
  private reconcilingWorkspaces = new Set<Uuid>();

  for(workspaceId: Uuid, categoryName: string): SliceList {
    return this.lists.get(key(workspaceId, categoryName)) ?? EMPTY_LIST;
  }

  countFor(workspaceId: Uuid, categoryName: string): number {
    return this.for(workspaceId, categoryName).entries.length;
  }

  syncStatusFor(workspaceId: Uuid, categoryName: string): CategorySyncStatus {
    const entries = this.for(workspaceId, categoryName).entries;
    if (entries.length === 0) return 'empty';
    let hasUploading = false;
    let hasLocal = false;
    for (const e of entries) {
      if (e.state === 'failed') return 'failed';
      if (e.state === 'uploading') hasUploading = true;
      else if (e.state === 'local') hasLocal = true;
    }
    if (hasUploading) return 'uploading';
    if (hasLocal) return 'pending';
    return 'synced';
  }

  latestRevisionFor(workspaceId: Uuid): number | null {
    return this.latestRevisions.get(workspaceId) ?? null;
  }

  setRevisionAtLeast(workspaceId: Uuid, revision: number): void {
    const prior = this.latestRevisions.get(workspaceId) ?? -1;
    if (revision > prior) this.latestRevisions.set(workspaceId, revision);
  }

  lastSyncedRevisionFor(workspaceId: Uuid): number | null {
    return this.lastSyncedRevisions.get(workspaceId) ?? null;
  }

  private setLastSyncedAtLeast(workspaceId: Uuid, revision: number): void {
    const prior = this.lastSyncedRevisions.get(workspaceId) ?? -1;
    if (revision > prior) this.lastSyncedRevisions.set(workspaceId, revision);
  }

  private loadedCategoryNames(workspaceId: Uuid): string[] {
    const prefix = `${workspaceId} `;
    const names: string[] = [];
    for (const k of this.lists.keys()) {
      if (k.startsWith(prefix)) names.push(k.slice(prefix.length));
    }
    return names;
  }

  beginMutation(workspaceId: Uuid): void {
    this.mutationsInFlight.set(workspaceId, (this.mutationsInFlight.get(workspaceId) ?? 0) + 1);
  }

  endMutation(workspaceId: Uuid): void {
    const cur = this.mutationsInFlight.get(workspaceId) ?? 0;
    if (cur <= 1) this.mutationsInFlight.delete(workspaceId);
    else this.mutationsInFlight.set(workspaceId, cur - 1);
  }

  mutationsInFlightFor(workspaceId: Uuid): number {
    return this.mutationsInFlight.get(workspaceId) ?? 0;
  }

  isStale(workspaceId: Uuid, categoryName: string): boolean {
    return this.staleKeys.has(key(workspaceId, categoryName));
  }

  isDeleting(workspaceId: Uuid, categoryName: string, id: string): boolean {
    return this.deletingIds.has(flightKey(workspaceId, categoryName, id));
  }

  // True iff ANY committed-slice delete is in flight. `syncStatusFor` reflects only upload state
  // (a deleting slice is still 'committed'), so category rename gates on this too: an in-flight
  // DELETE targets the OLD path and would 404 against the moved dir, leaving the "deleted" slice
  // alive under the new name.
  hasInflightDeletes(workspaceId: Uuid, categoryName: string): boolean {
    const prefix = `${workspaceId}/${categoryName}/`;
    for (const fk of this.deletingIds) {
      if (fk.startsWith(prefix)) return true;
    }
    return false;
  }

  // Poller invalidation on revision advance: stamp every loaded category stale (so an expanded
  // pane re-fires refresh on the tracked dep) AND kick a background workspace-wide reconcile so
  // collapsed-badge counts and the persisted sync record reach `workspaceRevision`.
  markStaleForWorkspace(workspaceId: Uuid, workspaceRevision: number): void {
    const prefix = `${workspaceId} `;
    const names: string[] = [];
    for (const k of this.lists.keys()) {
      if (k.startsWith(prefix)) {
        this.staleKeys.add(k);
        names.push(k.slice(prefix.length));
      }
    }
    if (names.length > 0) {
      void this.reconcileWorkspace(workspaceId, names, workspaceRevision);
    }
  }

  pendingFor(workspaceId: Uuid): SliceRecord[] {
    const result: SliceRecord[] = [];
    const prefix = `${workspaceId} `;
    for (const [k, list] of this.lists) {
      if (!k.startsWith(prefix)) continue;
      for (const entry of list.entries) {
        if (PENDING_STATES.has(entry.state)) result.push(entry);
      }
    }
    return result;
  }

  async refreshForWorkspace(
    workspaceId: Uuid,
    categoryNames: string[],
    workspaceRevision: number,
    force = false
  ): Promise<void> {
    // One indexed bulk query seeds all per-category lists, guarded idempotent.
    if (!this.workspacesLoaded.has(workspaceId)) {
      try {
        const db = await getDB();
        const rows = await db.getAllFromIndex(STORE_SLICES, 'by-workspace', workspaceId);
        const grouped = new SvelteMap<string, SliceRecord[]>();
        for (const name of categoryNames) grouped.set(name, []);
        for (const row of rows) {
          const list = grouped.get(row.category_name) ?? [];
          list.push(row);
          grouped.set(row.category_name, list);
        }
        for (const [name, entries] of grouped) {
          entries.sort(byCreatedAsc);
          const k = key(workspaceId, name);
          if (!this.lists.has(k)) {
            this.lists.set(k, {
              entries,
              loading: false,
              loaded: true,
              error: null
            });
          }
        }
        this.workspacesLoaded.add(workspaceId);
        this.resumePending(workspaceId);
      } catch (e) {
        console.warn('[slices] bulk refresh failed', e);
      }
    }

    // Seed the in-memory mirror from IDB on first sighting; cross-tab writes don't propagate, so
    // the failure mode is over-reconcile (wasteful but correct), never missed data. Re-read AFTER
    // the await: a concurrent setLastSyncedAtLeast is a stricter upper bound, so trust it.
    let synced = this.lastSyncedRevisions.get(workspaceId);
    if (synced === undefined) {
      const sync = await getWorkspaceSync(workspaceId).catch(() => undefined);
      if (sync !== undefined) {
        this.setLastSyncedAtLeast(workspaceId, sync.last_synced_revision_id);
      }
      synced = this.lastSyncedRevisions.get(workspaceId);
    }

    // Short-circuit when at/ahead of the caller's known revision (`synced > workspaceRevision` when
    // markCommitted auto-advanced since the caller's fetch): IDB holds the caller's state plus our
    // commits, and later external changes are caught by the next poller tick.
    if (!force && synced !== undefined && synced >= workspaceRevision) {
      return;
    }

    if (categoryNames.length === 0) return;
    void this.reconcileWorkspace(workspaceId, categoryNames, workspaceRevision);
  }

  async reconcileWorkspace(
    workspaceId: Uuid,
    categoryNames: readonly string[],
    workspaceRevision: number
  ): Promise<void> {
    if (this.reconcilingWorkspaces.has(workspaceId)) return;
    if (categoryNames.length === 0) return;
    this.reconcilingWorkspaces.add(workspaceId);
    let succeededAt: number | null = null;
    try {
      const outcomes = await withConcurrency(categoryNames, MAX_CONCURRENT_INDEX_FETCHES, (name) =>
        this.refresh(workspaceId, name, true).then(
          () => true,
          () => false
        )
      );
      const allSucceeded = outcomes.every((ok) => ok);
      const everyCategorySettled = categoryNames.every((name) => {
        const list = this.lists.get(key(workspaceId, name));
        return list?.loaded === true && list.error === null;
      });
      // Forget-race guard: if `forget(workspaceId)` ran during the reconcile (workspace-delete
      // chain), resurrecting a sync row would leave an IDB orphan a same-UUID re-create inherits.
      // `workspacesLoaded` (set by bulk-load, cleared by forget) closes the window.
      if (allSucceeded && everyCategorySettled && this.workspacesLoaded.has(workspaceId)) {
        // Persist max(reconcile rev, mirror): if markCommitted's auto-advance bumped mirror past
        // `workspaceRevision` during the fan-out, mirror is the stricter upper bound and writing
        // `workspaceRevision` would regress IDB past that fire-and-forget put.
        this.setLastSyncedAtLeast(workspaceId, workspaceRevision);
        const persistRev = this.lastSyncedRevisions.get(workspaceId) ?? workspaceRevision;
        await putWorkspaceSync({
          workspace_id: workspaceId,
          last_synced_revision_id: persistRev,
          last_synced_at: new Date().toISOString()
        }).catch(() => undefined);
        // Read the post-await mirror so catch-up only re-fires for advances past our synced rev that
        // markCommitted's strict-+1 auto-advance did NOT already cover (external peer uploads, or our
        // own gap-leaving commit whose receipt rev was not exactly +1).
        succeededAt = this.lastSyncedRevisions.get(workspaceId) ?? persistRev;
      }
    } finally {
      this.reconcilingWorkspaces.delete(workspaceId);
    }

    // Catch-up: if the daemon advanced past `workspaceRevision` while we synced (peer upload, or a
    // poll tick blocked by the in-flight guard), re-fire over the live loaded-category set. Success
    // path only; a failed reconcile self-heals via the poller's gate.
    if (succeededAt !== null) {
      const newest = this.latestRevisions.get(workspaceId);
      if (newest !== undefined && newest > succeededAt) {
        const live = this.loadedCategoryNames(workspaceId);
        if (live.length > 0) {
          void this.reconcileWorkspace(workspaceId, live, newest);
        }
      }
    }
  }

  // Per-category set-difference sync (rules in the file header). A 404 on the category dir is the
  // empty case (added but nothing uploaded yet); other errors surface on the list.
  async refresh(workspaceId: Uuid, categoryName: string, force = false): Promise<void> {
    const k = key(workspaceId, categoryName);
    const existing = this.lists.get(k);
    const stale = this.staleKeys.has(k);
    if (existing?.loaded && !force && !existing.error && !stale) return;
    if (existing?.loading && !force) return;

    // Snapshot entries at start so merge-on-finish can distinguish "deleted during refresh" (in
    // start, gone from current) from "synthesised by refresh" (in neither).
    const startEntries = existing?.entries ?? [];
    const startIds = new SvelteSet(startEntries.map((s) => s.id));

    this.lists.set(k, {
      ...EMPTY_LIST,
      entries: startEntries,
      loading: true,
      loaded: existing?.loaded ?? false
    });

    try {
      const [localRows, serverListing] = await Promise.all([
        listSlicesForCategory(workspaceId, categoryName),
        listAllCategoryEntries(workspaceId, categoryName).catch((e: unknown) => {
          if (isNotFound(e)) {
            // Category gone: a complete (empty) view, so committed IDB rows are orphans.
            return { entries: [] as DatasetListingEntries, complete: true };
          }
          throw e;
        })
      ]);

      // Forget-race guard: a workspace-delete may have cleared this entry during the await.
      if (!this.lists.has(k)) return;

      const serverFilenames = new SvelteMap<string, { name: string; mtime: string }>();
      for (const entry of serverListing.entries) {
        if (entry.kind !== 'file' || !entry.name.endsWith('.wav')) continue;
        serverFilenames.set(entry.name, { name: entry.name, mtime: entry.mtime });
      }

      const kept: SliceRecord[] = [];
      const toPut: SliceRecord[] = [];
      const toDeleteKeys: SliceKey[] = [];

      // Classify local rows: locally-mutable states always preserved; only committed rows obey
      // daemon-as-master. `seenFilenames` (populated for BOTH branches, since the daemon may already
      // hold a hash a still-pending local upload covers) stops the synthesise loop adding a
      // duplicate id -> Svelte keyed-{#each} duplicate-key warning + double-render.
      const seenFilenames = new SvelteSet<string>();
      for (const row of localRows) {
        if (row.state !== 'committed') {
          kept.push(row);
          seenFilenames.add(sliceFilename(row.id));
          continue;
        }
        const fname = sliceFilename(row.id);
        if (serverFilenames.has(fname)) {
          kept.push(row);
          seenFilenames.add(fname);
        } else if (serverListing.complete) {
          // Remote-deleted orphan, trustworthy only against a complete directory.
          toDeleteKeys.push(sliceKey(row));
        } else {
          // Incomplete view can't prove the row gone; keep rather than destroy a valid slice.
          kept.push(row);
          seenFilenames.add(fname);
        }
      }

      for (const [filename, remote] of serverFilenames) {
        if (seenFilenames.has(filename)) continue;
        const synthetic = synthesiseServerSlice(workspaceId, categoryName, filename, remote.mtime);
        if (synthetic !== null) {
          kept.push(synthetic);
          toPut.push(synthetic);
        }
      }

      if (toPut.length > 0) {
        // The listing predates any delete() that completed during the await, so a just-removed
        // filename can still synthesise into toPut; the in-memory merge drops it, but persisting it
        // leaves an IDB orphan a fresh mount re-surfaces as a phantom committed card. Skip any toPut
        // id whose delete() is still in flight.
        const persistable = toPut.filter(
          (row) => !this.deletingIds.has(flightKey(workspaceId, categoryName, row.id))
        );
        if (persistable.length > 0) {
          await bulkPutSlices(persistable).catch(() => undefined);
        }
      }
      if (toDeleteKeys.length > 0) {
        await bulkDeleteSlices(toDeleteKeys).catch(() => undefined);
      }

      // Re-check the forget-race after the IDB writes (workspace-delete may have cleared k).
      if (!this.lists.has(k)) return;

      // Merge `kept` with current in-memory state to absorb mutations (append/delete/state change)
      // during the await, else a new append is dropped or a mid-refresh commit reverts to the IDB
      // snapshot; current always wins for an overlapping id. The two startIds branches resolve
      // "committed-but-missing": in startIds = daemon-confirmed orphan, drop; not in startIds =
      // appended-and-committed during the window, preserve. (The "went local mid-refresh, listing
      // missed the PUT" case can't occur: markCommitted fires only after the PUT propagates.)
      const currentList = this.lists.get(k);
      const currentEntries = currentList?.entries ?? [];
      const currentById = new SvelteMap<string, SliceRecord>();
      for (const entry of currentEntries) currentById.set(entry.id, entry);

      const finalEntries: SliceRecord[] = [];
      const finalIds = new SvelteSet<string>();
      for (const row of kept) {
        const inCurrent = currentById.get(row.id);
        if (inCurrent === undefined) {
          if (startIds.has(row.id)) {
            continue;
          }
          finalEntries.push(row);
        } else {
          finalEntries.push(inCurrent);
        }
        finalIds.add(row.id);
      }
      for (const entry of currentEntries) {
        if (finalIds.has(entry.id)) continue;
        if (entry.state !== 'committed' || !startIds.has(entry.id)) {
          finalEntries.push(entry);
        }
      }

      finalEntries.sort(byCreatedAsc);
      this.lists.set(k, {
        entries: finalEntries,
        loading: false,
        loaded: true,
        error: null
      });
      this.staleKeys.delete(k);
    } catch (e) {
      if (!this.lists.has(k)) return;
      // Stamp the error on CURRENT state, not the pre-refresh snapshot, so mutations during the
      // failed refresh survive.
      const current = this.lists.get(k);
      if (!current) return;
      this.lists.set(k, {
        ...current,
        loading: false,
        error: errorCopy(e)
      });
      throw e;
    }
  }

  // Register the AbortController at ENQUEUE time, not on pool dispatch: a delete() landing while
  // queued must abort the task before it takes a slot and PUTs bytes to a just-removed slice. Eager
  // registration also closes a dedup race (two synchronous enqueues for one `fk` would both pass
  // the `has(fk)` gate if it were set inside runUpload).
  enqueueUpload(record: Pick<SliceRecord, 'workspace_id' | 'category_name' | 'id'>): Promise<void> {
    const fk = flightKey(record.workspace_id, record.category_name, record.id);
    if (this.inflightUploads.has(fk)) {
      return Promise.resolve();
    }
    const controller = new AbortController();
    this.inflightUploads.set(fk, controller);
    return this.uploadPool.submit(async () => {
      try {
        await this.runUpload(record.workspace_id, record.category_name, record.id, controller);
      } finally {
        // Clear only while still the registered owner: a re-slice enqueue after a delete aborted us
        // may have repopulated `fk` with a different controller we must not clobber.
        if (this.inflightUploads.get(fk) === controller) {
          this.inflightUploads.delete(fk);
        }
      }
    });
  }

  resumePending(workspaceId: Uuid): void {
    for (const slice of this.pendingFor(workspaceId)) {
      void this.enqueueUpload(slice);
    }
  }

  private async runUpload(
    workspaceId: Uuid,
    categoryName: string,
    id: string,
    controller: AbortController
  ): Promise<void> {
    // Aborted before we got a pool slot (delete fired while queued); bail before touching slice
    // state. Typed local needed so TS's single-shot narrowing doesn't flag the later catch-branch
    // re-read of `signal.aborted` as always-falsy.
    const initiallyAborted: boolean = controller.signal.aborted;
    if (initiallyAborted) return;
    const slice = this.findSlice(workspaceId, categoryName, id);
    if (!slice) return;
    if (!slice.blob || slice.blob.size === 0) {
      await this.markFailed(slice, 'No local bytes to upload.');
      return;
    }
    this.beginMutation(workspaceId);

    try {
      await this.markUploading(slice);

      const url = assets.slicePutPath(workspaceId, categoryName, sliceFilename(id));
      let lastError: unknown = null;
      for (let attempt = 1; attempt <= UPLOAD_RETRY_ATTEMPTS; attempt++) {
        if (attempt > 1) {
          this.setProgress(workspaceId, categoryName, id, 0);
          const wait = Math.min(
            UPLOAD_RETRY_BASE_MS * Math.pow(2, attempt - 2),
            UPLOAD_RETRY_MAX_MS
          );
          const jittered = wait * (0.75 + Math.random() * 0.5);
          try {
            await sleepAbortable(jittered, controller.signal);
          } catch {
            return;
          }
        }
        try {
          const receipt = await xhrPut<AssetReceipt>({
            url,
            body: slice.blob,
            contentType: 'audio/wav',
            onProgress: (loaded, total) => {
              if (total > 0) this.setProgress(workspaceId, categoryName, id, loaded / total);
            },
            signal: controller.signal
          });
          this.setRevisionAtLeast(workspaceId, receipt.workspace_revision_id);
          if (receipt.sha256 !== id) {
            // Daemon hash != our pre-computed id means transport corruption (or algo disagreement);
            // fail so the operator retries.
            lastError = new Error(
              `Daemon receipt sha256 (${receipt.sha256}) did not match slice id (${id}).`
            );
            break;
          }
          await this.markCommitted(slice, receipt.workspace_revision_id);
          return;
        } catch (e) {
          if (controller.signal.aborted) return;
          lastError = e;
          if (!isTransientUploadError(e)) break;
        }
      }

      await this.markFailed(slice, errorCopy(lastError));
    } finally {
      this.endMutation(workspaceId);
    }
  }

  private findSlice(workspaceId: Uuid, categoryName: string, id: string): SliceRecord | undefined {
    const list = this.lists.get(key(workspaceId, categoryName));
    if (!list) return undefined;
    return list.entries.find((s) => s.id === id);
  }

  private patchInMemory(
    workspaceId: Uuid,
    categoryName: string,
    id: string,
    transform: (s: SliceRecord) => SliceRecord
  ): SliceRecord | undefined {
    const k = key(workspaceId, categoryName);
    const list = this.lists.get(k);
    if (!list) return undefined;
    const idx = list.entries.findIndex((s) => s.id === id);
    if (idx < 0) return undefined;
    const next = transform(list.entries[idx]);
    const entries = list.entries.slice();
    entries[idx] = next;
    this.lists.set(k, { ...list, entries });
    return next;
  }

  private async markUploading(slice: SliceRecord): Promise<void> {
    const next = this.patchInMemory(slice.workspace_id, slice.category_name, slice.id, (s) => ({
      ...s,
      state: 'uploading',
      upload_progress: 0,
      last_error: undefined
    }));
    if (next) await putSlice(next).catch(() => undefined);
  }

  private setProgress(workspaceId: Uuid, categoryName: string, id: string, progress: number): void {
    // Every patch reallocates the shared `entries` array and re-runs the pane's $derived chain plus
    // selection-prune effect, so throttle to whole-percent transitions.
    const current = this.findSlice(workspaceId, categoryName, id);
    if (current !== undefined) {
      const priorPct = Math.round((current.upload_progress ?? 0) * 100);
      const nextPct = Math.round(progress * 100);
      if (priorPct === nextPct) return;
    }
    this.patchInMemory(workspaceId, categoryName, id, (s) => ({
      ...s,
      upload_progress: progress
    }));
  }

  private async markCommitted(slice: SliceRecord, revisionId: number): Promise<void> {
    const next = this.patchInMemory(slice.workspace_id, slice.category_name, slice.id, (s) => ({
      ...s,
      state: 'committed',
      blob: null,
      upload_progress: undefined,
      last_error: undefined,
      workspace_revision_id: revisionId
    }));
    if (next) await putSlice(next).catch(() => undefined);

    // Auto-advance the persisted sync record ONLY when the receipt rev is exactly +1 our synced
    // rev: our upload is then provably the sole mutation in the gap, so claiming synced can't miss
    // data; a multi-tab interleave leaves a gap, the strict `+1` fails, and the poller reconcile
    // takes over (without this, every local commit forces a needless reconcile next tick).
    // Fire-and-forget the IDB write (same-connection last-write-wins; unflushed re-reconciles).
    const wsId = slice.workspace_id;
    const priorSynced = this.lastSyncedRevisions.get(wsId) ?? -1;
    if (priorSynced + 1 === revisionId) {
      this.setLastSyncedAtLeast(wsId, revisionId);
      void putWorkspaceSync({
        workspace_id: wsId,
        last_synced_revision_id: revisionId,
        last_synced_at: new Date().toISOString()
      }).catch(() => undefined);
    }
  }

  private async markFailed(slice: SliceRecord, error: string): Promise<void> {
    const next = this.patchInMemory(slice.workspace_id, slice.category_name, slice.id, (s) => ({
      ...s,
      state: 'failed',
      upload_progress: undefined,
      last_error: error
    }));
    if (next) await putSlice(next).catch(() => undefined);
  }

  // Composite-key re-slice of byte-identical audio in one (workspace, category) overwrites the
  // prior row; the caller reads the post-put length for a "duplicates collapsed" hint.
  async append(record: SliceRecord): Promise<void> {
    await putSlice(record);
    const k = key(record.workspace_id, record.category_name);
    const existing = this.lists.get(k);
    const baseEntries = existing?.entries ?? [];
    // Tail-insert preserves `created_at` ascending order (the IDB query mirrors it).
    const replaceIdx = baseEntries.findIndex((s) => s.id === record.id);
    let entries: SliceRecord[];
    if (replaceIdx >= 0) {
      entries = baseEntries.slice();
      entries[replaceIdx] = record;
    } else {
      entries = [...baseEntries, record];
    }
    this.lists.set(k, {
      ...EMPTY_LIST,
      entries,
      loaded: true,
      loading: false,
      error: null
    });
  }

  async delete(record: SliceRecord): Promise<void> {
    const fk = flightKey(record.workspace_id, record.category_name, record.id);
    if (this.deletingIds.has(fk)) return;
    this.deletingIds.add(fk);
    const remoteDelete = record.state === 'committed';
    if (remoteDelete) this.beginMutation(record.workspace_id);
    try {
      const controller = this.inflightUploads.get(fk);
      if (controller) {
        controller.abort();
        this.inflightUploads.delete(fk);
      }
      if (remoteDelete) {
        try {
          await enqueueDelete(() => this.runRemoteDelete(record));
        } catch (e) {
          console.warn('[slices] remote delete failed', e);
          throw e;
        }
      }
      await idbDeleteSlice(record.workspace_id, record.category_name, record.id);
      // No cache eviction: content-addressed spectrogram/blob caches may still back another slice
      // with the same content, so they accumulate (`resetDB` is the only reset).
      const k = key(record.workspace_id, record.category_name);
      const existing = this.lists.get(k);
      if (!existing) return;
      this.lists.set(k, {
        ...existing,
        entries: existing.entries.filter((s) => s.id !== record.id)
      });
    } finally {
      this.deletingIds.delete(fk);
      if (remoteDelete) this.endMutation(record.workspace_id);
    }
  }

  async deleteMany(targets: SliceRecord[]): Promise<BulkSliceDeleteOutcome> {
    const failed: BulkSliceDeleteFailure[] = [];
    let succeeded = 0;
    await Promise.all(
      targets.map(async (record) => {
        try {
          await this.delete(record);
          succeeded++;
        } catch (e) {
          // ApiError.message is layer-prefixed thiserror text (e.g. "Fs: ...") that errorCopy
          // strips; awaitJobTerminal's plain Error only needs capFirst.
          const message = isApiError(e)
            ? errorCopy(e)
            : e instanceof Error && e.message
              ? capFirst(e.message, 'Delete failed.')
              : errorCopy(e);
          failed.push({ id: record.id, filename: sliceFilename(record.id), error: message });
        }
      })
    );
    return { succeeded, failed };
  }

  private async runRemoteDelete(slice: SliceRecord): Promise<void> {
    const ack = await assets.deleteSlice(
      slice.workspace_id,
      slice.category_name,
      sliceFilename(slice.id)
    );
    await awaitJobTerminal(ack.job_id);
  }

  // MOVE the in-memory list (not WIPE like clearForCategory), rewriting each row's `category_name`
  // and migrating the per-category staleKey, for immediate render; IDB rows move in the caller's
  // cross-store tx (revision/mutation maps are workspace-keyed, no move needed). Precondition:
  // callers gate rename on syncStatusFor 'synced'/'empty' because the upload URL bakes the old
  // name, so a mid-rename PUT would re-create the old daemon directory.
  renameCategory(workspaceId: Uuid, oldName: string, newName: string): void {
    if (oldName === newName) return;
    const oldKey = key(workspaceId, oldName);
    const list = this.lists.get(oldKey);
    if (!list) return;
    const newKey = key(workspaceId, newName);
    this.lists.set(newKey, {
      ...list,
      entries: list.entries.map((s) => ({ ...s, category_name: newName }))
    });
    this.lists.delete(oldKey);
    if (this.staleKeys.has(oldKey)) {
      this.staleKeys.delete(oldKey);
      this.staleKeys.add(newKey);
    }
  }

  async clearForCategory(workspaceId: Uuid, categoryName: string): Promise<void> {
    const existing = this.lists.get(key(workspaceId, categoryName));
    if (existing && existing.entries.length > 0) {
      for (const slice of existing.entries) {
        const fk = flightKey(workspaceId, categoryName, slice.id);
        const controller = this.inflightUploads.get(fk);
        if (controller) {
          controller.abort();
          this.inflightUploads.delete(fk);
        }
      }
    }
    await deleteSlicesForCategory(workspaceId, categoryName);
    const k = key(workspaceId, categoryName);
    this.lists.set(k, {
      ...EMPTY_LIST,
      loaded: true
    });
    this.staleKeys.delete(k);
  }

  forget(workspaceId: Uuid, categoryName?: string): void {
    const drop = (k: string, name: string): void => {
      const list = this.lists.get(k);
      if (list && list.entries.length > 0) {
        for (const slice of list.entries) {
          const fk = flightKey(workspaceId, name, slice.id);
          const controller = this.inflightUploads.get(fk);
          if (controller) {
            controller.abort();
            this.inflightUploads.delete(fk);
          }
        }
      }
      this.lists.delete(k);
      this.staleKeys.delete(k);
    };
    if (categoryName !== undefined) {
      drop(key(workspaceId, categoryName), categoryName);
      return;
    }
    const prefix = `${workspaceId} `;
    for (const k of Array.from(this.lists.keys())) {
      if (k.startsWith(prefix)) {
        const name = k.slice(prefix.length);
        drop(k, name);
      }
    }
    this.workspacesLoaded.delete(workspaceId);
    this.latestRevisions.delete(workspaceId);
    // Drop the mirror: a same-UUID re-created workspace (via local seed data) would otherwise
    // inherit the prior synced claim and mount-short-circuit a reconcile against the empty one.
    this.lastSyncedRevisions.delete(workspaceId);
    this.mutationsInFlight.delete(workspaceId);
    this.reconcilingWorkspaces.delete(workspaceId);
    // Workspace-delete chain also GCs this; firing here covers forget-without-delete.
    void deleteWorkspaceSync(workspaceId).catch(() => undefined);
  }
}

export const slices = new SlicesStore();
