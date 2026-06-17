// Listed from the daemon's view, not IDB, which can hold uncommitted/phantom rows the importer could never reproduce.

import type { Uuid } from './types';
import { ApiError } from './http';
import { assets } from './endpoints';
import { sliceIdFromFilename } from '$lib/idb/db';
import { getSliceBlob } from '$lib/audio/slice-fetch';
import type { AlpkgEntry } from '$lib/utils/alpkg';

export type DatasetEntriesPhase = 'listing' | 'fetching';

export interface DatasetEntriesProgress {
  phase: DatasetEntriesPhase;
  /// `fetching` only; fixed at the listing-to-fetching transition.
  itemsTotal?: number;
  /// `fetching` only; rises monotonically.
  itemsDone?: number;
}

/// `phase`/`category` let the exporter re-wrap failures structurally, not by message string.
export class DatasetEntriesError extends Error {
  readonly phase: DatasetEntriesPhase;
  /// null when no single category is to blame (fetching-phase abort).
  readonly category: string | null;
  constructor(
    phase: DatasetEntriesPhase,
    category: string | null,
    message: string,
    options?: { cause?: unknown }
  ) {
    super(message, options);
    this.name = 'DatasetEntriesError';
    this.phase = phase;
    this.category = category;
  }
}

const MAX_CONCURRENT_FETCHES = 6;
// Daemon clamps listings to 1000; we page offset->total so larger categories export fully.
const LISTING_LIMIT = 1000;

interface WorkItem {
  category: string;
  filename: string;
  id: string;
}

/// Build the `datasets/<category>/<sha256>.wav` AlpkgEntry list; empty array (not an error) when
/// no on-disk slices exist. Sorted category-then-filename so re-exports are byte-stable.
export async function buildDatasetEntries(
  workspaceId: Uuid,
  categories: readonly string[],
  signal: AbortSignal | undefined,
  onprogress?: (p: DatasetEntriesProgress) => void
): Promise<AlpkgEntry[]> {
  if (categories.length === 0) return [];

  emit(onprogress, { phase: 'listing' });
  const items: WorkItem[] = [];
  for (const cat of categories) {
    // `throwIfAborted` stays outside the try so aborts propagate as-is, not re-wrapped as listing errors.
    let offset = 0;
    for (;;) {
      throwIfAborted(signal, 'listing', cat);
      let page: Awaited<ReturnType<typeof assets.listCategory>> | null;
      try {
        page = await assets.listCategory(workspaceId, cat, { limit: LISTING_LIMIT, offset });
      } catch (e) {
        // 404 = IDB-only (never uploaded) or deleted mid-export: treat as empty.
        if (e instanceof ApiError && e.status === 404) break;
        throw new DatasetEntriesError('listing', cat, `Couldn't list slices in "${cat}".`, {
          cause: e
        });
      }
      for (const entry of page.entries) {
        if (entry.kind !== 'file') continue;
        const id = sliceIdFromFilename(entry.name);
        // Foreign-named files would fail the importer's content-addressed integrity check.
        if (id === null) continue;
        items.push({ category: cat, filename: entry.name, id });
      }
      // Stop when fully collected, or a page makes no progress (concurrent delete mid-walk).
      offset += page.entries.length;
      if (offset >= page.total || page.entries.length === 0) break;
    }
  }

  if (items.length === 0) return [];

  // `getSliceBlob` content-verifies bytes against the slice sha256 (rejecting wrong-file responses)
  // and caps its own concurrency, so this fan-out only saturates the queue and tracks progress.
  emit(onprogress, { phase: 'fetching', itemsTotal: items.length, itemsDone: 0 });
  const bytesById = new Map<string, Uint8Array>();
  let done = 0;
  let cursor = 0;
  // First-failure-wins: workers stop pulling once anything lands here, and only the head is re-thrown.
  const errors: DatasetEntriesError[] = [];
  const fanout = Math.min(MAX_CONCURRENT_FETCHES, items.length);
  const workers = Array.from({ length: fanout }, async () => {
    while (errors.length === 0) {
      const idx = cursor++;
      if (idx >= items.length) return;
      throwIfAborted(signal, 'fetching', null);
      const item = items[idx];
      // Dedup by content hash: identical sha256 across categories shares one fetch+buffer.
      if (bytesById.has(item.id)) {
        done++;
        emit(onprogress, { phase: 'fetching', itemsTotal: items.length, itemsDone: done });
        continue;
      }
      try {
        const blob = await getSliceBlob({
          id: item.id,
          workspace_id: workspaceId,
          category_name: item.category,
          // null `blob` forces the cache+network path; `state`/`created_at` only fill the shape.
          blob: null,
          state: 'committed',
          created_at: ''
        });
        const buf = await blob.arrayBuffer();
        bytesById.set(item.id, new Uint8Array(buf));
      } catch (e) {
        errors.push(
          new DatasetEntriesError(
            'fetching',
            item.category,
            `Couldn't fetch slice ${item.id.slice(0, 8)}… in "${item.category}".`,
            { cause: e }
          )
        );
        return;
      }
      done++;
      emit(onprogress, { phase: 'fetching', itemsTotal: items.length, itemsDone: done });
    }
  });
  await Promise.all(workers);
  if (errors.length > 0) throw errors[0];

  const sorted = items.slice().sort((a, b) => {
    if (a.category !== b.category) return a.category < b.category ? -1 : 1;
    return a.filename < b.filename ? -1 : 1;
  });
  const entries: AlpkgEntry[] = [];
  for (const item of sorted) {
    const bytes = bytesById.get(item.id);
    if (bytes === undefined) continue; // unreachable: every item populated above
    // Category/filename are within the asset-path allowlist (`[A-Za-z0-9._-]`), so `/`-joining is path-safe.
    entries.push({
      path: `datasets/${item.category}/${item.filename}`,
      bytes
    });
  }
  return entries;
}

function throwIfAborted(
  signal: AbortSignal | undefined,
  phase: DatasetEntriesPhase,
  category: string | null
): void {
  if (signal?.aborted === true) {
    const reason: unknown = signal.reason;
    const reasonMsg =
      reason instanceof Error
        ? reason.message
        : typeof reason === 'string'
          ? reason
          : 'export aborted';
    throw new DatasetEntriesError(phase, category, reasonMsg, { cause: reason });
  }
}

function emit(
  onprogress: ((p: DatasetEntriesProgress) => void) | undefined,
  p: DatasetEntriesProgress
): void {
  if (onprogress) onprogress(p);
}
