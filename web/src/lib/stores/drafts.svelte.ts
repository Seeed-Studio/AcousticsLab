import { SvelteMap } from 'svelte/reactivity';
import { deleteDraft as idbDeleteDraft, getDraft, putDraft } from '$lib/idb/drafts';
import type { DraftRecord } from '$lib/idb/db';
import type { Uuid } from '$lib/api/types';

// Single-slot per key (one IDB row). SvelteMap notifies on .set/.delete/.clear but values aren't
// deeply reactive, so every mutation must store a fresh object.

// String key dodges Map reference-equality (equal-content tuples hash apart); `\0` can't collide
// since no UUID or category name contains a NUL byte.
function key(workspaceId: Uuid, categoryName: string): string {
  return `${workspaceId}\0${categoryName}`;
}

interface DraftSlice {
  // `null` = read IDB and confirmed no draft; distinct from not-yet-read (`loaded` false).
  draft: DraftRecord | null;
  loading: boolean;
  loaded: boolean;
  error: string | null;
}

const EMPTY_SLICE: Readonly<DraftSlice> = Object.freeze({
  draft: null,
  loading: false,
  loaded: false,
  error: null
});

class DraftsStore {
  private slices = new SvelteMap<string, DraftSlice>();

  // Frozen empty slice for untouched keys so consumers never null-check.
  for(workspaceId: Uuid, categoryName: string): DraftSlice {
    return this.slices.get(key(workspaceId, categoryName)) ?? EMPTY_SLICE;
  }

  // loaded/in-flight guards break the reactive loop when called from a `$effect`; `force` bypasses
  // them to re-fetch on demand.
  async refresh(workspaceId: Uuid, categoryName: string, force = false): Promise<void> {
    const k = key(workspaceId, categoryName);
    const existing = this.slices.get(k);
    if (existing?.loaded && !force && !existing.error) return;
    if (existing?.loading && !force) return;

    // Keep the prior draft on the loading tick so the canvas doesn't flicker to empty.
    this.slices.set(k, {
      ...EMPTY_SLICE,
      draft: existing?.draft ?? null,
      loading: true,
      loaded: existing?.loaded ?? false
    });

    try {
      const draft = await getDraft(workspaceId, categoryName);
      this.slices.set(k, {
        draft: draft ?? null,
        loading: false,
        loaded: true,
        error: null
      });
    } catch (e) {
      this.slices.set(k, {
        draft: existing?.draft ?? null,
        loading: false,
        loaded: existing?.loaded ?? false,
        error: e instanceof Error ? e.message : String(e)
      });
    }
  }

  async save(record: DraftRecord): Promise<void> {
    await putDraft(record);
    this.slices.set(key(record.workspace_id, record.category_name), {
      draft: record,
      loading: false,
      loaded: true,
      error: null
    });
  }

  // Cache is set synchronously (fresh object) before the IDB write settles, else when the committed
  // record flows back through the parent's `$derived` its `$effect` would reset the trim handle to the
  // prior persisted values. Write is best-effort and swallowed: a `void`-discarded call must not raise
  // an unhandled rejection (cache holds until next refresh).
  async patchTrim(
    workspaceId: Uuid,
    categoryName: string,
    trimStartSamples: number,
    trimEndSamples: number
  ): Promise<void> {
    const k = key(workspaceId, categoryName);
    const slice = this.slices.get(k);
    if (!slice?.draft) return;
    const next: DraftRecord = {
      ...slice.draft,
      trim_start_samples: trimStartSamples,
      trim_end_samples: trimEndSamples
    };
    this.slices.set(k, { ...slice, draft: next });
    await putDraft(next).catch(() => undefined);
  }

  async clear(workspaceId: Uuid, categoryName: string): Promise<void> {
    await idbDeleteDraft(workspaceId, categoryName);
    this.slices.set(key(workspaceId, categoryName), {
      draft: null,
      loading: false,
      loaded: true,
      error: null
    });
  }

  // Migrates only the cache slot (caller moves the IDB row in its cross-store tx) so the renamed
  // pane reads the draft without an IDB round-trip.
  renameCategory(workspaceId: Uuid, oldName: string, newName: string): void {
    if (oldName === newName) return;
    const oldKey = key(workspaceId, oldName);
    const slice = this.slices.get(oldKey);
    if (!slice) return;
    this.slices.set(key(workspaceId, newName), {
      ...slice,
      draft: slice.draft ? { ...slice.draft, category_name: newName } : null
    });
    this.slices.delete(oldKey);
  }

  // Evict cache slices without touching IDB (rows GC'd separately on workspace deletion).
  forget(workspaceId: Uuid, categoryName?: string): void {
    if (categoryName !== undefined) {
      this.slices.delete(key(workspaceId, categoryName));
      return;
    }
    // Iterating SvelteMap is reactive on size, not membership, so this is safe in an effect.
    const prefix = `${workspaceId}\0`;
    for (const k of Array.from(this.slices.keys())) {
      if (k.startsWith(prefix)) this.slices.delete(k);
    }
  }
}

export const drafts = new DraftsStore();
