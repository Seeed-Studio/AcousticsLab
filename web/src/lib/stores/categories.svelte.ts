import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import { assets } from '$lib/api/endpoints';
import { enqueueDelete } from '$lib/api/delete-queue';
import { awaitJobTerminal } from '$lib/api/jobs';
import { errorCopy, isNotFound } from '$lib/utils/error-copy';
import {
  deleteCategoryRecord,
  listCategoriesForWorkspace,
  putCategoryRecord
} from '$lib/idb/categories';
import { renameCategoryForWorkspace } from '$lib/idb/db';
import { deleteDraft } from '$lib/idb/drafts';
import { deleteSlicesForCategory } from '$lib/idb/slices';
import { drafts } from '$lib/stores/drafts.svelte';
import { slices } from '$lib/stores/slices.svelte';
import { MANDATORY_BACKGROUND_NOISE, isMandatoryCategory } from '$lib/components/category/labels';
import { findCaseInsensitiveDuplicate } from '$lib/components/category/name-validate';
import { m } from '$lib/i18n';
import type { DatasetListing, RenameCategoryResp, Uuid } from '$lib/api/types';

// Per-workspace cache merging by byte-equal name (precedence server > idb > mandatory, mandatory
// first): synthetic `_background_noise_`, IDB rows (persisted to survive reload before any upload,
// since the daemon won't list an empty dir), and lazy server dirs under `<workspace>/datasets/`.
// SvelteMap values aren't deeply reactive, so every mutation replaces the slice object reference.

export type CategoryOrigin = 'mandatory' | 'idb' | 'server';

export interface Category {
  name: string;
  origin: CategoryOrigin;
}

interface WorkspaceSlice {
  entries: Category[];
  // At most one open at a time, null = all collapsed.
  expandedName: string | null;
  // Names with an async daemon DELETE in flight; UI dims the row until the job's SSE terminal.
  deleting: SvelteSet<string>;
  loading: boolean;
  loaded: boolean;
  error: string | null;
}

// Spread base for `this.slices` writes; frozen-shareable since every spreading literal overrides all
// mutable fields (incl. the `deleting` SvelteSet). Cache misses get a fresh `makeEmptySlice()`.
const EMPTY_SLICE: Readonly<WorkspaceSlice> = Object.freeze({
  entries: [] as Category[],
  expandedName: null as string | null,
  deleting: new SvelteSet<string>(),
  loading: false,
  loaded: false,
  error: null as string | null
});

function makeEmptySlice(): WorkspaceSlice {
  return {
    entries: [],
    expandedName: null,
    deleting: new SvelteSet<string>(),
    loading: false,
    loaded: false,
    error: null
  };
}

function sortCategories(entries: Category[]): Category[] {
  return entries.slice().sort((a, b) => {
    if (a.name === MANDATORY_BACKGROUND_NOISE) return -1;
    if (b.name === MANDATORY_BACKGROUND_NOISE) return 1;
    return a.name.localeCompare(b.name);
  });
}

function mergeSources(idbNames: string[], serverNames: string[]): Category[] {
  // SvelteMap not bare Map only to satisfy the `.svelte.ts` lint rule that flags Map even here.
  const seen = new SvelteMap<string, Category>();
  seen.set(MANDATORY_BACKGROUND_NOISE, {
    name: MANDATORY_BACKGROUND_NOISE,
    origin: 'mandatory'
  });
  for (const name of idbNames) {
    if (!seen.has(name)) seen.set(name, { name, origin: 'idb' });
  }
  for (const name of serverNames) {
    seen.set(name, {
      name,
      origin: name === MANDATORY_BACKGROUND_NOISE ? 'mandatory' : 'server'
    });
  }
  return sortCategories(Array.from(seen.values()));
}

class CategoriesStore {
  private slices = new SvelteMap<Uuid, WorkspaceSlice>();
  // Separate from `slices` so the consuming effect tracks the stale bit alone, without re-firing on
  // every internal `slices.set` the refresh / mutation paths make.
  private staleWorkspaces = new SvelteSet<Uuid>();

  // Fresh (not shared) empty slice on miss so consumers need no null guards (unloaded-vs-empty lives on
  // `loaded`) and their mutating its `deleting` set can't leak badges across unloaded workspaces.
  for(workspaceId: Uuid): WorkspaceSlice {
    return this.slices.get(workspaceId) ?? makeEmptySlice();
  }

  isStale(workspaceId: Uuid): boolean {
    return this.staleWorkspaces.has(workspaceId);
  }

  // No-op on a never-loaded list (nothing to invalidate).
  markStale(workspaceId: Uuid): void {
    if (this.slices.has(workspaceId)) this.staleWorkspaces.add(workspaceId);
  }

  // `force` re-fetches regardless. The in-flight guard is correctness not perf: the loading-flag write
  // invalidates a tracking effect reading `this.slices.get(...)`, which else re-enters to Svelte's fuse.
  async refresh(workspaceId: Uuid, force = false): Promise<void> {
    const existing = this.slices.get(workspaceId);
    const stale = this.staleWorkspaces.has(workspaceId);
    if (existing?.loaded && !force && !existing.error && !stale) return;
    if (existing?.loading && !force) return;

    // Keep prior `entries` so the list doesn't blink to empty mid-fetch.
    this.slices.set(workspaceId, {
      ...EMPTY_SLICE,
      entries: existing?.entries ?? [],
      expandedName: existing?.expandedName ?? null,
      deleting: existing?.deleting ?? new SvelteSet<string>(),
      loading: true,
      loaded: existing?.loaded ?? false
    });

    try {
      // `listDatasets` 404s on a fresh workspace (daemon creates `datasets/` on first upload); treat
      // 404 as "no server categories yet" so mandatory + IDB rows still surface.
      const emptyListing: DatasetListing = {
        entries: [],
        total: 0,
        offset: 0,
        limit: 1000
      };
      const [idbRows, serverListing] = await Promise.all([
        listCategoriesForWorkspace(workspaceId),
        assets.listDatasets(workspaceId, { limit: 1000 }).catch((e: unknown) => {
          if (isNotFound(e)) return emptyListing;
          throw e;
        })
      ]);
      const idbNames = idbRows.map((r) => r.name);
      // Only directories are categories (wire token `"directory"`); a stray file never renders.
      const serverNames = serverListing.entries
        .filter((e) => e.kind === 'directory')
        .map((e) => e.name);

      // Bail if `forget` cleared this during the await; else source `expandedName` / `deleting` from
      // the LIVE re-read (a concurrent toggle/delete may have moved them), not `?? existing` which
      // would restore a stale name over a legitimate `null`.
      const live = this.slices.get(workspaceId);
      if (!live) return;
      this.slices.set(workspaceId, {
        ...EMPTY_SLICE,
        entries: mergeSources(idbNames, serverNames),
        expandedName: live.expandedName,
        deleting: live.deleting,
        loading: false,
        loaded: true,
        error: null
      });
      this.staleWorkspaces.delete(workspaceId);
    } catch (e) {
      const live = this.slices.get(workspaceId);
      if (!live) return;
      this.slices.set(workspaceId, {
        ...EMPTY_SLICE,
        entries: live.entries.length > 0 ? live.entries : mergeSources([], []),
        expandedName: live.expandedName,
        deleting: live.deleting,
        loading: false,
        loaded: existing?.loaded ?? true,
        error: errorCopy(e)
      });
    }
  }

  // Caller owns AssetPath shape validation; this checks only uniqueness.
  async create(workspaceId: Uuid, name: string): Promise<void> {
    const existing = this.slices.get(workspaceId);
    if (existing?.entries.some((c) => c.name === name)) {
      throw new Error(m.category.add_dialog.error_exact_duplicate);
    }
    // Re-check case-insensitively at this sole write boundary: a byte-exact check alone lets a
    // cross-tab / refresh race persist an FS-colliding `Foo` vs `foo`.
    const ciDup = findCaseInsensitiveDuplicate(
      name,
      (existing?.entries ?? []).map((c) => c.name)
    );
    if (ciDup !== null) {
      throw new Error(m.category.add_dialog.error_case_insensitive_duplicate(ciDup));
    }
    await putCategoryRecord({
      workspace_id: workspaceId,
      name,
      created_at: new Date().toISOString()
    });
    // Build from the LIVE re-read so a concurrent mutation isn't clobbered (bail if `forget` cleared
    // it); dedup by name in case a concurrent refresh already surfaced this category server-side.
    const live = this.slices.get(workspaceId);
    if (!live) return;
    const newCat: Category = { name, origin: 'idb' };
    const entries = live.entries.some((c) => c.name === name)
      ? live.entries
      : sortCategories([...live.entries, newCat]);
    this.slices.set(workspaceId, {
      ...EMPTY_SLICE,
      entries,
      expandedName: live.expandedName,
      deleting: live.deleting,
      loaded: true,
      loading: false,
      error: null
    });
  }

  // Three paths: mandatory rejected unless `force` (defence in depth over the UI gate); IDB-only drops
  // the row, no daemon call; server-side or mandatory+force DELETEs via the global queue, awaits the
  // SSE terminal, drops both reps. `force` swallows a 404 (nothing to wipe satisfies wipe-and-reimport)
  // and keeps mandatory rows in `entries` (merge re-synthesises them) so the UI never flashes empty.
  async delete(workspaceId: Uuid, name: string, options: { force?: boolean } = {}): Promise<void> {
    const force = options.force ?? false;
    const mandatory = isMandatoryCategory(name);
    if (mandatory && !force) {
      throw new Error(m.category.delete_dialog.error_mandatory_required);
    }
    const slice = this.slices.get(workspaceId);
    const target = slice?.entries.find((c) => c.name === name);
    if (!slice || !target) throw new Error(m.category.delete_dialog.error_not_found);

    if (target.origin === 'idb') {
      await deleteCategoryRecord(workspaceId, name);
      // Drop the in-progress draft + local slices so they don't orphan in IDB; best-effort.
      await drafts.clear(workspaceId, name).catch(() => undefined);
      await slices.clearForCategory(workspaceId, name).catch(() => undefined);
      const fresh = this.slices.get(workspaceId);
      if (!fresh) return;
      this.slices.set(workspaceId, {
        ...fresh,
        entries: fresh.entries.filter((c) => c.name !== name),
        expandedName: fresh.expandedName === name ? null : fresh.expandedName
      });
      return;
    }

    // Skip the `deleting` badge for synthesised mandatory rows (confusing mid-wipe).
    if (!mandatory) {
      const startingDeleting = new SvelteSet(slice.deleting);
      startingDeleting.add(name);
      this.slices.set(workspaceId, { ...slice, deleting: startingDeleting });
    }
    // Bracket so the poller defers its revision check while the job is in flight; the slices store's
    // counter is one source of truth across slice + category mutations.
    slices.beginMutation(workspaceId);

    try {
      await enqueueDelete(async () => {
        try {
          await this.runRemoteDelete(workspaceId, name);
        } catch (e) {
          if (force && isNotFound(e)) return;
          throw e;
        }
      });
      // Folder gone server-side: drop the now-unreachable IDB shadow + draft + slices.
      await deleteCategoryRecord(workspaceId, name).catch(() => undefined);
      await drafts.clear(workspaceId, name).catch(() => undefined);
      await slices.clearForCategory(workspaceId, name).catch(() => undefined);
      const fresh = this.slices.get(workspaceId);
      if (fresh) {
        const deleting = new SvelteSet(fresh.deleting);
        deleting.delete(name);
        this.slices.set(workspaceId, {
          ...fresh,
          entries: mandatory ? fresh.entries : fresh.entries.filter((c) => c.name !== name),
          expandedName: fresh.expandedName === name ? null : fresh.expandedName,
          deleting
        });
      }
    } catch (e) {
      const fresh = this.slices.get(workspaceId);
      if (fresh) {
        const deleting = new SvelteSet(fresh.deleting);
        deleting.delete(name);
        this.slices.set(workspaceId, { ...fresh, deleting });
      }
      throw e;
    } finally {
      slices.endMutation(workspaceId);
    }
  }

  private async runRemoteDelete(workspaceId: Uuid, name: string): Promise<void> {
    const ack = await assets.deleteCategory(workspaceId, name);
    await awaitJobTerminal(ack.job_id);
  }

  // MOVES rows instead of wiping. IDB-only re-keys IDB + in-memory, no daemon call; server-side renames
  // the daemon directory FIRST (canonical under server-wins, so on failure IDB stays untouched and the
  // next refresh no-ops) then re-keys. Daemon rename is SYNCHRONOUS (atomic dir rename), no enqueue.
  async rename(workspaceId: Uuid, oldName: string, newName: string): Promise<void> {
    if (newName === oldName) return;
    if (isMandatoryCategory(oldName)) {
      throw new Error(m.category.rename_dialog.error_mandatory);
    }

    const slice = this.slices.get(workspaceId);
    const target = slice?.entries.find((c) => c.name === oldName);
    if (!slice || !target) throw new Error(m.category.delete_dialog.error_not_found);

    // Uniqueness against OTHER categories (exclude own name so a case-only rename isn't a
    // self-collision); re-checked here, not just the dialog, to close a cross-tab / refresh race.
    const others = slice.entries.filter((c) => c.name !== oldName).map((c) => c.name);
    if (others.includes(newName)) {
      throw new Error(m.category.add_dialog.error_exact_duplicate);
    }
    const ciDup = findCaseInsensitiveDuplicate(newName, others);
    if (ciDup !== null) {
      throw new Error(m.category.add_dialog.error_case_insensitive_duplicate(ciDup));
    }

    // Refuse while an async DELETE targeting the OLD directory is in flight: renaming out from under
    // it would race. Enforced at this sole write boundary, not just the menu.
    if (slice.deleting.has(oldName)) {
      throw new Error(m.category.rename_dialog.error_busy);
    }

    // An in-flight slice mutation bakes the OLD name (upload URL / DELETE path target it), so renaming
    // mid-flight re-creates / orphans under the old dir; refuse until uploads settle AND no delete runs.
    const status = slices.syncStatusFor(workspaceId, oldName);
    if (
      status === 'uploading' ||
      status === 'pending' ||
      status === 'failed' ||
      slices.hasInflightDeletes(workspaceId, oldName)
    ) {
      throw new Error(m.category.rename_dialog.error_busy);
    }

    if (target.origin === 'idb') {
      await renameCategoryForWorkspace(workspaceId, oldName, newName);
      drafts.renameCategory(workspaceId, oldName, newName);
      slices.renameCategory(workspaceId, oldName, newName);
      const fresh = this.slices.get(workspaceId);
      if (fresh) this.applyRename(workspaceId, fresh, oldName, newName, target.origin);
      return;
    }

    // Bracket so the poller defers its revision check while the rename is in flight.
    slices.beginMutation(workspaceId);
    try {
      const resp = await this.runRemoteRename(workspaceId, oldName, newName);
      // Advance only latestRevisions; let the poller's reconcile catch up lastSyncedRevisions (one
      // redundant listing) rather than auto-advancing the synced mirror here.
      slices.setRevisionAtLeast(workspaceId, resp.workspace_revision_id);
      // Daemon rename has COMMITTED and is canonical, so the local IDB re-key is best-effort, never
      // raising. On its atomic abort, drop every orphaned old-name IDB row (shadow else it ghosts
      // beside the new name in mergeSources; draft else an unreclaimable Blob; slices); the in-memory
      // move below still shows newName, and the poller reconcile rebuilds its slices from the daemon.
      try {
        await renameCategoryForWorkspace(workspaceId, oldName, newName);
      } catch {
        await deleteCategoryRecord(workspaceId, oldName).catch(() => undefined);
        await deleteDraft(workspaceId, oldName).catch(() => undefined);
        await deleteSlicesForCategory(workspaceId, oldName).catch(() => undefined);
      }
      drafts.renameCategory(workspaceId, oldName, newName);
      slices.renameCategory(workspaceId, oldName, newName);
      const fresh = this.slices.get(workspaceId);
      if (fresh) this.applyRename(workspaceId, fresh, oldName, newName, target.origin);
    } finally {
      slices.endMutation(workspaceId);
    }
  }

  private applyRename(
    workspaceId: Uuid,
    fresh: WorkspaceSlice,
    oldName: string,
    newName: string,
    origin: CategoryOrigin
  ): void {
    const entries = sortCategories(
      fresh.entries.map((c) => (c.name === oldName ? { name: newName, origin } : c))
    );
    let deleting = fresh.deleting;
    if (fresh.deleting.has(oldName)) {
      deleting = new SvelteSet(fresh.deleting);
      deleting.delete(oldName);
      deleting.add(newName);
    }
    this.slices.set(workspaceId, {
      ...fresh,
      entries,
      // Keep the pane open under the new name (delete collapses, rename keeps).
      expandedName: fresh.expandedName === oldName ? newName : fresh.expandedName,
      deleting
    });
  }

  private async runRemoteRename(
    workspaceId: Uuid,
    oldName: string,
    newName: string
  ): Promise<RenameCategoryResp> {
    return assets.renameCategory(workspaceId, oldName, newName);
  }

  toggleExpand(workspaceId: Uuid, name: string): void {
    const slice = this.slices.get(workspaceId);
    if (!slice) return;
    const next = slice.expandedName === name ? null : name;
    this.slices.set(workspaceId, { ...slice, expandedName: next });
  }

  collapseAll(workspaceId: Uuid): void {
    const slice = this.slices.get(workspaceId);
    if (!slice) return;
    if (slice.expandedName !== null) {
      this.slices.set(workspaceId, { ...slice, expandedName: null });
    }
  }

  // Drop per-workspace state on workspace delete (a separate step handles IDB cleanup); the LIVE
  // re-read guards above stop a concurrent in-flight refresh/mutation from re-creating this entry.
  forget(workspaceId: Uuid): void {
    this.slices.delete(workspaceId);
    this.staleWorkspaces.delete(workspaceId);
  }
}

export const categories = new CategoriesStore();
