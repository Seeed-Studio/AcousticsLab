import { SvelteSet } from 'svelte/reactivity';
import { isApiError } from '$lib/api/http';
import { workspaces as workspacesApi } from '$lib/api/endpoints';
import { awaitJobTerminal } from '$lib/api/jobs';
import { enqueueDelete } from '$lib/api/delete-queue';
import { deleteAllForWorkspace } from '$lib/idb/db';
import { categories as categoriesStore } from '$lib/stores/categories.svelte';
import { drafts as draftsStore } from '$lib/stores/drafts.svelte';
import { slices as slicesStore } from '$lib/stores/slices.svelte';
import { training as trainingStore } from '$lib/stores/training.svelte';
import { capFirst, errorCopy } from '$lib/utils/error-copy';
import type {
  WorkspaceCreateReq,
  WorkspaceListEntry,
  WorkspaceMutationResp,
  WorkspacePatchReq,
  Uuid
} from '$lib/api/types';

// UI-only hard cap (creation disabled at capacity); the daemon enforces no limit.
export const MAX_WORKSPACES = 16;

function byCreatedDesc(a: WorkspaceListEntry, b: WorkspaceListEntry): number {
  // Parse instants: variable-width fractional seconds break lexical compare (".1Z" > ".12Z").
  return Date.parse(b.created_at) - Date.parse(a.created_at);
}

export interface BulkDeleteFailure {
  id: Uuid;
  name: string;
  error: string;
}

export interface BulkDeleteOutcome {
  succeeded: number;
  failed: BulkDeleteFailure[];
}

class WorkspacesStore {
  // Raw source, sorted newest-first; both the Workspace and Converter tabs derive from it (one fetch).
  all = $state<WorkspaceListEntry[]>([]);
  deleting = new SvelteSet<Uuid>();
  // Pruned in refresh() (not just filtered) so direct selected.size readers stay accurate.
  selected = new SvelteSet<Uuid>();
  mode = $state<'normal' | 'selecting'>('normal');
  loading = $state(false);
  // False until first refresh() resolves: "no data yet" (Spinner) vs "really empty" (EmptyState).
  loaded = $state(false);
  error = $state<string | null>(null);

  // Delete-in-flight workspaces stay listed until the job lands so a failed delete doesn't flicker.
  entries = $derived(this.all);

  // Intersected with the live list so a workspace deleted elsewhere drops from this view.
  selectedEntries = $derived(this.all.filter((w) => this.selected.has(w.id)));

  async refresh(): Promise<void> {
    this.loading = true;
    try {
      const list = await workspacesApi.list();
      this.all = list.slice().sort(byCreatedDesc);
      for (const id of this.selected) {
        if (!this.all.some((w) => w.id === id)) this.selected.delete(id);
      }
      this.error = null;
    } catch (e) {
      this.error = errorCopy(e);
    } finally {
      this.loading = false;
      this.loaded = true;
    }
  }

  // Optimistic insert so the new card appears without a refresh round-trip.
  async create(req: WorkspaceCreateReq): Promise<WorkspaceMutationResp> {
    const resp = await workspacesApi.create(req);
    const entry: WorkspaceListEntry = {
      id: resp.id,
      name: resp.name,
      created_at: resp.created_at
    };
    this.all = [entry, ...this.all.filter((w) => w.id !== entry.id)].sort(byCreatedDesc);
    return resp;
  }

  // created_at is unchanged so sort order stays stable.
  async patch(id: Uuid, req: WorkspacePatchReq): Promise<WorkspaceMutationResp> {
    const resp = await workspacesApi.patch(id, req);
    this.all = this.all.map((w) =>
      w.id === id ? { id: resp.id, name: resp.name, created_at: resp.created_at } : w
    );
    return resp;
  }

  // enqueueDelete serializes against the daemon's single delete-family slot (max_delete_jobs = 1,
  // shared across categories/slices) and awaits the full lifecycle (queue -> ack -> SSE terminal).
  async delete(id: Uuid): Promise<void> {
    await enqueueDelete(() => this.runDelete(id));
  }

  private async runDelete(id: Uuid): Promise<void> {
    // Bracket as one mutation so the detail poller defers its revision check until we settle: the
    // daemon renames the tree before the ack, so an unbracketed poll could 404 and flash EmptyState.
    slicesStore.beginMutation(id);
    try {
      const ack = await workspacesApi.delete(id);
      this.deleting.add(id);
      try {
        await awaitJobTerminal(ack.job_id);
        this.all = this.all.filter((w) => w.id !== id);
        this.selected.delete(id);
        // categories.forget required or each delete leaks SvelteMap entries (asymmetric with IDB below).
        categoriesStore.forget(id);
        draftsStore.forget(id);
        slicesStore.forget(id);
        // forget only stops the local poller; the daemon's training task exits itself on WorkspaceDelete.
        trainingStore.forget(id);
        // Atomic IDB tx so a page close mid-cascade can't orphan rows a same-id recreate inherits;
        // catch is safe (stale local rows surface nowhere, resetDB is the escape hatch). Spectrogram
        // PNGs deliberately survive: sha256 content-addressed and shared, so a scoped evict could
        // drop rows another workspace still needs.
        await deleteAllForWorkspace(id).catch(() => undefined);
      } catch (e) {
        // Terminal failure leaves the workspace on disk; refresh so the list reflects truth.
        void this.refresh();
        throw e;
      } finally {
        this.deleting.delete(id);
      }
    } finally {
      slicesStore.endMutation(id);
    }
  }

  toggleSelect(id: Uuid): void {
    if (this.selected.has(id)) this.selected.delete(id);
    else this.selected.add(id);
  }

  // Skips delete-in-flight workspaces so "Select all" can't re-queue them.
  selectAllVisible(): void {
    for (const w of this.entries) {
      if (!this.deleting.has(w.id)) this.selected.add(w.id);
    }
  }

  clearSelection(): void {
    this.selected.clear();
  }

  enterSelecting(): void {
    if (this.mode !== 'selecting') this.mode = 'selecting';
  }

  exitSelecting(): void {
    if (this.mode !== 'normal') {
      this.mode = 'normal';
      // Clear so re-entering selecting mode starts empty, not with the last batch's checks.
      this.clearSelection();
    }
  }

  // Clears selection eagerly so a double-click can't double-fire; failed targets re-enter it for retry.
  async deleteSelected(
    targets: WorkspaceListEntry[] = this.selectedEntries.slice()
  ): Promise<BulkDeleteOutcome> {
    this.clearSelection();

    const failed: BulkDeleteFailure[] = [];
    let succeeded = 0;
    await Promise.all(
      targets.map(async (entry) => {
        try {
          await enqueueDelete(() => this.runDelete(entry.id));
          succeeded++;
        } catch (e) {
          // errorCopy strips ApiError's daemon prefix (e.g. `fs:`); capFirst handles the
          // already-clean SSE-terminal message awaitJobTerminal rejects with.
          const message = isApiError(e)
            ? errorCopy(e)
            : e instanceof Error && e.message
              ? capFirst(e.message, 'Delete failed.')
              : errorCopy(e);
          failed.push({ id: entry.id, name: entry.name, error: message });
        }
      })
    );

    for (const f of failed) this.selected.add(f.id);

    return { succeeded, failed };
  }
}

export const workspaces = new WorkspacesStore();
