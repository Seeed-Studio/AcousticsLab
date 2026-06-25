<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import { SvelteSet } from 'svelte/reactivity';
  import Modal from '$lib/components/ui/Modal.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import DownloadIcon from '$lib/components/ui/DownloadIcon.svelte';
  import { categories } from '$lib/stores/categories.svelte';
  import { slices } from '$lib/stores/slices.svelte';
  import { prettyCategoryName } from '$lib/components/category/labels';
  import {
    exportWorkspace,
    WorkspaceExportError,
    type WorkspaceExportProgress
  } from '$lib/api/workspace-export';
  import { workspaces as workspacesApi } from '$lib/api/endpoints';
  import { errorCopy } from '$lib/utils/error-copy';
  import { formatBytes } from '$lib/utils/format';
  import { m } from '$lib/i18n';
  import type { HeadRecord, Uuid } from '$lib/api/types';

  // `.alpkg` export dialog. No `done` surface: success fires browser SaveAs and closes same
  // tick; failure rolls back to `selecting` with a retry banner. Owns its data-loading so any
  // surface can open it cold: heads + revision come from the caller or are self-fetched on open.

  interface Props {
    open: boolean;
    workspaceId: Uuid;
    workspaceName: string;
    /// Skips the self-fetch. Supply both or neither (both come from one WorkspaceDetail;
    /// partial supply is a caller bug); absent => fetch on open.
    heads?: readonly HeadRecord[];
    workspaceRevisionId?: number;
    onclose: () => void;
  }
  let {
    open,
    workspaceId,
    workspaceName,
    heads: headsProp,
    workspaceRevisionId: revisionIdProp,
    onclose
  }: Props = $props();

  // Self-fetch slots: populated only when the caller supplies neither prop, else null.
  let fetchedHeads = $state<readonly HeadRecord[] | null>(null);
  let fetchedRevisionId = $state<number | null>(null);
  let loadingDetail = $state(false);
  let loadDetailError = $state<string | null>(null);

  const heads = $derived<readonly HeadRecord[]>(headsProp ?? fetchedHeads ?? []);
  const workspaceRevisionId = $derived<number | null>(revisionIdProp ?? fetchedRevisionId);

  // Gates a placeholder so an unhydrated self-fetch doesn't read as an empty workspace.
  const isInitialLoading = $derived(
    headsProp === undefined && fetchedHeads === null && loadDetailError === null
  );

  const allCategories = $derived(categories.for(workspaceId).entries);

  function sliceCountFor(name: string): number {
    return slices.countFor(workspaceId, name);
  }

  // SvelteSets (not bare Sets) so mutations stay reactive without clone-and-replace.
  const selectedCategories = new SvelteSet<string>();
  const selectedHeadIds = new SvelteSet<Uuid>();

  // Set by any manual category toggle so auto-top-up stops re-adding deliberate deselects.
  let selectionDirty = $state(false);

  // Open-edge latch: reset state once on the open transition, not on every reactive read
  // (e.g. `heads`) inside the effect, which would clobber operator selections.
  let lastOpenSeen = $state(false);
  // Separate latch: heads hydrate asynchronously, so bulk-select them ONCE on first arrival
  // without re-adding later deselects.
  let initialHeadsSelected = $state(false);

  type PipelineState = 'selecting' | 'running';
  let pipelineState = $state<PipelineState>('selecting');
  let progress = $state<WorkspaceExportProgress | null>(null);
  let errorMessage = $state<string | null>(null);
  let errorCategory = $state<string | null>(null);
  let errorHeadId = $state<Uuid | null>(null);
  let abortController = $state<AbortController | null>(null);

  $effect(() => {
    if (open && !lastOpenSeen) {
      lastOpenSeen = true;
      selectedCategories.clear();
      selectedHeadIds.clear();
      initialHeadsSelected = false;
      selectionDirty = false;
      pipelineState = 'selecting';
      progress = null;
      errorMessage = null;
      errorCategory = null;
      errorHeadId = null;
      abortController = null;
      // Reset self-load slots so a re-open re-fetches fresh detail, not stale data.
      fetchedHeads = null;
      fetchedRevisionId = null;
      loadingDetail = false;
      loadDetailError = null;
    } else if (!open && lastOpenSeen) {
      lastOpenSeen = false;
      // Abort any in-flight export on dismissal (Escape, backdrop, explicit close).
      if (abortController !== null) {
        abortController.abort();
        abortController = null;
      }
    }
  });

  $effect(() => {
    if (!open) return;
    if (initialHeadsSelected) return;
    const hs = heads;
    if (hs.length === 0) return;
    initialHeadsSelected = true;
    untrack(() => {
      for (const head of hs) selectedHeadIds.add(head.head_id);
    });
  });

  // Self-load + store-refresh, gated by `lastOpenSeen` so an already-open re-render doesn't
  // re-trigger.
  $effect(() => {
    if (!open || !lastOpenSeen) return;
    untrack(() => {
      void categories.refresh(workspaceId);
      if (headsProp !== undefined && revisionIdProp !== undefined) {
        return;
      }
      if (fetchedHeads !== null || loadingDetail || loadDetailError !== null) {
        return;
      }
      loadingDetail = true;
      // No "still open?" gate in .then/.catch: the open-edge reset clears these slots before
      // any UI reads a stale write (persistent detail-page caller), and the mount-gated
      // list-page caller destroys us on close.
      workspacesApi
        .get(workspaceId)
        .then((detail) => {
          fetchedHeads = detail.heads;
          fetchedRevisionId = detail.workspace_revision.id;
        })
        .catch((e: unknown) => {
          loadDetailError = errorCopy(e);
        })
        .finally(() => {
          loadingDetail = false;
        });
    });
  });

  interface CategoryRow {
    /// Wire-form name (on-disk directory + selection key), routed into the export verbatim so
    /// the archive's path layout matches the daemon's.
    name: string;
    display: string;
    count: number;
    disabled: boolean;
  }
  interface HeadRow {
    head_id: Uuid;
    short_id: string;
    n_classes: number;
    size_bytes: number;
  }

  const categoryRows = $derived<CategoryRow[]>(
    allCategories.map((cat) => {
      const count = sliceCountFor(cat.name);
      return {
        name: cat.name,
        display: prettyCategoryName(cat.name),
        count,
        disabled: count === 0
      };
    })
  );

  const headRows = $derived<HeadRow[]>(
    heads.map((h) => ({
      head_id: h.head_id,
      short_id: h.head_id.replace(/-/g, '').slice(0, 8),
      n_classes: h.n_classes,
      size_bytes: h.size_bytes
    }))
  );

  // Slice-count reconcile, gated until the category list hydrates AND the revision is known;
  // the refresh short-circuits an already-synced revision, untrack avoids the self-loop.
  $effect(() => {
    if (!open) return;
    const cats = categoryRows;
    const rev = workspaceRevisionId;
    if (cats.length === 0 || rev === null) return;
    untrack(() => {
      void slices.refreshForWorkspace(
        workspaceId,
        cats.map((c) => c.name),
        rev
      );
    });
  });

  // Auto-top-up: counts hydrate asynchronously (every row can be 0/disabled on the first tick
  // after navigation), so re-add non-disabled rows on each categoryRows update until
  // `selectionDirty` flips, letting a deliberate deselect survive a late count.
  $effect(() => {
    if (!open || selectionDirty) return;
    const rows = categoryRows;
    untrack(() => {
      for (const row of rows) {
        if (!row.disabled && !selectedCategories.has(row.name)) {
          selectedCategories.add(row.name);
        }
      }
    });
  });

  const selectableCategoryRows = $derived(categoryRows.filter((r) => !r.disabled));
  const selectedCategoryCount = $derived(
    selectableCategoryRows.filter((r) => selectedCategories.has(r.name)).length
  );
  const allCategoriesSelected = $derived(
    selectableCategoryRows.length > 0 && selectedCategoryCount === selectableCategoryRows.length
  );

  const selectedHeadCount = $derived(headRows.filter((r) => selectedHeadIds.has(r.head_id)).length);
  const allHeadsSelected = $derived(headRows.length > 0 && selectedHeadCount === headRows.length);

  // Export gate: >=1 item selected AND no self-fetch failure (else an unloadable workspace
  // hits a confusing deep "Nothing to export" instead of an upfront error). isInitialLoading
  // is excluded so a disabled-button hover reads as "loading", not "no selection".
  const canExport = $derived(
    (selectedCategoryCount > 0 || selectedHeadCount > 0) &&
      loadDetailError === null &&
      !isInitialLoading
  );

  // The export omits pending/uploading/failed slices, so the archive count may undercut the
  // on-screen count; drives a pre-flight hint.
  const hasPendingInSelection = $derived(
    categoryRows.some((r) => {
      if (!selectedCategories.has(r.name)) return false;
      const status = slices.syncStatusFor(workspaceId, r.name);
      return status === 'pending' || status === 'uploading' || status === 'failed';
    })
  );

  function toggleAllCategories(): void {
    selectionDirty = true;
    if (allCategoriesSelected) {
      selectedCategories.clear();
    } else {
      for (const r of selectableCategoryRows) selectedCategories.add(r.name);
    }
  }

  function toggleCategory(name: string): void {
    selectionDirty = true;
    if (selectedCategories.has(name)) selectedCategories.delete(name);
    else selectedCategories.add(name);
  }

  function toggleAllHeads(): void {
    if (allHeadsSelected) {
      selectedHeadIds.clear();
    } else {
      for (const r of headRows) selectedHeadIds.add(r.head_id);
    }
  }

  function toggleHead(id: Uuid): void {
    if (selectedHeadIds.has(id)) selectedHeadIds.delete(id);
    else selectedHeadIds.add(id);
  }

  async function startExport(): Promise<void> {
    if (pipelineState === 'running') return;
    if (!canExport) return;
    pipelineState = 'running';
    progress = { phase: 'preparing-workspace' };
    errorMessage = null;
    errorCategory = null;
    errorHeadId = null;
    const controller = new AbortController();
    abortController = controller;
    try {
      await exportWorkspace(
        {
          workspaceId,
          workspaceName,
          categories: Array.from(selectedCategories),
          heads: heads.filter((h) => selectedHeadIds.has(h.head_id))
        },
        {
          signal: controller.signal,
          onprogress: (p) => {
            progress = p;
          }
        }
      );
      onclose();
    } catch (e) {
      if (controller.signal.aborted) {
        // Silent rollback, no error banner.
        pipelineState = 'selecting';
        progress = null;
        return;
      }
      if (e instanceof WorkspaceExportError) {
        errorMessage = e.message;
        errorCategory = e.category;
        errorHeadId = e.headId;
      } else {
        errorMessage = errorCopy(e);
      }
      pipelineState = 'selecting';
      progress = null;
    } finally {
      if (abortController === controller) abortController = null;
    }
  }

  function cancelRunning(): void {
    if (pipelineState !== 'running') return;
    if (abortController !== null) abortController.abort();
  }

  // A parent unmount (route change, deletion) has no open->closed transition, so abort here
  // too, else the streaming export runs to completion against dead $state.
  onDestroy(() => {
    abortController?.abort();
  });

  const progressCopy = $derived.by((): string => {
    if (progress === null) return '';
    const t = m.workspace.export_dialog;
    switch (progress.phase) {
      case 'preparing-workspace':
        return t.progress_preparing_workspace;
      case 'preparing-datasets':
        if (progress.subphase === 'fetching') {
          if (typeof progress.itemsTotal === 'number' && typeof progress.itemsDone === 'number') {
            return t.progress_fetched_slices(progress.itemsDone, progress.itemsTotal);
          }
          return t.progress_fetching_slices;
        }
        return t.progress_listing_slices;
      case 'preparing-heads':
        if (typeof progress.itemsTotal === 'number' && typeof progress.itemsDone === 'number') {
          return t.progress_validated_heads(progress.itemsDone, progress.itemsTotal);
        }
        return t.progress_validating_heads;
      case 'packing':
        return t.progress_packing;
      case 'downloading':
        return t.progress_downloading;
    }
  });

  const progressFraction = $derived.by((): number => {
    if (progress === null) return 0;
    const total = progress.itemsTotal ?? 0;
    const done = progress.itemsDone ?? 0;
    if (total <= 0) return 0;
    return Math.min(1, done / total);
  });

  const errorHeadline = $derived.by((): string => {
    const t = m.workspace.export_dialog;
    if (errorCategory !== null) {
      // Same display formatter as the rows so the banner names the label the operator saw.
      return t.error_in_category(prettyCategoryName(errorCategory));
    }
    if (errorHeadId !== null) {
      const shortId = errorHeadId.replace(/-/g, '').slice(0, 8);
      return t.error_for_head(shortId);
    }
    return t.error_default;
  });
</script>

<!-- Name in the title so the dialog self-identifies when launched from a surface that doesn't
     name the workspace (e.g. list-page right-click). No truncation: names cap at 128 UTF-8
     bytes and the Modal `<h2>` wraps better than a mid-name ellipsis. -->
<Modal
  {open}
  title={m.workspace.export_dialog.title(workspaceName)}
  onclose={() => {
    if (pipelineState === 'running') cancelRunning();
    onclose();
  }}
  closeOnBackdrop={pipelineState !== 'running'}
  class="max-w-lg"
>
  {#if errorMessage !== null}
    <div
      class="rounded-md border border-danger-line bg-danger-soft px-3 py-2 text-xs text-danger-soft-fg"
      role="alert"
    >
      <p class="font-medium">{errorHeadline}</p>
      <p class="mt-0.5">{errorMessage}</p>
    </div>
  {/if}

  <!-- Separate from the errorMessage banner: "couldn't load" vs "couldn't run the export". -->
  {#if loadDetailError !== null}
    <div
      class="rounded-md border border-danger-line bg-danger-soft px-3 py-2 text-xs text-danger-soft-fg"
      role="alert"
    >
      <p class="font-medium">{m.workspace.export_dialog.load_error_title}</p>
      <p class="mt-0.5">{loadDetailError}</p>
    </div>
  {/if}

  {#if isInitialLoading}
    <!-- Placeholder so "nothing to export" doesn't flash before heads land. No spinner: the
         modal fade-in already telegraphs transient state at the fetch's few-ms scale. -->
    <p class="text-xs text-fg-muted">{m.workspace.export_dialog.loading}</p>
  {:else if categoryRows.length === 0 && headRows.length === 0}
    <p class="text-xs text-fg-muted">
      {m.workspace.export_dialog.nothing_to_export}
    </p>
  {/if}

  {#if categoryRows.length > 0}
    <section class="flex flex-col gap-2">
      <header class="flex items-center justify-between">
        <h3 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
          {m.workspace.export_dialog.datasets_heading}
        </h3>
        <div class="flex items-center gap-2">
          <span class="text-[11px] text-fg-muted">
            {selectedCategoryCount} / {selectableCategoryRows.length}
          </span>
          {#if selectableCategoryRows.length > 1}
            <button
              type="button"
              onclick={toggleAllCategories}
              disabled={pipelineState === 'running'}
              class="text-[11px] font-medium text-fg-secondary transition hover:text-fg disabled:cursor-not-allowed disabled:opacity-50"
            >
              {allCategoriesSelected
                ? m.workspace.export_dialog.deselect_all
                : m.workspace.export_dialog.select_all}
            </button>
          {/if}
        </div>
      </header>

      <ul class="flex flex-col gap-1.5">
        {#each categoryRows as row (row.name)}
          <li>
            <label
              class="flex cursor-pointer items-center justify-between gap-2 rounded-md border border-line px-3 py-1.5 text-xs hover:bg-surface-2"
              class:opacity-60={row.disabled}
              class:cursor-not-allowed={row.disabled || pipelineState === 'running'}
            >
              <span class="flex min-w-0 items-center gap-2">
                <input
                  type="checkbox"
                  checked={selectedCategories.has(row.name)}
                  disabled={row.disabled || pipelineState === 'running'}
                  onchange={() => toggleCategory(row.name)}
                  class="h-3.5 w-3.5 shrink-0 cursor-pointer disabled:cursor-not-allowed"
                />
                <!-- title = wire-form name so the row reconciles with the archive's on-disk
                     path (`datasets/<wire-name>/`). -->
                <span class="truncate text-fg-secondary" title={row.name}>{row.display}</span>
              </span>
              <span class="shrink-0 font-mono text-[10px] tabular-nums text-fg-muted">
                {#if row.disabled}
                  {m.workspace.export_dialog.row_empty}
                {:else}
                  {m.workspace.export_dialog.row_slice_count(row.count)}
                {/if}
              </span>
            </label>
          </li>
        {/each}
      </ul>
    </section>
  {/if}

  {#if headRows.length > 0}
    <section class="flex flex-col gap-2">
      <header class="flex items-center justify-between">
        <h3 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
          {m.workspace.export_dialog.heads_heading}
        </h3>
        <div class="flex items-center gap-2">
          <span class="text-[11px] text-fg-muted">
            {selectedHeadCount} / {headRows.length}
          </span>
          {#if headRows.length > 1}
            <button
              type="button"
              onclick={toggleAllHeads}
              disabled={pipelineState === 'running'}
              class="text-[11px] font-medium text-fg-secondary transition hover:text-fg disabled:cursor-not-allowed disabled:opacity-50"
            >
              {allHeadsSelected
                ? m.workspace.export_dialog.deselect_all
                : m.workspace.export_dialog.select_all}
            </button>
          {/if}
        </div>
      </header>

      <ul class="flex flex-col gap-1.5">
        {#each headRows as row (row.head_id)}
          <li>
            <label
              class="flex cursor-pointer items-center gap-3 rounded-md border border-line px-3 py-1.5 text-xs hover:bg-surface-2"
              class:cursor-not-allowed={pipelineState === 'running'}
            >
              <span class="flex shrink-0 items-center gap-2">
                <input
                  type="checkbox"
                  checked={selectedHeadIds.has(row.head_id)}
                  disabled={pipelineState === 'running'}
                  onchange={() => toggleHead(row.head_id)}
                  class="h-3.5 w-3.5 shrink-0 cursor-pointer disabled:cursor-not-allowed"
                />
                <!-- Lowercase hex (no `uppercase` class) so the chip matches the UUID's
                     wire-form casing, reading identically to the title tooltip. -->
                <span
                  class="shrink-0 rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] tracking-wider text-fg-secondary"
                  title={row.head_id}
                >
                  {row.short_id}
                </span>
              </span>
              <!-- Meta strip (size · classes): `min-w-0 overflow-hidden` plus `shrink-0` on
                   each span keeps both tokens whole and yields remaining space to the id chip. -->
              <span
                class="ml-auto flex min-w-0 items-center gap-2 overflow-hidden font-mono text-[10px] tabular-nums text-fg-muted"
                title={m.workspace.export_dialog.head_meta_title(
                  formatBytes(row.size_bytes),
                  row.n_classes
                )}
              >
                <span class="shrink-0">{formatBytes(row.size_bytes)}</span>
                <span aria-hidden="true" class="shrink-0 text-fg-subtle">·</span>
                <span class="shrink-0">
                  {m.workspace.export_dialog.head_meta_classes(row.n_classes)}
                </span>
              </span>
            </label>
          </li>
        {/each}
      </ul>
    </section>
  {/if}

  {#if hasPendingInSelection && pipelineState === 'selecting'}
    <p class="text-[11px] text-fg-muted">
      {m.workspace.export_dialog.pending_warning}
    </p>
  {/if}

  {#if pipelineState === 'running'}
    <div class="flex flex-col gap-1.5">
      <p class="text-xs text-fg-secondary">{progressCopy}</p>
      <div class="h-1 overflow-hidden rounded-full bg-surface-2">
        <div
          class="h-full bg-accent transition-[width] duration-200"
          style="width: {Math.round(progressFraction * 100)}%"
          aria-hidden="true"
        ></div>
      </div>
    </div>
  {/if}

  {#snippet footer()}
    {#if pipelineState === 'running'}
      <Button variant="secondary" onclick={cancelRunning}>{m.common.cancel}</Button>
      <Button disabled loading>{m.workspace.export_dialog.exporting}</Button>
    {:else}
      <Button variant="secondary" onclick={onclose}>{m.common.cancel}</Button>
      <Button
        onclick={() => void startExport()}
        disabled={!canExport}
        ariaLabel={m.workspace.export_dialog.export_aria}
      >
        <DownloadIcon />
        {m.workspace.export_dialog.export_button}
      </Button>
    {/if}
  {/snippet}
</Modal>
