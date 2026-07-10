<script lang="ts">
  import { onDestroy } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
  import { workspaces as wsApi } from '$lib/api/endpoints';
  import { errorCopy, isNotFound } from '$lib/utils/error-copy';
  import { m } from '$lib/i18n';
  import type { WorkspaceDetail, WorkspaceMutationResp } from '$lib/api/types';
  import LoadingRow from '$lib/components/LoadingRow.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import EmptyState from '$lib/components/ui/EmptyState.svelte';
  import ContextMenu, { type MenuSection } from '$lib/components/ui/ContextMenu.svelte';
  import DeleteWorkspaceDialog from '$lib/components/workspace/DeleteWorkspaceDialog.svelte';
  import RenameWorkspaceDialog from '$lib/components/workspace/RenameWorkspaceDialog.svelte';
  import WorkspaceToolIsland from '$lib/components/workspace/WorkspaceToolIsland.svelte';
  import { formatRelative } from '$lib/utils/time';
  import CategoryList from '$lib/components/category/CategoryList.svelte';
  import { slices } from '$lib/stores/slices.svelte';
  import { categories } from '$lib/stores/categories.svelte';
  import { WorkspacePoller } from '$lib/stores/workspace-poller';
  import TrainPane from '$lib/components/training/TrainPane.svelte';
  import DeployPane from '$lib/components/deploy/DeployPane.svelte';
  import { training as trainingStore } from '$lib/stores/training.svelte';
  import { config } from '$lib/stores/config.svelte';

  // Route-entry revalidation, as on the dashboard: DeployPane's deploy-state pill reads `config.active`.
  if (!config.loading) void config.refreshActive();

  let detail = $state<WorkspaceDetail | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let notFound = $state(false);
  // Drives the $effect's re-fetch-on-route-param-change guard.
  let lastId = $state<string | null>(null);

  const poller = new WorkspacePoller();

  async function load(id: string): Promise<void> {
    loading = true;
    error = null;
    notFound = false;
    // Adopt id BEFORE the await else a return-nav to this in-flight workspace
    // bails on the id===lastId guard and never polls.
    lastId = id;
    // Stop the prior poller pre-await so a stale tick can't land on the new detail.
    poller.stop();

    // Detail-independent mount reads, concurrent with the detail GET; they
    // swallow their own errors so only the detail catch drives notFound/error.
    void trainingStore.recover(id);
    void trainingStore.hydrateHistory(id);
    void categories.refresh(id);

    try {
      const fetched = await wsApi.get(id);
      // Stale-response guard (repeated in catch/finally): load() isn't request-
      // sequenced, so a late-resolving A must not overwrite B's detail or poll /B.
      if (page.params.id !== id) return;
      detail = fetched;
      poller.start(detail, {
        onDetail: (fresh) => {
          // Insurance over the poller's swap filter against a post-route-swap tick.
          if (lastId === fresh.id) detail = fresh;
        },
        onGone: () => {
          // Deleted under us: leave lastId on the gone id else the $effect re-fires a redundant 404 GET.
          detail = null;
          notFound = true;
        },
        onError: (e) => {
          console.warn('[workspace-poller] tick failed', e);
        }
      });
    } catch (e) {
      if (page.params.id !== id) return;
      detail = null;
      notFound = isNotFound(e);
      error = errorCopy(e);
    } finally {
      if (page.params.id === id) loading = false;
    }
  }

  $effect(() => {
    const id = page.params.id;
    if (id && id !== lastId) void load(id);
  });

  onDestroy(() => {
    poller.stop();
  });

  // Instance survives a route-param change: close any open menu on id change else it re-renders at the stale (x,y) anchor.
  $effect(() => {
    void page.params.id;
    menuOpen = false;
  });

  // The slices store stashes each upload receipt's workspace_revision_id so the
  // rev chip promotes (and revisionAdvanced lights the live badge) without a re-fetch.
  const sliceLatestRevision = $derived(detail ? slices.latestRevisionFor(detail.id) : null);
  const liveRevision = $derived(
    detail ? Math.max(detail.workspace_revision.id, sliceLatestRevision ?? 0) : 0
  );
  const revisionAdvanced = $derived(
    detail !== null &&
      sliceLatestRevision !== null &&
      sliceLatestRevision > detail.workspace_revision.id
  );

  // Re-pull detail without restarting the poller so head-list actions and the
  // training-terminal hook pick up a published/deleted head; a blip self-heals next tick.
  async function refreshDetail(): Promise<void> {
    const id = lastId;
    if (!id) return;
    try {
      const fresh = await wsApi.get(id);
      if (lastId === id) detail = fresh;
    } catch (e) {
      console.warn('[workspace] post-mutation refresh failed', e);
    }
  }

  // terminalSeq bumps on every terminal landing across all workspaces; refresh
  // only on this one's `completed` (failed/cancelled don't touch heads[]).
  // lastTerminalSeqSeen is plain `let`, not $state: this effect reads AND writes
  // it, and a reactive self-dependency would schedule a guard-skipped fire.
  let lastTerminalSeqSeen = 0;
  $effect(() => {
    const seq = trainingStore.terminalSeq;
    if (seq === lastTerminalSeqSeen) return;
    lastTerminalSeqSeen = seq;
    if (!detail) return;
    const t = trainingStore.terminalFor(detail.id);
    if (t?.view?.state === 'completed') void refreshDetail();
  });

  let renameOpen = $state(false);
  let exportOpen = $state(false);
  let importOpen = $state(false);
  let deleteOpen = $state(false);
  // Delete lives only in this right-click menu, not the tool island, so its high-cost-undo footprint must be opted into deliberately.
  let menuOpen = $state(false);
  let menuX = $state(0);
  let menuY = $state(0);
  let menuSections = $state<MenuSection[]>([]);

  function backToList(): void {
    void goto(resolve('/workspaces'));
  }

  function onRenamed(resp: WorkspaceMutationResp): void {
    // Rename PATCH is metadata-only (no revision bump): splice the name in, don't refetch.
    if (!detail) return;
    detail.name = resp.name;
  }

  function onPageContextMenu(e: MouseEvent): void {
    if (!detail) return;
    const target = e.target as HTMLElement | null;
    if (target?.closest('input, textarea')) return;
    e.preventDefault();
    menuX = e.clientX;
    menuY = e.clientY;
    const t = m.workspace.detail;
    menuSections = [
      {
        items: [
          {
            label: t.menu_rename,
            onclick: () => (renameOpen = true)
          },
          // Ungated: export is read-only; the dialog's pipelineState gate blocks concurrent SaveAs races.
          {
            label: t.menu_export,
            onclick: () => (exportOpen = true)
          },
          {
            label: t.menu_import,
            onclick: () => (importOpen = true)
          },
          {
            label: t.menu_delete,
            variant: 'destructive',
            onclick: () => (deleteOpen = true)
          }
        ]
      },
      {
        items: [
          {
            label: t.menu_back_to_list,
            onclick: backToList
          }
        ]
      }
    ];
    menuOpen = true;
  }
</script>

<svelte:head>
  <title
    >{detail
      ? m.routes.workspace_detail_title(detail.name, m.app.name)
      : m.routes.workspace_list_title(m.app.name)}</title
  >
</svelte:head>

<nav class="mb-4 text-xs text-fg-muted">
  <a href={resolve('/workspaces')} class="transition hover:text-fg"
    >{m.workspace.detail.back_link}</a
  >
</nav>

{#if loading && !detail}
  <LoadingRow label={m.workspace.detail.loading} />
{:else if notFound}
  <EmptyState
    title={m.workspace.detail.not_found_title}
    description={m.workspace.detail.not_found_description}
  >
    {#snippet action()}
      <Button onclick={backToList}>{m.workspace.detail.back_to_list_button}</Button>
    {/snippet}
  </EmptyState>
{:else if error && !detail}
  <div
    class="flex items-start gap-3 rounded-lg border border-warning-line bg-warning-soft px-4 py-3 text-xs"
  >
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2.5"
      class="h-4 w-4 shrink-0 animate-spin text-warning-soft-fg"
      aria-hidden="true"
    >
      <path d="M12 3a9 9 0 109 9" stroke-linecap="round" />
    </svg>
    <div class="min-w-0">
      <p class="font-medium text-warning-soft-fg">{m.workspace.detail.load_error_title}</p>
      <p class="mt-0.5 text-warning-soft-fg">{error}</p>
    </div>
  </div>
{:else if detail}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div oncontextmenu={onPageContextMenu}>
    <header class="mb-6 flex items-center justify-between gap-3">
      <div class="min-w-0 flex-1">
        <h1 class="truncate text-lg leading-tight font-semibold text-fg" title={detail.name}>
          {detail.name}
        </h1>
        <!-- · separators sit OUTSIDE every span so a hover maps 1:1 to its label's absolute-ISO title. -->
        <p class="mt-1 text-[11px] text-fg-muted">
          <span title={detail.created_at}
            >{m.workspace.detail.created_label(formatRelative(detail.created_at))}</span
          >
          · {m.workspace.detail.rev_label(liveRevision)} ·
          <span title={detail.workspace_revision.at}
            >{m.workspace.detail.modified_label(formatRelative(detail.workspace_revision.at))}</span
          >
          {#if revisionAdvanced}
            <span
              class="ml-1 rounded-full bg-accent-soft px-1.5 py-0.5 text-[10px] font-medium text-accent-soft-fg"
              title={m.workspace.detail.live_pill_title}
            >
              {m.workspace.detail.live_pill}
            </span>
          {/if}
        </p>
      </div>
      <WorkspaceToolIsland
        onrename={() => (renameOpen = true)}
        onexport={() => (exportOpen = true)}
        onimport={() => (importOpen = true)}
      />
    </header>

    <!-- workspaceRevision lets the slices store's Tier-1 short-circuit skip every
         per-category dataset GET when the persisted workspace_sync row matches it. -->
    <div class="mb-6">
      <CategoryList
        workspaceId={detail.id}
        workspaceRevision={detail.workspace_revision.id}
        workspaceName={detail.name}
      />
    </div>

    <div class="mb-6">
      <TrainPane workspaceId={detail.id} {liveRevision} heads={detail.heads} />
    </div>

    <!-- liveRevision folds in the upload receipt, so a head trained at the new revision wins "Latest"
         (and pre-upload heads lose it) immediately, without waiting for the poller to refetch detail. -->
    <DeployPane
      workspaceId={detail.id}
      workspaceName={detail.name}
      heads={detail.heads}
      {liveRevision}
      onchanged={refreshDetail}
    />
  </div>

  <DeleteWorkspaceDialog
    open={deleteOpen}
    workspaceId={detail.id}
    workspaceName={detail.name}
    onclose={() => (deleteOpen = false)}
    ondeleted={backToList}
  />

  <RenameWorkspaceDialog
    open={renameOpen}
    workspaceId={detail.id}
    currentName={detail.name}
    onclose={() => (renameOpen = false)}
    onsaved={onRenamed}
  />

  <!-- Lazy: dialog + export machinery load on first open, out of the route's initial bundle. -->
  {#if exportOpen && detail}
    {@const ws = detail}
    {#await import('$lib/components/workspace/WorkspaceExportDialog.svelte') then { default: WorkspaceExportDialog }}
      <WorkspaceExportDialog
        open={true}
        workspaceId={ws.id}
        workspaceName={ws.name}
        heads={ws.heads}
        workspaceRevisionId={ws.workspace_revision.id}
        onclose={() => (exportOpen = false)}
      />
    {/await}
  {/if}

  <!-- Dialog refreshes categories + slices itself; refreshDetail keeps the heads list current. Lazy-loaded: the import wizard loads on first open, not in the route's initial bundle. -->
  {#if importOpen && detail}
    {@const ws = detail}
    {#await import('$lib/components/workspace/ImportWorkspaceDialog.svelte') then { default: ImportWorkspaceDialog }}
      <ImportWorkspaceDialog
        open={true}
        mode="into-current"
        workspaceId={ws.id}
        workspaceName={ws.name}
        onclose={() => (importOpen = false)}
        onimported={() => void refreshDetail()}
      />
    {/await}
  {/if}

  <ContextMenu
    open={menuOpen}
    x={menuX}
    y={menuY}
    sections={menuSections}
    onclose={() => (menuOpen = false)}
  />
{/if}
