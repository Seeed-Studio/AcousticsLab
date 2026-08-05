<script lang="ts">
  import { goto } from '$app/navigation';
  import { resolve } from '$app/paths';
  import { workspaces, WORKSPACE_NOTICE_THRESHOLD } from '$lib/stores/workspaces.svelte';
  import { m } from '$lib/i18n';
  import Button from '$lib/components/ui/Button.svelte';
  import EmptyState from '$lib/components/ui/EmptyState.svelte';
  import LoadingRow from '$lib/components/LoadingRow.svelte';
  import PlusIcon from '$lib/components/ui/PlusIcon.svelte';
  import TrashIcon from '$lib/components/ui/TrashIcon.svelte';
  import ContextMenu, { type MenuSection } from '$lib/components/ui/ContextMenu.svelte';
  import CreateWorkspaceDialog from '$lib/components/workspace/CreateWorkspaceDialog.svelte';
  import DeleteWorkspaceDialog from '$lib/components/workspace/DeleteWorkspaceDialog.svelte';
  import BulkDeleteWorkspacesDialog from '$lib/components/workspace/BulkDeleteWorkspacesDialog.svelte';
  import WorkspaceCard from '$lib/components/workspace/WorkspaceCard.svelte';
  import UploadIcon from '$lib/components/ui/UploadIcon.svelte';
  import type { WorkspaceListEntry, WorkspaceMutationResp, Uuid } from '$lib/api/types';

  // Refresh on every visit: the list may have changed via another tab, the CLI, or an in-flight delete.
  $effect(() => {
    void workspaces.refresh();
  });

  let createOpen = $state(false);
  let editingId = $state<Uuid | null>(null);
  let deleteTarget = $state<WorkspaceListEntry | null>(null);
  let bulkOpen = $state(false);
  let bulkTargets = $state<WorkspaceListEntry[]>([]);
  // Right-click Export target; dialog self-loads heads + revision when absent, so id + name suffice here.
  let exportTarget = $state<WorkspaceListEntry | null>(null);
  let importOpen = $state(false);

  let menuOpen = $state(false);
  let menuX = $state(0);
  let menuY = $state(0);
  let menuSections = $state<MenuSection[]>([]);

  const mode = $derived(workspaces.mode);
  const selectableCount = $derived(
    workspaces.entries.filter((w) => !workspaces.deleting.has(w.id)).length
  );
  const isAllSelected = $derived(
    selectableCount > 0 && workspaces.selected.size >= selectableCount
  );
  const manyWorkspaces = $derived(workspaces.entries.length >= WORKSPACE_NOTICE_THRESHOLD);
  const overlayActive = $derived(
    createOpen ||
      deleteTarget !== null ||
      exportTarget !== null ||
      importOpen ||
      bulkOpen ||
      menuOpen ||
      editingId !== null
  );

  function onCreated(resp: WorkspaceMutationResp): void {
    void goto(resolve(`/workspaces/${resp.id}`));
  }

  function openCreate(): void {
    createOpen = true;
  }

  function openBulkDelete(): void {
    bulkTargets = workspaces.selectedEntries.slice();
    if (bulkTargets.length === 0) return;
    bulkOpen = true;
  }

  function startInlineEdit(ws: WorkspaceListEntry): void {
    if (workspaces.deleting.has(ws.id)) return;
    editingId = ws.id;
  }

  $effect(() => {
    if (mode !== 'selecting') return;
    if (overlayActive) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        e.preventDefault();
        workspaces.exitSelecting();
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  });

  function onPageContextMenu(e: MouseEvent): void {
    const target = e.target as HTMLElement | null;
    if (!target) return;
    if (target.closest('input, textarea')) return;
    const cardEl = target.closest<HTMLElement>('[data-workspace-id]');
    const id = cardEl?.dataset.workspaceId ?? null;
    const ws = id ? (workspaces.entries.find((w) => w.id === id) ?? null) : null;
    const sections = buildMenu(ws);
    if (sections.length === 0) return;
    e.preventDefault();
    menuX = e.clientX;
    menuY = e.clientY;
    menuSections = sections;
    menuOpen = true;
  }

  function buildMenu(ws: WorkspaceListEntry | null): MenuSection[] {
    const t = m.workspace.list;
    if (ws) {
      const isSelected = workspaces.selected.has(ws.id);
      const isDeleting = workspaces.deleting.has(ws.id);
      const sections: MenuSection[] = [];
      sections.push({
        items: [
          {
            label: t.menu_open,
            disabled: isDeleting,
            onclick: () => void goto(resolve(`/workspaces/${ws.id}`))
          },
          {
            label: t.menu_rename,
            disabled: isDeleting,
            onclick: () => startInlineEdit(ws)
          },
          // Bare-verb label to match siblings; the dialog title carries the noun + identity.
          {
            label: t.menu_export,
            disabled: isDeleting,
            onclick: () => (exportTarget = ws)
          },
          {
            label: t.menu_delete,
            variant: 'destructive',
            disabled: isDeleting,
            onclick: () => (deleteTarget = ws)
          }
        ]
      });
      if (mode === 'selecting') {
        sections.push({
          items: [
            {
              label: isSelected ? t.menu_deselect_one : t.menu_select_one,
              disabled: isDeleting,
              onclick: () => workspaces.toggleSelect(ws.id)
            },
            {
              label: t.menu_done_exit,
              onclick: () => workspaces.exitSelecting()
            }
          ]
        });
      } else {
        sections.push({
          items: [
            {
              label: t.menu_select_workspaces,
              disabled: isDeleting,
              onclick: () => {
                workspaces.enterSelecting();
                workspaces.toggleSelect(ws.id);
              }
            }
          ]
        });
      }
      return sections;
    }
    const items = [];
    items.push({
      label: t.menu_new,
      onclick: () => (createOpen = true)
    });
    items.push({
      label: t.menu_import,
      onclick: () => (importOpen = true)
    });
    if (mode === 'normal' && workspaces.entries.length > 0) {
      items.push({
        label: t.menu_select_workspaces,
        onclick: () => workspaces.enterSelecting()
      });
    }
    if (mode === 'selecting') {
      items.push({
        label: isAllSelected ? t.menu_deselect_all : t.menu_select_all,
        disabled: selectableCount === 0,
        onclick: () => (isAllSelected ? workspaces.clearSelection() : workspaces.selectAllVisible())
      });
      items.push({
        label: t.menu_done_exit,
        onclick: () => workspaces.exitSelecting()
      });
    }
    return items.length > 0 ? [{ items }] : [];
  }
</script>

<svelte:head>
  <title>{m.routes.workspace_list_title(m.app.name)}</title>
</svelte:head>

<header class="mb-5 flex flex-wrap items-center justify-between gap-3">
  <div>
    <h1 class="text-base font-semibold text-fg">{m.workspace.list.title}</h1>
    <p class="mt-0.5 text-xs text-fg-muted" role={manyWorkspaces ? 'status' : undefined}>
      {#if manyWorkspaces}
        <!-- Advisory status, not a gate (nothing is blocked; count is a weak proxy for the real
             resource, disk). Deliberately one rung below the bordered warning banner (that rung
             means faults): the TrainPane amber-subtitle tone, applied only to the emphasized
             fact while the guidance tail stays muted. Weight + wording (not color alone) carry
             the signal, so no glyph is needed. -->
        <span class="font-medium text-warning-soft-fg"
          >{m.workspace.list.many_workspaces_count(workspaces.entries.length)}</span
        >
        {m.workspace.list.many_workspaces_hint}
      {:else}
        {m.workspace.list.default_subtitle}
      {/if}
    </p>
  </div>
  {#if workspaces.loaded && workspaces.entries.length > 0}
    <!-- Selecting-mode buttons reuse this action group rather than a separate toolbar, avoiding a slide-in/layout shift; the sr-only live region announces the selection count. -->
    <span class="sr-only" aria-live="polite">
      {mode === 'selecting' ? m.workspace.list.selected_count_aria(workspaces.selected.size) : ''}
    </span>
    <div class="flex items-center gap-2">
      {#if mode === 'selecting'}
        <Button
          variant="secondary"
          onclick={() =>
            isAllSelected ? workspaces.clearSelection() : workspaces.selectAllVisible()}
          disabled={selectableCount === 0}
        >
          {isAllSelected ? m.workspace.list.deselect_all_label : m.workspace.list.select_all_label}
        </Button>
        <Button variant="secondary" onclick={() => workspaces.exitSelecting()}>
          {m.workspace.list.done_button_label}
        </Button>
        <Button
          variant="destructive"
          onclick={openBulkDelete}
          disabled={workspaces.selected.size === 0}
          ariaLabel={workspaces.selected.size > 0
            ? m.workspace.list.bulk_delete_aria_count(workspaces.selected.size)
            : m.workspace.list.bulk_delete_aria_fallback}
        >
          <TrashIcon />
          {workspaces.selected.size > 0
            ? m.workspace.list.bulk_delete_label_count(workspaces.selected.size)
            : m.workspace.list.bulk_delete_label_bare}
        </Button>
      {:else}
        <Button variant="secondary" onclick={() => workspaces.enterSelecting()}>
          {m.workspace.list.select_button_label}
        </Button>
        <Button
          variant="secondary"
          onclick={() => (importOpen = true)}
          ariaLabel={m.workspace.list.import_button_aria}
          title={m.workspace.list.import_button_title}
        >
          <UploadIcon />
          {m.workspace.list.import_button_label}
        </Button>
        <Button onclick={openCreate} ariaLabel={m.workspace.list.new_button_aria}>
          <PlusIcon />
          {m.workspace.list.new_button_label}
        </Button>
      {/if}
    </div>
  {/if}
</header>

{#if !workspaces.loaded}
  <LoadingRow label={m.workspace.list.loading} />
{:else if workspaces.error && workspaces.entries.length === 0}
  <div
    class="flex items-center gap-3 rounded-lg border border-warning-line bg-warning-soft px-4 py-3 text-xs"
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
      <p class="font-medium text-warning-soft-fg">{m.workspace.list.daemon_unavailable_title}</p>
      <p class="mt-0.5 truncate text-warning-soft-fg">{workspaces.error}</p>
    </div>
  </div>
{:else if workspaces.entries.length === 0}
  <EmptyState title={m.workspace.list.empty_title} description={m.workspace.list.empty_description}>
    {#snippet action()}
      <div class="flex items-center gap-2">
        <Button onclick={openCreate}>
          <PlusIcon />
          {m.workspace.list.new_button_label}
        </Button>
        <!-- Import is a valid first move (operator arrives with an .alpkg, no local workspaces yet). -->
        <Button variant="secondary" onclick={() => (importOpen = true)}>
          <UploadIcon />
          {m.workspace.list.import_button_label}
        </Button>
      </div>
    {/snippet}
  </EmptyState>
{:else}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div oncontextmenu={onPageContextMenu}>
    <ul class="flex flex-col gap-2">
      {#each workspaces.entries as ws (ws.id)}
        <WorkspaceCard
          workspace={ws}
          editing={editingId === ws.id}
          onendedit={() => (editingId = null)}
        />
      {/each}
    </ul>
  </div>
{/if}

<CreateWorkspaceDialog
  open={createOpen}
  onclose={() => (createOpen = false)}
  oncreated={onCreated}
/>

{#if deleteTarget}
  <DeleteWorkspaceDialog
    open={true}
    workspaceId={deleteTarget.id}
    workspaceName={deleteTarget.name}
    onclose={() => (deleteTarget = null)}
  />
{/if}

{#if bulkOpen}
  <BulkDeleteWorkspacesDialog
    open={true}
    targets={bulkTargets}
    onclose={() => (bulkOpen = false)}
  />
{/if}

<!-- `#if` mount-gating scopes the dialog's internal $state to one export cycle, so the next open starts fresh with no reset wiring. The dialog + export machinery load lazily on first open, out of the route's initial bundle. -->
{#if exportTarget}
  {@const ws = exportTarget}
  {#await import('$lib/components/workspace/WorkspaceExportDialog.svelte') then { default: WorkspaceExportDialog }}
    <WorkspaceExportDialog
      open={true}
      workspaceId={ws.id}
      workspaceName={ws.name}
      onclose={() => (exportTarget = null)}
    />
  {/await}
{/if}

<!-- Dialog reconciles the target's category + slice stores before firing onimported, so the detail page lands consistent. Lazy-loaded: the heavy import wizard (alpkg/TFJS machinery) loads on first open, not in the route's initial bundle. -->
{#if importOpen}
  {#await import('$lib/components/workspace/ImportWorkspaceDialog.svelte') then { default: ImportWorkspaceDialog }}
    <ImportWorkspaceDialog
      open={true}
      mode="pick-target"
      onclose={() => (importOpen = false)}
      onimported={(workspaceId) => void goto(resolve(`/workspaces/${workspaceId}`))}
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
