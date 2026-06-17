<script lang="ts">
  import { fade, fly } from 'svelte/transition';
  import { resolve } from '$app/paths';
  import { workspaces } from '$lib/stores/workspaces.svelte';
  import { formatRelative } from '$lib/utils/time';
  import { m } from '$lib/i18n';
  import InlineName from '$lib/components/ui/InlineName.svelte';
  import Spinner from '$lib/components/Spinner.svelte';
  import type { WorkspaceListEntry } from '$lib/api/types';

  // Only `editing` is per-card; mode/selected/deleting come from the singleton store. Rename/delete live in the right-click menu (parent owns one bubbling oncontextmenu) so the row stays free of hover icons.
  interface Props {
    workspace: WorkspaceListEntry;
    editing: boolean;
    onendedit: () => void;
  }
  let { workspace, editing, onendedit }: Props = $props();

  const isDeleting = $derived(workspaces.deleting.has(workspace.id));
  const isSelected = $derived(workspaces.selected.has(workspace.id));
  const mode = $derived(workspaces.mode);

  const detailHref = $derived(resolve(`/workspaces/${workspace.id}`));

  function onLinkClick(e: MouseEvent): void {
    if (mode === 'selecting') {
      e.preventDefault();
      if (!isDeleting) workspaces.toggleSelect(workspace.id);
      return;
    }
    if (isDeleting) e.preventDefault();
  }

  async function saveName(newValue: string): Promise<void> {
    await workspaces.patch(workspace.id, { name: newValue });
    onendedit();
  }
</script>

<!-- `pl-12` reserves checkbox room in selecting mode; `pr-28` reserves the in-flight deleting-badge slot. -->
<li
  data-workspace-id={workspace.id}
  class="relative rounded-lg border bg-surface transition hover:shadow-card {isSelected
    ? 'border-accent hover:border-accent-hover'
    : 'border-line hover:border-line-strong'}"
  class:opacity-60={isDeleting}
>
  {#if editing}
    <!-- `py-2` vs the static row's `py-3` cancels the 8-px growth from InlineName's `h-7` pill over the `<h2>`'s 20-px line box, pinning row height at 44 px across the toggle; `min-w-0 flex-1` gives the input the same slot as `<h2>` so the trailing `created` strip stays right-aligned. -->
    <div class="flex items-center gap-3 px-4 py-2">
      <div class="min-w-0 flex-1">
        <InlineName
          value={workspace.name}
          ariaLabel={m.workspace.card.rename_aria(workspace.name)}
          onsave={saveName}
          oncancel={onendedit}
        />
      </div>
      <span
        class="hidden shrink-0 text-[11px] text-fg-muted sm:inline"
        title={workspace.created_at}
      >
        {m.workspace.card.created_label(formatRelative(workspace.created_at))}
      </span>
    </div>
  {:else}
    <!-- Easing must equal Svelte's `cubicOut` (the checkbox `transition:fly` default); Tailwind's `ease-out` is a different bezier and desyncs the padding shift from the checkbox slide. -->
    <a
      href={detailHref}
      aria-disabled={isDeleting}
      class="flex items-center gap-3 px-4 py-3 transition-all duration-150 ease-[cubic-bezier(0.333,1,0.667,1)]"
      class:pl-12={mode === 'selecting'}
      class:pr-28={isDeleting}
      class:pointer-events-none={isDeleting}
      onclick={onLinkClick}
    >
      <h2 class="min-w-0 flex-1 truncate text-sm font-semibold text-fg" title={workspace.name}>
        {workspace.name}
      </h2>
      <span
        class="hidden shrink-0 text-[11px] text-fg-muted sm:inline"
        title={workspace.created_at}
      >
        {m.workspace.card.created_label(formatRelative(workspace.created_at))}
      </span>
    </a>

    {#if mode === 'selecting'}
      <!-- `fly` x=-32 slides the checkbox in lock-step with the name's 32-px padding shift (a plain `fade` would let the name pass through the half-opaque checkbox); `-translate-y-1/2` compiles to Tailwind v4's `translate` (separate from `transform`), so it stacks with fly's inline `transform: translate(...)` and vertical centring survives without a wrapper. -->
      <label
        transition:fly={{ x: -32, duration: 150 }}
        class="absolute top-1/2 left-2.5 flex h-7 w-7 -translate-y-1/2 cursor-pointer items-center justify-center rounded-md transition hover:bg-surface-2"
        class:cursor-not-allowed={isDeleting}
      >
        <input
          type="checkbox"
          class="h-4 w-4 cursor-pointer rounded border-line-strong accent-blue-500"
          checked={isSelected}
          disabled={isDeleting}
          onchange={() => workspaces.toggleSelect(workspace.id)}
          aria-label={m.workspace.card.select_aria(workspace.name)}
        />
      </label>
    {/if}

    {#if isDeleting}
      <span
        transition:fade={{ duration: 150 }}
        class="absolute top-1/2 right-3 inline-flex -translate-y-1/2 shrink-0 items-center gap-1 rounded-full bg-danger-soft px-2 py-0.5 text-[10px] font-medium text-danger-soft-fg capitalize"
      >
        <Spinner class="h-2.5 w-2.5 text-danger-soft-fg" />
        {m.workspace.card.deleting}
      </span>
    {/if}
  {/if}
</li>
