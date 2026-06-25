<script lang="ts">
  import { fade } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import DownloadIcon from '$lib/components/ui/DownloadIcon.svelte';
  import StatusBadge from '$lib/components/ui/StatusBadge.svelte';
  import Spinner from '$lib/components/Spinner.svelte';
  import HeadInfoPopover from './HeadInfoPopover.svelte';
  import { formatBytes } from '$lib/utils/format';
  import { formatRelative } from '$lib/utils/time';
  import { m } from '$lib/i18n';
  import type { HeadRecord, Uuid } from '$lib/api/types';

  interface Props {
    head: HeadRecord;
    // Read only by the model-card popover's manifest fetch, not the row.
    workspaceId: Uuid;
    // Sole freshness signal: this row is the single newest head at the live revision. Every other head
    // (older same-revision run or older revision) is "not Latest" and intentionally shows no pill.
    isLatest: boolean;
    // This row's head is the runtime-active one; drives the Active pill and chrome tint.
    isDeployed: boolean;
    // Another head on this list is mid-mutation; disables this one so two destructive actions can't race.
    busy?: boolean;
    // Parent-driven (not local) so spinners also fire for the context-menu path, not just icon/row-click.
    isExporting: boolean;
    isDeploying: boolean;
    ondeploy: (headId: Uuid) => Promise<void>;
    // Not gated on isDeployed: the daemon's export path is read-only, so backing up the running head is fine.
    onexport: (head: HeadRecord) => void;
  }
  let {
    head,
    workspaceId,
    isLatest,
    isDeployed,
    busy = false,
    isExporting,
    isDeploying,
    ondeploy,
    onexport
  }: Props = $props();

  // Row-click deploy path (isDeploying covers the context-menu path); gates re-entry.
  let deploying = $state(false);

  const canDeploy = $derived(!isDeployed && !busy && !deploying && !isExporting && !isDeploying);

  // activeMounted lags isDeployed by the spinner's out-duration so the badge mounts as the spinner
  // finishes leaving (no frame shows both); revert is immediate to avoid a stale badge.
  const SPINNER_OUT_DURATION_MS = 140;
  // Snapshot the prop so an already-deployed mount paints immediately; runtime flips go through $effect.
  // svelte-ignore state_referenced_locally
  let activeMounted = $state(isDeployed);
  let activeMountTimer: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    const target = isDeployed;
    if (target === activeMounted) return;
    if (target) {
      activeMountTimer = setTimeout(() => {
        activeMounted = true;
        activeMountTimer = null;
      }, SPINNER_OUT_DURATION_MS);
    } else {
      activeMounted = false;
    }
    return (): void => {
      if (activeMountTimer !== null) {
        clearTimeout(activeMountTimer);
        activeMountTimer = null;
      }
    };
  });

  // The "Latest" pill class-swap-hides so "id + Active" stays on one line; yield must span the WHOLE
  // Active-slot occupancy (spinner + mounted pill + its out:fade) or it wraps beside it. Release
  // exceeds the fade because the fade's WAAPI clock starts ~1 frame after this timer arms (Svelte primes
  // it in a zero-duration animation), so a bare-equal timer flashes the box back beside the still-fading one.
  // Only the Latest pill reads freshnessYield (inert elsewhere); kept ungated so a Latest flip can't
  // tear down an in-flight release timer.
  const FRESHNESS_RELEASE_MS = SPINNER_OUT_DURATION_MS + 60;
  // svelte-ignore state_referenced_locally
  let freshnessYield = $state(isDeployed);
  let freshnessReleaseTimer: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    const occupied = deploying || isDeploying || isDeployed || activeMounted;
    if (occupied === freshnessYield) return;
    if (occupied) {
      freshnessYield = true;
    } else {
      freshnessReleaseTimer = setTimeout(() => {
        freshnessYield = false;
        freshnessReleaseTimer = null;
      }, FRESHNESS_RELEASE_MS);
    }
    return (): void => {
      if (freshnessReleaseTimer !== null) {
        clearTimeout(freshnessReleaseTimer);
        freshnessReleaseTimer = null;
      }
    };
  });

  // Display toggle (hidden), not a mount, so StatusBadge keeps its instance + cached measured width on hide/show -- no remount/re-measure flash.
  const freshnessPillClass = $derived(
    freshnessYield ? 'hidden @[12rem]/headrow:inline-flex' : 'inline-flex'
  );

  async function onRowClick(): Promise<void> {
    if (!canDeploy) return;
    deploying = true;
    try {
      await ondeploy(head.head_id);
    } finally {
      deploying = false;
    }
  }

  // Enter/Space activate; preventDefault stops Space scrolling the page.
  function onRowKeydown(e: KeyboardEvent): void {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    if (!canDeploy) return;
    e.preventDefault();
    void onRowClick();
  }

  function onExportClick(e: MouseEvent): void {
    // Stop bubbling to the row onclick, which would deploy on top of the export.
    e.stopPropagation();
    if (isExporting || deploying || isDeploying || busy) return;
    onexport(head);
  }

  const rowTitle = $derived.by(() => {
    const t = m.deploy.head_row;
    if (isDeployed) return t.row_title_deployed;
    if (deploying || isDeploying) return t.row_title_deploying;
    if (isExporting) return t.row_title_exporting;
    if (busy) return t.row_title_busy;
    return t.row_title_idle;
  });
</script>

<!-- Clickable surface is the inner div[role=button], not the li (a11y lint refuses an interactive role
     on a list element; a native button would forbid the nested icon buttons). data-head-id stays on the
     li so the parent's context-menu handler resolves the row via closest('[data-head-id]') from any inner click. -->
<li data-head-id={head.head_id}>
  <div
    role="button"
    tabindex={canDeploy ? 0 : -1}
    aria-disabled={!canDeploy}
    aria-label={isDeployed
      ? m.deploy.head_row.row_aria_deployed(head.head_id.slice(0, 8))
      : m.deploy.head_row.row_aria_deploy(head.head_id.slice(0, 8))}
    title={rowTitle}
    onclick={onRowClick}
    onkeydown={onRowKeydown}
    class="group/row flex flex-wrap items-center justify-between gap-3 rounded-md border px-3 py-2.5 transition-colors focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none"
    class:border-accent-line={isDeployed}
    class:bg-accent-soft={isDeployed}
    class:border-line={!isDeployed}
    class:bg-surface={!isDeployed}
    class:hover:border-primary={canDeploy}
    class:hover:bg-accent-soft={canDeploy}
    class:cursor-pointer={canDeploy}
    class:cursor-wait={deploying || isDeploying || isExporting}
    class:cursor-default={isDeployed && !deploying && !isDeploying && !isExporting}
    class:cursor-not-allowed={busy && !isDeployed && !deploying && !isDeploying && !isExporting}
  >
    <!-- @container/headrow excludes the sibling fixed-width action icons so the thresholds track the
         space the text/badges actually get. -->
    <div class="@container/headrow min-w-0 flex-1">
      <div class="flex flex-wrap items-center gap-x-2 gap-y-1">
        <p class="font-mono text-sm font-semibold text-fg" title={head.head_id}>
          {head.head_id.slice(0, 8)}
        </p>
        {#if isLatest}
          <span class={freshnessPillClass}>
            <StatusBadge
              size="xs"
              label={m.deploy.head_row.pill_latest}
              tone="success"
              title={m.deploy.head_row.pill_latest_title}
            />
          </span>
        {/if}
        {#if isLatest && freshnessYield}
          <!-- The hidden visual pill leaves the a11y tree; announce it here, only below @[12rem],
               so it's heard exactly once. -->
          <span class="sr-only @[12rem]/headrow:hidden">
            {m.deploy.head_row.pill_latest}
          </span>
        {/if}
        {#if (deploying || isDeploying) && !isDeployed}
          <!-- Spinner in the Active pill's slot; its out:fade overlaps the activeMounted-lagged badge
               mount (see SPINNER_OUT_DURATION_MS). pointer-events-none so a stray hover can't intercept
               row clicks. -->
          <span
            in:fade={{ duration: 180, easing: cubicOut }}
            out:fade={{ duration: 140, easing: cubicOut }}
            aria-hidden="true"
            class="pointer-events-none inline-flex shrink-0 items-center justify-center"
          >
            <Spinner class="h-3.5 w-3.5 text-accent" />
          </span>
        {/if}
        {#if activeMounted}
          <StatusBadge
            size="xs"
            label={m.deploy.head_row.pill_active}
            tone="accent"
            title={m.deploy.head_row.pill_active_title}
          />
        {/if}
      </div>
      <!-- Meta degrades: size hides at @[14rem], rev at @[12rem]; classes+age always stay. Each droppable
           segment owns its aria-hidden separator so it vanishes cleanly. -->
      <p class="mt-1 text-[11px] text-fg-muted">
        <span class="hidden @[14rem]/headrow:inline">
          {formatBytes(head.size_bytes)}
          <span aria-hidden="true">·</span>
        </span>
        {m.deploy.head_row.meta_classes(head.n_classes)}
        <span class="hidden @[12rem]/headrow:inline">
          <span aria-hidden="true">·</span>
          {m.deploy.head_row.meta_rev(head.workspace_revision.id)}
        </span>
        <span aria-hidden="true">·</span>
        {formatRelative(head.created_at)}
      </p>
    </div>

    <!-- Hover-revealed via opacity+pointer-events (not display) so the row never reflows on hover. Export
         tooltip is on the wrapper span, not the button: a disabled button fires no pointer events in Firefox. -->
    <div class="flex shrink-0 items-center gap-1.5">
      <span
        class="inline-flex shrink-0"
        title={isExporting
          ? m.deploy.head_row.export_title_exporting
          : m.deploy.head_row.export_title_idle}
      >
        <!-- !opacity-100 / !pointer-events-auto pin the button visible while exporting so the spinner stays
             legible after the cursor leaves the row. -->
        <button
          type="button"
          onclick={onExportClick}
          disabled={deploying || isDeploying || busy}
          aria-label={isExporting
            ? m.deploy.head_row.export_aria_exporting(head.head_id.slice(0, 8))
            : m.deploy.head_row.export_aria_idle(head.head_id.slice(0, 8))}
          class="pointer-events-none inline-flex shrink-0 items-center justify-center rounded-md p-1.5 text-fg-subtle opacity-0 transition duration-200 ease-out group-hover/row:pointer-events-auto group-hover/row:opacity-100 focus-visible:pointer-events-auto focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none enabled:hover:bg-accent-soft enabled:hover:text-accent disabled:cursor-not-allowed disabled:text-fg-subtle pointer-coarse:pointer-events-auto pointer-coarse:opacity-100"
          class:!opacity-100={isExporting}
          class:!pointer-events-auto={isExporting}
        >
          {#if isExporting}
            <Spinner class="h-3.5 w-3.5 text-accent" />
          {:else}
            <DownloadIcon />
          {/if}
        </button>
      </span>
      <!-- Owns its trigger + manifest fetch; shows trained class labels the row doesn't carry. -->
      <HeadInfoPopover {head} {workspaceId} />
    </div>
  </div>
</li>
