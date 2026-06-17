<script lang="ts">
  import { SvelteSet } from 'svelte/reactivity';
  import { fade } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import ContextMenu, { type MenuSection } from '$lib/components/ui/ContextMenu.svelte';
  import TrainHistoryItem from './TrainHistoryItem.svelte';
  import {
    training as trainingStore,
    TRAINING_HISTORY_MAX_PER_WS,
    TRAINING_HISTORY_PAGE_SIZE,
    TRAINING_INITIAL_VISIBLE
  } from '$lib/stores/training.svelte';
  import type { TrackedTrainingJob } from '$lib/stores/training.svelte';
  import { m } from '$lib/i18n';
  import type { Uuid } from '$lib/api/types';

  // Keyed on job.jobId so active->terminal is a same-DOM-node update (instance/expansion/position
  // survive). Per-item expansion is explicit-only (auto-expand jumped the pane on every Train click);
  // older-accordion expansion lives on the store to survive remounts.

  interface Props {
    workspaceId: Uuid;
  }
  let { workspaceId }: Props = $props();

  const active = $derived(trainingStore.activeFor(workspaceId));
  const eagerHistory = $derived(trainingStore.eagerHistoryFor(workspaceId));
  const olderHistory = $derived(trainingStore.olderHistoryFor(workspaceId));
  const hydrating = $derived(trainingStore.hydratingFor(workspaceId));
  const loadingMore = $derived(trainingStore.loadingMoreFor(workspaceId));
  const olderExpanded = $derived(trainingStore.olderExpandedFor(workspaceId));
  const loadableOlder = $derived(trainingStore.loadableOlderCountFor(workspaceId));

  // active claims one of the INITIAL_VISIBLE slots (not added on top) so the tier renders exactly
  // that many rows across every transition; the displaced row stays reachable below.
  const eagerHistoryVisible = $derived<readonly TrackedTrainingJob[]>(
    active
      ? eagerHistory.slice(0, TRAINING_INITIAL_VISIBLE - 1)
      : eagerHistory.slice(0, TRAINING_INITIAL_VISIBLE)
  );
  const eagerItems = $derived<TrackedTrainingJob[]>(
    active ? [active, ...eagerHistoryVisible] : [...eagerHistoryVisible]
  );

  // Eager row evicted by the active slot, prepended onto the older list to stay reachable; non-empty
  // only while training is in flight AND the eager tier was already full.
  const displacedFromEager = $derived<readonly TrackedTrainingJob[]>(
    active ? eagerHistory.slice(TRAINING_INITIAL_VISIBLE - 1, TRAINING_INITIAL_VISIBLE) : []
  );

  // Distinct from olderHistoryFor: keeps displacement presentation-only and store accessors loop-free.
  const olderItems = $derived<readonly TrackedTrainingJob[]>([
    ...displacedFromEager,
    ...olderHistory
  ]);

  // olderItems (not olderHistory) so the count includes the displaced row.
  const olderTotal = $derived(olderItems.length + loadableOlder);

  // Nothing to render AND nothing to fetch -- distinct from "loading", which renders skeletons.
  const isEmpty = $derived(
    !hydrating &&
      !active &&
      eagerHistory.length === 0 &&
      olderHistory.length === 0 &&
      loadableOlder === 0
  );

  // Skeletons paint only while a load is incoming, else they'd linger beside real rows when
  // history.length < INITIAL_VISIBLE; active folded in so its slot isn't double-filled.
  const activeContribution = $derived(active ? 1 : 0);
  const skeletonsArePainting = $derived(
    trainingStore.hydratingFor(workspaceId) || trainingStore.autoRefillingFor(workspaceId)
  );
  const eagerSkeletonCount = $derived(
    skeletonsArePainting
      ? Math.max(0, TRAINING_INITIAL_VISIBLE - activeContribution - eagerHistoryVisible.length)
      : 0
  );

  // Store-snapshotted at click-time to the in-flight loadBatch's rows; rendered as <li> siblings of
  // loaded rows for an in-place swap. Zero when idle.
  const olderSkeletonCount = $derived(trainingStore.olderSkeletonCountFor(workspaceId));

  // Rows the NEXT "Load N more" click surfaces. Jitter-free across the await: the click bumps
  // olderSkeletonCount as the landing shrinks loadableOlder by the same amount, same tick.
  const nextLoadCount = $derived(
    Math.max(0, Math.min(loadableOlder - olderSkeletonCount, TRAINING_HISTORY_PAGE_SIZE))
  );

  const expanded = new SvelteSet<Uuid>();
  // Gates auto-collapse to once per id. Plain Set, not SvelteSet: SvelteSet would make the $effect
  // both read (has) and write (add) one signal, forcing a redundant re-run.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- plain Set intentional, see above
  const seenInOlder = new Set<Uuid>();

  // SvelteKit reuses +page.svelte across [id] navigation (no {#key} wrapper), so reset these
  // component-scoped sets on workspace change; otherwise stale jobId entries persist and a UUID
  // collision could let A's state taint B's auto-collapse.
  let lastWorkspaceIdSeen: Uuid | null = null;
  $effect(() => {
    const ws = workspaceId;
    if (lastWorkspaceIdSeen === ws) return;
    lastWorkspaceIdSeen = ws;
    expanded.clear();
    seenInOlder.clear();
  });

  // Collapse a row when it drops into the older tier, once per id (via seenInOlder) so a manual
  // re-expand still sticks.
  $effect(() => {
    for (const job of olderHistory) {
      if (seenInOlder.has(job.jobId)) continue;
      seenInOlder.add(job.jobId);
      expanded.delete(job.jobId);
    }
  });

  function toggle(jobId: Uuid): void {
    if (expanded.has(jobId)) expanded.delete(jobId);
    else expanded.add(jobId);
  }

  // No per-row Cancel by design: the TrainPane header's primary button is the single canonical
  // run-lifecycle surface (morphs to destructive Cancel while running).
  function onToggleOlder(): void {
    trainingStore.setOlderExpanded(workspaceId, !olderExpanded);
  }

  async function onLoadMore(): Promise<void> {
    await trainingStore.loadMoreHistory(workspaceId);
  }

  // One ContextMenu for the whole list; onListContextMenu walks closest('[data-job-id]') to the
  // row. Delete mirrors the daemon's per-workspace JobConflict: disabled while ANY Train producer
  // runs here, and the live row is never deletable (it IS the producer's open file).
  let menuOpen = $state(false);
  let menuX = $state(0);
  let menuY = $state(0);
  let menuSections = $state<MenuSection[]>([]);

  function onListContextMenu(e: MouseEvent): void {
    const target = e.target as HTMLElement | null;
    if (!target) return;
    if (target.closest('input, textarea')) return;
    const rowEl = target.closest<HTMLElement>('[data-job-id]');
    const jobId = rowEl?.dataset.jobId ?? null;
    if (!jobId) return;
    // active first: historyFor excludes the live entry.
    const live = trainingStore.activeFor(workspaceId);
    const fromActive = live?.jobId === jobId ? live : null;
    const fromHistory =
      trainingStore.historyFor(workspaceId).find((j) => j.jobId === jobId) ?? null;
    const job = fromActive ?? fromHistory;
    if (!job) return;
    const sections = buildMenu(job);
    if (sections.length === 0) return;
    e.preventDefault();
    // Stop the page root context menu opening at the same cursor.
    e.stopPropagation();
    menuX = e.clientX;
    menuY = e.clientY;
    menuSections = sections;
    menuOpen = true;
  }

  function buildMenu(job: TrackedTrainingJob): MenuSection[] {
    const live = trainingStore.activeFor(workspaceId);
    const trainActive = live !== null;
    const isLiveRow = live?.jobId === job.jobId;
    const deleting = trainingStore.historyDeletingForJob(workspaceId, job.jobId);
    // No hint while deleting (the label morphs); else name the obstacle: live row vs sibling train.
    let hint: string | undefined;
    if (!deleting) {
      if (isLiveRow) hint = m.training.history.menu_hint_live;
      else if (trainActive) hint = m.training.history.menu_hint_train_active;
    }
    return [
      {
        items: [
          {
            label: deleting ? m.training.history.menu_deleting : m.training.history.menu_delete,
            variant: 'destructive',
            disabled: trainActive || deleting,
            hint,
            onclick: () => void trainingStore.deleteHistoryEntry(workspaceId, job.jobId)
          }
        ]
      }
    ];
  }

  const deleteError = $derived(trainingStore.historyDeleteErrorFor(workspaceId));
  function onDismissDeleteError(): void {
    trainingStore.dismissHistoryDeleteError(workspaceId);
  }
</script>

<!-- No "Clear finished": the daemon's keep-last-N cap makes it redundant and per-entry fan-out
     would trip the daemon's single delete-admission slot. TRAINING_HISTORY_MAX_PER_WS mirrors the
     daemon cap - bump in lockstep. Right-clicks off any row early-return to the page root menu. -->
<div class="flex flex-col gap-2" oncontextmenu={onListContextMenu} role="presentation">
  <div class="flex items-baseline justify-between">
    <h3 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
      {m.training.history.heading}
    </h3>
    {#if eagerHistory.length + olderHistory.length + loadableOlder > 0}
      <span
        class="text-[10px] text-fg-subtle"
        title={m.training.history.retention_title(TRAINING_HISTORY_MAX_PER_WS)}
      >
        {m.training.history.keeps_last(TRAINING_HISTORY_MAX_PER_WS)}
      </span>
    {/if}
  </div>

  {#if deleteError}
    {@const hasMessage = deleteError.trim().length > 0}
    <!-- Padding collapses the chip when the daemon returns a code-only envelope (no message). -->
    <div
      in:fade={{ duration: 200, easing: cubicOut }}
      out:fade={{ duration: 160, easing: cubicOut }}
      class="flex justify-between gap-2 rounded-md border border-danger-line bg-danger-soft text-xs text-danger-soft-fg"
      class:items-start={hasMessage}
      class:items-center={!hasMessage}
      class:px-3={hasMessage}
      class:py-2={hasMessage}
      class:py-1={!hasMessage}
      class:pr-1={!hasMessage}
      class:pl-2.5={!hasMessage}
      role="alert"
    >
      <div class="min-w-0">
        <p class="font-medium">{m.training.history.delete_error_title}</p>
        {#if hasMessage}
          <p class="mt-0.5 wrap-break-word">{deleteError}</p>
        {/if}
      </div>
      <button
        type="button"
        onclick={onDismissDeleteError}
        aria-label={m.common.dismiss}
        class="shrink-0 rounded-md p-1 text-danger-soft-fg transition hover:bg-danger-soft"
        class:-mt-1={hasMessage}
        class:-mr-2={hasMessage}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          class="h-3.5 w-3.5"
          aria-hidden="true"
        >
          <path d="M6 6l12 12M6 18L18 6" />
        </svg>
      </button>
    </div>
  {/if}

  {#if isEmpty}
    <!-- Sized to ~one collapsed item so section height is stable between "no runs" and "one run"
         (first submit won't shift the heads list below). -->
    <div
      class="flex items-center justify-center gap-2 rounded-md border border-dashed border-line bg-surface-2/60 px-3 py-3 text-[11px] text-fg-muted"
    >
      <svg
        viewBox="0 0 20 20"
        fill="currentColor"
        aria-hidden="true"
        class="h-4 w-4 shrink-0 text-fg-subtle"
      >
        <path
          fill-rule="evenodd"
          d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-13a.75.75 0 00-1.5 0v5c0 .2.08.39.22.53l3 3a.75.75 0 101.06-1.06l-2.78-2.78V5z"
          clip-rule="evenodd"
        />
      </svg>
      <span class="text-center"
        >{m.training.history.empty_state_prefix}<b>{m.training.history.empty_state_button}</b>{m
          .training.history.empty_state_suffix}</span
      >
    </div>
  {:else}
    <ul class="flex flex-col gap-2">
      {#each eagerItems as job (job.jobId)}
        <TrainHistoryItem
          {job}
          isLive={active?.jobId === job.jobId}
          expanded={expanded.has(job.jobId)}
          ontoggle={() => toggle(job.jobId)}
          isDeleting={trainingStore.historyDeletingForJob(workspaceId, job.jobId)}
        />
      {/each}
      {#each Array.from({ length: eagerSkeletonCount }, (_, i) => i) as i (i)}
        <!-- h-10 must match the collapsed TrainHistoryItem resting height or the tier judders
             shrink->re-grow during the post-delete backfill fetch. -->
        <li
          aria-hidden="true"
          class="h-10 animate-pulse overflow-hidden rounded-md border border-line border-l-4 border-l-line bg-surface"
        >
          <div class="flex items-center gap-x-3 px-3 py-2.5">
            <span class="h-3 w-3 shrink-0 rounded-full bg-line"></span>
            <span class="h-3 w-16 shrink-0 rounded bg-line"></span>
            <span class="h-3 w-20 shrink-0 rounded bg-line"></span>
            <span class="h-3 w-12 shrink-0 rounded bg-line"></span>
          </div>
        </li>
      {/each}
    </ul>

    {#if olderTotal > 0}
      <!-- Opening auto-loads the first older batch when the older tier holds fewer than PAGE_SIZE rows (handleOlderExpand). -->
      <div class="flex flex-col gap-2">
        <!-- Rules sit OUTSIDE the button so the click target is the verb only. Collapsed -mb-4
             absorbs the section's p-5; expanded -mb-1 (full -mb-4 would overlap the first older card). -->
        <div
          class="-mt-1 flex items-center justify-center gap-2.5"
          class:-mb-4={!olderExpanded}
          class:-mb-1={olderExpanded}
        >
          <span class="h-px w-6 bg-line" aria-hidden="true"></span>
          <button
            type="button"
            onclick={onToggleOlder}
            aria-expanded={olderExpanded}
            class="rounded-md px-2 py-0.5 text-[11px] text-fg-secondary transition hover:bg-surface-2 hover:text-fg"
            title={olderExpanded
              ? m.training.history.hide_older_title
              : m.training.history.show_older_title}
          >
            {#if olderExpanded}
              {m.training.history.hide_older_label}
            {:else}
              {m.training.history.show_older_label(olderTotal)}
            {/if}
          </button>
          <span class="h-px w-6 bg-line" aria-hidden="true"></span>
        </div>

        {#if olderExpanded}
          <div in:fade={{ duration: 200, easing: cubicOut }} class="flex flex-col gap-2">
            <!-- Skeletons are <li> siblings inside this <ul> for a constant-height in-place swap;
                 one placeholder outside would imply "1 row" and shift 40px -> Nx40px. -->
            {#if olderItems.length > 0 || olderSkeletonCount > 0}
              <ul class="flex flex-col gap-2">
                {#each olderItems as job (job.jobId)}
                  <TrainHistoryItem
                    {job}
                    isLive={false}
                    expanded={expanded.has(job.jobId)}
                    ontoggle={() => toggle(job.jobId)}
                    isDeleting={trainingStore.historyDeletingForJob(workspaceId, job.jobId)}
                  />
                {/each}
                {#each Array.from({ length: olderSkeletonCount }, (_, i) => i) as i (i)}
                  <li
                    aria-hidden="true"
                    class="h-10 animate-pulse overflow-hidden rounded-md border border-line border-l-4 border-l-line bg-surface"
                  >
                    <div class="flex items-center gap-x-3 px-3 py-2.5">
                      <span class="h-3 w-3 shrink-0 rounded-full bg-line"></span>
                      <span class="h-3 w-16 shrink-0 rounded bg-line"></span>
                      <span class="h-3 w-24 shrink-0 rounded bg-line"></span>
                    </div>
                  </li>
                {/each}
              </ul>
            {/if}

            {#if nextLoadCount > 0}
              <!-- Disabled (not unmounted) while loadingMore: a click would hit loadMoreHistory's
                   re-entry guard and no-op, so disabling surfaces that. -->
              <div class="-mt-1 -mb-4 flex items-center justify-center gap-2.5">
                <span class="h-px w-6 bg-line" aria-hidden="true"></span>
                <button
                  type="button"
                  onclick={() => void onLoadMore()}
                  disabled={loadingMore}
                  class="rounded-md px-2 py-0.5 text-[11px] text-fg-muted transition enabled:hover:bg-surface-2 enabled:hover:text-fg disabled:cursor-not-allowed disabled:opacity-50"
                  title={m.training.history.load_more_title}
                >
                  {m.training.history.load_more_label(nextLoadCount)}
                </button>
                <span class="h-px w-6 bg-line" aria-hidden="true"></span>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  {/if}
</div>

<!-- Outside the wrapping <div> so its position:fixed chrome paints above the section's card
     boundaries (own z-50 handles page stacking). -->
<ContextMenu
  open={menuOpen}
  x={menuX}
  y={menuY}
  sections={menuSections}
  onclose={() => (menuOpen = false)}
/>
