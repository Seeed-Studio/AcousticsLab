<script lang="ts">
  import { onDestroy, untrack } from 'svelte';
  import { SvelteSet } from 'svelte/reactivity';
  import { slices } from '$lib/stores/slices.svelte';
  import { getSliceBlob } from '$lib/audio/slice-fetch';
  import { thresholdFor } from './labels';
  import SliceCard from './SliceCard.svelte';
  import Spinner from '$lib/components/Spinner.svelte';
  import LoadingRow from '$lib/components/LoadingRow.svelte';
  import Tips from '$lib/components/ui/Tips.svelte';
  import ContextMenu, {
    type MenuItem,
    type MenuSection
  } from '$lib/components/ui/ContextMenu.svelte';
  import { m } from '$lib/i18n';
  import type { Uuid } from '$lib/api/types';
  import type { SliceRecord } from '$lib/idb/db';

  // One lazily-built shared AudioContext (not one device per expanded category). The store's
  // `deletingIds` set is the source of truth for "row mid-delete" (store `delete()` early-returns on a
  // present id); every interactive surface gates on it. Page-close mid-batch self-heals: `deletingIds`
  // is in-memory, daemon delete jobs outlive tab death, and the next mount's `refresh()` GCs orphans.
  interface Props {
    workspaceId: Uuid;
    categoryName: string;
  }
  let { workspaceId, categoryName }: Props = $props();

  // Keyed by (workspace, category) too: content-addressed ids let one hash land in two categories.
  const isDeleting = (id: string): boolean => slices.isDeleting(workspaceId, categoryName, id);

  // Refresh on mount and on each stale-flag flip (poller flips it on workspace-revision advance).
  // Untracked because `refresh` writes the slice list this effect tracks, which would otherwise loop.
  $effect(() => {
    const id = workspaceId;
    const name = categoryName;
    void slices.isStale(id, name);
    untrack(() => {
      void slices.refresh(id, name);
    });
  });

  const list = $derived(slices.for(workspaceId, categoryName));
  const threshold = $derived(thresholdFor(categoryName));
  const count = $derived(list.entries.length);
  const satisfiesQuota = $derived(count >= threshold);

  let audioCtx: AudioContext | null = null;
  let activeSource: AudioBufferSourceNode | null = null;
  let playingId = $state<string | null>(null);
  // Decode-window generation: `activeSource` is set only post-decode, so without it a second click
  // mid-decode would `start(0)` twice and overlap. `play()` captures `++playGen`, bails post-await if
  // superseded; `stopPlayback()` bumps it to cancel a pending start.
  let playGen = 0;
  // Lets a mid-decode `play()` bail before touching the AudioContext onDestroy closes, instead of
  // throwing past its try/catch from createBufferSource.
  let destroyed = false;

  async function play(slice: SliceRecord): Promise<void> {
    // Keyboard Enter bypasses pointer-events-none, so guard vs a delete racing getSliceBlob on a row
    // vanishing between its daemon terminal and idbDeleteSlice.
    if (isDeleting(slice.id)) return;
    stopPlayback();
    const gen = ++playGen;
    // Pin the context locally so the post-decode continuation never reads `audioCtx` after a
    // concurrent unmount nulled/closed it.
    const ctx = (audioCtx ??= new AudioContext());
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    let buffer: AudioBuffer;
    try {
      // Decode every play (no buffer cache): ~5 ms at our 1 s clip vs holding buffers for hundreds of slices.
      const blob = await getSliceBlob(slice);
      const arrayBuffer = await blob.arrayBuffer();
      buffer = await ctx.decodeAudioData(arrayBuffer);
    } catch (e) {
      console.warn(`[slice ${slice.id}] play decode failed`, e);
      return;
    }
    // Bail before createBufferSource if superseded during decode: on a closed context it throws past
    // this try/catch as an unhandled rejection.
    if (gen !== playGen || destroyed || ctx.state === 'closed') return;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start(0);
    activeSource = source;
    playingId = slice.id;
    source.onended = (): void => {
      if (activeSource === source) {
        activeSource = null;
        playingId = null;
      }
    };
  }

  function stopPlayback(): void {
    // Bump even when activeSource is null so a stop landing mid-decode cancels the pending start at
    // its `gen !== playGen` guard.
    playGen++;
    if (activeSource) {
      activeSource.onended = null;
      try {
        activeSource.stop();
      } catch {
        // already stopped
      }
      activeSource = null;
      playingId = null;
    }
  }

  onDestroy(() => {
    destroyed = true;
    stopPlayback();
    if (audioCtx) {
      audioCtx.close().catch(() => undefined);
      audioCtx = null;
    }
  });

  async function deleteSlice(record: SliceRecord): Promise<void> {
    if (playingId === record.id) stopPlayback();
    try {
      await slices.delete(record);
    } catch (e) {
      console.error('[slices] delete failed', e);
    }
  }

  function retryUpload(record: SliceRecord): void {
    // Don't enqueue against a vanishing row: a PUT landing between delete()'s upload-abort and
    // idbDeleteSlice would resurrect a dropped row.
    if (isDeleting(record.id)) return;
    void slices.enqueueUpload(record);
  }

  // Holds ids (not records) so a store slice-swap doesn't invalidate it. The anchor backs shift-range
  // extension and drops on clear so a later shift-click can't resurrect a stale range.
  const selectedIds = new SvelteSet<string>();
  let selectionAnchor = $state<string | null>(null);
  const selectionCount = $derived(selectedIds.size);
  const hasSelection = $derived(selectionCount > 0);
  // Compare against the eligible (non-deleting) subset: a mid-delete row is never selected, so without
  // the filter the select-all/deselect-all label would stick on "Select all".
  const allSelected = $derived.by(() => {
    const eligible = list.entries.filter((s) => !isDeleting(s.id));
    return eligible.length > 0 && eligible.every((s) => selectedIds.has(s.id));
  });

  // Scope the count to THIS pane's prefix: `deletingIds` is one cross-pane set keyed `${ws}/${cat}/${id}`,
  // so a raw `.size` would leak another category's deletes during a row-switch.
  const flightPrefix = $derived(`${workspaceId}/${categoryName}/`);
  const inflightDeleteCount = $derived.by(() => {
    let n = 0;
    for (const k of slices.deletingIds) {
      if (k.startsWith(flightPrefix)) n++;
    }
    return n;
  });
  const isAnyDeleting = $derived(inflightDeleteCount > 0);

  let mode = $state<'normal' | 'selecting'>('normal');

  function enterSelecting(): void {
    if (mode !== 'selecting') mode = 'selecting';
  }

  // Done/Esc: clear the selection so a stale set doesn't surprise on re-entry.
  function exitSelecting(): void {
    if (mode === 'selecting') {
      mode = 'normal';
      selectedIds.clear();
      selectionAnchor = null;
    }
  }

  // Auto-exit selecting when the grid empties, else the toolbar lingers over an empty-state body.
  $effect(() => {
    if (list.entries.length === 0 && mode === 'selecting') {
      exitSelecting();
    }
  });

  // Prune selected ids whose slice left the store; reading `list.entries` keys the effect.
  $effect(() => {
    const entries = list.entries;
    if (selectedIds.size === 0) return;
    const live = new Set(entries.map((s) => s.id));
    let mutated = false;
    for (const id of selectedIds) {
      if (!live.has(id)) {
        selectedIds.delete(id);
        mutated = true;
      }
    }
    if (mutated && selectionAnchor !== null && !live.has(selectionAnchor)) {
      selectionAnchor = null;
    }
  });

  // "Deselect all": empties but stays in selecting mode (unlike Done) so the operator can re-pick.
  function clearSelection(): void {
    selectedIds.clear();
    selectionAnchor = null;
  }

  // Skips mid-delete rows (they leave `entries` on completion, so selecting one leaves a ghost id) and
  // anchors at the first non-deleting entry so a follow-up shift-click extends from the top.
  function selectAll(): void {
    if (list.entries.length === 0) return;
    enterSelecting();
    let anchor: string | null = null;
    for (const s of list.entries) {
      if (isDeleting(s.id)) continue;
      selectedIds.add(s.id);
      anchor ??= s.id;
    }
    selectionAnchor = anchor;
  }

  function toggleSelectAll(): void {
    if (allSelected) clearSelection();
    else selectAll();
  }

  function toggleSelection(id: string): void {
    // Card click can't reach a deleting row (pointer-events-none), but right-click "Select" and
    // keyboard paths can.
    if (isDeleting(id)) return;
    // Auto-enter mode only on the add branch so a Ctrl/Cmd-click in normal mode flips into selecting.
    if (selectedIds.has(id)) {
      selectedIds.delete(id);
      if (selectionAnchor === id) selectionAnchor = null;
    } else {
      enterSelecting();
      selectedIds.add(id);
      selectionAnchor = id;
    }
  }

  // Range-select from `selectionAnchor` (fallback: first selected id, then clicked id) over the
  // visible order of `list.entries` so the range matches what the operator sees.
  function selectRange(toId: string): void {
    const entries = list.entries;
    const toIdx = entries.findIndex((s) => s.id === toId);
    if (toIdx < 0) return;
    let fromId = selectionAnchor;
    if (fromId === null) {
      const firstSelected = entries.find((s) => selectedIds.has(s.id));
      fromId = firstSelected?.id ?? toId;
    }
    const fromIdx = entries.findIndex((s) => s.id === fromId);
    if (fromIdx < 0) {
      // Anchor went stale before the prune effect ran; fall back to a single toggle.
      toggleSelection(toId);
      return;
    }
    enterSelecting();
    const [lo, hi] = fromIdx <= toIdx ? [fromIdx, toIdx] : [toIdx, fromIdx];
    for (let i = lo; i <= hi; i++) {
      if (isDeleting(entries[i].id)) continue;
      selectedIds.add(entries[i].id);
    }
    selectionAnchor = toId;
  }

  function onPick(slice: SliceRecord, mods: { toggle: boolean; range: boolean }): void {
    if (mods.range) {
      selectRange(slice.id);
    } else if (mods.toggle) {
      toggleSelection(slice.id);
    }
  }

  // Targets snapshot synchronously (`.filter` before any await) so a background upload or concurrent
  // single-card delete can't shift the batch.
  function bulkDelete(): void {
    if (selectedIds.size === 0) return;
    // Pane-scoped gate so a drain in another category neither blocks this batch nor leaves the button
    // no-op'ing; the per-target filter below covers this pane's add-after-gate.
    if (isAnyDeleting) return;
    const targets = list.entries.filter((s) => selectedIds.has(s.id) && !isDeleting(s.id));
    if (targets.length === 0) return;
    // `slices.delete()` doesn't cut audio; stop only if the playing slice is targeted.
    if (playingId !== null && targets.some((s) => s.id === playingId)) {
      stopPlayback();
    }
    // Clear before kicking the pipeline so a follow-up click or held Backspace bails at the gate
    // above; the captured `targets` array is the source of truth.
    clearSelection();
    void runBulkDelete(targets);
  }

  // Re-add failures to the selection (re-entering selecting mode) so the operator can retry without
  // rebuilding the batch; re-selection is the only feedback (no toast surface yet).
  async function runBulkDelete(targets: SliceRecord[]): Promise<void> {
    const outcome = await slices.deleteMany(targets);
    for (const f of outcome.failed) selectedIds.add(f.id);
    if (outcome.failed.length > 0) enterSelecting();
  }

  function retrySelected(): void {
    const targets = list.entries.filter((s) => selectedIds.has(s.id) && s.state === 'failed');
    for (const record of targets) {
      retryUpload(record);
    }
  }

  // Keydown binds to the grid (not window) so a Backspace inside a form input in another pane never
  // fires a destructive slice action.
  let gridEl = $state<HTMLDivElement | undefined>();

  // Viewport-gated spectrogram loading so a large grid doesn't fetch+FFT every card per expand. The IO
  // is rooted on `gridEl` so its own overflow scroll drives intersection; `visibleIds` gates each
  // SliceCard's `visible`; a MutationObserver re-observes new card nodes on list mutation.
  const visibleIds = new SvelteSet<string>();
  $effect(() => {
    const root = gridEl;
    if (!root) return;
    // No IO fallback needed: every target shipping AudioWorklet (recorder requirement) also ships IO.
    const obs = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const id = (entry.target as HTMLElement).dataset.sliceId;
          if (!id) continue;
          if (entry.isIntersecting) visibleIds.add(id);
          else visibleIds.delete(id);
        }
      },
      {
        root,
        // Pre-fetch one card-height (~64 px) above/below so a smooth scroll lands on resolving cards.
        rootMargin: '64px 0px',
        threshold: 0.01
      }
    );
    for (const card of root.querySelectorAll<HTMLElement>('[data-slice-id]')) {
      obs.observe(card);
    }
    const mo = new MutationObserver((mutations) => {
      // Loop var is `mutation`, not `m`, to avoid shadowing the imported i18n catalog proxy.
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (node instanceof HTMLElement && node.dataset.sliceId) {
            obs.observe(node);
          } else if (node instanceof HTMLElement) {
            for (const card of node.querySelectorAll<HTMLElement>('[data-slice-id]')) {
              obs.observe(card);
            }
          }
        }
      }
    });
    mo.observe(root, { childList: true });
    return () => {
      obs.disconnect();
      mo.disconnect();
      visibleIds.clear();
    };
  });

  // Prune `visibleIds` of removed cards: IO drops a disconnected node WITHOUT an `isIntersecting:false`
  // callback, so a deleted card's id would otherwise linger until unmount.
  $effect(() => {
    const entries = list.entries;
    if (visibleIds.size === 0) return;
    const live = new Set(entries.map((s) => s.id));
    for (const id of visibleIds) {
      if (!live.has(id)) visibleIds.delete(id);
    }
  });

  function onGridKey(e: KeyboardEvent): void {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    if ((e.metaKey || e.ctrlKey) && (e.key === 'a' || e.key === 'A')) {
      if (list.entries.length === 0) return;
      // preventDefault also suppresses the browser's select-all-text on the grid chrome.
      e.preventDefault();
      toggleSelectAll();
      return;
    }
    if (e.key === 'Escape' && mode === 'selecting') {
      e.preventDefault();
      exitSelecting();
      return;
    }
    if ((e.key === 'Delete' || e.key === 'Backspace') && hasSelection && !isAnyDeleting) {
      // Refuse while draining so a held Backspace can't re-fire over its own drain.
      e.preventDefault();
      bulkDelete();
    }
  }

  let menuOpen = $state(false);
  let menuX = $state(0);
  let menuY = $state(0);
  let menuSections = $state<MenuSection[]>([]);

  function onGridContextMenu(e: MouseEvent): void {
    const target = e.target as HTMLElement | null;
    if (!target) return;
    const cardEl = target.closest<HTMLElement>('[data-slice-id]');
    const id = cardEl?.dataset.sliceId ?? null;
    const slice = id ? (list.entries.find((s) => s.id === id) ?? null) : null;
    if (!slice) return;
    // A keyboard-chord can resolve a mid-delete card's data-slice-id even though pointer-events-none
    // routes mouse contextmenu to the grid: never open a menu over an unactionable row.
    if (isDeleting(slice.id)) return;
    e.preventDefault();
    // Stop bubbling so the enclosing category-row and page menus don't also open and stack three menus
    // that orphan each other on close.
    e.stopPropagation();
    menuX = e.clientX;
    menuY = e.clientY;
    menuSections = buildSliceMenu(slice);
    menuOpen = true;
  }

  // Destructive item is batch "Delete N" only when in-selection AND selecting mode; every other case
  // (including selecting mode on an unselected card) is single-card "Delete".
  function buildSliceMenu(slice: SliceRecord): MenuSection[] {
    const t = m.category.slice_pane;
    const isInSelection = selectedIds.has(slice.id);
    const isPlaying = playingId === slice.id;

    const cardItems: MenuItem[] = [
      {
        label: isPlaying ? t.menu_stop : t.menu_play,
        onclick: (): void => {
          if (isPlaying) stopPlayback();
          else void play(slice);
        }
      }
    ];
    if (slice.state === 'failed') {
      cardItems.push({ label: t.menu_retry_upload, onclick: () => retryUpload(slice) });
    }

    const selItems: MenuItem[] = [];
    if (mode === 'selecting') {
      selItems.push({
        label: isInSelection ? t.menu_deselect : t.menu_select,
        onclick: () => toggleSelection(slice.id)
      });
      selItems.push({
        label: allSelected ? t.menu_deselect_all : t.menu_select_all,
        hint: t.menu_hint_a,
        onclick: toggleSelectAll
      });
      selItems.push({
        label: t.menu_done_exit,
        hint: t.menu_hint_esc,
        onclick: exitSelecting
      });
      // Shown only with a failed entry in the selection, else it is no-op clutter.
      const anyFailedInSelection = list.entries.some(
        (s) => selectedIds.has(s.id) && s.state === 'failed'
      );
      if (anyFailedInSelection) {
        selItems.push({
          label: t.menu_retry_failed_in_selection,
          onclick: retrySelected
        });
      }
    } else {
      selItems.push({
        label: t.menu_select,
        hint: t.menu_hint_ctrl_click,
        onclick: () => toggleSelection(slice.id)
      });
      if (list.entries.length > 1) {
        selItems.push({
          label: t.menu_select_all,
          hint: t.menu_hint_a,
          onclick: selectAll
        });
      }
    }

    const isBatchDelete = mode === 'selecting' && isInSelection && hasSelection;
    const destItem: MenuItem = isBatchDelete
      ? {
          label: t.menu_delete_batch(selectionCount),
          hint: t.menu_hint_del_backspace,
          variant: 'destructive',
          onclick: bulkDelete
        }
      : {
          label: t.menu_delete,
          variant: 'destructive',
          onclick: () => void deleteSlice(slice)
        };

    return [{ items: cardItems }, { items: selItems }, { items: [destItem] }];
  }
</script>

<!-- `contain-size` welds the parent grid row to its min-h floor; without it the inner grid's
     max-content height (walked past overflow-y-auto) lifts the row above the sibling pane and leaves
     an empty band. -->
<section
  class="flex h-full min-h-0 flex-col rounded-md border border-line bg-surface px-3 pt-1.5 pb-3 contain-size"
>
  <header class="mb-1.5 flex min-h-4.75 items-center justify-between gap-1.5">
    <!-- `min-h-4.75` (19 px) matches the sibling pane's header for equal-height boxes; load-bearing in
         selecting mode where toolbar pills ~1 px shorter than the quota pill would jiggle the header on
         mode switch. -->
    {#if mode === 'selecting'}
      <div class="flex min-w-0 items-center gap-1.5">
        <button
          type="button"
          onclick={toggleSelectAll}
          class="inline-flex items-center rounded-md border border-line bg-surface px-1.5 py-0.5 text-[10px] font-medium text-fg-secondary transition duration-200 ease-out hover:border-line-strong hover:bg-surface-2"
          title={allSelected
            ? m.category.slice_pane.deselect_all_title
            : m.category.slice_pane.select_all_title}
        >
          {allSelected
            ? m.category.slice_pane.deselect_all_label
            : m.category.slice_pane.select_all_label}
        </button>
        <button
          type="button"
          onclick={exitSelecting}
          class="inline-flex items-center rounded-md border border-line bg-surface px-1.5 py-0.5 text-[10px] font-medium text-fg-secondary transition duration-200 ease-out hover:border-line-strong hover:bg-surface-2"
          title={m.category.slice_pane.done_title}
        >
          {m.category.slice_pane.done_label}
        </button>
        <button
          type="button"
          onclick={bulkDelete}
          disabled={!hasSelection || isAnyDeleting}
          class="inline-flex items-center gap-1 rounded-md border border-danger-line bg-danger-soft px-1.5 py-0.5 text-[10px] font-medium text-danger-soft-fg transition duration-200 ease-out hover:border-danger-line hover:bg-danger-soft disabled:cursor-not-allowed disabled:border-line disabled:bg-surface-2 disabled:text-fg-subtle"
          title={isAnyDeleting
            ? m.category.slice_pane.delete_inflight_title(inflightDeleteCount)
            : hasSelection
              ? m.category.slice_pane.delete_title
              : m.category.slice_pane.delete_disabled_title}
          aria-label={isAnyDeleting
            ? m.category.slice_pane.delete_inflight_aria(inflightDeleteCount)
            : hasSelection
              ? m.category.slice_pane.delete_aria_count(selectionCount)
              : m.category.slice_pane.delete_aria_fallback}
          aria-live="polite"
        >
          {#if isAnyDeleting}
            <Spinner class="h-2.5 w-2.5" />
          {:else}
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              class="h-2.5 w-2.5"
              aria-hidden="true"
            >
              <path d="M3 6h18" />
              <path d="M8 6V4h8v2" />
              <path d="M19 6l-1 14H6L5 6" />
            </svg>
          {/if}
          {#if isAnyDeleting}
            {m.category.slice_pane.delete_label_inflight(inflightDeleteCount)}
          {:else if hasSelection}
            {m.category.slice_pane.delete_label_count(selectionCount)}
          {:else}
            {m.category.slice_pane.delete_label_bare}
          {/if}
        </button>
      </div>
    {:else}
      <!-- `translate-y-px` balances the heading; normal-mode only, as selecting-mode bordered buttons
           mask their own bias. -->
      <div class="flex translate-y-px items-center gap-1.5">
        <h4 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
          {m.category.slice_pane.heading}
        </h4>
        <!-- `translate-y-px` cancels the Tips button's cap-line `-translate-y-px` so its icon centres on
             the heading. -->
        <span class="inline-flex translate-y-px">
          <Tips label={m.category.slice_pane.tips_label}>
            <ul class="space-y-1.5">
              <li>
                <strong class="font-medium text-fg">
                  {m.category.slice_pane.tip_audition_title}
                </strong>
                {m.category.slice_pane.tip_audition_body}
              </li>
              <li>
                <strong class="font-medium text-fg">
                  {m.category.slice_pane.tip_diversity_title}
                </strong>
                {m.category.slice_pane.tip_diversity_body}
              </li>
            </ul>
          </Tips>
        </span>
      </div>
    {/if}
    <!-- `tabular-nums` keeps chip width stable while the count animates. -->
    <span
      class="inline-flex shrink-0 items-center rounded-full px-1.5 py-0.5 font-mono text-[10px] font-medium tabular-nums transition-colors"
      class:bg-success-soft={satisfiesQuota}
      class:text-success-soft-fg={satisfiesQuota}
      class:bg-warning-soft={!satisfiesQuota}
      class:text-warning-soft-fg={!satisfiesQuota}
      title={satisfiesQuota
        ? m.category.slice_pane.quota_above_title(threshold)
        : m.category.slice_pane.quota_below_title(threshold)}
    >
      {count}/{threshold}
    </span>
  </header>

  {#if !list.loaded}
    <LoadingRow size="fill" label={m.category.slice_pane.loading} />
  {:else if list.error && list.entries.length === 0}
    <div
      class="rounded-md border border-warning-line bg-warning-soft px-3 py-2 text-xs text-warning-soft-fg"
      role="alert"
    >
      {m.category.slice_pane.load_error(list.error)}
    </div>
  {:else if list.entries.length === 0}
    <div
      class="flex flex-1 flex-col items-center justify-center gap-1 rounded-md border border-dashed border-line bg-surface-2 p-3 text-center"
    >
      <p class="text-[11px] text-fg-muted">
        {m.category.slice_pane.empty_state_prefix}<span class="font-medium"
          >{m.category.slice_pane.empty_state_button}</span
        >{m.category.slice_pane.empty_state_suffix}
      </p>
    </div>
  {:else}
    <!-- Grid contract: `auto-fill` (not auto-fit) keeps empty tracks so a lone slice doesn't balloon;
         `content-start` stops align-content from stretching the `auto` rows and ballooning the inter-row
         gap; `scrollbar-gutter-stable` reserves the scrollbar so an overflow batch doesn't reflow the
         column count; `tabindex=0` scopes the Del/Backspace shortcut to focus inside this grid. -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <div
      bind:this={gridEl}
      tabindex="0"
      class="grid min-h-0 flex-1 grid-cols-[repeat(auto-fill,minmax(96px,1fr))] content-start gap-3 overflow-y-auto rounded-sm scrollbar-gutter-stable focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      oncontextmenu={onGridContextMenu}
      onkeydown={onGridKey}
    >
      {#each list.entries as slice (slice.id)}
        <SliceCard
          {slice}
          playing={playingId === slice.id}
          selected={selectedIds.has(slice.id)}
          multiSelectActive={mode === 'selecting'}
          deleting={isDeleting(slice.id)}
          visible={visibleIds.has(slice.id)}
          onPlay={() => void play(slice)}
          onPick={(mods: { toggle: boolean; range: boolean }) => onPick(slice, mods)}
          onDelete={() => void deleteSlice(slice)}
          onRetry={() => retryUpload(slice)}
        />
      {/each}
    </div>
  {/if}
</section>

<ContextMenu
  open={menuOpen}
  x={menuX}
  y={menuY}
  sections={menuSections}
  onclose={() => (menuOpen = false)}
/>
