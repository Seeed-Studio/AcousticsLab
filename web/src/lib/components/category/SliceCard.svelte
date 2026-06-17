<script lang="ts">
  import { untrack } from 'svelte';
  import { getSliceSpectrogramUrl } from '$lib/audio/spectrogram';
  import { sliceFilename } from '$lib/idb/db';
  import type { SliceRecord } from '$lib/idb/db';
  import { theme } from '$lib/stores/theme.svelte';
  import type { ResolvedTheme } from '$lib/stores/theme.svelte';
  import { m } from '$lib/i18n';

  // Stateless: selection state lives in the parent, routed through onPick.
  interface PickModifiers {
    toggle: boolean;
    range: boolean;
  }
  interface Props {
    slice: SliceRecord;
    playing: boolean;
    selected: boolean;
    multiSelectActive: boolean;
    // In-flight delete signal, pane-driven not derived from `slice.state` (any state can enter delete()).
    deleting: boolean;
    onPlay: () => void;
    onPick: (mods: PickModifiers) => void;
    onDelete: () => void;
    onRetry: () => void;
    // Viewport-intersection gate (incl. rootMargin buffer) so the WAV download + FFT only fire for in-view cards.
    visible: boolean;
  }
  let {
    slice,
    playing,
    selected,
    multiSelectActive,
    deleting,
    visible,
    onPlay,
    onPick,
    onDelete,
    onRetry
  }: Props = $props();

  const isUploading = $derived(slice.state === 'uploading');
  const isFailed = $derived(slice.state === 'failed');
  const isLocal = $derived(slice.state === 'local');
  const progressPct = $derived(
    isUploading ? Math.round(Math.max(0, Math.min(1, slice.upload_progress ?? 0)) * 100) : 0
  );
  const filename = $derived(sliceFilename(slice.id));

  let url = $state<string | null>(null);
  let pending = $state(false);

  // Viewport-gated, 150ms-debounced fetch: fast-scroll transits (<150ms visible) clear the timer
  // in cleanup and never fetch. Each PNG bakes the active palette so the cached url is theme-locked:
  // a theme flip drops the stale url and re-fetches, gated on a PREVIOUS lastSeenTheme so a
  // freshly-mounted card (url null) doesn't churn on its first theme observation. `slice` is read
  // inside untrack so a parent SvelteMap patch flipping its identity doesn't re-fetch (slice.id is
  // stable); a failed fetch leaves url=null so scroll-away-and-back re-fires fresh.
  const sliceId = $derived(slice.id);
  // Plain `let`, not `$state`: reactivity would feed the effect's own write back into a re-run, defeating the change-only gate.
  let lastSeenTheme: ResolvedTheme | null = null;
  $effect(() => {
    const id = sliceId;
    const inView = visible;
    const themeNow = theme.resolved;

    // Drop stale url on a genuine theme transition; skip first run.
    if (lastSeenTheme !== null && lastSeenTheme !== themeNow) {
      url = null;
      pending = false;
    }
    lastSeenTheme = themeNow;

    if (!inView) return;
    if (url !== null) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      if (cancelled) return;
      pending = true;
      untrack(() => {
        void getSliceSpectrogramUrl(slice, themeNow)
          .then((u) => {
            if (cancelled) return;
            url = u;
          })
          .catch((e: unknown) => {
            if (cancelled) return;
            console.warn(`[slice ${id}] spectrogram render failed`, e);
            url = null;
          })
          .finally(() => {
            if (cancelled) return;
            pending = false;
          });
      });
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      // `.finally` bails on `cancelled`, so reset here or a mid-flight cancel leaves the placeholder stuck.
      pending = false;
    };
  });

  function onDeleteClick(e: MouseEvent): void {
    e.stopPropagation();
    onDelete();
  }

  function onRetryClick(e: MouseEvent): void {
    e.stopPropagation();
    onRetry();
  }

  // ctrl/cmd toggles, shift range-extends, multi-select mode makes bare click toggle (file-manager idiom), else bare click plays. Branch here so the modifier read happens on the same event.
  function onCardClick(e: MouseEvent): void {
    if (e.ctrlKey || e.metaKey || e.shiftKey) {
      e.preventDefault();
      onPick({ toggle: e.ctrlKey || e.metaKey, range: e.shiftKey });
      return;
    }
    if (multiSelectActive) {
      e.preventDefault();
      onPick({ toggle: true, range: false });
      return;
    }
    onPlay();
  }

  function onSelectClick(e: MouseEvent): void {
    e.stopPropagation();
    onPick({ toggle: true, range: false });
  }
</script>

<!-- Parent's contextmenu handler walks closest() on data-slice-id. Descendant buttons also carry
     disabled={deleting} because pointer-events-none does NOT block keyboard activation of a focused descendant. -->
<div
  class="group relative transition-opacity duration-200 ease-out"
  class:opacity-50={deleting}
  class:pointer-events-none={deleting}
  data-slice-id={slice.id}
  aria-selected={selected}
  aria-busy={deleting}
>
  <!-- aspect-3/2 preserves the 96:64 PNG ratio so time-per-pixel stays constant across pane widths.
       Every state shares border-2 (not ring-*, which overflows the grid cell) for a constant box; selected yields to playing/failed. -->
  <button
    type="button"
    disabled={deleting}
    class="block aspect-3/2 w-full overflow-hidden rounded-md border-2 bg-surface-2 transition duration-200 ease-out focus:outline-none"
    class:border-line={!playing && !isFailed && !selected}
    class:border-accent={!isFailed && (playing || selected)}
    class:border-danger-line={isFailed}
    class:bg-accent-soft={selected && !playing && !isFailed}
    class:hover:border-line-strong={!playing && !isFailed && !selected}
    onclick={onCardClick}
    aria-label={multiSelectActive
      ? selected
        ? m.category.slice_card.aria_deselect(filename)
        : m.category.slice_card.aria_select(filename)
      : m.category.slice_card.aria_play(filename)}
    title={isFailed
      ? m.category.slice_card.title_failed(slice.last_error ?? m.category.slice_card.unknown_error)
      : isUploading
        ? m.category.slice_card.title_uploading(progressPct)
        : isLocal
          ? m.category.slice_card.title_local
          : multiSelectActive
            ? selected
              ? m.category.slice_card.title_multi_click_deselect
              : m.category.slice_card.title_multi_click_select
            : playing
              ? m.category.slice_card.title_playing
              : m.category.slice_card.title_idle}
  >
    {#if url}
      <!-- No loading="lazy": the card already viewport-gates the fetch and the url is in-tab blob data, not network, so eager is wanted; decoding="async" keeps the decode off the main thread. -->
      <img src={url} alt="" width="96" height="64" decoding="async" class="block h-full w-full" />
    {:else if pending}
      <!-- No spinner: cache hits resolve in a microtask and fresh renders in ~10-50ms, below the "is this loading?" threshold. -->
      <div class="h-full w-full bg-surface-2" aria-hidden="true"></div>
    {:else}
      <!-- Failed-render fallback: bg-line darkens past pending bg-surface-2 in light mode; both collapse to zinc-800 in dark, so only the wave icon distinguishes them. -->
      <div class="flex h-full w-full items-center justify-center bg-line">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          class="h-4 w-4 text-fg-subtle"
          aria-hidden="true"
        >
          <path d="M3 12h2l3-8 4 16 3-8h2" />
        </svg>
      </div>
    {/if}
  </button>

  <!-- White fill + dark drop-shadow read against either end of the grayscale ramp; pointer-events-none keeps the button the hit target. -->
  {#if !multiSelectActive && !playing && !isUploading && !isFailed && !deleting}
    <div
      class="pointer-events-none absolute inset-0 flex items-center justify-center opacity-0 transition-opacity duration-200 ease-out group-hover:opacity-100"
      aria-hidden="true"
    >
      <svg viewBox="0 0 24 24" class="h-7 w-7 fill-white drop-shadow-[0_1px_2px_rgb(0_0_0/0.55)]">
        <path d="M8 5v14l11-7z" />
      </svg>
    </div>
  {/if}

  <!-- No per-card spinner: N animate-spin cards in a bulk delete burned N transform recalcs + compositor layers/frame against the live-audio RAF loop, so the spinner lives in the toolbar; per-card feedback is root opacity/aria-busy plus this sr-only. -->
  {#if deleting}
    <span class="sr-only">{m.category.slice_card.sr_deleting(filename)}</span>
  {/if}

  {#if isUploading}
    <div
      class="pointer-events-none absolute right-0 bottom-0 left-0 h-1.5 bg-fg/40"
      aria-hidden="true"
    >
      <div
        class="h-full bg-accent transition-[width] duration-150"
        style:width="{progressPct}%"
      ></div>
    </div>
    <span class="sr-only">{m.category.slice_card.sr_uploading(progressPct)}</span>
  {/if}

  <!-- Retry badge always visible (not hover-gated) so failed slices read at a glance. -->
  {#if isFailed}
    <button
      type="button"
      disabled={deleting}
      class="absolute bottom-1 left-1 inline-flex items-center gap-0.5 rounded-md bg-danger-soft px-1 py-0.5 text-[9px] font-medium text-danger-soft-fg transition duration-200 ease-out hover:bg-danger-soft"
      onclick={onRetryClick}
      aria-label={m.category.slice_card.retry_aria(filename)}
      title={slice.last_error
        ? m.category.slice_card.retry_title_with_error(slice.last_error)
        : m.category.slice_card.retry_title_no_error}
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2.5"
        stroke-linecap="round"
        stroke-linejoin="round"
        class="h-2.5 w-2.5"
        aria-hidden="true"
      >
        <path d="M3 12a9 9 0 0115-6.7L21 8" />
        <path d="M21 3v5h-5" />
      </svg>
      {m.category.slice_card.retry_label}
    </button>
  {/if}

  <!-- Top-right checkbox in selecting mode, hover trash otherwise, sharing an anchor + hit area so the target doesn't shift on a mode flip. Trash is pointer-events-none at rest so the card's bare-click play handler stays unblocked. -->
  {#if multiSelectActive}
    <button
      type="button"
      disabled={deleting}
      onclick={onSelectClick}
      class="absolute top-1.5 right-1.5 inline-flex h-5 w-5 items-center justify-center rounded-md shadow-card transition duration-200 ease-out"
      class:bg-accent={selected}
      class:text-fg-on-accent={selected}
      class:hover:bg-accent-hover={selected}
      class:bg-surface={!selected}
      class:ring-1={!selected}
      class:ring-inset={!selected}
      class:ring-line-strong={!selected}
      class:hover:ring-accent-hover={!selected}
      class:hover:bg-accent-soft={!selected}
      aria-label={selected
        ? m.category.slice_card.slice_deselect_aria(filename)
        : m.category.slice_card.slice_select_aria(filename)}
      title={selected ? m.category.slice_card.deselect_title : m.category.slice_card.select_title}
    >
      {#if selected}
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="3"
          stroke-linecap="round"
          stroke-linejoin="round"
          class="h-3 w-3"
          aria-hidden="true"
        >
          <path d="M20 6L9 17l-5-5" />
        </svg>
      {/if}
    </button>
  {:else}
    <button
      type="button"
      disabled={deleting}
      class="pointer-events-none absolute top-1.5 right-1.5 inline-flex h-5 w-5 items-center justify-center rounded-md bg-surface text-danger-soft-fg opacity-0 shadow-card transition duration-200 ease-out group-hover:pointer-events-auto group-hover:opacity-100 focus-visible:pointer-events-auto focus-visible:opacity-100 hover:bg-danger-soft"
      onclick={onDeleteClick}
      aria-label={m.category.slice_card.delete_aria(filename)}
      title={m.category.slice_card.delete_title}
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
        class="h-3 w-3"
        aria-hidden="true"
      >
        <path d="M3 6h18" />
        <path d="M8 6V4h8v2" />
        <path d="M19 6l-1 14H6L5 6" />
      </svg>
    </button>
  {/if}
</div>
