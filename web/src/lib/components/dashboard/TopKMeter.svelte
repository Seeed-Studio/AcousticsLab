<script lang="ts">
  import { streams } from '$lib/stores/streams.svelte';
  import { prettyCategoryName } from '$lib/components/category/labels';
  import { m } from '$lib/i18n';

  // Container-query (host width tracks its grid share, not viewport): below 20rem fix the 7rem label,
  // above fix the bar and let minmax(0,1fr) shrink the label below its intrinsic width.
  let rows = $derived(streams.latestTopK);
</script>

<!-- @container: root is the inline-size container the rows' @[20rem]: resolves against. -->
<div class="@container space-y-2">
  {#if rows.length === 0}
    <!-- empty also covers released/stalled streams (store clears stale Top-K after 2s), so the pulse is
         neutral not "first frame". Mirrors the live grid 1:1 (same template/gap-3/items-center, h-5 =
         live label's 1.25rem line-box) so the swap never reflows; all three stubs kept so the fixed 7rem
         bar isn't floating against an empty label in the wide layout. space-y-2 re-declared since under
         {#if} the root's reaches only this child. bg-line is the only block colour readable on bg-surface
         in both themes (dark collapses --color-line/--color-surface-2 to zinc-800; surface-2 invisible on
         white). Live region ONLY here (populated meter re-renders every frame); rows aria-hidden, sr-only
         message on the trailing line. -->
    <div role="status" aria-live="polite" class="animate-pulse space-y-2">
      {#each [0, 1, 2] as i (i)}
        <div
          aria-hidden="true"
          class="grid h-5 grid-cols-[7rem_1fr_3rem] items-center gap-3 @[20rem]:grid-cols-[minmax(0,1fr)_7rem_3rem]"
        >
          <span class="h-3 w-20 rounded bg-line"></span>
          <span class="h-2 rounded-full bg-line"></span>
          <span class="h-3 w-9 justify-self-end rounded bg-line"></span>
        </div>
      {/each}
      <span class="sr-only">{m.dashboard.top_k_meter.awaiting_first_frame}</span>
    </div>
  {:else}
    {#each rows as row (row.class_idx)}
      {@const pct = Number.isFinite(row.prob) ? Math.max(0, Math.min(1, row.prob)) : 0}
      <div
        class="grid grid-cols-[7rem_1fr_3rem] items-center gap-3 @[20rem]:grid-cols-[minmax(0,1fr)_7rem_3rem]"
      >
        <!-- title = raw on-wire token; cell shows the prettified name. -->
        <span class="truncate text-sm font-medium text-fg-secondary" title={row.label}
          >{prettyCategoryName(row.label)}</span
        >
        <!-- overflow-hidden clips the fill to the pill so tiny values render as a sliver, not a circle. -->
        <div class="relative h-2 overflow-hidden rounded-full bg-surface-2">
          <div
            class="absolute inset-y-0 left-0 bg-accent transition-[width] duration-150"
            style="width: {pct * 100}%"
          ></div>
        </div>
        <span class="text-right font-mono text-xs text-fg-muted">{(pct * 100).toFixed(1)}%</span>
      </div>
    {/each}
  {/if}
</div>
