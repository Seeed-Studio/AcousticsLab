<script lang="ts">
  import { onMount } from 'svelte';
  import { streams } from '$lib/stores/streams.svelte';
  import TopKMeter from './TopKMeter.svelte';
  import ActiveHeadCard from './ActiveHeadCard.svelte';
  import { socketLabel, socketPillClass } from './socketPill';
  import { m } from '$lib/i18n';

  // Fade edges show only on the side where rows are hidden, signalling more content.
  let scrollEl = $state<HTMLDivElement | undefined>();
  let canScrollUp = $state(false);
  let canScrollDown = $state(false);

  function updateFades(el: HTMLDivElement): void {
    canScrollUp = el.scrollTop > 0;
    canScrollDown = el.scrollTop + el.clientHeight < el.scrollHeight - 1;
  }

  $effect(() => {
    // Re-measure on every Top-K change.
    void streams.latestTopK;
    const el = scrollEl;
    if (!el) return;
    queueMicrotask(() => {
      updateFades(el);
    });
  });

  onMount(() => {
    const el = scrollEl;
    if (!el) return;
    const onScroll = (): void => {
      updateFades(el);
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    const ro = new ResizeObserver(() => {
      updateFades(el);
    });
    ro.observe(el);
    updateFades(el);
    return () => {
      el.removeEventListener('scroll', onScroll);
      ro.disconnect();
    };
  });
</script>

<!-- Fixed `--vis-panel-h` at every breakpoint so the panel never reflows with width. -->
<section
  class="flex h-(--vis-panel-h) flex-col rounded-xl border border-line bg-surface px-5 pt-3.5 pb-5 shadow-card"
>
  <!-- font-mono on the Hz value prevents digit-width jitter as the rate fluctuates. -->
  <header class="mb-3 flex items-center justify-between gap-3">
    <div class="flex items-baseline gap-2">
      <h2 class="text-sm font-semibold text-fg">{m.dashboard.inference_panel.heading}</h2>
      <span class="font-mono text-[11px] text-fg-muted">{streams.inferenceFps.toFixed(1)} Hz</span>
    </div>
    <span
      class="rounded-full px-2 py-0.5 text-[11px] font-medium capitalize tracking-wide transition-colors duration-200 {socketPillClass(
        streams.inferStatus
      )}"
    >
      {socketLabel(streams.inferStatus)}
    </span>
  </header>

  <!-- min-h-0 lets this flex child clip+scroll so Top-K growth never resizes the panel. -->
  <div
    bind:this={scrollEl}
    class="min-h-0 flex-1 overflow-y-auto pr-1"
    class:fade-edge-top={canScrollUp}
    class:fade-edge-bottom={canScrollDown}
  >
    <TopKMeter />
  </div>

  <div class="pt-4">
    <ActiveHeadCard />
  </div>
</section>
