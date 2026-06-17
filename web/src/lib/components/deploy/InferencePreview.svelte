<script lang="ts">
  import { fade } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import { streams } from '$lib/stores/streams.svelte';
  import { config } from '$lib/stores/config.svelte';
  import SpectrogramCanvas from '$lib/components/dashboard/SpectrogramCanvas.svelte';
  import TopKMeter from '$lib/components/dashboard/TopKMeter.svelte';
  import { m } from '$lib/i18n';
  import { socketLabel, socketPillClass } from '$lib/components/dashboard/socketPill';

  // 60 Hz RAF+FFT loop runs only while ON; OFF by default (not persisted). Started by manual CTA or
  // when activation_id changes from its mount baseline (daemon reassigns it per head activation == a
  // deploy); stopped only by IntersectionObserver (+ unmount disconnect on route exit).
  let preview = $state(false);
  let sectionEl: HTMLElement | undefined = $state();

  // Visible-ratio floor below which the pane counts as out-of-focus (0.2 = >80% off-screen).
  const STOP_INTERSECTION_RATIO = 0.2;

  // Dwell below threshold before auto-stop, filtering scroll glance-aways (<~600 ms over-trips).
  const STOP_DEBOUNCE_MS = 800;

  // Plain `let`, not `$state`: reactivity would self-re-trigger the in-effect write. Null baseline is
  // fresh per mount so navigate-back re-baselines rather than treating the remount fetch as a deploy.
  let lastSeenActivationId: string | null = null;

  $effect(() => {
    const cur = config.active?.activation_id ?? null;
    if (cur === null) {
      // Pre-load/disconnect: keep baseline so reconnect doesn't false-positive a deploy.
      return;
    }
    if (lastSeenActivationId === null) {
      lastSeenActivationId = cur; // first non-null observation is the baseline, not a deploy
      return;
    }
    if (cur !== lastSeenActivationId) {
      lastSeenActivationId = cur;
      preview = true;
    }
  });

  // Refcount streams only while ON (preview=false runs acquire()'s teardown). `$effect.pre` so
  // acquire()'s optimistic 'connecting' write lands before the first {#if preview} DOM commit reads
  // streams.inferStatus, else the pill flashes one red frame.
  $effect.pre(() => {
    if (preview) return streams.acquire();
  });

  $effect(() => {
    if (!preview || !sectionEl) return;
    let offScreenTimer: ReturnType<typeof setTimeout> | null = null;
    const observer = new IntersectionObserver(
      (entries) => {
        // Thresholds crossed in one frame batch; only the last entry reflects current state.
        const entry = entries[entries.length - 1];
        if (entry.intersectionRatio < STOP_INTERSECTION_RATIO) {
          // `??=` so a later off-focus event doesn't replace an already-pending stop timer.
          offScreenTimer ??= setTimeout(() => {
            preview = false;
            offScreenTimer = null;
          }, STOP_DEBOUNCE_MS);
        } else if (offScreenTimer !== null) {
          clearTimeout(offScreenTimer); // back in focus within the debounce window: cancel the stop
          offScreenTimer = null;
        }
      },
      { threshold: [0, STOP_INTERSECTION_RATIO] }
    );
    observer.observe(sectionEl);
    return () => {
      observer.disconnect();
      if (offScreenTimer !== null) clearTimeout(offScreenTimer);
    };
  });

  // Scroller mounts conditionally with preview, so fade-edge listeners wire via an effect keyed on
  // scrollEl's bind, not component onMount.
  let scrollEl = $state<HTMLDivElement | undefined>();
  let canScrollUp = $state(false);
  let canScrollDown = $state(false);

  function updateFades(el: HTMLDivElement): void {
    canScrollUp = el.scrollTop > 0;
    // `- 1` absorbs sub-pixel rounding that otherwise leaves the bottom fade stuck on at rest.
    canScrollDown = el.scrollTop + el.clientHeight < el.scrollHeight - 1;
  }

  // queueMicrotask defers re-measure until after Svelte flushes new rows so scrollHeight is fresh.
  $effect(() => {
    void streams.latestTopK;
    const el = scrollEl;
    if (!el) return;
    queueMicrotask(() => updateFades(el));
  });

  $effect(() => {
    const el = scrollEl;
    if (!el) return;
    const onScroll = (): void => updateFades(el);
    el.addEventListener('scroll', onScroll, { passive: true });
    const ro = new ResizeObserver(() => updateFades(el));
    ro.observe(el);
    updateFades(el);
    return () => {
      el.removeEventListener('scroll', onScroll);
      ro.disconnect();
    };
  });
</script>

<section
  bind:this={sectionEl}
  class="flex h-full min-h-0 flex-col rounded-md border border-line bg-surface px-3 pt-1.5 pb-3"
>
  <!-- min-h-4.75 locks heading-row height when off-state drops the fps+status chrome, so adjacent
       cards in the row keep a shared heading-bottom strip. -->
  <header class="mb-1.5 flex min-h-4.75 items-center justify-between gap-1.5">
    <div class="flex translate-y-px items-baseline gap-1.5">
      <h4 class="text-[11px] font-semibold tracking-wider text-fg-muted uppercase">
        {m.deploy.inference_preview.heading}
      </h4>
      {#if preview}
        <!-- Live-only so a stale "0.0 Hz" never dangles against a paused renderer. -->
        <span
          in:fade={{ duration: 160, easing: cubicOut }}
          out:fade={{ duration: 140, easing: cubicOut }}
          class="font-mono text-[10px] text-fg-subtle tabular-nums"
        >
          {streams.inferenceFps.toFixed(1)} Hz
        </span>
      {/if}
    </div>
    {#if preview}
      <span
        in:fade={{ duration: 160, easing: cubicOut }}
        out:fade={{ duration: 140, easing: cubicOut }}
        class="rounded-full px-2 py-0.5 text-[11px] font-medium capitalize tracking-wide transition-colors duration-200 {socketPillClass(
          streams.inferStatus
        )}"
      >
        {socketLabel(streams.inferStatus)}
      </span>
    {/if}
  </header>

  {#if preview}
    <!-- Spectrogram shrink-0 so flex-1 gives Top-K the larger share; min-h-0 lets the scroller clip
         past pane height; -mr-1 reclaims the scrollbar inset to align the right edge with px-3. -->
    <div class="flex min-h-0 flex-1 flex-col gap-2">
      <div class="h-24 shrink-0">
        <SpectrogramCanvas seconds={3} />
      </div>
      <div
        bind:this={scrollEl}
        class="-mr-1 min-h-0 flex-1 overflow-y-auto pr-1"
        class:fade-edge-top={canScrollUp}
        class:fade-edge-bottom={canScrollDown}
      >
        <TopKMeter />
      </div>
    </div>
  {:else}
    <!-- Off-state placeholder holds the same vertical budget so toggling doesn't reflow the row. -->
    <div
      class="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 rounded-md border border-dashed border-line-strong bg-surface-2/60 px-6 py-8 text-center"
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.5"
        class="h-8 w-8 text-fg-subtle"
        aria-hidden="true"
      >
        <path stroke-linecap="round" stroke-linejoin="round" d="M9 19V6l11 7-11 6zM4 4v16" />
      </svg>
      <div class="min-w-0">
        <p class="text-sm font-medium text-fg-secondary">{m.deploy.inference_preview.off_title}</p>
        <p class="mt-1 text-xs text-fg-muted">
          {m.deploy.inference_preview.off_description}
        </p>
      </div>
      <button
        type="button"
        onclick={() => (preview = true)}
        class="mt-1 inline-flex shrink-0 items-center justify-center gap-1.5 rounded-md border border-accent bg-accent px-3.5 py-1.5 text-sm font-medium text-fg-on-accent transition hover:border-accent-hover hover:bg-accent-hover"
      >
        {m.deploy.inference_preview.start_button}
      </button>
    </div>
  {/if}
</section>
