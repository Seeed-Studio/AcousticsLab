<script lang="ts">
  import { tick } from 'svelte';
  import { stageLabel } from './labels';
  import { m } from '$lib/i18n';
  import { locale } from '$lib/stores/locale.svelte';
  import type { TrainingLogLine } from '$lib/stores/training.svelte';

  // Plain `Map` (not `SvelteMap`): `.set` runs from a template expr ({fmtTime}) on a cold locale, where Svelte 5's `state_unsafe_mutation` guard would turn `SvelteMap.set` into a mount-killing throw.
  // eslint-disable-next-line svelte/prefer-svelte-reactivity
  const TIME_FORMATTERS = new Map<string, Intl.DateTimeFormat>();
  function timeFormatter(loc: string): Intl.DateTimeFormat {
    let f = TIME_FORMATTERS.get(loc);
    if (!f) {
      f = new Intl.DateTimeFormat(loc, {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
      });
      TIME_FORMATTERS.set(loc, f);
    }
    return f;
  }

  // Daemon keeps no history, so this client-synthesised log is the only place to re-read a run's trace.
  interface Props {
    lines: readonly TrainingLogLine[];
  }
  let { lines }: Props = $props();

  const VIEWPORT_HEIGHT = 144;

  // Within-floor distance (px) counting as pinned to bottom; buffer absorbs DPR sub-pixel rounding.
  const STICK_PX = 4;

  let scrollEl: HTMLDivElement | undefined = $state();
  // True at mount so a fresh log auto-scrolls to bottom on first paint.
  let stuckToBottom = $state(true);

  function onScroll(): void {
    const el = scrollEl;
    if (!el) return;
    const distance = el.scrollHeight - el.clientHeight - el.scrollTop;
    stuckToBottom = distance <= STICK_PX;
  }

  // Auto-tail only when pinned, so a deliberate scroll-up survives. Dep on `lines.length` (not the array ref) skips no-op re-emits; `tick()` lets the DOM reflect new lines before reading `scrollHeight`, else the write hits the stale height.
  $effect(() => {
    void lines.length;
    if (!stuckToBottom) return;
    const el = scrollEl;
    if (!el) return;
    void tick().then(() => {
      el.scrollTop = el.scrollHeight;
    });
  });

  function fmtTime(at: string): string {
    // Daemon RFC 3339 and seed `Date.toISOString()` both round-trip through `Date`, so NaN means a malformed `at`.
    const d = new Date(at);
    if (Number.isNaN(d.getTime())) return '--:--:--';
    // Reading `locale.resolved` makes it a render dep so a locale switch re-runs every {fmtTime} with the new digit shaping.
    return timeFormatter(locale.resolved).format(d);
  }
</script>

<!-- Bounded height + internal scroll so accumulating messages can't grow the card. -->
<div class="overflow-hidden rounded-md border border-line bg-surface-2">
  <div class="flex items-center justify-between border-b border-line bg-surface px-3 py-1.5">
    <span class="text-[10px] font-medium text-fg-muted">{m.training.logs.heading}</span>
    <span class="font-mono text-[10px] text-fg-subtle tabular-nums">
      {m.training.logs.entry_count(lines.length)}
    </span>
  </div>
  <!-- `tabindex=0` makes the scrollback focusable so arrow keys can reach a clipped line's tail; without focus the horizontal scroll is pointer-only. -->
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div
    bind:this={scrollEl}
    onscroll={onScroll}
    tabindex="0"
    aria-label={m.training.logs.heading}
    class="overflow-auto px-3 py-2 text-[11px] focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-line focus-visible:outline-none"
    style="height: {VIEWPORT_HEIGHT}px;"
    role="log"
    aria-live="polite"
    aria-relevant="additions"
  >
    {#if lines.length === 0}
      <p class="font-mono text-[10px] text-fg-subtle">{m.training.logs.waiting_first_message}</p>
    {:else}
      <!-- Key by `line.seq` so a cap-driven shift recycles rows in place; daemon seq is per-job monotonic and the seed uses seq=-1, so keys never collide. `min-w-max` sizes the list to its widest line so messages scroll sideways instead of wrapping. -->
      <ol class="flex min-w-max flex-col gap-0.5 font-mono leading-snug">
        {#each lines as line (line.seq)}
          <li class="flex gap-2 text-fg-secondary">
            <span
              class="shrink-0 text-fg-subtle tabular-nums"
              title="{line.at} · {stageLabel(line.phase)}"
            >
              {fmtTime(line.at)}
            </span>
            <!-- `whitespace-pre` keeps each message one line; clipped tail recoverable via `title` hover or focus. -->
            <span class="whitespace-pre text-fg-secondary" title={line.message}>{line.message}</span
            >
          </li>
        {/each}
      </ol>
    {/if}
  </div>
</div>
