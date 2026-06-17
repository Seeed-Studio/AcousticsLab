<script lang="ts">
  import { health } from '$lib/stores/health.svelte';
  import { m } from '$lib/i18n';

  let open = $state(false);
  let wrapper = $state<HTMLDivElement | undefined>();
  let closeTimer: ReturnType<typeof setTimeout> | null = null;

  // No focus-to-open path: a tap focuses then clicks, so opening on focusin would race the click toggle.

  function cancelClose(): void {
    if (closeTimer !== null) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }
  }
  function scheduleClose(): void {
    cancelClose();
    closeTimer = setTimeout(() => {
      open = false;
      closeTimer = null;
    }, 120);
  }
  function openNow(): void {
    cancelClose();
    open = true;
  }
  function closeNow(): void {
    cancelClose();
    open = false;
  }
  function toggle(): void {
    if (open) closeNow();
    else openNow();
  }

  // Attached only while open so the closed state is event-free.
  $effect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent): void => {
      const root = wrapper;
      if (!root) return;
      if (!root.contains(e.target as Node | null)) closeNow();
    };
    document.addEventListener('pointerdown', onDown, true);
    return () => {
      document.removeEventListener('pointerdown', onDown, true);
    };
  });

  $effect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') closeNow();
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
    };
  });

  $effect(() => {
    return () => {
      cancelClose();
    };
  });

  // Dot color kept separate from the i18n label so it stays a stable Tailwind class across locale switch.
  const DOTS = {
    unknown: 'bg-fg-subtle',
    ok: 'bg-success-dot',
    degraded: 'bg-warning-dot',
    unhealthy: 'bg-danger-dot',
    unreachable: 'bg-danger-dot'
  } as const;

  let dotClass = $derived(DOTS[health.level]);
  let levelLabel = $derived(m.health.levels[health.level]);

  // Unit letters stay out of the i18n catalog: in font-mono they read as a stable machine label.
  function formatUptime(s: number): string {
    const h = Math.floor(s / 3600);
    const min = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}h ${min}m`;
    if (min > 0) return `${min}m ${sec}s`;
    return `${sec}s`;
  }

  // null relatedTarget (focus to a non-focusable element or out of the document) counts as leaving the scope.
  function onFocusOut(e: FocusEvent): void {
    const next = e.relatedTarget as Node | null;
    if (!next || !wrapper?.contains(next)) closeNow();
  }
</script>

<div
  bind:this={wrapper}
  class="relative inline-block"
  role="group"
  aria-label={m.health.aria_label}
  onmouseenter={openNow}
  onmouseleave={scheduleClose}
  onfocusout={onFocusOut}
>
  <!-- `py-3 sm:py-1.5`: below sm the label is hidden, so equal padding makes a 34x34 circle; at sm+ a 34x22 pill. -->
  <button
    type="button"
    class="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-3 text-xs font-medium text-fg-secondary capitalize shadow-card transition hover:border-line-strong sm:py-1.5"
    aria-expanded={open}
    aria-haspopup="dialog"
    onclick={toggle}
  >
    <span class="relative flex h-2.5 w-2.5">
      {#if health.level === 'ok'}
        <span class="absolute inline-flex h-full w-full animate-pulse-ring rounded-full {dotClass}"
        ></span>
      {/if}
      <span
        class="relative inline-flex h-2.5 w-2.5 rounded-full transition-colors duration-300 {dotClass}"
      ></span>
    </span>
    <span class="hidden sm:inline">{levelLabel}</span>
  </button>

  {#if open}
    <!-- Bridges the trigger-popover gap so the cursor never falls into dead space and starts the close timer. -->
    <div
      aria-hidden="true"
      class="absolute right-0 top-full z-30 h-2 w-80 max-w-[calc(100vw-var(--popover-edge-inset)*2)]"
    ></div>
    <div
      role="dialog"
      aria-label={m.health.aria_label}
      tabindex="-1"
      class="absolute right-0 z-30 mt-2 w-80 max-w-[calc(100vw-var(--popover-edge-inset)*2)] rounded-xl border border-line bg-elevated p-4 text-sm shadow-popover"
    >
      {#if health.lastError}
        <p class="font-medium text-danger-soft-fg">{m.health.popover.daemon_unreachable_title}</p>
        <p class="mt-1 text-xs text-fg-muted">{health.lastError}</p>
      {:else if !health.snapshot}
        <p class="text-fg-muted">{m.health.popover.waiting_first_snapshot}</p>
      {:else}
        {@const snap = health.snapshot}
        <div class="mb-3 flex items-center justify-between">
          <p class="text-xs font-semibold uppercase tracking-wide text-fg-muted">
            {m.health.popover.subsystems_heading}
          </p>
          {#if health.lastUpdated}
            <p class="text-[10px] font-mono text-fg-subtle">
              {m.health.popover.seconds_ago(
                Math.round((Date.now() - health.lastUpdated) / 100) / 10
              )}
            </p>
          {/if}
        </div>
        <ul class="space-y-1.5">
          {#each Object.entries(snap.subsystems) as [name, sub] (name)}
            <li class="flex items-center justify-between gap-2">
              <span class="font-mono text-xs text-fg-secondary">{name}</span>
              <span class="flex items-center gap-2">
                {#if sub.degraded_reason}
                  <span class="truncate text-xs text-warning-soft-fg" title={sub.degraded_reason}
                    >{sub.degraded_reason}</span
                  >
                {/if}
                <span
                  class="inline-block h-2 w-2 rounded-full"
                  class:bg-success-dot={sub.healthy && !sub.stale}
                  class:bg-warning-dot={sub.healthy && sub.stale}
                  class:bg-danger-dot={!sub.healthy}
                ></span>
              </span>
            </li>
          {/each}
        </ul>

        <div class="mt-3 grid grid-cols-3 gap-3 border-t border-line-subtle pt-3 text-xs">
          <div>
            <div class="font-mono text-fg">{snap.cpu_pct.toFixed(1)}%</div>
            <div class="text-fg-subtle">{m.health.popover.stat_cpu_label}</div>
          </div>
          <div>
            <div class="font-mono text-fg">{(snap.mem_rss_kb / 1024).toFixed(0)} MiB</div>
            <div class="text-fg-subtle">{m.health.popover.stat_rss_label}</div>
          </div>
          <div>
            <div class="font-mono text-fg">
              {(snap.disk_free_kb / 1024 / 1024).toFixed(1)} GiB
            </div>
            <div class="text-fg-subtle">{m.health.popover.stat_disk_free_label}</div>
          </div>
        </div>

        <div
          class="mt-3 flex items-center justify-between border-t border-line-subtle pt-3 text-[11px] text-fg-muted"
        >
          <span
            >{m.health.popover.uptime_label}
            <span class="font-mono text-fg-secondary">{formatUptime(snap.uptime_s)}</span></span
          >
          {#if snap.broadcast_audio_messages_dropped + snap.broadcast_inference_messages_dropped > 0}
            <span class="text-warning-soft-fg"
              >{m.health.popover.dropped_count(
                snap.broadcast_audio_messages_dropped + snap.broadcast_inference_messages_dropped
              )}</span
            >
          {/if}
        </div>
      {/if}
    </div>
  {/if}
</div>
