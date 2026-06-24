<script lang="ts">
  import { config } from '$lib/stores/config.svelte';
  import { workspaces as wsApi } from '$lib/api/endpoints';
  import { formatRelativeShort } from '$lib/utils/time';
  import { formatLabelsList } from '$lib/components/category/labels';
  import { m } from '$lib/i18n';

  let headActive = $derived(config.active?.origin === 'head' ? config.active : null);
  // Only explicit `false` is orphaned: POST omits the field (undefined) and GET sends `true` when alive; both mean not-detached.
  let orphaned = $derived(headActive !== null && headActive.source_workspace_alive === false);

  let origin = $derived(config.active?.origin ?? null);
  let nClasses = $derived(config.active?.n_classes ?? null);
  // Inline on the active record: powers the class-tile hover `title` with no per-head GET.
  let classLabels = $derived(config.active?.labels ?? null);
  let activatedAt = $derived(config.active?.activated_at ?? null);

  // Active record ships the workspace id but not its name: lazy-fetch for live heads (alive
  // true/undefined), and KEEP any session-resolved name on alive===false so the row reads "Test
  // (deleted)". Direct `wsApi.get` (not the workspaces store, whose slices/categories/drafts
  // imports add ~50 kB to this chunk) takes no AbortSignal, so the post-await id+alive re-check is
  // the only stale-write guard - the alive clause stops a mid-fetch deletion clobbering the name.
  let wsName = $state<string | null>(null);
  let wsNameStatus = $state<'idle' | 'loading' | 'error'>('idle');
  // Plain `let`, not `$state`: read/written only inside the effect below; a signal would self-loop.
  let prevId: string | null = null;

  $effect(() => {
    const cur = headActive;
    if (cur === null) {
      wsName = null;
      wsNameStatus = 'idle';
      prevId = null;
      return;
    }
    if (cur.source_workspace_alive === false) {
      wsNameStatus = 'idle';
      return;
    }
    const id = cur.source_workspace_id;
    // Clear only on a genuine head swap; same-id re-fetches (auto-reconnect config.refresh yields a
    // fresh reference, same id) keep the prior name until the response lands, killing flicker.
    if (id !== prevId) wsName = null;
    prevId = id;
    wsNameStatus = 'loading';
    // Cleanup-flipped: skips setters on a post-unmount `.then`, which the id/alive re-check (a
    // stale-swap guard) cannot distinguish from a live re-fetch.
    let cancelled = false;
    void wsApi.get(id).then(
      (detail) => {
        if (cancelled) return;
        const a = config.active;
        if (
          a?.origin === 'head' &&
          a.source_workspace_id === id &&
          a.source_workspace_alive !== false
        ) {
          wsName = detail.name;
          wsNameStatus = 'idle';
        }
      },
      () => {
        if (cancelled) return;
        const a = config.active;
        if (
          a?.origin === 'head' &&
          a.source_workspace_id === id &&
          a.source_workspace_alive !== false
        ) {
          wsNameStatus = 'error';
        }
      }
    );
    return () => {
      cancelled = true;
    };
  });

  // Adaptive tick: formatRelativeShort floors to integer units, so wake only at the next boundary
  // (60 s / 60 min / 24 h, then once per day), not at 1 Hz. Backgrounded tabs throttle long timers and fire
  // late on refocus, so visibilitychange forces an immediate update+reschedule; 250 ms floor avoids
  // a tight loop when a reschedule lands just before a boundary.
  let now = $state(Date.now());
  $effect(() => {
    if (!activatedAt) return;
    const t = Date.parse(activatedAt);
    if (Number.isNaN(t)) return;

    // `| undefined` makes a pre-`schedule()` read a TS error, not a silent `clearTimeout(undefined)`.
    let timer: ReturnType<typeof setTimeout> | undefined;
    const schedule = (): void => {
      const elapsedMs = Date.now() - t;
      let nextDeltaMs: number;
      if (elapsedMs < 60_000) {
        nextDeltaMs = 60_000 - elapsedMs;
      } else if (elapsedMs < 3_600_000) {
        nextDeltaMs = 60_000 - (elapsedMs % 60_000);
      } else if (elapsedMs < 86_400_000) {
        nextDeltaMs = 3_600_000 - (elapsedMs % 3_600_000);
      } else {
        nextDeltaMs = 86_400_000 - (elapsedMs % 86_400_000);
      }
      timer = setTimeout(
        () => {
          now = Date.now();
          schedule();
        },
        Math.max(250, nextDeltaMs)
      );
    };
    const onVisibilityChange = (): void => {
      if (document.visibilityState === 'visible') {
        clearTimeout(timer);
        now = Date.now();
        schedule();
      }
    };
    schedule();
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  });

  let activeRelative = $derived(
    activatedAt ? formatRelativeShort(activatedAt, new Date(now)) : null
  );

  // origin -> operator vocabulary: default (daemon-bundled), workspace (live head), detached (workspace deleted).
  let pillLabel = $derived<'default' | 'workspace' | 'detached' | null>(
    origin === null ? null : orphaned ? 'detached' : origin === 'head' ? 'workspace' : 'default'
  );

  // Workspace-cell tooltip always carries the full UUID for copy-paste (the body never shows it).
  let wsTitle = $derived<string | null>(
    headActive === null
      ? null
      : orphaned
        ? wsName !== null
          ? m.dashboard.active_head_card.ws_title_orphaned_with_name(
              wsName,
              headActive.source_workspace_id
            )
          : m.dashboard.active_head_card.ws_title_orphaned(headActive.source_workspace_id)
        : wsName !== null
          ? m.dashboard.active_head_card.ws_title_with_name(wsName, headActive.source_workspace_id)
          : headActive.source_workspace_id
  );
</script>

<!-- Asymmetric top/bottom padding makes text insets read optically equal (top compensates pill-driven
     flex centering + half-leading); the pill is anchored on its text glyph, not its rounded box. -->
<aside
  class="rounded-lg border px-3.5 pt-2.5 pb-3 transition-colors duration-200"
  class:border-warning-line={orphaned}
  class:bg-warning-soft={orphaned}
  class:border-line={!orphaned}
  class:bg-page={!orphaned}
>
  <header class="mb-3 flex items-center justify-between gap-2">
    <h3 class="text-[11px] font-semibold uppercase tracking-wider text-fg-muted">
      {m.dashboard.active_head_card.heading}
    </h3>
    {#if pillLabel}
      <span
        class="rounded-full px-2 py-0.5 text-[11px] font-medium capitalize tracking-wide transition-colors duration-200"
        class:bg-surface-2={pillLabel === 'default'}
        class:text-fg-secondary={pillLabel === 'default'}
        class:bg-accent-soft={pillLabel === 'workspace'}
        class:text-accent-soft-fg={pillLabel === 'workspace'}
        class:bg-warning-soft={pillLabel === 'detached'}
        class:text-warning-soft-fg={pillLabel === 'detached'}
        title={pillLabel === 'detached'
          ? m.dashboard.active_head_card.pill_detached_title
          : pillLabel === 'default'
            ? m.dashboard.active_head_card.pill_default_title
            : m.dashboard.active_head_card.pill_workspace_title}
      >
        {pillLabel === 'detached'
          ? m.dashboard.active_head_card.pill_detached
          : pillLabel === 'workspace'
            ? m.dashboard.active_head_card.pill_workspace
            : m.dashboard.active_head_card.pill_default}
      </span>
    {/if}
  </header>

  {#if config.active === null}
    <!-- Skeletons only the always-present stat-tile pair (matching line-boxes for height parity);
         the workspace/revision <dl> is workspace-origin-only, so reserving it would shrink on the
         common default-head swap. `bg-line` is theme-safe (dark collapses --color-line and
         --color-surface-2 to zinc-800). role/aria-live live ONLY here, else the populated card's
         per-tick relative time would be announced every second. -->
    <div role="status" aria-live="polite" class="animate-pulse">
      <div aria-hidden="true" class="grid grid-cols-2 divide-x divide-line">
        <div class="pr-3 text-center">
          <div class="text-xl">
            <span class="inline-block h-4 w-14 rounded bg-line align-middle"></span>
          </div>
          <div class="mt-1 text-[10px]">
            <span class="inline-block h-2 w-12 rounded bg-line align-middle"></span>
          </div>
        </div>
        <div class="pl-3 text-center">
          <div class="text-xl">
            <span class="inline-block h-4 w-8 rounded bg-line align-middle"></span>
          </div>
          <div class="mt-1 text-[10px]">
            <span class="inline-block h-2 w-10 rounded bg-line align-middle"></span>
          </div>
        </div>
      </div>
      <span class="sr-only">{m.dashboard.active_head_card.loading_active}</span>
    </div>
  {:else}
    <!-- `tabular-nums` stabilizes digit columns across ticks ("9" -> "10"); only the time tile gets
         `min-w-0 truncate` since an over-long locale form can overflow, but class counts stay short. -->
    <div
      class="grid grid-cols-2 divide-x"
      class:divide-warning-line={orphaned}
      class:divide-line={!orphaned}
    >
      <div class="min-w-0 pr-3 text-center" title={activatedAt ?? undefined}>
        <div class="truncate text-xl font-semibold tabular-nums text-fg">
          {#if activeRelative}{activeRelative}{:else}<span class="text-fg-subtle">-</span>{/if}
        </div>
        <div class="mt-1 text-[10px] text-fg-muted">
          {m.dashboard.active_head_card.activated_label}
        </div>
      </div>
      <div
        class="pl-3 text-center"
        title={classLabels && classLabels.length > 0 ? formatLabelsList(classLabels) : undefined}
      >
        <div class="text-xl font-semibold tabular-nums text-fg">
          {#if nClasses !== null}{nClasses}{:else}<span class="text-fg-subtle">-</span>{/if}
        </div>
        <div class="mt-1 text-[10px] text-fg-muted">
          {m.dashboard.active_head_card.class_count_label(nClasses ?? 0)}
        </div>
      </div>
    </div>

    {#if headActive}
      <dl
        class="mt-3 grid grid-cols-[max-content_1fr] items-baseline gap-x-3 gap-y-1.5 border-t pt-3 text-xs"
        class:border-warning-line={orphaned}
        class:border-line={!orphaned}
      >
        <dt class="text-fg-muted">{m.dashboard.active_head_card.workspace_dt}</dt>
        <dd class="min-w-0 truncate text-fg" title={wsTitle ?? undefined}>
          {#if orphaned}
            {#if wsName !== null}
              <span class="font-medium text-warning-soft-fg">{wsName}</span>
              <span class="ml-1 text-warning-soft-fg italic">
                {m.dashboard.active_head_card.deleted_tag}
              </span>
            {:else}
              <span class="text-warning-soft-fg italic"
                >{m.dashboard.active_head_card.deleted_tag}</span
              >
            {/if}
          {:else if wsName !== null}
            <span class="font-medium">{wsName}</span>
          {:else if wsNameStatus === 'loading'}
            <span class="text-fg-subtle">{m.dashboard.active_head_card.loading}</span>
          {:else}
            <span class="font-mono text-[10px]"
              >{headActive.source_workspace_id.slice(0, 8)}<span class="text-fg-subtle">…</span
              ></span
            >
          {/if}
        </dd>

        <dt class="text-fg-muted">{m.dashboard.active_head_card.revision_dt}</dt>
        <dd class="truncate text-fg">
          {m.dashboard.active_head_card.rev_value(headActive.workspace_revision.id)}
        </dd>
      </dl>
    {/if}
  {/if}
</aside>
