<script lang="ts">
  import InfoIcon from '$lib/components/ui/InfoIcon.svelte';
  import Spinner from '$lib/components/Spinner.svelte';
  import { heads as headsApi } from '$lib/api/endpoints';
  import { errorCopy } from '$lib/utils/error-copy';
  import { formatBytes } from '$lib/utils/format';
  import { formatRelative } from '$lib/utils/time';
  import { prettyCategoryName } from '$lib/components/category/labels';
  import { randomHex } from '$lib/utils/random';
  import { m } from '$lib/i18n';
  import type { HeadManifest, HeadRecord, Uuid } from '$lib/api/types';

  // Identity paints from the HeadRecord; only class labels (absent from the list response) gate on a per-instance-cached manifest fetch.

  interface Props {
    head: HeadRecord;
    workspaceId: Uuid;
  }
  let { head, workspaceId }: Props = $props();

  let open = $state(false);
  let triggerEl = $state<HTMLButtonElement | undefined>();
  let panelEl = $state<HTMLDivElement | undefined>();
  let closeTimer: ReturnType<typeof setTimeout> | null = null;
  // Set on open so an effect focuses the panel once visible: a hover peek never steals focus, a deliberate click/keyboard open does.
  let pendingFocus = false;

  // randomHex (not crypto.randomUUID) stays defined in insecure-context origins.
  const panelId = `head-info-panel-${randomHex(4)}`;

  // Panel portals to body, positioned fixed in viewport coords (right-aligned to icon, flipped above near the bottom edge): the icon's transformed/overflow-scroll ancestors would clip an absolute panel and mis-anchor a plain fixed one. Anchor rect captured once per open; scroll/resize closes rather than chase.
  let anchorRect = $state<DOMRect | null>(null);
  let measured = $state({ w: 0, h: 0 });
  // Matches --popover-edge-inset (1rem) so the clamp reserves the same inset as the width cap.
  const EDGE_GUTTER = 16;
  const ANCHOR_GAP = 8;
  const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(v, hi));

  $effect(() => {
    // Reading displayLoading re-measures on the loading->loaded/error swap, else a flipped-above bottom-row card overflows when the taller final manifest lands. Reset measured only on close (not here) so the panel doesn't blink hidden mid-session.
    void displayLoading;
    if (!open) {
      measured = { w: 0, h: 0 };
      return;
    }
    if (!panelEl) return;
    const el = panelEl;
    const raf = requestAnimationFrame(() => {
      measured = { w: el.offsetWidth, h: el.offsetHeight };
    });
    // Cancel a pending measure on re-run so a stale read can't land against a superseded node.
    return () => cancelAnimationFrame(raf);
  });

  const computedX = $derived.by(() => {
    if (!anchorRect) return 0;
    if (!measured.w) return anchorRect.right;
    return clamp(
      anchorRect.right - measured.w,
      EDGE_GUTTER,
      window.innerWidth - measured.w - EDGE_GUTTER
    );
  });
  const computedY = $derived.by(() => {
    if (!anchorRect) return 0;
    const below = anchorRect.bottom + ANCHOR_GAP;
    if (!measured.h) return below;
    if (below + measured.h <= window.innerHeight - EDGE_GUTTER) return below;
    const above = anchorRect.top - ANCHOR_GAP - measured.h;
    return above >= EDGE_GUTTER
      ? above
      : clamp(below, EDGE_GUTTER, window.innerHeight - measured.h - EDGE_GUTTER);
  });
  // Right-aligned placement is wrong until measured (pre-measure left = anchorRect.right); stay invisible that one frame so the panel never flashes left-edge-anchored at the icon's right edge before right-alignment is computed.
  const ready = $derived(measured.w > 0);
  const panelStyle = $derived(
    `position: fixed; left: ${computedX}px; top: ${computedY}px; visibility: ${ready ? 'visible' : 'hidden'};`
  );

  function cancelClose(): void {
    if (closeTimer !== null) {
      clearTimeout(closeTimer);
      closeTimer = null;
    }
  }
  // Deferred so the cursor can cross the gap into the panel (whose mouseenter cancels it); a focus-holding card stays open, left for focusout/Escape to own.
  function scheduleClose(): void {
    cancelClose();
    closeTimer = setTimeout(() => {
      closeTimer = null;
      const active = document.activeElement;
      if (panelEl?.contains(active) || triggerEl?.contains(active)) return;
      closeNow();
    }, 120);
  }
  function openNow(): void {
    cancelClose();
    if (open) return;
    if (triggerEl) anchorRect = triggerEl.getBoundingClientRect();
    open = true;
  }
  function closeNow(): void {
    cancelClose();
    open = false;
    // Reset focus intent so a pre-focus dismissal (scroll/Escape in the pre-measure frame) can't bleed into a later hover. Bumping requestSeq + dropping loading/error invalidates any in-flight fetch (no stale write post-dismissal), so a dismissed in-flight or errored card reopens into a clean GET; a manifest that already landed isn't reset, so it stays cached for an instant re-open.
    pendingFocus = false;
    requestSeq++;
    loading = false;
    loadError = null;
  }
  function closeAndRestore(): void {
    const t = triggerEl;
    closeNow();
    if (t?.isConnected) t.focus();
  }
  // Hover-to-peek gated to a real mouse: a touch tap synthesizes pointerenter THEN click, so an ungated open-on-enter would open then immediately toggle shut. Touch/pen fall through to the click toggle.
  function onPointerEnter(e: PointerEvent): void {
    if (e.pointerType === 'mouse') openNow();
  }
  function onPointerLeave(e: PointerEvent): void {
    if (e.pointerType === 'mouse') scheduleClose();
  }
  // Touch tap / keyboard Enter opens and pins (focus moves in); a mouse click on the already-hovered card closes it. stopPropagation keeps the tap off the row's deploy click.
  function onTriggerClick(e: MouseEvent): void {
    e.stopPropagation();
    if (open) {
      closeAndRestore();
      return;
    }
    pendingFocus = true;
    openNow();
  }
  // Panel branch needed because the portaled panel is no longer a DOM descendant of the trigger.
  function onFocusOut(e: FocusEvent): void {
    const next = e.relatedTarget as Node | null;
    if (next && (triggerEl?.contains(next) || panelEl?.contains(next))) return;
    closeNow();
  }

  // Focus into the panel only for a deliberate open, once visible (focus is a no-op while visibility:hidden).
  $effect(() => {
    if (!open || !ready || !pendingFocus || !panelEl) return;
    if (!panelEl.contains(document.activeElement)) panelEl.focus();
    pendingFocus = false;
  });

  // While open, dismiss on outside pointerdown, Escape/Tab, and scroll/resize. A pointerdown on the trigger isn't "outside", so its own click toggles the card shut.
  $effect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent): void => {
      const target = e.target as Node | null;
      if (triggerEl?.contains(target) || panelEl?.contains(target)) return;
      closeNow();
    };
    const onKey = (e: KeyboardEvent): void => {
      // Restore focus to the trigger only when the card holds focus (deliberate open); a hover-peek never focused it.
      const focused = !!panelEl?.contains(document.activeElement);
      if (e.key === 'Escape') {
        if (focused) closeAndRestore();
        else closeNow();
      } else if (e.key === 'Tab' && focused) {
        // Always preventDefault: portaled to body-end, a native Tab out lands in document-end limbo. If it holds a control (error-state Retry) and focus sits on the container, advance into it; else close and hand focus to the trigger.
        e.preventDefault();
        const firstControl = panelEl?.querySelector('button');
        if (firstControl && document.activeElement === panelEl) firstControl.focus();
        else closeAndRestore();
      }
    };
    // Fixed to the captured anchor, scroll would strand it; no internal scroller (chip box height-capped, overflow-hidden), so a capture-phase document listener closes unconditionally.
    const onScroll = (): void => closeNow();
    const onResize = (): void => closeNow();
    document.addEventListener('pointerdown', onDown, true);
    document.addEventListener('keydown', onKey);
    document.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onResize);
    return () => {
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onResize);
    };
  });

  $effect(() => () => cancelClose());

  // Hoists the panel to document.body so no transformed/overflow ancestor clips or mis-anchors it.
  function portal(node: HTMLElement): { destroy: () => void } {
    document.body.appendChild(node);
    return {
      destroy(): void {
        node.remove();
      }
    };
  }

  let manifest = $state<HeadManifest | null>(null);
  let loading = $state(false);
  let loadError = $state<string | null>(null);
  // Monotonic token: a late response is discarded on mismatch. Bumped by a superseding retry and by closeNow so a fetch in flight at dismissal can't settle into stale state.
  let requestSeq = 0;
  // True from first open until the manifest lands/errors, covering the tick before the fetch effect runs so the card opens straight into the loading state.
  const displayLoading = $derived(manifest === null && loadError === null);

  // Fetch once on first open; an error parks here (no auto-retry loop) until Retry refires it.
  $effect(() => {
    if (!open || manifest !== null || loading || loadError !== null) return;
    void fetchManifest();
  });

  async function fetchManifest(): Promise<void> {
    const myId = ++requestSeq;
    loading = true;
    loadError = null;
    try {
      // Named `resp` not `m` to avoid shadowing the imported i18n `m` proxy.
      const resp = await headsApi.manifest(workspaceId, head.head_id);
      if (myId !== requestSeq) return;
      manifest = resp;
    } catch (e) {
      if (myId !== requestSeq) return;
      loadError = errorCopy(e);
    } finally {
      if (myId === requestSeq) loading = false;
    }
  }

  function retry(): void {
    void fetchManifest();
  }
</script>

<!-- Trigger is the hover/focus bridge AND anchor. Revealed via opacity + pointer-events (not display) so the row never reflows on hover. Never disabled (read-only fetch), so title/aria-label sit on the button directly - no Export-style wrapper span (see HeadRow.svelte), which exists only because a disabled button surfaces no title tooltip / fires no pointer events in Firefox. -->
<button
  bind:this={triggerEl}
  type="button"
  onclick={onTriggerClick}
  onpointerenter={onPointerEnter}
  onpointerleave={onPointerLeave}
  onfocusout={onFocusOut}
  title={m.deploy.head_row.info_title}
  aria-label={m.deploy.head_row.info_aria(head.head_id.slice(0, 8))}
  aria-haspopup="dialog"
  aria-expanded={open}
  aria-controls={open ? panelId : undefined}
  class="pointer-events-none inline-flex shrink-0 items-center justify-center rounded-md p-1.5 text-fg-subtle opacity-0 transition duration-200 ease-out group-hover/row:pointer-events-auto group-hover/row:opacity-100 hover:bg-accent-soft hover:text-accent focus-visible:pointer-events-auto focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-accent-line focus-visible:outline-none pointer-coarse:pointer-events-auto pointer-coarse:opacity-100"
>
  <InfoIcon />
</button>

{#if open}
  <!-- Carries its own hover handlers: post-portal it isn't a DOM descendant of the trigger, so the trigger's handlers see no cursor traffic over it. z-40: above playback cursor (z-30) and trim handles (z-20), below the native-dialog top layer. -->
  <div
    bind:this={panelEl}
    use:portal
    id={panelId}
    role="dialog"
    aria-label={m.deploy.info_dialog.title_with_id(head.head_id.slice(0, 8))}
    tabindex="-1"
    style={panelStyle}
    onpointerenter={onPointerEnter}
    onpointerleave={onPointerLeave}
    onfocusout={onFocusOut}
    class="fixed z-40 w-72 max-w-[calc(100vw-var(--popover-edge-inset)*2)] rounded-md border border-line bg-surface shadow-popover ring-1 ring-black/5 focus:outline-none"
  >
    <div class="flex flex-col gap-2 px-3 py-2">
      <div class="flex flex-col gap-0.5">
        <code class="font-mono text-[11px] font-semibold break-all text-fg">{head.head_id}</code>
        <p class="text-[10px] text-fg-muted">
          {m.deploy.head_row.meta_line(
            formatBytes(head.size_bytes),
            head.n_classes,
            head.workspace_revision.id,
            formatRelative(head.created_at)
          )}
        </p>
      </div>

      {#if displayLoading}
        <div class="flex items-center gap-2 text-[10px] text-fg-muted" aria-live="polite">
          <Spinner class="h-3 w-3 text-accent" />
          {m.deploy.info_dialog.loading}
        </div>
      {:else if loadError}
        <!-- role="status" (polite), not "alert": the card can open on a passive hover, so an assertive interrupt would fire on an incidental graze of the icon. -->
        <div class="text-[10px]" role="status">
          <p class="text-danger-soft-fg">{m.deploy.info_dialog.error_title}</p>
          <button
            type="button"
            onclick={retry}
            class="mt-1 rounded-sm font-medium text-accent hover:underline focus-visible:ring-2 focus-visible:ring-accent-line focus-visible:outline-none"
          >
            {m.deploy.info_dialog.retry}
          </button>
        </div>
      {:else if manifest && manifest.labels.length > 0}
        <!-- Chips capped via max-h-20 overflow-hidden so a many-label head doesn't wall the card. prettyCategoryName formats reserved synthetics (e.g. `_background_noise_`); operator labels pass through verbatim. -->
        <div class="flex flex-col gap-1">
          <span class="text-[10px] font-medium tracking-wider text-fg-muted uppercase">
            {m.deploy.info_dialog.classes_heading}
          </span>
          <ul
            class="flex max-h-20 flex-wrap gap-1 overflow-hidden"
            aria-label={m.deploy.info_dialog.class_labels_aria}
          >
            {#each manifest.labels as label, idx (`${idx}-${label}`)}
              <li
                class="inline-flex max-w-full items-center rounded-full bg-surface-2 px-1.5 py-0.5 text-[10px] wrap-break-word text-fg-secondary"
              >
                {prettyCategoryName(label)}
              </li>
            {/each}
          </ul>
        </div>
      {/if}
    </div>
  </div>
{/if}
