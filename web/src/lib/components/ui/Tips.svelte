<script lang="ts">
  import type { Snippet } from 'svelte';
  import { randomHex } from '$lib/utils/random';

  // Panel portals to `document.body` to escape ancestor `overflow: hidden` clipping and a
  // `transform`/`filter`/`perspective` ancestor that would become the containing block for
  // `position: fixed` (painting viewport coords off-position). Cost: panel is no longer a
  // wrapper descendant, so wrapper `mouseleave` fires on leaving the icon -- the 120 ms
  // close-timer + panel `mouseenter` absorb the gap, and outside-tap/focusout treat it as
  // inside via `panelEl?.contains`. `z-40`: above trim handles (z-20)/cursor (z-30), below modals (z-50).
  interface Props {
    label: string;
    children: Snippet;
  }
  let { label, children }: Props = $props();

  // Per-instance `aria-controls` so AT follows trigger -> panel across the portal; `randomHex`
  // not `crypto.randomUUID` (undefined in insecure contexts).
  const panelId = `tips-panel-${randomHex(4)}`;

  let open = $state(false);
  let wrapper = $state<HTMLDivElement | undefined>();
  let panelEl = $state<HTMLDivElement | undefined>();
  let closeTimer: ReturnType<typeof setTimeout> | null = null;

  // Viewport-relative icon edges driving side-choice + horizontal clamp; null while closed.
  let anchor = $state<{ top: number; bottom: number; left: number } | null>(null);

  function refreshAnchor(): void {
    if (!wrapper) {
      anchor = null;
      return;
    }
    const rect = wrapper.getBoundingClientRect();
    anchor = { top: rect.top, bottom: rect.bottom, left: rect.left };
  }

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
    refreshAnchor();
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

  // Hoist node to `document.body`; `node.remove()` no-ops if already detached (idempotent teardown).
  function portal(node: HTMLElement): { destroy: () => void } {
    document.body.appendChild(node);
    return {
      destroy(): void {
        node.remove();
      }
    };
  }

  // Outside-tap dismissal; `panelEl?.contains` required since the panel's DOM parent is `document.body`
  // (a wrapper-only check would close on the first click inside the panel).
  $effect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent): void => {
      const target = e.target as Node | null;
      if (!target) return;
      if (wrapper?.contains(target)) return;
      if (panelEl?.contains(target)) return;
      closeNow();
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
    return () => cancelClose();
  });

  // `capture` catches scroll on any ancestor (scroll doesn't bubble); `passive` keeps it off the path.
  $effect(() => {
    if (!open) return;
    refreshAnchor();
    const update = (): void => refreshAnchor();
    window.addEventListener('scroll', update, { capture: true, passive: true });
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('scroll', update, { capture: true });
      window.removeEventListener('resize', update);
    };
  });

  // Close when focus leaves both trigger and panel; `panelEl?.contains(next)` keeps Tab into the panel inside.
  function onFocusOut(e: FocusEvent): void {
    const next = e.relatedTarget as Node | null;
    if (next && (wrapper?.contains(next) || panelEl?.contains(next))) return;
    closeNow();
  }

  // Must mirror the panel's classes: `w-72` = PANEL_WIDTH, `--popover-edge-inset` (1rem) = EDGE_INSET.
  const PANEL_WIDTH = 288;
  const EDGE_INSET = 16;
  const GAP = 8;

  // Position the fixed panel; CSS caps only width, so clamp here to keep the inset gutter on
  // every edge. Null anchor (pre-first-open or wrapper unmounted mid-tick) -> `display: none`
  // so the first paint never lands at the viewport origin.
  const panelStyle = $derived.by(() => {
    if (anchor === null) return 'display: none;';
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    // Left-align to the icon, clamped to both gutters; clamping `width` to vw first keeps the
    // valid range non-empty even when panel + gutters don't fit.
    const width = Math.min(PANEL_WIDTH, vw - EDGE_INSET * 2);
    const left = Math.max(EDGE_INSET, Math.min(anchor.left, vw - width - EDGE_INSET));
    // Open on the side with more room; `max-height` + `overflow-y-auto` scroll an over-tall
    // body internally. Stateless choice may oscillate sides near mid-viewport, harmless since
    // both have room there.
    const spaceBelow = vh - anchor.bottom - GAP - EDGE_INSET;
    const spaceAbove = anchor.top - GAP - EDGE_INSET;
    const vertical =
      spaceBelow >= spaceAbove
        ? `top: ${anchor.bottom + GAP}px; max-height: ${Math.max(0, spaceBelow)}px;`
        : `bottom: ${vh - anchor.top + GAP}px; max-height: ${Math.max(0, spaceAbove)}px;`;
    return `left: ${left}px; ${vertical}`;
  });
</script>

<div
  bind:this={wrapper}
  class="relative inline-flex"
  role="group"
  aria-label={label}
  onmouseenter={openNow}
  onmouseleave={scheduleClose}
  onfocusout={onFocusOut}
>
  <!-- 10 px icon (sub-WCAG-24 px, acceptable for a soft-discovery affordance; touch users get
       click-toggle). `-translate-y-px` corrects the ~1 px line-box-centre vs cap-centre gap,
       applied to the button not the wrapper so the wrapper's anchor rect stays at rest. -->
  <button
    type="button"
    class="inline-flex h-2.5 w-2.5 -translate-y-px items-center justify-center rounded-full text-fg-subtle transition hover:text-fg-secondary focus-visible:text-fg-secondary focus-visible:ring-2 focus-visible:ring-accent-line focus-visible:outline-none"
    aria-label={label}
    aria-expanded={open}
    aria-haspopup="dialog"
    aria-controls={open ? panelId : undefined}
    onclick={toggle}
  >
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2.25"
      stroke-linecap="round"
      stroke-linejoin="round"
      class="h-2.5 w-2.5"
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </svg>
  </button>
</div>

{#if open}
  <!-- Own hover handlers: post-portal the panel is not a wrapper descendant, so the wrapper's
       handlers see no cursor traffic over it. -->
  <div
    bind:this={panelEl}
    use:portal
    id={panelId}
    role="dialog"
    aria-label={label}
    tabindex="-1"
    class="fixed z-40 w-72 max-w-[calc(100vw-var(--popover-edge-inset)*2)] overflow-y-auto rounded-lg border border-line bg-elevated px-3 py-2 text-[11px] leading-snug text-fg-secondary shadow-popover"
    style={panelStyle}
    onmouseenter={openNow}
    onmouseleave={scheduleClose}
    onfocusout={onFocusOut}
  >
    {@render children()}
  </div>
{/if}
