<script lang="ts" module>
  export interface MenuItem {
    label: string;
    onclick: () => void;
    variant?: 'default' | 'destructive';
    disabled?: boolean;
    // Display only; the owner wires any actual shortcut listener.
    hint?: string;
  }

  export interface MenuSection {
    items: MenuItem[];
  }
</script>

<script lang="ts">
  import { fade } from 'svelte/transition';

  // Cursor menu (x, y) or button dropdown (anchorRect); clamps in-viewport and self-dismisses on
  // Escape/outside-click/scroll/resize. Focus contract: anchored menus focus item 0 on open;
  // cursor menus leave focus on the host (right-click must not steal it) until the first Arrow.
  // Arrow/Home/End rove; Escape/Tab close and restore focus to the trigger.
  interface Props {
    open: boolean;
    x: number;
    y: number;
    // Trigger rect for a button dropdown: the menu right-aligns to its right edge and opens below,
    // flipping above near the viewport bottom. Omit for cursor menus, where (x, y) is the click point.
    anchorRect?: DOMRect | null;
    // The opener button: a pointerdown on it isn't an outside-click, so its own click toggles the
    // menu closed instead of this listener racing it shut. Omit for cursor menus.
    triggerEl?: HTMLElement | null;
    sections: MenuSection[];
    onclose: () => void;
  }
  let { open, x, y, anchorRect = null, triggerEl = null, sections, onclose }: Props = $props();

  let menuEl = $state<HTMLDivElement | undefined>();
  // Measured after first paint to flip/align away from viewport edges; until then cursor menus
  // render at raw coords (fade masks the skew) and anchored menus stay hidden (`ready`).
  let measured = $state({ w: 0, h: 0 });

  $effect(() => {
    if (!open) {
      measured = { w: 0, h: 0 };
      return;
    }
    if (!menuEl) return;
    const el = menuEl;
    requestAnimationFrame(() => {
      measured = { w: el.offsetWidth, h: el.offsetHeight };
    });
  });

  const EDGE_GUTTER = 8;
  const ANCHOR_GAP = 4;
  const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(v, hi));

  const computedX = $derived.by(() => {
    if (anchorRect) {
      if (!measured.w) return anchorRect.right;
      return clamp(
        anchorRect.right - measured.w,
        EDGE_GUTTER,
        window.innerWidth - measured.w - EDGE_GUTTER
      );
    }
    if (!measured.w) return x;
    return clamp(x, EDGE_GUTTER, window.innerWidth - measured.w - EDGE_GUTTER);
  });
  const computedY = $derived.by(() => {
    if (anchorRect) {
      const below = anchorRect.bottom + ANCHOR_GAP;
      if (!measured.h) return below;
      if (below + measured.h <= window.innerHeight - EDGE_GUTTER) return below;
      const above = anchorRect.top - ANCHOR_GAP - measured.h;
      return above >= EDGE_GUTTER
        ? above
        : clamp(below, EDGE_GUTTER, window.innerHeight - measured.h - EDGE_GUTTER);
    }
    if (!measured.h) return y;
    return clamp(y, EDGE_GUTTER, window.innerHeight - measured.h - EDGE_GUTTER);
  });

  // Anchored placement is wrong until measured; keep it invisible for that frame.
  const ready = $derived(!anchorRect || measured.w > 0);

  function menuItems(): HTMLButtonElement[] {
    return menuEl
      ? Array.from(menuEl.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not([disabled])'))
      : [];
  }
  function focusItemAt(index: number): void {
    const items = menuItems();
    if (items.length === 0) return;
    items[((index % items.length) + items.length) % items.length].focus();
  }
  function moveFocus(delta: number): void {
    const items = menuItems();
    if (items.length === 0) return;
    const cur = items.indexOf(document.activeElement as HTMLButtonElement);
    focusItemAt(cur === -1 ? (delta > 0 ? 0 : -1) : cur + delta);
  }

  // Dismiss on scroll/resize too: the menu is fixed-positioned and would strand from its anchor.
  // `trigger` is captured before the focus effect moves focus in, so restore targets the opener;
  // activating an item leaves focus to its follow-up (e.g. a dialog), so only Escape/Tab restore.
  $effect(() => {
    if (!open) return;
    const trigger = document.activeElement as HTMLElement | null;
    const restoreAndClose = (): void => {
      if (trigger?.isConnected) trigger.focus();
      onclose();
    };
    const onDocPointer = (e: PointerEvent): void => {
      const root = menuEl;
      if (!root) return;
      const target = e.target as Node | null;
      if (root.contains(target) || triggerEl?.contains(target)) return;
      onclose();
    };
    const onKey = (e: KeyboardEvent): void => {
      switch (e.key) {
        case 'Escape':
        case 'Tab':
          e.preventDefault();
          restoreAndClose();
          break;
        case 'ArrowDown':
          e.preventDefault();
          moveFocus(1);
          break;
        case 'ArrowUp':
          e.preventDefault();
          moveFocus(-1);
          break;
        case 'Home':
          e.preventDefault();
          focusItemAt(0);
          break;
        case 'End':
          e.preventDefault();
          focusItemAt(-1);
          break;
      }
    };
    const onScroll = (): void => onclose();
    const onResize = (): void => onclose();
    document.addEventListener('pointerdown', onDocPointer, true);
    document.addEventListener('keydown', onKey);
    document.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onResize);
    return () => {
      document.removeEventListener('pointerdown', onDocPointer, true);
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onResize);
    };
  });

  // Auto-focus item 0, keyed on `ready` so it runs after visibility:visible (`.focus()` no-ops while
  // hidden); contains() fires once per open without yanking focus off a roved item. Declared after
  // the dismiss effect so that one captures `trigger` first. Anchored-only: cursor menus must not
  // steal focus from the host's focus-scoped shortcuts (they still rove via Arrow/Home/End).
  $effect(() => {
    if (!open || !ready || !menuEl || !anchorRect) return;
    if (menuEl.contains(document.activeElement)) return;
    focusItemAt(0);
  });

  function activate(item: MenuItem): void {
    if (item.disabled) return;
    item.onclick();
    onclose();
  }
</script>

{#if open}
  <!-- `p-1` gutter so each hovered item is an inset pill that never clips the card's outer curve. -->
  <div
    bind:this={menuEl}
    role="menu"
    aria-orientation="vertical"
    style="position: fixed; left: {computedX}px; top: {computedY}px; visibility: {ready
      ? 'visible'
      : 'hidden'};"
    class="z-50 min-w-48 rounded-md border border-line bg-elevated p-1 shadow-popover"
    transition:fade={{ duration: 100 }}
  >
    {#each sections as section, i (i)}
      {#if i > 0}
        <!-- No `-mx-1` full-bleed: keep the gutter so the separator inset matches the hover pills. -->
        <div class="my-1 h-px bg-line-subtle" role="separator"></div>
      {/if}
      {#each section.items as item, j (j)}
        <button
          type="button"
          role="menuitem"
          tabindex="-1"
          disabled={item.disabled}
          onclick={() => activate(item)}
          class="flex w-full items-center justify-between gap-4 rounded-sm px-2 py-1.5 text-left text-xs transition outline-none disabled:cursor-not-allowed disabled:text-fg-subtle {item.variant ===
          'destructive'
            ? 'text-danger-soft-fg hover:bg-danger-soft focus-visible:bg-danger-soft disabled:hover:bg-transparent'
            : 'text-fg-secondary hover:bg-surface-2 focus-visible:bg-surface-2 disabled:hover:bg-transparent'}"
        >
          <span>{item.label}</span>
          {#if item.hint}
            <span class="font-mono text-[10px] text-fg-subtle">{item.hint}</span>
          {/if}
        </button>
      {/each}
    {/each}
  </div>
{/if}
