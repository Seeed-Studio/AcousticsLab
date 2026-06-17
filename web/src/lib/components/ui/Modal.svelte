<script lang="ts">
  import type { Snippet } from 'svelte';
  import { randomHex } from '$lib/utils/random';

  interface Props {
    open: boolean;
    title?: string;
    headerRight?: Snippet;
    // Fires for Escape, backdrop click, and form[method=dialog] submit; owner must set open=false.
    onclose?: () => void;
    // false where accidental backdrop dismissal is costly; Escape/Cancel still close.
    closeOnBackdrop?: boolean;
    children?: Snippet;
    footer?: Snippet;
    // a11y label when no `title` (e.g. a custom header in the body snippet).
    ariaLabel?: string;
    class?: string;
  }
  let {
    open,
    title,
    headerRight,
    onclose,
    closeOnBackdrop = true,
    children,
    footer,
    ariaLabel,
    class: sizeClass = 'max-w-md'
  }: Props = $props();

  let dialogEl = $state<HTMLDialogElement | undefined>();

  // showModal() (not show()) for focus trap + backdrop; open-state guard avoids InvalidStateError from re-calling showModal() while open.
  $effect(() => {
    const d = dialogEl;
    if (!d) return;
    if (open && !d.open) {
      d.showModal();
    } else if (!open && d.open) {
      d.close();
    }
  });

  function onBackdropClick(e: MouseEvent): void {
    if (!closeOnBackdrop || !dialogEl) return;
    // target check rejects descendant clicks (incl. fixed children outside the rect a rect-only test would misread as backdrop); rect check then drops padding/gap clicks, which also report target===dialogEl.
    if (e.target !== dialogEl) return;
    const rect = dialogEl.getBoundingClientRect();
    const inside =
      e.clientX >= rect.left &&
      e.clientX <= rect.right &&
      e.clientY >= rect.top &&
      e.clientY <= rect.bottom;
    if (!inside) onclose?.();
  }

  function onNativeClose(): void {
    if (open) onclose?.();
  }

  // const not $derived: re-rolling each tick would break the aria-labelledby <-> <h2 id> binding. randomHex not crypto.randomUUID to survive insecure-context origins (runs even for open={false}).
  const titleId = `modal-title-${randomHex(4)}`;
</script>

<!-- Layout utils gated on `open:` so they don't override UA `dialog:not([open]){display:none}`. Width calc (not `w-full`) leaves an edge gap so it reads as a card, not fullscreen, when sizeClass exceeds the viewport. -->
<!-- Scroll lives in the inner child, not the dialog: Safari's overscroll bounce paints OVER the dialog bg and would flash the backdrop; a scrolling child keeps the frame behind the bounce. `overscroll-contain` stops chain leak; dialog `overflow-hidden` clips to rounded corners. -->
<dialog
  bind:this={dialogEl}
  onclick={onBackdropClick}
  onclose={onNativeClose}
  aria-labelledby={title ? titleId : undefined}
  aria-label={title ? undefined : ariaLabel}
  class="m-auto w-[calc(100%-var(--popover-edge-inset)*2)] rounded-xl border border-line bg-elevated shadow-modal backdrop:bg-scrim backdrop:backdrop-blur-[2px] open:flex open:max-h-[90vh] open:flex-col open:overflow-hidden {sizeClass}"
>
  <!-- `flex-1` fills the parent's available height (capped at `max-h-[90vh]`); `min-h-0` overrides the flex default `min-height:auto` so this child can shrink below content, giving `overflow-auto` a bounded height to scroll against - else content overflows the rounded corners past `max-h-[90vh]`. Padding sits inside the scroll box. -->
  <div class="flex min-h-0 flex-1 flex-col gap-3 overflow-auto overscroll-contain p-5">
    <!-- Truthy `||` not `??`: `??` keeps `""` (non-nullish), hiding a headerRight-only header when title is the empty string. -->
    <!-- eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing -->
    {#if title || headerRight}
      <header
        class="flex items-center gap-2 {title && headerRight
          ? 'justify-between'
          : title
            ? 'justify-start'
            : 'justify-end'}"
      >
        {#if title}
          <h2 id={titleId} class="text-sm font-semibold text-fg">{title}</h2>
        {/if}
        {#if headerRight}
          {@render headerRight()}
        {/if}
      </header>
    {/if}
    <div class="flex flex-col gap-3 text-sm text-fg-secondary">
      {@render children?.()}
    </div>
    {#if footer}
      <footer class="mt-1 flex justify-end gap-2">
        {@render footer()}
      </footer>
    {/if}
  </div>
</dialog>
