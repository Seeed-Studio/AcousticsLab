<script lang="ts">
  import { fade } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';

  // Label changes morph not switch: keyed span crossfades in one grid cell while width glides to
  // a JS-measured target. Measuring up front avoids the snap grid-cell max(out,in) sizing causes.

  type Size = 'xs' | 'sm';

  // Each tone needs a TONE_CLASSES entry legible in both light and dark.
  export type Tone = 'info' | 'accent' | 'success' | 'warning' | 'danger' | 'neutral';

  interface Props {
    /** Drives the crossfade + width glide on change. */
    label: string;
    tone: Tone;
    title?: string;
    size?: Size;
  }
  let { label, tone, title, size = 'sm' }: Props = $props();

  const TONE_CLASSES: Readonly<Record<Tone, string>> = {
    info: 'bg-accent-soft text-accent-soft-fg',
    // accent-tint: faintest fill still readable on an accent-soft row (where soft info dissolves).
    accent: 'bg-accent-tint text-accent-soft-fg',
    success: 'bg-success-soft text-success-soft-fg',
    warning: 'bg-warning-soft text-warning-soft-fg',
    danger: 'bg-danger-soft text-danger-soft-fg',
    neutral: 'bg-surface-2 text-fg-secondary'
  };
  const toneCls = $derived(TONE_CLASSES[tone]);

  const TEXT_SIZE: Readonly<Record<Size, string>> = {
    xs: 'text-[10px]',
    sm: 'text-[11px]'
  };
  // Size-constant so the text-to-edge gap holds at every font size.
  const PAD = 'px-2 py-0.5';

  // INVARIANT: typography only, no PAD; shared by wrapper and mirror. PAD here would widen the
  // mirror by +PAD/side, the padding-free wrapper centers that slack, and the pill's own PAD then
  // doubles the apparent text-to-edge gap.
  const TEXT_CLS = $derived(
    `${TEXT_SIZE[size]} font-medium capitalize tracking-wide whitespace-nowrap`
  );

  let measureEl: HTMLSpanElement | undefined = $state();
  // Float (not rounded) for sub-pixel-smooth width transition on Retina.
  let textWidth: number | null = $state(null);

  // Re-measures on label/size change (tone is typography-neutral). Inside a display:none ancestor
  // (e.g. unopened <dialog>) getBoundingClientRect() returns 0; textWidth=0 would collapse+clip
  // the pill, so skip w===0 (auto fallback) and let ResizeObserver re-measure when laid out.
  $effect(() => {
    void label;
    void size;
    if (!measureEl) return;
    const el = measureEl;
    const w = el.getBoundingClientRect().width;
    if (w > 0) {
      textWidth = w;
      return;
    }
    const observer = new ResizeObserver(() => {
      const w2 = el.getBoundingClientRect().width;
      if (w2 > 0) {
        textWidth = w2;
        observer.disconnect();
      }
    });
    observer.observe(el);
    return (): void => observer.disconnect();
  });
</script>

<span
  in:fade={{ duration: 180, easing: cubicOut }}
  out:fade={{ duration: 140, easing: cubicOut }}
  class="inline-flex items-center justify-center overflow-hidden rounded-full transition-[background-color,color] duration-200 ease-out {TEXT_CLS} {PAD} {toneCls}"
  {title}
>
  <!-- Width wrapper CSS-interpolates to the measured width; pre-effect auto == first measured, so no snap. -->
  <span
    class="inline-flex items-center justify-center overflow-hidden transition-[width] duration-200 ease-out"
    style:width={textWidth !== null ? `${textWidth}px` : 'auto'}
  >
    <span class="inline-grid grid-cols-1 grid-rows-1 items-center">
      {#key label}
        <span
          in:fade={{ duration: 180, easing: cubicOut }}
          out:fade={{ duration: 180, easing: cubicOut }}
          class="col-start-1 row-start-1"
        >
          {label}
        </span>
      {/key}
    </span>
  </span>
</span>

<!-- Off-screen mirror: visible-text typography for accurate width, no PAD (TEXT_CLS invariant). -->
<span
  bind:this={measureEl}
  aria-hidden="true"
  class="pointer-events-none invisible fixed top-0 left-0 {TEXT_CLS}"
>
  {label}
</span>
