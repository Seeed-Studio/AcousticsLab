<script lang="ts">
  // Stroke-only (no fill) to match the sibling action icons; the `i` is two strokes rather than a glyph path to stay crisp at small sizes.
  interface Props {
    class?: string;
  }
  let { class: extraClass = 'h-3.5 w-3.5' }: Props = $props();
</script>

<svg
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  class={extraClass}
  aria-hidden="true"
>
  <circle cx="12" cy="12" r="9" />
  <path d="M12 11v5" />
  <path d="M12 8h.01" />
</svg>
