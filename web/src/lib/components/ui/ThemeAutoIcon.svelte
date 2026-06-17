<script lang="ts">
  // Half-filled circle = 'auto' theme (half light, half dark); left-half fill <path> must precede the outline circle so the stroke paints over the seam.
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
  <path d="M12 3 A9 9 0 0 0 12 21 Z" fill="currentColor" stroke="none" />
  <circle cx="12" cy="12" r="9" />
</svg>
