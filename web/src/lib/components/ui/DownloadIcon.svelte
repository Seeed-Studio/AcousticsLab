<script lang="ts">
  // Outline download glyph; keep the viewBox/stroke geometry matched to the sibling action
  // icons (UploadIcon is a vertical mirror of this path) so the cluster reads as one family.
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
  <path d="M12 4v12" />
  <path d="M7 11l5 5 5-5" />
  <path d="M5 20h14" />
</svg>
