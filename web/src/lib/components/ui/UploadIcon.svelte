<script lang="ts">
  // Vertical mirror of DownloadIcon so the load/save pair reads as matched icons.
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
  <path d="M7 9l5-5 5 5" />
  <path d="M5 20h14" />
</svg>
