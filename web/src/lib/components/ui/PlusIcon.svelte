<script lang="ts">
  interface Props {
    class?: string;
  }
  let { class: extraClass = 'h-3.5 w-3.5' }: Props = $props();
</script>

<svg
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="2.5"
  stroke-linecap="round"
  stroke-linejoin="round"
  class={extraClass}
  aria-hidden="true"
>
  <path d="M12 5v14M5 12h14" />
</svg>
