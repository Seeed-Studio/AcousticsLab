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
  stroke-width="2"
  stroke-linecap="round"
  stroke-linejoin="round"
  class={extraClass}
  aria-hidden="true"
>
  <circle cx="12" cy="12" r="4" />
  <path d="M12 2v2" />
  <path d="M12 20v2" />
  <path d="M4.93 4.93l1.41 1.41" />
  <path d="M17.66 17.66l1.41 1.41" />
  <path d="M2 12h2" />
  <path d="M20 12h2" />
  <path d="M6.34 17.66l-1.41 1.41" />
  <path d="M19.07 4.93l-1.41 1.41" />
</svg>
