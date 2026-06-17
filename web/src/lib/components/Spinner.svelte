<script lang="ts">
  // Default sized for inline status use next to headings; callers override `class`.
  interface Props {
    class?: string;
  }
  let { class: cls = 'h-3 w-3 text-accent' }: Props = $props();
</script>

<svg
  class="animate-spin {cls}"
  xmlns="http://www.w3.org/2000/svg"
  fill="none"
  viewBox="0 0 24 24"
  aria-hidden="true"
>
  <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="3" class="opacity-25"></circle>
  <path d="M21 12a9 9 0 00-9-9" stroke="currentColor" stroke-width="3" stroke-linecap="round"
  ></path>
</svg>
