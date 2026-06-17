<script lang="ts">
  // size intent: 'page' centers in an empty route body, 'section' a tight band under an
  // already-rendered header, 'fill' occupies the eventual content's footprint. Spinner is
  // aria-hidden so role=status announces the bare label, not "graphic loading".
  import Spinner from './Spinner.svelte';

  interface Props {
    label: string;
    size?: 'page' | 'section' | 'fill';
  }
  let { label, size = 'page' }: Props = $props();

  const sizeClass = $derived(
    size === 'page' ? 'min-h-[60vh]' : size === 'section' ? 'py-6' : 'flex-1'
  );
</script>

<div
  role="status"
  aria-live="polite"
  class="flex items-center justify-center gap-2 text-xs text-fg-muted {sizeClass}"
>
  <Spinner />
  <span>{label}</span>
</div>
