<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    description?: string;
    action?: Snippet;
    icon?: Snippet;
  }
  let { title, description, action, icon }: Props = $props();
</script>

<div
  class="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-line-strong bg-surface-2 px-6 py-12 text-center"
>
  {#if icon}
    <div class="text-fg-subtle">
      {@render icon()}
    </div>
  {/if}
  <h3 class="text-sm font-semibold text-fg">{title}</h3>
  {#if description}
    <p class="max-w-sm text-xs text-fg-muted">{description}</p>
  {/if}
  {#if action}
    <div class="mt-1">
      {@render action()}
    </div>
  {/if}
</div>
