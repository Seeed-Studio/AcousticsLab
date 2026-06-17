<script lang="ts">
  import { config } from '$lib/stores/config.svelte';
  import ConfigurationControls from './ConfigurationControls.svelte';
  import { m } from '$lib/i18n';

  // Form body lives in ConfigurationControls so the deploy disclosure can reuse it without nesting another card.
  let unavailable = $derived<boolean>(
    (config.mic === null || config.inference === null) && config.error !== null
  );
</script>

<section class="rounded-xl border border-line bg-surface px-5 pt-3.5 pb-5 shadow-card">
  <header class="mb-3 flex items-baseline justify-between">
    <h2 class="text-sm font-semibold text-fg">{m.dashboard.configuration_panel.heading}</h2>
    {#if config.error && !unavailable}
      <span class="truncate text-xs text-danger-soft-fg">{config.error}</span>
    {/if}
  </header>

  <ConfigurationControls />
</section>
