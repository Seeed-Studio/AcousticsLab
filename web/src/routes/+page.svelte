<script lang="ts">
  import { config } from '$lib/stores/config.svelte';
  import { streams } from '$lib/stores/streams.svelte';
  import { m } from '$lib/i18n';
  import VisualizationPanel from '$lib/components/dashboard/VisualizationPanel.svelte';
  import InferencePanel from '$lib/components/dashboard/InferencePanel.svelte';
  import ConfigurationPanel from '$lib/components/dashboard/ConfigurationPanel.svelte';

  // Revalidate the active record on entry (another tab / the CLI can detach it between visits);
  // the guard skips the duplicate GET while the layout's bootstrap refresh is still in flight,
  // which is always the case on a fresh page load (parent script init runs first).
  if (!config.loading) void config.refreshActive();

  // Acquire page-level (not per-panel) to keep the worker alive across panels' independent
  // mount cycles; acquire() returns a dispose closure that Svelte runs as cleanup on route exit.
  // `$effect.pre` not `$effect`: children read stream status synchronously on first render, so
  // acquiring after mount flashes one "disconnected" frame before the optimistic 'connecting'
  // write lands; pre-mount makes it already 'connecting'.
  $effect.pre(() => streams.acquire());
</script>

<svelte:head>
  <title>{m.routes.dashboard_title(m.app.name)}</title>
</svelte:head>

{#if streams.unsupportedReason}
  <div
    class="mb-4 rounded-lg border border-warning-line bg-warning-soft px-4 py-3 text-sm text-warning-soft-fg"
  >
    <p class="font-medium">{m.dashboard.limited_support_title}</p>
    <p class="mt-1 text-xs">{streams.unsupportedReason}</p>
  </div>
{/if}

<div class="space-y-5">
  <div class="grid grid-cols-1 gap-5 lg:grid-cols-3">
    <div class="lg:col-span-2">
      <VisualizationPanel />
    </div>
    <div class="lg:col-span-1">
      <InferencePanel />
    </div>
  </div>

  <ConfigurationPanel />
</div>
