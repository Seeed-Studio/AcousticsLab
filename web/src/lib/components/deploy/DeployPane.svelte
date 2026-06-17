<script lang="ts">
  import { fade } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import { config as configStore } from '$lib/stores/config.svelte';
  import HeadsTable from './HeadsTable.svelte';
  import InferencePreview from './InferencePreview.svelte';
  import ConfigurationControls from '$lib/components/dashboard/ConfigurationControls.svelte';
  import StatusBadge, { type Tone } from '$lib/components/ui/StatusBadge.svelte';
  import { m } from '$lib/i18n';
  import type { HeadRecord, Uuid } from '$lib/api/types';

  interface Props {
    workspaceId: Uuid;
    /// Only seeds the alpkg export filename slug in the heads list; nothing else here reads it.
    workspaceName: string;
    heads: readonly HeadRecord[];
    liveRevision: number;
    onchanged: () => Promise<void> | void;
  }
  let { workspaceId, workspaceName, heads, liveRevision, onchanged }: Props = $props();

  const active = $derived(configStore.active);

  // 'elsewhere' -> 'standby' names this workspace's role (ready, not active), not the runtime's source.
  // 'detached' = source workspace deleted (orphaned runtime); 'unknown' = config.active not yet
  // landed (first-load / unreachable race), pill suppressed.
  type DeployState = 'workspace-deployed' | 'default' | 'elsewhere' | 'detached' | 'unknown';

  const deployState = $derived.by<DeployState>(() => {
    const a = active;
    if (a === null) return 'unknown';
    if (a.origin === 'default') return 'default';
    if (a.source_workspace_alive === false) return 'detached';
    return a.source_workspace_id === workspaceId ? 'workspace-deployed' : 'elsewhere';
  });

  const pillCopy = $derived<{ label: string; title: string; tone: Tone } | null>(
    deployState === 'workspace-deployed'
      ? {
          label: m.deploy.pane.pill_deployed,
          title: m.deploy.pane.pill_deployed_title,
          tone: 'info'
        }
      : deployState === 'default'
        ? {
            label: m.deploy.pane.pill_default,
            title: m.deploy.pane.pill_default_title,
            tone: 'neutral'
          }
        : deployState === 'elsewhere'
          ? {
              label: m.deploy.pane.pill_standby,
              title: m.deploy.pane.pill_standby_title,
              tone: 'warning'
            }
          : deployState === 'detached'
            ? {
                label: m.deploy.pane.pill_detached,
                title: m.deploy.pane.pill_detached_title,
                tone: 'warning'
              }
            : null
  );

  // Closed by default: the mainline action here is deploy-a-head, not tune-cadence.
  let configOpen = $state(false);

  // Daemon's fixed capture rate, duplicated (not imported) to avoid a cross-component dep for one
  // number; must stay equal to the cadence slider readout that renders the same Hz value.
  const CAPTURE_SAMPLE_RATE = 44_100;

  const configChips = $derived.by<string[]>(() => {
    const inf = configStore.inference;
    if (!inf) return [];
    const hz = CAPTURE_SAMPLE_RATE / Math.max(1, inf.hop_samples);
    const hzStr = hz >= 10 ? hz.toFixed(0) : hz >= 1 ? hz.toFixed(1) : hz.toFixed(2);
    return [m.deploy.pane.config_chip_freq(hzStr), m.deploy.pane.config_chip_top_k(inf.top_k)];
  });
</script>

<section class="rounded-xl border border-line bg-surface px-5 pt-3.5 pb-5 shadow-card">
  <!-- Badge wrapper must be `inline-flex`, not a bare `<span>`: a span's baseline strut in this
       flex context sinks the badge ~2px below centre. `-translate-y-px` is a 1px optical correction
       done as a transform so flow still measures the wrapper at its geometric centre. -->
  <header class="mb-3 flex items-center justify-between gap-3">
    <div class="min-w-0">
      <h2 class="text-sm font-semibold text-fg">{m.deploy.pane.heading}</h2>
      <p class="mt-0.5 text-xs text-fg-muted">{m.deploy.pane.description}</p>
    </div>
    {#if pillCopy}
      <span class="inline-flex shrink-0 -translate-y-px">
        <StatusBadge label={pillCopy.label} tone={pillCopy.tone} title={pillCopy.title} />
      </span>
    {/if}
  </header>

  <!-- Both cells pin to `h-80` so starting/stopping the preview or a long heads list never reflows
       the row: the heads pane scrolls internally instead of pushing the disclosure down. -->
  <div class="mb-3 grid grid-cols-1 gap-3 lg:grid-cols-5">
    <!-- 3/5 to the heads list so its trailing Deploy/Delete buttons don't wrap; 2/5 to preview. -->
    <div class="h-80 lg:col-span-3">
      <!-- Key on workspaceId: same-route nav reuses this page component, so without keying,
           HeadsTable's internal $state (busy/deploy/delete flags) would leak between workspaces. -->
      {#key workspaceId}
        <HeadsTable {workspaceId} {workspaceName} {heads} {liveRevision} {onchanged} />
      {/key}
    </div>
    <div class="h-80 lg:col-span-2">
      <InferencePreview />
    </div>
  </div>

  <!-- grid-rows 0fr/1fr animation keeps the body mounted across open/close so form state survives. -->
  <div class="rounded-md border border-line bg-surface-2/60">
    <button
      type="button"
      onclick={() => (configOpen = !configOpen)}
      aria-expanded={configOpen}
      aria-controls="deploy-config-panel"
      class="flex w-full items-center justify-between gap-3 px-3 py-2 text-left transition hover:bg-surface-2"
    >
      <span class="flex min-w-0 items-center gap-2">
        <svg
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
          class="h-3.5 w-3.5 shrink-0 text-fg-muted transition-transform duration-200"
          class:translate-y-px={!configOpen}
          class:rotate-90={configOpen}
        >
          <path
            fill-rule="evenodd"
            d="M7.21 5.23a.75.75 0 011.06.02L12 9l-3.73 3.71a.75.75 0 11-1.06-1.06L9.94 9 7.19 6.29a.75.75 0 01.02-1.06z"
            clip-rule="evenodd"
          />
        </svg>
        <span class="text-xs font-medium text-fg-secondary"
          >{m.deploy.pane.config_disclosure_label}</span
        >
      </span>
      {#if !configOpen}
        <span
          in:fade={{ duration: 180, easing: cubicOut }}
          class="hidden shrink-0 flex-wrap items-center justify-end gap-1 sm:flex"
          aria-hidden="true"
        >
          {#each configChips as chip (chip)}
            <span
              class="inline-flex items-center rounded-full bg-surface px-1.5 py-0.5 font-mono text-[10px] text-fg-secondary ring-1 ring-line"
            >
              {chip}
            </span>
          {/each}
        </span>
      {/if}
    </button>
    <div
      id="deploy-config-panel"
      class="grid transition-[grid-template-rows] duration-200 ease-out"
      class:grid-rows-[1fr]={configOpen}
      class:grid-rows-[0fr]={!configOpen}
    >
      <div class="min-h-0 overflow-hidden" inert={!configOpen} aria-hidden={!configOpen}>
        <div class="border-t border-line bg-surface px-4 py-4">
          <ConfigurationControls />
        </div>
      </div>
    </div>
  </div>
</section>
