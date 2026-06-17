<script lang="ts">
  import { fade } from 'svelte/transition';
  import { config } from '$lib/stores/config.svelte';
  import type { MicPolicy } from '$lib/api/types';
  import Spinner from '$lib/components/Spinner.svelte';
  import { m } from '$lib/i18n';

  // Mic + inference-cadence form body shared by the dashboard config panel and the deploy
  // pane; owns apply-floor and optimistic-revert. The outer section belongs to the host.

  // Daemon hop_samples contract is `SR*(1 - MAX_OVERLAP_RATIO)..=SR` (11_025..=44_100 at 44.1 kHz);
  // the slider exposes the dimension-free `overlap = hop / SR` (0.25..1.0), cadence = 1/overlap.
  const CAPTURE_SAMPLE_RATE = 44_100;
  const OVERLAP_MIN = 0.25;
  const OVERLAP_MAX = 1.0;
  const TOPK_MIN = 1;
  const TOPK_MAX = 20;

  function overlapToHop(o: number): number {
    return Math.round(o * CAPTURE_SAMPLE_RATE);
  }
  function approxHz(o: number): string {
    const hz = 1 / Math.max(0.001, o);
    const hzStr = hz >= 10 ? hz.toFixed(0) : hz >= 1 ? hz.toFixed(1) : hz.toFixed(2);
    return m.dashboard.configuration_controls.approx_hz(hzStr);
  }

  function formatRate(rate: unknown): string | null {
    if (typeof rate !== 'number' || !Number.isFinite(rate) || rate <= 0) return null;
    if (rate >= 1000) {
      const khz = rate / 1000;
      const khzStr = Number.isInteger(khz) ? khz.toFixed(0) : khz.toFixed(1);
      return m.dashboard.configuration_controls.khz(khzStr);
    }
    return m.dashboard.configuration_controls.hz(rate);
  }

  function formatSourceKind(kind: string): string {
    if (kind === 'alsa') return m.dashboard.configuration_controls.kind_alsa;
    if (!kind) return m.dashboard.configuration_controls.kind_unknown;
    return kind[0].toUpperCase() + kind.slice(1).replaceAll('_', ' ');
  }

  function sourceLabel(cand: (typeof candidates)[number]): string {
    const kind = formatSourceKind(cand.source.kind);
    const detail =
      cand.source.kind === 'alsa' ? cand.source.hw_spec : formatRate(cand.source.sample_rate);
    return [cand.id, kind, detail].filter(Boolean).join(' · ');
  }

  // 'auto' (= first_available/auto) or a candidate id / stringified channel; flattening the
  // policy's two-field shape into one dropdown each keeps the form stable (no rows on toggle).
  let sourceSel = $state<string>('auto');
  let channelSel = $state<string>('auto');

  // Daemon defaults; the config-sync effect overwrites with canonical server values.
  let overlap = $state(1.0);
  let topK = $state(20);

  // Disables controls (blocks a rapid re-fire racing the in-flight request) and shows a
  // heading spinner; cleared after MIN_APPLY_MS at earliest.
  let micApplying = $state(false);
  let inferApplying = $state(false);

  // Sync form from canonical config (initial load, reconnect refresh, successful apply); a
  // no-op on apply failure since config.mic is unchanged, and the failed apply self-reverts.
  $effect(() => {
    // Named `mic`, not `m`, to avoid shadowing the i18n `m` proxy.
    const mic = config.mic;
    if (!mic) return;
    sourceSel = mic.policy.mic.kind === 'fixed' ? mic.policy.mic.id : 'auto';
    channelSel = mic.policy.channel.kind === 'fixed' ? String(mic.policy.channel.channel) : 'auto';
  });

  $effect(() => {
    const c = config.inference;
    if (!c) return;
    overlap = c.hop_samples / CAPTURE_SAMPLE_RATE;
    topK = c.top_k;
  });

  let candidates = $derived(config.mic?.catalogue.candidates ?? []);

  let channelOptions = $derived.by(() => {
    const mic = config.mic;
    if (!mic) return [] as number[];
    const targetId = sourceSel === 'auto' ? (mic.catalogue.candidates[0]?.id ?? '') : sourceSel;
    const cand = mic.catalogue.candidates.find((c) => c.id === targetId);
    return cand?.channels ?? [];
  });

  // Fill fraction (0..100) fed to the `--slider-percent` CSS var gradienting the range track.
  let overlapPct = $derived(((overlap - OVERLAP_MIN) / (OVERLAP_MAX - OVERLAP_MIN)) * 100);
  let topKPct = $derived(((topK - TOPK_MIN) / (TOPK_MAX - TOPK_MIN)) * 100);

  // Failed first load with empty config: show "daemon unavailable" rather than an indefinite
  // "loading"; recovery comes from the auto-reconnect retrying config.refresh on health.
  let unavailable = $derived<boolean>(
    (config.mic === null || config.inference === null) && config.error !== null
  );

  // Snap the form back to canonical on apply failure; else the value lingers on the failed
  // pick, confusing cause/effect on retry.
  function revertMic(): void {
    const mic = config.mic;
    if (!mic) return;
    sourceSel = mic.policy.mic.kind === 'fixed' ? mic.policy.mic.id : 'auto';
    channelSel = mic.policy.channel.kind === 'fixed' ? String(mic.policy.channel.channel) : 'auto';
  }
  function revertInference(): void {
    const c = config.inference;
    if (!c) return;
    overlap = c.hop_samples / CAPTURE_SAMPLE_RATE;
    topK = c.top_k;
  }

  // Floor on the "applying" state: localhost round-trips (30-100 ms) finish before transitions
  // play out, so without it the user sees sub-perceptual flicker (transition reverses early).
  const MIN_APPLY_MS = 420;

  async function applyWithFloor(fn: () => Promise<void>): Promise<void> {
    const start = performance.now();
    try {
      await fn();
    } finally {
      const elapsed = performance.now() - start;
      if (elapsed < MIN_APPLY_MS) {
        await new Promise((r) => setTimeout(r, MIN_APPLY_MS - elapsed));
      }
    }
  }

  async function autoApplyMic(): Promise<void> {
    if (micApplying) return;
    // `channelSel` isn't reset on `sourceSel` change and `onchange` fires before binding
    // reconciliation, so guard against POSTing a channel index the new device lacks (e.g.
    // stereo ch 1 -> mono): snap to 'auto' when absent from the now-current channelOptions.
    if (channelSel !== 'auto' && !channelOptions.includes(Number(channelSel))) {
      channelSel = 'auto';
    }
    micApplying = true;
    try {
      await applyWithFloor(async () => {
        const policy: MicPolicy = {
          mic:
            sourceSel === 'auto' ? { kind: 'first_available' } : { kind: 'fixed', id: sourceSel },
          channel:
            channelSel === 'auto'
              ? { kind: 'auto' }
              : { kind: 'fixed', channel: Number(channelSel) }
        };
        await config.setMicPolicy(policy);
      });
    } catch {
      revertMic();
    } finally {
      micApplying = false;
    }
  }

  async function autoApplyInference(): Promise<void> {
    if (inferApplying) return;
    inferApplying = true;
    try {
      await applyWithFloor(async () => {
        await config.setInferenceCfg({ hop_samples: overlapToHop(overlap), top_k: topK });
      });
    } catch {
      revertInference();
    } finally {
      inferApplying = false;
    }
  }

  const selectCls =
    'select-chevron block w-full rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs text-fg transition-colors hover:border-line-strong disabled:cursor-wait disabled:bg-page disabled:text-fg-subtle disabled:hover:border-line';
</script>

{#if unavailable}
  <div
    class="flex items-center gap-3 rounded-lg border border-warning-line bg-warning-soft px-4 py-3"
  >
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2.5"
      class="h-4 w-4 shrink-0 animate-spin text-warning-soft-fg"
      aria-hidden="true"
    >
      <path d="M12 3a9 9 0 109 9" stroke-linecap="round" />
    </svg>
    <div class="min-w-0 text-xs">
      <p class="font-medium text-warning-soft-fg">
        {m.dashboard.configuration_controls.daemon_unavailable_title}
      </p>
      <p class="mt-0.5 truncate text-warning-soft-fg">
        {config.error ?? m.dashboard.configuration_controls.daemon_unavailable_default}
      </p>
    </div>
  </div>
{:else}
  <div class="grid grid-cols-1 gap-x-10 gap-y-6 md:grid-cols-2">
    <div class="flex flex-col">
      <h3
        class="mb-3 flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-fg-muted uppercase"
      >
        <span>{m.dashboard.configuration_controls.microphone_heading}</span>
        {#if micApplying}
          <span class="inline-flex" in:fade={{ duration: 160 }} out:fade={{ duration: 120 }}>
            <Spinner />
          </span>
        {/if}
      </h3>

      {#if !config.mic}
        <p class="text-xs text-fg-subtle">{m.dashboard.configuration_controls.loading}</p>
      {:else}
        <div
          class="space-y-3 transition-opacity duration-150 ease-out"
          class:opacity-60={micApplying}
        >
          <label for="mic-source" class="block text-xs">
            <span class="mb-1 block text-fg-secondary"
              >{m.dashboard.configuration_controls.source_label}</span
            >
            <select
              id="mic-source"
              name="mic-source"
              bind:value={sourceSel}
              onchange={autoApplyMic}
              disabled={micApplying}
              class={selectCls}
            >
              <option value="auto">{m.dashboard.configuration_controls.auto_first_available}</option
              >
              {#each candidates as cand (cand.id)}
                <option value={cand.id}>{sourceLabel(cand)}</option>
              {/each}
            </select>
          </label>

          <label for="mic-channel" class="block text-xs">
            <span class="mb-1 block text-fg-secondary"
              >{m.dashboard.configuration_controls.channel_label}</span
            >
            <select
              id="mic-channel"
              name="mic-channel"
              bind:value={channelSel}
              onchange={autoApplyMic}
              disabled={micApplying}
              class={selectCls}
            >
              <option value="auto">{m.dashboard.configuration_controls.auto_channel}</option>
              {#each channelOptions as ch (ch)}
                <option value={String(ch)}>{ch}</option>
              {/each}
            </select>
          </label>
        </div>
      {/if}
    </div>

    <div class="flex flex-col">
      <h3
        class="mb-3 flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-fg-muted uppercase"
      >
        <span>{m.dashboard.configuration_controls.inference_cadence_heading}</span>
        {#if inferApplying}
          <span class="inline-flex" in:fade={{ duration: 160 }} out:fade={{ duration: 120 }}>
            <Spinner />
          </span>
        {/if}
      </h3>

      {#if !config.inference}
        <p class="text-xs text-fg-subtle">{m.dashboard.configuration_controls.loading}</p>
      {:else}
        <div
          class="space-y-3 transition-opacity duration-150 ease-out"
          class:opacity-60={inferApplying}
        >
          <div class="flex flex-col gap-1">
            <div class="flex items-center justify-between">
              <label for="overlap-ratio" class="text-xs text-fg-secondary"
                >{m.dashboard.configuration_controls.overlap_ratio_label}</label
              >
              <span class="text-[11px] leading-4 text-fg-muted">
                <span class="font-mono text-fg-secondary">{overlap.toFixed(2)}</span>
                <span class="text-fg-muted">· ≈ {approxHz(overlap)}</span>
              </span>
            </div>
            <input
              id="overlap-ratio"
              type="range"
              min={OVERLAP_MIN}
              max={OVERLAP_MAX}
              step="0.01"
              bind:value={overlap}
              onchange={autoApplyInference}
              disabled={inferApplying}
              style="--slider-percent: {overlapPct}%"
            />
          </div>

          <div class="-mb-1.75 flex flex-col gap-1 md:mb-0">
            <div class="flex items-center justify-between">
              <label for="top-k" class="text-xs text-fg-secondary"
                >{m.dashboard.configuration_controls.top_k_label}</label
              >
              <span class="font-mono text-[11px] leading-4 text-fg-secondary">{topK}</span>
            </div>
            <input
              id="top-k"
              type="range"
              min={TOPK_MIN}
              max={TOPK_MAX}
              step="1"
              bind:value={topK}
              onchange={autoApplyInference}
              disabled={inferApplying}
              style="--slider-percent: {topKPct}%"
            />
          </div>
        </div>
      {/if}
    </div>
  </div>
{/if}
