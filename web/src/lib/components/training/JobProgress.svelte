<script lang="ts">
  import MetricsChart from './MetricsChart.svelte';
  import RunSummary from './RunSummary.svelte';
  import TrainLogs from './TrainLogs.svelte';
  import { stageLabel } from './labels';
  import { m } from '$lib/i18n';
  import type { TrackedTrainingJob } from '$lib/stores/training.svelte';
  import type { Stage, TrainingJobView } from '$lib/api/types';

  interface Props {
    // Parent gates on a live job slot, so this is non-null when rendered.
    job: TrackedTrainingJob;
  }
  let { job }: Props = $props();

  // null until the first poll lands; surfaces below tolerate that.
  const view = $derived<TrainingJobView | null>(job.view);
  const phase = $derived<Stage>(view?.progress.phase ?? 'prepare');

  // Catalog keyed by the closed Stage union, so a future out-of-union phase indexes to undefined, not a raw-string fallback.
  const phaseLabel = $derived<string>(stageLabel(phase));

  // Freshest metrics, else last epoch: history is monotonic so falling back is safe when a non-Train tick (e.g. Save) arrives without metrics.
  const latestMetrics = $derived(
    view?.progress.metrics ?? (job.epochs.length > 0 ? job.epochs[job.epochs.length - 1] : null)
  );

  // `validation_split = 0` makes val_acc NaN; detected from metrics since the split isn't echoed back on the view.
  const valDisabled = $derived(
    latestMetrics !== null &&
      (latestMetrics.val_acc === null || !Number.isFinite(latestMetrics.val_acc))
  );

  const isTerminal = $derived(
    view !== null &&
      (view.state === 'completed' || view.state === 'failed' || view.state === 'cancelled')
  );
  // null fill triggers the indeterminate animation: Prepare/Dataset-scan emit total:0 until the example count is known, so a running run with no total animates. Terminal never returns null (completed=100%, else last fill or 0).
  const progressPct = $derived.by(() => {
    const total = view?.progress.total ?? 0;
    const current = view?.progress.current ?? 0;
    if (isTerminal && view?.state === 'completed') return 100;
    if (isTerminal) return total > 0 ? Math.max(0, Math.min(100, (current / total) * 100)) : 0;
    if (total <= 0) return null;
    return Math.max(0, Math.min(100, (current / total) * 100));
  });

  // Non-finite reads as an em-dash so the readout never shows "NaN"/"undefined".
  function fmtLoss(v: number | undefined | null): string {
    if (v === undefined || v === null || !Number.isFinite(v)) return m.training.progress.em_dash;
    return v.toFixed(3);
  }
  function fmtAcc(v: number | undefined | null): string {
    if (v === undefined || v === null || !Number.isFinite(v)) return m.training.progress.em_dash;
    return (v * 100).toFixed(1) + '%';
  }
</script>

<div class="flex flex-col gap-2">
  {#if isTerminal && view !== null}
    <!-- Terminal: no bar (frozen 100%/0% is noise) and no readout strip (latest-tick misleads when the published head is the best-val epoch, not the last); RunSummary carries the verdict. -->
    <RunSummary {view} epochs={job.epochs} />
  {:else}
    <!-- Running: bar + phase caption + jobId chip. No "running" pill since the surrounding chrome already encodes that; the phase name is what's new. -->
    <div class="space-y-1">
      <div class="relative h-1.5 overflow-hidden rounded-full bg-surface-2">
        {#if progressPct !== null}
          <div
            class="absolute inset-y-0 left-0 bg-accent transition-[width] duration-150 ease-out"
            style="width: {progressPct}%"
          ></div>
        {:else}
          <div class="absolute inset-y-0 indeterminate-bar bg-accent"></div>
        {/if}
      </div>
      <p class="flex items-baseline justify-between gap-2 text-[11px] text-fg-muted">
        <span class="min-w-0 truncate">
          {#if view === null}
            <span class="text-fg-subtle">{m.training.progress.submitting}</span>
          {:else}
            <span class="font-medium text-fg-secondary">{phaseLabel}</span>
            {#if view.progress.total > 0}
              <span class="ml-1.5 font-mono tabular-nums text-fg-muted">
                {view.progress.current} / {view.progress.total}
              </span>
            {/if}
          {/if}
        </span>
        <span class="shrink-0 font-mono text-[10px] text-fg-subtle" title={job.jobId}>
          {m.training.progress.job_short_id(job.jobId.slice(0, 8))}
        </span>
      </p>
    </div>
  {/if}

  <!-- Always rendered (placeholder before Train); on terminal cards the best-val-epoch marker matters since the published head is that epoch, not the last. -->
  <MetricsChart epochs={job.epochs} {valDisabled} />

  <!-- Latest-tick readout for running runs only: on terminal cards a stale latest tick misleads when the published head is an earlier best-val epoch (RunSummary carries the verdict). -->
  {#if !isTerminal}
    <dl
      class="grid grid-cols-3 gap-x-3 gap-y-2 rounded-md border border-line bg-surface-2 p-3 text-xs"
    >
      <div>
        <dt class="text-[10px] uppercase tracking-wider text-fg-muted">
          {m.training.progress.train_loss_label}
        </dt>
        <dd class="mt-0.5 font-mono text-sm tabular-nums text-fg">
          {fmtLoss(latestMetrics?.train_loss)}
        </dd>
      </div>
      <div>
        <dt class="text-[10px] uppercase tracking-wider text-fg-muted">
          {m.training.progress.train_acc_label}
        </dt>
        <dd class="mt-0.5 font-mono text-sm tabular-nums text-fg">
          {fmtAcc(latestMetrics?.train_acc)}
        </dd>
      </div>
      <div>
        <dt class="text-[10px] uppercase tracking-wider text-fg-muted">
          {valDisabled
            ? m.training.progress.val_acc_disabled_label
            : m.training.progress.val_acc_label}
        </dt>
        <dd class="mt-0.5 font-mono text-sm tabular-nums text-fg">
          {fmtAcc(valDisabled ? undefined : latestMetrics?.val_acc)}
        </dd>
      </div>
    </dl>
  {/if}

  <!-- Rolling log synthesised client-side from per-tick progress.message deltas: the daemon keeps no message history, so this is the only place to re-read a run's trace. -->
  <TrainLogs lines={job.logLines} />
</div>

<style>
  /* 30% width balances "feels live" against "reads as a bar". */
  .indeterminate-bar {
    width: 30%;
    animation: indeterminate 1.4s cubic-bezier(0.4, 0, 0.6, 1) infinite;
  }
  @keyframes indeterminate {
    0% {
      transform: translateX(-100%);
    }
    100% {
      transform: translateX(370%);
    }
  }
</style>
