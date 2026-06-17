<script lang="ts">
  import { formatDurationHuman } from '$lib/utils/format';
  import { stageLabel as resolveStageLabel } from './labels';
  import { formatLabelsList } from '$lib/components/category/labels';
  import { m } from '$lib/i18n';
  import type { EpochMetrics, TrainingJobView } from '$lib/api/types';

  // Spotlights best-val epoch (not latest), since with validation_split > 0 the published head is
  // the best-val epoch, not the last; failure/cancellation collapse to one tone-tinted column.

  interface Props {
    // Always terminal (completed | failed | cancelled); parent gates before mounting.
    view: TrainingJobView;
    // Empty is valid: a run that fails before the train loop emits no epoch metric.
    epochs: readonly EpochMetrics[];
  }
  let { view, epochs }: Props = $props();

  // Clamp at 0: daemon/tab clock skew can make sub-second runs go slightly negative.
  const durationMs = $derived.by(() => {
    const start = Date.parse(view.started_at);
    const finish = view.finished_at ? Date.parse(view.finished_at) : Date.now();
    if (Number.isNaN(start) || Number.isNaN(finish)) return 0;
    return Math.max(0, finish - start);
  });

  // Argmax over observed val_acc (null when none); re-derived locally because the daemon's
  // best_val_acc carries the peak value but not the epoch index it landed at.
  const bestVal = $derived.by(() => {
    let best: EpochMetrics | null = null;
    let bestAcc = -Infinity;
    for (const e of epochs) {
      const v = e.val_acc;
      if (v === null || !Number.isFinite(v)) continue;
      if (v > bestAcc) {
        bestAcc = v;
        best = e;
      }
    }
    return best;
  });

  // epochsRun is the last-seen 1-indexed epoch; epochsTotal the run length each tick echoes.
  const epochsRun = $derived(epochs.length > 0 ? epochs[epochs.length - 1].epoch : 0);
  const epochsTotal = $derived(epochs.length > 0 ? epochs[epochs.length - 1].epochs : 0);

  // The daemon freezes progress.phase at the terminal moment, so this is where the run stopped.
  const stageLabel = $derived(resolveStageLabel(view.progress.phase));

  // Cancelled views carry no cancel reason from the daemon, so they fall back to a generic line.
  const reason = $derived.by(() => {
    if (view.state === 'failed') {
      const err = (view.error?.trim() ?? '') || view.progress.message.trim();
      return err || m.training.summary.failed_no_diagnostic;
    }
    if (view.state === 'cancelled') {
      // progress.message at cancel time is the last pre-checkpoint tick, rarely conclusive.
      return m.training.summary.cancelled_default_reason;
    }
    return '';
  });

  function fmtAcc(v: number | null | undefined): string {
    if (v === null || v === undefined || !Number.isFinite(v)) return m.training.progress.em_dash;
    return `${(v * 100).toFixed(1)}%`;
  }
</script>

<!-- Centered (not justify-between) so narrow values don't spread into orphaned chips in a wide card. -->
{#if view.state === 'completed'}
  <dl
    class="grid grid-cols-2 gap-x-3 gap-y-2 rounded-md border border-success-line bg-success-soft/50 p-3 text-xs sm:grid-cols-4"
    aria-label={m.training.summary.completed_aria}
  >
    <div class="text-center">
      <dt class="text-[10px] uppercase tracking-wider text-fg-muted">
        {m.training.summary.duration_label}
      </dt>
      <dd class="mt-0.5 font-mono text-sm tabular-nums text-fg">
        {formatDurationHuman(durationMs)}
      </dd>
    </div>
    <div class="text-center">
      <dt class="text-[10px] uppercase tracking-wider text-fg-muted">
        {m.training.summary.epochs_label}
      </dt>
      <dd
        class="mt-0.5 font-mono text-sm tabular-nums text-fg"
        title={epochsRun === epochsTotal && epochsRun > 0
          ? m.training.summary.epochs_tooltip_full
          : m.training.summary.epochs_tooltip_partial}
      >
        {epochsRun}/{epochsTotal || epochsRun || m.training.progress.em_dash}
      </dd>
    </div>
    <div class="text-center">
      <dt class="text-[10px] uppercase tracking-wider text-fg-muted">
        {#if bestVal}{m.training.summary.best_val_at(bestVal.epoch)}{:else}{m.training.summary
            .final_train_acc_label}{/if}
      </dt>
      <dd class="mt-0.5 font-mono text-sm tabular-nums text-fg">
        {#if bestVal}
          {fmtAcc(bestVal.val_acc)}
        {:else}
          {fmtAcc(view.result?.final_train_acc)}
        {/if}
      </dd>
    </div>
    <div
      class="text-center"
      title={view.result && view.result.classes.length > 0
        ? formatLabelsList(view.result.classes)
        : undefined}
    >
      <dt class="text-[10px] uppercase tracking-wider text-fg-muted">
        {m.training.summary.classes_label}
      </dt>
      <dd class="mt-0.5 font-mono text-sm tabular-nums text-fg">
        {view.result?.n_classes ?? m.training.progress.em_dash}
      </dd>
    </div>
  </dl>
{:else if view.state === 'failed'}
  <!-- Stage leads the reason: operators diagnose where it died before why. -->
  <div
    class="space-y-1.5 rounded-md border border-danger-line bg-danger-soft/40 p-3 text-xs"
    aria-label={m.training.summary.failed_aria}
  >
    <div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <span class="text-[10px] uppercase tracking-wider text-danger-soft-fg">
        {m.training.summary.stopped_at_label}
      </span>
      <span class="font-medium text-fg">{stageLabel}</span>
      {#if epochsRun > 0 && epochsTotal > 0 && view.progress.phase === 'train'}
        <span class="font-mono text-[11px] tabular-nums text-fg-muted">
          {m.training.summary.after_epochs(epochsRun, epochsTotal)}
        </span>
      {/if}
      <span class="ml-auto font-mono text-[10px] tabular-nums text-fg-subtle">
        {formatDurationHuman(durationMs)}
      </span>
    </div>
    <p class="wrap-break-word text-danger-soft-fg">{reason}</p>
  </div>
{:else if view.state === 'cancelled'}
  <!-- Same single-column shape as failed but neutral tone: a cancellation is intent, not a defect. -->
  <div
    class="space-y-1.5 rounded-md border border-line bg-surface-2 p-3 text-xs"
    aria-label={m.training.summary.cancelled_aria}
  >
    <div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <span class="text-[10px] uppercase tracking-wider text-fg-muted">
        {m.training.summary.cancelled_at_label}
      </span>
      <span class="font-medium text-fg">{stageLabel}</span>
      {#if epochsRun > 0 && epochsTotal > 0 && view.progress.phase === 'train'}
        <span class="font-mono text-[11px] tabular-nums text-fg-muted">
          {m.training.summary.after_epochs(epochsRun, epochsTotal)}
        </span>
      {/if}
      <span class="ml-auto font-mono text-[10px] tabular-nums text-fg-subtle">
        {formatDurationHuman(durationMs)}
      </span>
    </div>
    <p class="text-fg-secondary">{reason}</p>
  </div>
{/if}
