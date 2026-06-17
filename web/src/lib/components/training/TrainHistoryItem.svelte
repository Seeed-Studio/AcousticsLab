<script lang="ts">
  import { slide } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import JobProgress from './JobProgress.svelte';
  import { formatRelative } from '$lib/utils/time';
  import { stageLabel, trainingStateLabel } from './labels';
  import { formatLabelsList } from '$lib/components/category/labels';
  import { m } from '$lib/i18n';
  import type { TrackedTrainingJob } from '$lib/stores/training.svelte';

  // Parent keys by `job.jobId`, so the live->terminal flip swaps the view object under a stable item (no remount).

  interface Props {
    job: TrackedTrainingJob;
    // Parent-owned so expansion survives the active->terminal move (`store.active`->`store.history`).
    expanded: boolean;
    ontoggle: () => void;
    // Top-of-list active run; drives the state-word pulse.
    isLive: boolean;
    // Delete in flight; expansion stays interactive so the body is inspectable while the tombstone drains.
    isDeleting?: boolean;
  }
  let { job, expanded, ontoggle, isLive, isDeleting = false }: Props = $props();

  // `view === null` is the just-submitted, pre-first-poll window: show 'submitting'.
  type DisplayState = 'submitting' | 'running' | 'completed' | 'failed' | 'cancelled';
  const displayState = $derived<DisplayState>(job.view === null ? 'submitting' : job.view.state);
  // Separate from `displayState`: the null guard here narrows `job.view`, sidestepping a non-null assertion in markup.
  const stateLabel = $derived.by(() => {
    const v = job.view;
    if (v === null) return m.training.state_submitting;
    return trainingStateLabel(v.state);
  });

  // Live runs anchor relative-time on start, terminal runs on finish; full timestamps in title.
  const timeLabel = $derived.by(() => {
    const v = job.view;
    // Pre-ack has no daemon `started_at`; bare sentinel beats a relative count anchored on a frozen local Date.
    if (!v) return m.training.history_item.time_started_pre_ack;
    if (v.state === 'running')
      return m.training.history_item.time_started(formatRelative(v.started_at));
    if (v.finished_at) return m.training.history_item.time_finished(formatRelative(v.finished_at));
    return m.training.history_item.time_started(formatRelative(v.started_at));
  });
  const timeTitle = $derived.by(() => {
    const v = job.view;
    if (!v) return '';
    const parts = [m.training.history_item.time_title_started(v.started_at)];
    if (v.finished_at) parts.push(m.training.history_item.time_title_finished(v.finished_at));
    return parts.join(' · ');
  });

  // Peak val_acc across epochs; null when none observed (validation_split === 0 or pre-train fail).
  const bestValAcc = $derived.by<number | null>(() => {
    let best: number | null = null;
    for (const e of job.epochs) {
      const v = e.val_acc;
      if (v === null || !Number.isFinite(v)) continue;
      if (best === null || v > best) best = v;
    }
    return best;
  });

  // Array (not concatenated string) so each token stays nowrap and the flex wrapper breaks between them when narrow.
  interface TrailingToken {
    text: string;
    title?: string;
    // Lowest-precedence: hidden on a narrow row (@container thresholds in markup) to keep accuracy + head-id off a wrap line.
    hideNarrow?: boolean;
  }
  const trailingDetail = $derived.by<readonly TrailingToken[]>(() => {
    const v = job.view;
    if (!v) return [];
    if (v.state === 'running') {
      if (v.progress.phase === 'train' && v.progress.total > 0) {
        return [
          { text: m.training.history_item.detail_epoch(v.progress.current, v.progress.total) }
        ];
      }
      return [{ text: stageLabel(v.progress.phase).toLowerCase() }];
    }
    if (v.state === 'completed') {
      const tokens: TrailingToken[] = [];
      const nClasses = v.result?.n_classes ?? 0;
      if (nClasses > 0) {
        const classes = v.result?.classes ?? [];
        tokens.push({
          text: m.training.history_item.detail_class_count(nClasses),
          // formatLabelsList for reserved-synthetic pretty-form (`_unknown_` -> "Unknown").
          title: classes.length > 0 ? formatLabelsList(classes) : undefined,
          hideNarrow: true
        });
      }
      if (bestValAcc !== null) {
        tokens.push({
          text: m.training.history_item.detail_val_acc(`${(bestValAcc * 100).toFixed(1)}%`)
        });
      } else if (v.result && Number.isFinite(v.result.final_train_acc)) {
        tokens.push({
          text: m.training.history_item.detail_train_acc(
            `${(v.result.final_train_acc * 100).toFixed(1)}%`
          )
        });
      }
      return tokens;
    }
    // failed | cancelled share "stopped at <stage>" copy, distinguished only by the left-border accent.
    return [
      {
        text: m.training.history_item.detail_stopped_at(stageLabel(v.progress.phase).toLowerCase())
      }
    ];
  });

  // Head_id only on `completed`: failed/cancelled carry a pre-allocated head_id but no head landed, so showing it would imply a deployable artefact exists.
  const completedHeadId = $derived.by<string | null>(() => {
    const v = job.view;
    if (v?.state !== 'completed') return null;
    return v.result?.head_id ?? null;
  });
</script>

<!-- `data-job-id` is the parent's delegated right-click hook (handler walks `closest('[data-job-id]')`). -->
<li
  data-job-id={job.jobId}
  aria-busy={isDeleting}
  class="overflow-hidden rounded-md border border-line border-l-4 bg-surface transition-[opacity,background-color,border-color] duration-200"
  class:border-l-accent={displayState === 'running' || displayState === 'submitting'}
  class:border-l-success-dot={displayState === 'completed'}
  class:border-l-danger-dot={displayState === 'failed'}
  class:border-l-fg-subtle={displayState === 'cancelled'}
  class:opacity-60={isDeleting}
>
  <button
    type="button"
    onclick={ontoggle}
    aria-expanded={expanded}
    class="flex w-full items-center gap-2.5 px-3 py-2.5 text-left"
  >
    <!-- Chevron optical alignment: path centroid sits ~0.72px above box centre, so `translate-y-px` is gated on `!expanded` (a static translate would mis-center the rotated case). Relies on Tailwind v4 emitting `rotate`/`translate` as independent CSS props. -->
    <svg
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
      class="h-3.5 w-3.5 shrink-0 text-fg-subtle transition-transform duration-200"
      class:translate-y-px={!expanded}
      class:rotate-90={expanded}
    >
      <path
        fill-rule="evenodd"
        d="M7.21 5.23a.75.75 0 011.06.02L12 9l-3.73 3.71a.75.75 0 11-1.06-1.06L9.94 9 7.19 6.29a.75.75 0 01.02-1.06z"
        clip-rule="evenodd"
      />
    </svg>

    <!-- `@container/histrow` degrades precedence: narrowing hides low-value facts (time @[24rem], class count @[18rem]) rather than wrapping, flex-wrap as last net. display:none a11y drop is intentional since hidden facts are neutral and recoverable by expanding (unlike HeadRow's load-bearing freshness pill, which is sr-only). -->
    <span
      class="@container/histrow flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2 gap-y-0.5 text-xs"
    >
      <span
        class="shrink-0 font-medium capitalize"
        class:text-accent-soft-fg={displayState === 'running' || displayState === 'submitting'}
        class:animate-pulse={isLive}
        class:text-success-soft-fg={displayState === 'completed'}
        class:text-danger-soft-fg={displayState === 'failed'}
        class:text-fg-secondary={displayState === 'cancelled'}
      >
        {stateLabel}
      </span>
      <span class="hidden shrink-0 text-fg-muted @[24rem]/histrow:inline" title={timeTitle}>
        {timeLabel}
      </span>
      {#each trailingDetail as tok, i (i)}
        <span
          aria-hidden="true"
          class="shrink-0 text-fg-subtle {tok.hideNarrow ? 'hidden @[18rem]/histrow:inline' : ''}"
          >·</span
        >
        <span
          class="shrink-0 font-mono text-[11px] tabular-nums text-fg-muted {tok.hideNarrow
            ? 'hidden @[18rem]/histrow:inline'
            : ''}"
          title={tok.title}
        >
          {tok.text}
        </span>
      {/each}
      {#if completedHeadId}
        <span
          class="ml-auto shrink-0 font-mono text-[11px] tabular-nums text-fg-muted"
          title={completedHeadId}
        >
          {completedHeadId.slice(0, 8)}
        </span>
      {/if}
    </span>
  </button>

  <!-- Body remounts on every expand so JobProgress's chart re-runs its initial measurement. -->
  {#if expanded}
    <div
      transition:slide={{ duration: 220, easing: cubicOut }}
      class="border-t border-line bg-surface-2/60 px-3 py-3"
    >
      <JobProgress {job} />
    </div>
  {/if}
</li>
