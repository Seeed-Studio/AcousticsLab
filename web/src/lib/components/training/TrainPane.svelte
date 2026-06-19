<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import Button from '$lib/components/ui/Button.svelte';
  import { training as trainingStore } from '$lib/stores/training.svelte';
  import { categories } from '$lib/stores/categories.svelte';
  import { slices } from '$lib/stores/slices.svelte';
  import {
    isMandatoryCategory,
    MANDATORY_BACKGROUND_NOISE,
    thresholdFor
  } from '$lib/components/category/labels';
  import TrainForm from './TrainForm.svelte';
  import TrainHistory from './TrainHistory.svelte';
  import { DEFAULT_EPOCHS, DEFAULT_VALIDATION_SPLIT } from './labels';
  import { m } from '$lib/i18n';
  import type { HeadRecord, TrainingCfg, Uuid } from '$lib/api/types';

  // Layout deliberately stays stable across idle/running/finished: the live run is the top
  // card of the always-present history list (not a floating overlay), only its state word
  // morphs, so the eye never re-anchors. Activation is owned by the Heads section, so
  // history rows stay observational.

  interface Props {
    workspaceId: Uuid;
    // Max of the detail revision and any upload receipt the slices store has seen; a head
    // matching it morphs the button to "Re-train" (idle_trained).
    liveRevision: number;
    heads: readonly HeadRecord[];
  }
  let { workspaceId, liveRevision, heads }: Props = $props();

  const active = $derived(trainingStore.activeFor(workspaceId));
  const trainSlotHeld = $derived(trainingStore.active !== null);
  const otherWorkspaceRunning = $derived(
    trainingStore.active !== null && trainingStore.active.workspaceId !== workspaceId
  );

  // Existence of a head at the live revision (drives idle_trained). Only the boolean is read, so no
  // ordering is needed — HeadsTable owns the deterministic newest-at-liveRevision pick.
  const currentHead = $derived(heads.find((h) => h.workspace_revision.id === liveRevision) ?? null);

  // Readiness gate mirrors the daemon (refuses < 2 non-empty categories) plus per-category
  // thresholds (20 bg / 10 fg) matching the dataset module's "Synced" badge bar.
  function committedCountFor(name: string): number {
    let n = 0;
    for (const s of slices.for(workspaceId, name).entries) {
      if (s.state === 'committed') n++;
    }
    return n;
  }

  const datasetLoaded = $derived(categories.for(workspaceId).loaded);
  const categoryEntries = $derived(categories.for(workspaceId).entries);

  type Readiness =
    | { kind: 'loading' }
    | { kind: 'ready' }
    | { kind: 'no_categories' }
    | { kind: 'background_short'; have: number; need: number }
    | { kind: 'foreground_short' };

  const readiness = $derived.by<Readiness>(() => {
    if (!datasetLoaded) return { kind: 'loading' };
    const cats = categoryEntries;
    if (cats.length < 2) return { kind: 'no_categories' };
    const bgHave = committedCountFor(MANDATORY_BACKGROUND_NOISE);
    const bgNeed = thresholdFor(MANDATORY_BACKGROUND_NOISE);
    if (bgHave < bgNeed) return { kind: 'background_short', have: bgHave, need: bgNeed };
    const foregroundReady = cats
      .filter((c) => !isMandatoryCategory(c.name))
      .some((c) => committedCountFor(c.name) >= thresholdFor(c.name));
    if (!foregroundReady) return { kind: 'foreground_short' };
    return { kind: 'ready' };
  });

  function readinessReason(r: Readiness): string {
    switch (r.kind) {
      case 'loading':
        return m.training.pane.readiness_loading;
      case 'no_categories':
        return m.training.pane.readiness_no_categories;
      case 'background_short':
        return m.training.pane.readiness_background_short(r.need - r.have);
      case 'foreground_short':
        return m.training.pane.readiness_foreground_short;
      case 'ready':
        return '';
    }
  }

  // Lifted from TrainForm via bind: so the primary action can read validity/config.
  let cfg = $state<TrainingCfg | null>(null);
  let hasFieldErrors = $state(false);

  // Force-open on validation error so the operator sees the wrong field instead of a
  // disabled button with no visible reason; manual toggle still works afterward.
  let settingsOpen = $state(false);
  $effect(() => {
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
    if (hasFieldErrors) settingsOpen = true;
  });

  // Collapsed summary shows only epochs + validation split (glanceable: duration, and
  // whether a holdout score returns); rarely-tuned batch_size/learning_rate live behind
  // the disclosure.
  const summaryChips = $derived.by(() => {
    const c = cfg;
    const epochs = c?.epochs ?? DEFAULT_EPOCHS;
    const vs = c?.validation_split ?? DEFAULT_VALIDATION_SPLIT;
    return [
      m.training.pane.summary_chip_epochs(epochs),
      vs === 0
        ? m.training.pane.summary_chip_no_holdout
        : m.training.pane.summary_chip_val(`${Math.round(vs * 100)}%`)
    ];
  });

  // Primary action precedence: running > starting > loading > not-ready > busy > trained >
  // ready. Variant leaves `primary` only at two endpoints: `destructive` on
  // running/cancelling, and quiet `secondary` on idle_trained so Re-train doesn't compete
  // with the Heads section's Activate CTA; the first click re-lights `primary` via `starting`.
  type ButtonStateKind =
    | 'idle_ready'
    | 'idle_not_ready'
    | 'idle_trained'
    | 'idle_busy'
    | 'idle_loading'
    | 'starting'
    | 'running'
    | 'cancelling';

  const MIN_STARTING_MS = 350;
  let startingPin = $state(false);
  let startingPinTimer: ReturnType<typeof setTimeout> | null = null;

  function pinStarting(): void {
    startingPin = true;
    if (startingPinTimer !== null) clearTimeout(startingPinTimer);
    startingPinTimer = setTimeout(() => {
      startingPin = false;
      startingPinTimer = null;
    }, MIN_STARTING_MS);
  }

  const buttonStateKind = $derived.by<ButtonStateKind>(() => {
    if (active) {
      if (active.cancelling) return 'cancelling';
      if (startingPin) return 'starting';
      return 'running';
    }
    if (trainingStore.starting || startingPin) return 'starting';
    if (readiness.kind === 'loading') return 'idle_loading';
    if (readiness.kind !== 'ready') return 'idle_not_ready';
    if (otherWorkspaceRunning) return 'idle_busy';
    if (currentHead !== null) return 'idle_trained';
    return 'idle_ready';
  });

  const buttonLabel = $derived.by(() => {
    switch (buttonStateKind) {
      case 'starting':
        return m.training.pane.button_starting;
      case 'running':
        return m.training.pane.button_cancel;
      case 'cancelling':
        return m.training.pane.button_cancelling;
      case 'idle_trained':
        return m.training.pane.button_retrain;
      default:
        return m.training.pane.button_train;
    }
  });

  const buttonVariant = $derived.by<'primary' | 'secondary' | 'destructive'>(() => {
    if (buttonStateKind === 'running' || buttonStateKind === 'cancelling') return 'destructive';
    if (buttonStateKind === 'idle_trained') return 'secondary';
    return 'primary';
  });

  const buttonLoading = $derived(
    buttonStateKind === 'starting' || buttonStateKind === 'cancelling'
  );

  const buttonDisabled = $derived.by(() => {
    if (buttonStateKind === 'idle_ready' || buttonStateKind === 'idle_trained') {
      return cfg === null || hasFieldErrors;
    }
    if (buttonStateKind === 'running') return false;
    return true;
  });

  const buttonTitle = $derived.by(() => {
    const t = m.training.pane;
    switch (buttonStateKind) {
      case 'idle_loading':
        return t.button_title_loading;
      case 'idle_not_ready':
        return readinessReason(readiness);
      case 'idle_trained':
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (cfg === null || hasFieldErrors) {
          return t.button_title_form_errors;
        }
        return t.button_title_idle_trained;
      case 'idle_busy':
        return t.button_title_idle_busy;
      case 'idle_ready':
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (cfg === null || hasFieldErrors) {
          return t.button_title_form_errors;
        }
        return t.button_title_idle_ready;
      case 'starting':
        return t.button_title_starting;
      case 'running':
        return t.button_title_running;
      case 'cancelling':
        return t.button_title_cancelling;
    }
  });

  // Subtitle stays stable while training: a per-tick "started 3s ago" morph would tick
  // every poll and duplicate the live row's timestamp, and the running state is already
  // shown by the button and the row's blue border. Branches only on readiness and the
  // cross-workspace busy interlock.
  const subtitle = $derived.by(() => {
    // Readiness outranks busy so this agrees with `buttonStateKind` (which gates
    // idle_loading/idle_not_ready before otherWorkspaceRunning); ranking busy first would
    // let subtitle and button tooltip name different blockers for the same disabled button.
    if (readiness.kind !== 'loading' && readiness.kind !== 'ready') {
      return readinessReason(readiness);
    }
    if (otherWorkspaceRunning) {
      return m.training.pane.subtitle_other_running;
    }
    return m.training.pane.subtitle_default;
  });

  // Amber on a readiness/busy obstacle, else zinc; no blue variant since active training
  // is already signalled elsewhere.
  const subtitleTone = $derived.by<'zinc' | 'amber'>(() => {
    if (otherWorkspaceRunning) return 'amber';
    if (readiness.kind === 'loading' || readiness.kind === 'ready') return 'zinc';
    return 'amber';
  });

  async function onPrimaryClick(): Promise<void> {
    if (buttonStateKind === 'running') {
      try {
        await trainingStore.cancel();
      } catch {
        // Store logs the failure; the button re-enables itself.
      }
      return;
    }
    if (buttonStateKind !== 'idle_ready' && buttonStateKind !== 'idle_trained') return;
    if (cfg === null || hasFieldErrors) return;
    pinStarting();
    try {
      await trainingStore.start(workspaceId, cfg);
    } catch {
      if (startingPinTimer !== null) {
        clearTimeout(startingPinTimer);
        startingPinTimer = null;
      }
      startingPin = false;
    }
  }

  onMount(() => {
    void trainingStore.recover(workspaceId);
  });

  onDestroy(() => {
    if (startingPinTimer !== null) clearTimeout(startingPinTimer);
  });

  function dismissStartError(): void {
    trainingStore.startError = null;
  }
</script>

<section class="rounded-xl border border-line bg-surface px-5 pt-3.5 pb-5 shadow-card">
  <!-- `items-center` keeps the action button balanced against the title block's vertical
       centroid when the subtitle wraps to multiple lines. -->
  <header class="mb-3 flex items-center justify-between gap-3">
    <div class="min-w-0">
      <h2 class="text-sm font-semibold text-fg">{m.training.pane.heading}</h2>
      <p
        class="mt-0.5 text-xs"
        class:text-fg-muted={subtitleTone === 'zinc'}
        class:text-warning-soft-fg={subtitleTone === 'amber'}
      >
        {subtitle}
      </p>
    </div>
    <Button
      variant={buttonVariant}
      disabled={buttonDisabled}
      loading={buttonLoading}
      onclick={onPrimaryClick}
      title={buttonTitle}
      ariaLabel={buttonLabel}
    >
      <span class="relative inline-grid grid-cols-1 grid-rows-1 items-center">
        {#key buttonLabel}
          <span
            in:fade={{ duration: 150, easing: cubicOut }}
            out:fade={{ duration: 120, easing: cubicOut }}
            class="col-start-1 row-start-1 whitespace-nowrap"
          >
            {buttonLabel}
          </span>
        {/key}
      </span>
    </Button>
  </header>

  <!-- Geometry keyed on message presence: MULTI-LINE (typed daemon error) uses items-start
       + px-3 py-2 with -mt-1 -mr-2 pinning the X to a 4 px-inset top-right corner;
       SINGLE-LINE (defensive empty error) uses items-center + py-1 pr-1 pl-2.5, the extra
       left padding compensating items-center half-leading so cap-left ≈ cap-top ≈ bottom. -->
  {#if trainingStore.startError}
    {@const hasMessage = trainingStore.startError.trim().length > 0}
    <div
      in:fade={{ duration: 200, easing: cubicOut }}
      out:fade={{ duration: 160, easing: cubicOut }}
      class="mb-3 flex justify-between gap-2 rounded-md border border-danger-line bg-danger-soft text-xs text-danger-soft-fg"
      class:items-start={hasMessage}
      class:items-center={!hasMessage}
      class:px-3={hasMessage}
      class:py-2={hasMessage}
      class:py-1={!hasMessage}
      class:pr-1={!hasMessage}
      class:pl-2.5={!hasMessage}
      role="alert"
    >
      <div class="min-w-0">
        <p class="font-medium">{m.training.pane.start_error_title}</p>
        {#if hasMessage}
          <p class="mt-0.5 wrap-break-word">{trainingStore.startError}</p>
        {/if}
      </div>
      <button
        type="button"
        onclick={dismissStartError}
        aria-label={m.common.dismiss}
        class="shrink-0 rounded-md p-1 text-danger-soft-fg transition hover:bg-danger-soft"
        class:-mt-1={hasMessage}
        class:-mr-2={hasMessage}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          class="h-3.5 w-3.5"
          aria-hidden="true"
        >
          <path d="M6 6l12 12M6 18L18 6" />
        </svg>
      </button>
    </div>
  {/if}

  <!-- The `grid-template-rows: 0fr <-> 1fr` trick keeps the form panel mounted across
       open/close so it animates without losing operator-typed values. -->
  <div class="mb-3 rounded-md border border-line bg-surface-2/60">
    <button
      type="button"
      onclick={() => (settingsOpen = !settingsOpen)}
      aria-expanded={settingsOpen}
      aria-controls="train-settings-panel"
      class="flex w-full items-center justify-between gap-3 px-3 py-2 text-left transition hover:bg-surface-2"
    >
      <span class="flex min-w-0 items-center gap-2">
        <!-- The chevron path's centroid sits 0.72 px high unrotated and rotate-90 shifts it
             1 px down, so a static translate would fix one state and break the other; gating
             `translate-y-px` on `!settingsOpen` lands both at the same ~0.3 px residual so
             the chevron never hops during the disclosure animation. -->
        <svg
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
          class="h-3.5 w-3.5 shrink-0 text-fg-muted transition-transform duration-200"
          class:translate-y-px={!settingsOpen}
          class:rotate-90={settingsOpen}
        >
          <path
            fill-rule="evenodd"
            d="M7.21 5.23a.75.75 0 011.06.02L12 9l-3.73 3.71a.75.75 0 11-1.06-1.06L9.94 9 7.19 6.29a.75.75 0 01.02-1.06z"
            clip-rule="evenodd"
          />
        </svg>
        <span class="text-xs font-medium text-fg-secondary"
          >{m.training.pane.hyperparameters_disclosure_label}</span
        >
      </span>
      {#if !settingsOpen}
        <span
          in:fade={{ duration: 180, easing: cubicOut }}
          class="hidden shrink-0 flex-wrap items-center justify-end gap-1 sm:flex"
          aria-hidden="true"
        >
          {#each summaryChips as chip (chip)}
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
      id="train-settings-panel"
      class="grid transition-[grid-template-rows] duration-200 ease-out"
      class:grid-rows-[1fr]={settingsOpen}
      class:grid-rows-[0fr]={!settingsOpen}
    >
      <div class="min-h-0 overflow-hidden" inert={!settingsOpen} aria-hidden={!settingsOpen}>
        <div class="border-t border-line bg-surface px-3 py-3">
          <TrainForm disabled={trainSlotHeld} bind:cfg bind:hasFieldErrors />
        </div>
      </div>
    </div>
  </div>

  <TrainHistory {workspaceId} />
</section>
