<script lang="ts">
  import { inputClass } from '$lib/components/ui/inputClass';
  import { training as trainingStore } from '$lib/stores/training.svelte';
  import {
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_VALIDATION_SPLIT,
    MAX_BATCH_SIZE,
    MAX_EPOCHS,
    MAX_LEARNING_RATE,
    MAX_VALIDATION_SPLIT,
    MIN_BATCH_SIZE,
    MIN_EPOCHS,
    MIN_VALIDATION_SPLIT
  } from './labels';
  import {
    validateBatchSize,
    validateEpochs,
    validateLearningRate,
    validateSeed,
    validateValidationSplit
  } from './cfg-validate';
  import { m } from '$lib/i18n';
  import type { TrainingCfg } from '$lib/api/types';

  // Field state lives here; the parent owns the submit button and start-error surfacing.

  interface Props {
    // Held while a training job is in flight (daemon's max_train_jobs=1 is one global slot
    // across workspaces) and while `starting`, so fields can't change under an in-flight request.
    disabled?: boolean;
    // null when any required field is empty or invalid; parent gates submit on `cfg !== null`.
    cfg?: TrainingCfg | null;
    // Distinct from `cfg === null` (which also fires on empty required fields) so the parent can
    // distinguish "fix highlighted fields" from "fill in required fields".
    hasFieldErrors?: boolean;
  }
  let {
    disabled = false,
    cfg = $bindable(null),
    hasFieldErrors = $bindable(false)
  }: Props = $props();

  // `<input type="number">` binds null on empty input; modeling fields as `number | null` keeps
  // a cleared required field "required" (not read as 0) and lets optional fields treat null as absent.
  let epochs = $state<number | null>(DEFAULT_EPOCHS);
  let batchSize = $state<number | null>(DEFAULT_BATCH_SIZE);
  let learningRate = $state<number | null>(DEFAULT_LEARNING_RATE);
  let seed = $state<number | null>(null);
  let validationSplit = $state<number | null>(DEFAULT_VALIDATION_SPLIT);

  const epochsError = $derived(validateEpochs(epochs));
  const batchSizeError = $derived(validateBatchSize(batchSize));
  const learningRateError = $derived(validateLearningRate(learningRate));
  const seedError = $derived(validateSeed(seed));
  const validationSplitError = $derived(validateValidationSplit(validationSplit));

  // `seed` excluded: optional, null means let the daemon pick.
  const allRequiredPresent = $derived(
    epochs !== null && batchSize !== null && learningRate !== null && validationSplit !== null
  );
  const computedHasErrors = $derived(
    !!epochsError ||
      !!batchSizeError ||
      !!learningRateError ||
      !!seedError ||
      !!validationSplitError
  );

  // Separate $effects: bindable writes are reactive, so one shared effect would chain a no-op
  // write on one binding into a re-fire of the other.
  $effect(() => {
    hasFieldErrors = computedHasErrors;
  });
  $effect(() => {
    if (
      !allRequiredPresent ||
      computedHasErrors ||
      epochs === null ||
      batchSize === null ||
      learningRate === null ||
      validationSplit === null
    ) {
      cfg = null;
      return;
    }
    const next: TrainingCfg = {
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      validation_split: validationSplit
    };
    // Omit `seed` when null so the daemon's `Option<u64>` parses as None (per-job entropy).
    if (seed !== null) next.seed = seed;
    cfg = next;
  });

  // Read `starting` directly (not via parent gate) so the form locks the moment a submit lands,
  // even while the parent is mid-recompute.
  const fieldsDisabled = $derived(disabled || trainingStore.starting);
</script>

<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
  <label class="block">
    <span class="mb-1 block text-xs text-fg-secondary">{m.training.form.epochs_label}</span>
    <input
      type="number"
      bind:value={epochs}
      disabled={fieldsDisabled}
      min={MIN_EPOCHS}
      max={MAX_EPOCHS}
      step="1"
      inputmode="numeric"
      aria-invalid={epochsError ? true : undefined}
      aria-describedby={epochsError ? 'train-epochs-error' : undefined}
      class={inputClass(!!epochsError)}
    />
    {#if epochsError}
      <p id="train-epochs-error" class="mt-1 text-xs text-danger-soft-fg" role="alert">
        {epochsError}
      </p>
    {/if}
  </label>

  <label class="block">
    <span class="mb-1 block text-xs text-fg-secondary">{m.training.form.batch_size_label}</span>
    <input
      type="number"
      bind:value={batchSize}
      disabled={fieldsDisabled}
      min={MIN_BATCH_SIZE}
      max={MAX_BATCH_SIZE}
      step="1"
      inputmode="numeric"
      aria-invalid={batchSizeError ? true : undefined}
      aria-describedby={batchSizeError ? 'train-batch-error' : undefined}
      class={inputClass(!!batchSizeError)}
    />
    {#if batchSizeError}
      <p id="train-batch-error" class="mt-1 text-xs text-danger-soft-fg" role="alert">
        {batchSizeError}
      </p>
    {/if}
  </label>

  <label class="block">
    <span class="mb-1 block text-xs text-fg-secondary">{m.training.form.learning_rate_label}</span>
    <!-- No min: learning_rate is exclusive (0, MAX]; JS validation rejects v <= 0, and omitting
         min stops the UA stepper/constraint hints from advertising 0 as valid. -->
    <input
      type="number"
      bind:value={learningRate}
      disabled={fieldsDisabled}
      max={MAX_LEARNING_RATE}
      step="0.0001"
      inputmode="decimal"
      aria-invalid={learningRateError ? true : undefined}
      aria-describedby={learningRateError ? 'train-lr-error' : undefined}
      class={inputClass(!!learningRateError)}
    />
    {#if learningRateError}
      <p id="train-lr-error" class="mt-1 text-xs text-danger-soft-fg" role="alert">
        {learningRateError}
      </p>
    {/if}
  </label>

  <label class="block">
    <span class="mb-1 block text-xs text-fg-secondary">
      {m.training.form.validation_split_label}
      <span class="text-fg-subtle">{m.training.form.validation_split_hint}</span>
    </span>
    <input
      type="number"
      bind:value={validationSplit}
      disabled={fieldsDisabled}
      min={MIN_VALIDATION_SPLIT}
      max={MAX_VALIDATION_SPLIT}
      step="0.01"
      inputmode="decimal"
      aria-invalid={validationSplitError ? true : undefined}
      aria-describedby={validationSplitError ? 'train-vs-error' : undefined}
      class={inputClass(!!validationSplitError)}
    />
    {#if validationSplitError}
      <p id="train-vs-error" class="mt-1 text-xs text-danger-soft-fg" role="alert">
        {validationSplitError}
      </p>
    {/if}
  </label>

  <label class="block sm:col-span-2">
    <span class="mb-1 block text-xs text-fg-secondary">
      {m.training.form.seed_label}
      <span class="text-fg-subtle">{m.training.form.seed_hint}</span>
    </span>
    <input
      type="number"
      bind:value={seed}
      disabled={fieldsDisabled}
      min="0"
      step="1"
      inputmode="numeric"
      placeholder={m.training.form.seed_placeholder}
      aria-invalid={seedError ? true : undefined}
      aria-describedby={seedError ? 'train-seed-error' : undefined}
      class={inputClass(!!seedError)}
    />
    {#if seedError}
      <p id="train-seed-error" class="mt-1 text-xs text-danger-soft-fg" role="alert">
        {seedError}
      </p>
    {/if}
  </label>
</div>
