// Bounds mirror the daemon's training-config validation so the form rejects at keystroke
// what the daemon would 400 on. Label functions read the i18n catalog at the call site so
// Svelte tracks the reactive locale read and re-renders on locale switch.

import { m } from '$lib/i18n';
import type { Stage, TrainingJobState } from '$lib/api/types';

export const MIN_EPOCHS = 1;
export const MAX_EPOCHS = 1_000;
export const MIN_BATCH_SIZE = 1;
export const MAX_BATCH_SIZE = 4_096;
// learning_rate is (0, MAX]: exclusive low, inclusive high; NaN/Infinity rejected.
export const MAX_LEARNING_RATE = 1.0;
// validation_split is [0, 1): 0 disables the stratified split; near-1 leaves no training data.
export const MIN_VALIDATION_SPLIT = 0.0;
export const MAX_VALIDATION_SPLIT = 0.999; // operator-facing cap; daemon validates < 1.0

// Tuned for the device's small-dataset, few-minute runs; 0.2 split leaves holdout for best-epoch.
export const DEFAULT_EPOCHS = 50;
export const DEFAULT_BATCH_SIZE = 32;
export const DEFAULT_LEARNING_RATE = 1e-3;
export const DEFAULT_VALIDATION_SPLIT = 0.2;

// Catalog returns fully human-worded labels (e.g. feature_extract -> "Extracting features"); no CSS text-transform can derive these from the raw keys.
export function stageLabel(stage: Stage): string {
  return m.training.stage[stage];
}

// Strings are lowercase to match the pill's text-transform: capitalize convention.
export function trainingStateLabel(state: TrainingJobState): string {
  return m.training.state[state];
}

export const TERMINAL_TRAINING_STATES: ReadonlySet<TrainingJobState> = new Set([
  'completed',
  'failed',
  'cancelled'
]);
