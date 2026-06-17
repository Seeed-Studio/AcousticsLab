// Client-side mirror of the daemon's training-cfg validation for a no-round-trip verdict; each
// returns null on success or an operator-facing failure sentence (surfaced via aria-invalid + an
// inline role="alert" paragraph, not toasts). null/NaN short-circuits to null so the form's
// "required" affordance owns the empty case rather than red text on first paint.

import {
  MIN_EPOCHS,
  MAX_EPOCHS,
  MIN_BATCH_SIZE,
  MAX_BATCH_SIZE,
  MAX_LEARNING_RATE,
  MIN_VALIDATION_SPLIT,
  MAX_VALIDATION_SPLIT
} from './labels';
import { m } from '$lib/i18n';

export function validateEpochs(v: number | null): string | null {
  const t = m.validation.cfg;
  if (v === null || Number.isNaN(v)) return null;
  if (!Number.isInteger(v)) return t.epochs_whole;
  if (v < MIN_EPOCHS || v > MAX_EPOCHS) {
    return t.epochs_range(MIN_EPOCHS, MAX_EPOCHS);
  }
  return null;
}

export function validateBatchSize(v: number | null): string | null {
  const t = m.validation.cfg;
  if (v === null || Number.isNaN(v)) return null;
  if (!Number.isInteger(v)) return t.batch_whole;
  if (v < MIN_BATCH_SIZE || v > MAX_BATCH_SIZE) {
    return t.batch_range(MIN_BATCH_SIZE, MAX_BATCH_SIZE);
  }
  return null;
}

export function validateLearningRate(v: number | null): string | null {
  const t = m.validation.cfg;
  if (v === null || Number.isNaN(v)) return null;
  if (!Number.isFinite(v)) return t.lr_finite;
  if (v <= 0) return t.lr_greater_than_zero;
  if (v > MAX_LEARNING_RATE) return t.lr_max(MAX_LEARNING_RATE);
  return null;
}

// Cap at MAX_SAFE_INTEGER not the true u64 max: JS loses integer precision past 2^53, so larger
// values are already ambiguous; the daemon's Option<u64> deserialization rejects out-of-range
// values at parse time (validate_training_cfg leaves seed otherwise unconstrained).
export function validateSeed(v: number | null): string | null {
  const t = m.validation.cfg;
  if (v === null || Number.isNaN(v)) return null;
  if (!Number.isInteger(v)) return t.seed_whole;
  if (v < 0) return t.seed_non_negative;
  if (v > Number.MAX_SAFE_INTEGER) return t.seed_too_large;
  return null;
}

export function validateValidationSplit(v: number | null): string | null {
  const t = m.validation.cfg;
  if (v === null || Number.isNaN(v)) return null;
  if (!Number.isFinite(v)) return t.split_finite;
  if (v < MIN_VALIDATION_SPLIT) return t.split_min;
  if (v > MAX_VALIDATION_SPLIT) {
    return t.split_max(MAX_VALIDATION_SPLIT);
  }
  return null;
}
