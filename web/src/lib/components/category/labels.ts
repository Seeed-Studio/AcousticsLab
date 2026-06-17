// Reserved `_..._` synthetics (upstream Speech-Commands) render title-cased; operator labels pass through verbatim, safe because the wire-form validator forbids them a leading `_`.

export const MANDATORY_BACKGROUND_NOISE = '_background_noise_';

// `_background_noise_` held higher: as the negative class it needs more counterexamples than any one target.
export const THRESHOLD_BACKGROUND_NOISE = 20;
export const THRESHOLD_STANDARD = 10;

// Per-click amber warning only (no cumulative cap anywhere): a runaway batch past useful-training value costs ~88 KB WAV + a 96x64 spectrogram per slice in IDB (200 ≈ 18 MB; untrimmed 30 min = 1800).
export const SLICE_BATCH_WARN_THRESHOLD = 200;

export function isMandatoryCategory(name: string): boolean {
  return name === MANDATORY_BACKGROUND_NOISE;
}

export function thresholdFor(name: string): number {
  return isMandatoryCategory(name) ? THRESHOLD_BACKGROUND_NOISE : THRESHOLD_STANDARD;
}

// Hyphen split is for future label sets (no shipped synthetic uses one); an all-separator inner falls back to verbatim wire form so the row stays distinguishable.
export function prettyCategoryName(name: string): string {
  if (name.length >= 2 && name.startsWith('_') && name.endsWith('_')) {
    const parts = name
      .slice(1, -1)
      .split(/[\s_-]+/)
      .filter((p) => p.length > 0);
    if (parts.length === 0) return name;
    return parts.map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(' ');
  }
  return name;
}

// `opts.max` caps visible count, appending a trailing ellipsis only when the list exceeds it; omit it for the full list (HTML `title` tooltips).
export function formatLabelsList(labels: readonly string[], opts: { max?: number } = {}): string {
  const pretty = labels.map(prettyCategoryName);
  if (opts.max === undefined || pretty.length <= opts.max) {
    return pretty.join(', ');
  }
  return `${pretty.slice(0, opts.max).join(', ')}, …`;
}
