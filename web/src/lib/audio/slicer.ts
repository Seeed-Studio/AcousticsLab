import { SLICE_SAMPLES } from './wav';

// Full slices a range produces; trailing partials dropped to match chunkPcmToSlices.
export function sliceCountFor(
  startSamples: number,
  endSamples: number,
  sliceSamples: number = SLICE_SAMPLES
): number {
  if (sliceSamples <= 0) return 0;
  const span = Math.max(0, endSamples - startSamples);
  return Math.floor(span / sliceSamples);
}

// Slice a trimmed PCM range into full 1 s windows. Silence is NOT filtered (the daemon
// trains silent windows like any other); the sub-slice remainder is dropped, never
// zero-padded, so a partial recording can't dilute its category with padded silence.
export function chunkPcmToSlices(
  pcm: Float32Array,
  startSamples: number,
  endSamples: number,
  sliceSamples: number = SLICE_SAMPLES
): Float32Array[] {
  if (sliceSamples <= 0) {
    throw new Error('sliceSamples must be positive');
  }
  const clampedStart = Math.max(0, Math.min(startSamples, pcm.length));
  const clampedEnd = Math.max(clampedStart, Math.min(endSamples, pcm.length));
  const count = sliceCountFor(clampedStart, clampedEnd, sliceSamples);
  // `.slice` (not aliasing `subarray`) yields the fresh owned buffer the encoder requires.
  return Array.from({ length: count }, (_, i) => {
    const offset = clampedStart + i * sliceSamples;
    return pcm.slice(offset, offset + sliceSamples);
  });
}
