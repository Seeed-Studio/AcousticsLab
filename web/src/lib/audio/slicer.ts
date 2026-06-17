import { SLICE_SAMPLES } from './wav';
import { wouldNanAtPreproc } from './silence';

// Full slices a range produces; trailing partials dropped to match chunkPcmToValidSlices.
export function sliceCountFor(
  startSamples: number,
  endSamples: number,
  sliceSamples: number = SLICE_SAMPLES
): number {
  if (sliceSamples <= 0) return 0;
  const span = Math.max(0, endSamples - startSamples);
  return Math.floor(span / sliceSamples);
}

// Slice a trimmed PCM range, dropping windows the daemon's preproc would NaN-reject (digital
// silence in any FFT frame) so they never burn the operator's drop-ratio budget. Floor-divide
// so every slice holds 1 s of REAL audio; the sub-slice remainder is dropped, never
// silence-padded, since zero-tail slices poison training and the daemon re-pads anyway.
export function chunkPcmToValidSlices(
  pcm: Float32Array,
  startSamples: number,
  endSamples: number,
  sliceSamples: number = SLICE_SAMPLES
): { kept: Float32Array[]; silentDropped: number } {
  if (sliceSamples <= 0) {
    throw new Error('sliceSamples must be positive');
  }
  const clampedStart = Math.max(0, Math.min(startSamples, pcm.length));
  const clampedEnd = Math.max(clampedStart, Math.min(endSamples, pcm.length));
  const span = clampedEnd - clampedStart;
  const count = Math.floor(span / sliceSamples);
  if (count === 0) return { kept: [], silentDropped: 0 };

  const kept: Float32Array[] = [];
  let silentDropped = 0;
  for (let i = 0; i < count; i++) {
    const offset = clampedStart + i * sliceSamples;
    // Check the unallocated region directly (bit-equivalent to slice-then-filter, no transient
    // buffers on silent recordings).
    if (wouldNanAtPreproc(pcm, offset)) {
      silentDropped++;
      continue;
    }
    // `.slice` (not aliasing `subarray`) yields the fresh owned buffer the encoder requires.
    kept.push(pcm.slice(offset, offset + sliceSamples));
  }
  return { kept, silentDropped };
}
