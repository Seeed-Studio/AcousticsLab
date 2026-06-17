// PCM-16 mono WAV (S16LE RIFF/WAVE) at the daemon's target rate; one slice = 1 s = 44,100 samples, the trainer preprocessor's unit.

export const WAV_SAMPLE_RATE = 44_100;

// Single source of slice length so slicer and spectrogram engine cannot diverge the grid.
export const SLICE_SAMPLES = WAV_SAMPLE_RATE;
export const WAV_NUM_CHANNELS = 1;
export const WAV_BITS_PER_SAMPLE = 16;
export const WAV_MIME = 'audio/wav';

export const WAV_HEADER_BYTES = 44;

// On a LE host (every shipped browser) the quantise loop writes through an Int16Array view, landing bytes in WAV's required LE order and skipping DataView.setInt16's per-call swap (~5-10x faster).
const IS_LITTLE_ENDIAN = (() => {
  const probe = new ArrayBuffer(2);
  new Uint16Array(probe)[0] = 0x0102;
  return new Uint8Array(probe)[0] === 0x02;
})();

// Stamps the 44-byte header at offset 0; exposed so stream-encode helpers can fill a pre-sized WAV buffer in place rather than allocate a body.
export function writeWavHeader(
  view: DataView,
  numSamples: number,
  sampleRate: number = WAV_SAMPLE_RATE
): void {
  const dataLength = numSamples * 2;
  const blockAlign = WAV_NUM_CHANNELS * (WAV_BITS_PER_SAMPLE / 8);
  const byteRate = sampleRate * blockAlign;

  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataLength, true); // file length minus first 8 bytes
  writeAscii(view, 8, 'WAVE');

  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, WAV_NUM_CHANNELS, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, WAV_BITS_PER_SAMPLE, true);

  writeAscii(view, 36, 'data');
  view.setUint32(40, dataLength, true);
}

// Quantise Float32 into int16 at buffer[byteOffset], clamping to ±1 first (overflow would wrap to -32768 and click). Scale MUST be symmetric (×0x7FFF, -32768 never emitted): the daemon decodes via /0x8000, so symmetric encode is a uniform ~0.99997 gain that the spectrogram's log z-norm cancels, whereas asymmetric -1->-32768 attenuates only positive samples - a sign-dependent distortion z-norm cannot remove. The browser round-trip stays lossless: the decoder inverts with the matching /0x7FFF, and since -32768 never appears every decoded sample stays in [-1, 1]. byteOffset must be even for the Int16Array fast path (call sites use WAV_HEADER_BYTES + N*2).
export function quantiseFloat32ToInt16(
  buffer: ArrayBuffer,
  byteOffset: number,
  samples: Float32Array
): void {
  const n = samples.length;
  if (IS_LITTLE_ENDIAN) {
    // Math.round before assignment: raw Int16Array coerces via ToInt16 (truncate toward zero), losing a half-LSB and diverging from the DataView path.
    const out = new Int16Array(buffer, byteOffset, n);
    for (let i = 0; i < n; i++) {
      const s = samples[i];
      const clamped = s < -1 ? -1 : s > 1 ? 1 : s;
      out[i] = Math.round(clamped * 0x7fff);
    }
    return;
  }
  // Big-endian fallback, not reachable on any shipping web platform.
  const view = new DataView(buffer);
  let off = byteOffset;
  for (let i = 0; i < n; i++) {
    const s = samples[i];
    const clamped = s < -1 ? -1 : s > 1 ? 1 : s;
    const int16 = Math.round(clamped * 0x7fff);
    view.setInt16(off, int16, true);
    off += 2;
  }
}

// One-shot encode of a contiguous Float32 buffer already at sampleRate (no resample); long-form/stream paths instead use stream-encode helpers that fold resample + quantise into one allocation.
export function encodeWavPcm16(samples: Float32Array, sampleRate = WAV_SAMPLE_RATE): Blob {
  const buffer = new ArrayBuffer(WAV_HEADER_BYTES + samples.length * 2);
  const view = new DataView(buffer);
  writeWavHeader(view, samples.length, sampleRate);
  quantiseFloat32ToInt16(buffer, WAV_HEADER_BYTES, samples);
  return new Blob([buffer], { type: WAV_MIME });
}

function writeAscii(view: DataView, offset: number, s: string): void {
  for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
}
