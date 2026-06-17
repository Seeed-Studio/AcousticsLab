// Normalizes arbitrary audio to mono Float32 at the canonical rate (matching the daemon's training
// preprocessor) via OfflineAudioContext, averaging all channels so stereo phone-recording right-channel
// content survives. Encoders fold resample+quantise+blob-wrap into one in-place allocation pass (no
// intermediate copies); its single-pass peak (~1.07 GiB at the 50 min @ 48 kHz cap vs ~2.65 GiB if all
// stages coexisted) is what the recording duration cap is sized against.

import {
  WAV_HEADER_BYTES,
  WAV_MIME,
  WAV_SAMPLE_RATE,
  encodeWavPcm16,
  quantiseFloat32ToInt16,
  writeWavHeader
} from './wav';
import { m } from '$lib/i18n';

export function downmixToMono(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) {
    // slice copies the internal storage so the result outlives the AudioBuffer.
    return buffer.getChannelData(0).slice();
  }
  const n = buffer.length;
  const out = new Float32Array(n);
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < n; i++) out[i] += data[i];
  }
  const inv = 1 / buffer.numberOfChannels;
  for (let i = 0; i < n; i++) out[i] *= inv;
  return out;
}

// Decodes an arbitrary-format file to mono Float32 at the browser's native rate, returning that rate
// so the caller can re-resample to the canonical target (Chrome's decode-time resampling is fine
// since OAC re-resamples deterministically). Closes the context eagerly to free its device thread.
export async function decodeAudioFile(
  blob: Blob
): Promise<{ pcm: Float32Array; sampleRate: number }> {
  const arrayBuffer = await blob.arrayBuffer();
  const Ctor =
    typeof AudioContext !== 'undefined'
      ? AudioContext
      : (globalThis as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) {
    throw new Error(m.category.input_pane.error_web_audio_unavailable);
  }
  const ctx = new Ctor();
  try {
    // No resume(): decodeAudioData needs no output node and runs on the decoder thread.
    const decoded = await ctx.decodeAudioData(arrayBuffer);
    return {
      pcm: downmixToMono(decoded),
      sampleRate: decoded.sampleRate
    };
  } finally {
    // close() rejects in some old Safari builds; the context is done either way.
    try {
      await ctx.close();
    } catch {
      /* ignore */
    }
  }
}

// outputSamples rides alongside the blob so callers derive durationMs without re-measuring.
export interface EncodedWav {
  blob: Blob;
  outputSamples: number;
}

// Not mutated; caller retains ownership of `samples`.
export async function encodeWavFromFloat32(
  samples: Float32Array,
  inputRate: number,
  outputRate: number = WAV_SAMPLE_RATE
): Promise<EncodedWav> {
  if (samples.length === 0) return emptyWav(outputRate);
  if (inputRate === outputRate) {
    return {
      blob: encodeWavPcm16(samples, outputRate),
      outputSamples: samples.length
    };
  }
  const offline = makeOfflineContext(samples.length, inputRate, outputRate);
  const sourceBuf = offline.createBuffer(1, samples.length, inputRate);
  // set() into the storage view is copyToChannel's memcpy without its DOM-call overhead or cast.
  sourceBuf.getChannelData(0).set(samples);
  return renderAndQuantise(offline, sourceBuf, outputRate);
}

// `chunks` is moved-from (each entry nulled as folded in so its ArrayBuffer GCs mid-loop, not pinned
// through finalize); caller MUST drop its reference afterward and `totalSamples` MUST sum chunk lengths.
export async function encodeWavFromChunks(
  chunks: Float32Array[],
  totalSamples: number,
  inputRate: number,
  outputRate: number = WAV_SAMPLE_RATE
): Promise<EncodedWav> {
  if (totalSamples === 0) return emptyWav(outputRate);
  if (inputRate === outputRate) {
    return {
      blob: encodeMonoChunksToWav(chunks, totalSamples, outputRate),
      outputSamples: totalSamples
    };
  }
  const offline = makeOfflineContext(totalSamples, inputRate, outputRate);
  const sourceBuf = offline.createBuffer(1, totalSamples, inputRate);
  const dest = sourceBuf.getChannelData(0);
  // Null each folded slot to GC its ArrayBuffer during the loop, else 50 min of captured Float32
  // (~550 MiB at 48 kHz) stays pinned across the render atop the source buffer; safe since moved-from.
  const slots = chunks as (Float32Array | null)[];
  let off = 0;
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    dest.set(chunk, off);
    off += chunk.length;
    slots[i] = null;
  }
  return renderAndQuantise(offline, sourceBuf, outputRate);
}

// A BufferSource node is OAC's only render path, hence the connect/start dance before quantise.
async function renderAndQuantise(
  offline: OfflineAudioContext,
  sourceBuf: AudioBuffer,
  outputRate: number
): Promise<EncodedWav> {
  const node = offline.createBufferSource();
  node.buffer = sourceBuf;
  node.connect(offline.destination);
  node.start();
  const rendered = await offline.startRendering();
  // Storage view, not a copy: encodeWavPcm16 reads it straight into the WAV; AudioBuffer GCs on exit.
  const samples = rendered.getChannelData(0);
  return {
    blob: encodeWavPcm16(samples, outputRate),
    outputSamples: samples.length
  };
}

// ceil (not floor) keeps the fractional final sample (resampler silence-pads the trailing slot);
// the >= 1 clamp guards tiny inputs whose length would round to zero, which OAC rejects.
function makeOfflineContext(
  inputSamples: number,
  inputRate: number,
  outputRate: number
): OfflineAudioContext {
  const outputLength = Math.max(1, Math.ceil((inputSamples * outputRate) / inputRate));
  return new OfflineAudioContext(1, outputLength, outputRate);
}

// Rate-match fast path skipping the OAC round-trip.
function encodeMonoChunksToWav(
  chunks: Float32Array[],
  totalSamples: number,
  sampleRate: number
): Blob {
  const buffer = new ArrayBuffer(WAV_HEADER_BYTES + totalSamples * 2);
  const view = new DataView(buffer);
  writeWavHeader(view, totalSamples, sampleRate);
  // Null each folded slot to release captured frames during the encode (see encodeWavFromChunks).
  const slots = chunks as (Float32Array | null)[];
  let byteOff = WAV_HEADER_BYTES;
  for (let c = 0; c < chunks.length; c++) {
    const chunk = chunks[c];
    quantiseFloat32ToInt16(buffer, byteOff, chunk);
    byteOff += chunk.length * 2;
    slots[c] = null;
  }
  return new Blob([buffer], { type: WAV_MIME });
}

// Valid WAV with a zero-length data section; defensive end-stop since callers already guard empty input.
function emptyWav(sampleRate: number): EncodedWav {
  const buffer = new ArrayBuffer(WAV_HEADER_BYTES);
  const view = new DataView(buffer);
  writeWavHeader(view, 0, sampleRate);
  return {
    blob: new Blob([buffer], { type: WAV_MIME }),
    outputSamples: 0
  };
}
