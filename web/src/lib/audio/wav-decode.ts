// Canonical mono-PCM-16 drafts decode straight to Float32Array, bypassing decodeAudioData whose
// context-rate resample (~48 kHz on macOS) breaks alignment with the WAV_SAMPLE_RATE slice grid;
// other rate/depth/channel imports still use the browser decoder.

import { WAV_SAMPLE_RATE } from './wav';
import { m } from '$lib/i18n';

export interface WavMagicResult {
  valid: boolean;
  reason?: string;
}

// Pre-check magic (bytes 0-3 "RIFF", 8-11 "WAVE") so non-WAV imports fail with "not a WAV file"
// instead of decodeAudioData's generic "Unable to decode audio data".
export function verifyWavMagic(header: ArrayBuffer): WavMagicResult {
  if (header.byteLength < 12) {
    return {
      valid: false,
      reason: m.category.input_pane.error_wav_too_small_for_header
    };
  }
  const view = new DataView(header, 0, 12);
  const ascii = (offset: number, length: number): string => {
    let s = '';
    for (let i = 0; i < length; i++) s += String.fromCharCode(view.getUint8(offset + i));
    return s;
  };
  // Accept RIFX (big-endian) since decodeAudioData handles both.
  const riff = ascii(0, 4);
  if (riff !== 'RIFF' && riff !== 'RIFX') {
    return {
      valid: false,
      reason: m.category.input_pane.error_wav_missing_riff
    };
  }
  if (ascii(8, 4) !== 'WAVE') {
    return {
      valid: false,
      reason: m.category.input_pane.error_wav_missing_wave
    };
  }
  return { valid: true };
}

// Blob.slice is lazy: reads the 12-byte header without loading the whole body.
export async function readWavMagic(blob: Blob): Promise<WavMagicResult> {
  if (blob.size < 12) {
    return { valid: false, reason: m.category.input_pane.error_wav_empty };
  }
  const header = await blob.slice(0, 12).arrayBuffer();
  return verifyWavMagic(header);
}

export interface DecodedWav {
  pcm: Float32Array;
  sampleRate: number;
}

// Rate read from header (offset 24), not WAV_SAMPLE_RATE, so a canonical-rate change flows through.
// Divide by 0x7FFF (not 0x8000) to invert the encoder's symmetric *0x7FFF: most-negative code -32767
// maps to exactly -1.0, keeping samples in [-1, 1] (daemon's /0x8000 = uniform ~0.99997 gain z-norm removes).
export function decodeCanonicalWavSync(buf: ArrayBuffer): DecodedWav {
  if (buf.byteLength < 44) {
    throw new Error(m.category.input_pane.error_wav_buffer_too_small);
  }
  const view = new DataView(buf);
  // Assert canonical layout; a stereo/24-bit WAV or a LIST/fact chunk shifting `data` off offset 44
  // would otherwise decode as garbage.
  const audioFormat = view.getUint16(20, true);
  const numChannels = view.getUint16(22, true);
  const bitsPerSample = view.getUint16(34, true);
  const dataTag = String.fromCharCode(
    view.getUint8(36),
    view.getUint8(37),
    view.getUint8(38),
    view.getUint8(39)
  );
  if (audioFormat !== 1 || numChannels !== 1 || bitsPerSample !== 16 || dataTag !== 'data') {
    throw new Error(
      `decodeCanonicalWavSync: non-canonical WAV (format=${audioFormat}, channels=${numChannels}, bits=${bitsPerSample}, dataTag=${JSON.stringify(dataTag)}); expected mono PCM-16 with 'data' at offset 36`
    );
  }
  const sampleRate = view.getUint32(24, true);
  const sampleCount = (buf.byteLength - 44) >> 1;
  const pcm = new Float32Array(sampleCount);
  let offset = 44;
  for (let i = 0; i < sampleCount; i++) {
    const int16 = view.getInt16(offset, true);
    pcm[i] = int16 / 0x7fff;
    offset += 2;
  }
  return { pcm, sampleRate };
}

export async function decodeCanonicalWav(blob: Blob): Promise<DecodedWav> {
  return decodeCanonicalWavSync(await blob.arrayBuffer());
}

export { WAV_SAMPLE_RATE };
