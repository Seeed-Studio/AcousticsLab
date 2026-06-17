import { UploadPool } from '$lib/api/upload';
import { fftRadix2, hannWindow } from './fft';
import { buildSpectrogramLut, magnitudeToPaletteIndex, type SpectrogramTheme } from './palette';
import { decodeCanonicalWavSync } from './wav-decode';
import { getSliceBlob } from './slice-fetch';
import { getSpectrogramRecord, putSpectrogramRecord } from '$lib/idb/spectrograms';
import type { SliceRecord } from '$lib/idb/db';

// Content+theme-addressed (sha256 of WAV bytes x palette mode): PNG is a deterministic function of
// (hash, mode), never invalidated and never evicted (a deleted slice's row may be shared; last-ref
// needs a cross-store join). Cache only grows (~3-4 KB/pair, under origin quota and Chrome ~10k /
// Safari ~1k blob: URL caps); resetDB is the only reset.

const FFT_SIZE = 512;
const HOP_SIZE = 256;
const FREQ_BINS = FFT_SIZE / 2 + 1;

const CARD_WIDTH = 96;
const CARD_HEIGHT = 64;

const MAX_CONCURRENT_SPECTROGRAMS = 3;
const generatePool = new UploadPool(MAX_CONCURRENT_SPECTROGRAMS);

const HANN_512 = hannWindow(FFT_SIZE);
const PALETTE_N = 256;
// Both LUTs built eagerly at module load so the per-pixel loop indexes a ready Uint8ClampedArray;
// cost is client-only since the layout's ssr: false means SSG never evaluates this module.
const PALETTES: Record<SpectrogramTheme, Uint8ClampedArray> = {
  light: buildSpectrogramLut(PALETTE_N, 'light'),
  dark: buildSpectrogramLut(PALETTE_N, 'dark')
};

// urlCache/inflight key; pipe cannot collide with hex-only sha256. IDB instead keys rows on bare
// sha256 within a per-theme store.
function cacheKey(sha: string, theme: SpectrogramTheme): string {
  return `${sha}|${theme}`;
}

// blob: URLs live for the tab's lifetime (bytes persist across sessions via IDB); inflight dedups
// concurrent renders of one key.
const urlCache = new Map<string, string>();
const inflight = new Map<string, Promise<string>>();

export async function getSliceSpectrogramUrl(
  slice: SliceRecord,
  theme: SpectrogramTheme
): Promise<string> {
  const sha = slice.id;
  const key = cacheKey(sha, theme);
  const memUrl = urlCache.get(key);
  if (memUrl !== undefined) return memUrl;
  const pending = inflight.get(key);
  if (pending) return pending;

  const work = (async (): Promise<string> => {
    try {
      const cached = await getSpectrogramRecord(sha, theme).catch(() => undefined);
      const png = cached?.png ?? (await generatePool.submit(() => generatePng(slice, theme)));
      if (!cached) {
        // Best-effort persist; failure (e.g. origin quota) just re-renders next session.
        await putSpectrogramRecord(
          {
            sha256: sha,
            png,
            created_at: new Date().toISOString()
          },
          theme
        ).catch(() => undefined);
      }
      const url = URL.createObjectURL(png);
      urlCache.set(key, url);
      return url;
    } finally {
      inflight.delete(key);
    }
  })();
  inflight.set(key, work);
  return work;
}

async function generatePng(slice: SliceRecord, theme: SpectrogramTheme): Promise<Blob> {
  const sourceBlob = await getSliceBlob(slice);
  const buf = await sourceBlob.arrayBuffer();
  const { pcm } = decodeCanonicalWavSync(buf);

  const frames = Math.max(1, Math.floor((pcm.length - FFT_SIZE) / HOP_SIZE) + 1);
  const magnitudes = new Float32Array(frames * FREQ_BINS);

  const real = new Float32Array(FFT_SIZE);
  const imag = new Float32Array(FFT_SIZE);
  const normalise = FFT_SIZE / 2;

  for (let f = 0; f < frames; f++) {
    const start = f * HOP_SIZE;
    // Split the window at the PCM boundary so in-bounds reads stay monomorphic (a per-element
    // `pcm[start+i] ?? 0` is possibly-undefined and deopts the packed-Float32 path); the zero-padded
    // tail, with frames clamped >=1, also stops a sub-FFT_SIZE slice from NaN-poisoning the FFT.
    const windowed = Math.max(0, Math.min(FFT_SIZE, pcm.length - start));
    for (let i = 0; i < windowed; i++) real[i] = pcm[start + i] * HANN_512[i];
    for (let i = windowed; i < FFT_SIZE; i++) real[i] = 0;
    imag.fill(0);
    fftRadix2(real, imag);
    for (let k = 0; k < FREQ_BINS; k++) {
      const re = real[k];
      const im = imag[k];
      magnitudes[f * FREQ_BINS + k] = Math.sqrt(re * re + im * im) / normalise;
    }
  }

  if (typeof OffscreenCanvas === 'undefined') {
    throw new Error('OffscreenCanvas is unavailable in this browser.');
  }
  const canvas = new OffscreenCanvas(CARD_WIDTH, CARD_HEIGHT);
  const ctx = canvas.getContext('2d', { alpha: false });
  if (!ctx) throw new Error('Failed to acquire OffscreenCanvas 2D context.');

  const imageData = ctx.createImageData(CARD_WIDTH, CARD_HEIGHT);
  const pixels = imageData.data;
  const palette = PALETTES[theme];

  for (let y = 0; y < CARD_HEIGHT; y++) {
    const freqIdx = Math.min(
      FREQ_BINS - 1,
      Math.floor((1 - y / (CARD_HEIGHT - 1)) * (FREQ_BINS - 1))
    );
    for (let x = 0; x < CARD_WIDTH; x++) {
      const frameIdx = Math.min(frames - 1, Math.floor((x / CARD_WIDTH) * frames));
      const pi = magnitudeToPaletteIndex(magnitudes[frameIdx * FREQ_BINS + freqIdx], PALETTE_N);
      const src = pi * 3;
      const p = (y * CARD_WIDTH + x) * 4;
      pixels[p] = palette[src];
      pixels[p + 1] = palette[src + 1];
      pixels[p + 2] = palette[src + 2];
      pixels[p + 3] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}
