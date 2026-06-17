// Exact inverse of the daemon's NaN-drop filter (drops a slice with non-finite log-magnitude
// spectrogram; >10% drops fails the job). For mono PCM-i16 WAVs the only reachable trigger is an
// all-zero windowed FFT frame (log(0)=-inf -> NaN after z-norm), iff every sample quantizes to
// int16 0. INVARIANT: bundled Blackman window must have no exact-zero taps (w[0]=1.49e-8, rest
// >=8.6e-7); a zero tap (e.g. symmetric numpy.blackman) would let a non-zero PCM sample slip past
// this window-based check and requires revisiting.

// Must match the daemon's preproc framing byte-for-byte, else this detector silently disagrees.
export const PREPROC_FRAME_LEN = 2048;
export const PREPROC_HOP = 1024;
export const PREPROC_N_FRAMES = 43;
// Daemon truncates the trailing 68 of the 44_100-sample (1 s @ 44.1 kHz) slice before the spectrogram.
export const PREPROC_WAVEFORM_LEN = 44_032;
// Non-overlapping 1024-sample windows tiling the effective input (= WAVEFORM_LEN / HOP).
export const PREPROC_N_WINDOWS = 43;

// Compile-time pins: a typo trips type narrowing before prod; void suppresses no-unused-vars.
const _PREPROC_FRAME_OK: 2048 = PREPROC_FRAME_LEN;
const _PREPROC_HOP_OK: 1024 = PREPROC_HOP;
const _PREPROC_N_FRAMES_OK: 43 = PREPROC_N_FRAMES;
const _PREPROC_WAVEFORM_OK: 44_032 = PREPROC_WAVEFORM_LEN;
const _PREPROC_N_WINDOWS_OK: 43 = PREPROC_N_WINDOWS;
void _PREPROC_FRAME_OK;
void _PREPROC_HOP_OK;
void _PREPROC_N_FRAMES_OK;
void _PREPROC_WAVEFORM_OK;
void _PREPROC_N_WINDOWS_OK;

// Catches WAVEFORM_LEN/HOP/N_WINDOWS drift the literal pins can't (arithmetic isn't type-level);
// throws at load so the page errors rather than silently classifying every slice as non-silent.
if (PREPROC_N_WINDOWS * PREPROC_HOP !== PREPROC_WAVEFORM_LEN) {
  throw new Error(
    `silence framing invariant broken: N_WINDOWS(${PREPROC_N_WINDOWS}) * HOP(${PREPROC_HOP}) !== WAVEFORM_LEN(${PREPROC_WAVEFORM_LEN})`
  );
}

// Mirrors the WAV encoder's exact int16-0 band (scale 0x7FFF then Math.round, ties toward +Inf),
// an asymmetric [-0.5/32767, 0.5/32767), not one absolute threshold.
function quantizesToZero(s: number): boolean {
  // NaN encodes to int16 0; unreachable for finite FE PCM but stays encoder-aligned.
  if (Number.isNaN(s)) return true;
  // Encoder clamps to [-1, 1] before rounding; +/-Inf -> +/-1 -> +/-32767, non-zero.
  const clamped = s < -1 ? -1 : s > 1 ? 1 : s;
  return Math.round(clamped * 0x7fff) === 0;
}

// Samples past pcm.length count as zero, matching the daemon's zero-pad of short inputs.
function isWindowSilent(pcm: Float32Array, start: number, end: number): boolean {
  const lo = Math.max(0, start);
  const hi = Math.min(end, pcm.length);
  for (let i = lo; i < hi; i++) {
    if (!quantizesToZero(pcm[i])) return false;
  }
  return true;
}

// True iff the daemon would emit a NaN spectrogram for pcm[offset .. offset+WAVEFORM_LEN]. Frame
// 0 reads only window 0 (leading half zero-padded); frame i>=1 reads windows i-1 and i; so reduces
// to: window 0 silent OR any two adjacent windows both silent.
export function wouldNanAtPreproc(pcm: Float32Array, offset = 0): boolean {
  let prevSilent = false;
  for (let k = 0; k < PREPROC_N_WINDOWS; k++) {
    const start = offset + k * PREPROC_HOP;
    const end = start + PREPROC_HOP;
    const curSilent = isWindowSilent(pcm, start, end);
    if (k === 0) {
      if (curSilent) return true;
    } else if (prevSilent && curSilent) {
      return true;
    }
    prevSilent = curSilent;
  }
  return false;
}
