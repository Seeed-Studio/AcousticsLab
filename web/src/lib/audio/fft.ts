// Hand-rolled FFT instead of AnalyserNode, whose AudioContext stays suspended
// until a user gesture (blank spectrogram until click). Callers pre-window the
// input and zero `imag` per transform.

/** In-place radix-2 forward FFT on (real, imag). Length must be a power of two;
 *  only real.length is checked, so imag must match it (else silent OOB reads). */
export function fftRadix2(real: Float32Array, imag: Float32Array): void {
  const n = real.length;
  if ((n & (n - 1)) !== 0) {
    throw new Error(`fftRadix2: length must be a power of two, got ${n}`);
  }

  // Bit-reversal permutation so the iterative butterflies pair correctly.
  for (let i = 1, j = 0; i < n; ++i) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      const tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      const ti = imag[i];
      imag[i] = imag[j];
      imag[j] = ti;
    }
  }

  // Iterative Cooley-Tukey; twiddle advanced by complex multiply to avoid per-step trig.
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (-2 * Math.PI) / len;
    const wReal = Math.cos(ang);
    const wImag = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let rotR = 1;
      let rotI = 0;
      for (let k = 0; k < half; ++k) {
        const iK = i + k;
        const iKHalf = i + k + half;
        const aR = real[iK];
        const aI = imag[iK];
        const bR0 = real[iKHalf];
        const bI0 = imag[iKHalf];
        const bR = bR0 * rotR - bI0 * rotI;
        const bI = bR0 * rotI + bI0 * rotR;
        real[iK] = aR + bR;
        imag[iK] = aI + bI;
        real[iKHalf] = aR - bR;
        imag[iKHalf] = aI - bI;
        const nrotR = rotR * wReal - rotI * wImag;
        const nrotI = rotR * wImag + rotI * wReal;
        rotR = nrotR;
        rotI = nrotI;
      }
    }
  }
}

/** Precomputed Hann analysis window of length n (low spectral leakage). */
export function hannWindow(n: number): Float32Array {
  const w = new Float32Array(n);
  if (n === 1) {
    w[0] = 1;
    return w;
  }
  const denom = n - 1;
  for (let i = 0; i < n; ++i) {
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / denom);
  }
  return w;
}
