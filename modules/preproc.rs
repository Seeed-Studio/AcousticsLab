//! TF `sc_preproc_model` parity: 44032 mono PCM @ 44100 Hz f32 `[-1,1]` -> 43x232
//! z-normalized log-magnitude spectrogram. Contract: frame=2048, hop=1024, 43
//! frames, virtual front-pad of 1024 zeros (frame 0 = pad + `samples[0..1024]`),
//! span exactly 44032, no trailing pad; *periodic* Blackman window (denominator
//! M=2048), NOT symmetric `numpy.blackman`; log runs over a
//! [`MAG_SQ_FLOOR`]-clamped magnitude-squared: finite PCM (all-zero included)
//! yields an all-finite plane, nonzero bins stay bit-identical to TF.

// Public so `training` can reach it as `preproc::wav_io::*` (shares the sinc-resampler params).
pub mod wav_io;

use crate::common::dims::{HopSamples, NBins, NFrames, WaveformLen};
use realfft::{RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex32;
use std::sync::Arc;

pub const FRAME_LEN: usize = 2048;
/// Frame 0 begins this many samples before the first real PCM sample.
pub const FRONT_PAD: usize = 1024;

/// Z-norm epsilon added to std NOT under the sqrt: `(x-mean)/(sqrt(var)+1e-4)`.
/// Exact official value; omitting it is the dominant per-pixel divergence source.
pub const Z_NORM_EPSILON: f32 = 1e-4;

/// Floor for the magnitude-squared entering the log. A clamp, NOT `+ eps`
/// (additive would perturb every bin and break byte-exact TF parity): only bins
/// below `MIN_POSITIVE` move (`|z| < ~1.1e-19`, unreachable from real audio),
/// i.e. exact-zero bins of all-zero FFT frames, landing on `0.5 * ln` of this
/// ~= -43.7. An all-silent window thus becomes a constant plane that
/// [`Z_NORM_EPSILON`] z-norms to a bounded constant (|v| < 1), never NaN.
/// `MIN_POSITIVE`, not the smallest subnormal, so flush-to-zero FP can't turn
/// the clamped value back into `ln(0)`. Applied via `<`, NOT `f32::max`: `max`
/// ignores NaN and would launder a NaN bin (non-finite PCM, preproc fault) into
/// exact silence, blinding every downstream non-finite fault gate.
pub const MAG_SQ_FLOOR: f32 = f32::MIN_POSITIVE;

/// Compile-time guard that the last frame stays inside the waveform (the no-pad path assumes it).
#[allow(dead_code)]
const FRAME_BOUNDS_INVARIANT: () = assert!(
    (NFrames::USIZE - 1) * HopSamples::USIZE - FRONT_PAD + FRAME_LEN <= WaveformLen::USIZE,
    "frame bounds violation: last pcm index would exceed WaveformLen",
);

/// Cached real-input FFT plan + window + per-call scratch. FFT emits
/// `FRAME_LEN/2 + 1 = 1025` bins, of which the first 232 are read. [`Clone`] shares
/// the [`Arc`] plan but gives each clone private scratch, for rayon parallel preproc.
pub struct Preproc {
    r2c: Arc<dyn RealToComplex<f32>>,
    window: [f32; FRAME_LEN],
    // Pre-allocated per-call buffers for an alloc-free hot path; `frame` lives on
    // the struct so its 8 KB zero-init amortizes away, fully rewritten before each FFT.
    frame: [f32; FRAME_LEN],
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl Preproc {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(FRAME_LEN);
        let spectrum = r2c.make_output_vec();
        let scratch = vec![Complex32::default(); r2c.get_scratch_len()];
        Self {
            r2c,
            window: load_bundled_window(),
            frame: [0.0; FRAME_LEN],
            spectrum,
            scratch,
        }
    }

    /// 43x232 row-major spectrogram, all-finite for any finite PCM (zero-magnitude
    /// bins clamp to [`MAG_SQ_FLOOR`]); allocates per call, the zero-alloc hot
    /// loop uses [`Self::spectrogram_into`].
    pub fn spectrogram(
        &mut self,
        pcm: &[f32; WaveformLen::USIZE],
    ) -> Box<[[f32; NBins::USIZE]; NFrames::USIZE]> {
        let mut out = Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
        self.spectrogram_into(pcm, &mut out);
        out
    }

    /// Spectrogram into a caller-reused buffer (streaming hot path); every cell of
    /// `out` is overwritten before being read, so prior state is irrelevant.
    pub fn spectrogram_into(
        &mut self,
        pcm: &[f32; WaveformLen::USIZE],
        out: &mut [[f32; NBins::USIZE]; NFrames::USIZE],
    ) {
        // Destructure for disjoint borrows in the FFT call (`&r2c` + `&mut` rest).
        let Self {
            r2c,
            window,
            frame,
            spectrum,
            scratch,
        } = self;

        for (t, row) in out.iter_mut().enumerate() {
            // hop==FRONT_PAD==1024 so only frame 0 straddles the pad; the case-split
            // drops a per-bin pad branch. Do NOT factor the multiply into a helper
            // "for SIMD": thin-LTO emits scalar `fmul` regardless (verified).
            let start = t * HopSamples::USIZE;
            if start < FRONT_PAD {
                let zeros = FRONT_PAD - start;
                let take = FRAME_LEN - zeros;
                // 0.0 fill is bit-identical to `0.0 * w` (window non-negative).
                frame[..zeros].fill(0.0);
                let pcm_head = &pcm[..take];
                let win_tail = &window[zeros..];
                for ((f, &p), &w) in frame[zeros..].iter_mut().zip(pcm_head).zip(win_tail) {
                    *f = p * w;
                }
            } else {
                let s = start - FRONT_PAD;
                let pcm_slice = &pcm[s..s + FRAME_LEN];
                for ((f, &p), &w) in frame.iter_mut().zip(pcm_slice).zip(window.iter()) {
                    *f = p * w;
                }
            }

            r2c.process_with_scratch(frame, spectrum, scratch)
                .expect("realfft buffer sizes are fixed at Preproc::new");

            // log|z| = 0.5*ln(|z|^2): one fewer sqrt/bin, *0.5 exact in fp32.
            // `<` not `max` so NaN falls through (see MAG_SQ_FLOOR).
            for (r, s) in row.iter_mut().zip(&spectrum[..NBins::USIZE]) {
                let m = s.norm_sqr();
                *r = 0.5 * (if m < MAG_SQ_FLOOR { MAG_SQ_FLOOR } else { m }).ln();
            }
        }

        // Z-normalize the plane with population variance (/N), matching TF `moments`.
        // LANES independent accumulators let the ~10k-term reductions vectorize; a
        // sequential `sum += v` cannot, since LLVM may not reassociate IEEE adds. Do
        // NOT fold these into `f32::algebraic_add`: it reassociates but drops the
        // NaN/Inf propagation the engine's fault gate relies on. Splitting by hand
        // also lands nearer TF -- LANES partials round independently.
        const LANES: usize = 8;
        let count = (NFrames::USIZE * NBins::USIZE) as f32;
        let (lanes, rest) = out.as_slice().as_flattened().as_chunks::<LANES>();

        let mut acc = [0.0f32; LANES];
        for c in lanes {
            for (a, &v) in acc.iter_mut().zip(c) {
                *a += v;
            }
        }
        let mut sum: f32 = rest.iter().sum();
        for a in acc {
            sum += a;
        }
        let mean = sum / count;

        let mut acc = [0.0f32; LANES];
        for c in lanes {
            for (a, &v) in acc.iter_mut().zip(c) {
                let d = v - mean;
                *a += d * d;
            }
        }
        let mut sq: f32 = rest
            .iter()
            .map(|&v| {
                let diff = v - mean;
                diff * diff
            })
            .sum();
        for a in acc {
            sq += a;
        }
        let std = (sq / count).sqrt() + Z_NORM_EPSILON;
        // No `std == 0` guard needed: a constant plane (all-zero PCM) yields the
        // bounded constant d/(|d| + eps), |v| < 1 -- Z_NORM_EPSILON alone suffices.
        for row in out.iter_mut() {
            for v in row.iter_mut() {
                *v = (*v - mean) / std;
            }
        }
    }
}

impl Default for Preproc {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Preproc {
    fn clone(&self) -> Self {
        Self {
            r2c: self.r2c.clone(),
            window: self.window,
            frame: [0.0; FRAME_LEN],
            spectrum: vec![Complex32::default(); self.spectrum.len()],
            scratch: vec![Complex32::default(); self.scratch.len()],
        }
    }
}

/// Periodic Blackman window bytes from `sc_preproc_model`'s graph, shipped verbatim
/// NOT recomputed: recompute drifts up to 1.19e-7/tap, which `ln` amplifies 10x+ on
/// the DC bin of near-zero-mean frames.
const WINDOW_BYTES: &[u8; 8192] = include_bytes!("preproc/window_blackman_2048.bin");

fn load_bundled_window() -> [f32; FRAME_LEN] {
    const _: () = assert!(
        WINDOW_BYTES.len() == FRAME_LEN * std::mem::size_of::<f32>(),
        "bundled window byte count must equal FRAME_LEN * sizeof(f32)",
    );
    let mut w = [0.0f32; FRAME_LEN];
    for (i, &chunk) in WINDOW_BYTES.as_chunks::<4>().0.iter().enumerate() {
        w[i] = f32::from_le_bytes(chunk);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_periodic_shape() {
        let w = load_bundled_window();
        // Periodic-Blackman taps: w[0]=0.42-0.5+0.08=0, w[N/2]=1, w[N/4]=0.34.
        assert!(w[0].abs() < 1e-6, "w[0] = {}", w[0]);
        assert!(
            (w[FRAME_LEN / 2] - 1.0).abs() < 1e-6,
            "w[N/2] = {}",
            w[FRAME_LEN / 2]
        );
        assert!(
            (w[FRAME_LEN / 4] - 0.34).abs() < 1e-4,
            "w[N/4] = {}",
            w[FRAME_LEN / 4]
        );
    }

    #[test]
    fn output_shape() {
        let pcm = Box::new([0.1f32; WaveformLen::USIZE]);
        let mut p = Preproc::new();
        let s = p.spectrogram(&pcm);
        assert_eq!(s.len(), NFrames::USIZE);
        assert_eq!(s[0].len(), NBins::USIZE);
    }

    /// All-zero PCM -> all-finite CONSTANT plane, |v| < 1: silence is
    /// first-class input. Guards against bare-`ln` (NaN plane) and against an
    /// additive epsilon (would un-constant the plane).
    #[test]
    fn silence_input_is_finite_bounded_constant() {
        let pcm = Box::new([0.0f32; WaveformLen::USIZE]);
        let mut p = Preproc::new();
        let s = p.spectrogram(&pcm);
        let first = s[0][0];
        for (t, row) in s.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "silence: expected finite at t={t} k={k}, got {v}",
                );
                assert!(
                    v.abs() < 1.0,
                    "silence: |v| must stay under 1 (z-norm epsilon bound) at t={t} k={k}, got {v}",
                );
                assert_eq!(
                    v.to_bits(),
                    first.to_bits(),
                    "silence: plane must be constant; t={t} k={k} has {v}, expected {first}",
                );
            }
        }
    }

    /// A single all-zero FFT frame must NOT poison the plane; locks both edges:
    /// silent frame 0 (front-pad) and silent frame 42 (trailing).
    #[test]
    fn silent_frame_keeps_plane_finite() {
        let mut p = Preproc::new();

        // Case A: only frame 0 silent (pcm[0..1024]=0).
        let mut pcm_a = Box::new([0.1f32; WaveformLen::USIZE]);
        for s in &mut pcm_a[..HopSamples::USIZE] {
            *s = 0.0;
        }
        let s_a = p.spectrogram(&pcm_a);
        for (t, row) in s_a.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "leading-silence frame: expected finite at t={t} k={k}, got {v}",
                );
            }
        }

        // Case B: only the last frame silent (frame 42 spans real [41984, 44032)).
        let mut pcm_b = Box::new([0.1f32; WaveformLen::USIZE]);
        let frame42_start = 42 * HopSamples::USIZE - FRONT_PAD;
        for s in &mut pcm_b[frame42_start..frame42_start + FRAME_LEN] {
            *s = 0.0;
        }
        let s_b = p.spectrogram(&pcm_b);
        for (t, row) in s_b.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "trailing-silence frame: expected finite at t={t} k={k}, got {v}",
                );
            }
        }
    }

    /// NaN PCM must still poison the plane -- an `f32::max`-shaped clamp would
    /// launder it into silence (max ignores NaN), blinding the fault gates.
    #[test]
    fn nan_pcm_propagates_to_nonfinite_plane() {
        let mut p = Preproc::new();
        let mut pcm = Box::new([0.1f32; WaveformLen::USIZE]);
        pcm[WaveformLen::USIZE / 2] = f32::NAN;
        let s = p.spectrogram(&pcm);
        assert!(
            s.as_slice().as_flattened().iter().any(|v| !v.is_finite()),
            "a NaN PCM sample produced an all-finite plane; the clamp is \
             laundering non-finite bins and the fault gates downstream are blind",
        );
    }

    /// Non-silent audio still yields a finite, NON-constant plane; bit-exactness
    /// vs the TF references is locked separately by `tests/preproc_parity.rs`.
    #[test]
    fn clamp_leaves_nonsilent_plane_nonconstant() {
        let mut p = Preproc::new();
        // Deterministic broadband-ish content, no RNG dependency.
        let mut pcm = Box::new([0.0f32; WaveformLen::USIZE]);
        for (i, s) in pcm.iter_mut().enumerate() {
            let x = i as f32;
            *s = 0.3 * (0.031 * x).sin() + 0.2 * (0.173 * x).sin() + 0.05 * (0.719 * x).sin();
        }
        let s = p.spectrogram(&pcm);
        for (t, row) in s.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert!(v.is_finite(), "expected finite at t={t} k={k}, got {v}");
            }
            let (min, max) = row
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            assert!(
                max > min,
                "row {t} collapsed to a constant; clamp must not flatten real audio",
            );
        }
    }
}
