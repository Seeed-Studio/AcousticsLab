//! Streaming sinc resampler and single source of truth for sinc params
//! (re-exported by `preproc::wav_io`): train and serve must share these constants
//! or top-1 accuracy drifts silently. [`Streaming`] builds the resampler once and
//! feeds it contiguously, never resetting mid-stream (reset zeroes FIR history =
//! ~256-sample click), buffering partial input until a full chunk is available.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

/// Failure surface for [`Streaming::process`], wrapping upstream rubato types.
#[derive(Debug, thiserror::Error)]
pub enum StreamingResampleError {
    #[error("rubato input adapter: {source}")]
    InputAdapter {
        #[source]
        source: audioadapter_buffers::SizeError,
    },
    #[error("rubato output adapter: {source}")]
    OutputAdapter {
        #[source]
        source: audioadapter_buffers::SizeError,
    },
    /// Typically a `chunk_size`-cache invariant violation from a refactor.
    #[error("rubato process_into_buffer: {source}")]
    ProcessInto {
        #[source]
        source: rubato::ResampleError,
    },
}

impl crate::common::error::Categorized for StreamingResampleError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        // Always invariant violations, never operator input.
        crate::common::error::ErrorKind::Internal
    }
}

/// Sinc-async resampler with a fixed 1024-frame input chunk.
pub type SincResampler = Async<f32>;

/// Sinc-polyphase resampler matched to `scipy.signal.resample_poly` on speech.
/// Do NOT swap in [`rubato::Fft`]: it differs from scipy by up to 0.9 on a
/// +/-1.0 signal and can flip borderline top-1. Re-exported by `preproc::wav_io`,
/// so these constants are shared train/serve-wide.
pub fn sinc_resampler(from_sr: u32, to_sr: u32) -> SincResampler {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        // 0.95 Nyquist (~5% anti-alias guard-band, audible band un-attenuated) per
        // the scipy reference the classifier trained against.
        f_cutoff: Some(0.95),
        interpolation: SincInterpolationType::Cubic,
        // Polyphase depth; 512 = scipy high-quality default. Halving causes audible
        // imaging on borderline ratios.
        oversampling_factor: 512,
        window: WindowFunction::BlackmanHarris2,
    };
    Async::<f32>::new_sinc(
        to_sr as f64 / from_sr as f64,
        2.0,
        &params,
        1024,
        1,
        FixedAsync::Input,
    )
    .expect("resampler init")
}

/// Continuous-mode wrapper around [`SincResampler`] that never resets mid-stream.
pub struct Streaming {
    resampler: SincResampler,
    /// Partial input awaiting a full `input_frames_next()` chunk.
    accum: Vec<f32>,
    out: Vec<f32>,
    /// INVARIANT: equals `resampler.input_frames_next()` (constant under
    /// FixedAsync::Input, where it tracks `chunk_size` independent of the ratio); only
    /// `set_chunk_size` would break it, and this wrapper never calls it.
    chunk_size: usize,
    /// Reusable scratch sized to `output_frames_max()` (bound over all ratios within
    /// `max_relative_ratio` 2.0) to keep the hot path alloc-free.
    out_scratch: Vec<f32>,
}

impl Streaming {
    pub fn new(from_sr: u32, to_sr: u32) -> Self {
        let resampler = sinc_resampler(from_sr, to_sr);
        let chunk_size = resampler.input_frames_next();
        let out_scratch = vec![0.0f32; resampler.output_frames_max()];
        Self {
            resampler,
            // 2x chunk_size holds one chunk plus <=chunk_size-1 carry-over, so steady
            // state never reallocates.
            accum: Vec::with_capacity(chunk_size * 2),
            out: Vec::new(),
            chunk_size,
            out_scratch,
        }
    }

    /// Feed `input`, returning the count of new output samples (drain via
    /// [`Self::drain_output_into`]). Consumed prefixes coalesce into one end-of-call
    /// `accum.drain`; per-chunk drain would be O(K^2) tail shifts on a multi-period buffer.
    pub fn process(&mut self, input: &[f32]) -> Result<usize, StreamingResampleError> {
        if input.is_empty() {
            return Ok(0);
        }
        debug_assert_eq!(
            self.chunk_size,
            self.resampler.input_frames_next(),
            "chunk_size cache out of sync with resampler",
        );
        self.accum.extend_from_slice(input);
        let mut produced = 0;
        let mut consumed = 0;
        while consumed + self.chunk_size <= self.accum.len() {
            let in_adapter = InterleavedSlice::new(
                &self.accum[consumed..consumed + self.chunk_size],
                1,
                self.chunk_size,
            )
            .map_err(|source| StreamingResampleError::InputAdapter { source })?;
            let max_out = self.out_scratch.len();
            let mut out_adapter = InterleavedSlice::new_mut(&mut self.out_scratch, 1, max_out)
                .map_err(|source| StreamingResampleError::OutputAdapter { source })?;
            let (n_in, n_out) = self
                .resampler
                .process_into_buffer(&in_adapter, &mut out_adapter, None)
                .map_err(|source| StreamingResampleError::ProcessInto { source })?;
            debug_assert_eq!(n_in, self.chunk_size);
            self.out.extend_from_slice(&self.out_scratch[..n_out]);
            produced += n_out;
            consumed += self.chunk_size;
        }
        if consumed > 0 {
            self.accum.drain(..consumed);
        }
        Ok(produced)
    }

    /// Drain pending output into `sink`, preserving capacity (steady state stops allocating).
    pub fn drain_output_into(&mut self, sink: &mut Vec<f32>) {
        sink.append(&mut self.out);
    }

    /// Discard pending output. The mic arbitrator `process`-es every whitelisted
    /// channel each period (keeping FIR state current for a glitch-free switch) but
    /// writes only the active one, so non-active `out` Vecs must be dropped or they
    /// grow unbounded.
    pub fn drop_output(&mut self) {
        self.out.clear();
    }

    /// Drain pending output into a fresh [`Vec`]; hot paths prefer
    /// [`Self::drain_output_into`] (no allocation).
    pub fn take_output(&mut self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.out.len());
        out.append(&mut self.out);
        out
    }

    pub fn pending(&self) -> usize {
        self.out.len()
    }

    /// Clears FIR history + accumulators. Call ONLY on a real upstream discontinuity:
    /// on a continuous stream the reset injects a ~256-sample (the zeroed `sinc_len`
    /// FIR taps, ~6ms @44.1k) click, but legitimate callers (seek-after-`Lagged`, xrun
    /// FIR reset) are already absorbing a glitch.
    pub fn reset_after_discontinuity(&mut self) {
        self.accum.clear();
        self.out.clear();
        self.resampler.reset();
    }
}

impl std::fmt::Debug for Streaming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Streaming")
            .field("chunk_size", &self.chunk_size)
            .field("accum_len", &self.accum.len())
            .field("out_len", &self.out.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the upstream fixed chunk size that streaming logic assumes.
    #[test]
    fn upstream_chunk_size_is_1024() {
        let s = Streaming::new(48_000, 44_100);
        assert_eq!(s.chunk_size, 1024);
    }

    #[test]
    fn process_under_chunk_buffers_no_output() {
        let mut s = Streaming::new(48_000, 44_100);
        let input = vec![0.5_f32; 512];
        let produced = s.process(&input).expect("test resampler invariant");
        assert_eq!(produced, 0);
        assert_eq!(s.pending(), 0);
        assert_eq!(s.take_output(), Vec::<f32>::new());
    }

    #[test]
    fn process_one_chunk_produces_output() {
        let mut s = Streaming::new(48_000, 44_100);
        let input = vec![0.0_f32; 1024];
        let produced = s.process(&input).expect("test resampler invariant");
        assert!(
            produced > 0 && produced < 1024,
            "produced {produced}; expected 0 < produced < 1024 (downsampling)",
        );
        assert_eq!(s.pending(), produced);
    }

    #[test]
    fn streaming_total_output_matches_ratio() {
        let mut s = Streaming::new(48_000, 44_100);
        let total_in: usize = 48_000;
        let mut produced = 0;
        for chunk_start in (0..total_in).step_by(700) {
            let end = (chunk_start + 700).min(total_in);
            let input = vec![0.0_f32; end - chunk_start];
            produced += s.process(&input).expect("test resampler invariant");
        }
        // 46 full 1024-chunks (47104) * 44100/48000; +/-200 absorbs the ~120-sample
        // startup transient and rubato version drift.
        let expected_ideal = (47_104.0_f64 * 0.918_75) as usize;
        let lower = expected_ideal - 200;
        let upper = expected_ideal + 200;
        assert!(
            (lower..=upper).contains(&produced),
            "produced {produced}; expected ~{expected_ideal} (+/-200) for \
             48k to 44.1k of 48000 in",
        );
    }

    /// Identity rate (44.1->44.1): output approximates input, offset search
    /// absorbing the fixed ~128-sample sinc group delay.
    #[test]
    fn identity_rate_round_trips_pattern_within_tolerance() {
        let mut s = Streaming::new(44_100, 44_100);
        let input: Vec<f32> = (0..2048).map(|i| i as f32 / 2048.0).collect();
        s.process(&input).expect("test resampler invariant");
        let out = s.take_output();
        assert!(
            (1900..=2050).contains(&out.len()),
            "identity ratio produced {} samples for 2048 in",
            out.len(),
        );
        // Search +/-256 (the sinc filter length) for the best-fit offset.
        let win = 700..900;
        let mid_in = &input[win.clone()];
        let best = (-256i32..=256)
            .filter_map(|off| {
                let mut max_d = 0f32;
                for (i, &a) in mid_in.iter().enumerate() {
                    let j = (win.start as i32 + i as i32) + off;
                    if j < 0 || (j as usize) >= out.len() {
                        return None;
                    }
                    max_d = max_d.max((a - out[j as usize]).abs());
                }
                Some((off, max_d))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .expect("at least one offset must be in-bounds");
        assert!(
            best.1 < 0.05,
            "identity ratio: best-fit offset {} max|Δ| = {} (>= 0.05)",
            best.0,
            best.1,
        );
    }

    #[test]
    fn reset_after_discontinuity_clears_buffers() {
        let mut s = Streaming::new(48_000, 44_100);
        s.process(&vec![1.0; 1024])
            .expect("test resampler invariant");
        assert!(s.pending() > 0);
        s.reset_after_discontinuity();
        assert_eq!(s.pending(), 0);
        s.process(&vec![0.5; 1024])
            .expect("test resampler invariant");
        assert!(s.pending() > 0);
    }

    /// `drop_output` clears pending output but preserves FIR state (no transient on
    /// the next feed) - the property the mic arbitrator relies on.
    #[test]
    fn drop_output_preserves_fir_state() {
        let mut canonical = Streaming::new(48_000, 44_100);
        let mut dropped = Streaming::new(48_000, 44_100);

        let mut canonical_out: Vec<f32> = Vec::new();
        for _ in 0..4 {
            let input: Vec<f32> = (0..1024)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48_000.0).sin())
                .collect();
            canonical.process(&input).expect("test resampler invariant");
            dropped.process(&input).expect("test resampler invariant");
            canonical.drain_output_into(&mut canonical_out);
            dropped.drop_output();
        }

        let probe: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (4 * 1024 + i) as f32 / 48_000.0).sin())
            .collect();
        canonical.process(&probe).expect("test resampler invariant");
        dropped.process(&probe).expect("test resampler invariant");

        let probe_canonical = canonical.take_output();
        let probe_dropped = dropped.take_output();
        assert_eq!(
            probe_canonical.len(),
            probe_dropped.len(),
            "post-drop output length differs",
        );
        for (i, (a, b)) in probe_canonical.iter().zip(probe_dropped.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "drop_output altered FIR state at sample {i}: {a} vs {b}",
            );
        }
    }

    /// Unity-gain guard: a scale regression shifts Opus-monitor volume yet stays
    /// invisible to accuracy metrics (preproc z-norm is scale-invariant).
    #[test]
    fn resampler_preserves_amplitude_across_rates() {
        // 700-sample slices (not a chunk multiple) exercise the accumulator path.
        fn resample_all(from_sr: u32, input: &[f32]) -> Vec<f32> {
            let mut s = Streaming::new(from_sr, 44_100);
            let mut out = Vec::new();
            for slice in input.chunks(700) {
                s.process(slice).expect("test resampler invariant");
            }
            s.drain_output_into(&mut out);
            out
        }
        // Middle 80% skips the FIR ramp/group-delay tapers at the edges.
        fn mid_peak(s: &[f32]) -> f32 {
            let lo = s.len() / 10;
            let hi = s.len() * 9 / 10;
            s[lo..hi].iter().fold(0.0f32, |a, &v| a.max(v.abs()))
        }

        const AMP: f32 = 0.5;
        for &from_sr in &[8_000u32, 16_000, 22_050, 44_100, 48_000] {
            for &freq in &[100.0f32, 500.0] {
                let n = from_sr as usize;
                let input: Vec<f32> = (0..n)
                    .map(|i| {
                        AMP * (2.0 * std::f32::consts::PI * freq * i as f32 / from_sr as f32).sin()
                    })
                    .collect();
                let out = resample_all(from_sr, &input);
                assert!(
                    out.len() > n / 2,
                    "{from_sr} Hz: too little output ({})",
                    out.len(),
                );
                let gain = mid_peak(&out) / AMP;
                assert!(
                    (0.98..=1.02).contains(&gain),
                    "{from_sr} Hz -> 44.1 kHz, {freq} Hz tone: peak gain {gain:.4} outside \
                     +/-2% (resampler amplitude regression)",
                );
            }
        }

        // DC gain: tightest unity-gain check (no ripple, no group-delay taper).
        for &from_sr in &[16_000u32, 48_000] {
            let input = vec![0.3f32; from_sr as usize];
            let out = resample_all(from_sr, &input);
            let mid = &out[out.len() / 4..out.len() * 3 / 4];
            let gain = (mid.iter().sum::<f32>() / mid.len() as f32) / 0.3;
            assert!(
                (0.995..=1.005).contains(&gain),
                "{from_sr} Hz -> 44.1 kHz DC: gain {gain:.4} outside +/-0.5% \
                 (resampler DC-gain regression)",
            );
        }
    }
}
