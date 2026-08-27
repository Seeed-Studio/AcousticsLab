//! Shared WAV-to-44032-sample-f32 pipeline for training and reference tools.
//! Re-exports [`crate::dsp::resample`]'s [`SincResampler`] / [`sinc_resampler`]
//! so inference- and fine-tune-time resamplers share sinc params and cannot drift.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::Resampler;
use std::path::Path;
use thiserror::Error;

pub use crate::dsp::resample::{SincResampler, sinc_resampler};

/// Target sample rate for the inference pipeline (Hz).
pub const TARGET_SR: u32 = 44100;
use crate::common::dims::WaveformLen;

/// Hard ceiling on accepted WAV duration, rejected before the decode `Vec<f32>` alloc.
pub const MAX_INPUT_DURATION_SECS: u32 = 60;

/// Ceiling on the pre-downmix interleaved decode buffer (`frames * n_chan * 4`):
/// the duration cap alone permits 60s x 192 kHz x 8ch x 4 = 368 MiB (~1.5 GiB
/// resident at rayon W=4, OOMing a 1.5 GiB SBC); 64 MiB still admits 16-48 kHz
/// mono/stereo training shapes.
pub const MAX_DECODE_BUFFER_BYTES: usize = 64 * 1024 * 1024;

/// Failure shapes for [`read_wav_mono`] and [`to_waveform`], propagated via `?`
/// so one bad file doesn't kill the batch loop.
#[derive(Debug, Error)]
pub enum PreprocError {
    #[error("open wav {path}: {source}")]
    WavOpen {
        path: String,
        #[source]
        source: hound::Error,
    },
    #[error("decode wav {path} at sample {sample_idx}: {source}")]
    WavDecode {
        path: String,
        sample_idx: usize,
        #[source]
        source: hound::Error,
    },
    /// (format, bits) not one of the accepted PCM-i16/i24/i32 or IEEE-f32.
    #[error("unsupported wav format {format} / {bits} bits in {path}")]
    WavFormat {
        path: String,
        format: String,
        bits: u16,
    },
    /// Header field outside the accepted range; zero fields would divide-by-zero
    /// downstream (sr=0 in resampler init, channels=0 in downmix).
    #[error("invalid wav header in {path}: {field} = {got}")]
    WavInvalidHeader {
        path: String,
        field: &'static str,
        got: u64,
    },
    /// Non-finite float-32 sample; the finite-only PCM contract keeps NaN/Inf out of
    /// resamplers and inference frames. `value` may be `NaN`, so `PreprocError` must
    /// NOT derive `PartialEq` (IEEE `NaN != NaN`); tests assert via `is_nan()`.
    #[error("non-finite wav sample in {path} at sample {sample_idx}: {value}")]
    BadWavSample {
        path: String,
        sample_idx: usize,
        value: f32,
    },
    /// Input too short (post-resample) for one full training window: padding
    /// would fabricate labeled silence that now TRAINS, so snippet-chopping
    /// rejects (lands in training's `dropped_io`). [`to_waveform`] still pads
    /// for its diagnostic callers.
    #[error(
        "input too short: {got_samples} samples after resample, need >= {need} (one 1 s window)"
    )]
    TooShort { got_samples: usize, need: usize },
    /// Header duration exceeds [`MAX_INPUT_DURATION_SECS`]; caught before alloc.
    #[error(
        "wav {path} duration {duration_secs:.1}s exceeds cap {max_secs}s \
         ({sample_rate} Hz x {frames} frames)"
    )]
    WavTooLong {
        path: String,
        sample_rate: u32,
        frames: u32,
        duration_secs: f32,
        max_secs: u32,
    },
    /// Within the duration cap but the pre-downmix decode buffer would exceed
    /// [`MAX_DECODE_BUFFER_BYTES`] (rayon preproc OOM guard).
    #[error(
        "wav {path} pre-downmix decode buffer {observed_bytes} bytes \
         exceeds cap {max_bytes} bytes ({sample_rate} Hz x {channels} channels x {frames} frames)"
    )]
    WavTooLarge {
        path: String,
        sample_rate: u32,
        channels: u16,
        frames: u32,
        observed_bytes: usize,
        max_bytes: usize,
    },
    #[error("resample: {0}")]
    Resample(String),
}

/// Read a `.wav` file, returning `(sample_rate, mono f32 samples in [-1, 1])`,
/// downmixing by channel-averaging. Header duration is checked against
/// [`MAX_INPUT_DURATION_SECS`] before the decode loop runs.
///
/// # Errors
///
/// See [`PreprocError`].
pub fn read_wav_mono(path: &Path) -> Result<(u32, Vec<f32>), PreprocError> {
    let mut r = hound::WavReader::open(path).map_err(|e| PreprocError::WavOpen {
        path: path.display().to_string(),
        source: e,
    })?;
    let s = r.spec();
    let n_chan = s.channels as usize;
    let path_string = path.display().to_string();

    // hound accepts anything; reject out-of-range header fields before they reach
    // the resampler (sr=0 -> div-by-zero) or downmix (channels=0 -> chunks(0) panic).
    // Ranges reuse the capture-side caps so WAV ingest and live capture refuse the
    // same pathological metadata.
    use crate::audio_io::mic_arbitrator::{MAX_CHANNELS, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE};
    let header_err = |field: &'static str, got: u64| PreprocError::WavInvalidHeader {
        path: path_string.clone(),
        field,
        got,
    };
    if s.sample_rate == 0 {
        return Err(header_err("sample_rate", 0));
    }
    if s.channels == 0 {
        return Err(header_err("channels", 0));
    }
    if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&s.sample_rate) {
        return Err(header_err("sample_rate", s.sample_rate as u64));
    }
    if (s.channels as usize) > MAX_CHANNELS {
        return Err(header_err("channels", s.channels as u64));
    }

    // `r.duration()` is the header's per-channel sample count (no scan): O(1) cap
    // before any buffer growth; saturating mul guards a u32::MAX-sample claim.
    let frames = r.duration();
    let max_frames = (MAX_INPUT_DURATION_SECS as u64).saturating_mul(s.sample_rate as u64);
    if (frames as u64) > max_frames {
        // divisor non-zero: sample_rate gated >= MIN_SAMPLE_RATE above.
        let duration_secs = frames as f32 / s.sample_rate as f32;
        return Err(PreprocError::WavTooLong {
            path: path_string,
            sample_rate: s.sample_rate,
            frames,
            duration_secs,
            max_secs: MAX_INPUT_DURATION_SECS,
        });
    }

    let expected_samples = (frames as usize).saturating_mul(n_chan);
    // Gate byte size before `collect`'s `Vec::with_capacity` so the OOM-sized
    // allocation never happens.
    let expected_bytes = expected_samples.saturating_mul(std::mem::size_of::<f32>());
    if expected_bytes > MAX_DECODE_BUFFER_BYTES {
        return Err(PreprocError::WavTooLarge {
            path: path_string,
            sample_rate: s.sample_rate,
            channels: s.channels,
            frames,
            observed_bytes: expected_bytes,
            max_bytes: MAX_DECODE_BUFFER_BYTES,
        });
    }
    fn collect<S, F>(
        path: &str,
        iter: impl Iterator<Item = Result<S, hound::Error>>,
        expected_len: usize,
        mut scale: F,
    ) -> Result<Vec<f32>, PreprocError>
    where
        F: FnMut(S, usize) -> Result<f32, PreprocError>,
    {
        let mut out = Vec::with_capacity(expected_len);
        for (i, sample) in iter.enumerate() {
            let s = sample.map_err(|e| PreprocError::WavDecode {
                path: path.to_string(),
                sample_idx: i,
                source: e,
            })?;
            out.push(scale(s, i)?);
        }
        Ok(out)
    }
    // Full-scale divisors (`2^(bits-1)`) mapping signed PCM into [-1.0, 1.0].
    const SCALE_I16: f32 = (1u32 << 15) as f32;
    const SCALE_I24: f32 = (1u32 << 23) as f32;
    const SCALE_I32: f32 = (1u64 << 31) as f32;
    let samples: Vec<f32> = match (s.sample_format, s.bits_per_sample) {
        // Int formats are finite by construction; only float-32 needs the finite scan.
        (hound::SampleFormat::Int, 16) => collect(
            &path_string,
            r.samples::<i16>(),
            expected_samples,
            |v, _| Ok(v as f32 / SCALE_I16),
        )?,
        (hound::SampleFormat::Int, 24) => collect(
            &path_string,
            r.samples::<i32>(),
            expected_samples,
            |v, _| Ok(v as f32 / SCALE_I24),
        )?,
        (hound::SampleFormat::Int, 32) => collect(
            &path_string,
            r.samples::<i32>(),
            expected_samples,
            |v, _| Ok(v as f32 / SCALE_I32),
        )?,
        (hound::SampleFormat::Float, 32) => collect(
            &path_string,
            r.samples::<f32>(),
            expected_samples,
            // Finite check folded into the scale callback: single pass, fail on the
            // first non-finite sample with its index.
            |v, i| {
                if v.is_finite() {
                    Ok(v)
                } else {
                    Err(PreprocError::BadWavSample {
                        path: path_string.clone(),
                        sample_idx: i,
                        value: v,
                    })
                }
            },
        )?,
        (fmt, bits) => {
            return Err(PreprocError::WavFormat {
                path: path_string,
                format: format!("{fmt:?}"),
                bits,
            });
        }
    };
    let mono: Vec<f32> = if n_chan == 1 {
        samples
    } else {
        // A partial interleaved frame would scale the final chunk's amplitude; the
        // header gate should preclude it. Fail-loud in CI.
        debug_assert_eq!(
            samples.len() % n_chan,
            0,
            "post-decode sample count {} is not a multiple of n_chan {n_chan} -- \
             hound returned a partial interleaved frame; header-validation gate above \
             should have rejected this",
            samples.len(),
        );
        let mut downmixed = Vec::with_capacity(samples.len() / n_chan);
        for c in samples.chunks(n_chan) {
            downmixed.push(c.iter().sum::<f32>() / n_chan as f32);
        }
        downmixed
    };
    Ok((s.sample_rate, mono))
}

/// Rate-keyed [`SincResampler`] cache slot for [`to_waveform`]: keying by source
/// rate is mandatory because an unkeyed slot would apply the first file's ratio to
/// every later file, yielding valid-but-wrong spectrograms on a mixed-rate dataset.
#[derive(Debug, Default)]
pub struct ResamplerCache {
    inner: Option<(u32, SincResampler)>,
}

impl ResamplerCache {
    /// Empty cache; the first non-target-rate [`to_waveform`] lazy-builds the slot.
    #[inline]
    pub const fn empty() -> Self {
        Self { inner: None }
    }

    /// Source sample rate of the cached resampler, or `None` if none built.
    #[inline]
    pub fn cached_rate(&self) -> Option<u32> {
        self.inner.as_ref().map(|(sr, _)| *sr)
    }

    /// Drop the cached resampler so the next call rebuilds. Used on resampler errors
    /// where FIR history may be partial; safer than `reset()`, which keeps the
    /// now-suspect ratio.
    #[inline]
    pub fn clear(&mut self) {
        self.inner = None;
    }

    /// Borrow the resampler for `sr`, building it if empty or a different rate, and
    /// always `reset()`-ing FIR history so a prior file's taps don't bleed in.
    fn get_or_init(&mut self, sr: u32) -> &mut SincResampler {
        match &self.inner {
            Some((cached, _)) if *cached == sr => {}
            _ => self.inner = Some((sr, sinc_resampler(sr, TARGET_SR))),
        }
        let (_, r) = self
            .inner
            .as_mut()
            .expect("inner populated by the match arm above");
        r.reset();
        r
    }
}

/// Resample `mono` from `sr` to [`TARGET_SR`], at most `limit` output samples; the
/// loop early-exits at `limit` so long input doesn't burn CPU on discarded samples.
/// `sr == TARGET_SR` returns the input truncated to `limit`.
fn resample_to_target(
    sr: u32,
    mono: Vec<f32>,
    cache: &mut ResamplerCache,
    limit: usize,
) -> Result<Vec<f32>, PreprocError> {
    if sr == TARGET_SR {
        let mut mono = mono;
        mono.truncate(limit);
        return Ok(mono);
    }
    let r = cache.get_or_init(sr);
    let in_per = r.input_frames_next();
    let max_out = r.output_frames_max();
    let mut padded = mono;
    let pad = (in_per - (padded.len() % in_per)) % in_per;
    padded.extend(std::iter::repeat_n(0.0, pad));
    // Loop breaks one chunk past `limit`, so `limit + max_out` bounds the output,
    // dodging the worst-case `padded.len() * TARGET_SR / sr` alloc.
    let cap = limit.saturating_add(max_out);
    let mut out: Vec<f32> = Vec::with_capacity(cap);
    let mut out_scratch = vec![0.0f32; max_out];
    for c in padded.chunks(in_per) {
        let in_adapter = InterleavedSlice::new(c, 1, in_per)
            .map_err(|e| PreprocError::Resample(format!("input adapter: {e}")))?;
        let mut out_adapter = InterleavedSlice::new_mut(&mut out_scratch, 1, max_out)
            .map_err(|e| PreprocError::Resample(format!("output adapter: {e}")))?;
        let (_, n_out) = r
            .process_into_buffer(&in_adapter, &mut out_adapter, None)
            .map_err(|e| PreprocError::Resample(format!("process chunk: {e}")))?;
        out.extend_from_slice(&out_scratch[..n_out]);
        if out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

/// Mono f32 at `sr` into exactly [`WaveformLen::USIZE`] samples at [`TARGET_SR`]
/// (the first 1 s window): resample, then right-zero-pad short / tail-truncate long.
pub fn to_waveform(
    sr: u32,
    mono: Vec<f32>,
    cache: &mut ResamplerCache,
) -> Result<Box<[f32; WaveformLen::USIZE]>, PreprocError> {
    let resampled = resample_to_target(sr, mono, cache, WaveformLen::USIZE)?;
    Ok(pad_to_window(&resampled))
}

/// Copy up to one window of `resampled` into a fresh `WaveformLen`-sample box,
/// right-zero-padded short / tail-truncated long.
fn pad_to_window(resampled: &[f32]) -> Box<[f32; WaveformLen::USIZE]> {
    let mut arr = Box::new([0f32; WaveformLen::USIZE]);
    let n = resampled.len().min(WaveformLen::USIZE);
    arr[..n].copy_from_slice(&resampled[..n]);
    arr
}

/// Snippet-chop variant of [`to_waveform`] yielding every non-overlapping
/// [`WaveformLen::USIZE`]-sample window (multiple training examples per recording).
/// Sub-window input is [`PreprocError::TooShort`], never zero-padded; long input
/// yields up to `max_windows` full windows (clamped `>= 1`), trailing partial dropped.
pub fn to_waveform_windows(
    sr: u32,
    mono: Vec<f32>,
    cache: &mut ResamplerCache,
    max_windows: usize,
) -> Result<Vec<Box<[f32; WaveformLen::USIZE]>>, PreprocError> {
    let max_windows = max_windows.max(1);
    let limit = WaveformLen::USIZE.saturating_mul(max_windows);
    let resampled = resample_to_target(sr, mono, cache, limit)?;
    let n_full = resampled.len() / WaveformLen::USIZE;
    if n_full == 0 {
        return Err(PreprocError::TooShort {
            got_samples: resampled.len(),
            need: WaveformLen::USIZE,
        });
    }
    let n = n_full.min(max_windows);
    let mut out = Vec::with_capacity(n);
    for w in 0..n {
        let mut arr = Box::new([0f32; WaveformLen::USIZE]);
        let start = w * WaveformLen::USIZE;
        arr.copy_from_slice(&resampled[start..start + WaveformLen::USIZE]);
        out.push(arr);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_wav_mono_missing_file_returns_err() {
        let path = std::path::Path::new("/nonexistent/.acousticslab/no-such.wav");
        let err = read_wav_mono(path).expect_err("missing file must be Err");
        match err {
            PreprocError::WavOpen { path: p, .. } => {
                assert!(p.contains("no-such"), "diagnostic should embed path: {p}");
            }
            other => panic!("expected WavOpen, got {other:?}"),
        }
    }

    #[test]
    fn read_wav_mono_garbage_bytes_returns_err() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("garbage.wav");
        #[allow(clippy::disallowed_methods)]
        std::fs::write(&path, b"not a wav file at all, just garbage bytes").unwrap();
        let err = read_wav_mono(&path).expect_err("garbage bytes must be Err");
        // hound's exact dispatch among the three is not contractual.
        assert!(
            matches!(
                err,
                PreprocError::WavOpen { .. }
                    | PreprocError::WavDecode { .. }
                    | PreprocError::WavFormat { .. }
            ),
            "expected WavOpen / WavDecode / WavFormat, got {err:?}",
        );
    }

    fn write_silence_wav(dir: &std::path::Path, name: &str, sr: u32, n_frames: u32) -> PathBuf {
        let path = dir.join(name);
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).expect("wav writer");
        for _ in 0..n_frames {
            w.write_sample(0i16).expect("write sample");
        }
        w.finalize().expect("finalize wav");
        path
    }

    use std::path::PathBuf;

    /// Over-cap duration -> `WavTooLong` before per-sample decode.
    #[test]
    fn read_wav_mono_rejects_overlong_input() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sr: u32 = 16_000;
        let n_frames = MAX_INPUT_DURATION_SECS * sr + 1; // one frame past the cap
        let path = write_silence_wav(dir.path(), "long.wav", sr, n_frames);
        let err = read_wav_mono(&path).expect_err("overlong wav must be Err");
        match err {
            PreprocError::WavTooLong {
                sample_rate,
                frames,
                max_secs,
                ..
            } => {
                assert_eq!(sample_rate, sr);
                assert_eq!(frames, n_frames);
                assert_eq!(max_secs, MAX_INPUT_DURATION_SECS);
            }
            other => panic!("expected WavTooLong, got {other:?}"),
        }
    }

    /// Mono float-32 WAV preserving exact sample bits (incl. NaN/Inf) for the
    /// finite-only PCM contract tests.
    fn write_float_wav(dir: &std::path::Path, name: &str, sr: u32, samples: &[f32]) -> PathBuf {
        let path = dir.join(name);
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut w = hound::WavWriter::create(&path, spec).expect("wav writer");
        for s in samples {
            w.write_sample(*s).expect("write sample");
        }
        w.finalize().expect("finalize wav");
        path
    }

    fn write_silence_multichannel_wav(
        dir: &std::path::Path,
        name: &str,
        sr: u32,
        channels: u16,
        n_frames: u32,
    ) -> PathBuf {
        let path = dir.join(name);
        let spec = hound::WavSpec {
            channels,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).expect("wav writer");
        for _ in 0..n_frames {
            for _ in 0..channels {
                w.write_sample(0i16).expect("write sample");
            }
        }
        w.finalize().expect("finalize wav");
        path
    }

    #[test]
    fn read_wav_mono_downmixes_stereo() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).expect("wav writer");
        const N_FRAMES: usize = 100;
        const LEFT_I16: i16 = 1000;
        const RIGHT_I16: i16 = -2000;
        for _ in 0..N_FRAMES {
            w.write_sample(LEFT_I16).expect("write L");
            w.write_sample(RIGHT_I16).expect("write R");
        }
        w.finalize().expect("finalize wav");

        let (sr, mono) = read_wav_mono(&path).expect("stereo wav must decode");
        assert_eq!(sr, 16_000, "sample rate");
        assert_eq!(
            mono.len(),
            N_FRAMES,
            "downmix length must equal frame count"
        );

        let expected = (LEFT_I16 as f32 + RIGHT_I16 as f32) / (1u32 << 15) as f32 / 2.0;
        for (i, &v) in mono.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-9,
                "frame {i}: downmix expected {expected}, got {v}",
            );
        }
    }

    /// Float-32 NaN -> `BadWavSample` carrying the offending index.
    #[test]
    fn read_wav_mono_rejects_nan_in_float() {
        let dir = tempfile::tempdir().expect("tempdir");
        let samples = [0.0f32, 0.5, f32::NAN, -0.25];
        let path = write_float_wav(dir.path(), "nan.wav", 16_000, &samples);
        let err = read_wav_mono(&path).expect_err("NaN sample must reject");
        match err {
            PreprocError::BadWavSample {
                sample_idx, value, ..
            } => {
                assert_eq!(sample_idx, 2, "NaN was at index 2");
                assert!(value.is_nan(), "rejected value must be NaN, got {value}");
            }
            other => panic!("expected BadWavSample, got {other:?}"),
        }
    }

    #[test]
    fn read_wav_mono_rejects_inf_in_float() {
        let dir = tempfile::tempdir().expect("tempdir");
        let samples = [0.0f32, f32::INFINITY, 0.5];
        let path = write_float_wav(dir.path(), "inf.wav", 16_000, &samples);
        let err = read_wav_mono(&path).expect_err("Inf sample must reject");
        match err {
            PreprocError::BadWavSample {
                sample_idx, value, ..
            } => {
                assert_eq!(sample_idx, 1);
                assert!(value.is_infinite() && value.is_sign_positive());
            }
            other => panic!("expected BadWavSample, got {other:?}"),
        }
    }

    #[test]
    fn read_wav_mono_rejects_sub_min_sample_rate() {
        let dir = tempfile::tempdir().expect("tempdir");
        // 100 Hz: below the MIN_SAMPLE_RATE floor.
        let path = write_silence_wav(dir.path(), "slow.wav", 100, 50);
        let err = read_wav_mono(&path).expect_err("100 Hz must reject");
        match err {
            PreprocError::WavInvalidHeader { field, got, .. } => {
                assert_eq!(field, "sample_rate");
                assert_eq!(got, 100);
            }
            other => panic!("expected WavInvalidHeader, got {other:?}"),
        }
    }

    #[test]
    fn read_wav_mono_rejects_super_max_sample_rate() {
        let dir = tempfile::tempdir().expect("tempdir");
        // 384 kHz: above the MAX_SAMPLE_RATE ceiling.
        let path = write_silence_wav(dir.path(), "fast.wav", 384_000, 50);
        let err = read_wav_mono(&path).expect_err("384 kHz must reject");
        match err {
            PreprocError::WavInvalidHeader { field, got, .. } => {
                assert_eq!(field, "sample_rate");
                assert_eq!(got, 384_000);
            }
            other => panic!("expected WavInvalidHeader, got {other:?}"),
        }
    }

    #[test]
    fn read_wav_mono_rejects_oversized_channels() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = write_silence_multichannel_wav(dir.path(), "wide.wav", 16_000, 9, 50);
        let err = read_wav_mono(&path).expect_err("9-channel must reject");
        match err {
            PreprocError::WavInvalidHeader { field, got, .. } => {
                assert_eq!(field, "channels");
                assert_eq!(got, 9);
            }
            other => panic!("expected WavInvalidHeader, got {other:?}"),
        }
    }

    /// A WAV exactly at the cap is accepted (off-by-one guard).
    #[test]
    fn read_wav_mono_accepts_at_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sr: u32 = 8_000; // 8 kHz keeps the at-cap buffer small
        let n_frames = MAX_INPUT_DURATION_SECS * sr;
        let path = write_silence_wav(dir.path(), "at_cap.wav", sr, n_frames);
        let (got_sr, mono) = read_wav_mono(&path).expect("at-cap wav must succeed");
        assert_eq!(got_sr, sr);
        assert_eq!(mono.len(), n_frames as usize);
    }

    /// Cache rebuilds on every rate change, including switching back to a
    /// previously-seen rate; the at-target path never touches the cache.
    #[test]
    fn to_waveform_cache_rebuilds_on_rate_change() {
        let mono_16k: Vec<f32> = vec![0.0; 16_000];
        let mono_48k: Vec<f32> = vec![0.0; 48_000];
        let mut cache = ResamplerCache::empty();
        assert_eq!(cache.cached_rate(), None, "fresh cache must be empty");

        let _ = to_waveform(16_000, mono_16k.clone(), &mut cache)
            .expect("16 kHz to_waveform must succeed");
        assert_eq!(
            cache.cached_rate(),
            Some(16_000),
            "after first call, cache must hold the 16 kHz resampler",
        );

        let _ = to_waveform(48_000, mono_48k.clone(), &mut cache)
            .expect("48 kHz to_waveform must succeed");
        assert_eq!(
            cache.cached_rate(),
            Some(48_000),
            "cache must rebuild on rate change; got rate {:?}",
            cache.cached_rate(),
        );

        let _ = to_waveform(16_000, mono_16k, &mut cache)
            .expect("repeat 16 kHz to_waveform must succeed");
        assert_eq!(
            cache.cached_rate(),
            Some(16_000),
            "cache must rebuild when switching back to a previously-seen rate",
        );

        let prev = cache.cached_rate();
        let mono_44k: Vec<f32> = vec![0.0; 44_100];
        let _ = to_waveform(TARGET_SR, mono_44k, &mut cache)
            .expect("target-rate to_waveform must succeed without cache use");
        assert_eq!(
            cache.cached_rate(),
            prev,
            "target-rate path must not modify the cache",
        );
    }

    /// A long input is tail-truncated to a full `WaveformLen` buffer; the
    /// resample-loop early-exit must not undercut the window size.
    #[test]
    fn to_waveform_truncates_long_input_at_target_rate() {
        let n = WaveformLen::USIZE * 2;
        let mono: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.001).collect();
        let mut cache = ResamplerCache::empty();
        let arr = to_waveform(TARGET_SR, mono.clone(), &mut cache)
            .expect("at-target to_waveform must succeed");
        assert_eq!(arr.len(), WaveformLen::USIZE);
        for (i, &v) in arr.iter().enumerate() {
            assert_eq!(v, mono[i], "expected pass-through at sample {i}");
        }
    }

    /// Long input chops into non-overlapping windows; sub-window input is
    /// `TooShort` while `to_waveform` still pads; `max_windows` caps and drops
    /// the trailing partial.
    #[test]
    fn to_waveform_windows_chops_long_and_rejects_short() {
        let mut cache = ResamplerCache::empty();

        let short: Vec<f32> = (0..1000).map(|i| (i % 7) as f32 * 0.01).collect();
        let err = to_waveform_windows(TARGET_SR, short.clone(), &mut cache, 8)
            .expect_err("sub-window input must reject, not pad");
        match err {
            PreprocError::TooShort { got_samples, need } => {
                assert_eq!(got_samples, 1000);
                assert_eq!(need, WaveformLen::USIZE);
            }
            other => panic!("expected TooShort, got {other:?}"),
        }
        // The single-window path keeps padding (diagnostics score what exists).
        let single = to_waveform(TARGET_SR, short.clone(), &mut cache).expect("to_waveform short");
        assert_eq!(&single[..1000], &short[..], "audio prefix preserved");
        assert!(
            single[1000..].iter().all(|&s| s == 0.0),
            "to_waveform pads the tail with zeros",
        );

        let n = WaveformLen::USIZE * 3;
        let long: Vec<f32> = (0..n).map(|i| (i % 101) as f32 * 0.001).collect();
        let w = to_waveform_windows(TARGET_SR, long.clone(), &mut cache, 8)
            .expect("long to_waveform_windows");
        assert_eq!(w.len(), 3, "3-window input must yield 3 windows");
        for (k, win) in w.iter().enumerate() {
            let start = k * WaveformLen::USIZE;
            assert_eq!(
                &win[..],
                &long[start..start + WaveformLen::USIZE],
                "window {k} must be the k-th slice of the stream",
            );
        }

        // 3.5 windows, cap 2 -> 2 windows (trailing partial dropped).
        let n = WaveformLen::USIZE * 3 + WaveformLen::USIZE / 2;
        let long: Vec<f32> = vec![0.25f32; n];
        let w = to_waveform_windows(TARGET_SR, long, &mut cache, 2).expect("capped windows");
        assert_eq!(w.len(), 2, "max_windows must cap the window count");
    }
}
