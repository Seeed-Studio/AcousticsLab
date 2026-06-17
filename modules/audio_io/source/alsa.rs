//! Multi-channel ALSA capture source (Linux + `alsa-real` only).
//!
//! Negotiates at the device's actual channel count (no ALSA-side downmix) so the
//! arbitrator can do RMS-based channel selection. Float LE preferred, s16 LE fallback.

use crate::audio_io::mic_arbitrator::{CandidateSource, MicCandidate};
use crate::audio_io::source::{ReadError, ReadOutcome};
use crate::common::dims::SampleRate;
use crate::common::ids::MicId;
use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::{Direction, PollDescriptors, ValueOr};
use std::num::NonZeroUsize;
use std::time::Duration;

/// Outcome of a bounded-timeout ALSA read; the explicit `Timeout` lets the run loop re-check
/// its stop flag instead of blocking on `readi`, so a hung USB device can't pin the thread.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AlsaReadOutcome {
    /// `n` interleaved frames read into `out[..n * channels]`.
    Frames(usize),
    /// `poll` found 0 ready events within `timeout`; `out` is untouched.
    Timeout,
}

pub struct AlsaSource {
    id: MicId,
    pcm: PCM,
    effective_whitelist: Vec<u16>,
    /// Negotiated channel count; interleaved frames are this wide.
    actual_channels: u16,
    /// Frames per `readi`; read buffer is `period_size * actual_channels` samples.
    period_size: usize,
    is_float: bool,
    /// Negotiated rate; arbitrator resamples iff != [`SampleRate::VALUE`].
    rate: u32,
    /// Reusable s16-fallback scratch (alloc-free hot path); empty when `is_float`.
    i16_scratch: Vec<i16>,
    /// PCM pollfds cached at open (stable until drop): the read path reuses this (resetting
    /// `revents`) rather than allocating per period on the RT thread, which would spike
    /// worst-case latency via malloc-lock contention.
    pollfds: Vec<alsa::poll::pollfd>,
    overlong_read_warned: bool,
}

// `alsa::PCM` has no `Debug`; print the negotiated values.
impl std::fmt::Debug for AlsaSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlsaSource")
            .field("id", &self.id)
            .field("actual_channels", &self.actual_channels)
            .field("effective_whitelist", &self.effective_whitelist)
            .field("period_size", &self.period_size)
            .field("rate", &self.rate)
            .field("is_float", &self.is_float)
            .finish()
    }
}

impl AlsaSource {
    /// Open the candidate's ALSA device + negotiate channels/format/rate/period. Errors are
    /// stringified to keep `alsa::Error` out of [`super::OpenError`]; re-validates to guard direct
    /// (test) callers that skip the config-load path.
    pub(crate) fn open(candidate: &MicCandidate) -> Result<Self, String> {
        candidate
            .validate()
            .map_err(|e| format!("candidate failed validation: {e}"))?;
        let CandidateSource::Alsa {
            hw_spec,
            period_size,
            buffer_size,
        } = &candidate.source
        else {
            return Err("AlsaSource::open called on a Mock candidate".into());
        };
        let (hw_spec, period_size, buffer_size) = (hw_spec.clone(), *period_size, *buffer_size);

        // Non-blocking (3rd arg) so `readi` returns EAGAIN instead of blocking, letting the
        // bounded poll keep a wedged USB device from pinning the thread.
        let pcm = PCM::new(&hw_spec, Direction::Capture, true)
            .map_err(|e| format!("PCM::new({hw_spec}): {e}"))?;

        // Minimum channel count exposing every whitelisted index; wider costs bandwidth.
        let whitelist_max = candidate
            .channels
            .iter()
            .copied()
            .max()
            .ok_or("candidate channels whitelist is empty")?;
        let needed = whitelist_max as u32 + 1;

        let (actual_channels, is_float, actual_rate, actual_period);
        {
            let hwp = HwParams::any(&pcm).map_err(|e| format!("HwParams::any: {e}"))?;
            hwp.set_access(Access::RWInterleaved)
                .map_err(|e| format!("set_access: {e}"))?;

            // Prefer FloatLE; s16 fallback for USB mics without float.
            is_float = if hwp.set_format(Format::float()).is_ok() {
                true
            } else {
                hwp.set_format(Format::s16())
                    .map_err(|e| format!("set_format(s16): {e}"))?;
                false
            };

            // Try exact, fall back to channels_max; MAX_NEGOTIABLE_CHANNELS on both legs bounds
            // `period * n_channels * 4` against the multi-GB alloc a `channels_max = u16::MAX`
            // backend could force.
            use crate::audio_io::mic_arbitrator::MAX_NEGOTIABLE_CHANNELS;
            if hwp.set_channels(needed).is_ok() {
                debug_assert!(
                    needed <= MAX_NEGOTIABLE_CHANNELS,
                    "negotiated channel count {needed} exceeds MAX_NEGOTIABLE_CHANNELS",
                );
                actual_channels = needed as u16;
            } else {
                let max = hwp
                    .get_channels_max()
                    .map_err(|e| format!("get_channels_max: {e}"))?;
                if max == 0 {
                    return Err(format!("device {hw_spec} reports 0 channels"));
                }
                if max > MAX_NEGOTIABLE_CHANNELS {
                    return Err(format!(
                        "device {hw_spec} reports {max} channels -- exceeds MAX_NEGOTIABLE_CHANNELS \
                         ({MAX_NEGOTIABLE_CHANNELS})",
                    ));
                }
                hwp.set_channels(max)
                    .map_err(|e| format!("set_channels({max}) fallback: {e}"))?;
                actual_channels = max as u16;
                tracing::warn!(
                    target: "audio_io.source.alsa",
                    device = %candidate.id,
                    hw_spec = %hw_spec,
                    requested = needed,
                    actual = max,
                    "device refused requested channel count; using device max",
                );
            }

            // `_near` so 48 kHz-only USB mics negotiate; arbitrator resamples non-native rates.
            hwp.set_rate_near(SampleRate::VALUE, ValueOr::Nearest)
                .map_err(|e| format!("set_rate_near: {e}"))?;
            // `_near` snaps to nearest supported period (exact fails on mics quantizing to
            // 480/960/1920); negotiated value read back after `hw_params`.
            hwp.set_period_size_near(period_size as alsa::pcm::Frames, ValueOr::Nearest)
                .map_err(|e| format!("set_period_size_near: {e}"))?;
            hwp.set_buffer_size_near(buffer_size as alsa::pcm::Frames)
                .map_err(|e| format!("set_buffer_size_near: {e}"))?;
            pcm.hw_params(&hwp).map_err(|e| format!("hw_params: {e}"))?;
            // `set_channels` is exact; a post-commit mismatch means a wrong interleaved stride downstream.
            let neg_channels = hwp
                .get_channels()
                .map_err(|e| format!("get_channels: {e}"))?;
            if neg_channels == 0 {
                return Err(format!(
                    "device {hw_spec} reported 0 channels after hw_params"
                ));
            }
            if neg_channels > MAX_NEGOTIABLE_CHANNELS {
                return Err(format!(
                    "device {hw_spec} reports {neg_channels} channels after hw_params -- exceeds \
                     MAX_NEGOTIABLE_CHANNELS ({MAX_NEGOTIABLE_CHANNELS})",
                ));
            }
            if neg_channels != actual_channels as u32 {
                return Err(format!(
                    "device {hw_spec} committed {neg_channels} channels after hw_params, expected {actual_channels}",
                ));
            }
            actual_rate = hwp.get_rate().map_err(|e| format!("get_rate: {e}"))?;
            // Bound the rate: out-of-range builds a resampler with an absurd ratio (multi-MB sinc
            // tables) or overflows the arbitrator's `cached_rate` math.
            use crate::audio_io::mic_arbitrator::{MAX_SAMPLE_RATE, MIN_SAMPLE_RATE};
            if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&actual_rate) {
                return Err(format!(
                    "device {hw_spec} negotiated rate {actual_rate} Hz outside \
                     [{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}]",
                ));
            }
            // Read back the negotiated period (device may snap larger): sizing scratch off the
            // requested value causes chronic short reads. `Frames` is `i64`; reject non-positive.
            let neg_period = hwp
                .get_period_size()
                .map_err(|e| format!("get_period_size: {e}"))?;
            if neg_period <= 0 {
                return Err(format!(
                    "device {hw_spec} reported non-positive period size {neg_period}",
                ));
            }
            // Bound the period (a buggy 100M-frame value would allocate GB); 2x the ceiling for snap slack.
            use crate::audio_io::mic_arbitrator::MAX_PERIOD_FRAMES;
            let max_negotiable = MAX_PERIOD_FRAMES.saturating_mul(2);
            if (neg_period as u64) > max_negotiable as u64 {
                return Err(format!(
                    "device {hw_spec} negotiated period {neg_period} frames > 2 * \
                     MAX_PERIOD_FRAMES ({max_negotiable})",
                ));
            }
            actual_period = neg_period as usize;
        }

        // Out-of-range entries get one consolidated warning (per-entry would spam journalctl).
        let dropped: Vec<u16> = candidate
            .channels
            .iter()
            .copied()
            .filter(|&ch| ch >= actual_channels)
            .collect();
        if !dropped.is_empty() {
            tracing::warn!(
                target: "audio_io.source.alsa",
                device = %candidate.id,
                dropped_channels = ?dropped,
                actual_channels,
                "whitelist entries exceed device channel count; dropping",
            );
        }
        let effective_whitelist = intersect_whitelist(&candidate.channels, actual_channels);
        if effective_whitelist.is_empty() {
            return Err(format!(
                "candidate {} whitelist {:?} has no entries within \
                 device's {} channels",
                candidate.id, candidate.channels, actual_channels,
            ));
        }

        pcm.prepare().map_err(|e| format!("prepare: {e}"))?;
        pcm.start().map_err(|e| format!("start: {e}"))?;

        let pollfds = pcm
            .get()
            .map_err(|e| format!("PollDescriptors::get: {e}"))?;

        if actual_period != period_size {
            tracing::info!(
                target: "audio_io.source.alsa",
                device = %candidate.id,
                requested_period = period_size,
                actual_period,
                "device snapped to a different period size",
            );
        }

        // Non-native rate forces a per-slot rubato resampler (~140 KB sinc table + CPU/L2
        // contention); warn so operators pick 44.1 kHz-native hardware.
        if actual_rate != SampleRate::VALUE {
            tracing::warn!(
                target: "audio_io.source.alsa",
                device = %candidate.id,
                hw_spec = %hw_spec,
                requested_rate = SampleRate::VALUE,
                actual_rate,
                "device negotiated a non-44.1 kHz rate; arbitrator will resample per slot -- \
                 prefer 44.1 kHz-native hardware",
            );
        }

        let i16_scratch = if is_float {
            Vec::new()
        } else {
            vec![0i16; actual_period * actual_channels as usize]
        };

        tracing::info!(
            target: "audio_io.source.alsa",
            device = %candidate.id,
            hw_spec = %hw_spec,
            actual_channels,
            actual_rate,
            is_float,
            actual_period,
            effective_whitelist = ?effective_whitelist,
            "ALSA source opened",
        );
        Ok(Self {
            id: candidate.id.clone(),
            pcm,
            effective_whitelist,
            actual_channels,
            period_size: actual_period,
            is_float,
            rate: actual_rate,
            i16_scratch,
            pollfds,
            overlong_read_warned: false,
        })
    }

    pub fn id(&self) -> &MicId {
        &self.id
    }

    pub fn channels(&self) -> u16 {
        self.actual_channels
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    pub fn period_size(&self) -> usize {
        self.period_size
    }

    /// Whitelist usable on this device (candidate `channels` intersect `[0..actual_channels)`);
    /// drives the arbitrator's per-slot RMS.
    pub fn effective_whitelist(&self) -> &[u16] {
        &self.effective_whitelist
    }

    /// Default poll timeout: two periods at the negotiated rate. The 2x absorbs scheduler jitter
    /// so a healthy device doesn't spuriously hit [`AlsaReadOutcome::Timeout`].
    pub fn default_read_timeout(&self) -> Duration {
        Duration::from_secs_f64(self.period_size as f64 / self.rate as f64).saturating_mul(2)
    }

    /// Read up to one period of interleaved frames into `out`, polling with a bounded `timeout`
    /// first so a wedged USB device can't pin the arbitrator thread. `timeout` >= one period plus
    /// slack keeps a healthy device on `Frames` not `Timeout`; `out` must be >= `period_size *
    /// channels()`; short reads (EOF/hot-unplug) return `Frames(n)`.
    pub fn read_with_timeout(
        &mut self,
        out: &mut [f32],
        timeout: Duration,
    ) -> alsa::Result<AlsaReadOutcome> {
        let needed = self.period_size * self.actual_channels as usize;
        debug_assert!(
            out.len() >= needed,
            "alsa read buffer too small: {} < {needed}",
            out.len(),
        );

        // Clear sticky `revents` so this poll only reports fds ready now.
        for p in &mut self.pollfds {
            p.revents = 0;
        }
        // Floor at 1 ms: `as_millis()` floors a sub-ms timeout to 0, degenerating poll into a
        // busy-spin; i32::MAX ceiling avoids overflow.
        let timeout_ms: i32 = timeout.as_millis().clamp(1, i32::MAX as u128) as i32;
        let n_ready = match alsa::poll::poll(&mut self.pollfds, timeout_ms) {
            Ok(n) => n,
            Err(e) => {
                // EINTR (typically `unpark` from `signal_stop`) is benign; Timeout fires the stop check.
                if e.errno() == libc::EINTR {
                    return Ok(AlsaReadOutcome::Timeout);
                }
                return Err(e);
            }
        };
        if n_ready == 0 {
            return Ok(AlsaReadOutcome::Timeout);
        }

        let target = &mut out[..needed];
        // SAFETY: `io_unchecked` requires (1) `S` matches the stream sample type -- `is_float`
        // reflects the format verified at `open()`, immutable for the source's lifetime; (2) no
        // concurrent `IO` -- each is created and dropped within this `&mut self` call on a
        // thread-owned PCM. Safe `io_f32()/io_i16()` re-verify the unchangeable format via a
        // per-period malloc + syscall, avoided on the RT thread.
        let frames = if self.is_float {
            let io = unsafe { self.pcm.io_unchecked::<f32>() };
            match io.readi(target) {
                Ok(n) => n,
                Err(e) if is_eagain(&e) => return Ok(AlsaReadOutcome::Timeout),
                Err(e) => return Err(e),
            }
        } else {
            let io = unsafe { self.pcm.io_unchecked::<i16>() };
            let raw_frames = match io.readi(&mut self.i16_scratch) {
                Ok(n) => n,
                Err(e) if is_eagain(&e) => return Ok(AlsaReadOutcome::Timeout),
                Err(e) => return Err(e),
            };
            // Clamp before slice indexing: a driver returning > buffer size would panic here,
            // before the outer `read_interleaved` clamp can fire.
            let frames = raw_frames.min(self.period_size);
            // /32_768.0 (2^15) maps -32768 to -1.0 exactly; /i16::MAX would break that symmetry.
            let n_samples = frames * self.actual_channels as usize;
            for (dst, &src) in target[..n_samples]
                .iter_mut()
                .zip(self.i16_scratch[..n_samples].iter())
            {
                *dst = src as f32 / 32_768.0;
            }
            frames
        };
        Ok(AlsaReadOutcome::Frames(frames))
    }

    /// Standard ALSA xrun/suspend recovery. `Ok(())` if the PCM is ready again; `Err` if
    /// unrecoverable (typically ENODEV).
    pub fn try_recover(&mut self, e: alsa::Error) -> alsa::Result<()> {
        self.pcm.try_recover(e, true)?; // silent=true: tracing, not stderr
        // Recovery lands PREPARED (xrun) or RUNNING (resume); gate `start` on PREPARED since
        // `snd_pcm_start` on a RUNNING PCM returns -EBADFD (spurious recovery failure). Use
        // `state_raw()`, not safe `state()` which panics on any kernel value outside its 9-variant
        // enum (aborting through mic_arbitrator's catch_unwind); the raw constant fails safe.
        const SND_PCM_STATE_PREPARED: std::os::raw::c_int = 2;
        if self.pcm.state_raw() == SND_PCM_STATE_PREPARED {
            self.pcm.start()?;
        }
        Ok(())
    }
}

// Maps [`AlsaReadOutcome`] onto [`super::ReadOutcome`]: `Frames(0)` -> `EndOfStream` (EOF on a
// closed device); recovery stays off-trait on the concrete variant (mock has no analogue).
impl super::MicSource for AlsaSource {
    fn id(&self) -> &MicId {
        AlsaSource::id(self)
    }
    fn channels(&self) -> u16 {
        AlsaSource::channels(self)
    }
    fn rate(&self) -> u32 {
        AlsaSource::rate(self)
    }
    fn period_size(&self) -> usize {
        AlsaSource::period_size(self)
    }
    fn effective_whitelist(&self) -> &[u16] {
        AlsaSource::effective_whitelist(self)
    }
    fn read_interleaved(&mut self, out: &mut [f32]) -> Result<ReadOutcome, ReadError> {
        let timeout = self.default_read_timeout();
        match self.read_with_timeout(out, timeout) {
            Ok(AlsaReadOutcome::Timeout) => Ok(ReadOutcome::Timeout),
            Ok(AlsaReadOutcome::Frames(0)) => Ok(ReadOutcome::EndOfStream),
            Ok(AlsaReadOutcome::Frames(n)) => {
                // Clamp so a driver returning > period_size gets a bounded outcome instead of an
                // OOB panic in `process_period`; warn once per source.
                if n > self.period_size && !self.overlong_read_warned {
                    self.overlong_read_warned = true;
                    tracing::warn!(
                        target: "audio_io",
                        mic_id = %self.id,
                        observed = n,
                        period_size = self.period_size,
                        "ALSA readi returned more frames than negotiated period_size; clamping \
                         (further occurrences silenced)",
                    );
                }
                let clamped = n.min(self.period_size);
                Ok(ReadOutcome::Frames(
                    NonZeroUsize::new(clamped).expect("n > 0 in this match arm; clamp preserves"),
                ))
            }
            Err(e) => Err(ReadError::Alsa(e)),
        }
    }
}

/// True iff `e` is `EAGAIN` / `EWOULDBLOCK` (equal on Linux; both checked for portability).
fn is_eagain(e: &alsa::Error) -> bool {
    let code = e.errno();
    code == libc::EAGAIN || code == libc::EWOULDBLOCK
}

/// Intersect `raw` with `[0..actual_channels)`, sorted + deduped. Out-of-range entries are
/// dropped silently here; [`AlsaSource::open`] logs them first.
fn intersect_whitelist(raw: &[u16], actual_channels: u16) -> Vec<u16> {
    let mut effective: Vec<u16> = raw
        .iter()
        .copied()
        .filter(|&ch| ch < actual_channels)
        .collect();
    effective.sort_unstable();
    effective.dedup();
    effective
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ids::MicId;

    #[test]
    #[ignore = "requires a Linux ALSA environment with no bogus123 PCM"]
    fn opening_invalid_device_returns_error() {
        let cand = MicCandidate {
            id: MicId::from_static("test-bogus"),
            source: CandidateSource::Alsa {
                hw_spec: "bogus_device_does_not_exist_42".into(),
                period_size: 1024,
                buffer_size: 4096,
            },
            channels: vec![0],
        };
        let res = AlsaSource::open(&cand);
        assert!(res.is_err(), "expected open to fail; got {res:?}");
    }

    #[test]
    fn intersect_whitelist_drops_out_of_range_entries() {
        let raw: Vec<u16> = vec![0, 1, 5, 7];
        let effective = intersect_whitelist(&raw, 2);
        assert_eq!(effective, vec![0_u16, 1]);
    }

    #[test]
    fn intersect_whitelist_returns_empty_when_all_entries_are_out_of_range() {
        let raw: Vec<u16> = vec![5, 7];
        let effective = intersect_whitelist(&raw, 2);
        assert!(effective.is_empty(), "all entries should be dropped");
    }

    #[test]
    fn intersect_whitelist_sorts_and_dedups() {
        // A duplicate index would corrupt per-slot RMS.
        let raw: Vec<u16> = vec![3, 0, 2, 0, 3];
        let effective = intersect_whitelist(&raw, 4);
        assert_eq!(effective, vec![0_u16, 2, 3]);
    }

    #[test]
    fn intersect_whitelist_passes_through_in_range_entries() {
        let raw: Vec<u16> = vec![0, 1, 2, 3];
        let effective = intersect_whitelist(&raw, 4);
        assert_eq!(effective, vec![0_u16, 1, 2, 3]);
    }

    /// Pin [`AlsaReadOutcome`]'s shape (public contract with the arbitrator's match); fails to
    /// compile if a variant disappears.
    #[test]
    fn alsa_read_outcome_variants_pin_shape() {
        let f = AlsaReadOutcome::Frames(42);
        let t = AlsaReadOutcome::Timeout;
        assert_ne!(f, t);
        match f {
            AlsaReadOutcome::Frames(n) => assert_eq!(n, 42),
            AlsaReadOutcome::Timeout => panic!("expected Frames"),
        }
        match t {
            AlsaReadOutcome::Frames(_) => panic!("expected Timeout"),
            AlsaReadOutcome::Timeout => {}
        }
    }

    /// `is_eagain` recognises EAGAIN/EWOULDBLOCK only; other errnos (EPIPE/ENODEV) must surface
    /// to `try_recover`.
    #[test]
    fn is_eagain_recognises_eagain_and_ewouldblock_only() {
        assert!(
            is_eagain(&alsa::Error::new("test", libc::EAGAIN)),
            "EAGAIN must be classified as eagain",
        );
        assert!(
            is_eagain(&alsa::Error::new("test", libc::EWOULDBLOCK)),
            "EWOULDBLOCK must be classified as eagain",
        );
        for code in [
            libc::EPIPE,
            libc::ESTRPIPE,
            libc::ENODEV,
            libc::EINTR,
            libc::EIO,
        ] {
            assert!(
                !is_eagain(&alsa::Error::new("test", code)),
                "errno {code} must NOT be classified as eagain",
            );
        }
    }
}
