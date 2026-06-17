//! Multi-channel synthetic capture source: one [`Waveform`] per channel,
//! interleaved into the buffer the arbitrator reads.
//!
//! `read_interleaved` paces in wall-clock inside the call (the arbitrator is
//! single-threaded) so the head advances at real-time cadence, not memory
//! speed; otherwise consumers observe "future" audio. A stall exceeding one
//! `block_dur` resets the pacing target to `now` rather than bursting
//! stale-timestamped periods (ALSA xrun-recover). `stop` is polled in 2 ms
//! slices so teardown latency is ~2 ms. `WhiteNoise { seed }` is bit-identical
//! across runs with independent per-channel rng state.

use crate::audio_io::mic_arbitrator::{CandidateSource, MicCandidate};
use crate::audio_io::mock::Waveform;
use crate::audio_io::source::{ReadError, ReadOutcome};
use crate::common::ids::MicId;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Synthetic multi-channel capture source.
#[derive(Debug)]
pub struct MockSource {
    id: MicId,
    waveforms: Vec<Waveform>,
    /// Sorted + deduped for parity with `AlsaSource`.
    effective_whitelist: Vec<u16>,
    period_size: usize,
    sample_rate: u32,
    /// Per-channel LCG state (seed from `WhiteNoise.seed` or 1), preserved
    /// across periods so noise is continuous.
    rng_states: Vec<u64>,
    /// Wall-clock target for the next period, advanced one `block_dur` per
    /// read; reset to `now` by the skew clamp when drift exceeds one
    /// `block_dur`.
    next_block_at: Instant,
    /// Samples-per-channel emitted so far; indexes each waveform's analytic
    /// form to keep phase/counter continuous across periods.
    absolute_idx: u64,
    stop: Arc<AtomicBool>,
}

impl MockSource {
    /// Re-validates so callers bypassing the dispatcher still uphold the
    /// invariants the synthesis path's `expect`s/divides rely on.
    pub(crate) fn open(candidate: &MicCandidate, stop: Arc<AtomicBool>) -> Result<Self, String> {
        candidate
            .validate()
            .map_err(|e| format!("MockSource::open candidate failed validation: {e}"))?;
        let CandidateSource::Mock {
            waveforms,
            period_size,
            sample_rate,
        } = &candidate.source
        else {
            return Err("MockSource::open called on an Alsa candidate".into());
        };
        let (waveforms, period_size, sample_rate) = (waveforms.clone(), *period_size, *sample_rate);
        let rng_states = waveforms
            .iter()
            .map(|w| match w {
                // LCG seed 0 yields the all-zero stream; bump to 1.
                Waveform::WhiteNoise { seed, .. } => (*seed).max(1),
                _ => 1,
            })
            .collect();
        let mut effective_whitelist = candidate.channels.clone();
        effective_whitelist.sort_unstable();
        effective_whitelist.dedup();
        Ok(Self {
            id: candidate.id.clone(),
            waveforms,
            effective_whitelist,
            period_size,
            sample_rate,
            rng_states,
            next_block_at: Instant::now(),
            absolute_idx: 0,
            stop,
        })
    }

    pub fn id(&self) -> &MicId {
        &self.id
    }

    /// Number of interleaved channels (equals waveform count).
    pub fn channels(&self) -> u16 {
        self.waveforms.len() as u16
    }

    pub fn rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn period_size(&self) -> usize {
        self.period_size
    }

    /// Sorted + deduped; one accessor across variants like `AlsaSource`.
    pub fn effective_whitelist(&self) -> &[u16] {
        &self.effective_whitelist
    }

    /// Synthesize one period of interleaved frames into `out` (>=
    /// `period_size * channels`), pacing to the wall-clock target while
    /// polling `stop` in 2 ms slices. Returns [`ReadOutcome::StopRequested`]
    /// iff stop fired during the sleep, else [`ReadOutcome::Frames`] of
    /// `NonZero(period_size)`. Never fails.
    pub fn read_interleaved(&mut self, out: &mut [f32]) -> Result<ReadOutcome, ReadError> {
        let n_channels = self.waveforms.len();
        let needed = self.period_size * n_channels;
        debug_assert!(
            out.len() >= needed,
            "mock read buffer too small: {} < {needed}",
            out.len(),
        );

        // First call: `next_block_at` is open's `now`, so no sleep.
        let block_dur = Duration::from_secs_f64(self.period_size as f64 / self.sample_rate as f64);
        let now = Instant::now();
        // Skew clamp: a stall leaving `next_block_at` >1 block_dur in the past
        // would burst stale-timestamped periods; reset to now. The one-block
        // threshold lets legitimate small drift pass.
        if now > self.next_block_at + block_dur {
            let skew = now.saturating_duration_since(self.next_block_at);
            tracing::debug!(
                target: "audio_io",
                skew_ms = skew.as_millis() as u64,
                block_dur_ms = block_dur.as_millis() as u64,
                "mock source skew clamp: consumer-side stall detected, resetting pacing",
            );
            self.next_block_at = now;
        }
        if now < self.next_block_at {
            let mut remaining = self.next_block_at - now;
            const STOP_POLL_SLICE: Duration = Duration::from_millis(2);
            while remaining > STOP_POLL_SLICE {
                if self.stop.load(Ordering::Acquire) {
                    return Ok(ReadOutcome::StopRequested);
                }
                thread::sleep(STOP_POLL_SLICE);
                let now2 = Instant::now();
                remaining = self.next_block_at.saturating_duration_since(now2);
            }
            if remaining > Duration::ZERO {
                thread::sleep(remaining);
            }
        }
        self.next_block_at += block_dur;

        let target = &mut out[..needed];
        // Per-waveform loops write only their strided slots; `Silence` writes
        // nothing and relies on this pre-zero.
        target.fill(0.0);
        for (ch, waveform) in self.waveforms.iter().enumerate() {
            let rng = &mut self.rng_states[ch];
            generate_channel_into(
                target,
                ch,
                n_channels,
                self.period_size,
                *waveform,
                self.sample_rate,
                self.absolute_idx,
                rng,
            );
        }
        self.absolute_idx += self.period_size as u64;
        // `period_size` validated nonzero at `MicCandidate::validate`.
        Ok(ReadOutcome::Frames(
            std::num::NonZeroUsize::new(self.period_size)
                .expect("MockSource::period_size must be nonzero"),
        ))
    }
}

impl super::MicSource for MockSource {
    fn id(&self) -> &MicId {
        MockSource::id(self)
    }
    fn channels(&self) -> u16 {
        MockSource::channels(self)
    }
    fn rate(&self) -> u32 {
        MockSource::rate(self)
    }
    fn period_size(&self) -> usize {
        MockSource::period_size(self)
    }
    fn effective_whitelist(&self) -> &[u16] {
        MockSource::effective_whitelist(self)
    }
    fn read_interleaved(
        &mut self,
        out: &mut [f32],
    ) -> Result<super::ReadOutcome, super::ReadError> {
        MockSource::read_interleaved(self, out)
    }
}

/// Write `period_size` samples of `waveform` into channel's strided slots.
#[allow(clippy::too_many_arguments)]
fn generate_channel_into(
    interleaved: &mut [f32],
    channel: usize,
    n_channels: usize,
    period_size: usize,
    waveform: Waveform,
    sample_rate: u32,
    start: u64,
    rng_state: &mut u64,
) {
    match waveform {
        Waveform::Silence => {
            // Pre-filled to 0.
        }
        Waveform::Sine { freq_hz, amplitude } => {
            // f64 phase: f32 quantizes integers past 2^24 (~6 min @ 44.1k),
            // distorting the sine; f64's 2^53 covers any mock duration.
            let omega = 2.0 * std::f64::consts::PI * freq_hz as f64 / sample_rate as f64;
            let amp = amplitude as f64;
            for f in 0..period_size {
                let n = (start + f as u64) as f64;
                interleaved[f * n_channels + channel] = (amp * (omega * n).sin()) as f32;
            }
        }
        Waveform::WhiteNoise { amplitude, .. } => {
            // Knuth's MMIX LCG: reproducible, not cryptographic.
            const MUL: u64 = 6_364_136_223_846_793_005;
            const ADD: u64 = 1_442_695_040_888_963_407;
            for f in 0..period_size {
                *rng_state = rng_state.wrapping_mul(MUL).wrapping_add(ADD);
                let bits = (*rng_state >> 32) as u32;
                let signed = bits as i32;
                let normalized = signed as f32 / i32::MAX as f32;
                interleaved[f * n_channels + channel] = normalized * amplitude;
            }
        }
        Waveform::Counter => {
            for f in 0..period_size {
                let absolute = start + f as u64;
                interleaved[f * n_channels + channel] = (absolute & 0xFFFF) as f32;
            }
        }
        Waveform::PingPongSine {
            freq_hz,
            high_amp,
            low_amp,
            half_period_samples,
            inverted,
        } => {
            // Continuous sine phase (f64 as in `Sine`); only the amplitude
            // envelope flips at half-cycle boundaries.
            let omega = 2.0 * std::f64::consts::PI * freq_hz as f64 / sample_rate as f64;
            let high = high_amp as f64;
            let low = low_amp as f64;
            // `.max(1)` guards the modulo against divide-by-zero when
            // `half_period_samples == 0` (degenerate output, no panic).
            let half = (half_period_samples as u64).max(1);
            let cycle = half.saturating_mul(2);
            for f in 0..period_size {
                let n = start + f as u64;
                let cycle_pos = n % cycle;
                let in_high_half = cycle_pos < half;
                let amp = if in_high_half ^ inverted { high } else { low };
                let n_f = n as f64;
                interleaved[f * n_channels + channel] = (amp * (omega * n_f).sin()) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ids::MicId;

    fn mock_candidate(channels: Vec<u16>, waveforms: Vec<Waveform>) -> MicCandidate {
        MicCandidate {
            id: MicId::from_static("mock:test"),
            source: CandidateSource::Mock {
                waveforms,
                period_size: 256,
                sample_rate: 44_100,
            },
            channels,
        }
    }

    fn frames_or_zero(r: Result<ReadOutcome, ReadError>) -> usize {
        match r.expect("mock read cannot fail") {
            ReadOutcome::Frames(n) => n.get(),
            ReadOutcome::StopRequested | ReadOutcome::Timeout | ReadOutcome::EndOfStream => 0,
        }
    }

    #[test]
    fn open_round_trips_id_and_shape() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = mock_candidate(vec![0, 1], vec![Waveform::Silence, Waveform::Silence]);
        let s = MockSource::open(&cand, stop).expect("open");
        assert_eq!(s.id(), &MicId::from_static("mock:test"));
        assert_eq!(s.channels(), 2);
        assert_eq!(s.rate(), 44_100);
        assert_eq!(s.period_size(), 256);
    }

    #[test]
    fn open_rejects_alsa_candidate() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("alsa-impostor"),
            source: CandidateSource::Alsa {
                hw_spec: "hw:1,0".into(),
                period_size: 1024,
                buffer_size: 4096,
            },
            channels: vec![0],
        };
        let err = MockSource::open(&cand, stop).expect_err("must reject");
        assert!(err.contains("Alsa"), "err = {err:?}");
    }

    /// `period_size = 0` would later panic on the `NonZeroUsize` `expect` in
    /// `read_interleaved`, so the constructor must reject it.
    #[test]
    fn open_rejects_zero_period_size() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("zero-period"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence; 2],
                period_size: 0,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let err = MockSource::open(&cand, stop).expect_err("must reject zero period");
        assert!(
            err.contains("validation") || err.contains("period_size"),
            "err = {err:?}",
        );
    }

    /// Rejects out-of-range sample_rate, guarding the resampler ratio.
    #[test]
    fn open_rejects_sample_rate_out_of_range() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("bad-rate"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence; 2],
                period_size: 512,
                // Above MAX_SAMPLE_RATE (192 kHz).
                sample_rate: 1_000_000,
            },
            channels: vec![0],
        };
        let err = MockSource::open(&cand, stop).expect_err("must reject extreme rate");
        assert!(
            err.contains("validation") || err.contains("sample_rate"),
            "err = {err:?}",
        );
    }

    /// First read immediate, subsequent reads pace to real-time; 4096-frame
    /// period (~93 ms) and 80% lower bound absorb `thread::sleep`
    /// over-delivery / scheduler noise.
    #[test]
    fn read_paces_to_real_time() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("mock:pace"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence],
                period_size: 4096,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop).expect("open");
        let mut buf = vec![0.0f32; s.period_size() * s.channels() as usize];

        let block_dur = Duration::from_secs_f64(s.period_size() as f64 / s.rate() as f64);
        let t0 = Instant::now();
        let n0 = frames_or_zero(s.read_interleaved(&mut buf));
        let t1 = Instant::now();
        let n1 = frames_or_zero(s.read_interleaved(&mut buf));
        let t2 = Instant::now();

        assert_eq!(n0, s.period_size());
        assert_eq!(n1, s.period_size());

        let first_dur = t1 - t0;
        assert!(
            first_dur < Duration::from_millis(5),
            "first read should be ~immediate; took {first_dur:?}",
        );
        let second_dur = t2 - t1;
        let lower = block_dur.mul_f32(0.80);
        assert!(
            second_dur >= lower,
            "second read should pace to >= {lower:?} (80% of {block_dur:?}); took {second_dur:?}",
        );
        // Upper bound catches a sleep-too-much regression.
        assert!(
            second_dur < block_dur * 2,
            "second read paced too long: {second_dur:?} > 2x {block_dur:?}",
        );
    }

    /// Per-channel waveforms land at the right interleaved positions.
    #[test]
    fn read_interleaves_per_channel_waveforms() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = mock_candidate(
            vec![0, 1, 2],
            vec![
                Waveform::Silence,
                Waveform::Counter,
                Waveform::Sine {
                    freq_hz: 1000.0,
                    amplitude: 1.0,
                },
            ],
        );
        let mut s = MockSource::open(&cand, stop).expect("open");
        let n_ch = s.channels() as usize;
        let mut buf = vec![1234.0f32; s.period_size() * n_ch];
        let n = frames_or_zero(s.read_interleaved(&mut buf));
        assert_eq!(n, s.period_size());

        for f in 0..s.period_size() {
            assert_eq!(buf[f * n_ch], 0.0, "ch0 frame {f}");
        }
        for f in 0..s.period_size() {
            let expected = (f as u64 & 0xFFFF) as f32;
            assert_eq!(buf[f * n_ch + 1], expected, "ch1 frame {f}");
        }
        let max = (0..s.period_size())
            .map(|f| buf[f * n_ch + 2])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max <= 1.0 + 1e-6, "ch2 sine max {max} > 1.0");
        let min = (0..s.period_size())
            .map(|f| buf[f * n_ch + 2])
            .fold(f32::INFINITY, f32::min);
        assert!(min >= -1.0 - 1e-6, "ch2 sine min {min} < -1.0");
    }

    /// Waveforms stay continuous across reads (`absolute_idx` advances
    /// by `period_size` per period).
    #[test]
    fn waveforms_are_continuous_across_reads() {
        let stop = Arc::new(AtomicBool::new(false));
        // Tiny period to keep wall-clock under a second.
        let cand = MicCandidate {
            id: MicId::from_static("mock:cont"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Counter],
                period_size: 64,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop).expect("open");
        let mut buf = vec![0.0f32; 64];

        let mut concat: Vec<f32> = Vec::new();
        for _ in 0..3 {
            let _ = s.read_interleaved(&mut buf);
            concat.extend_from_slice(&buf);
        }
        for (i, &v) in concat.iter().enumerate() {
            let expected = (i as u64 & 0xFFFF) as f32;
            assert_eq!(v, expected, "discontinuity at absolute idx {i}");
        }
    }

    /// Identical seed -> bit-identical samples across separate sources.
    #[test]
    fn white_noise_is_seed_deterministic() {
        let cand = MicCandidate {
            id: MicId::from_static("mock:noise"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::WhiteNoise {
                    amplitude: 1.0,
                    seed: 42,
                }],
                period_size: 128,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut a = MockSource::open(&cand, Arc::new(AtomicBool::new(false))).expect("a");
        let mut b = MockSource::open(&cand, Arc::new(AtomicBool::new(false))).expect("b");
        let mut buf_a = vec![0.0f32; 128];
        let mut buf_b = vec![0.0f32; 128];
        let _ = a.read_interleaved(&mut buf_a);
        let _ = b.read_interleaved(&mut buf_b);
        for (i, (x, y)) in buf_a.iter().zip(buf_b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "noise drift at sample {i}: {x} vs {y}",
            );
        }
    }

    /// Stop during the pacing sleep returns 0 promptly.
    #[test]
    fn stop_during_sleep_returns_zero_quickly() {
        let stop = Arc::new(AtomicBool::new(false));
        // Long period (~93 ms) so the second read has a real sleep to interrupt.
        let cand = MicCandidate {
            id: MicId::from_static("mock:stop"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence],
                period_size: 4096,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop.clone()).expect("open");
        let mut buf = vec![0.0f32; 4096];
        let n0 = frames_or_zero(s.read_interleaved(&mut buf));
        assert_eq!(n0, 4096);

        let stop_clone = stop.clone();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            stop_clone.store(true, Ordering::Release);
        });
        let t0 = Instant::now();
        let n1 = frames_or_zero(s.read_interleaved(&mut buf));
        let elapsed = t0.elapsed();
        t.join().unwrap();

        assert_eq!(n1, 0, "stop signaled mid-sleep should return 0");
        // 10 ms pre-signal sleep + 2 ms slice + slop << the 93 ms period.
        assert!(
            elapsed < Duration::from_millis(20),
            "stop response too slow: {elapsed:?}",
        );
    }

    /// Mid-sleep stop surfaces as [`ReadOutcome::StopRequested`], not `Frames`
    /// or `EndOfStream` (ALSA's closed-PCM variant).
    #[test]
    fn stop_during_sleep_returns_stop_requested_variant() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("mock:stop_variant"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence],
                period_size: 4096,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop.clone()).expect("open");
        let mut buf = vec![0.0f32; 4096];
        let _ = s.read_interleaved(&mut buf);
        let stop_clone = stop.clone();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            stop_clone.store(true, Ordering::Release);
        });
        let outcome = s.read_interleaved(&mut buf).expect("mock infallible");
        t.join().unwrap();
        assert_eq!(
            outcome,
            ReadOutcome::StopRequested,
            "mid-sleep stop must surface as StopRequested, not Frames/EndOfStream",
        );
    }

    /// Stop preset before `read_interleaved` is honoured by the pacing sleep.
    #[test]
    fn read_with_stop_preset_returns_zero() {
        let stop = Arc::new(AtomicBool::new(true));
        let cand = mock_candidate(vec![0], vec![Waveform::Silence]);
        let mut s = MockSource::open(&cand, stop).expect("open");
        let mut buf = vec![0.0f32; s.period_size()];
        // First call doesn't pace; the second does.
        let _ = s.read_interleaved(&mut buf);
        let t0 = Instant::now();
        let n = frames_or_zero(s.read_interleaved(&mut buf));
        assert_eq!(n, 0);
        assert!(
            t0.elapsed() < Duration::from_millis(5),
            "stop pre-set should return ~immediately",
        );
    }

    /// Skew clamp: after a long stall, pacing resumes from `now` (one read
    /// clamps, the next paces normally) rather than bursting N periods.
    #[test]
    fn read_clamps_skew_after_long_stall() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("mock:skew"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::Silence],
                period_size: 4096,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop).expect("open");
        let mut buf = vec![0.0f32; s.period_size()];

        // Prime pacing.
        let n0 = frames_or_zero(s.read_interleaved(&mut buf));
        assert_eq!(n0, s.period_size());

        // Simulate a long stall by rolling `next_block_at` 1 s into the past.
        let block_dur = Duration::from_secs_f64(s.period_size() as f64 / s.rate() as f64);
        s.next_block_at = Instant::now() - Duration::from_secs(1);

        // First post-stall read: clamped, so ~immediate.
        let t0 = Instant::now();
        let n1 = frames_or_zero(s.read_interleaved(&mut buf));
        let first_post = t0.elapsed();
        assert_eq!(n1, s.period_size());
        assert!(
            first_post < Duration::from_millis(5),
            "first post-stall read should return ~immediately (clamp resets pacing); \
             took {first_post:?}",
        );

        // Second post-stall read: pacing resumed from the clamp's `now`, so
        // it MUST sleep ~one block_dur (else we're bursting).
        let t1 = Instant::now();
        let n2 = frames_or_zero(s.read_interleaved(&mut buf));
        let second_post = t1.elapsed();
        assert_eq!(n2, s.period_size());
        let lower = block_dur.mul_f32(0.80);
        assert!(
            second_post >= lower,
            "second post-stall read should pace >= {lower:?} (80% of \
             {block_dur:?}); took {second_post:?} (clamp regression?)",
        );
    }

    /// `PingPongSine` alternates high/low amplitude at the half-period
    /// boundary, with `inverted` swapping which half is loud.
    #[test]
    fn ping_pong_sine_alternates_amplitude_at_half_period() {
        let stop = Arc::new(AtomicBool::new(false));
        // ch0 high-first, ch1 inverted (low-first), ch2 silence; half-period
        // 64 -> full cycle 128.
        let cand = MicCandidate {
            id: MicId::from_static("mock:pp"),
            source: CandidateSource::Mock {
                waveforms: vec![
                    Waveform::PingPongSine {
                        freq_hz: 4_410.0,
                        high_amp: 0.8,
                        low_amp: 0.05,
                        half_period_samples: 64,
                        inverted: false,
                    },
                    Waveform::PingPongSine {
                        freq_hz: 4_410.0,
                        high_amp: 0.8,
                        low_amp: 0.05,
                        half_period_samples: 64,
                        inverted: true,
                    },
                    Waveform::Silence,
                ],
                period_size: 256, // 2 full cycles
                sample_rate: 44_100,
            },
            channels: vec![0, 1, 2],
        };
        let mut s = MockSource::open(&cand, stop).expect("open");
        let n_ch = s.channels() as usize;
        let mut buf = vec![0.0f32; s.period_size() * n_ch];
        let _ = s.read_interleaved(&mut buf);

        // Peak |sample| over a window centered in a half-cycle.
        let peak_in = |ch: usize, start: usize, len: usize| -> f32 {
            (start..start + len)
                .map(|f| buf[f * n_ch + ch].abs())
                .fold(0.0_f32, f32::max)
        };
        // ch0: high [0,64), low [64,128), high [128,192).
        let ch0_high0 = peak_in(0, 16, 32);
        let ch0_low = peak_in(0, 80, 32);
        let ch0_high1 = peak_in(0, 144, 32);
        assert!(ch0_high0 > 0.5, "ch0 high half 0: peak {ch0_high0}");
        assert!(ch0_low < 0.1, "ch0 low half: peak {ch0_low}");
        assert!(ch0_high1 > 0.5, "ch0 high half 1: peak {ch0_high1}");

        // ch1 inverted: low [0,64), high [64,128).
        let ch1_low = peak_in(1, 16, 32);
        let ch1_high = peak_in(1, 80, 32);
        assert!(ch1_low < 0.1, "ch1 low half (inverted): peak {ch1_low}");
        assert!(ch1_high > 0.5, "ch1 high half (inverted): peak {ch1_high}");

        let ch2_peak = peak_in(2, 0, 256);
        assert_eq!(ch2_peak, 0.0);
    }

    /// `half_period_samples = 0` must not panic (`.max(1)` guard); a
    /// misconfigured setup must not crash the arbitrator thread.
    #[test]
    fn ping_pong_sine_zero_half_period_does_not_panic() {
        let stop = Arc::new(AtomicBool::new(false));
        let cand = MicCandidate {
            id: MicId::from_static("mock:pp-zero"),
            source: CandidateSource::Mock {
                waveforms: vec![Waveform::PingPongSine {
                    freq_hz: 1_000.0,
                    high_amp: 0.5,
                    low_amp: 0.05,
                    half_period_samples: 0,
                    inverted: false,
                }],
                period_size: 64,
                sample_rate: 44_100,
            },
            channels: vec![0],
        };
        let mut s = MockSource::open(&cand, stop).expect("open");
        let mut buf = vec![0.0f32; 64];
        let n = frames_or_zero(s.read_interleaved(&mut buf));
        assert_eq!(n, 64);
        assert!(buf.iter().all(|s| s.is_finite()));
    }

    /// Open + close many sources without leaking.
    #[test]
    fn many_open_close_cycles() {
        let cand = mock_candidate(vec![0], vec![Waveform::Silence]);
        for _ in 0..50 {
            let stop = Arc::new(AtomicBool::new(false));
            let mut s = MockSource::open(&cand, stop).expect("open");
            let mut buf = vec![0.0f32; s.period_size()];
            let _ = s.read_interleaved(&mut buf);
        }
    }
}
