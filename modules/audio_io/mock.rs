//! Synthetic [`Waveform`] shapes carried by
//! [`crate::audio_io::mic_arbitrator::CandidateSource::Mock`]; synthesis lives
//! in `crate::audio_io::source::mock`.

/// Tests/dev only; production uses
/// [`crate::audio_io::mic_arbitrator::CandidateSource::Alsa`].
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Waveform {
    /// Low-RMS reference for channel-arbitration tests.
    Silence,
    /// Phase continuous across blocks.
    Sine {
        freq_hz: f32,
        /// Linear; 1.0 saturates downstream nonlinearities.
        amplitude: f32,
    },
    /// Deterministic white noise via a 64-bit LCG; same `seed` is bit-identical.
    WhiteNoise { amplitude: f32, seed: u64 },
    /// Index ramp `(absolute_sample_idx & 0xFFFF) as f32`.
    Counter,
    /// Sine alternating `high_amp`/`low_amp` on a 50/50 duty cycle with phase
    /// continuous across transitions (no click); two channels with opposite
    /// `inverted` exercise the arbitrator's dwell+hysteresis auto-switch, so any
    /// discontinuity isolates an arbitrator-side issue.
    PingPongSine {
        freq_hz: f32,
        high_amp: f32,
        low_amp: f32,
        /// Per half-cycle; `0` is clamped to 1 in synthesis (div-by-zero modulo).
        half_period_samples: u32,
        /// `false` starts in `high_amp` half, `true` in `low_amp`.
        inverted: bool,
    },
}
