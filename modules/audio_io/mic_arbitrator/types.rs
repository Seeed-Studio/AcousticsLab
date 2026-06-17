//! Mic-arbitrator data model: candidates, catalogue, policy, and validation.

use crate::audio_io::mock::Waveform;
use crate::common::ids::MicId;
use std::sync::Arc;

/// A microphone the daemon may use; a willingness, not an open device (at most one open at a time).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MicCandidate {
    /// Stable id used by policy (NOT `hw_spec`, which churns on USB hot-plug).
    pub id: MicId,
    pub source: CandidateSource,
    /// Channel-index whitelist into the interleaved frame; only these arbitrate, intersected with the device's real channel count at open time (no OOB read).
    pub channels: Vec<u16>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CandidateSource {
    /// ALSA hardware, real only on Linux + `alsa-real`; elsewhere open fails (arbitrator falls through or stays inert per [`MicSelection`]).
    Alsa {
        /// E.g. `"hw:1,0"` or `"plughw:1,0"` (`plug` adds conversion latency but tolerates quirky USB devices).
        hw_spec: String,
        /// Frames per `readi` call (1024 ~= 23 ms at 44.1 kHz).
        #[serde(default = "default_period_size")]
        period_size: usize,
        /// 4x `period_size` is the canonical ALSA default.
        #[serde(default = "default_buffer_size")]
        buffer_size: usize,
    },
    Mock {
        /// One waveform per device-channel (`channels` indexes this Vec); out-of-whitelist entries are synthesized for LCG determinism but dropped before the audio buffer.
        waveforms: Vec<Waveform>,
        /// Synthesis chunk paced in real time.
        #[serde(default = "default_mock_period")]
        period_size: usize,
        /// Non-canonical values exercise the per-channel resampler.
        #[serde(default = "default_mock_rate")]
        sample_rate: u32,
    },
}

fn default_period_size() -> usize {
    1024
}
fn default_buffer_size() -> usize {
    1024 * 4
}
fn default_mock_period() -> usize {
    512
}
fn default_mock_rate() -> u32 {
    crate::common::dims::SampleRate::VALUE
}

/// Launch-time manifest of available mics, immutable for the daemon's lifetime (edited via launch-config TOML + restart); [`MicPolicy`] references by id.
#[derive(Clone, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MicCatalogue {
    pub candidates: Vec<MicCandidate>,
}

impl MicCatalogue {
    /// Per-candidate well-formedness plus catalogue-level id uniqueness at launch-config load; duplicate-id returns the second occurrence.
    pub fn validate(&self) -> Result<(), (MicId, CandidateError)> {
        for c in &self.candidates {
            if let Err(e) = c.validate() {
                return Err((c.id.clone(), e));
            }
        }
        let mut seen = std::collections::HashSet::with_capacity(self.candidates.len());
        for c in &self.candidates {
            if !seen.insert(&c.id) {
                return Err((c.id.clone(), CandidateError::DuplicateMicId(c.id.clone())));
            }
        }
        Ok(())
    }

    pub fn find(&self, id: &MicId) -> Option<&MicCandidate> {
        self.candidates.iter().find(|c| &c.id == id)
    }
}

/// Catalogue + live policy read by the arbitrator's hot loop via `Arc<arc_swap::ArcSwap<MicSettings>>`; catalogue is `Arc`-wrapped so a policy-change rebuild reuses it without cloning the `Vec<MicCandidate>`.
#[derive(Clone, Debug, PartialEq)]
pub struct MicSettings {
    pub catalogue: Arc<MicCatalogue>,
    pub policy: MicPolicy,
}

impl Default for MicSettings {
    fn default() -> Self {
        Self {
            catalogue: Arc::new(MicCatalogue::default()),
            policy: MicPolicy::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MicPolicy {
    pub mic: MicSelection,
    pub channel: ChannelSelection,
}

impl Default for MicPolicy {
    fn default() -> Self {
        Self {
            mic: MicSelection::FirstAvailable,
            channel: ChannelSelection::Auto,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum MicSelection {
    /// First candidate (declaration order) that opens wins; re-walks from the top on hot-unplug of the active mic.
    FirstAvailable,
    /// Always the named mic; retries forever (rate-limited), never fails over.
    Fixed { id: MicId },
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChannelSelection {
    /// RMS-arbitrated across the whitelist with hysteresis + dwell ("loudest wins").
    Auto,
    /// Named index; falls back to [`Self::Auto`] for THIS mic (not silence) if the index isn't in the active mic's whitelist.
    Fixed { channel: u16 },
}

/// Errors from [`MicCandidate::validate`] and [`MicCatalogue::validate`]; errors depending on the device's actual channel count surface at open time instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateError {
    EmptyChannels,
    /// Distinct indices required for the per-slot RMS state.
    DuplicateChannel(u16),
    /// Shared [`MicId`] (operator copy-paste typo): only the first resolves, the second is silently dead.
    DuplicateMicId(MicId),
    /// Above [`MAX_CHANNEL_INDEX`]; defends downstream `u16 -> u32 + 1` ALSA arithmetic against silent overflow.
    ChannelIndexTooLarge {
        channel: u16,
        cap: u16,
    },
    EmptyHwSpec,
    InvalidPeriodSize(usize),
    /// `buffer_size < period_size`; ALSA would refuse this anyway.
    InvalidBufferSize {
        period: usize,
        buffer: usize,
    },
    EmptyMockWaveforms,
    /// Too few `Mock` waveforms to cover the whitelist's max index.
    MockWhitelistOutOfRange {
        whitelist_max: u16,
        waveform_count: u16,
    },
    InvalidMockSampleRate(u32),
    PeriodSizeTooLarge {
        period: usize,
        cap: usize,
    },
    /// `16 * period_size` or absolute cap exceeded (samples-vs-ms typo).
    BufferSizeTooLarge {
        buffer: usize,
        period: usize,
        absolute_cap: usize,
        period_multiplier_cap: usize,
    },
    TooManyChannels {
        count: usize,
        cap: usize,
    },
    /// `sample_rate` outside [`MIN_SAMPLE_RATE`]..=[`MAX_SAMPLE_RATE`]; the resampler ratio either wedges or thrashes multi-MB sinc tables.
    SampleRateOutOfRange {
        rate: u32,
        min: u32,
        max: u32,
    },
    /// `period_size * channels` overflows `usize`; defence in depth via `checked_mul` should a per-field cap later be relaxed.
    ScratchSizeOverflow {
        period: usize,
        channels: usize,
    },
}

/// Sanity cap on whitelist channel indices: above any real device yet keeps `whitelist_max + 1` within [`u16::MAX`] so ALSA's `u16 -> u32 -> u16` channel-negotiation round-trip cannot wrap.
pub const MAX_CHANNEL_INDEX: u16 = 1023;

/// Cap on channels negotiated with an ALSA device: a driver reporting `channels_max = u16::MAX` would let `boot()` allocate `period * n * 4 bytes` ~= 4 GiB on one probe, but the daemon never indexes past [`MAX_CHANNEL_INDEX`] so `+ 1` is the largest legitimate need.
pub const MAX_NEGOTIABLE_CHANNELS: u32 = MAX_CHANNEL_INDEX as u32 + 1;

/// Cap on whitelist channel / `Mock` waveform count: real arrays expose 1-8 channels, and per-slot state + per-channel resamplers (~140 KB sinc each) make excess expensive.
pub const MAX_CHANNELS: usize = 8;

/// Cap on `period_size` (frames per `read_interleaved`): [`crate::audio_buffer::Writer::push`] needs `<= capacity / 4`, and at canonical capacity 262 144 (margin 65 536) 8192 stays 8x below (~186 ms at 44.1 kHz; production 1024 = ~23 ms). Const because validation predates the buffer.
pub const MAX_PERIOD_FRAMES: usize = 8192;

/// Absolute cap on `buffer_size`, defending against an oversized period that passed its own cap; the companion `16 * period_size` check catches ms typos.
pub const MAX_BUFFER_FRAMES: usize = MAX_PERIOD_FRAMES * 16;

/// Lower bound on any source rate (`Mock { sample_rate }`, ALSA-negotiated rate, WAV-ingest header rate); below 8 kHz the resampler ratio `44_100 / from_sr` explodes and the sinc kernel aliases audibly.
pub const MIN_SAMPLE_RATE: u32 = 8_000;

/// Upper bound on any source rate (`Mock { sample_rate }`, ALSA-negotiated rate, WAV-ingest header rate); above 192 kHz per-period sample counts balloon and such rates are studio-equipment-only.
pub const MAX_SAMPLE_RATE: u32 = 192_000;

impl std::fmt::Display for CandidateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CandidateError::EmptyChannels => f.write_str("channels whitelist is empty"),
            CandidateError::DuplicateChannel(c) => {
                write!(f, "duplicate channel {c} in whitelist")
            }
            CandidateError::DuplicateMicId(id) => {
                write!(f, "duplicate mic id '{id}' in catalogue")
            }
            CandidateError::ChannelIndexTooLarge { channel, cap } => {
                write!(f, "channel index {channel} exceeds cap {cap}",)
            }
            CandidateError::EmptyHwSpec => f.write_str("alsa hw_spec is empty"),
            CandidateError::InvalidPeriodSize(n) => {
                write!(f, "period_size must be > 0, got {n}")
            }
            CandidateError::InvalidBufferSize { period, buffer } => {
                write!(f, "buffer_size {buffer} must be >= period_size {period}",)
            }
            CandidateError::EmptyMockWaveforms => f.write_str("mock waveforms is empty"),
            CandidateError::MockWhitelistOutOfRange {
                whitelist_max,
                waveform_count,
            } => write!(
                f,
                "whitelist asks for channel {whitelist_max} but mock has only {waveform_count} waveforms",
            ),
            CandidateError::InvalidMockSampleRate(r) => {
                write!(f, "mock sample_rate must be > 0, got {r}")
            }
            CandidateError::PeriodSizeTooLarge { period, cap } => {
                write!(f, "period_size {period} exceeds cap {cap} frames",)
            }
            CandidateError::BufferSizeTooLarge {
                buffer,
                period,
                absolute_cap,
                period_multiplier_cap,
            } => write!(
                f,
                "buffer_size {buffer} exceeds cap (absolute {absolute_cap} or \
                 {period_multiplier_cap}x period_size = {})",
                period.saturating_mul(*period_multiplier_cap),
            ),
            CandidateError::TooManyChannels { count, cap } => {
                write!(f, "channels/waveforms count {count} exceeds cap {cap}",)
            }
            CandidateError::SampleRateOutOfRange { rate, min, max } => write!(
                f,
                "sample_rate {rate} outside supported range [{min}, {max}] Hz",
            ),
            CandidateError::ScratchSizeOverflow { period, channels } => write!(
                f,
                "period_size {period} * channels {channels} overflows usize",
            ),
        }
    }
}

impl std::error::Error for CandidateError {}

impl MicCandidate {
    /// Static validation catching operator typos at config load; does NOT check the device's actual channel count (the source-open path's job).
    pub fn validate(&self) -> Result<(), CandidateError> {
        if self.channels.is_empty() {
            return Err(CandidateError::EmptyChannels);
        }
        // Length cap before per-index sanity so the count, not a per-index error, is reported.
        if self.channels.len() > MAX_CHANNELS {
            return Err(CandidateError::TooManyChannels {
                count: self.channels.len(),
                cap: MAX_CHANNELS,
            });
        }
        // Cap before duplicate detection so the offending index beats "duplicate at 65535".
        for &ch in &self.channels {
            if ch > MAX_CHANNEL_INDEX {
                return Err(CandidateError::ChannelIndexTooLarge {
                    channel: ch,
                    cap: MAX_CHANNEL_INDEX,
                });
            }
        }
        // Clone before sorting to preserve the caller's whitelist order.
        let mut sorted = self.channels.clone();
        sorted.sort_unstable();
        for w in sorted.windows(2) {
            if w[0] == w[1] {
                return Err(CandidateError::DuplicateChannel(w[0]));
            }
        }

        match &self.source {
            CandidateSource::Alsa {
                hw_spec,
                period_size,
                buffer_size,
            } => {
                if hw_spec.is_empty() {
                    return Err(CandidateError::EmptyHwSpec);
                }
                if *period_size == 0 {
                    return Err(CandidateError::InvalidPeriodSize(*period_size));
                }
                if *period_size > MAX_PERIOD_FRAMES {
                    return Err(CandidateError::PeriodSizeTooLarge {
                        period: *period_size,
                        cap: MAX_PERIOD_FRAMES,
                    });
                }
                if *buffer_size < *period_size {
                    return Err(CandidateError::InvalidBufferSize {
                        period: *period_size,
                        buffer: *buffer_size,
                    });
                }
                // Either cap catches a samples-vs-ms typo (e.g. 44_100 = "1 second").
                let multiplier_cap = period_size.saturating_mul(16);
                if *buffer_size > MAX_BUFFER_FRAMES || *buffer_size > multiplier_cap {
                    return Err(CandidateError::BufferSizeTooLarge {
                        buffer: *buffer_size,
                        period: *period_size,
                        absolute_cap: MAX_BUFFER_FRAMES,
                        period_multiplier_cap: 16,
                    });
                }
                // Device channel count unknown here; bound open-time scratch (`period_size * channels`) by whitelist `max + 1`.
                let worst_case_channels =
                    self.channels.iter().copied().max().unwrap_or(0) as usize + 1;
                if period_size.checked_mul(worst_case_channels).is_none() {
                    return Err(CandidateError::ScratchSizeOverflow {
                        period: *period_size,
                        channels: worst_case_channels,
                    });
                }
            }
            CandidateSource::Mock {
                waveforms,
                period_size,
                sample_rate,
            } => {
                if waveforms.is_empty() {
                    return Err(CandidateError::EmptyMockWaveforms);
                }
                // Mock's device channel count is `waveforms.len()`; cap before whitelist-out-of-range so the diagnostic error wins.
                if waveforms.len() > MAX_CHANNELS {
                    return Err(CandidateError::TooManyChannels {
                        count: waveforms.len(),
                        cap: MAX_CHANNELS,
                    });
                }
                let max_ch = *self.channels.iter().max().expect("non-empty above");
                if max_ch as usize >= waveforms.len() {
                    return Err(CandidateError::MockWhitelistOutOfRange {
                        whitelist_max: max_ch,
                        waveform_count: waveforms.len() as u16,
                    });
                }
                if *period_size == 0 {
                    return Err(CandidateError::InvalidPeriodSize(*period_size));
                }
                if *period_size > MAX_PERIOD_FRAMES {
                    return Err(CandidateError::PeriodSizeTooLarge {
                        period: *period_size,
                        cap: MAX_PERIOD_FRAMES,
                    });
                }
                if *sample_rate == 0 {
                    return Err(CandidateError::InvalidMockSampleRate(*sample_rate));
                }
                if *sample_rate < MIN_SAMPLE_RATE || *sample_rate > MAX_SAMPLE_RATE {
                    return Err(CandidateError::SampleRateOutOfRange {
                        rate: *sample_rate,
                        min: MIN_SAMPLE_RATE,
                        max: MAX_SAMPLE_RATE,
                    });
                }
                if period_size.checked_mul(waveforms.len()).is_none() {
                    return Err(CandidateError::ScratchSizeOverflow {
                        period: *period_size,
                        channels: waveforms.len(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PolicyValidationError {
    #[error("policy.mic.id = '{0}' does not match any catalogue candidate")]
    UnknownMicId(MicId),
    /// Fixed channel isn't in the Fixed mic's whitelist; only checkable for Fixed mics (`FirstAvailable` resolves at runtime).
    #[error("policy.channel = {channel} not in candidate '{mic}' channels {available:?}")]
    ChannelNotAvailable {
        mic: MicId,
        channel: u16,
        available: Vec<u16>,
    },
}

impl MicPolicy {
    /// Verify the policy is satisfiable against `catalogue` (boot, hot-reload, `POST /mic/policy`): `FirstAvailable` is always valid, a Fixed mic must match a candidate, and a Fixed channel must be in that candidate's whitelist.
    pub fn validate_against(&self, catalogue: &MicCatalogue) -> Result<(), PolicyValidationError> {
        let MicSelection::Fixed { id } = &self.mic else {
            return Ok(());
        };
        let cand = catalogue
            .find(id)
            .ok_or_else(|| PolicyValidationError::UnknownMicId(id.clone()))?;
        if let ChannelSelection::Fixed { channel } = &self.channel
            && !cand.channels.contains(channel)
        {
            return Err(PolicyValidationError::ChannelNotAvailable {
                mic: id.clone(),
                channel: *channel,
                available: cand.channels.clone(),
            });
        }
        Ok(())
    }
}
