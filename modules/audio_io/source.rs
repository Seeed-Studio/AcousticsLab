//! Capture sources: ALSA (production, Linux only) and Mock (cross-platform).
//!
//! [`MicSource`] is a sealed trait unifying the read surface; the arbitrator
//! pattern-matches [`ActiveSource`] for source-specific recovery but reads via
//! the trait. `AlsaSource` is plain text not an intra-doc link so default-feature
//! (alsa-less) doc builds don't warn.

#[cfg(all(target_os = "linux", feature = "alsa-real"))]
pub mod alsa;
pub mod mock;

#[cfg(all(target_os = "linux", feature = "alsa-real"))]
pub use alsa::AlsaSource;
pub use mock::MockSource;

use crate::audio_io::mic_arbitrator::{CandidateError, CandidateSource, MicCandidate};
use crate::common::ids::MicId;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Sealed marker so external crates can't add [`MicSource`] impls.
mod sealed {
    pub trait Sealed {}
}

/// Outcome of one [`MicSource::read_interleaved`] call; `out` is untouched for
/// every variant except [`Self::Frames`]. `Timeout`/`EndOfStream` are only
/// constructed by the cfg-gated alsa impl.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
pub enum ReadOutcome {
    /// `n` interleaved frames written into `out[..n * channels]`.
    Frames(NonZeroUsize),
    /// Bounded poll/sleep elapsed without data; arbitrator re-checks stop then re-reads the same source.
    Timeout,
    /// Cooperative stop observed during pacing; handled like [`Self::Timeout`], distinct for logs/tests.
    StopRequested,
    /// Source exhausted (ALSA short-read on a closed PCM); arbitrator tears down and re-resolves.
    EndOfStream,
}

/// Unified per-period read error, cfg-gated to the impls that compile.
#[derive(Debug)]
#[allow(dead_code)]
pub enum ReadError {
    #[cfg(all(target_os = "linux", feature = "alsa-real"))]
    Alsa(::alsa::Error),
    /// Uninhabited (sources that cannot fail mid-stream, e.g. [`MockSource`]) so an exhaustive `match` needs no wildcard.
    Mock(std::convert::Infallible),
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ReadError::Alsa(e) => std::fmt::Display::fmt(e, _f),
            ReadError::Mock(infallible) => match *infallible {},
        }
    }
}

impl std::error::Error for ReadError {}

/// Sealed read-dispatch trait. `Send` required (source owned on a `std::thread`),
/// `Sync` not; the unified [`ReadError`] (no associated type) keeps it `dyn`-compatible.
#[allow(dead_code)] // accessors consumed only by the alsa-real impl + tests
pub trait MicSource: sealed::Sealed + Send + std::fmt::Debug {
    fn id(&self) -> &MicId;
    fn channels(&self) -> u16;
    fn rate(&self) -> u32;
    fn period_size(&self) -> usize;
    fn effective_whitelist(&self) -> &[u16];
    /// Read up to one period of interleaved frames (sources own their pacing); see [`ReadOutcome`] for per-variant semantics.
    fn read_interleaved(&mut self, out: &mut [f32]) -> Result<ReadOutcome, ReadError>;
}

impl sealed::Sealed for MockSource {}
impl sealed::Sealed for ActiveSource {}
#[cfg(all(target_os = "linux", feature = "alsa-real"))]
impl sealed::Sealed for AlsaSource {}

/// Currently-active capture source; the arbitrator owns at most one. Variants cfg-gated so an `alsa-real`-less build carries no unreachable variant.
#[derive(Debug)]
pub enum ActiveSource {
    Mock(MockSource),
    #[cfg(all(target_os = "linux", feature = "alsa-real"))]
    Alsa(AlsaSource),
}

impl ActiveSource {
    /// Stable id this source was opened for; the loop diffs it against the desired id to detect mic-change requests.
    pub fn id(&self) -> &MicId {
        match self {
            ActiveSource::Mock(s) => s.id(),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.id(),
        }
    }

    /// Total interleaved channels; may exceed whitelist length (non-whitelisted indices are demuxed and discarded, never buffered).
    pub fn channels(&self) -> u16 {
        match self {
            ActiveSource::Mock(s) => s.channels(),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.channels(),
        }
    }

    /// Source-native rate; arbitrator builds a resampler iff it differs from [`crate::common::dims::SampleRate::VALUE`].
    pub fn rate(&self) -> u32 {
        match self {
            ActiveSource::Mock(s) => s.rate(),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.rate(),
        }
    }

    /// Frames per `read_interleaved` call; sizes the arbitrator's per-channel scratch buffers once at open.
    pub fn period_size(&self) -> usize {
        match self {
            ActiveSource::Mock(s) => s.period_size(),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.period_size(),
        }
    }

    /// Whitelist intersected once at open with the device's channel count: `AlsaSource` drops operator-listed indices the device lacks, [`MockSource`] keeps every requested index (sorted + deduped, no device-count intersection).
    pub fn effective_whitelist(&self) -> &[u16] {
        match self {
            ActiveSource::Mock(s) => s.effective_whitelist(),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.effective_whitelist(),
        }
    }
}

/// Mic-open failures from [`open_source`]; per-period read errors are [`ReadError`] instead.
#[derive(Debug)]
pub enum OpenError {
    /// Static [`MicCandidate::validate`] rejected the candidate.
    InvalidCandidate(MicId, CandidateError),
    /// `CandidateSource::Alsa` requested but built without `alsa-real`; [`crate::audio_io::mic_arbitrator::MicSelection::FirstAvailable`] callers fall through to the next candidate, `Fixed` callers stay inert.
    #[allow(dead_code)] // constructed only on non-Linux / no-`alsa-real` builds
    AlsaNotCompiledIn(MicId),
    /// Source-specific open failure; `String` is operator-readable text (ALSA's `alsa::Error` stringified so callers don't depend on alsa-rs).
    SourceUnavailable(MicId, String),
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::InvalidCandidate(id, e) => {
                write!(f, "candidate {id} invalid: {e}")
            }
            OpenError::AlsaNotCompiledIn(id) => write!(
                f,
                "candidate {id} requires the `alsa-real` feature (built without it)",
            ),
            OpenError::SourceUnavailable(id, msg) => {
                write!(f, "candidate {id} unavailable: {msg}")
            }
        }
    }
}

impl std::error::Error for OpenError {}

impl OpenError {
    pub(crate) fn mic_id(&self) -> &MicId {
        match self {
            OpenError::InvalidCandidate(id, _)
            | OpenError::AlsaNotCompiledIn(id)
            | OpenError::SourceUnavailable(id, _) => id,
        }
    }
}

/// Open a capture source for `candidate`, validating statically first. `stop`
/// lets [`MockSource`] interrupt its real-time pacing sleep on shutdown; ALSA
/// ignores it since `readi` returns fast enough for the per-period stop check.
pub fn open_source(
    candidate: &MicCandidate,
    stop: Arc<AtomicBool>,
) -> Result<ActiveSource, OpenError> {
    candidate
        .validate()
        .map_err(|e| OpenError::InvalidCandidate(candidate.id.clone(), e))?;

    match &candidate.source {
        CandidateSource::Mock { .. } => MockSource::open(candidate, stop)
            .map(ActiveSource::Mock)
            .map_err(|msg| OpenError::SourceUnavailable(candidate.id.clone(), msg)),
        CandidateSource::Alsa { .. } => open_alsa(candidate),
    }
}

/// Isolates the `alsa-real` cfg gate so [`open_source`] stays cfg-free.
#[cfg(all(target_os = "linux", feature = "alsa-real"))]
fn open_alsa(candidate: &MicCandidate) -> Result<ActiveSource, OpenError> {
    AlsaSource::open(candidate)
        .map(ActiveSource::Alsa)
        .map_err(|msg| OpenError::SourceUnavailable(candidate.id.clone(), msg))
}

#[cfg(not(all(target_os = "linux", feature = "alsa-real")))]
fn open_alsa(candidate: &MicCandidate) -> Result<ActiveSource, OpenError> {
    Err(OpenError::AlsaNotCompiledIn(candidate.id.clone()))
}

// Compile-time guard that `MicSource` stays dyn-compatible.
#[cfg(test)]
const _: fn() = || {
    fn assert_obj_safe<T: ?Sized>() {}
    assert_obj_safe::<dyn MicSource>();
};

impl MicSource for ActiveSource {
    fn id(&self) -> &MicId {
        ActiveSource::id(self)
    }
    fn channels(&self) -> u16 {
        ActiveSource::channels(self)
    }
    fn rate(&self) -> u32 {
        ActiveSource::rate(self)
    }
    fn period_size(&self) -> usize {
        ActiveSource::period_size(self)
    }
    fn effective_whitelist(&self) -> &[u16] {
        ActiveSource::effective_whitelist(self)
    }
    fn read_interleaved(&mut self, out: &mut [f32]) -> Result<ReadOutcome, ReadError> {
        match self {
            ActiveSource::Mock(s) => s.read_interleaved(out),
            #[cfg(all(target_os = "linux", feature = "alsa-real"))]
            ActiveSource::Alsa(s) => s.read_interleaved(out),
        }
    }
}
