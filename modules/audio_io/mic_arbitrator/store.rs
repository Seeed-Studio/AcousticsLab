//! Read-only `MicSettingsStore` trait for wait-free reads of the live [`MicSettings`].
//!
//! Lives in `audio_io` (not `common`): the [`Arc<MicSettings>`] return type would
//! pull this module's ALSA/mock/candidate-source deps into the contract module.

use crate::audio_io::mic_arbitrator::MicSettings;
use crate::common::version::ResourceVersion;
use std::sync::Arc;

/// Wait-free read-side view of the live mic settings (catalogue + policy).
///
/// `Send + Sync + 'static` so an `Arc<dyn MicSettingsStore>` can live on the
/// [`crate::audio_io::mic_arbitrator::MicArbitrator`]'s spawned thread state.
pub trait MicSettingsStore: Send + Sync + 'static {
    /// Wait-free read; the returned `Arc` is safe to hold across mutations.
    fn snapshot(&self) -> Arc<MicSettings>;

    /// Bumps on each successful mutation through the matching `MicSettingsHandle`.
    fn version(&self) -> ResourceVersion;

    /// Atomic `(snapshot, version)` read: a writer slipping between separate
    /// `snapshot()` + `version()` calls could pair a stale value with a newer
    /// version. Default = two non-atomic reads; `config::MicSettingsCell`
    /// overrides with a single `VersionedSwap::snapshot_with_version` load.
    fn snapshot_with_version(&self) -> (Arc<MicSettings>, ResourceVersion) {
        (self.snapshot(), self.version())
    }
}

/// Adapter from a bare `Arc<ArcSwap<MicSettings>>` to `Arc<dyn MicSettingsStore>`.
/// Version always reads [`ResourceVersion::ZERO`] (`ArcSwap` is versionless); real
/// versioning needs the production `config::MicSettingsCell`.
#[derive(Debug)]
pub struct ArcSwapStore(pub Arc<arc_swap::ArcSwap<MicSettings>>);

impl MicSettingsStore for ArcSwapStore {
    fn snapshot(&self) -> Arc<MicSettings> {
        self.0.load_full()
    }
    fn version(&self) -> ResourceVersion {
        ResourceVersion::ZERO
    }
}
