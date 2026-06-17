//! Mic settings store + handle traits + the `MicSettingsCell` impl.
//!
//! In `config` not `common`: signatures reference `audio_io` types, so `common`
//! would inherit `audio_io`'s alsa/rubato deps. `try_set_policy` is concrete,
//! not a generic `try_mutate`, to keep the trait object-safe (an `impl FnOnce`
//! arg would not be).

use crate::audio_io::mic_arbitrator::{MicCatalogue, MicPolicy, MicSettings, MicSettingsStore};
use crate::common::error::{Categorized, ErrorKind};
use crate::common::version::{ResourceVersion, SwapReceipt, VersionedSwap};
use crate::config::ConfigError;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum MicError {
    #[error("mic policy rejected: {0}")]
    Rejected(String),
    #[error("persist mic policy: {0}")]
    Persist(#[from] ConfigError),
}

impl Categorized for MicError {
    fn kind(&self) -> ErrorKind {
        match self {
            MicError::Rejected(_) => ErrorKind::UserInput,
            MicError::Persist(e) => e.kind(),
        }
    }
}

/// Extends read-only [`MicSettingsStore`] so one cell serves both trait objects
/// (api gets read+write, arbitrator gets wait-free reads).
pub trait MicSettingsHandle: MicSettingsStore {
    /// Validate against the catalogue, then atomically publish + persist to
    /// user-config TOML. Blocking (fsync): async callers must `spawn_blocking`.
    fn try_set_policy(&self, policy: MicPolicy) -> Result<SwapReceipt, MicError>;

    /// `try_set_policy` without persistence. ONLY for the watcher path: it runs
    /// inside `ConfigCell::mutate`'s callback, so persisting would re-enter
    /// `mutate` and trip the `IN_MUTATE` guard, and the on-disk edit it reacted
    /// to is already the source of truth.
    fn try_set_policy_no_persist(&self, policy: MicPolicy) -> Result<SwapReceipt, MicError>;
}

/// Both `dyn MicSettingsStore` and `dyn MicSettingsHandle` cast from the same
/// `Arc<MicSettingsCell>`.
#[derive(Debug)]
pub struct MicSettingsCell {
    inner: VersionedSwap<MicSettings>,
    /// Launch-immutable; Arc-cloned into every new `MicSettings` snapshot.
    catalogue: Arc<MicCatalogue>,
    config: Arc<crate::config::ConfigCell>,
}

impl MicSettingsCell {
    pub fn new(
        catalogue: Arc<MicCatalogue>,
        initial_policy: MicPolicy,
        config: Arc<crate::config::ConfigCell>,
    ) -> Self {
        let initial = MicSettings {
            catalogue: catalogue.clone(),
            policy: initial_policy,
        };
        Self {
            inner: VersionedSwap::new(initial),
            catalogue,
            config,
        }
    }
}

impl MicSettingsStore for MicSettingsCell {
    fn snapshot(&self) -> Arc<MicSettings> {
        self.inner.snapshot()
    }

    fn version(&self) -> ResourceVersion {
        self.inner.version()
    }

    fn snapshot_with_version(&self) -> (Arc<MicSettings>, ResourceVersion) {
        // Both halves share one guard, so a concurrent swap can't pair OLD
        // policy with NEW version.
        self.inner.snapshot_with_version()
    }
}

impl MicSettingsHandle for MicSettingsCell {
    fn try_set_policy(&self, policy: MicPolicy) -> Result<SwapReceipt, MicError> {
        crate::config::launch::validate_policy_against_catalogue(
            &policy,
            &self.catalogue,
            self.config.path(),
        )
        .map_err(|e| MicError::Rejected(e.to_string()))?;

        // Persist + swap collapse into one `mutate_lock` critical section so
        // concurrent `spawn_blocking` POSTs can't interleave to disk=T2/mem=T1;
        // persist-first (swap in `after`) makes Err mean neither disk nor mem
        // changed. Cost: readers lag the swap by one arbitrator pump.
        //
        // This `Arc::new` MUST stay outside the `after` closure (which runs
        // after disk+config commit): an alloc panic inside it would strand
        // inner at OLD while config is NEW (GET /config vs GET /mic divergence);
        // hoisting puts any alloc panic before all side effects.
        let new_settings_arc = Arc::new(MicSettings {
            catalogue: self.catalogue.clone(),
            policy: policy.clone(),
        });
        let receipt_slot: std::cell::RefCell<Option<SwapReceipt>> = std::cell::RefCell::new(None);
        self.config.mutate_then(
            move |c| {
                c.mic = policy;
            },
            |_committed| {
                // mutate_lock held + persist landed: publish the pre-allocated
                // Arc as an infallible bump.
                let arc_for_swap = new_settings_arc.clone();
                let (receipt, _) = self
                    .inner
                    .try_mutate::<(), MicError>(move |_cur| Ok((arc_for_swap, ())))
                    .expect("infallible mutator");
                *receipt_slot.borrow_mut() = Some(receipt);
            },
        )?;

        Ok(receipt_slot
            .into_inner()
            .expect("mutate_then's after callback must run on the Ok path"))
    }

    fn try_set_policy_no_persist(&self, policy: MicPolicy) -> Result<SwapReceipt, MicError> {
        crate::config::launch::validate_policy_against_catalogue(
            &policy,
            &self.catalogue,
            self.config.path(),
        )
        .map_err(|e| MicError::Rejected(e.to_string()))?;

        // Hoisted for symmetry with try_set_policy; here only defense-in-depth,
        // since nothing has committed on the watcher path.
        let new_settings_arc = Arc::new(MicSettings {
            catalogue: self.catalogue.clone(),
            policy,
        });
        let (receipt, _) = self
            .inner
            .try_mutate::<(), MicError>(move |_cur| Ok((new_settings_arc, ())))
            .expect("infallible mutator");

        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_io::mic_arbitrator::{ChannelSelection, MicSelection};

    /// Persist failure returns Err AND leaves the in-memory snapshot unchanged
    /// (no silent live-mic apply behind an Err).
    #[cfg(unix)]
    #[test]
    fn try_set_policy_persist_failure_leaves_in_memory_unchanged() {
        use std::os::unix::fs::PermissionsExt;

        // Restore writable perms on drop so tempdir cleanup survives panic.
        struct RestorePerms(std::path::PathBuf);
        impl Drop for RestorePerms {
            fn drop(&mut self) {
                let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
            }
        }

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let cfg_path = tmpdir.path().join("config.toml");
        let initial_cfg = crate::config::Config::default_for();
        let initial_policy = initial_cfg.mic.clone();
        let cfg_cell = Arc::new(
            crate::config::ConfigCell::from_value(initial_cfg, cfg_path)
                .expect("valid initial config"),
        );

        let launch = crate::config::LaunchConfig::default_for();
        let mic_cell = MicSettingsCell::new(Arc::new(launch.mic), initial_policy.clone(), cfg_cell);

        let pre = mic_cell.snapshot();
        assert_eq!(pre.policy, initial_policy);

        // Read-only parent dir fails write_toml_atomically at staging.
        std::fs::set_permissions(tmpdir.path(), std::fs::Permissions::from_mode(0o555))
            .expect("chmod 555");
        let _restore = RestorePerms(tmpdir.path().to_path_buf());

        // Different but catalogue-valid policy (channel 0 is whitelisted).
        let new_policy = MicPolicy {
            mic: MicSelection::FirstAvailable,
            channel: ChannelSelection::Fixed { channel: 0 },
        };
        assert_ne!(
            new_policy, initial_policy,
            "test target must differ from initial"
        );

        let err = mic_cell
            .try_set_policy(new_policy.clone())
            .expect_err("persist must fail under chmod 555");
        assert!(
            matches!(err, MicError::Persist(_)),
            "expected MicError::Persist, got {err:?}",
        );

        let post = mic_cell.snapshot();
        assert_eq!(
            post.policy, initial_policy,
            "in-memory must stay at initial policy when persist fails; \
             got {:?}",
            post.policy,
        );
    }
}
