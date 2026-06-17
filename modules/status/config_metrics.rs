//! Process-wide config-reload counters for
//! [`crate::status::StatusSnapshot::config_reload`], from `ConfigCell`'s atomic
//! handles installed once via [`install_global`]; [`global`] is `None` until
//! then (test fixtures read default-zero).

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Config-reload counters; monotonic per-process (reset on restart).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigReloadSnapshot {
    /// Successful reloads; only counts an actual value change (no-op self-writes short-circuited).
    pub reloads_succeeded_total: u64,
    /// Rejected reloads (parse/validation/callback Err/panic); transient ENOENT file-read races are NOT counted.
    pub reloads_rejected_total: u64,
}

#[derive(Clone, Debug)]
pub struct ConfigReloadHandles {
    succeeded: Arc<AtomicU64>,
    rejected: Arc<AtomicU64>,
}

impl ConfigReloadHandles {
    pub fn new(succeeded: Arc<AtomicU64>, rejected: Arc<AtomicU64>) -> Self {
        Self {
            succeeded,
            rejected,
        }
    }

    /// Lock-free snapshot: two relaxed atomic loads.
    pub fn snapshot(&self) -> ConfigReloadSnapshot {
        ConfigReloadSnapshot {
            reloads_succeeded_total: self.succeeded.load(Ordering::Relaxed),
            reloads_rejected_total: self.rejected.load(Ordering::Relaxed),
        }
    }
}

static GLOBAL: OnceLock<ConfigReloadHandles> = OnceLock::new();

/// One-shot install: `Err(existing)` on re-init.
pub fn install_global(handles: ConfigReloadHandles) -> Result<(), ConfigReloadHandles> {
    GLOBAL.set(handles)
}

/// Installed handles, or `None` (counters disabled) when never installed.
pub fn global() -> Option<&'static ConfigReloadHandles> {
    GLOBAL.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_default_is_zero() {
        let s = ConfigReloadSnapshot::default();
        assert_eq!(s.reloads_succeeded_total, 0);
        assert_eq!(s.reloads_rejected_total, 0);
    }

    #[test]
    fn handles_snapshot_reads_underlying_atomics() {
        let succ = Arc::new(AtomicU64::new(7));
        let rej = Arc::new(AtomicU64::new(2));
        let h = ConfigReloadHandles::new(succ.clone(), rej.clone());
        let s = h.snapshot();
        assert_eq!(s.reloads_succeeded_total, 7);
        assert_eq!(s.reloads_rejected_total, 2);

        succ.fetch_add(1, Ordering::Relaxed);
        rej.fetch_add(3, Ordering::Relaxed);
        let s2 = h.snapshot();
        assert_eq!(s2.reloads_succeeded_total, 8);
        assert_eq!(s2.reloads_rejected_total, 5);
    }
}
