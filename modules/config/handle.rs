//! Object-safe `ConfigHandle` trait + `ConfigGuard` pattern; concrete impl is
//! [`crate::config::ConfigCell`].
//!
//! Atomicity contract: read-modify-write runs the mutator AND the after-hook
//! under the mutate lock, so handlers updating both on-disk config and a runtime
//! ArcSwap (e.g. `inference_cfg`, `mic_settings`) cannot interleave into an
//! out-of-sync state. Dropping the guard without `commit` rolls back.

use std::path::Path;
use std::sync::Arc;

use crate::config::Config;
use crate::config::error::ConfigError;

/// Object-safe handle on the daemon configuration, held as `Arc<dyn ConfigHandle>`
/// outside the `config` crate so the impl is substitutable. `Send + Sync`: shared
/// across the tokio runtime (api handlers + watcher worker thread).
pub trait ConfigHandle: Send + Sync {
    /// Current snapshot; wait-free `ArcSwap::load_full`.
    fn snapshot(&self) -> Arc<Config>;

    /// Open a write guard, holding the mutate lock until it commits or rolls back.
    /// [`ConfigError::ReentrantMutate`] if this thread is already inside a guard
    /// (re-entrant locking would deadlock). The `'_` bars the guard from outliving
    /// the handle.
    fn open_mutation(&self) -> Result<Box<dyn ConfigGuard + '_>, ConfigError>;

    /// Diagnostic-only path the handle reads + writes; treat as opaque.
    fn path(&self) -> &Path;
}

/// RAII write guard holding the mutate lock for its lifetime; dropping without
/// calling either method acts as `rollback`.
pub trait ConfigGuard {
    /// Mutable access to the cloned snapshot; nothing persists until `commit`.
    fn config(&mut self) -> &mut Config;

    /// Validate, atomically write a fresh TOML file (tempfile + rename), store the
    /// new `Arc<Config>` snapshot, then run `after` with it WHILE STILL HOLDING the
    /// mutate lock -- the hook keeping a caller's runtime ArcSwap consistent with
    /// disk. Pass `Box::new(|_| ())` when no after-hook is needed.
    fn commit(self: Box<Self>, after: Box<dyn FnOnce(&Config) + Send>) -> Result<(), ConfigError>;

    /// Discard without persisting; releases the lock, snapshot unchanged.
    fn rollback(self: Box<Self>);
}
