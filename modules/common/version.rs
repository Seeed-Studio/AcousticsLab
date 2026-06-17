//! Hot-state versioning: [`ResourceVersion`], [`SwapReceipt`], and the
//! wait-free-read / writer-mutex-serialised [`VersionedSwap<T>`] (concrete, not
//! a trait: `try_mutate` over `FnOnce(&T) -> Result<(Arc<T>, R), _>` is not object-safe).

/// Monotonic version stamped onto each mutation's [`SwapReceipt`]; `u64` overflow is a non-concern.
#[derive(
    Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct ResourceVersion(u64);

impl ResourceVersion {
    /// Seed for a fresh [`VersionedSwap<T>`]; first mutation produces `1`.
    pub const ZERO: Self = Self(0);

    #[inline]
    pub const fn new(v: u64) -> Self {
        Self(v)
    }

    #[inline]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Saturating at [`u64::MAX`].
    #[inline]
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl Default for ResourceVersion {
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for ResourceVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// Post-mutation version enabling read-your-write: a later `?min_version=version` read reflects this mutation or newer.
#[must_use = "the SwapReceipt carries the post-mutation version; drop only when no caller needs read-your-write semantics"]
#[derive(Copy, Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SwapReceipt {
    pub version: ResourceVersion,
}

use arc_swap::ArcSwap;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Value snapshot plus its producing version; `Arc` lets readers refcount-bump, not copy.
#[derive(Debug)]
struct Versioned<T> {
    version: ResourceVersion,
    value: Arc<T>,
}

/// Wait-free `ArcSwap::load` reads; writes serialise through a mutex so the
/// read-modify-write is linearisable and monotonic-versioned. The mutex covers
/// the in-memory swap ONLY: persist outside `try_mutate`'s closure (an fsync
/// under the lock blocks every later writer, and the value is already
/// `ArcSwap`-published once the lock releases).
#[derive(Debug)]
pub struct VersionedSwap<T> {
    inner: ArcSwap<Versioned<T>>,
    writer: Mutex<()>,
    counter: AtomicU64,
}

impl<T: Send + Sync + 'static> VersionedSwap<T> {
    /// Seeds `initial` at version `0`.
    pub fn new(initial: T) -> Self {
        Self {
            inner: ArcSwap::from_pointee(Versioned {
                version: ResourceVersion::ZERO,
                value: Arc::new(initial),
            }),
            writer: Mutex::new(()),
            counter: AtomicU64::new(0),
        }
    }

    /// Wait-free; the returned `Arc<T>` is safe to hold across mutations until its last refcount drops.
    pub fn snapshot(&self) -> Arc<T> {
        self.inner.load_full().value.clone()
    }

    pub fn version(&self) -> ResourceVersion {
        self.inner.load().version
    }

    /// Atomic value+version read, avoiding the tear of separate `snapshot`/`version` calls.
    pub fn snapshot_with_version(&self) -> (Arc<T>, ResourceVersion) {
        let g = self.inner.load_full();
        (g.value.clone(), g.version)
    }

    /// Runs `f` against the current value under the writer mutex and atomically
    /// installs the result: `Ok` bumps to the next version and returns its
    /// [`SwapReceipt`] plus `R`; `Err` bails with no bump.
    pub fn try_mutate<R, E>(
        &self,
        f: impl FnOnce(&T) -> Result<(Arc<T>, R), E>,
    ) -> Result<(SwapReceipt, R), E> {
        let _g = self.writer.lock();
        let cur = self.inner.load_full();
        let (new_value, ret) = f(&cur.value)?;
        // Allocate (the only fallible step) BEFORE the counter bump: bumping
        // first risks an OOM-panic leaving the counter at N+1 with no store,
        // hanging a `?min_version=N+1` poller forever. Patch the real version
        // below while the carrier is still unshared.
        let mut carrier = Arc::new(Versioned {
            version: ResourceVersion::ZERO,
            value: new_value,
        });
        let v = ResourceVersion::new(self.counter.fetch_add(1, Ordering::AcqRel) + 1);
        Arc::get_mut(&mut carrier)
            .expect("freshly-constructed carrier must be unshared before store")
            .version = v;
        self.inner.store(carrier);
        Ok((SwapReceipt { version: v }, ret))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_version_zero_is_canonical_seed() {
        assert_eq!(ResourceVersion::ZERO.get(), 0);
        assert_eq!(ResourceVersion::default(), ResourceVersion::ZERO);
    }

    #[test]
    fn resource_version_next_monotonic() {
        let v0 = ResourceVersion::ZERO;
        let v1 = v0.next();
        let v2 = v1.next();
        assert_eq!(v0.get(), 0);
        assert_eq!(v1.get(), 1);
        assert_eq!(v2.get(), 2);
        assert!(v0 < v1);
        assert!(v1 < v2);
    }

    #[test]
    fn resource_version_next_saturates_at_u64_max() {
        let max = ResourceVersion::new(u64::MAX);
        assert_eq!(max.next(), max);
    }

    #[test]
    fn swap_receipt_carries_version() {
        let r = SwapReceipt {
            version: ResourceVersion::new(42),
        };
        assert_eq!(r.version.get(), 42);
    }

    #[test]
    fn display_writes_inner() {
        assert_eq!(format!("{}", ResourceVersion::new(7)), "7");
    }

    #[test]
    fn versioned_swap_initial_state() {
        let s: VersionedSwap<u32> = VersionedSwap::new(42);
        assert_eq!(s.version(), ResourceVersion::ZERO);
        assert_eq!(*s.snapshot(), 42);
        let (val, v) = s.snapshot_with_version();
        assert_eq!(*val, 42);
        assert_eq!(v, ResourceVersion::ZERO);
    }

    #[test]
    fn versioned_swap_try_mutate_bumps_version() {
        let s: VersionedSwap<u32> = VersionedSwap::new(10);
        let (receipt, ret) = s
            .try_mutate(|cur| Ok::<_, ()>((Arc::new(*cur + 1), "ok")))
            .expect("mutate");
        assert_eq!(receipt.version.get(), 1);
        assert_eq!(ret, "ok");
        assert_eq!(*s.snapshot(), 11);
        assert_eq!(s.version().get(), 1);
    }

    #[test]
    fn versioned_swap_try_mutate_err_does_not_bump() {
        let s: VersionedSwap<u32> = VersionedSwap::new(7);
        let res: Result<(SwapReceipt, ()), &'static str> = s.try_mutate(|_cur| Err("nope"));
        assert!(matches!(res, Err("nope")));
        assert_eq!(*s.snapshot(), 7);
        assert_eq!(s.version(), ResourceVersion::ZERO);
    }

    #[test]
    fn versioned_swap_concurrent_writers_monotonic() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let s: StdArc<VersionedSwap<u32>> = StdArc::new(VersionedSwap::new(0));
        let n_tasks = 100;
        let handles: Vec<_> = (0..n_tasks)
            .map(|_| {
                let s = s.clone();
                thread::spawn(move || {
                    let (receipt, _) = s
                        .try_mutate(|cur| Ok::<_, ()>((Arc::new(*cur + 1), ())))
                        .expect("mutate");
                    receipt.version.get()
                })
            })
            .collect();
        let mut versions: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        versions.sort_unstable();
        assert_eq!(versions, (1..=n_tasks as u64).collect::<Vec<_>>());
        assert_eq!(s.version().get(), n_tasks as u64);
        assert_eq!(*s.snapshot(), n_tasks);
    }
}
