//! Per-workspace `ArcSwap` snapshots of workspace core + head index so hot
//! paths never re-parse JSON. Reads are wait-free (`Arc<T>`); writers must hold
//! the per-workspace mutation mutex so disk and cache agree at every observable
//! instant.

use crate::common::workspace::{HeadIndex, WorkspaceCore};
use crate::file_mgr::error::FileError;
use crate::file_mgr::schema::{read_head_index, read_workspace_core};
use arc_swap::ArcSwap;
use std::path::Path;
use std::sync::Arc;

/// Both fields in one `ArcSwap` so cross-field reads see a consistent pair, not
/// two tear-prone loads.
#[derive(Debug, Clone)]
struct CachedPair {
    core: Arc<WorkspaceCore>,
    heads: Arc<HeadIndex>,
}

/// Cached `workspace.json`/`heads.json`, one cell per workspace. Callers needing
/// BOTH fields MUST use [`Self::snapshot`]: separate `core()`+`heads()` can
/// straddle a head-rotation publish (heads updated, core not yet).
#[derive(Debug)]
pub struct WorkspaceCacheCell {
    pair: ArcSwap<CachedPair>,
}

impl WorkspaceCacheCell {
    pub fn new(core: WorkspaceCore, heads: HeadIndex) -> Self {
        Self {
            pair: ArcSwap::from_pointee(CachedPair {
                core: Arc::new(core),
                heads: Arc::new(heads),
            }),
        }
    }

    /// Seed from disk; a missing file fails closed (`FileError::Io`/`NotFound`)
    /// rather than defaulting, so a half-initialised workspace is not masked.
    pub fn load_from_disk(workspace_dir: &Path) -> Result<Self, FileError> {
        let core = read_workspace_core(workspace_dir)?;
        let heads = read_head_index(workspace_dir)?;
        Ok(Self::new(core, heads))
    }

    /// Wait-free; the returned `Arc` pins the value against concurrent
    /// publishers. For both fields use [`Self::snapshot`].
    #[inline]
    pub fn core(&self) -> Arc<WorkspaceCore> {
        self.pair.load().core.clone()
    }

    /// Wait-free; for both fields use [`Self::snapshot`].
    #[inline]
    pub fn heads(&self) -> Arc<HeadIndex> {
        self.pair.load().heads.clone()
    }

    /// Both fields from the same published commit, so cross-field invariants
    /// (e.g. `core.head_count == heads.heads.len()`) hold by construction.
    #[inline]
    pub fn snapshot(&self) -> (Arc<WorkspaceCore>, Arc<HeadIndex>) {
        let p = self.pair.load();
        (p.core.clone(), p.heads.clone())
    }

    /// Replace the cached core. Caller MUST hold the mutation mutex: the
    /// load-then-store is non-atomic, so concurrent publishers lose updates.
    #[inline]
    pub fn publish_core(&self, core: WorkspaceCore) {
        let prev = self.pair.load_full();
        self.pair.store(Arc::new(CachedPair {
            core: Arc::new(core),
            heads: prev.heads.clone(),
        }));
    }

    /// Replace the cached head index under the same mutex discipline as
    /// [`Self::publish_core`] (non-atomic load-then-store).
    #[inline]
    pub fn publish_heads(&self, heads: HeadIndex) {
        let prev = self.pair.load_full();
        self.pair.store(Arc::new(CachedPair {
            core: prev.core.clone(),
            heads: Arc::new(heads),
        }));
    }

    /// Publish both fields in one swap; use from paths touching both (e.g. head
    /// rotation updates heads + head_count) so a cross-field reader sees only
    /// pre- or post-publish state.
    #[inline]
    pub fn publish_pair(&self, core: WorkspaceCore, heads: HeadIndex) {
        self.pair.store(Arc::new(CachedPair {
            core: Arc::new(core),
            heads: Arc::new(heads),
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ids::{HeadId, WorkspaceId};
    use crate::common::workspace::{HeadRecord, WorkspaceRevision};
    use crate::file_mgr::schema::{write_head_index, write_workspace_core};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn sample_core(rev_id: u64) -> WorkspaceCore {
        WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(rev_id),
            head_count: 0,
        }
    }

    fn sample_record() -> HeadRecord {
        HeadRecord {
            head_id: HeadId::parse("11111111-2222-4333-8444-555555555556").unwrap(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 12,
            size_bytes: 4096,
            created_at: "2026-05-07T12:34:56Z".to_string(),
        }
    }

    #[test]
    fn new_seeds_cache_with_supplied_values() {
        let core = sample_core(5);
        let mut idx = HeadIndex::default();
        idx.heads.push(sample_record());
        let cell = WorkspaceCacheCell::new(core.clone(), idx.clone());
        assert_eq!(*cell.core(), core);
        assert_eq!(*cell.heads(), idx);
    }

    #[test]
    fn load_from_disk_round_trips_through_schema_helpers() {
        let tmp = tempfile::tempdir().unwrap();
        let core = sample_core(5);
        write_workspace_core(tmp.path(), &core).unwrap();
        let mut idx = HeadIndex::default();
        idx.heads.push(sample_record());
        write_head_index(tmp.path(), &idx).unwrap();
        let cell = WorkspaceCacheCell::load_from_disk(tmp.path()).unwrap();
        assert_eq!(*cell.core(), core);
        assert_eq!(*cell.heads(), idx);
    }

    #[test]
    fn load_from_disk_missing_files_surface_as_io_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        match WorkspaceCacheCell::load_from_disk(tmp.path()) {
            Err(FileError::Io { source, .. }) => {
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected FileError::Io NotFound, got {other:?}"),
        }
        // Only workspace.json present -- still fails on heads.
        write_workspace_core(tmp.path(), &sample_core(0)).unwrap();
        match WorkspaceCacheCell::load_from_disk(tmp.path()) {
            Err(FileError::Io { source, .. }) => {
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected FileError::Io NotFound, got {other:?}"),
        }
    }

    #[test]
    fn publish_core_replaces_snapshot_atomically() {
        let cell = WorkspaceCacheCell::new(sample_core(5), HeadIndex::default());
        assert_eq!(cell.core().workspace_revision, rev(5));
        cell.publish_core(sample_core(6));
        assert_eq!(cell.core().workspace_revision, rev(6));
        cell.publish_core(sample_core(7));
        assert_eq!(cell.core().workspace_revision, rev(7));
    }

    #[test]
    fn publish_heads_replaces_snapshot_atomically() {
        let cell = WorkspaceCacheCell::new(sample_core(5), HeadIndex::default());
        assert!(cell.heads().heads.is_empty());
        let mut idx = HeadIndex::default();
        idx.heads.push(sample_record());
        cell.publish_heads(idx.clone());
        assert_eq!(cell.heads().heads.len(), 1);
    }

    /// Pins wait-free reads: a concurrent reader sees old or new value, never
    /// partial state, never panics.
    #[test]
    fn concurrent_reads_during_publish_observe_consistent_snapshots() {
        let cell = Arc::new(WorkspaceCacheCell::new(
            sample_core(0),
            HeadIndex::default(),
        ));
        let stop = Arc::new(AtomicBool::new(false));
        let reader_cell = Arc::clone(&cell);
        let reader_stop = Arc::clone(&stop);
        let reader = thread::spawn(move || {
            let mut last = 0u64;
            while !reader_stop.load(Ordering::Relaxed) {
                let snap = reader_cell.core();
                let obs = snap.workspace_revision.id;
                // Revisions never go backward across publishes.
                assert!(
                    obs >= last,
                    "reader saw non-monotonic revision: {obs} < {last}"
                );
                last = obs;
            }
            last
        });
        for i in 1..=200u64 {
            cell.publish_core(sample_core(i));
        }
        stop.store(true, Ordering::Relaxed);
        let last_seen = reader.join().unwrap();
        assert!(last_seen <= 200);
        assert_eq!(cell.core().workspace_revision.id, 200);
    }

    /// Pins that a `core()` Arc survives a subsequent publish (no UAF).
    #[test]
    fn snapshot_arc_outlives_subsequent_publish() {
        let cell = WorkspaceCacheCell::new(sample_core(5), HeadIndex::default());
        let snap = cell.core();
        cell.publish_core(sample_core(6));
        assert_eq!(snap.workspace_revision, rev(5));
        assert_eq!(cell.core().workspace_revision, rev(6));
    }
}
