//! Trained-head sliding-window rotation under the caller-held per-workspace
//! mutation mutex.  `heads.json` is the publish point: a crash before the index
//! commit leaves `<head_id>.{mpk,json}` orphans swept by boot recovery, after
//! it the head is durable; post-commit cleanup of displaced bytes is best-effort.

use std::path::{Path, PathBuf};

use crate::common::ids::HeadId;
use crate::common::workspace::{HeadIndex, HeadManifest, HeadRecord, MAX_HEADS_PER_WORKSPACE};
use crate::file_mgr::cache::WorkspaceCacheCell;
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::fs_atomic::put_atomic;
use crate::file_mgr::schema::{
    head_artifact_path, head_manifest_path, heads_dir, read_workspace_core, write_head_index,
    write_workspace_core,
};
use crate::file_mgr::validate::fsync_dir;

/// Inputs to [`publish_trained_head`]; `manifest.head_id` MUST equal `head_id`.
#[derive(Clone, Debug)]
pub struct PendingHead {
    pub head_id: HeadId,
    /// Staged + fsynced `.mpk` tempfile under `<workspace_dir>/.tmp/` (same FS as
    /// `heads/` so the rename into `heads/<head_id>.mpk` is intra-FS atomic).
    pub mpk_tempfile: PathBuf,
    pub manifest: HeadManifest,
}

#[derive(Clone, Copy, Debug)]
pub struct HeadRotationResult {
    /// Head id displaced by eviction, or `None` if the workspace was below
    /// `MAX_HEADS_PER_WORKSPACE` before the publish.
    pub displaced_head_id: Option<HeadId>,
}

/// Publish a trained head into the workspace's head index.  Caller MUST hold the
/// per-workspace mutation mutex for the full call; sync, never `.await`.
///
/// `pinned_head` (typically the active source) MUST NOT be displaced: eviction
/// drops the tail-most NON-pinned entry, and `None`/a pin absent from the prior
/// index yields the original LRU tail.
///
/// Crash contract is step ordering: `.mpk` + `.json` land under `heads/` and
/// `heads/` is fsynced BEFORE the `heads.json` index commit (the publish point)
/// so an index entry never outruns its files, then derived boot-repairable
/// `workspace.json.head_count` and best-effort unlink of displaced bytes.  A
/// `heads/` file unreferenced by `heads.json` is orphan residue swept by boot
/// recovery; an index entry referencing a missing file is genuine corruption.
///
/// Two known forward-loss races, both fixable by stamping `published_at`
/// post-commit and dropping unstamped manifests (deferred): a crash between the
/// `.mpk` rename and the commit, ONLY when the prior `heads.json` is also
/// missing, lets `reconstruct_head_index_from_disk` promote the paired
/// `<id>.{mpk,json}` to a real head; and a commit that fails while the prior
/// `heads.json` survives leaves the durable new pair unreferenced for
/// `sweep_head_orphans` to unlink next boot.
pub fn publish_trained_head(
    workspace_dir: &Path,
    cache: &WorkspaceCacheCell,
    pending: PendingHead,
    pinned_head: Option<HeadId>,
) -> Result<HeadRotationResult, FileError> {
    if pending.manifest.head_id != pending.head_id {
        return Err(io_err(
            workspace_dir.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "publish_trained_head: PendingHead.head_id != manifest.head_id",
            ),
        ));
    }
    if !pending.mpk_tempfile.is_file() {
        return Err(io_err(
            pending.mpk_tempfile.display(),
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "publish_trained_head: mpk_tempfile is missing or not a regular file",
            ),
        ));
    }
    // Fail closed on a reused head_id, else the rotation overwrites the existing
    // `<id>.{mpk,json}` AND duplicates the `HeadRecord` (next_records does not
    // dedupe).  Canonical snapshot under the caller-held mutex.
    let prev_heads = cache.heads();
    if let Some(existing) = prev_heads
        .heads
        .iter()
        .find(|h| h.head_id == pending.head_id)
    {
        return Err(FileError::HeadIdCollision {
            head_id: pending.head_id.to_string(),
            got_sha256: pending.manifest.sha256.clone(),
            stored_sha256: existing.sha256.clone(),
        });
    }
    let heads = heads_dir(workspace_dir);
    std::fs::create_dir_all(&heads).map_err(|e| io_err(heads.display(), e))?;

    // Manifest durable (put_atomic = tempfile+fsync+rename+parent-fsync) before
    // the index.
    let manifest_path = head_manifest_path(workspace_dir, pending.head_id);
    let manifest_bytes = serde_json::to_vec(&pending.manifest)?;
    put_atomic(&manifest_path, &manifest_bytes)?;

    // fsync the source before the rename: rename moves dirents not data blocks,
    // so an un-fsynced source leaves heads.json pointing at an empty-on-disk
    // `.mpk` post-crash.  Read-write fd because read-only fsync no-ops on some
    // non-Linux filesystems.
    {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .read(true)
            .open(&pending.mpk_tempfile)
            .map_err(|e| io_err(pending.mpk_tempfile.display(), e))?;
        f.sync_all()
            .map_err(|e| io_err(pending.mpk_tempfile.display(), e))?;
    }
    let mpk_path = head_artifact_path(workspace_dir, pending.head_id);
    let mpk_src_parent = pending
        .mpk_tempfile
        .parent()
        .map(std::path::Path::to_path_buf);
    std::fs::rename(&pending.mpk_tempfile, &mpk_path).map_err(|e| io_err(mpk_path.display(), e))?;

    // Both new dirents durable before the index commit.
    fsync_dir(&heads).map_err(|e| io_err(heads.display(), e))?;
    // fsync the source `.tmp/` parent too to drop the stale staging dirent
    // (rename touches dirents in BOTH parents); warn-only since the destination
    // fsync already locked in publish-succeeded semantics.
    if let Some(parent) = mpk_src_parent
        && parent.exists()
        && let Err(e) = fsync_dir(&parent)
    {
        tracing::warn!(
            target: "file_mgr",
            err = %e,
            path = %parent.display(),
            "publish_trained_head: source-parent fsync failed after step-5 rename; \
             stale .tmp/ dirent may persist across crash until storage_reaper sweep",
        );
    }

    // Reuses the collision-guard `prev_heads` snapshot (still valid: caller-held
    // mutex excludes concurrent mutation).
    let new_record = HeadRecord {
        head_id: pending.manifest.head_id,
        workspace_revision: pending.manifest.workspace_revision.clone(),
        sha256: pending.manifest.sha256.clone(),
        n_classes: pending.manifest.n_classes,
        size_bytes: pending.manifest.size_bytes,
        created_at: pending.manifest.created_at.clone(),
    };
    let mut next_records: Vec<HeadRecord> = Vec::with_capacity(prev_heads.heads.len() + 1);
    next_records.push(new_record);
    next_records.extend(prev_heads.heads.iter().cloned());
    let displaced_head_id = if next_records.len() > MAX_HEADS_PER_WORKSPACE {
        // Scan from the tail skipping index 0 (the new head is never evicted);
        // `unwrap_or(len-1)` covers the impossible all-pinned case.
        let drop_idx = (1..next_records.len())
            .rev()
            .find(|&i| pinned_head.is_none_or(|pin| next_records[i].head_id != pin))
            .unwrap_or(next_records.len() - 1);
        Some(next_records.remove(drop_idx).head_id)
    } else {
        None
    };
    debug_assert!(next_records.len() <= MAX_HEADS_PER_WORKSPACE);
    let next_index = HeadIndex {
        heads: next_records,
    };

    // Read workspace_core from disk (guarding a torn cache snapshot) BEFORE the
    // publish point so a corrupt workspace.json fails closed without publishing an
    // index whose head_count never updated.
    let mut next_core = read_workspace_core(workspace_dir)?;
    next_core.head_count = next_index.heads.len() as u8;

    // Publish point.
    if let Err(e) = write_head_index(workspace_dir, &next_index) {
        tracing::warn!(
            target: "file_mgr",
            err = %e,
            head_id = %pending.head_id,
            mpk = %mpk_path.display(),
            json = %manifest_path.display(),
            "publish_trained_head: heads.json commit failed; durable .mpk + .json pair is unindexed and will be swept by boot recovery",
        );
        return Err(e);
    }

    // workspace.json failure: head_count converges at boot via repair.  Sync the
    // cache's heads-half to the just-published disk state first, else a
    // same-process retry rebuilds from the OLD snapshot and overwrites heads.json
    // from stale state, dropping the new head.
    if let Err(e) = write_workspace_core(workspace_dir, &next_core) {
        cache.publish_heads(next_index);
        return Err(e);
    }

    // Dual-publish so head_count never disagrees with heads.heads.len().
    cache.publish_pair(next_core, next_index);

    // Best-effort; boot recovery sweeps stragglers.
    if let Some(old) = displaced_head_id {
        let old_mpk = head_artifact_path(workspace_dir, old);
        let old_json = head_manifest_path(workspace_dir, old);
        if let Err(e) = std::fs::remove_file(&old_mpk)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %old_mpk.display(),
                "publish_trained_head: failed to remove displaced .mpk; boot recovery will sweep",
            );
        }
        if let Err(e) = std::fs::remove_file(&old_json)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %old_json.display(),
                "publish_trained_head: failed to remove displaced .json; boot recovery will sweep",
            );
        }
        if let Err(e) = fsync_dir(&heads) {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %heads.display(),
                "publish_trained_head: failed to fsync heads/ after displaced removal",
            );
        }
    }

    Ok(HeadRotationResult { displaced_head_id })
}

use crate::common::ids::WorkspaceId;
use crate::common::workspace::JobReference;
use crate::file_mgr::WorkspaceMgr;
use crate::file_mgr::schema::workspace_core_path;
use std::sync::Arc;

impl WorkspaceMgr {
    /// Remove a single completed head; index-atomic removal mirroring
    /// [`publish_trained_head`].  Errors: [`FileError::JobConflict`] (running job
    /// references the workspace), [`FileError::ActiveSourcePinned`] (head is the
    /// active source), [`FileError::AssetNotFound`] (absent from `heads.json`).
    ///
    /// Holds the per-workspace mutation mutex throughout.  The production entry
    /// point [`crate::file_mgr::FsService::delete_head`] holds the activation lock
    /// (active -> per-workspace order) around the unlocked
    /// `active_source_head_in_workspace` read below; bare-`WorkspaceMgr` test
    /// callers MUST replicate it if they race activation.
    pub(crate) fn delete_head(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        head_id: HeadId,
    ) -> Result<(), FileError> {
        let workspace_dir = self.workspace_dir(ws);
        let core_path = workspace_core_path(&workspace_dir);
        if !core_path.exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        // Workspace-wide lease; an overlapping running job bumps us out with
        // `JobConflict`.
        let _ref_guard = self
            .jobs
            .try_acquire_lease(vec![JobReference::Workspace { workspace_id: *ws }])
            .map_err(FileError::from)?;

        // Per-workspace mutation mutex.  Sync; never `.await`.
        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();
        // Re-check under the lock: a concurrent `WorkspaceDelete` could have
        // renamed the tree away since the pre-lock check.
        if !core_path.exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }

        // Refuse to delete the active source; the entry point's `active_mutex`
        // excludes this read against `POST /active`.
        if let Some(active_source) =
            crate::file_mgr::active_source_head_in_workspace(&self.root, *ws)
            && active_source == head_id
        {
            return Err(FileError::ActiveSourcePinned {
                workspace_id: ws.to_string(),
                head_id: head_id.to_string(),
            });
        }

        let cell = self.cache_cell(ws)?;
        let prev_heads = cell.heads();
        let mut next_records = Vec::with_capacity(prev_heads.heads.len());
        let mut found = false;
        for rec in prev_heads.heads.iter() {
            if rec.head_id == head_id {
                found = true;
                continue;
            }
            next_records.push(rec.clone());
        }
        if !found {
            return Err(FileError::AssetNotFound {
                ws: ws.to_string(),
                kind: crate::file_mgr::AssetKind::HeadMpk,
                name: format!("{head_id}.mpk"),
            });
        }
        let next_index = HeadIndex {
            heads: next_records,
        };

        // Read workspace.json BEFORE the publish point so a corrupt one fails
        // closed without lagging head_count (same hoist as
        // `publish_trained_head`).
        let mut next_core = read_workspace_core(&workspace_dir)?;
        next_core.head_count = next_index.heads.len() as u8;

        // Publish point.
        write_head_index(&workspace_dir, &next_index)?;

        // Sync cache.heads on workspace.json failure so a same-process retry does
        // not rebuild from a stale snapshot and re-insert the just-deleted head.
        if let Err(e) = write_workspace_core(&workspace_dir, &next_core) {
            cell.publish_heads(next_index);
            return Err(e);
        }

        // Dual-publish so head_count never disagrees with heads.heads.
        cell.publish_pair(next_core, next_index);

        // Best-effort; boot recovery sweeps stragglers.
        let mpk_path = head_artifact_path(&workspace_dir, head_id);
        let json_path = head_manifest_path(&workspace_dir, head_id);
        if let Err(e) = std::fs::remove_file(&mpk_path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %mpk_path.display(),
                "delete_head: failed to remove .mpk; boot recovery will sweep",
            );
        }
        if let Err(e) = std::fs::remove_file(&json_path)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %json_path.display(),
                "delete_head: failed to remove .json; boot recovery will sweep",
            );
        }
        let heads = heads_dir(&workspace_dir);
        if let Err(e) = fsync_dir(&heads) {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %heads.display(),
                "delete_head: failed to fsync heads/ after removal",
            );
        }
        Ok(())
    }

    /// Index-atomic publish of a freshly trained head into the workspace
    /// rotation, under the same per-workspace mutex + cache cell as `delete_head`.
    /// Sync; never `.await`.  Caller guarantees `pending.mpk_tempfile` is fsynced
    /// under `<workspace_dir>/.tmp/` and `pending.manifest.head_id == pending.head_id`.
    pub(crate) fn publish_trained_head_for_workspace(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        pending: PendingHead,
    ) -> Result<HeadRotationResult, FileError> {
        let workspace_dir = self.workspace_dir(ws);
        let core_path = workspace_core_path(&workspace_dir);
        if !core_path.exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();
        // Re-check under the lock, else a concurrent `WorkspaceDelete` lets
        // `publish_trained_head`'s `create_dir_all(heads/)` re-materialize a split
        // tree.
        if !core_path.exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        // Active-mutex held by the trait entry point; bare-`WorkspaceMgr` test
        // callers must replicate it if they race activation.
        let pinned_head = crate::file_mgr::active_source_head_in_workspace(&self.root, *ws);
        let cell = self.cache_cell(ws)?;
        publish_trained_head(&workspace_dir, &cell, pending, pinned_head)
    }

    /// Index-atomic publish of an *imported* head (`.alpkg` convert path).  Same
    /// discipline as [`Self::publish_trained_head_for_workspace`] plus an
    /// idempotency/collision check under the same mutex so no producer interleaves
    /// between check and publish.  Outcomes: `AlreadyExists` (same `head_id` +
    /// `sha256`, no publish); `Published(rotation)`; `Err(HeadIdCollision)` (same
    /// `head_id`, different `sha256` -- refuses to overwrite so references pinned
    /// to the old sha256 stay valid).
    pub(crate) fn publish_imported_head_for_workspace(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        pending: PendingHead,
    ) -> Result<HeadImportResult, FileError> {
        let workspace_dir = self.workspace_dir(ws);
        let core_path = workspace_core_path(&workspace_dir);
        if !core_path.exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();
        // Re-check under the lock (see `publish_trained_head_for_workspace`).
        if !core_path.exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }

        // Idempotency/collision check under the mutex so no rotation interleaves
        // before the delegate below.
        let cell = self.cache_cell(ws)?;
        let prev_heads = cell.heads();
        for existing in &prev_heads.heads {
            if existing.head_id == pending.head_id {
                if existing.sha256 == pending.manifest.sha256 {
                    return Ok(HeadImportResult::AlreadyExists);
                }
                return Err(FileError::HeadIdCollision {
                    head_id: pending.head_id.to_string(),
                    got_sha256: pending.manifest.sha256.clone(),
                    stored_sha256: existing.sha256.clone(),
                });
            }
        }

        // Active-mutex held by the trait entry point (see
        // `publish_trained_head_for_workspace`).
        let pinned_head = crate::file_mgr::active_source_head_in_workspace(&self.root, *ws);
        let rotation = publish_trained_head(&workspace_dir, &cell, pending, pinned_head)?;
        Ok(HeadImportResult::Published(rotation))
    }
}

/// Outcome of `publish_imported_head_for_workspace`.
#[derive(Clone, Copy, Debug)]
pub enum HeadImportResult {
    /// A rotation ran; inner field carries any evicted predecessor.
    Published(HeadRotationResult),
    /// Already present with a matching sha256; no publish ran.
    AlreadyExists,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]
    // Orphan-fixture setup intentionally bypasses atomic publish helpers.

    use super::*;
    use crate::common::ids::WorkspaceId;
    use crate::common::workspace::{WorkspaceCore, WorkspaceRevision};
    use crate::file_mgr::schema::{HEADS_DIR_NAME, write_head_index, write_workspace_core};
    use std::io::Write;

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555550").unwrap()
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn sample_core(rev_id: u64, head_count: u8) -> WorkspaceCore {
        WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(rev_id),
            head_count,
        }
    }

    fn sample_manifest(head_id: HeadId, rev_id: u64) -> HeadManifest {
        HeadManifest {
            head_id,
            workspace_id: ws_id(),
            workspace_revision: rev(rev_id),
            sha256: "def".to_string(),
            n_classes: 3,
            size_bytes: 1024,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
        }
    }

    fn stage_mpk_tempfile(workspace_dir: &Path, head_id: HeadId) -> PathBuf {
        let tmp_dir = workspace_dir.join(".tmp");
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let path = tmp_dir.join(format!("staged-{head_id}.mpk"));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(format!("MPK-{head_id}").as_bytes()).unwrap();
        f.sync_all().unwrap();
        path
    }

    fn fresh_workspace() -> (tempfile::TempDir, WorkspaceCacheCell) {
        let tmp = tempfile::tempdir().unwrap();
        let core = sample_core(0, 0);
        write_workspace_core(tmp.path(), &core).unwrap();
        write_head_index(tmp.path(), &HeadIndex::default()).unwrap();
        std::fs::create_dir_all(tmp.path().join(HEADS_DIR_NAME)).unwrap();
        let cache = WorkspaceCacheCell::new(core, HeadIndex::default());
        (tmp, cache)
    }

    /// Publish one head: index, head_count, and cache all reflect it.
    #[test]
    fn publish_trained_head_happy_path() {
        let (tmp, cache) = fresh_workspace();
        let head_id = HeadId::new();
        let mpk = stage_mpk_tempfile(tmp.path(), head_id);
        let manifest = sample_manifest(head_id, 5);

        let result = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id,
                mpk_tempfile: mpk.clone(),
                manifest: manifest.clone(),
            },
            None,
        )
        .unwrap();

        assert!(
            result.displaced_head_id.is_none(),
            "first publish never displaces"
        );
        let mpk_final = head_artifact_path(tmp.path(), head_id);
        let json_final = head_manifest_path(tmp.path(), head_id);
        assert!(mpk_final.is_file(), "mpk landed at {}", mpk_final.display());
        assert!(json_final.is_file());
        assert!(!mpk.exists(), "tempfile was renamed away");
        let on_disk = crate::file_mgr::schema::read_head_index(tmp.path()).unwrap();
        assert_eq!(on_disk.heads.len(), 1);
        assert_eq!(on_disk.heads[0].head_id, head_id);
        let core = crate::file_mgr::schema::read_workspace_core(tmp.path()).unwrap();
        assert_eq!(core.head_count, 1);
        assert_eq!(cache.heads().heads.len(), 1);
        assert_eq!(cache.core().head_count, 1);
    }

    /// Sliding window: publishing `cap + 1` keeps only the newest `cap` and
    /// removes the displaced head's files.
    #[test]
    fn publish_overflowing_cap_displaces_oldest() {
        let (tmp, cache) = fresh_workspace();
        let cap = MAX_HEADS_PER_WORKSPACE;
        // ids[0] is oldest; the trailing id triggers displacement.
        let ids: Vec<HeadId> = (0..(cap + 1)).map(|_| HeadId::new()).collect();

        for (i, &h) in ids[..cap].iter().enumerate() {
            let mpk = stage_mpk_tempfile(tmp.path(), h);
            publish_trained_head(
                tmp.path(),
                &cache,
                PendingHead {
                    head_id: h,
                    mpk_tempfile: mpk,
                    manifest: sample_manifest(h, (i + 1) as u64),
                },
                None,
            )
            .unwrap();
        }
        assert_eq!(cache.heads().heads.len(), cap);
        assert_eq!(cache.core().head_count, cap as u8);

        let h_last = ids[cap];
        let mpk = stage_mpk_tempfile(tmp.path(), h_last);
        let result = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h_last,
                mpk_tempfile: mpk,
                manifest: sample_manifest(h_last, (cap + 1) as u64),
            },
            None,
        )
        .unwrap();
        assert_eq!(
            result.displaced_head_id,
            Some(ids[0]),
            "ids[0] was the oldest, so it gets displaced",
        );
        let on_disk = crate::file_mgr::schema::read_head_index(tmp.path()).unwrap();
        assert_eq!(on_disk.heads.len(), cap);
        assert_eq!(on_disk.heads[0].head_id, h_last, "newest first");
        for i in 1..cap {
            assert_eq!(
                on_disk.heads[i].head_id,
                ids[cap - i],
                "newest-first ordering at index {i}",
            );
        }
        assert!(!head_artifact_path(tmp.path(), ids[0]).exists());
        assert!(!head_manifest_path(tmp.path(), ids[0]).exists());
        for &h in &ids[1..=cap] {
            assert!(head_artifact_path(tmp.path(), h).is_file());
            assert!(head_manifest_path(tmp.path(), h).is_file());
        }
        let core = crate::file_mgr::schema::read_workspace_core(tmp.path()).unwrap();
        assert_eq!(core.head_count, cap as u8);
    }

    /// The rotation touches only its own displaced entry, never
    /// index-unreachable orphans under `heads/`.
    #[test]
    fn publish_does_not_disturb_unrelated_orphans() {
        let (tmp, cache) = fresh_workspace();
        let orphan_id = HeadId::new();
        let orphan_mpk = head_artifact_path(tmp.path(), orphan_id);
        let orphan_json = head_manifest_path(tmp.path(), orphan_id);
        std::fs::write(&orphan_mpk, b"orphan-mpk").unwrap();
        std::fs::write(&orphan_json, b"{}").unwrap();

        let h1 = HeadId::new();
        let mpk = stage_mpk_tempfile(tmp.path(), h1);
        publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h1,
                mpk_tempfile: mpk,
                manifest: sample_manifest(h1, 1),
            },
            None,
        )
        .unwrap();
        assert!(orphan_mpk.exists(), "rotation must not touch orphans");
        assert!(orphan_json.exists(), "rotation must not touch orphans");
    }

    /// Cache load treats the index as truth (no per-`.mpk` stat), so it tolerates
    /// an index referencing a phantom head whose `.mpk` never landed; physical
    /// consistency is boot recovery's concern.
    #[test]
    fn cache_load_tolerates_index_referencing_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let core = sample_core(0, 1);
        write_workspace_core(tmp.path(), &core).unwrap();
        let phantom = HeadId::new();
        let phantom_record = HeadRecord {
            head_id: phantom,
            workspace_revision: rev(0),
            sha256: "y".into(),
            n_classes: 1,
            size_bytes: 0,
            created_at: "2026-05-07T12:00:00Z".to_string(),
        };
        let bad_index = HeadIndex {
            heads: vec![phantom_record],
        };
        write_head_index(tmp.path(), &bad_index).unwrap();
        let cache = WorkspaceCacheCell::load_from_disk(tmp.path()).unwrap();
        assert_eq!(cache.heads().heads.len(), 1);
        assert!(!head_artifact_path(tmp.path(), phantom).exists());
        assert!(!head_manifest_path(tmp.path(), phantom).exists());
    }

    /// Best-effort displaced cleanup: a displaced file already gone (FS race /
    /// orphan sweep) still yields a successful rotation since the index commit is
    /// the source of truth.
    #[test]
    fn publish_succeeds_when_displaced_files_are_already_gone() {
        let (tmp, cache) = fresh_workspace();
        let cap = MAX_HEADS_PER_WORKSPACE;
        let ids: Vec<HeadId> = (0..(cap + 1)).map(|_| HeadId::new()).collect();
        for (i, &h) in ids[..cap].iter().enumerate() {
            let mpk = stage_mpk_tempfile(tmp.path(), h);
            publish_trained_head(
                tmp.path(),
                &cache,
                PendingHead {
                    head_id: h,
                    mpk_tempfile: mpk,
                    manifest: sample_manifest(h, (i + 1) as u64),
                },
                None,
            )
            .unwrap();
        }
        // Remove ids[0]'s files before the publish that displaces them.
        std::fs::remove_file(head_artifact_path(tmp.path(), ids[0])).unwrap();
        std::fs::remove_file(head_manifest_path(tmp.path(), ids[0])).unwrap();
        let h_last = ids[cap];
        let mpk = stage_mpk_tempfile(tmp.path(), h_last);
        let result = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h_last,
                mpk_tempfile: mpk,
                manifest: sample_manifest(h_last, (cap + 1) as u64),
            },
            None,
        )
        .unwrap();
        assert_eq!(result.displaced_head_id, Some(ids[0]));
        let on_disk = crate::file_mgr::schema::read_head_index(tmp.path()).unwrap();
        assert_eq!(on_disk.heads.len(), cap);
        assert_eq!(on_disk.heads[0].head_id, h_last);
        for i in 1..cap {
            assert_eq!(on_disk.heads[i].head_id, ids[cap - i]);
        }
    }

    #[test]
    fn publish_rejects_mismatched_head_id() {
        let (tmp, cache) = fresh_workspace();
        let h1 = HeadId::new();
        let h2 = HeadId::new();
        let mpk = stage_mpk_tempfile(tmp.path(), h1);
        let bad_manifest = sample_manifest(h2, 1);
        let result = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h1,
                mpk_tempfile: mpk,
                manifest: bad_manifest,
            },
            None,
        );
        assert!(matches!(result, Err(FileError::Io { .. })));
        assert!(!head_artifact_path(tmp.path(), h1).exists());
        assert!(!head_manifest_path(tmp.path(), h1).exists());
        assert!(!head_artifact_path(tmp.path(), h2).exists());
        assert!(!head_manifest_path(tmp.path(), h2).exists());
    }

    /// Reject a missing staged `.mpk` tempfile, else an empty rename commits.
    #[test]
    fn publish_rejects_missing_mpk_tempfile() {
        let (tmp, cache) = fresh_workspace();
        let h1 = HeadId::new();
        let result = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h1,
                mpk_tempfile: tmp.path().join(".tmp/never-staged.mpk"),
                manifest: sample_manifest(h1, 1),
            },
            None,
        );
        assert!(matches!(result, Err(FileError::Io { .. })));
        assert!(!head_artifact_path(tmp.path(), h1).exists());
    }

    /// Pinned head survives eviction: with the LRU-tail head pinned, the
    /// (cap + 1)th publish displaces the next-oldest non-pinned head.
    #[test]
    fn publish_skips_pinned_head_during_eviction() {
        let (tmp, cache) = fresh_workspace();
        let cap = MAX_HEADS_PER_WORKSPACE;
        let ids: Vec<HeadId> = (0..(cap + 1)).map(|_| HeadId::new()).collect();

        // After the loop the index is [ids[cap-1], ..., ids[0]] with ids[0] the
        // LRU tail.
        for (i, &h) in ids[..cap].iter().enumerate() {
            let mpk = stage_mpk_tempfile(tmp.path(), h);
            publish_trained_head(
                tmp.path(),
                &cache,
                PendingHead {
                    head_id: h,
                    mpk_tempfile: mpk,
                    manifest: sample_manifest(h, (i + 1) as u64),
                },
                None,
            )
            .unwrap();
        }
        assert_eq!(cache.heads().heads.len(), cap);

        let h_last = ids[cap];
        let mpk = stage_mpk_tempfile(tmp.path(), h_last);
        let result = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h_last,
                mpk_tempfile: mpk,
                manifest: sample_manifest(h_last, (cap + 1) as u64),
            },
            Some(ids[0]),
        )
        .unwrap();
        assert_eq!(
            result.displaced_head_id,
            Some(ids[1]),
            "pinned ids[0] must survive; ids[1] (next-oldest non-pinned) gets evicted",
        );
        let on_disk = crate::file_mgr::schema::read_head_index(tmp.path()).unwrap();
        assert_eq!(on_disk.heads.len(), cap);
        assert_eq!(on_disk.heads[0].head_id, h_last, "newest first");
        assert_eq!(
            on_disk.heads[cap - 1].head_id,
            ids[0],
            "pinned survivor at tail",
        );
        for i in 1..(cap - 1) {
            assert_eq!(
                on_disk.heads[i].head_id,
                ids[cap - i],
                "newest-first ordering at index {i}",
            );
        }
        assert!(!head_artifact_path(tmp.path(), ids[1]).exists());
        assert!(!head_manifest_path(tmp.path(), ids[1]).exists());
        assert!(head_artifact_path(tmp.path(), ids[0]).is_file());
        assert!(head_manifest_path(tmp.path(), ids[0]).is_file());
        assert!(head_artifact_path(tmp.path(), h_last).is_file());
        assert!(head_manifest_path(tmp.path(), h_last).is_file());
        for &h in &ids[2..cap] {
            assert!(head_artifact_path(tmp.path(), h).is_file());
            assert!(head_manifest_path(tmp.path(), h).is_file());
        }
    }

    /// Stale-pin fallback: a pin absent from the index yields the original LRU
    /// tail, never silently protecting a phantom slot.
    #[test]
    fn publish_falls_through_when_pinned_id_not_in_index() {
        let (tmp, cache) = fresh_workspace();
        let cap = MAX_HEADS_PER_WORKSPACE;
        let ids: Vec<HeadId> = (0..(cap + 1)).map(|_| HeadId::new()).collect();
        let phantom = HeadId::new();

        for (i, &h) in ids[..cap].iter().enumerate() {
            let mpk = stage_mpk_tempfile(tmp.path(), h);
            publish_trained_head(
                tmp.path(),
                &cache,
                PendingHead {
                    head_id: h,
                    mpk_tempfile: mpk,
                    manifest: sample_manifest(h, (i + 1) as u64),
                },
                None,
            )
            .unwrap();
        }

        let h_last = ids[cap];
        let mpk = stage_mpk_tempfile(tmp.path(), h_last);
        let result = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id: h_last,
                mpk_tempfile: mpk,
                manifest: sample_manifest(h_last, (cap + 1) as u64),
            },
            Some(phantom),
        )
        .unwrap();
        assert_eq!(
            result.displaced_head_id,
            Some(ids[0]),
            "stale pin must not protect anyone; chronological tail evicted",
        );
    }

    /// Re-publishing the same `head_id` MUST fail with `HeadIdCollision`, never
    /// silently overwrite `<id>.{mpk,json}` and duplicate the `HeadRecord`.
    #[test]
    fn publish_rejects_repeat_head_id_with_collision_error() {
        let (tmp, cache) = fresh_workspace();
        let head_id = HeadId::new();
        let mpk_first = stage_mpk_tempfile(tmp.path(), head_id);
        let manifest_first = sample_manifest(head_id, 5);
        publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id,
                mpk_tempfile: mpk_first,
                manifest: manifest_first.clone(),
            },
            None,
        )
        .unwrap();
        // Re-publish under the same id with a different sha256.
        let mpk_second = stage_mpk_tempfile(tmp.path(), head_id);
        let mut manifest_second = sample_manifest(head_id, 6);
        manifest_second.sha256 = "deadbeef".repeat(8);
        let err = publish_trained_head(
            tmp.path(),
            &cache,
            PendingHead {
                head_id,
                mpk_tempfile: mpk_second.clone(),
                manifest: manifest_second.clone(),
            },
            None,
        )
        .unwrap_err();
        match err {
            FileError::HeadIdCollision {
                head_id: hid,
                got_sha256,
                stored_sha256,
            } => {
                assert_eq!(hid, head_id.to_string());
                assert_eq!(got_sha256, manifest_second.sha256);
                assert_eq!(stored_sha256, manifest_first.sha256);
            }
            other => panic!("expected HeadIdCollision, got {other:?}"),
        }
        // Second tempfile survives (no rename) so the caller can clean up.
        assert!(
            mpk_second.is_file(),
            "tempfile must survive guard rejection"
        );
        let on_disk = crate::file_mgr::schema::read_head_index(tmp.path()).unwrap();
        assert_eq!(on_disk.heads.len(), 1, "no duplicate record added");
        assert_eq!(on_disk.heads[0].sha256, manifest_first.sha256);
    }
}
