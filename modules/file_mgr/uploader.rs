//! Admission caps plus the async-streaming [`WorkspaceMgr::upload`] and sync
//! [`WorkspaceMgr::install_from_path`] paths; both commit via atomic rename +
//! metadata under the per-workspace lock, with fail-fast over-cap admission.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::common::ids::{AssetId, WorkspaceId};
use sha2::{Digest, Sha256};

use crate::file_mgr::error::{FileError, io_err, metadata_parse_err};
use crate::file_mgr::metadata::{AssetKind, AssetRecord};
use crate::file_mgr::schema::workspace_core_path;
use crate::file_mgr::validate::{
    fsync_dir, hex_lowercase, sha256_file_streaming, validate_extension,
};
use crate::file_mgr::{AssetReceipt, WorkspaceMgr, validate_asset_name};

/// Admission caps for [`WorkspaceMgr::upload`].
#[derive(Clone, Copy, Debug)]
pub struct AdmissionCfg {
    /// Per-request hard ceiling on uncompressed upload bytes. Default 256 MiB.
    pub max_upload_bytes: u64,
    /// Max in-flight uploads (default 4); global, not per-workspace, so one hostile
    /// workspace can't saturate the file-handle budget.
    pub max_concurrent_uploads: u32,
}

impl Default for AdmissionCfg {
    fn default() -> Self {
        Self {
            max_upload_bytes: 256 * 1024 * 1024,
            max_concurrent_uploads: 4,
        }
    }
}

/// Semaphore is [`Arc`]-shared so all [`WorkspaceMgr`] clones count against one global cap.
#[derive(Debug)]
pub(crate) struct AdmissionState {
    pub(crate) cfg: AdmissionCfg,
    pub(crate) semaphore: Arc<tokio::sync::Semaphore>,
}

impl AdmissionState {
    pub(crate) fn new(cfg: AdmissionCfg) -> Self {
        Self {
            cfg,
            semaphore: Arc::new(tokio::sync::Semaphore::new(
                cfg.max_concurrent_uploads as usize,
            )),
        }
    }
}

impl WorkspaceMgr {
    /// Stream a reader into a tempfile, hashing as it goes, then atomic rename +
    /// `metadata.json` update. Validates name + extension, rejecting path components
    /// (`..`, `/`) to prevent escape.
    pub async fn upload<R>(
        &self,
        id: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        mut body: R,
    ) -> Result<AssetReceipt, FileError>
    where
        R: tokio::io::AsyncRead + Unpin,
    {
        // Held for the whole fn so Drop releases the slot at return; `None` = uncapped.
        let _admission_permit = self.try_acquire_upload_permit()?;

        validate_asset_name(name)?;
        validate_extension(name, kind.allowed_ext())?;

        // Prelude syscalls run in `spawn_blocking` so a disk-pressure stall can't park a
        // tokio worker; its no-lock preflight collision check rejects before consuming the
        // body, but the authoritative check is under the lock at commit time.
        let workspace_dir = self.workspace_dir(id);
        let final_path = self.asset_path(id, kind, name);
        let id_for_err = *id;
        let name_for_check = name.to_string();
        let tmp_dir_for_closure = workspace_dir.join(".tmp");
        let final_parent_for_closure = final_path.parent().map(std::path::Path::to_path_buf);
        let metadata_path_for_closure = workspace_dir.join("metadata.json");
        let (tmp, sync_fd) = tokio::task::spawn_blocking(
            move || -> Result<(tempfile::NamedTempFile, std::fs::File), FileError> {
                if !workspace_dir.exists() {
                    return Err(FileError::NotFound(id_for_err.to_string()));
                }
                // Read directly (not via `&self`) to keep the closure Send-clean; gating
                // the schema here rejects a bad/too-new metadata.json (missing -> NotFound)
                // before the body streams to disk.
                let meta_bytes = match crate::file_mgr::schema::read_capped(
                    &metadata_path_for_closure,
                    crate::file_mgr::schema::MAX_WORKSPACE_METADATA_BYTES,
                ) {
                    Ok(b) => b,
                    Err(FileError::Io { source, .. })
                        if source.kind() == std::io::ErrorKind::NotFound =>
                    {
                        return Err(FileError::NotFound(id_for_err.to_string()));
                    }
                    Err(e) => return Err(e),
                };
                let meta_preflight: crate::file_mgr::WorkspaceMetadata =
                    serde_json::from_slice(&meta_bytes)
                        .map_err(|e| metadata_parse_err(metadata_path_for_closure.display(), e))?;
                if meta_preflight.schema_version > crate::file_mgr::WorkspaceMetadata::CURRENT {
                    return Err(FileError::SchemaTooNew {
                        path: metadata_path_for_closure.display().to_string(),
                        found: meta_preflight.schema_version,
                        max: crate::file_mgr::WorkspaceMetadata::CURRENT,
                    });
                }
                if meta_preflight.schema_version
                    < crate::file_mgr::WorkspaceMetadata::MIN_COMPATIBLE
                {
                    return Err(FileError::SchemaTooOld {
                        path: metadata_path_for_closure.display().to_string(),
                        found: meta_preflight.schema_version,
                        min: crate::file_mgr::WorkspaceMetadata::MIN_COMPATIBLE,
                    });
                }
                check_name_conflict(&meta_preflight, kind, &name_for_check)?;
                std::fs::create_dir_all(&tmp_dir_for_closure)
                    .map_err(|e| io_err(tmp_dir_for_closure.display(), e))?;
                if let Some(parent) = &final_parent_for_closure {
                    std::fs::create_dir_all(parent).map_err(|e| io_err(parent.display(), e))?;
                }
                let tmp = tempfile::NamedTempFile::new_in(&tmp_dir_for_closure)
                    .map_err(|e| io_err(tmp_dir_for_closure.display(), e))?;
                // Sibling fd on the same inode: writer uses `tokio::fs`; `NamedTempFile`
                // keeps its own fd for the eventual `persist` rename.
                let sync_fd = tmp.reopen().map_err(|e| io_err(tmp.path().display(), e))?;
                Ok((tmp, sync_fd))
            },
        )
        .await
        // JoinError (panic/cancel) wrapped as Io to preserve the FileError::Io shape.
        .map_err(|je| io_err("<upload-prelude-spawn-blocking>", std::io::Error::other(je)))??;
        let tmp_path = tmp.path().to_path_buf();
        let mut writer = tokio::fs::File::from_std(sync_fd);

        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 64 * 1024];
        let mut total: u64 = 0;
        // Cap rejects the chunk that crosses it; the tempfile's Drop then unlinks, so
        // no partial commit reaches the asset path.
        let max_upload_bytes = self.max_upload_bytes();
        loop {
            let n = body
                .read(&mut buf)
                .await
                .map_err(|e| io_err("<upload-stream>", e))?;
            if n == 0 {
                break;
            }
            total = total.saturating_add(n as u64);
            if total > max_upload_bytes {
                return Err(FileError::PayloadTooLarge {
                    observed: total,
                    max: max_upload_bytes,
                });
            }
            hasher.update(&buf[..n]);
            writer
                .write_all(&buf[..n])
                .await
                .map_err(|e| io_err(tmp_path.display(), e))?;
        }
        writer
            .flush()
            .await
            .map_err(|e| io_err(tmp_path.display(), e))?;
        writer
            .sync_all()
            .await
            .map_err(|e| io_err(tmp_path.display(), e))?;
        drop(writer);

        let digest = hex_lowercase(&hasher.finalize());

        // Rename + collision-check + commit under one per-workspace lock so concurrent
        // `Foo.mpk`/`foo.mpk` uploads serialize (2nd rejects `NameConflict`) even on a
        // case-sensitive fs. Sync `parking_lot`, never held across an await (no cancel-strand).
        let lock = self.metadata_lock(id);
        let _guard = lock.lock();
        // Re-check under the lock: a concurrent WorkspaceDelete may have renamed the tree
        // away mid-stream. `NotFound` (not `Io{NotFound}`) matches the persist that follows.
        if !workspace_core_path(&self.workspace_dir(id)).exists() {
            return Err(FileError::NotFound(id.to_string()));
        }
        let meta = self.read_metadata(id)?;
        check_name_conflict(&meta, kind, name)?;
        // On overwrite the rename replaces prior bytes, so a later commit failure must NOT
        // unlink `final_path`; capture BEFORE persist.
        let was_overwrite = meta.find_index(kind, name).is_some();
        tmp.persist(&final_path)?;
        self.commit_asset_after_rename(
            id,
            kind,
            name,
            digest,
            total,
            final_path,
            meta,
            was_overwrite,
        )
    }

    /// Sync rename + commit for callers that already staged bytes to a tempfile. `src` MUST
    /// share the workspace root's filesystem so the rename is atomic; cross-device `EXDEV`
    /// surfaces as [`FileError::Io`] (the rename is `std::fs::rename` wrapped by `io_err`, not
    /// `tmp.persist`). Same lock-held section as [`Self::upload`].
    pub fn install_from_path(
        &self,
        id: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        src: &Path,
    ) -> Result<AssetReceipt, FileError> {
        validate_asset_name(name)?;
        let ws = self.workspace_dir(id);
        if !ws.exists() {
            return Err(FileError::NotFound(id.to_string()));
        }
        validate_extension(name, kind.allowed_ext())?;

        let final_path = self.asset_path(id, kind, name);
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io_err(parent.display(), e))?;
        }

        // Stream-hash so multi-GB installs don't load the whole file into RAM.
        let metadata = std::fs::metadata(src).map_err(|e| io_err(src.display(), e))?;
        let total = metadata.len();
        let digest = sha256_file_streaming(src)?;

        let lock = self.metadata_lock(id);
        let _guard = lock.lock();
        // Re-check under the lock; the pre-lock `ws.exists()` is advisory.
        if !workspace_core_path(&ws).exists() {
            return Err(FileError::NotFound(id.to_string()));
        }
        let meta = self.read_metadata(id)?;
        check_name_conflict(&meta, kind, name)?;
        // Capture BEFORE rename so a commit failure won't unlink replaced bytes.
        let was_overwrite = meta.find_index(kind, name).is_some();
        std::fs::rename(src, &final_path).map_err(|e| io_err(final_path.display(), e))?;
        self.commit_asset_after_rename(
            id,
            kind,
            name,
            digest,
            total,
            final_path,
            meta,
            was_overwrite,
        )
    }

    /// Fail-fast global concurrent-upload permit: `Ok(Some)` slot free, `Ok(None)` uncapped
    /// (admission not engaged), `Err(TooManyConcurrentUploads)` full. Drop releases the slot.
    pub fn try_acquire_upload_permit(
        &self,
    ) -> Result<Option<tokio::sync::OwnedSemaphorePermit>, FileError> {
        let Some(state) = &self.admission else {
            return Ok(None);
        };
        match state.semaphore.clone().try_acquire_owned() {
            Ok(p) => Ok(Some(p)),
            Err(_) => {
                let available = state.semaphore.available_permits() as u32;
                let active = state.cfg.max_concurrent_uploads.saturating_sub(available);
                Err(FileError::TooManyConcurrentUploads {
                    active,
                    max: state.cfg.max_concurrent_uploads,
                })
            }
        }
    }

    /// Per-request upload byte cap; [`u64::MAX`] if admission is not engaged.
    pub fn max_upload_bytes(&self) -> u64 {
        self.admission
            .as_ref()
            .map(|s| s.cfg.max_upload_bytes)
            .unwrap_or(u64::MAX)
    }

    /// Post-rename commit shared by [`Self::upload`] and [`Self::install_from_path`]; on
    /// failure runs orphan cleanup gated on `!was_overwrite`. Caller MUST hold the
    /// per-workspace metadata lock for the whole call (not acquired here): otherwise after a
    /// failed `write_metadata` + lock release, a peer uploading the same `name` commits at
    /// `final_path` and our cleanup unlinks its bytes, dangling its row.
    #[allow(clippy::too_many_arguments)]
    fn commit_asset_after_rename(
        &self,
        id: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        digest: String,
        total: u64,
        final_path: PathBuf,
        meta: crate::file_mgr::WorkspaceMetadata,
        was_overwrite: bool,
    ) -> Result<AssetReceipt, FileError> {
        match self.try_commit_asset(id, kind, name, digest, total, &final_path, meta) {
            Ok(receipt) => Ok(receipt),
            Err(e) => {
                // Unlink only if metadata.json provably lacks the row: `!was_overwrite`
                // alone is unsafe since `write_metadata` may have committed the row then
                // failed its dir-fsync, so the readback keeps the file unless absence proven.
                if !was_overwrite && self.asset_orphan_safe_to_remove(id, kind, name) {
                    cleanup_orphan_after_failed_commit(&final_path);
                }
                Err(e)
            }
        }
    }

    /// True only when metadata.json provably lacks `(kind, name)`. `false` when the row is
    /// present (must survive) or the readback fails (a leaked orphan only surfaces in
    /// `validate()`, but a wrong delete is silent data loss). Caller holds the per-workspace
    /// lock so the readback is a consistent post-write snapshot.
    fn asset_orphan_safe_to_remove(&self, id: &WorkspaceId, kind: AssetKind, name: &str) -> bool {
        match self.read_metadata(id) {
            Ok(meta) => meta.find_index(kind, name).is_none(),
            Err(read_err) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %read_err,
                    name = %name,
                    "post-rename commit failed and metadata readback also failed; \
                     refusing to unlink the asset (cannot prove it is unreferenced); \
                     orphan/row surfaces in validate()",
                );
                false
            }
        }
    }

    /// Fallible inner of [`Self::commit_asset_after_rename`]; failure leaves `final_path`
    /// on disk for the caller's cleanup arm.
    #[allow(clippy::too_many_arguments)]
    fn try_commit_asset(
        &self,
        id: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        digest: String,
        total: u64,
        final_path: &Path,
        mut meta: crate::file_mgr::WorkspaceMetadata,
    ) -> Result<AssetReceipt, FileError> {
        // fsync the parent so the rename's dirent is durable; content fsync is the caller's
        // contract (writer.sync_all / the staging caller).
        if let Some(parent) = final_path.parent() {
            fsync_dir(parent).map_err(|e| io_err(parent.display(), e))?;
        }
        // Re-parse the validated name into the typed id; `?` guards validator drift.
        let asset_id = AssetId::parse(name)?;
        let record = AssetRecord {
            kind,
            name: asset_id,
            sha256: digest.clone(),
            size_bytes: total,
        };
        let upsert_record = record.clone();
        if let Some(idx) = meta.find_index(kind, name) {
            meta.assets[idx] = upsert_record;
        } else {
            meta.assets.push(upsert_record);
        }
        self.write_metadata(id, &meta)?;
        Ok(AssetReceipt {
            kind: record.kind,
            name: record.name.into(),
            sha256: digest,
            size_bytes: total,
            path: final_path.to_path_buf(),
        })
    }
}

/// Reject `name` only when it case-insensitively collides with an existing `kind` record
/// yet differs in case (an exact match is an allowed overwrite).
fn check_name_conflict(
    meta: &crate::file_mgr::WorkspaceMetadata,
    kind: AssetKind,
    name: &str,
) -> Result<(), FileError> {
    if let Some(existing) = meta.find_case_insensitive(kind, name)
        && existing.name.as_str() != name
    {
        return Err(FileError::NameConflict(format!(
            "{kind:?} asset {name:?} collides case-insensitively with existing {:?}",
            existing.name
        )));
    }
    Ok(())
}

/// Best-effort unlink of an orphan asset file (renamed but with no `metadata.json` row)
/// after a failed commit. Unlink failures warn (caller still propagates the commit error);
/// `NotFound` is silent (steady state when the rename's dirent wasn't durable).
fn cleanup_orphan_after_failed_commit(final_path: &Path) {
    match std::fs::remove_file(final_path) {
        Ok(()) => {
            tracing::debug!(
                target: "file_mgr",
                path = %final_path.display(),
                "post-rename commit failed; orphan asset file removed",
            );
            // fsync the unlink durable: asset orphans aren't on boot recovery's sweep
            // list, so otherwise a crash post-unlink can re-expose the dirent.
            if let Some(parent) = final_path.parent()
                && let Err(e) = fsync_dir(parent)
            {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %parent.display(),
                    "post-rename commit cleanup unlink succeeded but parent fsync \
                     failed; orphan dirent may persist across crash until validate()",
                );
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Steady state for a non-durable rename dirent; silent by design.
        }
        Err(e) => {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %final_path.display(),
                "post-rename commit failed and orphan cleanup also failed; \
                 file will surface in validate()",
            );
        }
    }
}

#[cfg(test)]
mod cleanup_tests {
    //! Pin the commit-failure cleanup helper's contract.
    use super::cleanup_orphan_after_failed_commit;

    #[test]
    fn cleanup_removes_existing_orphan() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("orphan.bin");
        #[allow(clippy::disallowed_methods)]
        std::fs::write(&path, b"orphan bytes").expect("seed orphan");
        assert!(path.is_file(), "fixture must materialize the orphan");
        cleanup_orphan_after_failed_commit(&path);
        assert!(
            !path.exists(),
            "cleanup_orphan_after_failed_commit must unlink an existing orphan",
        );
    }

    #[test]
    fn cleanup_is_silent_on_missing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("never-existed");
        assert!(!path.exists());
        cleanup_orphan_after_failed_commit(&path);
        assert!(!path.exists());
    }
}
