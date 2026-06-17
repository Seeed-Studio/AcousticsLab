//! Object-safe workspace + asset filesystem trait surface, held by api as
//! `Arc<dyn FsService>`; production impl [`FsServiceImpl`] facades [`WorkspaceMgr`].
//!
//! Trait + DTOs live here not in `common` because hoisting [`WorkspaceMetadata`] would force a
//! `time` dep on common (its zero-workspace-dep rule). All methods are sync; async callers wrap
//! the critical section in [`tokio::task::spawn_blocking`].

use crate::common::asset_path::AssetPath;
use crate::common::error::{Categorized, ErrorKind};
use crate::common::ids::{HeadId, JobId, WorkspaceId};
use crate::file_mgr::dataset::{DatasetListing, DatasetUploadReceipt, RenameCategoryReceipt};
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::metadata::{AssetKind, AssetRecord, WorkspaceMetadata};
use crate::file_mgr::{AssetReceipt, WorkspaceReport, WorkspaceSummary};
use std::path::{Path, PathBuf};

/// Type-erased error returned by every [`FsService`] method. Captures [`ErrorKind`] alongside the
/// boxed [`FileError`] so HTTP status mapping via [`Categorized`] works without downcasts.
#[derive(Debug)]
pub struct FsError {
    inner: Box<dyn std::error::Error + Send + Sync + 'static>,
    kind: ErrorKind,
}

impl FsError {
    pub fn new<E>(e: E) -> Self
    where
        E: std::error::Error + Send + Sync + Categorized + 'static,
    {
        let kind = e.kind();
        Self {
            inner: Box::new(e),
            kind,
        }
    }
}

impl std::fmt::Display for FsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.inner, f)
    }
}

impl std::error::Error for FsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.inner.as_ref())
    }
}

impl Categorized for FsError {
    fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl From<FileError> for FsError {
    fn from(e: FileError) -> Self {
        Self::new(e)
    }
}

/// RAII permit gating concurrent uploads, held across staging + [`FsService::install_from_path`];
/// drop releases the slot. Opaque `Box<dyn Send>` lets the impl stash a tokio permit without
/// pulling tokio into the trait.
pub struct UploadPermit {
    _inner: Box<dyn Send + 'static>,
}

impl UploadPermit {
    pub fn from_guard<T: Send + 'static>(guard: T) -> Self {
        Self {
            _inner: Box::new(guard),
        }
    }
}

impl std::fmt::Debug for UploadPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UploadPermit").finish_non_exhaustive()
    }
}

/// Per-workspace metadata RMW critical section from [`FsService::metadata_mut`], holding the
/// per-workspace metadata lock for its lifetime. `!Send` (parking_lot mutex) so holding it across
/// an `.await` is compile-rejected; callers must keep it inside one [`tokio::task::spawn_blocking`].
pub trait MetadataGuard {
    fn metadata(&self) -> &WorkspaceMetadata;

    fn metadata_mut(&mut self) -> &mut WorkspaceMetadata;

    /// Atomic write + release; drop without commit discards the mutations.
    fn commit(self: Box<Self>) -> Result<(), FsError>;
}

/// Path readers are pure joins (no I/O); lifecycle/read/upload methods do blocking I/O
/// (`spawn_blocking`); the RMW pair ([`Self::metadata_mut`] + [`MetadataGuard::commit`]) holds the
/// per-workspace mutex.
pub trait FsService: Send + Sync + 'static {
    /// Workspace root; callers build sibling tempdirs on the same filesystem.
    fn root(&self) -> &Path;

    /// `<workspace>/.tmp/` (pure join). Per-writer lazy-mkdir, so callers staging tempfiles here
    /// must `create_dir_all` first (as [`Self::install_bytes`] does).
    fn workspace_tmpdir(&self, ws: &WorkspaceId) -> PathBuf;

    /// On-disk asset path (pure join); trusts the caller to have run
    /// [`crate::file_mgr::validate_asset_name`] on `name`.
    fn asset_path(&self, ws: &WorkspaceId, kind: AssetKind, name: &str) -> PathBuf;

    fn create(&self, name: &str) -> Result<WorkspaceId, FsError>;
    /// Tags are trimmed + validated; empty -> `tags = []`.
    fn create_with_tags(&self, name: &str, tags: &[String]) -> Result<WorkspaceId, FsError>;
    /// Atomic name + tag edit (≥1 of `name`/`tags` must be `Some`); returns the published core.
    fn patch_workspace(
        &self,
        ws: &WorkspaceId,
        name: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<Arc<crate::common::workspace::WorkspaceCore>, FsError>;
    /// Synchronous workspace delete. Lock order active -> workspace: impls MUST hold the activation
    /// serialisation lock for the call ([`FsServiceImpl`] holds `active_mutex`).
    fn delete(&self, ws: &WorkspaceId) -> Result<(), FsError>;
    /// Begin async workspace delete, returning the owning `JobId`. The tree moves under root
    /// `.tmp/delete-workspace-<job_id>/payload`; drain runs off-mutex on the blocking pool and
    /// boot recovery resumes any crash-interrupted drain. Lock order active -> workspace: impls
    /// MUST hold the activation lock for the rename-to-staging step (releasable before drain).
    fn start_delete_workspace(&self, ws: &WorkspaceId) -> Result<JobId, FsError>;
    /// Idempotently create the root dir; subdirs are materialized lazily by their writers.
    fn ensure_root_layout(&self) -> Result<(), FsError>;
    fn list_workspaces(&self) -> Result<Vec<WorkspaceId>, FsError>;
    fn read_metadata(&self, ws: &WorkspaceId) -> Result<WorkspaceMetadata, FsError>;
    /// Hot-path read off the per-workspace eager cache; never walks `datasets/`.
    fn summary(&self, ws: &WorkspaceId) -> Result<WorkspaceSummary, FsError>;
    fn list_assets(&self, ws: &WorkspaceId, kind: AssetKind) -> Result<Vec<AssetRecord>, FsError>;
    fn validate(&self, ws: &WorkspaceId) -> Result<WorkspaceReport, FsError>;

    /// Locks the per-workspace metadata for RMW. Targets the RETIRED legacy `<ws>/metadata.json`
    /// store that `create` no longer writes; in-crate-test-only - production wiring would hit a store
    /// the daemon never populates. Deadlock hazard: while the guard holds `metadata_lock`, any
    /// active-mutex-locked method ([`Self::delete`], [`Self::start_delete_workspace`],
    /// [`Self::delete_head`], [`Self::publish_trained_head`], [`Self::publish_imported_head`]) on the
    /// same Arc deadlocks; drop the guard first.
    fn metadata_mut(&self, ws: &WorkspaceId) -> Result<Box<dyn MetadataGuard + '_>, FsError>;

    /// Fail-fast (non-blocking) upload permit; [`ErrorKind::Conflict`] when the global in-flight
    /// cap is hit. Hold it across the staging stream + [`Self::install_from_path`].
    fn acquire_upload_permit(&self) -> Result<UploadPermit, FsError>;

    /// Per-request upload byte cap from [`crate::file_mgr::AdmissionCfg`] for mid-stream
    /// enforcement; [`u64::MAX`] when uncapped.
    fn max_upload_bytes(&self) -> u64;

    /// Install a pre-staged asset by atomic rename + metadata commit (under the per-workspace
    /// metadata lock; validates name/extension, rejects case-insensitive collisions, hashes `src`
    /// before rename). `src` MUST be on the same filesystem as the workspace root for the intra-FS
    /// rename guarantee (typically `NamedTempFile::new_in(fs.workspace_tmpdir(ws))`).
    fn install_from_path(
        &self,
        ws: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        src: &Path,
    ) -> Result<AssetReceipt, FsError>;

    /// Workspace-rooted asset path resolution; `path` carries the tree-root (`datasets/...` or
    /// `converters/...`), `.tmp/` is rejected.
    fn workspace_asset_path(&self, ws: &WorkspaceId, path: &AssetPath) -> Result<PathBuf, FsError>;
    fn open_workspace_file(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<(PathBuf, std::fs::Metadata), FsError>;
    fn list_workspace_children(
        &self,
        ws: &WorkspaceId,
        relative: Option<&AssetPath>,
        offset: usize,
        limit: usize,
    ) -> Result<DatasetListing, FsError>;

    /// Commit a single-file workspace upload from a pre-staged tempfile; `path` must start with
    /// `datasets/` or `converters/` and carry ≥1 child.
    fn upload_workspace_file(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
        src_tmpfile: &Path,
        observed_sha256: &str,
        observed_size: u64,
    ) -> Result<DatasetUploadReceipt, FsError>;

    /// Begin async workspace asset delete, returning the owning [`JobId`]. All four mutable trees
    /// (`datasets/`, `converters/`, `training_logs/`, `converter_logs/`) share one tombstone+stage+
    /// drain shape; datasets/converters bump `workspace_revision`, log trees don't.
    fn start_workspace_asset_delete(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<JobId, FsError>;

    /// Rename a dataset category directory `datasets/<from>` -> `datasets/<to>` in place.
    /// Synchronous (one atomic `rename(2)`, moves no bytes), bumps `workspace_revision`. Rejects:
    /// missing source (404); existing target incl. case-insensitive sibling collision (409); a
    /// rename while a Train job for the workspace is active (409).
    fn rename_dataset_category(
        &self,
        ws: &WorkspaceId,
        from: &AssetPath,
        to: &AssetPath,
    ) -> Result<RenameCategoryReceipt, FsError>;

    /// Synchronous head delete: under the per-workspace mutation mutex, atomically rewrites
    /// `heads.json` + `workspace.json`, then best-effort unlinks `heads/<head_id>.{mpk,json}`. Lock
    /// order active -> workspace: impls MUST hold the activation lock so the active-pin check + index
    /// rewrite see one consistent state.
    fn delete_head(&self, ws: &WorkspaceId, head_id: HeadId) -> Result<(), FsError>;

    /// Publish a freshly trained head into the workspace head index via the per-workspace mutation
    /// mutex + cache cell. Lock order active -> workspace (as [`Self::delete_head`]): the active-pin
    /// read + heads.json rewrite must exclude `POST /active`.
    fn publish_trained_head(
        &self,
        ws: &WorkspaceId,
        pending: crate::file_mgr::PendingHead,
    ) -> Result<crate::file_mgr::HeadRotationResult, FsError>;

    /// Publish an imported head (`.alpkg` convert path) into the head index; mirrors
    /// [`Self::publish_trained_head`]'s mutex/cache-cell discipline plus an idempotency/collision
    /// check under the same mutex (three-way success / idempotent-no-op / collision contract in
    /// [`crate::file_mgr::HeadImportResult`]). Same lock order as [`Self::delete_head`].
    fn publish_imported_head(
        &self,
        ws: &WorkspaceId,
        pending: crate::file_mgr::PendingHead,
    ) -> Result<crate::file_mgr::HeadImportResult, FsError>;

    /// Path-scoped atomic write via the tempfile + sync_all + rename + dir-fsync primitive
    /// ([`crate::file_mgr::put_atomic`]); parent dir must exist (staging tempfile lands as a sibling
    /// for the intra-FS rename guarantee). Unlike [`Self::install_bytes`], writes outside the
    /// workspace asset tree (no admission/metadata commit), e.g. the converter's staging area.
    fn put_atomic(&self, path: &Path, bytes: &[u8]) -> Result<(), FsError> {
        crate::file_mgr::fs_atomic::put_atomic(path, bytes).map_err(FsError::from)
    }

    /// Stage `bytes` to a tempfile in [`Self::workspace_tmpdir`], then [`Self::install_from_path`].
    /// For training checkpoint-emit; streaming uploads use the explicit two-phase shape instead to
    /// enforce [`Self::max_upload_bytes`] mid-stream.
    fn install_bytes(
        &self,
        ws: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        bytes: &[u8],
    ) -> Result<AssetReceipt, FsError> {
        use std::io::Write;
        let tmp_dir = self.workspace_tmpdir(ws);
        // `.tmp/` is per-writer-lazy: mkdir first, else the first checkpoint emit on a fresh workspace ENOENTs.
        std::fs::create_dir_all(&tmp_dir)
            .map_err(|e| FsError::new(io_err(tmp_dir.display(), e)))?;
        let mut tmp = tempfile::NamedTempFile::new_in(&tmp_dir)
            .map_err(|e| FsError::new(io_err(tmp_dir.display(), e)))?;
        tmp.write_all(bytes)
            .map_err(|e| FsError::new(io_err(tmp.path().display(), e)))?;
        tmp.as_file()
            .sync_all()
            .map_err(|e| FsError::new(io_err(tmp.path().display(), e)))?;
        self.install_from_path(ws, kind, name, tmp.path())
    }
}

use std::sync::Arc;

use crate::file_mgr::{AdmissionCfg, WorkspaceMgr};

/// Production [`FsService`], built once at boot. Facades [`WorkspaceMgr`] (vs moving bodies onto
/// `Self`) to keep [`WorkspaceMgr`]'s in-crate unit tests on the concrete type.
#[derive(Debug, Clone)]
pub struct FsServiceImpl {
    mgr: Arc<WorkspaceMgr>,
    /// Activation serialisation lock, shared by-Arc with `api::AppState::active_mutex` and acquired
    /// internally by every active-mutex-locked trait method so the lock-order contract (active ->
    /// per-workspace metadata) lives at the trait surface, not each call site. `POST /active` holds
    /// the same Arc externally end-to-end (its non-trait calls must stay atomic and never re-enter a
    /// trait method).
    active_mutex: Arc<parking_lot::Mutex<()>>,
}

impl FsServiceImpl {
    /// Test-fixture construct: no admission control, fresh (unshared) active-mutex. Production uses
    /// [`Self::with_admission_jobs_and_active_mutex`] to share the Arc with `AppState`.
    pub fn new(root: PathBuf) -> Self {
        Self {
            mgr: Arc::new(WorkspaceMgr::new(root)),
            active_mutex: Arc::new(parking_lot::Mutex::new(())),
        }
    }

    /// Production construct: admission caps, shared [`crate::file_mgr::JobRegistry`], and the
    /// caller-owned `active_mutex` Arc shared with `api::AppState` so its external lock and the
    /// trait-internal locks serialise.
    pub fn with_admission_jobs_and_active_mutex(
        root: PathBuf,
        cfg: AdmissionCfg,
        jobs: Arc<crate::file_mgr::JobRegistry>,
        active_mutex: Arc<parking_lot::Mutex<()>>,
    ) -> Self {
        Self {
            mgr: Arc::new(WorkspaceMgr::with_admission_and_jobs(root, cfg, jobs)),
            active_mutex,
        }
    }
}

/// Production [`MetadataGuard`]. Holds an OWNED per-workspace mutex (`lock_arc`, not a borrow) so it
/// detaches from `&self` and crosses the [`tokio::task::spawn_blocking`] boundary as a `Box<dyn>`.
/// `!Send` (parking_lot default): holding one across an `.await` is compile-rejected.
pub struct FileMetadataGuard {
    // Drop runs in declaration order: `_arc_guard` last, releasing the lock after `meta`.
    mgr: Arc<WorkspaceMgr>,
    ws: WorkspaceId,
    meta: WorkspaceMetadata,
    committed: bool,
    dirty: bool,
    _arc_guard: parking_lot::ArcMutexGuard<parking_lot::RawMutex, ()>,
}

impl std::fmt::Debug for FileMetadataGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileMetadataGuard")
            .field("ws", &self.ws)
            .field("dirty", &self.dirty)
            .field("committed", &self.committed)
            .finish_non_exhaustive()
    }
}

impl MetadataGuard for FileMetadataGuard {
    fn metadata(&self) -> &WorkspaceMetadata {
        &self.meta
    }

    fn metadata_mut(&mut self) -> &mut WorkspaceMetadata {
        self.dirty = true;
        &mut self.meta
    }

    fn commit(mut self: Box<Self>) -> Result<(), FsError> {
        self.mgr
            .write_metadata(&self.ws, &self.meta)
            .map_err(FsError::new)?;
        self.committed = true; // suppress Drop's uncommitted-mutation warn
        Ok(())
    }
}

impl Drop for FileMetadataGuard {
    fn drop(&mut self) {
        if self.dirty && !self.committed {
            tracing::warn!(
                target: "file_mgr",
                ws = %self.ws,
                "MetadataGuard dropped without commit; mutations rolled back",
            );
        }
    }
}

impl FsService for FsServiceImpl {
    fn root(&self) -> &Path {
        self.mgr.root()
    }

    fn workspace_tmpdir(&self, ws: &WorkspaceId) -> PathBuf {
        crate::file_mgr::schema::workspace_dir_for(self.mgr.root(), ws).join(".tmp")
    }

    fn asset_path(&self, ws: &WorkspaceId, kind: AssetKind, name: &str) -> PathBuf {
        self.mgr.asset_path(ws, kind, name)
    }

    fn create(&self, name: &str) -> Result<WorkspaceId, FsError> {
        self.mgr.create(name).map_err(FsError::new)
    }

    fn create_with_tags(&self, name: &str, tags: &[String]) -> Result<WorkspaceId, FsError> {
        self.mgr.create_with_tags(name, tags).map_err(FsError::new)
    }

    fn patch_workspace(
        &self,
        ws: &WorkspaceId,
        name: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<Arc<crate::common::workspace::WorkspaceCore>, FsError> {
        self.mgr
            .patch_workspace(ws, name, tags)
            .map_err(FsError::new)
    }

    fn delete(&self, ws: &WorkspaceId) -> Result<(), FsError> {
        // Lock order active -> workspace; without it delete races POST /active and the tree vanishes
        // mid-activation, flipping `source_workspace_alive` false.
        let _active_guard = self.active_mutex.lock();
        self.mgr.delete(ws).map_err(FsError::new)
    }

    fn start_delete_workspace(&self, ws: &WorkspaceId) -> Result<JobId, FsError> {
        // Lock order active -> workspace: the rename-to-staging is observable to a concurrent
        // activation reading the source path. Drain runs unlocked off the blocking pool.
        let _active_guard = self.active_mutex.lock();
        self.mgr.start_delete_workspace(ws).map_err(FsError::new)
    }

    fn ensure_root_layout(&self) -> Result<(), FsError> {
        self.mgr.ensure_root_layout().map_err(FsError::new)
    }

    fn list_workspaces(&self) -> Result<Vec<WorkspaceId>, FsError> {
        self.mgr.list_workspaces().map_err(FsError::new)
    }

    fn read_metadata(&self, ws: &WorkspaceId) -> Result<WorkspaceMetadata, FsError> {
        self.mgr.read_metadata(ws).map_err(FsError::new)
    }

    fn summary(&self, ws: &WorkspaceId) -> Result<WorkspaceSummary, FsError> {
        self.mgr.summary(ws).map_err(FsError::new)
    }

    fn list_assets(&self, ws: &WorkspaceId, kind: AssetKind) -> Result<Vec<AssetRecord>, FsError> {
        self.mgr.list_assets(ws, kind).map_err(FsError::new)
    }

    fn validate(&self, ws: &WorkspaceId) -> Result<WorkspaceReport, FsError> {
        self.mgr.validate(ws).map_err(FsError::new)
    }

    fn metadata_mut(&self, ws: &WorkspaceId) -> Result<Box<dyn MetadataGuard + '_>, FsError> {
        // Pre-check `workspace.json` BEFORE `metadata_lock`, else the lock map's lazy
        // `entry().or_insert_with()` strands a never-ejected entry for a missing id.
        if !crate::file_mgr::schema::workspace_core_path(
            &crate::file_mgr::schema::workspace_dir_for(self.mgr.root(), ws),
        )
        .exists()
        {
            return Err(FsError::new(FileError::NotFound(ws.to_string())));
        }
        let lock = self.mgr.metadata_lock(ws);
        // `lock_arc` yields an owned guard (no `self` lifetime) so the boxed guard can move into a
        // `spawn_blocking` closure capturing only the Arc.
        let arc_guard = lock.lock_arc();
        let meta = self.mgr.read_metadata(ws).map_err(FsError::new)?;
        Ok(Box::new(FileMetadataGuard {
            mgr: self.mgr.clone(),
            ws: *ws,
            meta,
            committed: false,
            dirty: false,
            _arc_guard: arc_guard,
        }))
    }

    fn acquire_upload_permit(&self) -> Result<UploadPermit, FsError> {
        match self.mgr.try_acquire_upload_permit().map_err(FsError::new)? {
            Some(p) => Ok(UploadPermit::from_guard(p)),
            None => Ok(UploadPermit::from_guard(())),
        }
    }

    fn max_upload_bytes(&self) -> u64 {
        self.mgr.max_upload_bytes()
    }

    fn install_from_path(
        &self,
        ws: &WorkspaceId,
        kind: AssetKind,
        name: &str,
        src: &Path,
    ) -> Result<AssetReceipt, FsError> {
        self.mgr
            .install_from_path(ws, kind, name, src)
            .map_err(FsError::new)
    }

    fn workspace_asset_path(&self, ws: &WorkspaceId, path: &AssetPath) -> Result<PathBuf, FsError> {
        self.mgr
            .workspace_asset_path(ws, path)
            .map_err(FsError::new)
    }

    fn open_workspace_file(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<(PathBuf, std::fs::Metadata), FsError> {
        self.mgr.open_workspace_file(ws, path).map_err(FsError::new)
    }

    fn list_workspace_children(
        &self,
        ws: &WorkspaceId,
        relative: Option<&AssetPath>,
        offset: usize,
        limit: usize,
    ) -> Result<DatasetListing, FsError> {
        self.mgr
            .list_workspace_children(ws, relative, offset, limit)
            .map_err(FsError::new)
    }

    fn upload_workspace_file(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
        src_tmpfile: &Path,
        observed_sha256: &str,
        observed_size: u64,
    ) -> Result<DatasetUploadReceipt, FsError> {
        self.mgr
            .upload_workspace_file(ws, path, src_tmpfile, observed_sha256, observed_size)
            .map_err(FsError::new)
    }

    fn start_workspace_asset_delete(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<JobId, FsError> {
        self.mgr
            .start_workspace_asset_delete(ws, path)
            .map_err(FsError::new)
    }

    fn rename_dataset_category(
        &self,
        ws: &WorkspaceId,
        from: &AssetPath,
        to: &AssetPath,
    ) -> Result<RenameCategoryReceipt, FsError> {
        // No `active_mutex`: a category rename touches no active-head state.
        self.mgr
            .rename_dataset_category(ws, from, to)
            .map_err(FsError::new)
    }

    fn delete_head(&self, ws: &WorkspaceId, head_id: HeadId) -> Result<(), FsError> {
        let _active_guard = self.active_mutex.lock();
        self.mgr.delete_head(ws, head_id).map_err(FsError::new)
    }

    fn publish_trained_head(
        &self,
        ws: &WorkspaceId,
        pending: crate::file_mgr::PendingHead,
    ) -> Result<crate::file_mgr::HeadRotationResult, FsError> {
        let _active_guard = self.active_mutex.lock();
        self.mgr
            .publish_trained_head_for_workspace(ws, pending)
            .map_err(FsError::new)
    }

    fn publish_imported_head(
        &self,
        ws: &WorkspaceId,
        pending: crate::file_mgr::PendingHead,
    ) -> Result<crate::file_mgr::HeadImportResult, FsError> {
        let _active_guard = self.active_mutex.lock();
        self.mgr
            .publish_imported_head_for_workspace(ws, pending)
            .map_err(FsError::new)
    }
}
