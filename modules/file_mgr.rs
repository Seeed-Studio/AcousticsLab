//! Workspace + asset file manager.
//!
//! Workspace index and artifact writes go through `fs_atomic::put_atomic` (tempfile, fsync,
//! rename, parent-dir fsync) so a partial write is never visible (the per-job JSONL event log
//! instead appends with an open-time dir-fsync and a torn-tail `set_len` discipline; see
//! [`jsonl_event_log`]). Artifact writes (`<head_id>.mpk` + `.json`) pair with
//! index commits (`heads.json` + `workspace.json`) under the per-workspace mutation mutex; a
//! crash between the two leaves orphan bytes that boot recovery ([`recovery::recover_all`],
//! run once before the api goes live) sweeps.

#![warn(missing_debug_implementations)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(test)]
use crate::common::ids::AssetId;
use crate::common::ids::WorkspaceId;
use crate::common::workspace::{HeadIndex, HeadStatus, WorkspaceCore};
use serde::Serialize;
#[cfg(test)]
use sha2::{Digest, Sha256};

mod error;
pub mod fs_atomic;
// Colocated with its DTO types so `common` need not depend on `time`.
pub mod cache;
pub mod fs_service;
mod metadata;
pub mod schema;
pub mod staging;
pub mod time_util;
mod validate;

pub mod log_page;

pub mod active_head_writer;
mod asset;
pub mod dataset;
pub mod head_rotation;
pub mod job_registry;
pub mod mime;
pub mod recovery;
mod registry;
pub mod request_payload;
pub mod storage_reaper;
// Cap enforced synchronously from `JsonlEventLog::open`, the only moment it can be exceeded.
pub mod log_retention;
// Wire envelope `{seq, at, ...event}` must match the [`log_page::LogEvent`] reader.
pub mod jsonl_event_log;
mod uploader;

// Inversion hook: daemon publishes into `status::WorkspaceMetrics` without `file_mgr`
// referencing `status`, preserving the `file_mgr -> common` dep-edge guard.
pub mod metrics_hooks;

pub use active_head_writer::{
    ActivationError, ActivationOriginInput, ActivationResult, HeadInnerLoader, PendingActivation,
    active_source_head_in_workspace, prune_old_generations, publish_active_generation,
    stage_and_validate_activation, staging_path_for,
};
pub use cache::WorkspaceCacheCell;
pub use dataset::{
    CONVERTER_LOGS_DIR_NAME, DATASETS_DIR_NAME, DEFAULT_DATASET_LIST_LIMIT, DatasetEntry,
    DatasetListing, DatasetUploadReceipt, EntryKind, MAX_DATASET_LIST_LIMIT, RenameCategoryReceipt,
    TRAINING_LOGS_DIR_NAME,
};
pub use error::FileError;
pub(crate) use error::io_err;
pub use fs_atomic::put_atomic;
pub use fs_service::{
    FileMetadataGuard, FsError, FsService, FsServiceImpl, MetadataGuard, UploadPermit,
};
pub use head_rotation::{HeadImportResult, HeadRotationResult, PendingHead, publish_trained_head};
pub use job_registry::{
    EventGap, EventStream, EventStreamError, JobEvent, JobHandle, JobProgress, JobRegistry,
    JobRegistryCfg, JobRegistryCounters, JobResult as RegistryJobResult, JobSnapshot,
    JobState as RegistryJobState, LeaseGuard as JobRegistryLeaseGuard, RegistryConflict,
    RegistryEvent,
};
pub use jsonl_event_log::JsonlEventLog;
pub use log_retention::{LOG_RETENTION_KEEP_COUNT, RetentionReport, enforce_keep_last_n};
pub use metadata::{AssetKind, AssetRecord, WorkspaceMetadata};
pub use mime::content_type_from_path;
pub use recovery::{
    RecoveryActiveResult, RecoveryError, RecoveryReport, RecoveryRootReport,
    RecoveryWorkspaceReport, recover_active_head, recover_all, recover_root_staging,
    recover_workspaces,
};
pub use request_payload::{
    ConvertRequest, ConverterPath, ConverterPathError, LabelsFormat, MAX_BATCH_SIZE, MAX_EPOCHS,
    MAX_LEARNING_RATE, MIN_BATCH_SIZE, MIN_EPOCHS, TfjsConvertParams, TrainRequest, TrainingCfg,
    ValidationError, canonical_training_cfg_sha256, from_manifest_value, to_manifest_value,
    validate_convert_request, validate_training_cfg,
};
pub use schema::{
    ACTIVE_CURRENT_FILENAME, ACTIVE_DIR_NAME, ACTIVE_GENERATIONS_DIR_NAME, ACTIVE_HEAD_FILENAME,
    ACTIVE_LABELS_FILENAME, ACTIVE_MANIFEST_FILENAME, ACTIVE_TMP_DIR_NAME, ActiveCurrentPointer,
    HEAD_ARTIFACT_EXTENSION, HEAD_INDEX_FILENAME, HEAD_MANIFEST_EXTENSION, HEADS_DIR_NAME,
    MAX_WORKSPACE_CORE_BYTES, ROOT_TMP_DIR_NAME, WORKSPACE_CORE_FILENAME, WORKSPACES_DIR_NAME,
    active_current_path, active_dir, active_generation_dir, active_generations_dir,
    active_staging_dir, head_artifact_path, head_index_path, head_manifest_path, heads_dir,
    read_active_current, read_active_manifest, read_head_index, read_head_manifest,
    read_workspace_core, root_tmp_dir, workspace_core_path, workspace_dir_for, workspaces_dir,
    write_active_current, write_active_manifest, write_head_index, write_head_manifest,
    write_workspace_core,
};
pub use staging::{
    DATASET_TOMBSTONE_PREFIX, DEFAULT_DELETE_BATCH_ENTRIES, DeleteTombstone, DrainResult,
    STAGED_PAYLOAD_NAME, StagedDelete, WORKSPACE_TOMBSTONE_PREFIX, drain_staged_payload,
    finalize_staged_delete, read_tombstone, stage_payload, write_tombstone,
};
pub use storage_reaper::{SweepConfig, SweepReport, sweep_once};
pub use time_util::now_rfc3339;
pub use uploader::AdmissionCfg;
pub(crate) use validate::hex_lowercase;
pub(crate) use validate::sha256_file_streaming;
pub use validate::validate_asset_name;

#[cfg(test)]
use validate::validate_extension;

/// Workspace + asset manager. Thread-safe (`&self`), cheap to clone (`Arc` bumps);
/// external consumers reach it through [`fs_service::FsService`].
///
/// Index-atomic publishes (`heads.json` + `workspace.json`) serialize per workspace via
/// a `DashMap`-sharded `Mutex<()>`, so distinct workspaces don't contend. `metadata_locks`
/// and `caches` stay bounded by live workspace count (inserted on create/lazy-load, ejected
/// in lockstep on delete); every `metadata_lock` caller pre-checks `workspace.json` existence
/// so a never-created id cannot strand a lock entry.
#[derive(Clone, Debug)]
pub struct WorkspaceMgr {
    pub(crate) root: PathBuf,
    pub(crate) metadata_locks: Arc<dashmap::DashMap<WorkspaceId, Arc<parking_lot::Mutex<()>>>>,
    /// Serializes [`Self::create`]'s name-uniqueness check + commit (in-process only) so two
    /// concurrent `create("main")` can't both observe an empty registry and both succeed.
    pub(crate) registry_lock: Arc<parking_lot::Mutex<()>>,
    /// `None` disables admission control; `Some` enforces caps + a concurrency gate.
    pub(crate) admission: Option<Arc<uploader::AdmissionState>>,
    /// Per-workspace cache of `workspace.json` + `heads.json`; disk is read only on cold miss.
    pub(crate) caches: Arc<dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>>>,
    pub(crate) jobs: Arc<job_registry::JobRegistry>,
}

impl WorkspaceMgr {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            metadata_locks: Arc::new(dashmap::DashMap::new()),
            registry_lock: Arc::new(parking_lot::Mutex::new(())),
            admission: None,
            caches: Arc::new(dashmap::DashMap::new()),
            jobs: Arc::new(job_registry::JobRegistry::new(
                job_registry::JobRegistryCfg::default(),
            )),
        }
    }

    /// Admission caps enforced with a private job registry; production boot uses
    /// [`Self::with_admission_and_jobs`] to share one.
    #[cfg(test)]
    pub(crate) fn with_admission(root: PathBuf, cfg: AdmissionCfg) -> Self {
        Self {
            root,
            metadata_locks: Arc::new(dashmap::DashMap::new()),
            registry_lock: Arc::new(parking_lot::Mutex::new(())),
            admission: Some(Arc::new(uploader::AdmissionState::new(cfg))),
            caches: Arc::new(dashmap::DashMap::new()),
            jobs: Arc::new(job_registry::JobRegistry::new(
                job_registry::JobRegistryCfg::default(),
            )),
        }
    }

    /// Shares a pre-configured [`JobRegistry`] so the api `AppState`'s registry (read by
    /// `GET /jobs`) is the same instance admission paths register against.
    pub fn with_admission_and_jobs(
        root: PathBuf,
        cfg: AdmissionCfg,
        jobs: Arc<job_registry::JobRegistry>,
    ) -> Self {
        Self {
            root,
            metadata_locks: Arc::new(dashmap::DashMap::new()),
            registry_lock: Arc::new(parking_lot::Mutex::new(())),
            admission: Some(Arc::new(uploader::AdmissionState::new(cfg))),
            caches: Arc::new(dashmap::DashMap::new()),
            jobs,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Wire receipt from [`WorkspaceMgr::upload`] / [`WorkspaceMgr::install_from_path`].
#[derive(Debug, Clone, Eq, PartialEq, Serialize)]
pub struct AssetReceipt {
    pub kind: AssetKind,
    pub name: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub path: PathBuf,
}

/// [`WorkspaceMgr::validate`] result: `ok` iff `missing` and `corrupt` are both empty;
/// `extra` holds on-disk files unknown to the metadata.
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceReport {
    pub ok: bool,
    pub missing: Vec<(AssetKind, String)>,
    pub corrupt: Vec<(AssetKind, String)>,
    pub extra: Vec<(PathBuf, String)>,
}

/// [`WorkspaceMgr::summary`] result; `head_statuses` is derived at summary time from the
/// current `workspace_revision.id`, never persisted.
#[derive(Debug, Clone)]
pub struct WorkspaceSummary {
    pub core: Arc<WorkspaceCore>,
    pub heads: Arc<HeadIndex>,
    pub head_statuses: Vec<HeadStatus>,
}

#[cfg(test)]
mod tests;
