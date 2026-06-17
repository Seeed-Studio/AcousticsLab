//! Workspace + asset error type.

use crate::file_mgr::metadata::AssetKind;
use thiserror::Error;

/// HTTP statuses via the [`crate::common::error::Categorized`] impl.
#[derive(Debug, Error)]
pub enum FileError {
    #[error("workspace not found: {0}")]
    NotFound(String),
    #[error("asset not found in workspace {ws}: {kind:?} {name}")]
    AssetNotFound {
        ws: String,
        kind: AssetKind,
        name: String,
    },
    #[error("invalid identifier: {0}")]
    Id(#[from] crate::common::ids::IdError),
    #[error("invalid asset name: {0}")]
    InvalidName(String),
    #[error("invalid asset extension: got {got}, expected one of {expected:?}")]
    InvalidExtension {
        got: String,
        expected: Vec<&'static str>,
    },
    #[error("workspace name conflict: {0}")]
    NameConflict(String),
    /// Refused so an older daemon can't overwrite a newer-shape file and silently lose
    /// fields; bump [`crate::file_mgr::WorkspaceMetadata::CURRENT`] when adding such fields.
    #[error(
        "workspace {path} schema version {found} is newer than this build (max {max}); upgrade the daemon"
    )]
    SchemaTooNew { path: String, found: u32, max: u32 },
    /// Below [`crate::file_mgr::WorkspaceMetadata::MIN_COMPATIBLE`].
    #[error(
        "workspace {path} schema version {found} is older than this build supports (min {min}); migrate or recreate the workspace"
    )]
    SchemaTooOld { path: String, found: u32, min: u32 },
    #[error("io {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("metadata parse {path}: {source}")]
    MetadataParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    /// `#[from]` keeps `serde_json::to_*` sites boilerplate-free; trades path context
    /// for surface (trigger is allocator failure / non-serializable type: rare, fatal).
    #[error("metadata serialize: {0}")]
    MetadataSerialize(#[from] serde_json::Error),
    /// Common cause is `EXDEV`: the daemon's `.tmp/` must share a filesystem with the
    /// persist target. `#[from]` trades path context for surface.
    #[error("persist tempfile: {0}")]
    Persist(#[from] tempfile::PersistError),
    /// Detected mid-stream so the tempfile is dropped (no partial commit, no metadata
    /// row). Maps to 400 (canonical 413 folds into `UserInput`).
    #[error("upload exceeded max_upload_bytes: {observed} > {max}")]
    PayloadTooLarge { observed: u64, max: u64 },
    /// At [`crate::file_mgr::AdmissionCfg::max_concurrent_uploads`]; retriable once an
    /// in-flight upload finishes. Maps to 409.
    #[error("too many concurrent uploads: {active}/{max}")]
    TooManyConcurrentUploads { active: u32, max: u32 },
    /// On-disk metadata exceeded its per-file size cap (daemon-internal corruption or
    /// tampering with a daemon-owned file); each metadata file has its own cap in
    /// `crate::file_mgr::schema` (e.g. `MAX_WORKSPACE_CORE_BYTES` = 64 KiB for
    /// `workspace.json`), enforced by `read_capped`.
    #[error("metadata at {path} too large: {observed} bytes > {max}")]
    MetadataTooLarge {
        path: String,
        observed: u64,
        max: u64,
    },
    /// Another running job references the target workspace or an ancestor/descendant
    /// dataset path (`DatasetRefRegistry` asset scope, `JobRegistry` job scope). 409.
    #[error("job conflict: {message}")]
    JobConflict { message: String },
    /// `max_train_jobs = 1` invariant: one unfinished train job daemon-wide. Distinct
    /// from `JobConflict` so the api layer renders the `another_train_running`
    /// discriminator. 409.
    #[error("another train job is already running daemon-wide (max_train_jobs = 1)")]
    AnotherTrainRunning,
    /// Would remove the head the active generation sourced from. Inference survives
    /// (active gen owns a copy), but `heads.json`'s row would dangle under `GET
    /// /active`'s `source_head_id`. 409.
    #[error(
        "head {head_id} is the current active source for workspace {workspace_id}; \
         activate a different head before evicting / deleting it"
    )]
    ActiveSourcePinned {
        workspace_id: String,
        head_id: String,
    },
    /// `.alpkg` import: same `head_id` in the destination but different `sha256`.
    /// Refused rather than overwritten to avoid silently invalidating external refs
    /// pinned to the original sha256. 409, `head_id_collision` discriminator.
    #[error(
        "head {head_id} already exists in this workspace with a different sha256 \
         (got {got_sha256}, stored {stored_sha256}); delete the existing head before re-importing"
    )]
    HeadIdCollision {
        head_id: String,
        got_sha256: String,
        stored_sha256: String,
    },
}

pub(crate) fn io_err(path: impl std::fmt::Display, source: std::io::Error) -> FileError {
    FileError::Io {
        path: path.to_string(),
        source,
    }
}

pub(crate) fn metadata_parse_err(
    path: impl std::fmt::Display,
    source: serde_json::Error,
) -> FileError {
    FileError::MetadataParse {
        path: path.to_string(),
        source,
    }
}

impl crate::common::error::Categorized for FileError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            FileError::NotFound(_) | FileError::AssetNotFound { .. } => NotFound,
            FileError::Id(e) => e.kind(),
            FileError::InvalidName(_) | FileError::InvalidExtension { .. } => UserInput,
            FileError::NameConflict(_) => Conflict,
            FileError::PayloadTooLarge { .. } => UserInput,
            FileError::TooManyConcurrentUploads { .. } => Conflict,
            FileError::SchemaTooNew { .. } | FileError::SchemaTooOld { .. } => Conflict,
            FileError::JobConflict { .. } => Conflict,
            FileError::AnotherTrainRunning => Conflict,
            FileError::ActiveSourcePinned { .. } => Conflict,
            FileError::HeadIdCollision { .. } => Conflict,
            FileError::Io { .. }
            | FileError::MetadataParse { .. }
            | FileError::MetadataSerialize(_)
            | FileError::MetadataTooLarge { .. }
            | FileError::Persist(_) => Internal,
        }
    }
}
