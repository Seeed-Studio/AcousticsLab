//! Workspace, head, active-head, and job schema contracts (no I/O).
//!
//! [`HeadRecord`] (on-disk index entry) is distinct from the Burn-derived `model::HeadRecord`; refer by qualified path.

use crate::common::error::{Categorized, ErrorKind};
use crate::common::ids::{HeadId, WorkspaceId, default_runtime_head_id};
use thiserror::Error;

/// Monotonic mutation counter over `datasets/`+`converters/`, snapshotted by
/// heads at producer start so stale-vs-current is one integer compare; boot
/// recovery may bump conservatively but must never leave a head current after a file mutation. Name/tag edits do not bump.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceRevision {
    /// Strictly increases on accepted file mutations, 0 on create.
    pub id: u64,
    /// RFC3339 wall-clock of the bump.
    pub at: String,
}

/// Hot-path workspace metadata at `workspaces/<id>/workspace.json`, held in the
/// `ArcSwap` core cache; listing/summary reads this + `heads.json` only, never walking `datasets/`/`converters/`.
///
/// No `schema_version` field: adding one preemptively would trap downgrade via
/// `deny_unknown_fields`; a future bump adds `schema_version: u32` with `#[serde(default)]` plus version-gated reads.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkspaceCore {
    /// Matches the directory name.
    pub id: WorkspaceId,
    pub name: String,
    /// Tag edits do not advance `workspace_revision` or affect head status.
    pub tags: Vec<String>,
    pub created_at: String,
    /// Bumped before bytes mutate so a crash can't leave a head current after a file change.
    pub workspace_revision: WorkspaceRevision,
    /// Count of heads in `heads.json`; boot recovery repairs from len.
    pub head_count: u8,
}

/// On-disk head index at `workspaces/<id>/heads.json`: sliding window,
/// most-recent-first; only successful publishes (failed jobs never appear).
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadIndex {
    /// Up to [`MAX_HEADS_PER_WORKSPACE`] entries, most-recent-first.
    pub heads: Vec<HeadRecord>,
}

/// Retained-heads cap; the next publish evicts the oldest non-pinned slot.
pub const MAX_HEADS_PER_WORKSPACE: usize = 3;

/// `head_count` is `u8` and rotation/recovery cast `heads.len() as u8`, so a cap past 255 would silently truncate.
const _: () = assert!(
    MAX_HEADS_PER_WORKSPACE <= u8::MAX as usize,
    "MAX_HEADS_PER_WORKSPACE must fit in u8 for the persisted head_count cast"
);

/// Compact head index entry; input metadata and labels live in the per-head [`HeadManifest`] so summaries stay small.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadRecord {
    pub head_id: HeadId,
    /// Snapshotted at producer start.
    pub workspace_revision: WorkspaceRevision,
    /// Hex SHA-256 of `<head_id>.mpk`; checked by activation pre-load + boot verify.
    pub sha256: String,
    /// Duplicated from the manifest so the summary API surfaces it without opening the `.mpk`.
    pub n_classes: u32,
    pub size_bytes: u64,
    pub created_at: String,
}

/// Per-head manifest beside the `.mpk` at `heads/<head_id>.json`. Index-atomic
/// publish: staged `.mpk`+`.json` are renamed before `heads.json` references them, so a pre-commit crash leaves only unreferenced files.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadManifest {
    pub head_id: HeadId,
    pub workspace_id: WorkspaceId,
    pub workspace_revision: WorkspaceRevision,
    /// Hex SHA-256 of `<head_id>.mpk`.
    pub sha256: String,
    /// Must equal `labels.len()`.
    pub n_classes: u32,
    pub size_bytes: u64,
    pub created_at: String,
    /// Inference-order labels, inline so publish is index-atomic; activation derives `active/labels.txt` from this.
    pub labels: Vec<String>,
}

/// Derived freshness of a [`HeadRecord`] vs its workspace's current revision;
/// computed on demand, never persisted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadStatus {
    /// Produced at the workspace's current revision id.
    Current,
    /// Produced at an older revision; may still serve.
    Stale,
}

impl HeadStatus {
    /// Compares `id` only; a head id above the workspace's is corruption read as `Stale` to fail closed.
    #[inline]
    pub fn from_revisions(head: &WorkspaceRevision, workspace: &WorkspaceRevision) -> Self {
        if head.id == workspace.id {
            Self::Current
        } else {
            Self::Stale
        }
    }
}

/// Structural-invariant failure raised by [`HeadManifest::validate`].
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum HeadValidationError {
    #[error("head manifest has n_classes = 0")]
    ZeroClasses,
    #[error("head manifest n_classes ({n_classes}) != labels.len() ({labels_len})")]
    ClassCountMismatch { n_classes: u32, labels_len: usize },
    /// Control char (C0/DEL/C1) breaks the `labels.txt` `\n`-join / `\n`+`trim_end_matches('\r')`-split round-trip and `labels_sha256`.
    #[error(
        "head manifest label at index {index} contains a control character (U+{codepoint:04X})"
    )]
    LabelControlChar { index: usize, codepoint: u32 },
    /// Empty/blank: the runtime loader drops blank lines, desyncing the count from `n_classes`.
    #[error("head manifest label at index {index} is empty")]
    EmptyLabel { index: usize },
    /// Over [`MAX_LABEL_BYTES`]: `head::load_inner` rejects it; gate at the byte-owning boundary, not a late activation.
    #[error(
        "head manifest label at index {index} is {len} bytes, over the {MAX_LABEL_BYTES}-byte cap"
    )]
    LabelTooLong { index: usize, len: usize },
}

impl Categorized for HeadValidationError {
    fn kind(&self) -> ErrorKind {
        // Producer bug or hand-tampered manifest, never operator input.
        ErrorKind::Internal
    }
}

impl HeadManifest {
    /// Structural invariants beyond serde; call before trusting on-disk values.
    pub fn validate(&self) -> Result<(), HeadValidationError> {
        if self.n_classes == 0 {
            return Err(HeadValidationError::ZeroClasses);
        }
        let labels_len = self.labels.len();
        if (self.n_classes as usize) != labels_len {
            return Err(HeadValidationError::ClassCountMismatch {
                n_classes: self.n_classes,
                labels_len,
            });
        }
        validate_labels::<HeadValidationError>(&self.labels)?;
        Ok(())
    }
}

/// Per-label byte ceiling, single source of truth: `head::load_inner` rejects
/// longer labels with `HeaderCorrupt`, so gating producer/import here moves rejection to the byte-owning boundary, not a late `POST /active`.
pub const MAX_LABEL_BYTES: usize = 256;

/// Reject control chars (C0/DEL/C1), empty/whitespace-only, or over
/// [`MAX_LABEL_BYTES`]; shared by both validators so the labels-render contract is single-rooted.
fn validate_labels<E: From<LabelError>>(labels: &[String]) -> Result<(), E> {
    for (index, label) in labels.iter().enumerate() {
        // Trim mirrors `head::load_inner`; bare `is_empty()` would let whitespace-only inflate `n_classes` then fail late at activation.
        if label.trim().is_empty() {
            return Err(E::from(LabelError::Empty { index }));
        }
        // Trimmed length matches the loader's post-trim cap check.
        let trimmed_len = label.trim().len();
        if trimmed_len > MAX_LABEL_BYTES {
            return Err(E::from(LabelError::TooLong {
                index,
                len: trimmed_len,
            }));
        }
        if let Some(ch) = label.chars().find(|c| c.is_control()) {
            return Err(E::from(LabelError::ControlChar {
                index,
                codepoint: ch as u32,
            }));
        }
    }
    Ok(())
}

/// Shared-gate failure, lifted via `From` into each validator's variant.
enum LabelError {
    Empty { index: usize },
    TooLong { index: usize, len: usize },
    ControlChar { index: usize, codepoint: u32 },
}

impl From<LabelError> for HeadValidationError {
    fn from(e: LabelError) -> Self {
        match e {
            LabelError::Empty { index } => Self::EmptyLabel { index },
            LabelError::TooLong { index, len } => Self::LabelTooLong { index, len },
            LabelError::ControlChar { index, codepoint } => {
                Self::LabelControlChar { index, codepoint }
            }
        }
    }
}

/// Provenance of an active head generation; flattens into
/// [`ActiveHeadManifest`] under the `"origin"` discriminator.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "origin", rename_all = "snake_case")]
pub enum ActiveOrigin {
    /// Bundled deployment-default head; no source fields.
    Default,
    /// Per-workspace head; provenance recorded so a deleted source surfaces `source_workspace_alive: false`.
    Head {
        source_workspace_id: WorkspaceId,
        source_head_id: HeadId,
        workspace_revision: WorkspaceRevision,
    },
}

/// Active-head manifest at `active/generations/<id>/manifest.json`. The
/// generation owns independent bytes (`head.mpk`, `labels.txt`) so deleting the source workspace cannot break inference.
/// Boot recovery stream-hashes both against `sha256`/`labels_sha256`; on `labels.txt`-only mismatch it regenerates from `labels[]`.
///
/// `deny_unknown_fields` omitted: serde rejects it with `flatten` +
/// internally-tagged [`ActiveOrigin`], so forward-compat fields are ignored; the discriminant is validated by the inner enum's shape.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ActiveHeadManifest {
    #[serde(flatten)]
    pub origin: ActiveOrigin,
    /// Stamped on every `InferenceFrame.head_id`: bundled-default UUID for `Default`, else `source_head_id`.
    pub runtime_head_id: HeadId,
    /// Hex SHA-256 of `head.mpk`.
    pub sha256: String,
    /// Hex SHA-256 of the materialized `labels.txt`.
    pub labels_sha256: String,
    pub n_classes: u32,
    /// Inference order; canonical recovery source if `labels.txt` is lost/stale.
    pub labels: Vec<String>,
    pub activated_at: String,
}

/// Structural validation failure for [`ActiveHeadManifest`], catching what
/// serde can't (flatten + internally-tagged enum defeat `deny_unknown_fields`); writer + recovery call `validate` to fail closed.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ActiveHeadValidationError {
    /// `origin=default` with a non-bundled-default `runtime_head_id`: the discriminator would lie about identity.
    #[error(
        "active head manifest has origin=default but runtime_head_id ({got}) \
         differs from the bundled default ({expected})"
    )]
    DefaultRuntimeIdMismatch { got: HeadId, expected: HeadId },
    /// `origin=head` with `runtime_head_id != source_head_id`: frames would carry an id consumers can't disambiguate.
    #[error(
        "active head manifest has origin=head with mismatched runtime_head_id ({got}) \
         vs source_head_id ({expected})"
    )]
    HeadRuntimeIdMismatch { got: HeadId, expected: HeadId },
    #[error("active head manifest has n_classes = 0")]
    ZeroClasses,
    #[error("active head manifest n_classes ({n_classes}) != labels.len() ({labels_len})")]
    ClassCountMismatch { n_classes: u32, labels_len: usize },
    #[error(
        "active head manifest label at index {index} contains a control character (U+{codepoint:04X})"
    )]
    LabelControlChar { index: usize, codepoint: u32 },
    #[error("active head manifest label at index {index} is empty")]
    EmptyLabel { index: usize },
    #[error(
        "active head manifest label at index {index} is {len} bytes, over the {MAX_LABEL_BYTES}-byte cap"
    )]
    LabelTooLong { index: usize, len: usize },
}

impl From<LabelError> for ActiveHeadValidationError {
    fn from(e: LabelError) -> Self {
        match e {
            LabelError::Empty { index } => Self::EmptyLabel { index },
            LabelError::TooLong { index, len } => Self::LabelTooLong { index, len },
            LabelError::ControlChar { index, codepoint } => {
                Self::LabelControlChar { index, codepoint }
            }
        }
    }
}

impl Categorized for ActiveHeadValidationError {
    fn kind(&self) -> ErrorKind {
        ErrorKind::Internal
    }
}

impl ActiveHeadManifest {
    /// Structural invariants serde can't enforce; call before trusting a
    /// deserialized manifest.
    pub fn validate(&self) -> Result<(), ActiveHeadValidationError> {
        match &self.origin {
            ActiveOrigin::Default => {
                let expected = default_runtime_head_id();
                if self.runtime_head_id != expected {
                    return Err(ActiveHeadValidationError::DefaultRuntimeIdMismatch {
                        got: self.runtime_head_id,
                        expected,
                    });
                }
            }
            ActiveOrigin::Head { source_head_id, .. } => {
                if self.runtime_head_id != *source_head_id {
                    return Err(ActiveHeadValidationError::HeadRuntimeIdMismatch {
                        got: self.runtime_head_id,
                        expected: *source_head_id,
                    });
                }
            }
        }
        if self.n_classes == 0 {
            return Err(ActiveHeadValidationError::ZeroClasses);
        }
        let labels_len = self.labels.len();
        if (self.n_classes as usize) != labels_len {
            return Err(ActiveHeadValidationError::ClassCountMismatch {
                n_classes: self.n_classes,
                labels_len,
            });
        }
        validate_labels::<ActiveHeadValidationError>(&self.labels)?;
        Ok(())
    }
}

/// Convert-pipeline selector (wire shape snake_case).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConverterType {
    /// TFJS-bundled-graph -> head conversion.
    Tfjs,
    /// `.alpkg` import: `.mpk`+`.json` verified against the embedded manifest's
    /// size+sha256, published via the training rotation primitive; idempotent on (head_id, sha256).
    Alpkg,
}

/// Discriminator for typed job snapshots: long-running operations the daemon
/// bounds via the `JobRegistry`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobType {
    /// Bounded to one unfinished job daemon-wide.
    Train,
    /// Bounded to one unfinished job daemon-wide.
    Convert,
    /// Async delete under `datasets/`; tombstoned+staged, boot resumes.
    DatasetDelete,
    /// Async delete under `converters/`; tombstoned+staged, boot resumes.
    ConverterDelete,
    /// Async delete under `training_logs/`; does NOT bump `workspace_revision` (logs aren't workspace state).
    TrainingLogsDelete,
    /// Mirror of [`Self::TrainingLogsDelete`] for converter logs.
    ConverterLogsDelete,
    /// Stages the whole workspace dir under root `.tmp/`, drains in batches.
    WorkspaceDelete,
}

impl JobType {
    /// The async-delete family shares one `max_delete_jobs` admission slot; sole classifier for [`crate::file_mgr::JobRegistry::try_acquire`].
    pub(crate) fn is_delete_subtype(self) -> bool {
        matches!(
            self,
            JobType::DatasetDelete
                | JobType::ConverterDelete
                | JobType::TrainingLogsDelete
                | JobType::ConverterLogsDelete
                | JobType::WorkspaceDelete
        )
    }
}

/// State a running job touches: one whole-workspace reference per job for
/// `WorkspaceDelete` exclusion (conflict = same workspace_id + a `WorkspaceDelete` on either side).
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum JobReference {
    Workspace { workspace_id: WorkspaceId },
}

impl JobReference {
    pub fn workspace_id(&self) -> WorkspaceId {
        match self {
            JobReference::Workspace { workspace_id } => *workspace_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::error::{Categorized, ErrorKind};
    use crate::common::ids::{HeadId, WorkspaceId, default_runtime_head_id};

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }

    fn head_id() -> HeadId {
        HeadId::parse("11111111-2222-4333-8444-555555555556").unwrap()
    }

    #[test]
    fn head_status_current_when_revisions_match() {
        assert_eq!(
            HeadStatus::from_revisions(&rev(5), &rev(5)),
            HeadStatus::Current
        );
    }

    #[test]
    fn head_status_stale_when_revision_id_differs() {
        assert_eq!(
            HeadStatus::from_revisions(&rev(4), &rev(5)),
            HeadStatus::Stale
        );
        // Head ahead of workspace (boot-recovery conservative bump) is also stale.
        assert_eq!(
            HeadStatus::from_revisions(&rev(6), &rev(5)),
            HeadStatus::Stale
        );
    }

    #[test]
    fn workspace_core_round_trips() {
        let core = WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: vec!["pet-noises".to_string(), "field".to_string()],
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(5),
            head_count: 2,
        };
        let json = serde_json::to_string(&core).unwrap();
        let parsed: WorkspaceCore = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, core);
    }

    #[test]
    fn workspace_core_round_trips_with_empty_tags() {
        let core = WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(0),
            head_count: 0,
        };
        let json = serde_json::to_string(&core).unwrap();
        assert!(json.contains("\"tags\":[]"));
        let parsed: WorkspaceCore = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, core);
    }

    #[test]
    fn workspace_core_rejects_unknown_fields() {
        let bad = r#"{
            "id": "11111111-2222-4333-8444-555555555555",
            "name": "main",
            "tags": [],
            "created_at": "2026-05-07T12:34:56Z",
            "workspace_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
            "head_count": 2,
            "schema_version": 1
        }"#;
        let res: Result<WorkspaceCore, _> = serde_json::from_str(bad);
        assert!(res.is_err(), "deny_unknown_fields must reject extra keys");
    }

    #[test]
    fn workspace_revision_rejects_unknown_fields() {
        let bad = r#"{ "id": 5, "at": "2026-05-07T13:00:00Z", "extra": true }"#;
        assert!(serde_json::from_str::<WorkspaceRevision>(bad).is_err());
    }

    #[test]
    fn head_index_default_empty() {
        let idx = HeadIndex::default();
        assert!(idx.heads.is_empty());
        let json = serde_json::to_string(&idx).unwrap();
        let parsed: HeadIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, idx);
    }

    #[test]
    fn head_record_round_trips() {
        let rec = HeadRecord {
            head_id: head_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 12,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
        };
        let json = serde_json::to_string(&rec).unwrap();
        let parsed: HeadRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, rec);
    }

    /// Guards against legacy provenance fields leaking back into the schema.
    #[test]
    fn head_record_drops_round_1_provenance_fields() {
        let rec = HeadRecord {
            head_id: head_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 12,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
        };
        let json = serde_json::to_string(&rec).unwrap();
        for stale in [
            "dataset_path",
            "dataset_revision_at_train",
            "training_cfg_sha256",
            "training_cfg",
        ] {
            assert!(
                !json.contains(stale),
                "legacy field `{stale}` must not appear in serialized HeadRecord"
            );
        }
    }

    #[test]
    fn head_manifest_round_trips() {
        let manifest = HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 2,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["cat".to_string(), "dog".to_string()],
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: HeadManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, manifest);
    }

    /// Legacy shape (dataset_path/training_cfg/...) must fail to parse.
    #[test]
    fn head_manifest_rejects_unknown_fields() {
        let bad = r#"{
            "head_id": "11111111-2222-4333-8444-555555555556",
            "workspace_id": "11111111-2222-4333-8444-555555555555",
            "dataset_path": "audio_dataset",
            "dataset_revision_at_train": { "id": 5, "at": "2026-05-07T13:00:00Z" },
            "training_cfg_sha256": "abc",
            "training_cfg": {},
            "sha256": "def",
            "n_classes": 12,
            "size_bytes": 2048,
            "created_at": "2026-05-07T12:34:56Z",
            "labels": [],
            "schema_version": 2
        }"#;
        assert!(serde_json::from_str::<HeadManifest>(bad).is_err());
    }

    /// `n_classes == labels.len()` enforced in both directions.
    #[test]
    fn head_manifest_validate_class_count_consistent() {
        let mut manifest = HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 2,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["cat".to_string(), "dog".to_string()],
        };
        assert!(manifest.validate().is_ok());

        manifest.n_classes = 3;
        let err = manifest.validate().unwrap_err();
        assert!(matches!(
            err,
            HeadValidationError::ClassCountMismatch {
                n_classes: 3,
                labels_len: 2
            }
        ));

        manifest.n_classes = 2;
        manifest.labels.push("bird".to_string());
        let err = manifest.validate().unwrap_err();
        assert!(matches!(
            err,
            HeadValidationError::ClassCountMismatch {
                n_classes: 2,
                labels_len: 3
            }
        ));
    }

    /// Embedded `\n` corrupts the labels.txt round-trip.
    #[test]
    fn head_manifest_validate_rejects_embedded_newline_in_label() {
        let manifest = HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 2,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["good".to_string(), "bad\nlabel".to_string()],
        };
        let err = manifest.validate().unwrap_err();
        assert!(matches!(
            err,
            HeadValidationError::LabelControlChar {
                index: 1,
                codepoint: 0x0A
            }
        ));
    }

    /// Embedded `\r`: loader's `trim_end_matches('\r')` would break sha256 symmetry.
    #[test]
    fn head_manifest_validate_rejects_embedded_carriage_return_in_label() {
        let manifest = HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 1,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["bad\rlabel".to_string()],
        };
        let err = manifest.validate().unwrap_err();
        assert!(matches!(
            err,
            HeadValidationError::LabelControlChar {
                index: 0,
                codepoint: 0x0D
            }
        ));
    }

    /// NUL: downstream C-string parsers would truncate.
    #[test]
    fn head_manifest_validate_rejects_embedded_nul_in_label() {
        let manifest = HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 1,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["bad\0label".to_string()],
        };
        let err = manifest.validate().unwrap_err();
        assert!(matches!(
            err,
            HeadValidationError::LabelControlChar {
                index: 0,
                codepoint: 0x00
            }
        ));
    }

    /// Empty AND whitespace-only rejected, matching `head::load_inner`'s trim-then-drop (else the label desyncs the count and never activates).
    #[test]
    fn head_manifest_validate_rejects_empty_label() {
        let manifest = HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 2,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["ok".to_string(), String::new()],
        };
        let err = manifest.validate().unwrap_err();
        assert!(
            matches!(err, HeadValidationError::EmptyLabel { index: 1 }),
            "{err:?}",
        );

        let ws_only = HeadManifest {
            labels: vec!["ok".to_string(), "   \t ".to_string()],
            ..manifest
        };
        let err = ws_only.validate().unwrap_err();
        assert!(
            matches!(err, HeadValidationError::EmptyLabel { index: 1 }),
            "{err:?}",
        );
    }

    /// C1 control U+0085 (NEL): valid UTF-8 yet some line iterators split on it.
    #[test]
    fn head_manifest_validate_rejects_c1_control_in_label() {
        let manifest = HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 1,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["bad\u{0085}label".to_string()],
        };
        let err = manifest.validate().unwrap_err();
        assert!(matches!(
            err,
            HeadValidationError::LabelControlChar {
                index: 0,
                codepoint: 0x85
            }
        ));
    }

    #[test]
    fn head_manifest_validate_accepts_unicode_and_printable_labels() {
        let manifest = HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 4,
            size_bytes: 2048,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec![
                "cat".to_string(),
                "犬 (dog)".to_string(),
                "noise-source".to_string(),
                "класс_3".to_string(),
            ],
        };
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn head_validation_error_classifies_internal() {
        let err = HeadValidationError::ClassCountMismatch {
            n_classes: 3,
            labels_len: 2,
        };
        assert_eq!(err.kind(), ErrorKind::Internal);
    }

    #[test]
    fn active_head_manifest_default_origin_round_trips() {
        let manifest = ActiveHeadManifest {
            origin: ActiveOrigin::Default,
            runtime_head_id: default_runtime_head_id(),
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 1,
            labels: vec!["unknown".to_string()],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"origin\":\"default\""));
        assert!(!json.contains("source_workspace_id"));
        let parsed: ActiveHeadManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, manifest);
    }

    #[test]
    fn active_head_manifest_head_origin_round_trips() {
        let manifest = ActiveHeadManifest {
            origin: ActiveOrigin::Head {
                source_workspace_id: ws_id(),
                source_head_id: head_id(),
                workspace_revision: rev(5),
            },
            runtime_head_id: head_id(),
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 2,
            labels: vec!["cat".to_string(), "dog".to_string()],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"origin\":\"head\""));
        assert!(json.contains("source_workspace_id"));
        assert!(json.contains("\"workspace_revision\""));
        assert!(!json.contains("source_dataset_revision"));
        let parsed: ActiveHeadManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, manifest);
    }

    #[test]
    fn active_head_manifest_head_origin_requires_source_fields() {
        let bad = r#"{
            "origin": "head",
            "runtime_head_id": "11111111-2222-4333-8444-555555555556",
            "sha256": "aa",
            "labels_sha256": "bb",
            "n_classes": 12,
            "labels": [],
            "activated_at": "2026-05-07T12:34:56Z"
        }"#;
        assert!(serde_json::from_str::<ActiveHeadManifest>(bad).is_err());
    }

    #[test]
    fn active_head_validate_default_requires_bundled_runtime_id() {
        let bad = ActiveHeadManifest {
            origin: ActiveOrigin::Default,
            runtime_head_id: head_id(),
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 1,
            labels: vec!["unknown".to_string()],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        assert!(matches!(
            bad.validate(),
            Err(ActiveHeadValidationError::DefaultRuntimeIdMismatch { .. })
        ));

        let good = ActiveHeadManifest {
            runtime_head_id: default_runtime_head_id(),
            ..bad
        };
        assert!(good.validate().is_ok());
    }

    #[test]
    fn active_head_validate_head_requires_runtime_eq_source() {
        let other = HeadId::parse("11111111-2222-4333-8444-666666666666").unwrap();
        let bad = ActiveHeadManifest {
            origin: ActiveOrigin::Head {
                source_workspace_id: ws_id(),
                source_head_id: head_id(),
                workspace_revision: rev(5),
            },
            runtime_head_id: other,
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 1,
            labels: vec!["cat".to_string()],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        assert!(matches!(
            bad.validate(),
            Err(ActiveHeadValidationError::HeadRuntimeIdMismatch { .. })
        ));

        let good = ActiveHeadManifest {
            runtime_head_id: head_id(),
            ..bad
        };
        assert!(good.validate().is_ok());
    }

    /// Mirrors the `HeadValidationError` control-char gate so recovery regen can't render a `labels.txt` line count differing from `n_classes`.
    #[test]
    fn active_head_validate_rejects_embedded_newline_in_label() {
        let bad = ActiveHeadManifest {
            origin: ActiveOrigin::Default,
            runtime_head_id: default_runtime_head_id(),
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 1,
            labels: vec!["bad\nlabel".to_string()],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        let err = bad.validate().unwrap_err();
        assert!(matches!(
            err,
            ActiveHeadValidationError::LabelControlChar {
                index: 0,
                codepoint: 0x0A
            }
        ));
    }

    #[test]
    fn active_head_validate_rejects_zero_n_classes() {
        let bad = ActiveHeadManifest {
            origin: ActiveOrigin::Default,
            runtime_head_id: default_runtime_head_id(),
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 0,
            labels: vec![],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        assert!(matches!(
            bad.validate(),
            Err(ActiveHeadValidationError::ZeroClasses)
        ));
    }

    #[test]
    fn active_head_validate_rejects_class_count_mismatch() {
        let bad = ActiveHeadManifest {
            origin: ActiveOrigin::Default,
            runtime_head_id: default_runtime_head_id(),
            sha256: "aa".to_string(),
            labels_sha256: "bb".to_string(),
            n_classes: 3,
            labels: vec!["a".to_string(), "b".to_string()],
            activated_at: "2026-05-07T12:34:56Z".to_string(),
        };
        let err = bad.validate().unwrap_err();
        assert!(matches!(
            err,
            ActiveHeadValidationError::ClassCountMismatch {
                n_classes: 3,
                labels_len: 2
            }
        ));
    }

    #[test]
    fn active_head_validation_error_classifies_internal() {
        let err = ActiveHeadValidationError::DefaultRuntimeIdMismatch {
            got: head_id(),
            expected: default_runtime_head_id(),
        };
        assert_eq!(err.kind(), ErrorKind::Internal);
    }

    #[test]
    fn converter_type_round_trips_in_snake_case() {
        assert_eq!(
            serde_json::to_string(&ConverterType::Tfjs).unwrap(),
            "\"tfjs\""
        );
        let parsed: ConverterType = serde_json::from_str("\"tfjs\"").unwrap();
        assert_eq!(parsed, ConverterType::Tfjs);
    }

    #[test]
    fn converter_type_rejects_unknown_variant() {
        assert!(serde_json::from_str::<ConverterType>("\"onnx\"").is_err());
    }

    #[test]
    fn job_reference_workspace_id_returns_owner() {
        let r = JobReference::Workspace {
            workspace_id: ws_id(),
        };
        assert_eq!(r.workspace_id(), ws_id());
    }

    #[test]
    fn job_reference_round_trips() {
        let r = JobReference::Workspace {
            workspace_id: ws_id(),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"kind\":\"workspace\""));
        let parsed: JobReference = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, r);
    }

    /// Legacy dataset-tree/file variants must not parse (no path smuggling).
    #[test]
    fn job_reference_rejects_round_1_variants() {
        for stale in [
            r#"{"kind":"dataset_tree","workspace_id":"11111111-2222-4333-8444-555555555555","path":"audio"}"#,
            r#"{"kind":"dataset_file","workspace_id":"11111111-2222-4333-8444-555555555555","path":"audio/cat"}"#,
        ] {
            assert!(
                serde_json::from_str::<JobReference>(stale).is_err(),
                "legacy variant `{stale}` must not parse"
            );
        }
    }

    #[test]
    fn job_type_round_trips_in_snake_case() {
        for (jt, expected) in [
            (JobType::Train, "\"train\""),
            (JobType::Convert, "\"convert\""),
            (JobType::DatasetDelete, "\"dataset_delete\""),
            (JobType::ConverterDelete, "\"converter_delete\""),
            (JobType::TrainingLogsDelete, "\"training_logs_delete\""),
            (JobType::ConverterLogsDelete, "\"converter_logs_delete\""),
            (JobType::WorkspaceDelete, "\"workspace_delete\""),
        ] {
            assert_eq!(serde_json::to_string(&jt).unwrap(), expected);
            let parsed: JobType = serde_json::from_str(expected).unwrap();
            assert_eq!(parsed, jt);
        }
    }

    #[test]
    fn max_heads_per_workspace_is_three() {
        // Bumping cascades through rotation + schema tests (cap+1 fixtures).
        assert_eq!(MAX_HEADS_PER_WORKSPACE, 3);
    }
}
