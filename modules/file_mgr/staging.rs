//! Atomic delete-staging primitives for async asset and workspace deletes.
//! Steps under the per-workspace mutation mutex: (1) `write_tombstone` durable
//! intent marker; (2) caller bumps `workspace.json` revision BEFORE the byte
//! rename, so a crash can only conservatively stale heads, never leave a head
//! current after a dataset change; (3) `stage_payload` atomic-renames `target`
//! into staging; (4) caller publishes cache and unlocks; then off-mutex
//! (5) `drain_staged_payload` with budget until `Done`; (6)
//! `finalize_staged_delete`. Order is enforced by function-doc preconditions
//! and the `staging_publish_order` test, not type-state. A crash at any step
//! is resumable from the tombstone; boot recovery is the sole consumer that
//! walks existing `.tmp/delete-*.json`.

use crate::common::asset_path::AssetPath;
use crate::common::ids::{JobId, WorkspaceId};
use crate::file_mgr::error::{FileError, io_err, metadata_parse_err};
use crate::file_mgr::fs_atomic::put_atomic;
use crate::file_mgr::validate::fsync_dir;
use std::path::{Path, PathBuf};

/// Per-call entry budget for `drain_staged_payload`. Hardcoded (no TOML
/// override) because 0 would spin the inline drain loop forever.
pub const DEFAULT_DELETE_BATCH_ENTRIES: usize = 256;

/// Tombstone filename prefix (`<prefix><job_id>.json`) per delete kind.
pub const DATASET_TOMBSTONE_PREFIX: &str = "delete-assets-";
pub const CONVERTER_TOMBSTONE_PREFIX: &str = "delete-converters-";
pub const TRAINING_LOGS_TOMBSTONE_PREFIX: &str = "delete-training-logs-";
pub const CONVERTER_LOGS_TOMBSTONE_PREFIX: &str = "delete-converter-logs-";
pub const WORKSPACE_TOMBSTONE_PREFIX: &str = "delete-workspace-";

/// Boot recovery dispatches by prefix, so no prefix may `starts_with`-prefix
/// another (else the longer prefix's files misclassify); asserted below.
const TOMBSTONE_PREFIXES: [&str; 5] = [
    DATASET_TOMBSTONE_PREFIX,
    CONVERTER_TOMBSTONE_PREFIX,
    TRAINING_LOGS_TOMBSTONE_PREFIX,
    CONVERTER_LOGS_TOMBSTONE_PREFIX,
    WORKSPACE_TOMBSTONE_PREFIX,
];

const fn bytes_is_prefix_of(prefix: &[u8], whole: &[u8]) -> bool {
    if prefix.len() > whole.len() {
        return false;
    }
    let mut i = 0;
    while i < prefix.len() {
        if prefix[i] != whole[i] {
            return false;
        }
        i += 1;
    }
    true
}

const _: () = {
    let n = TOMBSTONE_PREFIXES.len();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n {
            let a = TOMBSTONE_PREFIXES[i].as_bytes();
            let b = TOMBSTONE_PREFIXES[j].as_bytes();
            assert!(
                !bytes_is_prefix_of(a, b),
                "tombstone prefix collision: one prefix is a starts-with prefix of another \
                 (see TOMBSTONE_PREFIXES; reorder + rename so no pair overlaps)",
            );
            assert!(
                !bytes_is_prefix_of(b, a),
                "tombstone prefix collision: one prefix is a starts-with prefix of another \
                 (see TOMBSTONE_PREFIXES; reorder + rename so no pair overlaps)",
            );
            j += 1;
        }
        i += 1;
    }
};

/// Name of the staged payload (file or dir tree) inside a stage dir;
/// `drain_staged_payload` removes whatever shape it finds.
pub const STAGED_PAYLOAD_NAME: &str = "payload";

/// Tombstone JSON written before staging a delete payload; boot recovery
/// deserializes these to resume an interrupted delete. The variant is also
/// encoded in the filename prefix so a boot scan dispatches without parsing.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeleteTombstone {
    /// Dataset file or subtree delete under `datasets/`.
    Dataset {
        job_id: JobId,
        workspace_id: WorkspaceId,
        /// `None` = whole-tree wipe; `serde(default)` keeps pre-whole-tree
        /// tombstones parsing.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<AssetPath>,
        /// Revision already bumped on disk before this tombstone was written.
        workspace_revision_id: u64,
        created_at: String,
    },
    /// Converter file or subtree delete under `converters/`.
    Converter {
        job_id: JobId,
        workspace_id: WorkspaceId,
        /// `None` = whole-tree wipe.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<AssetPath>,
        workspace_revision_id: u64,
        created_at: String,
    },
    /// Training-logs delete under `training_logs/`. Logs aren't tracked by
    /// `workspace.json.workspace_revision`, hence no `workspace_revision_id`.
    TrainingLogs {
        job_id: JobId,
        workspace_id: WorkspaceId,
        /// `None` = whole-tree wipe.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<AssetPath>,
        created_at: String,
    },
    /// Converter-logs delete under `converter_logs/`.
    ConverterLogs {
        job_id: JobId,
        workspace_id: WorkspaceId,
        /// `None` = whole-tree wipe.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        path: Option<AssetPath>,
        created_at: String,
    },
    /// Whole-workspace delete (no `path`).
    Workspace {
        job_id: JobId,
        workspace_id: WorkspaceId,
        created_at: String,
    },
}

impl DeleteTombstone {
    pub fn job_id(&self) -> JobId {
        match self {
            DeleteTombstone::Dataset { job_id, .. }
            | DeleteTombstone::Converter { job_id, .. }
            | DeleteTombstone::TrainingLogs { job_id, .. }
            | DeleteTombstone::ConverterLogs { job_id, .. }
            | DeleteTombstone::Workspace { job_id, .. } => *job_id,
        }
    }

    pub fn workspace_id(&self) -> WorkspaceId {
        match self {
            DeleteTombstone::Dataset { workspace_id, .. }
            | DeleteTombstone::Converter { workspace_id, .. }
            | DeleteTombstone::TrainingLogs { workspace_id, .. }
            | DeleteTombstone::ConverterLogs { workspace_id, .. }
            | DeleteTombstone::Workspace { workspace_id, .. } => *workspace_id,
        }
    }

    fn prefix(&self) -> &'static str {
        match self {
            DeleteTombstone::Dataset { .. } => DATASET_TOMBSTONE_PREFIX,
            DeleteTombstone::Converter { .. } => CONVERTER_TOMBSTONE_PREFIX,
            DeleteTombstone::TrainingLogs { .. } => TRAINING_LOGS_TOMBSTONE_PREFIX,
            DeleteTombstone::ConverterLogs { .. } => CONVERTER_LOGS_TOMBSTONE_PREFIX,
            DeleteTombstone::Workspace { .. } => WORKSPACE_TOMBSTONE_PREFIX,
        }
    }

    /// Tombstone filename `<prefix><job_id>.json`.
    pub fn filename(&self) -> String {
        format!("{}{}.json", self.prefix(), self.job_id())
    }

    /// Stage-directory name holding the renamed payload (`filename` minus `.json`).
    pub fn stage_dir_name(&self) -> String {
        format!("{}{}", self.prefix(), self.job_id())
    }
}

/// Resolved filesystem paths for one in-flight async delete, built from a
/// `DeleteTombstone` and the staging-dir parent (workspace `.tmp/` for
/// dataset/converter/log-tree deletes; root `.tmp/` for whole-workspace).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedDelete {
    pub tombstone: PathBuf,
    pub stage_dir: PathBuf,
    pub payload: PathBuf,
}

impl StagedDelete {
    pub fn for_tombstone(staging_dir: &Path, tombstone: &DeleteTombstone) -> Self {
        let tombstone_path = staging_dir.join(tombstone.filename());
        let stage_dir = staging_dir.join(tombstone.stage_dir_name());
        let payload = stage_dir.join(STAGED_PAYLOAD_NAME);
        Self {
            tombstone: tombstone_path,
            stage_dir,
            payload,
        }
    }
}

/// Step 1: atomically write a delete tombstone under `staging_dir` (via
/// `put_atomic`: tempfile, fsync, rename, fsync dir). Returns the resolved
/// `StagedDelete` for `stage_payload`.
pub fn write_tombstone(
    staging_dir: &Path,
    tombstone: &DeleteTombstone,
) -> Result<StagedDelete, FileError> {
    std::fs::create_dir_all(staging_dir).map_err(|e| io_err(staging_dir.display(), e))?;
    let staged = StagedDelete::for_tombstone(staging_dir, tombstone);
    let bytes = serde_json::to_vec(tombstone)?;
    put_atomic(&staged.tombstone, &bytes)?;
    Ok(staged)
}

/// Step 3: atomically rename `target` (file or directory) into
/// `staged.payload`.
///
/// Preconditions: `write_tombstone` ran; caller advanced+synced any revision
/// counter; `target` exists, staging dir exists, payload path does not. Fsyncs
/// the dirents it touches (stage_dir parent, old parent, stage_dir) but NOT
/// target file data, so target blocks must already be durable: a crash before
/// they flush leaves staged dirents pointing at unfsynced inodes -- in-tree
/// producers always feed via the atomic-put helper, but an operator drop under
/// `<workspace>/<tree>/` would expose the gap. On failure the staged path's
/// existence is undefined; callers treat it as in-progress for boot recovery.
pub fn stage_payload(target: &Path, staged: &StagedDelete) -> Result<(), FileError> {
    let old_parent = target.parent().ok_or_else(|| {
        io_err(
            target.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "stage_payload: target has no parent directory",
            ),
        )
    })?;
    std::fs::create_dir_all(&staged.stage_dir)
        .map_err(|e| io_err(staged.stage_dir.display(), e))?;
    // Make stage_dir's dirent durable BEFORE the rename makes its child the
    // on-disk source of truth, else a crash could orphan the payload.
    if let Some(staging_parent) = staged.stage_dir.parent() {
        fsync_dir(staging_parent).map_err(|e| io_err(staging_parent.display(), e))?;
    }
    std::fs::rename(target, &staged.payload).map_err(|e| io_err(target.display(), e))?;
    // The old parent's unlink must reach stable storage before we ack.
    fsync_dir(old_parent).map_err(|e| io_err(old_parent.display(), e))?;
    fsync_dir(&staged.stage_dir).map_err(|e| io_err(staged.stage_dir.display(), e))?;
    Ok(())
}

/// Outcome of one `drain_staged_payload` call.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DrainResult {
    /// Payload fully drained: a dir payload leaves its now-empty root for
    /// `finalize_staged_delete`, while a non-dir payload is already unlinked.
    Done,
    /// Budget exhausted before the payload drained; call again to resume.
    More,
}

/// Step 5: remove up to `budget` filesystem entries from `staged.payload`,
/// returning `Done` once empty.
///
/// Idempotent: a missing payload returns `Done`, so crash+retry converges.
/// Symlinks are unlinked, not followed (operator-tamper defense). Walk is
/// leaf-first depth-first so each `remove_dir` sees an empty dir; each removed
/// entry costs one budget unit, descending does not.
pub fn drain_staged_payload(
    staged: &StagedDelete,
    budget: usize,
) -> Result<DrainResult, FileError> {
    // `symlink_metadata`, not `Path::exists`: detect presence+type without
    // following, so a dangling symlink at the payload is not seen as absence.
    let metadata = match std::fs::symlink_metadata(&staged.payload) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(DrainResult::Done);
        }
        Err(e) => {
            return Err(io_err(staged.payload.display(), e));
        }
    };
    if !metadata.is_dir() {
        std::fs::remove_file(&staged.payload).map_err(|e| io_err(staged.payload.display(), e))?;
        // fsync stage_dir so the unlink is durable before Done: finalize fsyncs
        // only the staging root, so a power loss between Done and finalize's
        // stage_dir removal could otherwise resurrect the payload.
        fsync_dir(&staged.stage_dir).map_err(|e| io_err(staged.stage_dir.display(), e))?;
        return Ok(DrainResult::Done);
    }
    let mut removed = 0usize;
    let outcome = drain_dir(&staged.payload, budget, &mut removed)?;
    // On Done, fsync the payload root: drain_dir fsyncs each level it fully
    // drains but never the root, and finalize removes it (after the tombstone
    // unlink) without fsyncing it -- else a crash between Done and finalize
    // could roll back the root's unlinks while the tombstone is gone, leaving
    // an unresumable orphan.
    if matches!(outcome, DrainResult::Done) {
        fsync_dir(&staged.payload).map_err(|e| io_err(staged.payload.display(), e))?;
    }
    Ok(outcome)
}

/// Recursive directory-drain helper; preserves the root dir on `Done`.
fn drain_dir(dir: &Path, budget: usize, removed: &mut usize) -> Result<DrainResult, FileError> {
    let entries = std::fs::read_dir(dir).map_err(|e| io_err(dir.display(), e))?;
    for entry_result in entries {
        if *removed >= budget {
            return Ok(DrainResult::More);
        }
        let entry = entry_result.map_err(|e| io_err(dir.display(), e))?;
        let path = entry.path();
        // `entry.file_type()` reuses the readdir dirent type and does NOT follow
        // symlinks: a symlinked dir reports non-dir and takes the unlink branch
        // rather than recursing across the link.
        let ft = entry.file_type().map_err(|e| io_err(path.display(), e))?;
        if ft.is_dir() {
            match drain_dir(&path, budget, removed)? {
                DrainResult::More => return Ok(DrainResult::More),
                DrainResult::Done => {
                    if *removed >= budget {
                        return Ok(DrainResult::More);
                    }
                    std::fs::remove_dir(&path).map_err(|e| io_err(path.display(), e))?;
                    *removed += 1;
                }
            }
        } else {
            // Regular file OR symlink: `remove_file` unlinks without following.
            std::fs::remove_file(&path).map_err(|e| io_err(path.display(), e))?;
            *removed += 1;
        }
    }
    // Fully drained: fsync this dirent so its unlinks/rmdirs are durable before
    // the parent rmdirs this now-empty dir, else a power loss could roll them
    // back, leaving a non-empty dir whose rmdir fails post-remount.
    fsync_dir(dir).map_err(|e| io_err(dir.display(), e))?;
    Ok(DrainResult::Done)
}

/// Step 6: remove the (now empty) payload, stage dir, and tombstone, then fsync
/// the staging parent. Idempotent: missing entries are treated as already gone
/// so a crash mid-finalize converges on retry.
///
/// Precondition: `drain_staged_payload` returned `Done`. A non-empty payload
/// surfaces as `FileError::Io` (`ENOTEMPTY`).
pub fn finalize_staged_delete(staged: &StagedDelete) -> Result<(), FileError> {
    // Remove the payload dir if present; a single-file payload is gone post-drain.
    match std::fs::symlink_metadata(&staged.payload) {
        Ok(md) if md.is_dir() => {
            std::fs::remove_dir(&staged.payload)
                .map_err(|e| io_err(staged.payload.display(), e))?;
        }
        Ok(_) => {
            // Stale file or symlink -- unlink for safety.
            std::fs::remove_file(&staged.payload)
                .map_err(|e| io_err(staged.payload.display(), e))?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(io_err(staged.payload.display(), e));
        }
    }
    match std::fs::remove_dir(&staged.stage_dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(io_err(staged.stage_dir.display(), e));
        }
    }
    match std::fs::remove_file(&staged.tombstone) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(io_err(staged.tombstone.display(), e));
        }
    }
    // fsync the staging parent (shared by tombstone + stage_dir) so the three
    // removals are durable.
    if let Some(staging_dir) = staged.tombstone.parent() {
        // no-op if a parallel cleaner already removed it
        if staging_dir.exists() {
            fsync_dir(staging_dir).map_err(|e| io_err(staging_dir.display(), e))?;
        }
    }
    Ok(())
}

/// Read a tombstone JSON from disk. A malformed tombstone surfaces as
/// `FileError::MetadataParse`; the recovery caller decides delete-and-proceed
/// vs abort.
pub fn read_tombstone(path: &Path) -> Result<DeleteTombstone, FileError> {
    let bytes =
        crate::file_mgr::schema::read_capped(path, crate::file_mgr::schema::MAX_TOMBSTONE_BYTES)?;
    serde_json::from_slice(&bytes).map_err(|source| metadata_parse_err(path.display(), source))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixtures write payload bytes directly

    use super::*;
    use std::fs;
    use std::io::Write;

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }

    fn job_id() -> JobId {
        JobId::parse("22222222-3333-4444-8555-666666666666").unwrap()
    }

    fn dataset_tombstone() -> DeleteTombstone {
        DeleteTombstone::Dataset {
            job_id: job_id(),
            workspace_id: ws_id(),
            path: Some(AssetPath::parse("audio_dataset/cat").unwrap()),
            workspace_revision_id: 6,
            created_at: "2026-05-07T13:00:00Z".to_string(),
        }
    }

    fn converter_tombstone() -> DeleteTombstone {
        DeleteTombstone::Converter {
            job_id: job_id(),
            workspace_id: ws_id(),
            path: Some(AssetPath::parse("tfjs/model.json").unwrap()),
            workspace_revision_id: 7,
            created_at: "2026-05-07T13:00:00Z".to_string(),
        }
    }

    fn workspace_tombstone() -> DeleteTombstone {
        DeleteTombstone::Workspace {
            job_id: job_id(),
            workspace_id: ws_id(),
            created_at: "2026-05-07T13:00:00Z".to_string(),
        }
    }

    fn training_logs_tombstone() -> DeleteTombstone {
        DeleteTombstone::TrainingLogs {
            job_id: job_id(),
            workspace_id: ws_id(),
            path: Some(AssetPath::parse("11111111-2222-4333-8444-555555555555.jsonl").unwrap()),
            created_at: "2026-05-07T13:00:00Z".to_string(),
        }
    }

    fn converter_logs_tombstone() -> DeleteTombstone {
        DeleteTombstone::ConverterLogs {
            job_id: job_id(),
            workspace_id: ws_id(),
            path: None,
            created_at: "2026-05-07T13:00:00Z".to_string(),
        }
    }

    fn make_target_dir(root: &Path, layout: &[(&str, &[u8])]) -> PathBuf {
        let target = root.join("target");
        for (rel, bytes) in layout {
            let full = target.join(rel);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            let mut f = fs::File::create(&full).unwrap();
            f.write_all(bytes).unwrap();
        }
        target
    }

    fn count_filesystem_entries(root: &Path) -> usize {
        if !root.exists() {
            return 0;
        }
        let mut total = 0usize;
        for entry in fs::read_dir(root).unwrap() {
            let entry = entry.unwrap();
            total += 1;
            if entry.file_type().unwrap().is_dir() {
                total += count_filesystem_entries(&entry.path());
            }
        }
        total
    }

    #[test]
    fn dataset_tombstone_round_trips_and_names_files() {
        let t = dataset_tombstone();
        let json = serde_json::to_string(&t).unwrap();
        let parsed: DeleteTombstone = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
        assert!(t.filename().starts_with(DATASET_TOMBSTONE_PREFIX));
        assert!(t.filename().ends_with(".json"));
        assert_eq!(t.filename(), format!("{}.json", t.stage_dir_name()));
    }

    #[test]
    fn workspace_tombstone_round_trips_and_names_files() {
        let t = workspace_tombstone();
        let json = serde_json::to_string(&t).unwrap();
        let parsed: DeleteTombstone = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
        assert!(t.filename().starts_with(WORKSPACE_TOMBSTONE_PREFIX));
    }

    #[test]
    fn converter_tombstone_round_trips_and_names_files() {
        let t = converter_tombstone();
        let json = serde_json::to_string(&t).unwrap();
        let parsed: DeleteTombstone = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
        assert!(json.contains("\"kind\":\"converter\""));
        assert!(t.filename().starts_with(CONVERTER_TOMBSTONE_PREFIX));
        assert!(!t.filename().starts_with(DATASET_TOMBSTONE_PREFIX));
        assert_eq!(t.filename(), format!("{}.json", t.stage_dir_name()));
    }

    #[test]
    fn training_logs_tombstone_round_trips_and_names_files() {
        let t = training_logs_tombstone();
        let json = serde_json::to_string(&t).unwrap();
        let parsed: DeleteTombstone = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
        assert!(json.contains("\"kind\":\"training_logs\""));
        assert!(
            !json.contains("workspace_revision_id"),
            "training-logs tombstone must not carry workspace_revision_id; got {json}",
        );
        assert!(t.filename().starts_with(TRAINING_LOGS_TOMBSTONE_PREFIX));
        assert!(!t.filename().starts_with(CONVERTER_LOGS_TOMBSTONE_PREFIX));
        assert!(!t.filename().starts_with(DATASET_TOMBSTONE_PREFIX));
        assert!(!t.filename().starts_with(CONVERTER_TOMBSTONE_PREFIX));
        assert_eq!(t.filename(), format!("{}.json", t.stage_dir_name()));
    }

    #[test]
    fn converter_logs_tombstone_round_trips_and_names_files() {
        let t = converter_logs_tombstone();
        let json = serde_json::to_string(&t).unwrap();
        let parsed: DeleteTombstone = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
        assert!(json.contains("\"kind\":\"converter_logs\""));
        assert!(
            !json.contains("workspace_revision_id"),
            "converter-logs tombstone must not carry workspace_revision_id; got {json}",
        );
        assert!(t.filename().starts_with(CONVERTER_LOGS_TOMBSTONE_PREFIX));
        // Guards the near-collision: `delete-converter-logs-` vs `delete-converters-`.
        assert!(!t.filename().starts_with(CONVERTER_TOMBSTONE_PREFIX));
        assert!(!t.filename().starts_with(TRAINING_LOGS_TOMBSTONE_PREFIX));
        assert!(!t.filename().starts_with(DATASET_TOMBSTONE_PREFIX));
        assert_eq!(t.filename(), format!("{}.json", t.stage_dir_name()));
    }

    #[test]
    fn tombstone_rejects_unknown_fields() {
        let bad = r#"{
            "kind": "dataset",
            "job_id": "22222222-3333-4444-8555-666666666666",
            "workspace_id": "11111111-2222-4333-8444-555555555555",
            "path": "audio_dataset",
            "workspace_revision_id": 1,
            "created_at": "2026-05-07T13:00:00Z",
            "extra": "no"
        }"#;
        assert!(serde_json::from_str::<DeleteTombstone>(bad).is_err());
    }

    #[test]
    fn write_tombstone_creates_staging_dir_and_atomic_file() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        assert!(!staging.exists());
        let t = dataset_tombstone();
        let staged = write_tombstone(&staging, &t).unwrap();
        assert!(staging.is_dir(), "staging dir was created");
        assert!(staged.tombstone.is_file(), "tombstone written");
        let recovered = read_tombstone(&staged.tombstone).unwrap();
        assert_eq!(recovered, t);
    }

    #[test]
    fn write_tombstone_is_idempotent_on_existing_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        fs::create_dir_all(&staging).unwrap();
        let t = dataset_tombstone();
        // Re-writing the same tombstone (boot rewrites mid-stage) is allowed.
        write_tombstone(&staging, &t).unwrap();
        write_tombstone(&staging, &t).unwrap();
        let staged = StagedDelete::for_tombstone(&staging, &t);
        assert!(staged.tombstone.exists());
    }

    #[test]
    fn stage_payload_renames_file_target_into_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("dataset.wav");
        fs::write(&target, b"audio bytes").unwrap();
        let staging = tmp.path().join(".tmp");
        let t = dataset_tombstone();
        let staged = write_tombstone(&staging, &t).unwrap();
        stage_payload(&target, &staged).unwrap();
        assert!(!target.exists(), "target removed from original");
        assert!(staged.payload.is_file(), "payload rename in place");
        assert_eq!(fs::read(&staged.payload).unwrap(), b"audio bytes");
    }

    #[test]
    fn stage_payload_renames_directory_target() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(
            tmp.path(),
            &[
                ("a.wav", b"a"),
                ("nested/b.wav", b"b"),
                ("nested/c.wav", b"c"),
            ],
        );
        let staging = tmp.path().join(".tmp");
        let t = dataset_tombstone();
        let staged = write_tombstone(&staging, &t).unwrap();
        stage_payload(&target, &staged).unwrap();
        assert!(!target.exists());
        assert!(staged.payload.is_dir());
        assert!(staged.payload.join("a.wav").is_file());
        assert!(staged.payload.join("nested/c.wav").is_file());
    }

    #[test]
    fn drain_staged_payload_done_on_missing_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        let t = dataset_tombstone();
        let staged = write_tombstone(&staging, &t).unwrap();
        let res = drain_staged_payload(&staged, 100).unwrap();
        assert_eq!(res, DrainResult::Done);
    }

    #[test]
    fn drain_staged_payload_removes_single_file() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("dataset.wav");
        fs::write(&target, b"audio").unwrap();
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        let res = drain_staged_payload(&staged, 1).unwrap();
        assert_eq!(res, DrainResult::Done);
        assert!(!staged.payload.exists());
    }

    #[test]
    fn drain_staged_payload_recursively_empties_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(
            tmp.path(),
            &[
                ("a.wav", b"a"),
                ("nested/b.wav", b"b"),
                ("nested/deeper/c.wav", b"c"),
                ("nested/deeper/d.wav", b"d"),
            ],
        );
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        let res = drain_staged_payload(&staged, 1024).unwrap();
        assert_eq!(res, DrainResult::Done);
        // Payload root remains as an empty dir for finalize to remove.
        assert!(staged.payload.is_dir());
        assert_eq!(count_filesystem_entries(&staged.payload), 0);
    }

    #[test]
    fn drain_staged_payload_respects_budget_and_resumes() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(
            tmp.path(),
            &[
                ("a", b"1"),
                ("b", b"2"),
                ("c", b"3"),
                ("d", b"4"),
                ("e", b"5"),
            ],
        );
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        let mut iterations = 0usize;
        loop {
            iterations += 1;
            let res = drain_staged_payload(&staged, 2).unwrap();
            if res == DrainResult::Done {
                break;
            }
            assert!(iterations <= 5, "drain failed to converge");
        }
        assert!(staged.payload.exists());
        assert_eq!(count_filesystem_entries(&staged.payload), 0);
    }

    /// Operator-tamper defense: a symlink in the payload is unlinked, never
    /// followed to delete its target outside the tree.
    #[cfg(unix)]
    #[test]
    fn drain_staged_payload_unlinks_symlinks_without_following() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let outside = tmp.path().join("outside.txt");
        fs::write(&outside, b"do not delete").unwrap();
        let target = make_target_dir(tmp.path(), &[("real.wav", b"real")]);
        symlink(&outside, target.join("link")).unwrap();
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        let res = drain_staged_payload(&staged, 1024).unwrap();
        assert_eq!(res, DrainResult::Done);
        assert!(outside.is_file());
        assert_eq!(fs::read(&outside).unwrap(), b"do not delete");
    }

    #[test]
    fn finalize_staged_delete_clears_tombstone_and_stage_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("dataset.wav");
        fs::write(&target, b"audio").unwrap();
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        drain_staged_payload(&staged, 1024).unwrap();
        finalize_staged_delete(&staged).unwrap();
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
        assert!(!staged.payload.exists());
        // Shared staging dir survives finalize for future deletes.
        assert!(staging.is_dir());
    }

    #[test]
    fn finalize_staged_delete_is_idempotent_on_missing_pieces() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        fs::create_dir_all(&staging).unwrap();
        let staged = StagedDelete::for_tombstone(&staging, &dataset_tombstone());
        finalize_staged_delete(&staged).unwrap();
    }

    /// `finalize` must refuse a non-empty payload, catching a skipped-drain caller.
    #[test]
    fn finalize_staged_delete_rejects_nonempty_payload() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(tmp.path(), &[("a.wav", b"a")]);
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();
        // Skip the drain: finalize must fail on the non-empty payload dir.
        let res = finalize_staged_delete(&staged);
        assert!(
            res.is_err(),
            "non-empty payload must not be silently removed"
        );
    }

    /// Canonical guard that the four steps leave a consistent filesystem at
    /// every boundary; reordering breaks a boot-recovery invariant.
    #[test]
    fn staging_publish_order_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(tmp.path(), &[("dataset.wav", b"a"), ("nested/b.wav", b"b")]);
        let staging = tmp.path().join(".tmp");

        let t = dataset_tombstone();
        let staged = write_tombstone(&staging, &t).unwrap();
        assert!(staged.tombstone.is_file(), "step 1: tombstone present");
        assert!(target.exists(), "step 1: target unchanged");
        assert!(!staged.payload.exists(), "step 1: payload not yet staged");

        stage_payload(&target, &staged).unwrap();
        assert!(
            staged.tombstone.is_file(),
            "step 3: tombstone still present"
        );
        assert!(!target.exists(), "step 3: target gone from original");
        assert!(staged.payload.is_dir(), "step 3: payload renamed in place");

        let res = drain_staged_payload(&staged, 1024).unwrap();
        assert_eq!(res, DrainResult::Done);
        assert!(
            staged.payload.is_dir(),
            "step 5: empty payload root remains"
        );
        assert_eq!(count_filesystem_entries(&staged.payload), 0);

        finalize_staged_delete(&staged).unwrap();
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
        assert!(staging.is_dir(), "staging dir survives for reuse");
    }

    /// Crash-resume: drain+finalize re-derive paths from a tombstone alone and
    /// converge, simulating a fresh boot after stage.
    #[test]
    fn drain_finalize_resumes_from_post_stage_state() {
        let tmp = tempfile::tempdir().unwrap();
        let target = make_target_dir(
            tmp.path(),
            &[
                ("a.wav", b"a"),
                ("nested/b.wav", b"b"),
                ("nested/deeper/c.wav", b"c"),
            ],
        );
        let staging = tmp.path().join(".tmp");
        let staged = write_tombstone(&staging, &dataset_tombstone()).unwrap();
        stage_payload(&target, &staged).unwrap();

        let recovered_tombstone = read_tombstone(&staged.tombstone).unwrap();
        let recovered_staged = StagedDelete::for_tombstone(&staging, &recovered_tombstone);
        assert_eq!(recovered_staged, staged);

        loop {
            let res = drain_staged_payload(&recovered_staged, 1).unwrap();
            if res == DrainResult::Done {
                break;
            }
        }
        finalize_staged_delete(&recovered_staged).unwrap();
        assert!(!staged.tombstone.exists());
    }

    /// Each filename prefix classifies its tombstone without collision (boot
    /// recovery dispatches by prefix before opening the file).
    #[test]
    fn tombstone_filename_prefix_dispatches_kind() {
        let dataset = dataset_tombstone();
        let converter = converter_tombstone();
        let training_logs = training_logs_tombstone();
        let converter_logs = converter_logs_tombstone();
        let workspace = workspace_tombstone();
        let all_prefixes = [
            DATASET_TOMBSTONE_PREFIX,
            CONVERTER_TOMBSTONE_PREFIX,
            TRAINING_LOGS_TOMBSTONE_PREFIX,
            CONVERTER_LOGS_TOMBSTONE_PREFIX,
            WORKSPACE_TOMBSTONE_PREFIX,
        ];
        for (variant, own) in [
            (&dataset, DATASET_TOMBSTONE_PREFIX),
            (&converter, CONVERTER_TOMBSTONE_PREFIX),
            (&training_logs, TRAINING_LOGS_TOMBSTONE_PREFIX),
            (&converter_logs, CONVERTER_LOGS_TOMBSTONE_PREFIX),
            (&workspace, WORKSPACE_TOMBSTONE_PREFIX),
        ] {
            let name = variant.filename();
            assert!(
                name.starts_with(own),
                "{variant:?} must use its own prefix `{own}`; got {name}"
            );
            for other in all_prefixes {
                if other == own {
                    continue;
                }
                assert!(
                    !name.starts_with(other),
                    "{variant:?} filename `{name}` must not collide with `{other}` -- \
                     boot recovery dispatches by prefix",
                );
            }
        }
    }

    #[test]
    fn default_delete_batch_entries_matches_storage_table() {
        assert_eq!(DEFAULT_DELETE_BATCH_ENTRIES, 256);
    }
}
