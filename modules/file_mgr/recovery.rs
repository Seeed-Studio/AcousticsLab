//! Boot-time recovery sweeps. Run once on daemon start, AFTER
//! `ensure_root_layout` and BEFORE any API/inference goes live, in fixed
//! order root staging -> per-workspace -> active staging -> active head:
//! active head is last because activation may reference a head in a
//! workspace the root-staging step's tombstone deletes.
//!
//! [`recover_active_head`] returns the validated manifest rather than a
//! runtime `HotHead` (install left to the daemon's `HotHead::load`) so
//! `file_mgr` stays independent of `inference`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::common::ids::{HeadId, WorkspaceId};
use crate::common::workspace::{
    ActiveHeadManifest, HeadIndex, HeadRecord, MAX_HEADS_PER_WORKSPACE,
};
use crate::file_mgr::active_head_writer::{
    ActivationOriginInput, DefaultHeadSource, HeadInnerLoader, PendingActivation,
    publish_active_generation, stage_and_validate_activation, staging_path_for,
};
use crate::file_mgr::cache::WorkspaceCacheCell;
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::fs_atomic::put_atomic;
use crate::file_mgr::schema::{
    ACTIVE_HEAD_FILENAME, ACTIVE_LABELS_FILENAME, ActiveCurrentPointer, active_generation_dir,
    active_generations_dir, active_staging_dir, head_artifact_path, heads_dir, read_active_current,
    read_active_manifest, read_head_index, read_head_manifest, read_workspace_core,
    workspace_core_path, workspaces_dir, write_active_current, write_head_index,
    write_workspace_core,
};
use crate::file_mgr::staging::{
    CONVERTER_LOGS_TOMBSTONE_PREFIX, CONVERTER_TOMBSTONE_PREFIX, DATASET_TOMBSTONE_PREFIX,
    DEFAULT_DELETE_BATCH_ENTRIES, DeleteTombstone, DrainResult, StagedDelete,
    TRAINING_LOGS_TOMBSTONE_PREFIX, WORKSPACE_TOMBSTONE_PREFIX, drain_staged_payload,
    finalize_staged_delete, read_tombstone, stage_payload,
};
use crate::file_mgr::time_util::{now_rfc3339, parse_rfc3339_or_epoch};
use crate::file_mgr::validate::{fsync_dir, hex_lowercase};

/// 64 KiB streaming-hash chunk, matching the other `file_mgr` hash sites.
const STREAM_HASH_CHUNK: usize = 64 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum RecoveryError {
    #[error("file: {0}")]
    File(#[from] FileError),
}

impl crate::common::error::Categorized for RecoveryError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        crate::common::error::ErrorKind::Internal
    }
}

/// Outcome of [`recover_active_head`]; caller installs the resolved
/// manifest. For `PromotedPrevious`/`DefaultedFromBundle`, `current.json`
/// was already rewritten to point at the new generation.
#[derive(Debug)]
pub enum RecoveryActiveResult {
    /// Current generation passed verify; no on-disk pointer change.
    Current {
        activation_id: String,
        manifest: ActiveHeadManifest,
    },
    /// Current failed verify; the previous generation was promoted.
    PromotedPrevious {
        activation_id: String,
        manifest: ActiveHeadManifest,
    },
    /// No retained generation passed verify; bundled default activated.
    DefaultedFromBundle {
        activation_id: String,
        manifest: ActiveHeadManifest,
    },
    /// Bundled default missing or its activation failed; daemon boots
    /// without inference, caller surfaces `reason` as degraded.
    Unhealthy { reason: String },
}

/// Counters returned by [`recover_workspaces`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecoveryWorkspaceReport {
    /// Workspaces whose `workspace.json` parsed; incomplete creates count
    /// under [`Self::incomplete_creates_removed`] instead.
    pub workspaces_scanned: usize,
    /// `<head_id>.{mpk,json}` files dropped because the id was not in
    /// `heads.json.heads[]`.
    pub head_orphans_swept: usize,
    pub head_count_repaired: usize,
    pub dataset_tombstones_completed: usize,
    pub dataset_stage_orphans_swept: usize,
    pub converter_tombstones_completed: usize,
    pub converter_stage_orphans_swept: usize,
    pub training_logs_tombstones_completed: usize,
    pub training_logs_stage_orphans_swept: usize,
    pub converter_logs_tombstones_completed: usize,
    pub converter_logs_stage_orphans_swept: usize,
    /// Incomplete-create dirs (no `workspace.json`) removed.
    pub incomplete_creates_removed: usize,
    /// Workspace-level recovery `Err` (logged + continued); NOT in
    /// [`Self::workspaces_scanned`], distinct from the dirent-level
    /// [`Self::workspace_enumeration_failures`].
    pub workspace_recovery_failures: usize,
    /// Per-dirent `read_dir`/`file_type()` failures (logged + skipped);
    /// kept apart from [`Self::workspace_recovery_failures`] to distinguish
    /// "couldn't enumerate" from "swept Err".
    pub workspace_enumeration_failures: usize,
}

impl RecoveryWorkspaceReport {
    /// Fold per-workspace counters into the aggregate; `scanned` +
    /// `workspace_recovery_failures` are bumped by the orchestrator's
    /// Ok/Err arms, not here, else double-count.
    fn fold_per_workspace(&mut self, per: PerWorkspaceCounts) {
        // Exhaustive destructure (no `..`) so a new PerWorkspaceCounts field
        // compile-fails here rather than silently undercounting.
        let PerWorkspaceCounts {
            head_orphans_swept,
            head_count_repaired,
            dataset_tombstones_completed,
            dataset_stage_orphans_swept,
            converter_tombstones_completed,
            converter_stage_orphans_swept,
            training_logs_tombstones_completed,
            training_logs_stage_orphans_swept,
            converter_logs_tombstones_completed,
            converter_logs_stage_orphans_swept,
        } = per;
        self.head_orphans_swept += head_orphans_swept;
        self.head_count_repaired += head_count_repaired;
        self.dataset_tombstones_completed += dataset_tombstones_completed;
        self.dataset_stage_orphans_swept += dataset_stage_orphans_swept;
        self.converter_tombstones_completed += converter_tombstones_completed;
        self.converter_stage_orphans_swept += converter_stage_orphans_swept;
        self.training_logs_tombstones_completed += training_logs_tombstones_completed;
        self.training_logs_stage_orphans_swept += training_logs_stage_orphans_swept;
        self.converter_logs_tombstones_completed += converter_logs_tombstones_completed;
        self.converter_logs_stage_orphans_swept += converter_logs_stage_orphans_swept;
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecoveryRootReport {
    pub workspace_tombstones_completed: usize,
    pub workspace_stage_orphans_swept: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecoveryActiveStagingReport {
    /// Orphan staging dirs removed from `<root>/active/.tmp/`: published
    /// activations rename out, so anything left is a failed/crashed staging.
    pub active_staging_orphans_swept: usize,
}

#[derive(Debug)]
pub struct RecoveryReport {
    pub active: RecoveryActiveResult,
    pub workspaces: RecoveryWorkspaceReport,
    pub root_staging: RecoveryRootReport,
    pub active_staging: RecoveryActiveStagingReport,
}

/// Run every boot-recovery sweep in the module-header dependency order.
pub fn recover_all(
    root: &Path,
    default_head: Option<DefaultHeadSource<'_>>,
    caches: &dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>>,
    head_inner_loader: &HeadInnerLoader,
) -> Result<RecoveryReport, RecoveryError> {
    let root_staging = recover_root_staging(root, caches)?;
    let workspaces = recover_workspaces(root)?;
    let active_staging = recover_active_staging(root)?;
    let active = recover_active_head(root, default_head, head_inner_loader)?;
    Ok(RecoveryReport {
        active,
        workspaces,
        root_staging,
        active_staging,
    })
}

/// Verify the active-head generation and pick the runtime candidate:
/// read `current.json` (absent -> bundled default), verify the pointed
/// generation, else promote the newest passing previous generation, else
/// the bundled default.
///
/// `head_inner_loader` is only used by the bundled-default fallback; the
/// current/promote paths don't load (daemon re-loads via `HotHead::load`).
pub fn recover_active_head(
    root: &Path,
    default_head: Option<DefaultHeadSource<'_>>,
    head_inner_loader: &HeadInnerLoader,
) -> Result<RecoveryActiveResult, RecoveryError> {
    // One read distinguishes "missing" (-> default) from "corrupt"
    // (-> fallback) via NotFound, avoiding an exists()+read TOCTOU that
    // misreports a raced unlink as corrupt.
    let pointer = match read_active_current(root) {
        Ok(p) => p,
        Err(FileError::Io { ref source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            return activate_bundled_default(root, default_head, head_inner_loader);
        }
        Err(e) => {
            // Corrupt pointer: treat current as missing.
            tracing::warn!(
                target: "file_mgr::recovery",
                err = %e,
                "active current.json read/parse failed; falling back",
            );
            return promote_or_default(root, None, default_head, head_inner_loader);
        }
    };

    match verify_generation(root, &pointer.activation_id)? {
        VerifyOutcome::Ok { manifest } => Ok(RecoveryActiveResult::Current {
            activation_id: pointer.activation_id,
            manifest,
        }),
        VerifyOutcome::Failed => promote_or_default(
            root,
            Some(&pointer.activation_id),
            default_head,
            head_inner_loader,
        ),
    }
}

/// `Failed` is unstructured; the diagnostic is logged at the failure site.
enum VerifyOutcome {
    Ok { manifest: ActiveHeadManifest },
    Failed,
}

/// Verify one generation in place: read + validate manifest,
/// streaming-hash `head.mpk`, then `labels.txt` (regenerating from
/// `manifest.labels[]` on mismatch). On `Ok` both files match the
/// manifest's recorded hashes.
fn verify_generation(root: &Path, activation_id: &str) -> Result<VerifyOutcome, RecoveryError> {
    let manifest = match read_active_manifest(root, activation_id) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %activation_id,
                err = %e,
                "active manifest read/parse failed",
            );
            return Ok(VerifyOutcome::Failed);
        }
    };
    if let Err(e) = manifest.validate() {
        tracing::warn!(
            target: "file_mgr::recovery",
            activation_id = %activation_id,
            err = %e,
            "active manifest validation failed",
        );
        return Ok(VerifyOutcome::Failed);
    }
    let gen_dir = active_generation_dir(root, activation_id);
    let head_mpk = gen_dir.join(ACTIVE_HEAD_FILENAME);
    match sha256_stream(&head_mpk) {
        Ok(observed) if observed == manifest.sha256 => {}
        Ok(observed) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %activation_id,
                expected = %manifest.sha256,
                observed = %observed,
                "active head.mpk hash mismatch",
            );
            return Ok(VerifyOutcome::Failed);
        }
        Err(e) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %activation_id,
                path = %head_mpk.display(),
                err = %e,
                "active head.mpk read failed",
            );
            return Ok(VerifyOutcome::Failed);
        }
    }
    let labels_path = gen_dir.join(ACTIVE_LABELS_FILENAME);
    let labels_ok = match sha256_stream(&labels_path) {
        Ok(observed) => observed == manifest.labels_sha256,
        Err(e) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %activation_id,
                path = %labels_path.display(),
                err = %e,
                "active labels.txt read failed; regenerating from manifest",
            );
            false
        }
    };
    if !labels_ok {
        if let Err(e) = regenerate_labels_from_manifest(&labels_path, &manifest) {
            // Regen WRITE failure (ENOSPC/EROFS/EIO): skip this candidate (a
            // later one or the bundled default may still verify), not fail boot.
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %activation_id,
                path = %labels_path.display(),
                err = %e,
                "active labels.txt regeneration write failed",
            );
            return Ok(VerifyOutcome::Failed);
        }
        // Post-regen recheck: the regenerator emits only canonical
        // `labels.join("\n")`, so a `labels_sha256` recorded over
        // non-canonical bytes (trailing newline / CRLFs) can never match and
        // must not be promoted; without this each boot regens and loops forever.
        match sha256_stream(&labels_path) {
            Ok(observed) if observed == manifest.labels_sha256 => {
                tracing::info!(
                    target: "file_mgr::recovery",
                    activation_id = %activation_id,
                    "regenerated active labels.txt from manifest.labels[]",
                );
            }
            Ok(observed) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    activation_id = %activation_id,
                    expected = %manifest.labels_sha256,
                    observed = %observed,
                    "regenerated active labels.txt does not match manifest labels_sha256",
                );
                return Ok(VerifyOutcome::Failed);
            }
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    activation_id = %activation_id,
                    path = %labels_path.display(),
                    err = %e,
                    "regenerated active labels.txt unreadable post-regen",
                );
                return Ok(VerifyOutcome::Failed);
            }
        }
    }
    Ok(VerifyOutcome::Ok { manifest })
}

/// Promote the most recently published valid generation other
/// than `current_id`, falling back to the bundled default.
fn promote_or_default(
    root: &Path,
    current_id: Option<&str>,
    default_head: Option<DefaultHeadSource<'_>>,
    head_inner_loader: &HeadInnerLoader,
) -> Result<RecoveryActiveResult, RecoveryError> {
    let generations_root = active_generations_dir(root);
    if !generations_root.is_dir() {
        return activate_bundled_default(root, default_head, head_inner_loader);
    }
    // Newest-first by `activated_at` PARSED to instants (lexical is
    // non-monotonic against `now_rfc3339`'s variable-width fractions and could
    // promote an older generation); tiebreak mtime then dir-name,
    // empty/unparseable -> epoch (oldest), un-stat-able dirs skipped.
    let mut candidates: Vec<(String, std::time::SystemTime, String)> = Vec::new();
    let entries =
        std::fs::read_dir(&generations_root).map_err(|e| io_err(generations_root.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| io_err(generations_root.display(), e))?;
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if Some(name.as_str()) == current_id {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %entry.path().display(),
                    err = %e,
                    "skipping unreadable generation entry during fallback",
                );
                continue;
            }
        };
        if !metadata.is_dir() {
            continue;
        }
        // Empty on unparseable manifest so mtime breaks the tie.
        let activated_at = read_active_manifest(root, &name)
            .map(|m| m.activated_at)
            .unwrap_or_default();
        let mtime = metadata
            .modified()
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        candidates.push((activated_at, mtime, name));
    }
    candidates.sort_by(|a, b| {
        parse_rfc3339_or_epoch(&b.0)
            .cmp(&parse_rfc3339_or_epoch(&a.0))
            .then(b.1.cmp(&a.1))
            .then(a.2.cmp(&b.2))
    });
    for (_, _, candidate_id) in candidates {
        if let VerifyOutcome::Ok { manifest } = verify_generation(root, &candidate_id)? {
            // write_active_current -> put_atomic already fsynced active/.
            write_active_current(
                root,
                &ActiveCurrentPointer {
                    activation_id: candidate_id.clone(),
                },
            )?;
            tracing::warn!(
                target: "file_mgr::recovery",
                activation_id = %candidate_id,
                "promoted previous active generation; current.json rewritten",
            );
            return Ok(RecoveryActiveResult::PromotedPrevious {
                activation_id: candidate_id,
                manifest,
            });
        }
    }
    activate_bundled_default(root, default_head, head_inner_loader)
}

/// Activate the bundled default via the standard stage + publish pipeline;
/// any failure (or `default_head: None`) becomes
/// [`RecoveryActiveResult::Unhealthy`] so the daemon boots without inference.
fn activate_bundled_default(
    root: &Path,
    default_head: Option<DefaultHeadSource<'_>>,
    head_inner_loader: &HeadInnerLoader,
) -> Result<RecoveryActiveResult, RecoveryError> {
    let Some(default_head) = default_head else {
        return Ok(RecoveryActiveResult::Unhealthy {
            reason: "head.default not configured in launch config".into(),
        });
    };
    let pending = PendingActivation {
        root,
        origin_input: ActivationOriginInput::Default {
            source: default_head,
        },
        now_rfc3339: now_rfc3339(),
    };
    let result = match stage_and_validate_activation(pending, head_inner_loader) {
        Ok(r) => r,
        Err(e) => {
            return Ok(RecoveryActiveResult::Unhealthy {
                reason: format!("bundled default activation failed: {e}"),
            });
        }
    };
    let staging = staging_path_for(root, &result.activation_id);
    if let Err(e) =
        publish_active_generation(root, &staging, &result.manifest, &result.activation_id)
    {
        return Ok(RecoveryActiveResult::Unhealthy {
            reason: format!("bundled default publish failed: {e}"),
        });
    }
    tracing::warn!(
        target: "file_mgr::recovery",
        activation_id = %result.activation_id,
        "boot recovery activated bundled default",
    );
    Ok(RecoveryActiveResult::DefaultedFromBundle {
        activation_id: result.activation_id,
        manifest: result.manifest,
    })
}

/// Render `manifest.labels[]` to `labels.txt` atomically via the canonical
/// `labels_to_text`, byte-exact against a fresh activation (the post-regen
/// `labels_sha256` recheck relies on this).
fn regenerate_labels_from_manifest(
    labels_path: &Path,
    manifest: &ActiveHeadManifest,
) -> Result<(), FileError> {
    let bytes = crate::file_mgr::active_head_writer::labels_to_text(&manifest.labels).into_bytes();
    put_atomic(labels_path, &bytes)
}

/// Per-workspace sweep: complete delete tombstones, drop daemon-owned
/// head orphans, repair derived `head_count`, remove incomplete-create
/// directories.
pub fn recover_workspaces(root: &Path) -> Result<RecoveryWorkspaceReport, RecoveryError> {
    let workspaces_root = workspaces_dir(root);
    if !workspaces_root.is_dir() {
        return Ok(RecoveryWorkspaceReport::default());
    }
    let mut report = RecoveryWorkspaceReport::default();
    let mut incomplete_dirs: Vec<PathBuf> = Vec::new();
    // Shared UUID-v4 + dir filter; branch on workspace.json present (recover)
    // vs absent (incomplete-create cleanup). dirent_errors get their own
    // counter, distinct from workspace-level sweep failures.
    let outcome = crate::file_mgr::registry::walk_workspace_dirs(&workspaces_root)?;
    report.workspace_enumeration_failures = report
        .workspace_enumeration_failures
        .saturating_add(outcome.dirent_errors);
    for entry in outcome.entries {
        if !entry.has_core {
            // Incomplete create: defer remove_dir_all until the walk finishes
            // so we don't perturb the iterator's view.
            incomplete_dirs.push(entry.path);
            continue;
        }
        // Per-workspace failures are logged, not fatal.
        match recover_one_workspace(&entry.path) {
            Ok(per) => {
                report.workspaces_scanned += 1;
                report.fold_per_workspace(per);
            }
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    workspace = %entry.path.display(),
                    err = %e,
                    "per-workspace recovery failed; leaving for next boot",
                );
                report.workspace_recovery_failures =
                    report.workspace_recovery_failures.saturating_add(1);
            }
        }
    }
    // Remove incomplete-create dirs, fsync the parent once at the end.
    // removed_any is set on Err too: remove_dir_all can fail partway after
    // unlinking some dirents, so the fsync must still flush what landed.
    let mut removed_any = false;
    for path in incomplete_dirs {
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                report.incomplete_creates_removed += 1;
                removed_any = true;
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    "removed incomplete-create workspace directory",
                );
            }
            Err(e) => {
                removed_any = true;
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "incomplete-create cleanup failed; leaving for next boot",
                );
            }
        }
    }
    if removed_any && let Err(e) = fsync_dir(&workspaces_root) {
        tracing::warn!(
            target: "file_mgr::recovery",
            err = %e,
            "fsync workspaces/ after incomplete-create sweep failed (best-effort)",
        );
    }
    Ok(report)
}

/// Per-workspace sweep counters, folded into [`RecoveryWorkspaceReport`].
#[derive(Default)]
struct PerWorkspaceCounts {
    head_orphans_swept: usize,
    head_count_repaired: usize,
    dataset_tombstones_completed: usize,
    dataset_stage_orphans_swept: usize,
    converter_tombstones_completed: usize,
    converter_stage_orphans_swept: usize,
    training_logs_tombstones_completed: usize,
    training_logs_stage_orphans_swept: usize,
    converter_logs_tombstones_completed: usize,
    converter_logs_stage_orphans_swept: usize,
}

/// Recover one workspace dir: read `heads.json`, sweep head orphans,
/// repair `head_count`, then drain + finalize the four `.tmp/delete-*`
/// tombstone prefixes (one pass each) and their orphan stage dirs.
fn recover_one_workspace(workspace_dir: &Path) -> Result<PerWorkspaceCounts, FileError> {
    let mut counts = PerWorkspaceCounts::default();

    // Head-orphan sweep. A non-NotFound heads.json read failure
    // (parse/schema/EACCES) logs + skips the head half so one bad byte doesn't
    // block the tombstone drain; rebuild/write Err still propagates.
    let heads_for_sweep: Option<HeadIndex> = match read_head_index(workspace_dir) {
        Ok(idx) => Some(idx),
        Err(FileError::Io { ref source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            // heads.json missing under valid workspace.json: rebuild from
            // on-disk per-head manifests. An empty index would feed an empty
            // `known` set into sweep_head_orphans and unlink every surviving
            // head pair -- catastrophic data loss.
            let rebuilt = reconstruct_head_index_from_disk(workspace_dir)?;
            if rebuilt.heads.is_empty() {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    workspace = %workspace_dir.display(),
                    "heads.json missing under valid workspace.json; rewriting empty index",
                );
            } else {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    workspace = %workspace_dir.display(),
                    reconstructed = rebuilt.heads.len(),
                    "heads.json missing under valid workspace.json; rebuilt index from per-head manifests",
                );
            }
            write_head_index(workspace_dir, &rebuilt)?;
            Some(rebuilt)
        }
        Err(e) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                workspace = %workspace_dir.display(),
                err = %e,
                "heads.json unreadable; skipping head sweep + count repair",
            );
            None
        }
    };
    if let Some(heads) = heads_for_sweep.as_ref() {
        counts.head_orphans_swept += sweep_head_orphans(workspace_dir, heads)?;
        counts.head_count_repaired += repair_head_count(workspace_dir, heads)?;
    }

    // Tombstone drain runs even on heads-half failure so a corrupt heads.json
    // doesn't strand operator-committed deletes; one pass per `.tmp/delete-*`
    // prefix.
    let staging_dir = workspace_dir.join(".tmp");
    if staging_dir.is_dir() {
        let (completed, orphan_swept) = drain_staging_dir(&staging_dir, StagingScope::Dataset)?;
        counts.dataset_tombstones_completed += completed;
        counts.dataset_stage_orphans_swept += orphan_swept;
        let (completed, orphan_swept) = drain_staging_dir(&staging_dir, StagingScope::Converter)?;
        counts.converter_tombstones_completed += completed;
        counts.converter_stage_orphans_swept += orphan_swept;
        let (completed, orphan_swept) =
            drain_staging_dir(&staging_dir, StagingScope::TrainingLogs)?;
        counts.training_logs_tombstones_completed += completed;
        counts.training_logs_stage_orphans_swept += orphan_swept;
        let (completed, orphan_swept) =
            drain_staging_dir(&staging_dir, StagingScope::ConverterLogs)?;
        counts.converter_logs_tombstones_completed += completed;
        counts.converter_logs_stage_orphans_swept += orphan_swept;
    }
    Ok(counts)
}

/// Rebuild a [`HeadIndex`] from on-disk per-head manifests when
/// `heads.json` is missing: enumerate `<id>.json` with a paired `<id>.mpk`,
/// validate, require `head_id == filename_stem`. A structurally-invalid pair
/// is left for `sweep_head_orphans`, but a TRANSIENT read failure propagates
/// as `Err` so the caller DEFERS recovery rather than excluding a maybe-valid
/// head and letting the orphan sweep unlink its pair. Sorted newest-first by
/// `created_at` PARSED to an instant (lexical is non-monotonic against
/// `now_rfc3339`'s variable-width fractions), capped to
/// [`MAX_HEADS_PER_WORKSPACE`] (over-cap entries become orphans).
fn reconstruct_head_index_from_disk(workspace_dir: &Path) -> Result<HeadIndex, FileError> {
    let dir = heads_dir(workspace_dir);
    if !dir.is_dir() {
        return Ok(HeadIndex::default());
    }
    let entries = std::fs::read_dir(&dir).map_err(|e| io_err(dir.display(), e))?;
    let mut candidates: Vec<HeadId> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| io_err(dir.display(), e))?;
        let ft = match entry.file_type() {
            Ok(t) => t,
            // Transient stat failure: propagate (defer) vs dropping a valid head.
            Err(e) => return Err(io_err(dir.display(), e)),
        };
        if !ft.is_file() {
            continue;
        }
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some((stem, ext)) = name.rsplit_once('.') else {
            continue;
        };
        if ext != "json" {
            continue;
        }
        let Ok(id) = HeadId::parse(stem) else {
            continue;
        };
        // A manifest with no paired `.mpk` weights is not recoverable.
        if !head_artifact_path(workspace_dir, id).is_file() {
            continue;
        }
        candidates.push(id);
    }
    if candidates.is_empty() {
        return Ok(HeadIndex::default());
    }
    let mut records: Vec<HeadRecord> = Vec::with_capacity(candidates.len());
    for id in candidates {
        let manifest = match read_head_manifest(workspace_dir, id) {
            Ok(m) => m,
            // Transient read failure: propagate to DEFER (leaves heads.json
            // unwritten) vs excluding a maybe-valid head.
            Err(e @ FileError::Io { .. }) => return Err(e),
            // Structural failure (parse/schema): skip; the sweep reclaims the pair.
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    head_id = %id,
                    err = %e,
                    "skipping corrupt head manifest during index reconstruction",
                );
                continue;
            }
        };
        if manifest.head_id != id {
            tracing::warn!(
                target: "file_mgr::recovery",
                head_id = %id,
                manifest_head_id = %manifest.head_id,
                "head manifest head_id mismatches filename during index reconstruction; skipping",
            );
            continue;
        }
        records.push(HeadRecord {
            head_id: manifest.head_id,
            workspace_revision: manifest.workspace_revision,
            sha256: manifest.sha256,
            n_classes: manifest.n_classes,
            size_bytes: manifest.size_bytes,
            created_at: manifest.created_at,
        });
    }
    // head_id tiebreak keeps the truncate deterministic across reboots (else
    // phantom head appearances).
    records.sort_by(|a, b| {
        parse_rfc3339_or_epoch(&b.created_at)
            .cmp(&parse_rfc3339_or_epoch(&a.created_at))
            .then_with(|| a.head_id.to_string().cmp(&b.head_id.to_string()))
    });
    if records.len() > MAX_HEADS_PER_WORKSPACE {
        let dropped: Vec<String> = records[MAX_HEADS_PER_WORKSPACE..]
            .iter()
            .map(|r| r.head_id.to_string())
            .collect();
        tracing::warn!(
            target: "file_mgr::recovery",
            workspace = %workspace_dir.display(),
            cap = MAX_HEADS_PER_WORKSPACE,
            reconstructed = records.len(),
            dropped = ?dropped,
            "reconstructed head index exceeds MAX_HEADS_PER_WORKSPACE; the orphan sweep \
             will unlink the .mpk + .json pair for each dropped id (forward-loss). \
             Operator action: accept the eviction (oldest-first by created_at) or \
             rebuild against a daemon build with a larger cap before the next boot",
        );
        records.truncate(MAX_HEADS_PER_WORKSPACE);
    }
    Ok(HeadIndex { heads: records })
}

/// Unlink any `<workspace>/heads/<head_id>.{mpk,json}` whose stem is not
/// in `heads.json.heads[]`. Returns files removed (mpk + json counted
/// separately, so a fully-orphaned pair is 2).
fn sweep_head_orphans(workspace_dir: &Path, heads: &HeadIndex) -> Result<usize, FileError> {
    let dir = heads_dir(workspace_dir);
    if !dir.is_dir() {
        return Ok(0);
    }
    let known: std::collections::HashSet<HeadId> = heads.heads.iter().map(|h| h.head_id).collect();
    let mut removed = 0usize;
    let entries = std::fs::read_dir(&dir).map_err(|e| io_err(dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| io_err(dir.display(), e))?;
        let path = entry.path();
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "skipping unreadable heads/ entry",
                );
                continue;
            }
        };
        if !ft.is_file() {
            continue;
        }
        // Only `.mpk`/`.json` are daemon-owned; leave operator-pasted files.
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some((stem, ext)) = name.rsplit_once('.') else {
            continue;
        };
        if !matches!(ext, "mpk" | "json") {
            continue;
        }
        // Non-UUID filename: not daemon-produced, leave for triage.
        let Ok(id) = HeadId::parse(stem) else {
            continue;
        };
        if known.contains(&id) {
            continue;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => {
                removed += 1;
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    "removed daemon-owned head orphan",
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "failed to remove head orphan; leaving for next boot",
                );
            }
        }
    }
    if removed > 0
        && let Err(e) = fsync_dir(&dir)
    {
        tracing::warn!(
            target: "file_mgr::recovery",
            err = %e,
            "fsync heads/ after orphan sweep failed (best-effort)",
        );
    }
    Ok(removed)
}

/// Rewrite `workspace.json.head_count` to `heads.len()` if they disagree;
/// returns `1` on a write, `0` on a no-op.
fn repair_head_count(workspace_dir: &Path, heads: &HeadIndex) -> Result<usize, FileError> {
    let core_path = workspace_core_path(workspace_dir);
    let core = match read_workspace_core(workspace_dir) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                target: "file_mgr::recovery",
                path = %core_path.display(),
                err = %e,
                "read workspace.json failed during head_count repair; skipping",
            );
            return Ok(0);
        }
    };
    let expected = u8::try_from(heads.heads.len()).unwrap_or(u8::MAX);
    if core.head_count == expected {
        return Ok(0);
    }
    let mut updated = core.clone();
    updated.head_count = expected;
    write_workspace_core(workspace_dir, &updated)?;
    tracing::warn!(
        target: "file_mgr::recovery",
        workspace = %workspace_dir.display(),
        observed = core.head_count,
        repaired = expected,
        "repaired workspace.json.head_count",
    );
    Ok(1)
}

/// `<root>/active/.tmp/` sweep: remove orphan staging dirs from
/// failed/crashed activations. Publish atomically renames staging into
/// `generations/<id>/`, so anything left at boot is an orphan, and the API
/// isn't live yet so nothing races the sweep. Per-entry failures logged +
/// skipped; the post-sweep fsync makes the unlinks durable.
pub fn recover_active_staging(root: &Path) -> Result<RecoveryActiveStagingReport, RecoveryError> {
    let staging_dir = active_staging_dir(root);
    if !staging_dir.is_dir() {
        return Ok(RecoveryActiveStagingReport::default());
    }
    let mut report = RecoveryActiveStagingReport::default();
    let entries = match std::fs::read_dir(&staging_dir) {
        Ok(it) => it,
        // NotFound raced the is_dir() precheck: nothing to sweep.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok(report);
        }
        // EACCES/EIO/ELOOP: propagate so it surfaces at boot, not silently
        // accumulating orphans.
        Err(e) => {
            return Err(RecoveryError::File(io_err(staging_dir.display(), e)));
        }
    };
    let mut fs_mutation_observed = false;
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %staging_dir.display(),
                    err = %e,
                    "dirent read failed; continuing",
                );
                continue;
            }
        };
        let path = entry.path();
        // file_type does NOT follow symlinks, so a planted symlink is unlinked.
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "file_type failed; continuing",
                );
                continue;
            }
        };
        let removed = if ft.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match removed {
            Ok(()) => {
                report.active_staging_orphans_swept += 1;
                fs_mutation_observed = true;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Raced gone; count as swept.
                report.active_staging_orphans_swept += 1;
                fs_mutation_observed = true;
            }
            Err(e) => {
                // remove_dir_all can fail partway; mark mutation so the fsync
                // covers the half-done unlinks.
                fs_mutation_observed = true;
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "active staging orphan removal failed; will retry next boot",
                );
            }
        }
    }
    if fs_mutation_observed && let Err(e) = fsync_dir(&staging_dir) {
        tracing::warn!(
            target: "file_mgr::recovery",
            path = %staging_dir.display(),
            err = %e,
            "active staging dir fsync failed post-sweep",
        );
    }
    Ok(report)
}

/// Root `.tmp/` sweep: complete pending workspace-delete tombstones
/// (drain + finalize), eject the targeted workspace from `caches`, and
/// remove orphan stage dirs with no matching tombstone.
pub fn recover_root_staging(
    root: &Path,
    caches: &dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>>,
) -> Result<RecoveryRootReport, RecoveryError> {
    let staging_dir = root.join(".tmp");
    if !staging_dir.is_dir() {
        return Ok(RecoveryRootReport::default());
    }
    let mut report = RecoveryRootReport::default();
    let (completed, orphan_swept) = drain_workspace_staging_dir(&staging_dir, |ws_id| {
        caches.remove(&ws_id);
    })?;
    report.workspace_tombstones_completed += completed;
    report.workspace_stage_orphans_swept += orphan_swept;
    Ok(report)
}

/// Which tombstone prefix a workspace-scoped `.tmp/` sweep considers.
#[derive(Clone, Copy)]
enum StagingScope {
    Dataset,
    Converter,
    TrainingLogs,
    ConverterLogs,
}

/// Drain tombstones matching the scope's prefix + their orphan dirs;
/// no cache-eviction hook. Returns `(completed, orphan_swept)`.
fn drain_staging_dir(staging_dir: &Path, scope: StagingScope) -> Result<(usize, usize), FileError> {
    let prefix = match scope {
        StagingScope::Dataset => DATASET_TOMBSTONE_PREFIX,
        StagingScope::Converter => CONVERTER_TOMBSTONE_PREFIX,
        StagingScope::TrainingLogs => TRAINING_LOGS_TOMBSTONE_PREFIX,
        StagingScope::ConverterLogs => CONVERTER_LOGS_TOMBSTONE_PREFIX,
    };
    // No reclaim resolver: these scopes can't resurrect (no live sub-path
    // derivable from the tombstone).
    walk_staging(staging_dir, prefix, |_| {}, None)
}

/// Workspace-delete variant: drain + sweep, calling `evict` with each
/// completed tombstone's `WorkspaceId` so the cache map drops it before
/// any API consumer observes the vacated workspace.
fn drain_workspace_staging_dir<F>(staging_dir: &Path, evict: F) -> Result<(usize, usize), FileError>
where
    F: FnMut(WorkspaceId),
{
    // Resolver so a bare workspace tombstone (payload never staged out)
    // reclaims the live `workspaces/<id>/` instead of letting
    // recover_workspaces resurrect it. Only a dir carrying workspace.json
    // reclaims; a core-less dir is an incomplete create.
    let reclaim = |t: &DeleteTombstone| -> Option<PathBuf> {
        let root = staging_dir.parent()?;
        let ws = workspaces_dir(root).join(t.workspace_id().to_string());
        workspace_core_path(&ws).is_file().then_some(ws)
    };
    walk_staging(
        staging_dir,
        WORKSPACE_TOMBSTONE_PREFIX,
        evict,
        Some(&reclaim),
    )
}

/// Maps a tombstone to its still-live source dir so [`walk_staging`] can
/// reclaim an interrupted delete (`None` => nothing live). Only the
/// workspace scope supplies one.
type LiveSourceResolver<'a> = &'a dyn Fn(&DeleteTombstone) -> Option<PathBuf>;

/// Generic two-pass sweep: classify entries under `staging_dir` as
/// `prefix` tombstones vs stage dirs, pair + drain/finalize each tombstone
/// (bare ones just unlink), then `remove_dir_all` any unpaired stage dir.
/// `evict` fires after each completed tombstone with its `workspace_id`
/// (root-staging drops the cache cell; other scopes pass a no-op).
fn walk_staging<F>(
    staging_dir: &Path,
    prefix: &str,
    mut evict: F,
    reclaim_live_source: Option<LiveSourceResolver<'_>>,
) -> Result<(usize, usize), FileError>
where
    F: FnMut(WorkspaceId),
{
    let mut tombstone_files: Vec<PathBuf> = Vec::new();
    let mut stage_dirs: std::collections::HashMap<String, PathBuf> =
        std::collections::HashMap::new();
    let entries = std::fs::read_dir(staging_dir).map_err(|e| io_err(staging_dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| io_err(staging_dir.display(), e))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let ft = match entry.file_type() {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    err = %e,
                    "skipping unreadable staging entry",
                );
                continue;
            }
        };
        if ft.is_file() {
            if name.starts_with(prefix) && name.ends_with(".json") {
                tombstone_files.push(path);
            }
        } else if ft.is_dir() && name.starts_with(prefix) {
            // Key by bare name to match tombstone.stage_dir_name().
            stage_dirs.insert(name.to_string(), path);
        }
    }

    let mut completed = 0usize;
    let mut orphan_swept = 0usize;
    // Drives the fsync independently of the success counters so an errored
    // partial-unlink still flushes its dirent changes.
    let mut fs_mutation_observed = false;

    for tombstone_path in tombstone_files {
        let tombstone = match read_tombstone(&tombstone_path) {
            Ok(t) => t,
            Err(e) => {
                // Corrupt tombstone: best-effort unlink, mark mutation so the
                // fsync makes it durable, else power loss replays it forever.
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %tombstone_path.display(),
                    err = %e,
                    "removing corrupt tombstone",
                );
                let _ = std::fs::remove_file(&tombstone_path);
                fs_mutation_observed = true;
                continue;
            }
        };
        // Downstream paths re-derive from the PARSED kind/job_id, not the
        // scanned name; if a tampered body disagrees with the filename,
        // finalize would target a different path and never unlink the scanned
        // file, re-looping every boot, so unlink the scanned path here.
        if tombstone_path.file_name().and_then(|n| n.to_str())
            != Some(tombstone.filename().as_str())
        {
            tracing::warn!(
                target: "file_mgr::recovery",
                path = %tombstone_path.display(),
                parsed = %tombstone.filename(),
                "tombstone filename disagrees with parsed kind/job_id; removing",
            );
            let _ = std::fs::remove_file(&tombstone_path);
            fs_mutation_observed = true;
            continue;
        }
        let staged = StagedDelete::for_tombstone(staging_dir, &tombstone);
        // Pair the stage dir (if any) so the orphan sweep ignores it; the
        // idempotent finalize handles the tombstone-only case.
        stage_dirs.remove(&tombstone.stage_dir_name());
        // Reconcile an interrupted delete: durable tombstone but absent
        // PAYLOAD while live `workspaces/<id>/` still exists. Gate on the
        // payload's absence (not the stage dir's) to cover both a bare
        // tombstone AND the empty-stage-dir crash window; else the drain reads
        // the missing payload as "already drained", unlinks the tombstone, and
        // recover_workspaces resurrects the destroyed workspace. A
        // genuinely-drained delete returns None and falls to finalize.
        let payload_absent = matches!(
            std::fs::symlink_metadata(&staged.payload),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound
        );
        if payload_absent
            && let Some(reclaim) = reclaim_live_source
            && let Some(live_source) = reclaim(&tombstone)
        {
            match stage_payload(&live_source, &staged) {
                Ok(()) => {
                    tracing::warn!(
                        target: "file_mgr::recovery",
                        workspace_id = %tombstone.workspace_id(),
                        "reclaimed live workspace for an interrupted delete (tombstone present, payload never staged)",
                    );
                }
                Err(e) => {
                    fs_mutation_observed = true;
                    tracing::warn!(
                        target: "file_mgr::recovery",
                        workspace_id = %tombstone.workspace_id(),
                        err = %e,
                        "failed to reclaim live workspace for interrupted delete; leaving for next boot",
                    );
                    continue;
                }
            }
        }
        // Bounded batches with NO iteration cap: each More removes >=1 so it
        // terminates once entries are exhausted, whereas a cap would leave the
        // tombstone + stage_dir and replay forever. Only an I/O Err breaks
        // without finalization.
        let drain_ok = loop {
            match drain_staged_payload(&staged, DEFAULT_DELETE_BATCH_ENTRIES) {
                Ok(DrainResult::Done) => {
                    // May have unlinked this call too; mark for fsync.
                    fs_mutation_observed = true;
                    break true;
                }
                Ok(DrainResult::More) => {
                    continue;
                }
                Err(e) => {
                    // Per-batch ops may have unlinked before the Err; mark so
                    // the fsync runs on the early-break path.
                    fs_mutation_observed = true;
                    tracing::warn!(
                        target: "file_mgr::recovery",
                        tombstone = %tombstone_path.display(),
                        err = %e,
                        "staged delete drain failed; leaving for next boot",
                    );
                    break false;
                }
            }
        };
        if !drain_ok {
            continue;
        }
        if let Err(e) = finalize_staged_delete(&staged) {
            tracing::warn!(
                target: "file_mgr::recovery",
                tombstone = %tombstone_path.display(),
                err = %e,
                "staged delete finalize failed; leaving for next boot",
            );
            continue;
        }
        // Log the resumed job_id so operators correlate a `/jobs/{job_id}`
        // that vanished across the restart (no Completed transition) with the
        // on-disk drain that finished.
        tracing::info!(
            target: "file_mgr::recovery",
            job_id = %tombstone.job_id(),
            workspace_id = %tombstone.workspace_id(),
            "resumed pending delete drain to completion",
        );
        completed += 1;
        evict(tombstone.workspace_id());
    }

    // Unpaired stage dirs are unreferenced residue: remove_dir_all.
    for (name, path) in stage_dirs {
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                orphan_swept += 1;
                fs_mutation_observed = true;
                tracing::warn!(
                    target: "file_mgr::recovery",
                    path = %path.display(),
                    "removed orphan stage directory without tombstone",
                );
            }
            Err(e) => {
                // remove_dir_all can fail after some unlinks; mark so the
                // fsync covers the half-done changes.
                fs_mutation_observed = true;
                tracing::warn!(
                    target: "file_mgr::recovery",
                    name = %name,
                    err = %e,
                    "orphan stage cleanup failed; leaving for next boot",
                );
            }
        }
    }

    if fs_mutation_observed && let Err(e) = fsync_dir(staging_dir) {
        tracing::warn!(
            target: "file_mgr::recovery",
            err = %e,
            "fsync staging dir after sweep failed (best-effort)",
        );
    }

    Ok((completed, orphan_swept))
}

/// Lowercase-hex SHA-256 of a file, read in [`STREAM_HASH_CHUNK`] chunks.
fn sha256_stream(path: &Path) -> Result<String, FileError> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| io_err(path.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; STREAM_HASH_CHUNK];
    loop {
        let n = f.read(&mut buf).map_err(|e| io_err(path.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lowercase(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]

    use super::*;
    use crate::common::asset_path::AssetPath;
    use crate::common::ids::JobId;
    use crate::common::workspace::{
        HeadIndex, HeadManifest, HeadRecord, WorkspaceCore, WorkspaceRevision,
    };
    use crate::file_mgr::schema::{
        head_artifact_path, head_manifest_path, write_head_index, write_head_manifest,
        write_workspace_core,
    };
    use crate::file_mgr::staging::{DeleteTombstone, stage_payload, write_tombstone};
    use std::path::PathBuf;

    fn ws_id(byte: u8) -> WorkspaceId {
        let s = format!("11111111-2222-4333-8444-5555555555{byte:02x}");
        WorkspaceId::parse(&s).unwrap()
    }

    fn head_id(byte: u8) -> HeadId {
        let s = format!("11111111-2222-4333-8444-5555555555{byte:02x}");
        HeadId::parse(&s).unwrap()
    }

    fn job_id() -> JobId {
        JobId::parse("22222222-3333-4444-8555-666666666666").unwrap()
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn synth_workspace_core(id: WorkspaceId, head_count: u8) -> WorkspaceCore {
        WorkspaceCore {
            id,
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(5),
            head_count,
        }
    }

    fn synth_head_manifest(workspace: WorkspaceId, hid: HeadId, mpk: &[u8]) -> HeadManifest {
        HeadManifest {
            head_id: hid,
            workspace_id: workspace,
            workspace_revision: rev(5),
            sha256: hex_sha256(mpk),
            n_classes: 2,
            size_bytes: mpk.len() as u64,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["alpha".to_string(), "beta".to_string()],
        }
    }

    fn hex_sha256(bytes: &[u8]) -> String {
        let digest = Sha256::digest(bytes);
        crate::common::hex::hex_lowercase(digest.as_slice())
    }

    /// `()` candidate so the activation pipeline runs without `inference`.
    fn synth_loader_ok() -> Box<HeadInnerLoader> {
        Box::new(|_mpk: &Path, _labels: &Path, _id: HeadId| {
            Ok(Box::new(()) as Box<dyn std::any::Any + Send>)
        })
    }

    fn fresh_bundled_default(root: &Path, mpk: &[u8], labels_text: &str) -> (PathBuf, PathBuf) {
        let dir = root.join("bundled_default");
        std::fs::create_dir_all(&dir).unwrap();
        let head = dir.join("head.mpk");
        let labels = dir.join("labels.txt");
        std::fs::write(&head, mpk).unwrap();
        std::fs::write(&labels, labels_text).unwrap();
        (head, labels)
    }

    fn default_source<'a>(path: &'a Path, labels_path: &'a Path) -> DefaultHeadSource<'a> {
        DefaultHeadSource { path, labels_path }
    }

    fn default_origin<'a>(path: &'a Path, labels_path: &'a Path) -> ActivationOriginInput<'a> {
        ActivationOriginInput::Default {
            source: default_source(path, labels_path),
        }
    }

    /// One workspace dir with canonical files + a single head.
    fn fresh_workspace_with_head(root: &Path, ws: WorkspaceId, head: HeadId) -> PathBuf {
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(root, &ws);
        std::fs::create_dir_all(crate::file_mgr::schema::heads_dir(&ws_dir)).unwrap();
        std::fs::create_dir_all(ws_dir.join(".tmp")).unwrap();
        let mpk = b"MPK-CONTENT";
        let manifest = synth_head_manifest(ws, head, mpk);
        let mut idx = HeadIndex::default();
        idx.heads.push(HeadRecord {
            head_id: head,
            workspace_revision: manifest.workspace_revision.clone(),
            sha256: manifest.sha256.clone(),
            n_classes: manifest.n_classes,
            size_bytes: manifest.size_bytes,
            created_at: manifest.created_at.clone(),
        });
        write_head_index(&ws_dir, &idx).unwrap();
        write_head_manifest(&ws_dir, &manifest).unwrap();
        std::fs::write(head_artifact_path(&ws_dir, head), mpk).unwrap();
        write_workspace_core(&ws_dir, &synth_workspace_core(ws, 1)).unwrap();
        ws_dir
    }

    /// Activate the bundled default for a known-good current generation.
    fn seed_default_active_generation(root: &Path) -> (PathBuf, PathBuf, String) {
        let (head, labels) = fresh_bundled_default(root, b"DEFAULT-MPK", "cat\ndog\n");
        let pending = PendingActivation {
            root,
            origin_input: default_origin(&head, &labels),
            now_rfc3339: now_rfc3339(),
        };
        let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();
        let staging = staging_path_for(root, &result.activation_id);
        publish_active_generation(root, &staging, &result.manifest, &result.activation_id).unwrap();
        (head, labels, result.activation_id)
    }

    #[test]
    fn recover_active_current_passes() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels, current_id) = seed_default_active_generation(tmp.path());
        let result = recover_active_head(
            tmp.path(),
            Some(default_source(&head, &labels)),
            &*synth_loader_ok(),
        )
        .unwrap();
        match result {
            RecoveryActiveResult::Current { activation_id, .. } => {
                assert_eq!(activation_id, current_id);
            }
            other => panic!("expected Current, got {other:?}"),
        }
    }

    #[test]
    fn recover_active_corrupt_head_falls_back_to_previous() {
        let tmp = tempfile::tempdir().unwrap();
        // First generation becomes "previous" after the second is published.
        let (head1, labels1) = fresh_bundled_default(tmp.path(), b"DEFAULT-MPK-A", "cat\ndog\n");
        let pending1 = PendingActivation {
            root: tmp.path(),
            origin_input: default_origin(&head1, &labels1),
            now_rfc3339: now_rfc3339(),
        };
        let r1 = stage_and_validate_activation(pending1, &*synth_loader_ok()).unwrap();
        publish_active_generation(
            tmp.path(),
            &staging_path_for(tmp.path(), &r1.activation_id),
            &r1.manifest,
            &r1.activation_id,
        )
        .unwrap();
        // Distinct mtimes between the two generations.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let (head2, labels2) =
            fresh_bundled_default(tmp.path(), b"DEFAULT-MPK-B", "cat\ndog\nbird\n");
        let pending2 = PendingActivation {
            root: tmp.path(),
            origin_input: default_origin(&head2, &labels2),
            now_rfc3339: now_rfc3339(),
        };
        let r2 = stage_and_validate_activation(pending2, &*synth_loader_ok()).unwrap();
        publish_active_generation(
            tmp.path(),
            &staging_path_for(tmp.path(), &r2.activation_id),
            &r2.manifest,
            &r2.activation_id,
        )
        .unwrap();
        // Tamper the CURRENT head.mpk so verify fails and r1 is promoted.
        let current_head =
            active_generation_dir(tmp.path(), &r2.activation_id).join(ACTIVE_HEAD_FILENAME);
        std::fs::write(&current_head, b"TAMPERED").unwrap();
        let result = recover_active_head(
            tmp.path(),
            Some(default_source(&head2, &labels2)),
            &*synth_loader_ok(),
        )
        .unwrap();
        match result {
            RecoveryActiveResult::PromotedPrevious { activation_id, .. } => {
                assert_eq!(activation_id, r1.activation_id);
            }
            other => panic!("expected PromotedPrevious, got {other:?}"),
        }
        let pointer = read_active_current(tmp.path()).unwrap();
        assert_eq!(pointer.activation_id, r1.activation_id);
    }

    #[test]
    fn recover_active_corrupt_labels_regenerates_from_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels, current_id) = seed_default_active_generation(tmp.path());
        // Tampered labels.txt mismatches its hash; recovery regenerates.
        let labels_path =
            active_generation_dir(tmp.path(), &current_id).join(ACTIVE_LABELS_FILENAME);
        std::fs::write(&labels_path, b"TAMPERED LABELS").unwrap();
        let result = recover_active_head(
            tmp.path(),
            Some(default_source(&head, &labels)),
            &*synth_loader_ok(),
        )
        .unwrap();
        match result {
            RecoveryActiveResult::Current { activation_id, .. } => {
                assert_eq!(activation_id, current_id);
            }
            other => panic!("expected Current after labels regen, got {other:?}"),
        }
        let regen = std::fs::read_to_string(&labels_path).unwrap();
        assert_eq!(regen, "cat\ndog");
    }

    #[test]
    fn recover_active_no_valid_generation_falls_back_to_default() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels, current_id) = seed_default_active_generation(tmp.path());
        // Tamper the only generation's head.mpk: no previous exists, so
        // recovery re-activates the bundled default.
        let head_path = active_generation_dir(tmp.path(), &current_id).join(ACTIVE_HEAD_FILENAME);
        std::fs::write(&head_path, b"TAMPERED").unwrap();
        let result = recover_active_head(
            tmp.path(),
            Some(default_source(&head, &labels)),
            &*synth_loader_ok(),
        )
        .unwrap();
        match result {
            RecoveryActiveResult::DefaultedFromBundle { activation_id, .. } => {
                assert_ne!(activation_id, current_id);
            }
            other => panic!("expected DefaultedFromBundle, got {other:?}"),
        }
    }

    #[test]
    fn recover_active_bundled_default_missing_returns_unhealthy() {
        let tmp = tempfile::tempdir().unwrap();
        let missing_head = tmp.path().join("does_not_exist.mpk");
        let missing_labels = tmp.path().join("does_not_exist.labels.txt");
        let result = recover_active_head(
            tmp.path(),
            Some(default_source(&missing_head, &missing_labels)),
            &*synth_loader_ok(),
        )
        .unwrap();
        assert!(matches!(result, RecoveryActiveResult::Unhealthy { .. }));
    }

    #[test]
    fn recover_active_no_default_head_returns_unhealthy() {
        let tmp = tempfile::tempdir().unwrap();
        let result = recover_active_head(tmp.path(), None, &*synth_loader_ok()).unwrap();
        match result {
            RecoveryActiveResult::Unhealthy { reason } => {
                assert!(
                    reason.contains("head.default not configured"),
                    "diagnostic should surface the root cause: {reason}",
                );
            }
            other => panic!("expected Unhealthy with head.default reason, got {other:?}"),
        }
    }

    #[test]
    fn recover_workspace_orphan_head_files_swept() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let orphan_id = head_id(0xCC);
        let orphan_mpk = head_artifact_path(&ws_dir, orphan_id);
        let orphan_json = head_manifest_path(&ws_dir, orphan_id);
        std::fs::write(&orphan_mpk, b"ORPHAN").unwrap();
        std::fs::write(&orphan_json, b"{}").unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.workspaces_scanned, 1);
        assert_eq!(report.head_orphans_swept, 2);
        assert!(head_artifact_path(&ws_dir, head).is_file());
        assert!(head_manifest_path(&ws_dir, head).is_file());
        assert!(!orphan_mpk.exists());
        assert!(!orphan_json.exists());
    }

    #[test]
    fn recover_workspace_missing_heads_json_rebuilds_from_per_head_manifests() {
        // Missing heads.json with valid pairs must rebuild, not default to []
        // (which would unlink every real head).
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let heads_json = ws_dir.join("heads.json");
        std::fs::remove_file(&heads_json).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.workspaces_scanned, 1);
        assert!(
            head_artifact_path(&ws_dir, head).is_file(),
            "legitimate head .mpk must survive heads.json loss"
        );
        assert!(
            head_manifest_path(&ws_dir, head).is_file(),
            "legitimate head .json must survive heads.json loss"
        );
        let rebuilt = read_head_index(&ws_dir).unwrap();
        assert_eq!(rebuilt.heads.len(), 1);
        assert_eq!(rebuilt.heads[0].head_id, head);
        let core = read_workspace_core(&ws_dir).unwrap();
        assert_eq!(core.head_count, 1);
        // Reconstructed pair is in `known`, so not swept.
        assert_eq!(report.head_orphans_swept, 0);
    }

    #[test]
    fn recover_workspace_missing_heads_json_with_no_per_head_manifests_writes_empty() {
        // No valid pairs: write the empty index so boots converge, sweep the
        // half-formed .mpk as an orphan.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        std::fs::remove_file(ws_dir.join("heads.json")).unwrap();
        // Remove the head's .json so the rebuild finds no valid pairs.
        std::fs::remove_file(head_manifest_path(&ws_dir, head)).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.workspaces_scanned, 1);
        let rebuilt = read_head_index(&ws_dir).unwrap();
        assert_eq!(rebuilt.heads.len(), 0);
        // Orphan .mpk (no paired manifest) is swept.
        assert_eq!(report.head_orphans_swept, 1);
        assert!(!head_artifact_path(&ws_dir, head).exists());
    }

    #[test]
    fn recover_workspace_head_count_repaired() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        // head_count=0 though heads.json has 1 entry.
        let mut core = read_workspace_core(&ws_dir).unwrap();
        core.head_count = 0;
        write_workspace_core(&ws_dir, &core).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.head_count_repaired, 1);
        let core = read_workspace_core(&ws_dir).unwrap();
        assert_eq!(core.head_count, 1);
    }

    #[test]
    fn recover_workspace_dataset_tombstone_completed() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let staging = ws_dir.join(".tmp");
        let tombstone = DeleteTombstone::Dataset {
            job_id: job_id(),
            workspace_id: ws,
            path: Some(AssetPath::parse("audio/cat").unwrap()),
            workspace_revision_id: 6,
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        let target = tmp.path().join("dataset.wav");
        std::fs::write(&target, b"data").unwrap();
        stage_payload(&target, &staged).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.dataset_tombstones_completed, 1);
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
    }

    #[test]
    fn recover_workspace_dataset_stage_orphan_without_tombstone_swept() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let staging = ws_dir.join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let orphan_dir = staging.join("delete-assets-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
        std::fs::create_dir_all(orphan_dir.join("payload")).unwrap();
        std::fs::write(orphan_dir.join("payload/leftover"), b"x").unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.dataset_stage_orphans_swept, 1);
        assert!(!orphan_dir.exists());
    }

    #[test]
    fn recover_workspace_converter_tombstone_completed() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let staging = ws_dir.join(".tmp");
        let tombstone = DeleteTombstone::Converter {
            job_id: job_id(),
            workspace_id: ws,
            path: Some(AssetPath::parse("tfjs/model.json").unwrap()),
            workspace_revision_id: 6,
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        let target = tmp.path().join("model.json");
        std::fs::write(&target, b"manifest-bytes").unwrap();
        stage_payload(&target, &staged).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.converter_tombstones_completed, 1);
        // Prefix-dispatch keeps the trees independent.
        assert_eq!(report.dataset_tombstones_completed, 0);
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
    }

    #[test]
    fn recover_workspace_converter_stage_orphan_without_tombstone_swept() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let staging = ws_dir.join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let orphan_dir = staging.join("delete-converters-cccccccc-dddd-4eee-8fff-aaaaaaaaaaaa");
        std::fs::create_dir_all(orphan_dir.join("payload")).unwrap();
        std::fs::write(orphan_dir.join("payload/leftover"), b"x").unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.converter_stage_orphans_swept, 1);
        assert!(!orphan_dir.exists());
    }

    #[test]
    fn recover_workspace_training_logs_tombstone_completed() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        // Logs aren't workspace state, so this variant has no
        // workspace_revision_id (unlike Dataset/Converter).
        let staging = ws_dir.join(".tmp");
        let tombstone = DeleteTombstone::TrainingLogs {
            job_id: job_id(),
            workspace_id: ws,
            path: Some(AssetPath::parse("aaaaaaaa-1111-4222-8333-444444444444.jsonl").unwrap()),
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        let target = tmp.path().join("training_log.jsonl");
        std::fs::write(&target, b"{\"seq\":1}\n").unwrap();
        stage_payload(&target, &staged).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.training_logs_tombstones_completed, 1);
        assert_eq!(report.dataset_tombstones_completed, 0);
        assert_eq!(report.converter_tombstones_completed, 0);
        assert_eq!(report.converter_logs_tombstones_completed, 0);
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
    }

    #[test]
    fn recover_workspace_training_logs_stage_orphan_without_tombstone_swept() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let staging = ws_dir.join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let orphan_dir = staging.join("delete-training-logs-11111111-2222-4333-8444-555555555555");
        std::fs::create_dir_all(orphan_dir.join("payload")).unwrap();
        std::fs::write(orphan_dir.join("payload/leftover.jsonl"), b"{}").unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.training_logs_stage_orphans_swept, 1);
        assert!(!orphan_dir.exists());
    }

    #[test]
    fn recover_workspace_converter_logs_tombstone_completed() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        // Whole-tree wipe shape: `path = None`.
        let staging = ws_dir.join(".tmp");
        let tombstone = DeleteTombstone::ConverterLogs {
            job_id: job_id(),
            workspace_id: ws,
            path: None,
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        let target = tmp.path().join("converter_logs_tree");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("a.jsonl"), b"{}").unwrap();
        std::fs::write(target.join("b.jsonl"), b"{}").unwrap();
        stage_payload(&target, &staged).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.converter_logs_tombstones_completed, 1);
        assert_eq!(report.training_logs_tombstones_completed, 0);
        assert_eq!(report.dataset_tombstones_completed, 0);
        assert_eq!(report.converter_tombstones_completed, 0);
        assert!(!staged.tombstone.exists());
        assert!(!staged.stage_dir.exists());
    }

    #[test]
    fn recover_workspace_converter_logs_stage_orphan_without_tombstone_swept() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);
        let staging = ws_dir.join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let orphan_dir = staging.join("delete-converter-logs-22222222-3333-4444-8555-666666666666");
        std::fs::create_dir_all(orphan_dir.join("payload")).unwrap();
        std::fs::write(orphan_dir.join("payload/x.jsonl"), b"{}").unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.converter_logs_stage_orphans_swept, 1);
        assert!(!orphan_dir.exists());
    }

    /// Fires 4 distinct recovery conditions at once and asserts each lands in
    /// its own counter while the other 6 stay 0, catching a `fold_per_workspace`
    /// cross-wire that per-counter tests miss when every value is 1.
    #[test]
    fn recover_workspace_cross_product_fold_correctness() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let head = head_id(0xBB);
        let ws_dir = fresh_workspace_with_head(tmp.path(), ws, head);

        // head_orphans_swept = 2
        let orphan_id = head_id(0xCC);
        std::fs::write(head_artifact_path(&ws_dir, orphan_id), b"ORPHAN").unwrap();
        std::fs::write(head_manifest_path(&ws_dir, orphan_id), b"{}").unwrap();

        // head_count_repaired = 1
        let mut core = read_workspace_core(&ws_dir).unwrap();
        core.head_count = 0;
        write_workspace_core(&ws_dir, &core).unwrap();

        // dataset_tombstones_completed = 1
        let staging = ws_dir.join(".tmp");
        let tombstone = DeleteTombstone::Dataset {
            job_id: job_id(),
            workspace_id: ws,
            path: Some(AssetPath::parse("audio/cat").unwrap()),
            workspace_revision_id: 6,
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        let target = tmp.path().join("dataset.wav");
        std::fs::write(&target, b"data").unwrap();
        stage_payload(&target, &staged).unwrap();

        // converter_stage_orphans_swept = 1
        std::fs::create_dir_all(&staging).unwrap();
        let conv_orphan = staging.join("delete-converters-cccccccc-dddd-4eee-8fff-aaaaaaaaaaaa");
        std::fs::create_dir_all(conv_orphan.join("payload")).unwrap();
        std::fs::write(conv_orphan.join("payload/leftover"), b"x").unwrap();

        let report = recover_workspaces(tmp.path()).unwrap();

        assert_eq!(report.head_orphans_swept, 2, "head_orphans_swept");
        assert_eq!(report.head_count_repaired, 1, "head_count_repaired");
        assert_eq!(
            report.dataset_tombstones_completed, 1,
            "dataset_tombstones_completed",
        );
        assert_eq!(
            report.converter_stage_orphans_swept, 1,
            "converter_stage_orphans_swept",
        );

        // A cross-wired fold surfaces as a nonzero here.
        assert_eq!(
            report.dataset_stage_orphans_swept, 0,
            "dataset_stage_orphans_swept must stay 0",
        );
        assert_eq!(
            report.converter_tombstones_completed, 0,
            "converter_tombstones_completed must stay 0",
        );
        assert_eq!(
            report.training_logs_tombstones_completed, 0,
            "training_logs_tombstones_completed must stay 0",
        );
        assert_eq!(
            report.training_logs_stage_orphans_swept, 0,
            "training_logs_stage_orphans_swept must stay 0",
        );
        assert_eq!(
            report.converter_logs_tombstones_completed, 0,
            "converter_logs_tombstones_completed must stay 0",
        );
        assert_eq!(
            report.converter_logs_stage_orphans_swept, 0,
            "converter_logs_stage_orphans_swept must stay 0",
        );

        // Caller-driven counters.
        assert_eq!(report.workspaces_scanned, 1, "workspaces_scanned");
        assert_eq!(
            report.workspace_recovery_failures, 0,
            "workspace_recovery_failures",
        );
        assert_eq!(
            report.workspace_enumeration_failures, 0,
            "workspace_enumeration_failures",
        );
    }

    /// A symlinked `workspace.json` must be skipped (neither recovered nor
    /// auto-deleted) and surfaced via `workspace_enumeration_failures`, else
    /// recovery would walk an out-of-tree path as workspace metadata.
    #[test]
    #[cfg(unix)]
    fn recover_skips_workspace_with_symlinked_workspace_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(tmp.path(), &ws);
        std::fs::create_dir_all(&ws_dir).unwrap();
        let foreign = tmp.path().join("foreign_workspace.json");
        std::fs::write(&foreign, "{}").unwrap();
        std::os::unix::fs::symlink(
            &foreign,
            crate::file_mgr::schema::workspace_core_path(&ws_dir),
        )
        .unwrap();

        let report = recover_workspaces(tmp.path()).unwrap();
        // Not walked AND not auto-deleted; operator triages.
        assert_eq!(report.workspaces_scanned, 0);
        assert_eq!(report.incomplete_creates_removed, 0);
        assert!(
            ws_dir.exists(),
            "operator-triage stance: do not auto-delete"
        );
        assert_eq!(
            report.workspace_enumeration_failures, 1,
            "symlinked workspace.json must bump workspace_enumeration_failures",
        );
        assert_eq!(
            report.workspace_recovery_failures, 0,
            "enumeration failures do NOT account into workspace_recovery_failures",
        );
    }

    #[test]
    fn recover_incomplete_workspace_create_directory_removed() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(workspaces_dir(tmp.path())).unwrap();
        let ws = ws_id(0xAA);
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(tmp.path(), &ws);
        std::fs::create_dir_all(ws_dir.join("heads")).unwrap();
        std::fs::create_dir_all(ws_dir.join(".tmp")).unwrap();
        let report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(report.incomplete_creates_removed, 1);
        assert!(!ws_dir.exists());
        assert_eq!(report.workspaces_scanned, 0);
    }

    #[test]
    fn recover_root_workspace_tombstone_completed() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let ws = ws_id(0xAA);
        let tombstone = DeleteTombstone::Workspace {
            job_id: job_id(),
            workspace_id: ws,
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        let target = tmp.path().join("victim_ws");
        std::fs::create_dir_all(target.join("heads")).unwrap();
        std::fs::write(target.join("workspace.json"), b"{}").unwrap();
        stage_payload(&target, &staged).unwrap();
        let caches: dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>> =
            dashmap::DashMap::new();
        caches.insert(
            ws,
            Arc::new(WorkspaceCacheCell::new(
                synth_workspace_core(ws, 0),
                HeadIndex::default(),
            )),
        );
        let report = recover_root_staging(tmp.path(), &caches).unwrap();
        assert_eq!(report.workspace_tombstones_completed, 1);
        assert!(!staged.tombstone.exists());
        assert!(caches.get(&ws).is_none(), "cache cell evicted");
    }

    #[test]
    fn recover_root_workspace_stage_orphan_swept() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let orphan = staging.join("delete-workspace-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee");
        std::fs::create_dir_all(orphan.join("payload/heads")).unwrap();
        std::fs::write(orphan.join("payload/workspace.json"), b"{}").unwrap();
        let caches: dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>> =
            dashmap::DashMap::new();
        let report = recover_root_staging(tmp.path(), &caches).unwrap();
        assert_eq!(report.workspace_stage_orphans_swept, 1);
        assert!(!orphan.exists());
    }

    /// Resurrection regression: a durable workspace tombstone whose payload
    /// was never staged out must reclaim + drain the live workspace so the
    /// subsequent recover_workspaces pass can't resurrect it.
    #[test]
    fn recover_root_bare_workspace_tombstone_reclaims_live_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let ws = ws_id(0xBB);
        // Durable tombstone, NO stage dir (payload never renamed out).
        let tombstone = DeleteTombstone::Workspace {
            job_id: job_id(),
            workspace_id: ws,
            created_at: now_rfc3339(),
        };
        write_tombstone(&staging, &tombstone).unwrap();
        // Workspace still fully alive.
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(tmp.path(), &ws);
        std::fs::create_dir_all(ws_dir.join("heads")).unwrap();
        std::fs::write(workspace_core_path(&ws_dir), b"{}").unwrap();

        let caches: dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>> =
            dashmap::DashMap::new();
        caches.insert(
            ws,
            Arc::new(WorkspaceCacheCell::new(
                synth_workspace_core(ws, 0),
                HeadIndex::default(),
            )),
        );
        let report = recover_root_staging(tmp.path(), &caches).unwrap();
        // Completed via reclaim + drain, not a no-op nor orphan sweep.
        assert_eq!(report.workspace_tombstones_completed, 1);
        assert_eq!(report.workspace_stage_orphans_swept, 0);
        assert!(
            !ws_dir.exists(),
            "workspace must be deleted, not resurrected",
        );
        assert!(caches.get(&ws).is_none(), "cache cell evicted");
        let ws_report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(
            ws_report.workspaces_scanned, 0,
            "no workspace resurrected after an interrupted delete",
        );
    }

    /// Empty-stage-dir variant: reclaim gates on payload absence (not the
    /// stage dir's), so an empty stage dir is still reclaimed + drained rather
    /// than completing as a no-op.
    #[test]
    fn recover_root_empty_stage_dir_tombstone_reclaims_live_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = tmp.path().join(".tmp");
        std::fs::create_dir_all(&staging).unwrap();
        let ws = ws_id(0xCC);
        let tombstone = DeleteTombstone::Workspace {
            job_id: job_id(),
            workspace_id: ws,
            created_at: now_rfc3339(),
        };
        let staged = write_tombstone(&staging, &tombstone).unwrap();
        // Empty stage dir: rename never happened, so no payload inside.
        std::fs::create_dir_all(&staged.stage_dir).unwrap();
        assert!(staged.stage_dir.exists() && !staged.payload.exists());
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(tmp.path(), &ws);
        std::fs::create_dir_all(ws_dir.join("heads")).unwrap();
        std::fs::write(workspace_core_path(&ws_dir), b"{}").unwrap();

        let caches: dashmap::DashMap<WorkspaceId, Arc<WorkspaceCacheCell>> =
            dashmap::DashMap::new();
        let report = recover_root_staging(tmp.path(), &caches).unwrap();
        assert_eq!(report.workspace_tombstones_completed, 1);
        assert_eq!(report.workspace_stage_orphans_swept, 0);
        assert!(
            !ws_dir.exists(),
            "workspace must be deleted, not resurrected via the empty-stage-dir window",
        );
        let ws_report = recover_workspaces(tmp.path()).unwrap();
        assert_eq!(ws_report.workspaces_scanned, 0, "no workspace resurrected");
    }

    #[test]
    fn recover_active_staging_returns_default_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let report = recover_active_staging(tmp.path()).expect("sweep ok on missing tree");
        assert_eq!(report.active_staging_orphans_swept, 0);
    }

    #[test]
    fn recover_active_staging_sweeps_orphan_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let staging = active_staging_dir(tmp.path());
        std::fs::create_dir_all(&staging).unwrap();
        let orphan = staging.join("11111111-2222-4333-8444-555555555555");
        std::fs::create_dir_all(&orphan).unwrap();
        std::fs::write(orphan.join("head.mpk"), b"partial-bytes").unwrap();
        let report = recover_active_staging(tmp.path()).unwrap();
        assert_eq!(report.active_staging_orphans_swept, 1);
        assert!(!orphan.exists(), "orphan staging dir must be removed");
    }

    #[test]
    fn recover_active_staging_sweeps_loose_files_too() {
        // The dir is daemon-owned, so a loose file is removed too.
        let tmp = tempfile::tempdir().unwrap();
        let staging = active_staging_dir(tmp.path());
        std::fs::create_dir_all(&staging).unwrap();
        let loose = staging.join("stray.txt");
        std::fs::write(&loose, b"junk").unwrap();
        let report = recover_active_staging(tmp.path()).unwrap();
        assert_eq!(report.active_staging_orphans_swept, 1);
        assert!(!loose.exists());
    }

    #[test]
    fn recover_active_staging_unlinks_symlinks_without_following() {
        // A symlink pointing outside is unlinked, not followed.
        let tmp = tempfile::tempdir().unwrap();
        let staging = active_staging_dir(tmp.path());
        std::fs::create_dir_all(&staging).unwrap();
        let target_dir = tempfile::tempdir().unwrap();
        let target_file = target_dir.path().join("must-survive");
        std::fs::write(&target_file, b"outside-the-workspace").unwrap();
        let link_path = staging.join("11111111-2222-4333-8444-666666666666");
        #[cfg(unix)]
        std::os::unix::fs::symlink(target_dir.path(), &link_path).unwrap();
        #[cfg(not(unix))]
        return;
        #[allow(unreachable_code)]
        {
            let report = recover_active_staging(tmp.path()).unwrap();
            assert_eq!(report.active_staging_orphans_swept, 1);
            assert!(!link_path.exists(), "symlink must be unlinked");
            assert!(
                target_file.exists(),
                "symlink target outside staging must survive",
            );
        }
    }
}
