//! Active-head activation pipeline: [`stage_and_validate_activation`] stages +
//! hashes + validates into `active/.tmp/<id>/`, [`publish_active_generation`]
//! atomic-renames into `active/generations/<id>/` and rewrites `current.json`,
//! [`prune_old_generations`] retains only current + previous.
//!
//! Lock order: caller takes the global `active/` mutex first, then (Head origin
//! only) the per-workspace mutation mutex. The runtime candidate installs only
//! after publish succeeds, so on-disk state is durable before `HotHead` rotates.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::common::ids::{HeadId, WorkspaceId, default_runtime_head_id};
use crate::common::workspace::{
    ActiveHeadManifest, ActiveHeadValidationError, ActiveOrigin, HeadValidationError,
    WorkspaceRevision,
};
use crate::file_mgr::error::{FileError, io_err, metadata_parse_err};
use crate::file_mgr::fs_atomic::put_atomic;
use crate::file_mgr::schema::{
    ACTIVE_HEAD_FILENAME, ACTIVE_LABELS_FILENAME, ActiveCurrentPointer, active_current_path,
    active_dir, active_generation_dir, active_generations_dir, active_staging_dir,
    head_artifact_path, read_active_current, read_active_manifest, read_head_manifest,
    write_active_current,
};
use crate::file_mgr::validate::{fsync_dir, hex_lowercase};

/// Source file pair (not a directory) for a deployment-bundled default head, so
/// launch config owns the exact artifact paths.
#[derive(Clone, Copy, Debug)]
pub struct DefaultHeadSource<'a> {
    pub path: &'a Path,
    pub labels_path: &'a Path,
}

/// Origin descriptor for a pending activation; carries the sources staging copies.
#[derive(Clone, Debug)]
pub enum ActivationOriginInput<'a> {
    /// Workspace trained head at `<workspace_dir>/heads/<head_id>.{mpk,json}`.
    Head {
        workspace_dir: &'a Path,
        workspace_id: WorkspaceId,
        /// Becomes the activation's `runtime_head_id` for Head origin.
        head_id: HeadId,
    },
    Default {
        source: DefaultHeadSource<'a>,
    },
}

/// Inputs for [`stage_and_validate_activation`].
#[derive(Debug)]
pub struct PendingActivation<'a> {
    /// Daemon `WORKSPACE_ROOT`.
    pub root: &'a Path,
    pub origin_input: ActivationOriginInput<'a>,
    /// RFC3339 wall-clock for `manifest.activated_at`; caller-supplied so tests pin it.
    pub now_rfc3339: String,
}

/// Successful result of [`stage_and_validate_activation`]. The caller downcasts
/// `candidate` and installs it via
/// [`crate::common::traits::head_store::HeadStore::install_prevalidated`] only
/// AFTER [`publish_active_generation`] makes `current.json` durable.
pub struct ActivationResult {
    /// Validated manifest; `manifest.json` was written to staging.
    pub manifest: ActiveHeadManifest,
    /// Generation dir name (UUID-v4): staging `active/.tmp/<id>/`, published `active/generations/<id>/`.
    pub activation_id: String,
    /// Prevalidated runtime candidate (production `Box<inference::HeadInner>`);
    /// boxed `Any` keeps the primitive in `file_mgr` without inverting deps.
    pub candidate: Box<dyn std::any::Any + Send>,
}

impl std::fmt::Debug for ActivationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActivationResult")
            .field("manifest", &self.manifest)
            .field("activation_id", &self.activation_id)
            .field("candidate", &"Box<dyn Any + Send>")
            .finish()
    }
}

/// Failure shapes; mapped to HTTP statuses via [`crate::common::error::Categorized`].
#[derive(Debug, thiserror::Error)]
pub enum ActivationError {
    #[error("activation source not found: {what}")]
    NotFound { what: String },
    /// Staged bytes disagree with the source manifest `sha256`; refuse to publish.
    #[error("hash mismatch for {what}: expected {expected}, observed {observed}")]
    HashMismatch {
        what: String,
        expected: String,
        observed: String,
    },
    /// Constructed manifest rejected by validation; daemon-internal kind.
    #[error("active head manifest validation: {0}")]
    Validation(#[from] ActiveHeadValidationError),
    /// Source per-head manifest invalid; refuse to publish from corrupt metadata.
    #[error("source head manifest validation: {0}")]
    SourceManifestInvalid(#[from] HeadValidationError),
    #[error("file: {0}")]
    File(#[from] FileError),
    /// Runtime-candidate pre-load failed; the `inference` loader stringifies its error.
    #[error("preload candidate: {message}")]
    Preload { message: String },
}

impl crate::common::error::Categorized for ActivationError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            ActivationError::NotFound { .. } => NotFound,
            ActivationError::HashMismatch { .. } | ActivationError::Preload { .. } => UserInput,
            ActivationError::Validation(_) => Internal,
            ActivationError::SourceManifestInvalid(e) => e.kind(),
            ActivationError::File(e) => e.kind(),
        }
    }
}

/// Runtime-candidate factory `(staged_mpk, staged_labels, runtime_head_id) ->`
/// boxed impl-specific candidate; `String` errors keep `file_mgr` independent of
/// `inference`'s error type.
pub type HeadInnerLoader =
    dyn Fn(&Path, &Path, HeadId) -> Result<Box<dyn std::any::Any + Send>, String> + Send + Sync;

/// Stage the source into `<root>/active/.tmp/<activation_id>/`, hash, build +
/// validate the manifest, pre-load the runtime candidate, write `manifest.json`.
/// `head_inner_loader` runs sync (caller wraps in `spawn_blocking`); tests
/// substitute a synthetic candidate.
pub fn stage_and_validate_activation(
    pending: PendingActivation<'_>,
    head_inner_loader: &HeadInnerLoader,
) -> Result<ActivationResult, ActivationError> {
    let activation_id = uuid::Uuid::new_v4().hyphenated().to_string();
    let staging_root = active_staging_dir(pending.root);
    std::fs::create_dir_all(&staging_root)
        .map_err(|e| ActivationError::File(io_err(staging_root.display(), e)))?;
    let staging_dir = staging_root.join(&activation_id);
    if staging_dir.exists() {
        // UUID-v4 reuse is fatal, never overlaid onto a partial earlier generation.
        return Err(ActivationError::File(io_err(
            staging_dir.display(),
            std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "activation staging dir already exists",
            ),
        )));
    }
    std::fs::create_dir_all(&staging_dir)
        .map_err(|e| ActivationError::File(io_err(staging_dir.display(), e)))?;

    let staged = match &pending.origin_input {
        ActivationOriginInput::Head {
            workspace_dir,
            workspace_id,
            head_id,
        } => stage_head_origin(workspace_dir, *workspace_id, *head_id, &staging_dir)?,
        ActivationOriginInput::Default { source } => stage_default_origin(*source, &staging_dir)?,
    };

    // validate() below fails closed on origin-vs-runtime_head_id drift.
    let runtime_head_id = match &pending.origin_input {
        ActivationOriginInput::Head { head_id, .. } => *head_id,
        ActivationOriginInput::Default { .. } => default_runtime_head_id(),
    };
    let manifest = ActiveHeadManifest {
        origin: staged.origin,
        runtime_head_id,
        sha256: staged.head_sha256,
        labels_sha256: staged.labels_sha256,
        n_classes: staged.n_classes,
        labels: staged.labels,
        activated_at: pending.now_rfc3339,
    };
    manifest.validate()?;

    // Pre-load before any publish so a load failure refuses the activation.
    let head_mpk_staged = staging_dir.join(ACTIVE_HEAD_FILENAME);
    let labels_staged = staging_dir.join(ACTIVE_LABELS_FILENAME);
    let candidate = head_inner_loader(&head_mpk_staged, &labels_staged, runtime_head_id)
        .map_err(|message| ActivationError::Preload { message })?;

    // Write manifest.json into staging directly (the schema helper targets the
    // published path); the publish rename carries it along.
    let manifest_bytes = serde_json::to_vec(&manifest).map_err(FileError::MetadataSerialize)?;
    put_atomic(
        &staging_dir.join(crate::file_mgr::schema::ACTIVE_MANIFEST_FILENAME),
        &manifest_bytes,
    )?;

    Ok(ActivationResult {
        manifest,
        activation_id,
        candidate,
    })
}

struct StagedSource {
    origin: ActiveOrigin,
    head_sha256: String,
    labels_sha256: String,
    n_classes: u32,
    labels: Vec<String>,
}

/// Stage a Head-origin activation: copy the trained `.mpk`, render `labels.txt`
/// from the manifest `labels[]`, verify the mpk hash against the recorded
/// `sha256`, emit the provenance triple.
fn stage_head_origin(
    workspace_dir: &Path,
    workspace_id: WorkspaceId,
    head_id: HeadId,
    staging_dir: &Path,
) -> Result<StagedSource, ActivationError> {
    let manifest = read_head_manifest(workspace_dir, head_id).map_err(|e| match e {
        FileError::Io { ref source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            ActivationError::NotFound {
                what: format!("head {head_id} in workspace {workspace_id}"),
            }
        }
        other => ActivationError::File(other),
    })?;
    // Fail closed on a tampered manifest before any heavy IO.
    manifest.validate()?;

    let mpk_path = head_artifact_path(workspace_dir, head_id);
    if !mpk_path.is_file() {
        return Err(ActivationError::NotFound {
            what: format!("head {head_id} mpk in workspace {workspace_id}"),
        });
    }

    // Stream-copy+hash in one pass: avoids the ~80 MiB heap a full fs::read would
    // need at the convert cap. Partial staging bytes get swept by boot recovery.
    let staged_mpk = staging_dir.join(ACTIVE_HEAD_FILENAME);
    let head_sha = copy_and_hash(&mpk_path, &staged_mpk).map_err(|e| match e {
        FileError::Io { ref source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            ActivationError::NotFound {
                what: format!("head {head_id} mpk in workspace {workspace_id}"),
            }
        }
        other => ActivationError::File(other),
    })?;
    if head_sha != manifest.sha256 {
        return Err(ActivationError::HashMismatch {
            what: format!("head {head_id} mpk"),
            expected: manifest.sha256.clone(),
            observed: head_sha,
        });
    }

    // Render from manifest `labels[]` (not a re-read) so the generation's labels
    // file is the single canonical source, insulated from later edits.
    let labels_text = labels_to_text(&manifest.labels);
    let labels_bytes = labels_text.as_bytes().to_vec();
    let labels_sha = sha256_hex_of(&labels_bytes);
    put_atomic(&staging_dir.join(ACTIVE_LABELS_FILENAME), &labels_bytes)?;

    let n_classes = u32::try_from(manifest.labels.len()).map_err(|_| {
        ActivationError::File(metadata_parse_err(
            staging_dir.display(),
            serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "labels count {} overflows u32 in workspace {workspace_id} head {head_id}",
                    manifest.labels.len()
                ),
            )),
        ))
    })?;

    Ok(StagedSource {
        origin: ActiveOrigin::Head {
            source_workspace_id: workspace_id,
            source_head_id: head_id,
            workspace_revision: WorkspaceRevision {
                id: manifest.workspace_revision.id,
                at: manifest.workspace_revision.at.clone(),
            },
        },
        head_sha256: head_sha,
        labels_sha256: labels_sha,
        n_classes,
        labels: manifest.labels,
    })
}

/// Stage a Default-origin activation: copy bundled `head.mpk`, parse bundled
/// `labels.txt`, re-render to canonical `labels.join("\n")` BEFORE hashing.
fn stage_default_origin(
    source: DefaultHeadSource<'_>,
    staging_dir: &Path,
) -> Result<StagedSource, ActivationError> {
    let src_mpk = source.path;
    let src_labels = source.labels_path;
    let mpk_bytes = std::fs::read(src_mpk).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            ActivationError::NotFound {
                what: format!("default head at {}", src_mpk.display()),
            }
        } else {
            ActivationError::File(io_err(src_mpk.display(), e))
        }
    })?;
    let labels_bytes = std::fs::read(src_labels).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            ActivationError::NotFound {
                what: format!("default head labels at {}", src_labels.display()),
            }
        } else {
            ActivationError::File(io_err(src_labels.display(), e))
        }
    })?;

    let labels_text = std::str::from_utf8(&labels_bytes).map_err(|e| {
        ActivationError::File(io_err(
            src_labels.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("labels utf-8: {e}"),
            ),
        ))
    })?;
    let labels: Vec<String> = labels_text
        .lines()
        .map(|s| s.trim_end_matches('\r'))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    if labels.is_empty() {
        return Err(ActivationError::File(io_err(
            src_labels.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bundled labels.txt is empty",
            ),
        )));
    }
    let n_classes = u32::try_from(labels.len()).map_err(|_| {
        ActivationError::File(io_err(
            src_labels.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("labels count {} overflows u32", labels.len()),
            ),
        ))
    })?;

    let head_sha = sha256_hex_of(&mpk_bytes);
    // Hash the canonical labels.join("\n") form, not raw bundled bytes: boot
    // recovery regenerates labels.txt from `labels[]`, so a trailing newline /
    // CRLF / blank line would fail the labels_sha256 recheck every boot.
    let canonical_labels_text = labels_to_text(&labels);
    let canonical_labels_bytes = canonical_labels_text.as_bytes().to_vec();
    let labels_sha = sha256_hex_of(&canonical_labels_bytes);

    put_atomic(&staging_dir.join(ACTIVE_HEAD_FILENAME), &mpk_bytes)?;
    put_atomic(
        &staging_dir.join(ACTIVE_LABELS_FILENAME),
        &canonical_labels_bytes,
    )?;

    Ok(StagedSource {
        origin: ActiveOrigin::Default,
        head_sha256: head_sha,
        labels_sha256: labels_sha,
        n_classes,
        labels,
    })
}

/// Classification of the post-`write_active_current` failure branch of
/// [`publish_active_generation`], split out so the 3-way decision is unit-testable
/// without a fault-injection seam in `put_atomic`.
#[derive(Debug)]
enum PublishScenario {
    /// Readback shows the OLD pointer (or NotFound): rename never committed; safe
    /// to rollback `final_dir -> staging`.
    A,
    /// Readback shows our `activation_id`: rename committed, only post-rename fsync
    /// failed -- published with degraded durability.
    B,
    /// Readback itself failed (non-NotFound): state unobservable, so refuse to
    /// rollback (doing so over an actual scenario B would strand a phantom pointer);
    /// carries the read failure for the operator log.
    Ambiguous(FileError),
}

fn classify_publish_failure(
    readback: Result<ActiveCurrentPointer, FileError>,
    activation_id: &str,
) -> PublishScenario {
    match readback {
        Ok(p) if p.activation_id == activation_id => PublishScenario::B,
        Ok(_) => PublishScenario::A,
        Err(FileError::Io { ref source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            PublishScenario::A
        }
        Err(read_err) => PublishScenario::Ambiguous(read_err),
    }
}

/// Atomic-rename the staged generation into `active/generations/<id>/`, fsync,
/// rewrite `active/current.json`, fsync `active/`. MUST run AFTER
/// [`stage_and_validate_activation`] Ok and BEFORE installing the runtime candidate.
pub fn publish_active_generation(
    root: &Path,
    staging: &Path,
    _manifest: &ActiveHeadManifest,
    activation_id: &str,
) -> Result<(), ActivationError> {
    let generations_root = active_generations_dir(root);
    let staging_root = active_staging_dir(root);
    std::fs::create_dir_all(&generations_root)
        .map_err(|e| ActivationError::File(io_err(generations_root.display(), e)))?;
    let final_dir = active_generation_dir(root, activation_id);
    std::fs::rename(staging, &final_dir)
        .map_err(|e| ActivationError::File(io_err(final_dir.display(), e)))?;
    // fsync BOTH rename parents (source `active/.tmp/`, dest
    // `active/generations/`): without the source-side fsync, power loss after the
    // rename but before `.tmp/` metadata flushes leaves a stale staging dirent
    // post-remount (self-heals via recover_active_staging).
    fsync_dir(&generations_root)
        .map_err(|e| ActivationError::File(io_err(generations_root.display(), e)))?;
    if let Err(e) = fsync_dir(&staging_root) {
        // Warn not Err: destination is durable and recovery handles the leftover,
        // so propagating would report failure on a successful publish. No
        // `.exists()` pre-check (false on transient EACCES/EIO hides the gap);
        // active_mutex precludes concurrent unlink of staging_root.
        tracing::warn!(
            target: "file_mgr",
            err = %e,
            path = %staging_root.display(),
            "publish_active_generation: source-parent fsync failed; \
             stale .tmp/ dirent may persist across crash until \
             recover_active_staging sweeps",
        );
    }
    // On write_active_current failure the rename above already moved staging ->
    // final_dir, leaving one of three on-disk states classified from a
    // current.json readback ([`PublishScenario`]); recovery per match arm below.
    if let Err(write_err) = write_active_current(
        root,
        &ActiveCurrentPointer {
            activation_id: activation_id.to_string(),
        },
    ) {
        let scenario = classify_publish_failure(
            crate::file_mgr::schema::read_active_current(root),
            activation_id,
        );
        match scenario {
            PublishScenario::B => {
                // current.json already points at us, so blind rollback would strand
                // a phantom pointer: re-fsync `active/` and return Ok either way
                // (rename committed). Error vs warn distinguishes re-fsync Err
                // (pointer may revert across power loss until the next durability
                // barrier) from Ok (recovered in-band).
                if let Err(refsync_err) = fsync_dir(&active_dir(root)) {
                    tracing::error!(
                        target: "file_mgr",
                        inner_err = %write_err,
                        refsync_err = %refsync_err,
                        activation_id = %activation_id,
                        "publish_active_generation: write_active_current returned Err but \
                         current.json already reflects the new pointer; re-fsync of active/ \
                         also failed -- the new pointer may revert across power loss until \
                         the next durability barrier",
                    );
                } else {
                    tracing::warn!(
                        target: "file_mgr",
                        inner_err = %write_err,
                        activation_id = %activation_id,
                        "publish_active_generation: write_active_current returned Err but \
                         current.json already reflects the new pointer; re-fsync of active/ \
                         succeeded -- treating as published",
                    );
                }
                return Ok(());
            }
            PublishScenario::Ambiguous(read_err) => {
                // Refuse to rollback (see PublishScenario::Ambiguous); orphan
                // persists until prune_old_generations evicts it.
                tracing::error!(
                    target: "file_mgr",
                    inner_err = %write_err,
                    read_err = %read_err,
                    activation_id = %activation_id,
                    orphan_path = %final_dir.display(),
                    "publish_active_generation: write_active_current returned Err AND \
                     readback of current.json also failed; refusing to rollback.  \
                     Operator: inspect <root>/active/current.json + generations/<id>/ -- \
                     either re-fsync and complete the publish manually, or remove the \
                     orphan dir if current.json still points at the prior generation",
                );
                return Err(ActivationError::File(write_err));
            }
            PublishScenario::A => {}
        }
        match std::fs::rename(&final_dir, staging) {
            Ok(()) => {
                // Log before the fsync diagnostics so the rollback outcome leads.
                tracing::warn!(
                    target: "file_mgr",
                    activation_id = %activation_id,
                    orphan_path = %final_dir.display(),
                    staging_path = %staging.display(),
                    "publish_active_generation: write_active_current failed; \
                     rolled the generation back to staging for recover_active_staging \
                     to sweep on next boot",
                );
                // fsync both rollback-rename parents (staging_root gained the
                // dirent, generations_root lost it) so a crash doesn't leave the
                // entry observable in `generations/`. Warn not Err (we already
                // return write_err; a fsync Err would only mask the root cause).
                if let Err(e) = fsync_dir(&staging_root) {
                    tracing::warn!(
                        target: "file_mgr",
                        err = %e,
                        path = %staging_root.display(),
                        "publish_active_generation: post-rollback fsync of staging root failed",
                    );
                }
                if let Err(e) = fsync_dir(&generations_root) {
                    tracing::warn!(
                        target: "file_mgr",
                        err = %e,
                        path = %generations_root.display(),
                        "publish_active_generation: post-rollback fsync of generations root failed",
                    );
                }
            }
            Err(rollback_err) => {
                tracing::error!(
                    target: "file_mgr",
                    activation_id = %activation_id,
                    orphan_path = %final_dir.display(),
                    rollback_err = %rollback_err,
                    "publish_active_generation: write_active_current failed AND roll-back rename failed; \
                     orphan generation directory will persist under generations/ until prune_old_generations \
                     evicts it on a future activation",
                );
            }
        }
        return Err(ActivationError::File(write_err));
    }
    // Redundantly fsync `active/`: put_atomic fsyncs it as current.json's parent
    // today, but if current.json ever moves into a subdir put_atomic would fsync
    // only that subdir, leaving the new `active/` dirent unflushed. No `.exists()`
    // pre-check (false on transient EACCES/EIO); the debug_assert below pins the
    // invariant this relies on.
    let active_root = active_dir(root);
    if let Err(e) = fsync_dir(&active_root) {
        tracing::warn!(
            target: "file_mgr",
            err = %e,
            path = %active_root.display(),
            "publish_active_generation: fsync of active/ failed after current.json rewrite; \
             rename durability may degrade to the kernel writeback window",
        );
    }
    debug_assert_eq!(
        active_current_path(root).parent(),
        Some(active_dir(root).as_path()),
        "active_current_path must live directly under active_dir; \
         current.json publish relies on put_atomic's parent fsync \
         to make the rename durable inside `active/`",
    );
    Ok(())
}

/// Remove one stale generation entry: `remove_dir_all` for a dir, else the
/// non-dir orphan sweep; returns whether a removal happened. Caller resolves
/// `is_dir` so its branch's stat choice survives (no-follow `file_type()` for
/// non-UTF-8 names, follow-symlink `metadata()` for the keep-list path).
fn remove_generation_entry(path: &Path, is_dir: bool) -> Result<bool, ActivationError> {
    if is_dir {
        std::fs::remove_dir_all(path)
            .map_err(|e| ActivationError::File(io_err(path.display(), e)))?;
        Ok(true)
    } else {
        Ok(sweep_non_dir_orphan(path))
    }
}

/// Unlink a non-dir orphan under `generations/` (the daemon never creates non-dir
/// entries there; survivors are operator-placed or OS residue). `remove_file` is
/// lstat-semantic so symlinks unlink the link itself. Returns whether the unlink
/// succeeded; on failure logs and returns false so the prune loop still reaches
/// its post-loop fsync.
fn sweep_non_dir_orphan(path: &Path) -> bool {
    match std::fs::remove_file(path) {
        Ok(()) => {
            tracing::warn!(
                target: "file_mgr",
                path = %path.display(),
                "prune_old_generations: reaped non-daemon-managed entry under generations/ \
                 (operator-placed file or OS residue such as NFS .nfs* / fsck-rescued node)",
            );
            true
        }
        Err(e) => {
            tracing::warn!(
                target: "file_mgr",
                err = %e,
                path = %path.display(),
                "prune_old_generations: failed to unlink non-dir orphan; \
                 leaving in place and continuing prune",
            );
            false
        }
    }
}

/// Retain only generations named in `keep`; remove every other entry under
/// `active/generations/`, returning the removed count.
///
/// **Caller MUST hold `api::AppState::active_mutex`** across the whole `read_dir`
/// -> `remove_dir_all` window, else a concurrent publish lands a generation
/// outside `keep` and it gets deleted, leaving `current.json` dangling.
pub fn prune_old_generations<S: AsRef<str>>(
    root: &Path,
    keep: &[S],
) -> Result<usize, ActivationError> {
    let generations_root = active_generations_dir(root);
    // read_dir (not Path::exists) so a transient stat error isn't swallowed as
    // "nothing to prune".
    let entries = match std::fs::read_dir(&generations_root) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(ActivationError::File(io_err(generations_root.display(), e))),
    };
    let mut removed = 0usize;
    // Capture the loop Err to an outer Result so the post-loop fsync still runs
    // for already-removed entries; a mid-loop `?` would skip it, leaving prior
    // unlinks at the writeback window where power loss could resurrect them as
    // promote_or_default candidates.
    let loop_result: Result<(), ActivationError> = (|| {
        for entry in entries {
            let entry =
                entry.map_err(|e| ActivationError::File(io_err(generations_root.display(), e)))?;
            let name = match entry.file_name().into_string() {
                Ok(s) => s,
                // Non-UTF-8 names can't match a UUID-v4 keep list; sweep as orphan
                // residue (same dir-vs-non-dir split as the keep path).
                Err(_) => {
                    let path = entry.path();
                    let is_dir = entry
                        .file_type()
                        .map_err(|e| ActivationError::File(io_err(path.display(), e)))?
                        .is_dir();
                    if remove_generation_entry(&path, is_dir)? {
                        removed += 1;
                    }
                    continue;
                }
            };
            if keep.iter().any(|k| k.as_ref() == name) {
                continue;
            }
            let path = entry.path();
            let is_dir = entry
                .metadata()
                .map_err(|e| ActivationError::File(io_err(path.display(), e)))?
                .is_dir();
            if remove_generation_entry(&path, is_dir)? {
                removed += 1;
            }
        }
        Ok(())
    })();
    // Fsync regardless of loop_result: successful unlinks must reach stable storage
    // before we propagate a later Err.
    if removed > 0
        && let Err(e) = fsync_dir(&generations_root)
    {
        tracing::warn!(
            target: "file_mgr",
            err = %e,
            path = %generations_root.display(),
            "fsync generations/ after prune failed; \
             evicted generations may resurrect post-power-loss",
        );
    }
    // Surface the partial-progress count before propagating the loop Err: the
    // caller sees only the Err and would otherwise lose that `removed` entries
    // were unlinked first.
    if removed > 0 && loop_result.is_err() {
        tracing::warn!(
            target: "file_mgr",
            removed,
            "prune_old_generations: mid-iteration Err after partial success; \
             {removed} entries were unlinked before the failure \
             (durability reported by the preceding fsync_dir warn, if any)",
        );
    }
    loop_result?;
    Ok(removed)
}

/// Staging directory path for an in-flight activation.
pub fn staging_path_for(root: &Path, activation_id: &str) -> PathBuf {
    active_staging_dir(root).join(activation_id)
}

/// Source head id of the current active generation scoped to `workspace_id`:
/// `Some` iff the published manifest is [`ActiveOrigin::Head`] with a matching
/// `source_workspace_id`, else `None` (any IO/parse failure included --
/// best-effort, never errors). Read-only, takes no lock itself.
///
/// **Caller contract:** to serialise against `POST /active`, hold
/// `api::AppState::active_mutex`; without it a concurrent publish between this
/// read and the caller's mutation can drop the workspace `heads.json` entry the
/// active manifest references, surfacing as a phantom `source_head_id` on
/// `GET /active`. `FsService` trait entry points take the lock internally.
pub fn active_source_head_in_workspace(root: &Path, workspace_id: WorkspaceId) -> Option<HeadId> {
    let pointer = read_active_current(root).ok()?;
    let manifest = read_active_manifest(root, &pointer.activation_id).ok()?;
    match manifest.origin {
        ActiveOrigin::Head {
            source_workspace_id,
            source_head_id,
            ..
        } if source_workspace_id == workspace_id => Some(source_head_id),
        _ => None,
    }
}

/// Canonical active-head `labels.txt` body: `labels.join("\n")`, UTF-8, NO
/// trailing newline. `pub(crate)` so recovery's label regeneration shares the
/// EXACT shape -- any divergence (trailing newline, NFC) breaks the
/// `labels_sha256` recheck for every Default-origin activation. Control chars
/// (which render as extra labels) are gated in release by `HeadManifest::validate`
/// for the Head origin (run before this) and by [`ActiveHeadManifest::validate`]
/// for the Default origin (its bundled labels skip the per-head manifest); the
/// debug_assert catches an unvalidated in-memory manifest.
pub(crate) fn labels_to_text(labels: &[String]) -> String {
    debug_assert!(
        !labels.iter().any(|l| l.chars().any(char::is_control)),
        "labels_to_text: a label contains a control character; breaks the \
         labels.txt render and manifest.labels_sha256 recheck.  Callers must \
         invoke HeadManifest::validate / ActiveHeadManifest::validate first",
    );
    labels.join("\n")
}

fn sha256_hex_of(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_lowercase(&hasher.finalize())
}

/// Stream-copy `src` to `dst` and SHA-256 in one pass (64 KiB buffer, constant
/// memory), returning the lowercase-hex digest. `dst.parent()` must exist; `dst`
/// is `sync_all`'d but the parent dir fsync is the caller's.
fn copy_and_hash(src: &Path, dst: &Path) -> Result<String, FileError> {
    use std::io::{Read, Write};
    let mut reader = std::fs::File::open(src).map_err(|source| io_err(src.display(), source))?;
    let mut writer = std::fs::File::create(dst).map_err(|source| io_err(dst.display(), source))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|source| io_err(src.display(), source))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        writer
            .write_all(&buf[..n])
            .map_err(|source| io_err(dst.display(), source))?;
    }
    writer
        .sync_all()
        .map_err(|source| io_err(dst.display(), source))?;
    Ok(hex_lowercase(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)]

    use super::*;
    use crate::common::ids::{HeadId, WorkspaceId};
    use crate::common::workspace::{HeadIndex, HeadManifest, HeadRecord, WorkspaceCore};
    use crate::file_mgr::schema::{write_head_index, write_head_manifest, write_workspace_core};

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn synth_head_manifest(head_id: HeadId, mpk_bytes: &[u8]) -> HeadManifest {
        HeadManifest {
            head_id,
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: sha256_hex_of(mpk_bytes),
            n_classes: 2,
            size_bytes: mpk_bytes.len() as u64,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["alpha".to_string(), "beta".to_string()],
        }
    }

    fn synth_workspace_core() -> WorkspaceCore {
        WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(5),
            head_count: 1,
        }
    }

    fn fresh_workspace_with_head(root: &Path, mpk_bytes: &[u8]) -> (PathBuf, HeadId) {
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(root, &ws_id());
        std::fs::create_dir_all(&ws_dir).unwrap();
        std::fs::create_dir_all(crate::file_mgr::schema::heads_dir(&ws_dir)).unwrap();
        write_workspace_core(&ws_dir, &synth_workspace_core()).unwrap();
        let mut idx = HeadIndex::default();
        let head_id = HeadId::parse("11111111-2222-4333-8444-555555555556").unwrap();
        let manifest = synth_head_manifest(head_id, mpk_bytes);
        idx.heads.push(HeadRecord {
            head_id,
            workspace_revision: manifest.workspace_revision.clone(),
            sha256: manifest.sha256.clone(),
            n_classes: manifest.n_classes,
            size_bytes: manifest.size_bytes,
            created_at: manifest.created_at.clone(),
        });
        write_head_index(&ws_dir, &idx).unwrap();
        write_head_manifest(&ws_dir, &manifest).unwrap();
        std::fs::write(
            crate::file_mgr::schema::head_artifact_path(&ws_dir, head_id),
            mpk_bytes,
        )
        .unwrap();
        (ws_dir, head_id)
    }

    /// Returns a `()` candidate so the pipeline runs without `inference`.
    fn synth_loader_ok() -> Box<HeadInnerLoader> {
        Box::new(|_mpk: &Path, _labels: &Path, _id: HeadId| {
            Ok(Box::new(()) as Box<dyn std::any::Any + Send>)
        })
    }

    /// Always-fail loader, exercising the `Preload` path.
    fn synth_loader_fail() -> Box<HeadInnerLoader> {
        Box::new(|_mpk: &Path, _labels: &Path, _id: HeadId| {
            Err("synthetic preload failure".to_string())
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

    fn default_origin<'a>(path: &'a Path, labels_path: &'a Path) -> ActivationOriginInput<'a> {
        ActivationOriginInput::Default {
            source: DefaultHeadSource { path, labels_path },
        }
    }

    #[test]
    fn head_origin_stages_and_validates() {
        let tmp = tempfile::tempdir().unwrap();
        let mpk = b"MPK-CONTENT-AAA";
        let (ws_dir, head_id) = fresh_workspace_with_head(tmp.path(), mpk);
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: ActivationOriginInput::Head {
                workspace_dir: &ws_dir,
                workspace_id: ws_id(),
                head_id,
            },
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();

        match &result.manifest.origin {
            ActiveOrigin::Head {
                source_workspace_id,
                source_head_id,
                ..
            } => {
                assert_eq!(*source_workspace_id, ws_id());
                assert_eq!(*source_head_id, head_id);
            }
            other => panic!("expected Head origin, got {other:?}"),
        }
        assert_eq!(result.manifest.runtime_head_id, head_id);
        assert_eq!(result.manifest.n_classes, 2);
        assert_eq!(
            result.manifest.labels,
            vec!["alpha".to_string(), "beta".to_string()]
        );

        let staging = staging_path_for(tmp.path(), &result.activation_id);
        assert!(staging.join(ACTIVE_HEAD_FILENAME).is_file());
        assert!(staging.join(ACTIVE_LABELS_FILENAME).is_file());
        assert!(
            staging
                .join(crate::file_mgr::schema::ACTIVE_MANIFEST_FILENAME)
                .is_file()
        );

        let labels_disk = std::fs::read_to_string(staging.join(ACTIVE_LABELS_FILENAME)).unwrap();
        assert_eq!(labels_disk, "alpha\nbeta");
    }

    #[test]
    fn head_origin_rejects_hash_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let mpk = b"MPK-CONTENT-BBB";
        let (ws_dir, head_id) = fresh_workspace_with_head(tmp.path(), mpk);

        // Tamper after sha256 was recorded -> HashMismatch.
        std::fs::write(
            crate::file_mgr::schema::head_artifact_path(&ws_dir, head_id),
            b"TAMPERED",
        )
        .unwrap();

        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: ActivationOriginInput::Head {
                workspace_dir: &ws_dir,
                workspace_id: ws_id(),
                head_id,
            },
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let err = stage_and_validate_activation(pending, &*synth_loader_ok())
            .expect_err("hash mismatch must reject");
        assert!(matches!(err, ActivationError::HashMismatch { .. }));
    }

    #[test]
    fn head_origin_missing_head_id_surfaces_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let ws_dir = crate::file_mgr::schema::workspace_dir_for(tmp.path(), &ws_id());
        std::fs::create_dir_all(&ws_dir).unwrap();
        std::fs::create_dir_all(crate::file_mgr::schema::heads_dir(&ws_dir)).unwrap();
        write_workspace_core(&ws_dir, &synth_workspace_core()).unwrap();
        write_head_index(&ws_dir, &HeadIndex::default()).unwrap();
        let unknown = HeadId::parse("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee").unwrap();
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: ActivationOriginInput::Head {
                workspace_dir: &ws_dir,
                workspace_id: ws_id(),
                head_id: unknown,
            },
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let err = stage_and_validate_activation(pending, &*synth_loader_ok())
            .expect_err("missing head id must reject");
        assert!(matches!(err, ActivationError::NotFound { .. }));
    }

    #[test]
    fn default_origin_stages_and_validates() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels) = fresh_bundled_default(tmp.path(), b"DEFAULT-MPK", "cat\ndog\nbird\n");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: default_origin(&head, &labels),
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();
        assert!(matches!(result.manifest.origin, ActiveOrigin::Default));
        assert_eq!(result.manifest.runtime_head_id, default_runtime_head_id());
        assert_eq!(result.manifest.n_classes, 3);
        assert_eq!(
            result.manifest.labels,
            vec!["cat".to_string(), "dog".to_string(), "bird".to_string()]
        );

        let staging = staging_path_for(tmp.path(), &result.activation_id);
        assert!(staging.join(ACTIVE_HEAD_FILENAME).is_file());
        assert!(staging.join(ACTIVE_LABELS_FILENAME).is_file());
    }

    #[test]
    fn default_origin_missing_fixture_surfaces_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let missing_head = tmp.path().join("does_not_exist.mpk");
        let missing_labels = tmp.path().join("does_not_exist.labels.txt");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: default_origin(&missing_head, &missing_labels),
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let err = stage_and_validate_activation(pending, &*synth_loader_ok())
            .expect_err("missing fixture must reject");
        assert!(matches!(err, ActivationError::NotFound { .. }));
    }

    #[test]
    fn preload_failure_propagates() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels) = fresh_bundled_default(tmp.path(), b"DEFAULT-MPK", "cat\n");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: default_origin(&head, &labels),
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let err = stage_and_validate_activation(pending, &*synth_loader_fail())
            .expect_err("preload failure must reject");
        assert!(matches!(err, ActivationError::Preload { .. }));
    }

    #[test]
    fn publish_renames_and_writes_pointer() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels) = fresh_bundled_default(tmp.path(), b"MPK", "x\n");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: default_origin(&head, &labels),
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();
        let staging = staging_path_for(tmp.path(), &result.activation_id);

        publish_active_generation(
            tmp.path(),
            &staging,
            &result.manifest,
            &result.activation_id,
        )
        .unwrap();

        assert!(active_generation_dir(tmp.path(), &result.activation_id).is_dir());
        let pointer = crate::file_mgr::schema::read_active_current(tmp.path()).unwrap();
        assert_eq!(pointer.activation_id, result.activation_id);
        assert!(!staging.exists());
    }

    fn synth_io_err(kind: std::io::ErrorKind) -> FileError {
        io_err("test/path", std::io::Error::new(kind, "synthetic"))
    }

    #[test]
    fn classify_matching_pointer_is_b() {
        let id = "abc-123";
        let readback = Ok(ActiveCurrentPointer {
            activation_id: id.to_string(),
        });
        assert!(matches!(
            classify_publish_failure(readback, id),
            PublishScenario::B
        ));
    }

    #[test]
    fn classify_non_matching_pointer_is_a() {
        let readback = Ok(ActiveCurrentPointer {
            activation_id: "prior-pointer".to_string(),
        });
        assert!(matches!(
            classify_publish_failure(readback, "our-activation"),
            PublishScenario::A
        ));
    }

    #[test]
    fn classify_notfound_readback_is_a() {
        let readback = Err(synth_io_err(std::io::ErrorKind::NotFound));
        assert!(matches!(
            classify_publish_failure(readback, "any"),
            PublishScenario::A
        ));
    }

    #[test]
    fn classify_other_io_err_is_ambiguous() {
        // Transient readback EIO could hide a real scenario B -> Ambiguous, not A.
        let readback = Err(synth_io_err(std::io::ErrorKind::PermissionDenied));
        let scenario = classify_publish_failure(readback, "any");
        let preserved_err = match scenario {
            PublishScenario::Ambiguous(e) => e,
            other => panic!("expected Ambiguous; got {other:?}"),
        };
        // Original FileError flows through unchanged for the operator log.
        assert!(matches!(
            preserved_err,
            FileError::Io { ref source, .. }
                if source.kind() == std::io::ErrorKind::PermissionDenied
        ));
    }

    #[test]
    fn classify_metadata_parse_err_is_ambiguous() {
        // current.json exists but won't parse (external corruption) -> Ambiguous.
        let serde_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let readback = Err(metadata_parse_err("test/path", serde_err));
        assert!(matches!(
            classify_publish_failure(readback, "any"),
            PublishScenario::Ambiguous(FileError::MetadataParse { .. })
        ));
    }

    #[test]
    fn prune_keeps_only_listed_generations() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels) = fresh_bundled_default(tmp.path(), b"MPK", "x\n");
        let mut ids = Vec::new();
        for _ in 0..3 {
            let pending = PendingActivation {
                root: tmp.path(),
                origin_input: default_origin(&head, &labels),
                now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
            };
            let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();
            let staging = staging_path_for(tmp.path(), &result.activation_id);
            publish_active_generation(
                tmp.path(),
                &staging,
                &result.manifest,
                &result.activation_id,
            )
            .unwrap();
            ids.push(result.activation_id);
        }
        let keep = vec![ids[1].clone(), ids[2].clone()];
        let removed = prune_old_generations(tmp.path(), &keep).unwrap();
        assert_eq!(removed, 1);
        assert!(!active_generation_dir(tmp.path(), &ids[0]).exists());
        assert!(active_generation_dir(tmp.path(), &ids[1]).is_dir());
        assert!(active_generation_dir(tmp.path(), &ids[2]).is_dir());
    }

    #[test]
    fn prune_on_missing_generations_dir_is_ok() {
        let tmp = tempfile::tempdir().unwrap();
        let removed = prune_old_generations::<&str>(tmp.path(), &[]).unwrap();
        assert_eq!(removed, 0);
    }

    /// The UTF-8 no-keep branch reaps non-dir orphans (UUID- and non-UUID-named)
    /// while sparing the kept generation dir.
    #[test]
    fn prune_sweeps_non_dir_orphans_alongside_generations() {
        let tmp = tempfile::tempdir().unwrap();
        let (head, labels) = fresh_bundled_default(tmp.path(), b"MPK", "x\n");
        // Real generation so the gen-root exists.
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: default_origin(&head, &labels),
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();
        let staging = staging_path_for(tmp.path(), &result.activation_id);
        publish_active_generation(
            tmp.path(),
            &staging,
            &result.manifest,
            &result.activation_id,
        )
        .unwrap();
        let gens_root = active_generations_dir(tmp.path());
        // Both a non-UUID- and a UUID-named regular-file orphan must be swept
        // regardless of name shape.
        let notes = gens_root.join("NOTES.md");
        let rescued = gens_root.join("00000000-0000-0000-0000-000000000000");
        std::fs::write(&notes, b"operator placed").unwrap();
        std::fs::write(&rescued, b"fsck residue").unwrap();
        let keep = vec![result.activation_id.clone()];
        let removed = prune_old_generations(tmp.path(), &keep).unwrap();
        assert_eq!(removed, 2, "both non-dir orphans must be swept");
        assert!(!notes.exists(), "NOTES.md must be unlinked");
        assert!(
            !rescued.exists(),
            "UUID-named regular file must be unlinked"
        );
        assert!(
            active_generation_dir(tmp.path(), &result.activation_id).is_dir(),
            "the real kept generation must survive",
        );
    }

    /// Pins the non-UTF-8 `into_string` Err arm: a non-dir orphan with a non-UTF-8
    /// filename is reaped. Linux-only (macOS rejects non-UTF-8 dirents).
    #[cfg(target_os = "linux")]
    #[test]
    fn prune_sweeps_non_utf8_orphans() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let tmp = tempfile::tempdir().unwrap();
        let (head, labels) = fresh_bundled_default(tmp.path(), b"MPK", "x\n");
        let pending = PendingActivation {
            root: tmp.path(),
            origin_input: default_origin(&head, &labels),
            now_rfc3339: "2026-05-07T12:34:56Z".to_string(),
        };
        let result = stage_and_validate_activation(pending, &*synth_loader_ok()).unwrap();
        let staging = staging_path_for(tmp.path(), &result.activation_id);
        publish_active_generation(
            tmp.path(),
            &staging,
            &result.manifest,
            &result.activation_id,
        )
        .unwrap();
        let gens_root = active_generations_dir(tmp.path());
        // 0xff is an invalid UTF-8 leading byte, so into_string() Errs.
        let non_utf8 = gens_root.join(OsString::from_vec(vec![0xff]));
        std::fs::write(&non_utf8, b"non-utf8 residue").unwrap();
        let keep = vec![result.activation_id.clone()];
        let removed = prune_old_generations(tmp.path(), &keep).unwrap();
        assert_eq!(removed, 1, "non-UTF-8 non-dir orphan must be swept");
        assert!(!non_utf8.exists(), "non-UTF-8 file must be unlinked");
    }

    #[test]
    fn labels_to_text_is_newline_joined() {
        assert_eq!(labels_to_text(&[]), "");
        assert_eq!(labels_to_text(&["a".into()]), "a");
        assert_eq!(
            labels_to_text(&["a".into(), "b".into(), "c".into()]),
            "a\nb\nc"
        );
    }
}
