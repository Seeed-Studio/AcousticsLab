//! Workspace asset-tree surface: `AssetPath`-shaped methods on [`WorkspaceMgr`]
//! backing `GET/PUT/DELETE /assets/{*path}`. Mutations hold the per-workspace
//! mutation mutex across conflict-check, revision-before-bytes, atomic commit,
//! and cache publish; reads never take it. Conflict admission consumes
//! [`crate::file_mgr::job_registry::JobRegistry`] -> HTTP 409 `JobConflict`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::common::asset_path::AssetPath;
use crate::common::ids::{JobId, WorkspaceId};
use crate::common::workspace::{JobReference, JobType, WorkspaceCore, WorkspaceRevision};
use crate::file_mgr::WorkspaceMgr;
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::job_registry::{JobHandle, RegistryConflict};
use crate::file_mgr::schema::{workspace_core_path, write_workspace_core};
use crate::file_mgr::staging::{
    DEFAULT_DELETE_BATCH_ENTRIES, DeleteTombstone, DrainResult, StagedDelete, drain_staged_payload,
    finalize_staged_delete, stage_payload, write_tombstone,
};
use crate::file_mgr::time_util::{now_rfc3339, rfc3339_from_result};
use crate::file_mgr::validate::fsync_dir;

pub const DATASETS_DIR_NAME: &str = "datasets";

pub const CONVERTERS_DIR_NAME: &str = "converters";

/// Trainer's per-job JSONL log backstop; daemon-only producer. Deletes 409
/// while a Train job for the workspace runs.
pub const TRAINING_LOGS_DIR_NAME: &str = "training_logs";

/// Converter's per-job JSONL log backstop; daemon-only producer.
pub const CONVERTER_LOGS_DIR_NAME: &str = "converter_logs";

/// Which workspace tree an asset path targets. Drives the tombstone variant +
/// `JobType` for async deletes, and (log trees) the producer-conflict pre-check.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssetTree {
    /// `datasets/`; operator uploads.
    Datasets,
    /// `converters/`; operator uploads.
    Converters,
    /// `training_logs/`; daemon-only producer.
    TrainingLogs,
    /// `converter_logs/`; daemon-only producer.
    ConverterLogs,
}

impl AssetTree {
    pub fn dir_name(self) -> &'static str {
        match self {
            AssetTree::Datasets => DATASETS_DIR_NAME,
            AssetTree::Converters => CONVERTERS_DIR_NAME,
            AssetTree::TrainingLogs => TRAINING_LOGS_DIR_NAME,
            AssetTree::ConverterLogs => CONVERTER_LOGS_DIR_NAME,
        }
    }

    /// Operator uploads target only datasets/converters; log dirs are
    /// daemon-only producers.
    pub fn accepts_uploads(self) -> bool {
        matches!(self, AssetTree::Datasets | AssetTree::Converters)
    }

    fn async_delete_job_type(self) -> JobType {
        match self {
            AssetTree::Datasets => JobType::DatasetDelete,
            AssetTree::Converters => JobType::ConverterDelete,
            AssetTree::TrainingLogs => JobType::TrainingLogsDelete,
            AssetTree::ConverterLogs => JobType::ConverterLogsDelete,
        }
    }
}

/// Bumped clone of `prev_core` (revision id `saturating_add(1)`, now timestamp);
/// caller owns the cache publish and `workspace.json` write. Log-tree deletes
/// skip this -- logs aren't workspace state.
fn bump_workspace_revision(prev_core: &WorkspaceCore) -> (u64, WorkspaceCore) {
    let next_revision_id = prev_core.workspace_revision.id.saturating_add(1);
    let mut next_core = prev_core.clone();
    next_core.workspace_revision = WorkspaceRevision {
        id: next_revision_id,
        at: now_rfc3339(),
    };
    (next_revision_id, next_core)
}

/// Parse a workspace-rooted path into `(tree, sub)`: `sub` is `None` for the
/// bare-tree whole-tree-wipe form, `Some(rel)` for a child-rooted path.
fn parse_mutable_path(path: &AssetPath) -> Result<(AssetTree, Option<AssetPath>), FileError> {
    let mut comps = path.components();
    let first = comps.next().ok_or_else(|| {
        FileError::InvalidName(format!("workspace path empty: {}", path.as_str()))
    })?;
    let tree = match first {
        DATASETS_DIR_NAME => AssetTree::Datasets,
        CONVERTERS_DIR_NAME => AssetTree::Converters,
        TRAINING_LOGS_DIR_NAME => AssetTree::TrainingLogs,
        CONVERTER_LOGS_DIR_NAME => AssetTree::ConverterLogs,
        other => {
            return Err(FileError::InvalidName(format!(
                "workspace mutation path top-level must be one of \
                 `datasets` / `converters` / `training_logs` / `converter_logs`; got {other:?}",
            )));
        }
    };
    // No separator after `<tree>` = bare-tree (whole-tree wipe).
    let Some(sub_raw) = path
        .as_str()
        .strip_prefix(tree.dir_name())
        .and_then(|tail| tail.strip_prefix('/'))
    else {
        return Ok((tree, None));
    };
    let sub = AssetPath::parse(sub_raw).map_err(|e| {
        FileError::InvalidName(format!(
            "internal: failed to re-parse sub-path {sub_raw:?} from {}: {e}",
            path.as_str()
        ))
    })?;
    Ok((tree, Some(sub)))
}

/// Datasets uploads require `datasets/<class>/<file>` so the trainer's
/// first-subdir-as-class convention finds the bytes. Deletes are exempt.
fn validate_dataset_upload_depth(tree: AssetTree, path: &AssetPath) -> Result<(), FileError> {
    if tree != AssetTree::Datasets {
        return Ok(());
    }
    if path.components().count() < 3 {
        return Err(FileError::InvalidName(format!(
            "dataset upload requires `datasets/<class>/<file>` minimum; got `{}`",
            path.as_str(),
        )));
    }
    Ok(())
}

/// Emit the per-tree mutation-rejection counter; log trees bucketize on the
/// sibling dataset/converter counter.
fn emit_mutation_rejected(tree: AssetTree) {
    match tree {
        AssetTree::Datasets | AssetTree::TrainingLogs => {
            crate::file_mgr::metrics_hooks::emit_dataset_mutation_rejected()
        }
        AssetTree::Converters | AssetTree::ConverterLogs => {
            crate::file_mgr::metrics_hooks::emit_converter_mutation_rejected()
        }
    }
}

/// Map an admission conflict to [`FileError`], emitting the per-tree rejected counter.
fn reject_conflict(tree: AssetTree, e: RegistryConflict) -> FileError {
    emit_mutation_rejected(tree);
    FileError::from(e)
}

/// Default `?limit=` for [`WorkspaceMgr::list_workspace_children`], capped at
/// [`MAX_DATASET_LIST_LIMIT`].
pub const DEFAULT_DATASET_LIST_LIMIT: usize = 100;
/// Hard ceiling on `?limit=`.
pub const MAX_DATASET_LIST_LIMIT: usize = 1000;

/// Log-tree sub-path must be a single `.jsonl` filename (`<tree>/<id>.jsonl`);
/// nested paths and non-`.jsonl` reject before any state mutation.
fn validate_log_subpath(sub: &AssetPath) -> Result<(), FileError> {
    let comp_count = sub.components().count();
    if comp_count != 1 {
        return Err(FileError::InvalidName(format!(
            "log file path must be a single component (e.g. `<job_id>.jsonl`); \
             got {comp_count} components in {:?}",
            sub.as_str(),
        )));
    }
    let name = sub.as_str();
    if !name.ends_with(".jsonl") {
        return Err(FileError::InvalidName(format!(
            "log file path must end in `.jsonl`; got {name:?}",
        )));
    }
    Ok(())
}

/// On-disk delete target `<workspace_dir>/<tree>[/<sub>]`.
fn build_delete_target(workspace_dir: &Path, tree: AssetTree, sub: Option<&AssetPath>) -> PathBuf {
    let mut p = workspace_dir.join(tree.dir_name());
    if let Some(rel) = sub {
        for component in rel.components() {
            p.push(component);
        }
    }
    p
}

/// Display target_path for a delete: bare tree name for whole-tree wipes, else
/// `<tree>/<sub>`. The `parse` can't fail because `sub` is a tree-prefix strip of
/// a validated [`AssetPath`], so the rejoin reconstructs it byte-for-byte (an
/// arbitrary `sub` at `MAX_DEPTH` would overflow and is unsupported).
fn build_display_path(tree: AssetTree, sub: Option<&AssetPath>) -> AssetPath {
    let raw = match sub {
        Some(rel) => format!("{}/{}", tree.dir_name(), rel.as_str()),
        None => tree.dir_name().to_string(),
    };
    AssetPath::parse(&raw).expect(
        "build_display_path precondition violated: \
         join exceeds AssetPath limits (sub must come from a tree-prefix strip)",
    )
}

/// Filesystem entry kind reported by [`WorkspaceMgr::list_workspace_children`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum EntryKind {
    File,
    Directory,
}

/// One direct child of a workspace asset directory.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct DatasetEntry {
    /// Filename component (no path separators).
    pub name: String,
    #[serde(flatten)]
    pub kind: EntryKind,
    /// Byte size for files; absent for directories.
    pub size_bytes: Option<u64>,
    /// RFC3339 mtime (UTC), always present. Birth time is not exposed (needs
    /// unlanded `statx(STATX_BTIME)`).
    pub mtime: String,
}

/// Paginated direct-child listing under a workspace asset directory.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct DatasetListing {
    /// Name-sorted page of entries.
    pub entries: Vec<DatasetEntry>,
    /// Total under the parent (pre-pagination).
    pub total: usize,
    pub offset: usize,
    /// Echoed limit, clamped to [`MAX_DATASET_LIST_LIMIT`].
    pub limit: usize,
}

/// Receipt returned by [`WorkspaceMgr::upload_workspace_file`].
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct DatasetUploadReceipt {
    /// Validated workspace-rooted asset path (includes the tree-root component).
    pub path: AssetPath,
    /// Lowercase-hex sha256 observed during streaming (not recomputed).
    pub sha256: String,
    /// Body byte count observed during streaming.
    pub size_bytes: u64,
    /// `workspace_revision.id` after the bump.
    pub workspace_revision_id: u64,
}

/// Receipt returned by [`WorkspaceMgr::rename_dataset_category`].
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct RenameCategoryReceipt {
    /// `workspace_revision.id` after the bump (unchanged on a same-name no-op).
    pub workspace_revision_id: u64,
}

/// Validate `path` is exactly `datasets/<name>` (one class dir) and return
/// `<name>`; bare tree, nested paths, and converter/log trees all reject.
fn single_dataset_component(path: &AssetPath) -> Result<String, FileError> {
    let (tree, sub) = parse_mutable_path(path)?;
    if tree != AssetTree::Datasets {
        return Err(FileError::InvalidName(format!(
            "category rename targets `datasets/<name>`; got tree `{}`",
            tree.dir_name(),
        )));
    }
    let sub = sub.ok_or_else(|| {
        FileError::InvalidName(
            "category rename requires `datasets/<name>`, not the bare `datasets` tree".into(),
        )
    })?;
    if sub.components().count() != 1 {
        return Err(FileError::InvalidName(format!(
            "category name must be a single path component; got {:?}",
            sub.as_str(),
        )));
    }
    Ok(sub.as_str().to_string())
}

impl WorkspaceMgr {
    /// Pure path join `<workspace_dir>/<asset_path>`; `path` already carries the
    /// tree-root component.
    pub(crate) fn workspace_asset_path_join(&self, ws: &WorkspaceId, path: &AssetPath) -> PathBuf {
        let mut p = self.workspace_dir(ws);
        for component in path.components() {
            p.push(component);
        }
        p
    }

    /// Resolve a workspace-rooted asset path to its absolute disk path after
    /// validating the workspace exists; does not stat. Rejects `.tmp/` to keep
    /// the staging surface unaddressable (defense in depth).
    pub fn workspace_asset_path(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<PathBuf, FileError> {
        let dir = self.workspace_dir(ws);
        if !workspace_core_path(&dir).exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        if let Some(first) = path.components().next()
            && first == ".tmp"
        {
            return Err(FileError::InvalidName(format!(
                "internal staging path is not externally addressable: {}",
                path.as_str()
            )));
        }
        Ok(self.workspace_asset_path_join(ws, path))
    }

    /// Validate the target is a regular file (not dir/symlink) and return its
    /// path + metadata; the route streams it off-mutex.
    pub fn open_workspace_file(
        &self,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<(PathBuf, std::fs::Metadata), FileError> {
        let target = self.workspace_asset_path(ws, path)?;
        // No-follow stat: a symlink fails the regular-file gate below, defending
        // against operator tampering.
        let md = std::fs::symlink_metadata(&target)
            .map_err(|source| io_err(target.display(), source))?;
        if !md.is_file() {
            return Err(io_err(
                target.display(),
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "open_workspace_file: target is not a regular file",
                ),
            ));
        }
        Ok((target, md))
    }

    /// Bounded direct-child listing rooted at the workspace dir. `None` lists
    /// the top-level (excluding `.tmp/`); `Some(p)` lists `<workspace_dir>/<p>/`.
    /// Name-sorted, `limit` clamped to [`MAX_DATASET_LIST_LIMIT`], never holds
    /// the mutation mutex.
    pub fn list_workspace_children(
        &self,
        ws: &WorkspaceId,
        relative: Option<&AssetPath>,
        offset: usize,
        limit: usize,
    ) -> Result<DatasetListing, FileError> {
        let workspace_dir = self.workspace_dir(ws);
        if !workspace_core_path(&workspace_dir).exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        let target = match relative {
            Some(p) => self.workspace_asset_path_join(ws, p),
            None => workspace_dir.clone(),
        };
        // Explicit `.tmp` guard beyond `AssetPath`'s leading-dot rule; listing
        // must not expose internal staging entries.
        if let Some(p) = relative
            && let Some(first) = p.components().next()
            && first == ".tmp"
        {
            return Err(FileError::InvalidName(format!(
                "internal staging path is not externally addressable: {}",
                p.as_str()
            )));
        }
        if !target.exists() {
            return Ok(DatasetListing {
                entries: Vec::new(),
                total: 0,
                offset,
                limit: limit.min(MAX_DATASET_LIST_LIMIT),
            });
        }
        let md = std::fs::symlink_metadata(&target)
            .map_err(|source| io_err(target.display(), source))?;
        if !md.is_dir() {
            return Err(io_err(
                target.display(),
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "list_workspace_children: target is not a directory",
                ),
            ));
        }
        let exclude_tmp = relative.is_none();
        let read = std::fs::read_dir(&target).map_err(|source| io_err(target.display(), source))?;
        // Two-pass: pass 1 reads names + cheap `d_type`; pass 2 stats only the
        // offset/limit slice (~100 stats not ~100k on 100k entries). `DirEntry`
        // held across the sort keeps the stat valid after the iterator drops
        // (Unix `metadata()` stats by path).
        struct Probe {
            name: String,
            kind: EntryKind,
            entry: std::fs::DirEntry,
        }
        let mut probes: Vec<Probe> = Vec::new();
        for entry in read {
            let entry = entry.map_err(|source| io_err(target.display(), source))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if exclude_tmp && name == ".tmp" {
                continue;
            }
            let ft = match entry.file_type() {
                Ok(ft) => ft,
                Err(source) => return Err(io_err(entry.path().display(), source)),
            };
            let kind = if ft.is_dir() {
                EntryKind::Directory
            } else if ft.is_file() {
                EntryKind::File
            } else {
                // Non-file/dir (symlink/socket/fifo/device): excluded from `total`.
                continue;
            };
            probes.push(Probe { name, kind, entry });
        }
        probes.sort_by(|a, b| a.name.cmp(&b.name));
        let total = probes.len();
        let limit = limit.min(MAX_DATASET_LIST_LIMIT);
        let mut entries: Vec<DatasetEntry> = Vec::with_capacity(limit.min(total));
        for p in probes.into_iter().skip(offset).take(limit) {
            // A concurrent delete (lockless listing vs mutex-held `stage_payload`
            // rename) can vacate an entry; skip it rather than mis-promote
            // NotFound to a 404 "workspace not found". `total` may overcount --
            // already non-authoritative for a live tree.
            let md = match p.entry.metadata() {
                Ok(md) => md,
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
                Err(source) => return Err(io_err(p.entry.path().display(), source)),
            };
            let mtime = rfc3339_from_result(md.modified());
            let size_bytes = match p.kind {
                EntryKind::File => Some(md.len()),
                EntryKind::Directory => None,
            };
            entries.push(DatasetEntry {
                name: p.name,
                kind: p.kind,
                size_bytes,
                mtime,
            });
        }
        Ok(DatasetListing {
            entries,
            total,
            offset,
            limit,
        })
    }

    /// Commit a single-file workspace asset upload.
    ///
    /// Caller preconditions: `src_tmpfile` is on the same filesystem as the
    /// workspace root; `observed_sha256`/`observed_size` reflect its streamed
    /// contents.
    ///
    /// Takes the job-conflict lease (1) first, then holds the per-workspace
    /// mutation mutex across (2) case-insensitive sibling-collision check,
    /// (3) `workspace.json` revision-bump, (4) tempfile rename + parent fsync,
    /// (5) cache publish.
    /// Revision-before-bytes: a crash between (3) and (4) stales heads with bytes
    /// unchanged, and after (4) the new bytes carry an already-bumped revision --
    /// neither leaves a head current across a mutation.
    pub fn upload_workspace_file(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        path: &AssetPath,
        src_tmpfile: &Path,
        observed_sha256: &str,
        observed_size: u64,
    ) -> Result<DatasetUploadReceipt, FileError> {
        // Validate before the lease so a malformed path 400s without contending
        // the registry.
        let (tree, sub) = parse_mutable_path(path)?;
        if !tree.accepts_uploads() {
            emit_mutation_rejected(tree);
            return Err(FileError::InvalidName(format!(
                "uploads to {} are reserved for the daemon producer",
                tree.dir_name(),
            )));
        }
        let Some(_sub) = sub else {
            emit_mutation_rejected(tree);
            return Err(FileError::InvalidName(format!(
                "upload path requires at least one child component below `{}/`: {}",
                tree.dir_name(),
                path.as_str(),
            )));
        };
        if let Err(e) = validate_dataset_upload_depth(tree, path) {
            emit_mutation_rejected(tree);
            return Err(e);
        }

        // Workspace-scoped lease blocks only `WorkspaceDelete`; uploads and
        // file-deletes overlap each other.
        let _ref_guard = self
            .jobs
            .try_acquire_lease(vec![JobReference::Workspace { workspace_id: *ws }])
            .map_err(|e| reject_conflict(tree, e))?;
        let workspace_dir = self.workspace_dir(ws);
        let core_path = workspace_core_path(&workspace_dir);
        if !core_path.exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        let final_path = self.workspace_asset_path_join(ws, path);

        // Per-workspace mutation mutex; sync, never `.await` inside. Shares the
        // `metadata_locks` key with the legacy metadata write path.
        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();
        // Re-check under the lock: else `create_dir_all(parent)` below
        // re-materialises a dir under a `WorkspaceDelete`-finalising workspace.
        if !core_path.exists() {
            emit_mutation_rejected(tree);
            return Err(FileError::NotFound(ws.to_string()));
        }

        // Case-insensitive sibling scan held INSIDE the lock so concurrent
        // uploads of `Foo.mpk`/`foo.mpk` on a CI FS can't both pass before either
        // commits. Missing parent = no siblings.
        if let Some(parent) = final_path.parent()
            && parent.exists()
        {
            let target_name = final_path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| FileError::InvalidName(path.as_str().to_string()))?;
            for entry in
                std::fs::read_dir(parent).map_err(|source| io_err(parent.display(), source))?
            {
                let entry = entry.map_err(|source| io_err(parent.display(), source))?;
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.eq_ignore_ascii_case(target_name) && name_str != target_name {
                    emit_mutation_rejected(tree);
                    return Err(FileError::NameConflict(format!(
                        "workspace upload {target_name:?} collides case-insensitively with \
                         existing {name_str:?}",
                    )));
                }
            }
        }

        let cell = self.cache_cell(ws)?;
        let (next_revision_id, next_core) = bump_workspace_revision(&cell.core());

        // (3) Revision-before-bytes: atomically rewrite `workspace.json` (fsyncs
        // file + parent) before renaming the staged tempfile in.
        write_workspace_core(&workspace_dir, &next_core)?;

        // (4) Atomic rename + parent fsync. Any failure through the publish below
        // must still publish the already-durable `next_core`, else a same-process
        // retry bumps the stale cache to match disk and the next
        // `write_workspace_core` writes a duplicate `revision_id`, breaking
        // cache-invalidation.
        let post_write_result: Result<(), FileError> = (|| {
            // create_dir_all first so a deeply nested upload doesn't ENOENT.
            if let Some(parent) = final_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|source| io_err(parent.display(), source))?;
            }
            std::fs::rename(src_tmpfile, &final_path)
                .map_err(|source| io_err(final_path.display(), source))?;
            if let Some(parent) = final_path.parent() {
                fsync_dir(parent).map_err(|source| io_err(parent.display(), source))?;
            }
            Ok(())
        })();

        // (5) Publish on success AND post-write Err to keep cache==durable
        // workspace_core.
        cell.publish_core(next_core);
        post_write_result?;

        // Count after the publish so a mid-flight failure doesn't overcount.
        crate::file_mgr::metrics_hooks::emit_upload(observed_size);

        Ok(DatasetUploadReceipt {
            path: path.clone(),
            sha256: observed_sha256.to_string(),
            size_bytes: observed_size,
            workspace_revision_id: next_revision_id,
        })
    }

    /// Rename a dataset category dir `datasets/<from>` -> `datasets/<to>` in
    /// place. A category's identity IS its dir name (the trainer reads it as the
    /// class label; no metadata row or manifest), so this is one intra-FS dirent
    /// `std::fs::rename` -- atomic, zero bytes moved (slices stay
    /// content-addressed), no tombstone/drain. Bumps `workspace_revision` because
    /// heads' frozen training-time labels now desync from the live dir (inference
    /// unaffected -- labels never re-derived from the dir at serve time).
    ///
    /// Rejects: non-`datasets/<single-component>` (`InvalidName` -> 400); missing
    /// source (`Io{NotFound}` -> 404); existing target incl. case-insensitive
    /// sibling collision (`NameConflict` -> 409); active Train for the workspace
    /// (`JobConflict` -> 409) -- `scan_dataset` snapshots absolute per-file paths
    /// at job start, so a mid-train rename strips those paths and silently aborts.
    pub fn rename_dataset_category(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        from: &AssetPath,
        to: &AssetPath,
    ) -> Result<RenameCategoryReceipt, FileError> {
        let from_name = single_dataset_component(from)?;
        let to_name = single_dataset_component(to)?;

        // Same-name no-op: a bump would pointlessly stale every head; return the
        // current (unbumped) revision (idempotent).
        if from_name == to_name {
            let cell = self.cache_cell(ws)?;
            return Ok(RenameCategoryReceipt {
                workspace_revision_id: cell.core().workspace_revision.id,
            });
        }

        // Workspace-scoped lease blocks only `WorkspaceDelete`.
        let _ref_guard = self
            .jobs
            .try_acquire_lease(vec![JobReference::Workspace { workspace_id: *ws }])
            .map_err(|e| reject_conflict(AssetTree::Datasets, e))?;

        // The lease does NOT exclude Train, so 409 explicitly: a mid-train rename
        // strips scan_dataset's start-of-job path snapshot -> abort. KNOWN-RACE:
        // this fires before metadata_lock(ws), which Train admission skips, so a
        // Train admitted between here and the rename still aborts (best-effort).
        if self.jobs.has_active_train_for(*ws) {
            emit_mutation_rejected(AssetTree::Datasets);
            return Err(FileError::JobConflict {
                message: format!(
                    "dataset category cannot be renamed while a train job for {ws} is active",
                ),
            });
        }

        let workspace_dir = self.workspace_dir(ws);
        let core_path = workspace_core_path(&workspace_dir);
        if !core_path.exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        let old_dir = self.workspace_asset_path_join(ws, from);
        let new_dir = self.workspace_asset_path_join(ws, to);

        // Per-workspace mutation mutex; sync, never `.await` inside.
        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();
        // Re-check under the lock: a concurrent `WorkspaceDelete` could have
        // renamed the tree away since the pre-lock check.
        if !core_path.exists() {
            emit_mutation_rejected(AssetTree::Datasets);
            return Err(FileError::NotFound(ws.to_string()));
        }

        // Source must be a real dir; symlink-safe stat (a symlink rejects, never
        // followed out of the tree).
        let old_md = match std::fs::symlink_metadata(&old_dir) {
            Ok(md) if md.is_dir() => md,
            Ok(_) => {
                emit_mutation_rejected(AssetTree::Datasets);
                return Err(FileError::InvalidName(format!(
                    "dataset category {from_name:?} is not a directory",
                )));
            }
            // Count the missing-source 404 but not non-NotFound IO faults, so
            // internal 500s don't inflate the counter.
            Err(source) => {
                if source.kind() == std::io::ErrorKind::NotFound {
                    emit_mutation_rejected(AssetTree::Datasets);
                }
                return Err(io_err(old_dir.display(), source));
            }
        };

        // Target must not exist -- except a legitimate case-only rename where it
        // resolves to the SOURCE dir on a CI FS (APFS-CI collapses `Foo`/`foo`),
        // told apart by dev+inode identity rather than name equality.
        use std::os::unix::fs::MetadataExt as _;
        if let Ok(new_md) = std::fs::symlink_metadata(&new_dir) {
            let same_dir = new_md.ino() == old_md.ino() && new_md.dev() == old_md.dev();
            if !same_dir {
                emit_mutation_rejected(AssetTree::Datasets);
                return Err(FileError::NameConflict(format!(
                    "dataset category {to_name:?} already exists",
                )));
            }
        }
        // Case-insensitive sibling scan INSIDE the lock (the trainer hard-fails
        // the next run on a coexisting `Foo`/`foo` pair). Skip the source dir so
        // a case-only rename isn't flagged against itself.
        if let Some(parent) = new_dir.parent()
            && parent.exists()
        {
            for entry in
                std::fs::read_dir(parent).map_err(|source| io_err(parent.display(), source))?
            {
                let entry = entry.map_err(|source| io_err(parent.display(), source))?;
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str == from_name.as_str() {
                    continue;
                }
                if name_str.eq_ignore_ascii_case(&to_name) && name_str != to_name.as_str() {
                    emit_mutation_rejected(AssetTree::Datasets);
                    return Err(FileError::NameConflict(format!(
                        "dataset category {to_name:?} collides case-insensitively \
                         with existing {name_str:?}",
                    )));
                }
            }
        }

        let cell = self.cache_cell(ws)?;
        let (next_revision_id, next_core) = bump_workspace_revision(&cell.core());

        // Revision-before-bytes: durably record the bump before the rename.
        write_workspace_core(&workspace_dir, &next_core)?;

        // Atomic dirent move + one parent fsync (shared `datasets/` dir records
        // both unlink-of-old and link-of-new). Publish on BOTH paths to keep
        // cache==disk revision (revision-id-collision rationale in
        // upload_workspace_file).
        let post_write_result: Result<(), FileError> = (|| {
            std::fs::rename(&old_dir, &new_dir)
                .map_err(|source| io_err(new_dir.display(), source))?;
            if let Some(parent) = new_dir.parent() {
                fsync_dir(parent).map_err(|source| io_err(parent.display(), source))?;
            }
            Ok(())
        })();
        cell.publish_core(next_core);
        post_write_result?;

        Ok(RenameCategoryReceipt {
            workspace_revision_id: next_revision_id,
        })
    }

    /// Workspace-asset delete dispatcher returning the owning [`JobId`] (route
    /// maps to `202 Accepted`).
    ///
    /// All four trees share one async tombstone+stage+drain shape: rename +
    /// tombstone land durably under the per-workspace mutex, then the payload
    /// drains off-mutex (boot recovery resumes an interrupted drain). Datasets
    /// and converters bump `workspace_revision`; log trees skip the bump and
    /// best-effort pre-check for an active producer.
    pub fn start_workspace_asset_delete(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        path: &AssetPath,
    ) -> Result<JobId, FileError> {
        let (tree, sub) = parse_mutable_path(path)?;
        let workspace_dir = self.workspace_dir(ws);
        if !workspace_core_path(&workspace_dir).exists() {
            return Err(FileError::NotFound(ws.to_string()));
        }
        match tree {
            AssetTree::Datasets | AssetTree::Converters => {
                self.start_async_tree_delete(ws, &workspace_dir, tree, sub)
            }
            AssetTree::TrainingLogs | AssetTree::ConverterLogs => {
                self.start_async_log_delete(ws, &workspace_dir, tree, sub)
            }
        }
    }

    /// Async-delete branch for datasets/converters. `sub = None` stages the whole
    /// tree dir; `Some(rel)` stages only `<tree>/<rel>`.
    fn start_async_tree_delete(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        workspace_dir: &Path,
        tree: AssetTree,
        sub: Option<AssetPath>,
    ) -> Result<JobId, FileError> {
        debug_assert!(
            matches!(tree, AssetTree::Datasets | AssetTree::Converters),
            "start_async_tree_delete called with non-async tree {tree:?}",
        );
        let target = build_delete_target(workspace_dir, tree, sub.as_ref());
        // Symlink-safe; missing target -> `Io { NotFound }` -> 404.
        std::fs::symlink_metadata(&target).map_err(|source| io_err(target.display(), source))?;
        // Workspace reference gates only `WorkspaceDelete`; the shared
        // `max_delete_jobs` slot serializes deletes across the family.
        let job_type = tree.async_delete_job_type();
        let display_path = build_display_path(tree, sub.as_ref());
        let job_handle = self
            .jobs
            .try_acquire(
                job_type,
                vec![JobReference::Workspace { workspace_id: *ws }],
                Some(display_path),
            )
            .map_err(|e| reject_conflict(tree, e))?;
        let job_id = job_handle.job_id();

        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();
        // Re-check under the lock: else the whole-tree `create_dir_all` +
        // `fsync_dir` below re-materialise a tree under a removed workspace.
        if !workspace_core_path(workspace_dir).exists() {
            emit_mutation_rejected(tree);
            return Err(FileError::NotFound(ws.to_string()));
        }

        let cell = self.cache_cell(ws)?;
        let (next_revision_id, next_core) = bump_workspace_revision(&cell.core());

        // Tombstone `path` is diagnostic; recovery consumes only the staging-dir
        // location from the filename.
        let tombstone = match tree {
            AssetTree::Datasets => DeleteTombstone::Dataset {
                job_id,
                workspace_id: *ws,
                path: sub.clone(),
                workspace_revision_id: next_revision_id,
                created_at: now_rfc3339(),
            },
            AssetTree::Converters => DeleteTombstone::Converter {
                job_id,
                workspace_id: *ws,
                path: sub.clone(),
                workspace_revision_id: next_revision_id,
                created_at: now_rfc3339(),
            },
            AssetTree::TrainingLogs | AssetTree::ConverterLogs => {
                unreachable!("debug_assert above forbids log trees in this branch")
            }
        };
        let staging_dir = workspace_dir.join(".tmp");
        let staged = write_tombstone(&staging_dir, &tombstone)?;
        // Revision-before-bytes: bump BEFORE renaming the target into staging.
        write_workspace_core(workspace_dir, &next_core)?;
        // Publish on both paths to keep cache==durable workspace_core
        // (revision-id-collision rationale in `upload_workspace_file`).
        let post_write_result: Result<(), FileError> = (|| {
            stage_payload(&target, &staged)?;
            // Whole-tree wipe leaves no tree dir; recreate empty for the canonical
            // structural shape.
            if sub.is_none() {
                std::fs::create_dir_all(&target)
                    .map_err(|source| io_err(target.display(), source))?;
                fsync_dir(workspace_dir)
                    .map_err(|source| io_err(workspace_dir.display(), source))?;
            }
            Ok(())
        })();
        cell.publish_core(next_core);
        post_write_result?;

        drop(_guard); // drain runs off-mutex
        spawn_asset_drain(staged, job_handle);

        Ok(job_id)
    }

    /// Async-delete branch for log trees. Same shape as
    /// [`Self::start_async_tree_delete`] minus the `workspace_revision` bump
    /// (logs aren't workspace state); best-effort `JobConflict` pre-check for an
    /// active producer before any state mutation.
    fn start_async_log_delete(
        self: &Arc<Self>,
        ws: &WorkspaceId,
        workspace_dir: &Path,
        tree: AssetTree,
        sub: Option<AssetPath>,
    ) -> Result<JobId, FileError> {
        debug_assert!(
            matches!(tree, AssetTree::TrainingLogs | AssetTree::ConverterLogs),
            "start_async_log_delete called with non-log tree {tree:?}",
        );

        // Shape check BEFORE the producer-active check: a malformed sub-path is a
        // 400 regardless of producer state.
        if let Some(rel) = &sub {
            validate_log_subpath(rel)?;
        }

        let (job_kind, conflict_label, producer_active) = match tree {
            AssetTree::TrainingLogs => (
                "train",
                "training_logs",
                self.jobs.has_active_train_for(*ws),
            ),
            AssetTree::ConverterLogs => (
                "convert",
                "converter_logs",
                self.jobs.has_active_convert_for(*ws),
            ),
            AssetTree::Datasets | AssetTree::Converters => {
                unreachable!("start_async_log_delete called with non-log tree {tree:?}")
            }
        };
        // KNOWN-RACE: fires before `metadata_lock(ws)`, which producer admission
        // skips, so a producer admitted before the `stage_payload` rename keeps an
        // append fd writing into the staged-for-delete payload (lost on drain).
        // Symptom = a concurrent job loses scroll-back; no committed data loss.
        if producer_active {
            return Err(FileError::JobConflict {
                message: format!(
                    "{conflict_label} cannot be cleared while a {job_kind} job for {ws} is active",
                ),
            });
        }

        let target = build_delete_target(workspace_dir, tree, sub.as_ref());
        // Symlink-safe; missing target -> `Io { NotFound }` -> 404 (the operator's
        // idempotent "clear logs" handles a never-created log dir).
        std::fs::symlink_metadata(&target).map_err(|source| io_err(target.display(), source))?;

        let job_type = tree.async_delete_job_type();
        let display_path = build_display_path(tree, sub.as_ref());
        let job_handle = self
            .jobs
            .try_acquire(
                job_type,
                vec![JobReference::Workspace { workspace_id: *ws }],
                Some(display_path),
            )
            .map_err(|e| reject_conflict(tree, e))?;
        let job_id = job_handle.job_id();

        let lock = self.metadata_lock(ws);
        let _guard = lock.lock();
        // Re-check under the lock: else the `.tmp/` tombstone write re-creates a
        // dir under a finalising workspace.
        if !workspace_core_path(workspace_dir).exists() {
            emit_mutation_rejected(tree);
            return Err(FileError::NotFound(ws.to_string()));
        }

        // Logs skip `workspace_revision_id`; `path` is diagnostic (recovery uses
        // only the staging-dir location from the filename).
        let tombstone = match tree {
            AssetTree::TrainingLogs => DeleteTombstone::TrainingLogs {
                job_id,
                workspace_id: *ws,
                path: sub.clone(),
                created_at: now_rfc3339(),
            },
            AssetTree::ConverterLogs => DeleteTombstone::ConverterLogs {
                job_id,
                workspace_id: *ws,
                path: sub.clone(),
                created_at: now_rfc3339(),
            },
            AssetTree::Datasets | AssetTree::Converters => {
                unreachable!("debug_assert above forbids non-log trees in this branch")
            }
        };
        let staging_dir = workspace_dir.join(".tmp");
        let staged = write_tombstone(&staging_dir, &tombstone)?;
        // No revision bump; tombstone-then-stage is the durable pair.
        stage_payload(&target, &staged)?;
        // Whole-tree wipe leaves no log dir; recreate empty so producer runs find
        // the canonical structural shape.
        if sub.is_none() {
            std::fs::create_dir_all(&target).map_err(|source| io_err(target.display(), source))?;
            fsync_dir(workspace_dir).map_err(|source| io_err(workspace_dir.display(), source))?;
        }

        drop(_guard); // drain runs off-mutex
        spawn_asset_drain(staged, job_handle);

        Ok(job_id)
    }

    /// [`JobType::WorkspaceDelete`] handle holding a [`JobReference::Workspace`]
    /// until the drain finishes so follow-up requests observe the conflict.
    pub(crate) fn register_workspace_delete(
        self: &Arc<Self>,
        workspace_id: WorkspaceId,
    ) -> Result<JobHandle, FileError> {
        self.jobs
            .try_acquire(
                JobType::WorkspaceDelete,
                vec![JobReference::Workspace { workspace_id }],
                None,
            )
            .map_err(FileError::from)
    }

    /// Register a convert job: global `JobType::Convert` slot
    /// (`max_convert_jobs = 1`) + workspace reference for `WorkspaceDelete`
    /// exclusion. A second convert surfaces 409.
    pub fn register_convert_job(
        self: &Arc<Self>,
        workspace_id: WorkspaceId,
    ) -> Result<JobHandle, FileError> {
        self.jobs
            .try_acquire(
                JobType::Convert,
                vec![JobReference::Workspace { workspace_id }],
                None,
            )
            .map_err(FileError::from)
    }

    /// Register a train job: global `JobType::Train` slot (`max_train_jobs = 1`)
    /// plus a workspace reference for `WorkspaceDelete` exclusion. A second train
    /// returns `AnotherTrainRunning`.
    pub fn register_train_job(
        self: &Arc<Self>,
        workspace_id: WorkspaceId,
    ) -> Result<JobHandle, FileError> {
        self.jobs
            .try_acquire(
                JobType::Train,
                vec![JobReference::Workspace { workspace_id }],
                None,
            )
            .map_err(FileError::from)
    }
}

/// Drain `staged` to completion, finalize, and terminate `job_handle`
/// (`succeed(None)` / `fail(...)`). `log_suffix` tags failure traces by caller.
/// `max_iters` caps the loop on the sync-test path so a hung drain fails loudly;
/// the async path passes `None` (tokio abandons a stuck worker).
fn drain_to_completion(
    staged: &StagedDelete,
    job_handle: JobHandle,
    log_suffix: &str,
    max_iters: Option<usize>,
) {
    let mut iter = 0usize;
    loop {
        if let Some(max) = max_iters {
            iter += 1;
            if iter > max {
                tracing::error!(
                    target: "file_mgr",
                    "asset delete drain failed to converge after {max} iterations",
                );
                break;
            }
        }
        match drain_staged_payload(staged, DEFAULT_DELETE_BATCH_ENTRIES) {
            Ok(DrainResult::Done) => break,
            Ok(DrainResult::More) => continue,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    "asset delete drain failed{log_suffix}; boot recovery will resume",
                );
                job_handle.fail(format!("asset delete drain failed: {e}"));
                return;
            }
        }
    }
    if let Err(e) = finalize_staged_delete(staged) {
        tracing::warn!(
            target: "file_mgr",
            err = %e,
            "asset delete finalize failed{log_suffix}; boot recovery will resume",
        );
        job_handle.fail(format!("asset delete finalize failed: {e}"));
        return;
    }
    job_handle.succeed(None);
}

/// Spawn the off-mutex drain, holding `job_handle` for its lifetime so a
/// follow-up request observes the conflict. Tokio blocking pool when a runtime
/// exists, inline otherwise.
fn spawn_asset_drain(staged: StagedDelete, job_handle: JobHandle) {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn_blocking(move || {
            drain_to_completion(&staged, job_handle, "", None);
        });
    } else {
        // Sync test path: drain inline. The 1M-iter cap fails a hung drain
        // instead of hanging the runner; staged state is boot-recoverable.
        drain_to_completion(&staged, job_handle, " (sync path)", Some(1_000_000));
    }
}

/// Stage `bytes` to a tempfile under `<workspace_dir>/.tmp/`. Test-only.
#[cfg(test)]
pub(crate) fn stage_test_tempfile(
    workspace_dir: &Path,
    bytes: &[u8],
) -> Result<tempfile::NamedTempFile, FileError> {
    use std::io::Write;
    let tmp_dir = workspace_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_dir).map_err(|source| io_err(tmp_dir.display(), source))?;
    let mut tmp = tempfile::NamedTempFile::new_in(&tmp_dir)
        .map_err(|source| io_err(tmp_dir.display(), source))?;
    tmp.write_all(bytes)
        .map_err(|source| io_err(tmp.path().display(), source))?;
    tmp.as_file()
        .sync_all()
        .map_err(|source| io_err(tmp.path().display(), source))?;
    Ok(tmp)
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
// Tests stage fixtures with `std::fs::write`; the clippy.toml atomic-writer
// constraint doesn't apply to test setup.
mod tests {
    use super::*;
    use crate::file_mgr::WorkspaceMgr;

    fn fresh_mgr(root: PathBuf) -> Arc<WorkspaceMgr> {
        let mgr = Arc::new(WorkspaceMgr::new(root));
        mgr.ensure_root_layout().expect("layout");
        mgr
    }

    fn new_workspace(mgr: &Arc<WorkspaceMgr>, name: &str) -> WorkspaceId {
        mgr.create(name).expect("create workspace")
    }

    #[test]
    fn workspace_asset_path_join_resolves_under_workspace_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("datasets/audio/cat/sample.wav").unwrap();
        let resolved = mgr.workspace_asset_path_join(&ws, &p);
        assert_eq!(
            resolved,
            mgr.workspace_dir(&ws)
                .join("datasets")
                .join("audio")
                .join("cat")
                .join("sample.wav")
        );
    }

    #[test]
    fn upload_workspace_file_round_trip_datasets() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("datasets/audio_dataset/cat/sample.wav").unwrap();
        let bytes = b"hello world";
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), bytes).unwrap();
        let receipt = mgr
            .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", bytes.len() as u64)
            .expect("upload");
        assert_eq!(receipt.workspace_revision_id, 1);
        assert_eq!(receipt.size_bytes, bytes.len() as u64);
        let final_path = mgr.workspace_asset_path_join(&ws, &p);
        let read = std::fs::read(&final_path).expect("read final");
        assert_eq!(read, bytes);
        let summary = mgr.summary(&ws).expect("summary");
        assert_eq!(summary.core.workspace_revision.id, 1);
    }

    #[test]
    fn upload_workspace_file_round_trip_converters() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("converters/tfjs/model.json").unwrap();
        let bytes = br#"{"format":"tfjs"}"#;
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), bytes).unwrap();
        let receipt = mgr
            .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", bytes.len() as u64)
            .expect("upload");
        assert_eq!(receipt.workspace_revision_id, 1);
        let final_path = mgr.workspace_asset_path_join(&ws, &p);
        assert!(final_path.is_file());
        let read = std::fs::read(&final_path).expect("read final");
        assert_eq!(read, bytes);
    }

    /// Non-`datasets`/`converters` top-levels reject at the lib boundary.
    #[test]
    fn upload_workspace_file_rejects_non_mutable_top_level() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for bad in [
            "heads/x.mpk",
            "training_logs/foo.jsonl",
            "workspace.json",
            "scratch/file.bin",
        ] {
            let p = AssetPath::parse(bad).unwrap();
            let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
            let err = mgr
                .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
                .unwrap_err();
            assert!(
                matches!(err, FileError::InvalidName(_)),
                "expected InvalidName for `{bad}`; got {err:?}"
            );
        }
    }

    /// Bare `datasets`/`converters` reject; uploads require a child component.
    #[test]
    fn upload_workspace_file_rejects_tree_root_without_child() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for bare in ["datasets", "converters"] {
            let p = AssetPath::parse(bare).unwrap();
            let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
            let err = mgr
                .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
                .unwrap_err();
            assert!(
                matches!(err, FileError::InvalidName(_)),
                "expected InvalidName for bare `{bare}`; got {err:?}"
            );
        }
    }

    #[test]
    fn upload_bumps_revision_each_time() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for (i, name) in ["a.json", "b.json", "c.json"].iter().enumerate() {
            let p = AssetPath::parse(&format!("datasets/cls/{name}")).unwrap();
            let bytes = format!("{i}").into_bytes();
            let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), &bytes).unwrap();
            let receipt = mgr
                .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", bytes.len() as u64)
                .expect("upload");
            assert_eq!(receipt.workspace_revision_id, (i as u64) + 1);
        }
    }

    /// `datasets/<file>` (no class folder) rejects; `converters/<file>` uploads.
    #[test]
    fn upload_rejects_dataset_without_class_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("datasets/sample.wav").unwrap();
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        let err = mgr
            .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
            .unwrap_err();
        assert!(
            matches!(err, FileError::InvalidName(_)),
            "expected InvalidName for datasets/<file>; got {err:?}",
        );
        let p = AssetPath::parse("converters/loose.json").unwrap();
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        mgr.upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
            .expect("converters/<file> uploads");
    }

    /// `DELETE datasets/<class>` works; the depth gate applies only to uploads.
    #[test]
    fn delete_dataset_class_folder_remains_supported() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("datasets/cat/sample.wav").unwrap();
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        mgr.upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
            .expect("upload");
        let class_root = AssetPath::parse("datasets/cat").unwrap();
        mgr.start_workspace_asset_delete(&ws, &class_root)
            .expect("class folder delete admits at depth 2");
    }

    #[test]
    fn upload_then_delete_round_trip_sync() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("datasets/audio/sample.wav").unwrap();
        let staged_tmp = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        mgr.upload_workspace_file(&ws, &p, staged_tmp.path(), "deadbeef", 1)
            .unwrap();
        let pre = mgr.summary(&ws).unwrap().core.workspace_revision.id;
        let _job = mgr.start_workspace_asset_delete(&ws, &p).expect("delete");
        let post = mgr.summary(&ws).unwrap().core.workspace_revision.id;
        assert_eq!(post, pre + 1, "delete bumps revision");
        let final_path = mgr.workspace_asset_path_join(&ws, &p);
        assert!(!final_path.exists(), "{final_path:?} still present");
    }

    /// A converter delete writes a `Converter` tombstone, admits as
    /// `JobType::ConverterDelete`, and still bumps the revision.
    #[test]
    fn converter_upload_then_delete_round_trip_sync() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let p = AssetPath::parse("converters/tfjs/model.json").unwrap();
        let staged_tmp = stage_test_tempfile(&mgr.workspace_dir(&ws), b"manifest").unwrap();
        mgr.upload_workspace_file(&ws, &p, staged_tmp.path(), "deadbeef", 8)
            .unwrap();
        let pre = mgr.summary(&ws).unwrap().core.workspace_revision.id;
        let _job = mgr
            .start_workspace_asset_delete(&ws, &p)
            .expect("converter delete");
        let post = mgr.summary(&ws).unwrap().core.workspace_revision.id;
        assert_eq!(post, pre + 1, "converter delete bumps revision");
        let final_path = mgr.workspace_asset_path_join(&ws, &p);
        assert!(!final_path.exists());
    }

    /// `start_workspace_asset_delete` rejects any non-mutable top-level.
    #[test]
    fn start_workspace_asset_delete_rejects_non_mutable_top_level() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        // `training_logs`/`converter_logs` are mutable; rejection set is the rest.
        for bad in [
            "heads/x.mpk",
            "heads",
            "workspace.json",
            "scratch/file.bin",
            "scratch",
        ] {
            let p = AssetPath::parse(bad).unwrap();
            let err = mgr.start_workspace_asset_delete(&ws, &p).unwrap_err();
            assert!(
                matches!(err, FileError::InvalidName(_)),
                "expected InvalidName for `{bad}`; got {err:?}"
            );
        }
    }

    /// Single-file log delete of a missing file is `NotFound` (the async path
    /// stats before staging). Bare-tree wipes are excluded: creation lays log
    /// dirs down empty so "clear all logs" stays idempotent without a 404.
    #[test]
    fn start_workspace_asset_delete_log_file_returns_not_found_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for path in [
            "training_logs/00000000-0000-4000-8000-000000000001.jsonl",
            "converter_logs/00000000-0000-4000-8000-000000000002.jsonl",
        ] {
            let p = AssetPath::parse(path).unwrap();
            let err = mgr.start_workspace_asset_delete(&ws, &p).unwrap_err();
            assert!(
                matches!(
                    &err,
                    FileError::Io { source, .. }
                        if source.kind() == std::io::ErrorKind::NotFound
                ),
                "expected NotFound for missing `{path}`; got {err:?}",
            );
        }
    }

    /// Whole-tree wipe of an empty log dir succeeds (empty payload drains as a
    /// no-op); exercises the "exists but empty" path.
    #[test]
    fn start_workspace_asset_delete_log_whole_tree_succeeds_on_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let ws_dir = mgr.workspace_dir(&ws);
        for path in ["training_logs", "converter_logs"] {
            std::fs::create_dir_all(ws_dir.join(path))
                .unwrap_or_else(|e| panic!("mkdir `{path}` failed: {e}"));
            let p = AssetPath::parse(path).unwrap();
            let _job = mgr
                .start_workspace_asset_delete(&ws, &p)
                .unwrap_or_else(|e| panic!("whole-tree empty wipe of `{path}` failed: {e:?}"));
            let dir = ws_dir.join(path);
            assert!(dir.exists(), "{path}/ recreated after whole-tree wipe");
        }
    }

    /// Single-file log delete and whole-dir wipe both stage + drain inline;
    /// the wipe recreates the empty dir for the canonical shape.
    #[test]
    fn start_workspace_asset_delete_log_paths_async_and_drain_into_staging() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let ws_dir = mgr.workspace_dir(&ws);
        let log_dir = ws_dir.join("training_logs");
        std::fs::create_dir_all(&log_dir).unwrap();
        let a = log_dir.join("00000000-0000-4000-8000-000000000001.jsonl");
        let b = log_dir.join("00000000-0000-4000-8000-000000000002.jsonl");
        std::fs::write(&a, b"{}").unwrap();
        std::fs::write(&b, b"{}").unwrap();

        let p =
            AssetPath::parse("training_logs/00000000-0000-4000-8000-000000000001.jsonl").unwrap();
        let _job = mgr
            .start_workspace_asset_delete(&ws, &p)
            .expect("single-file log delete");
        assert!(!a.exists(), "single-file log delete drained `a`");
        assert!(b.exists(), "sibling jsonl untouched by single-file delete");

        let p = AssetPath::parse("training_logs").unwrap();
        let _job = mgr
            .start_workspace_asset_delete(&ws, &p)
            .expect("whole-tree log delete");
        assert!(!b.exists(), "whole-dir log wipe drained `b`");
        let recreated = ws_dir.join("training_logs");
        assert!(
            recreated.exists() && recreated.is_dir(),
            "empty training_logs/ recreated after whole-tree wipe",
        );
    }

    /// Log sub-paths not matching `<dir>/<id>.jsonl` reject before any mutation.
    #[test]
    fn start_workspace_asset_delete_log_subpath_shape_constraints() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        for bad in [
            "training_logs/sub/x.jsonl",
            "training_logs/note.txt",
            "converter_logs/sub/y.jsonl",
            "converter_logs/keep.txt",
        ] {
            let p = AssetPath::parse(bad).unwrap();
            let err = mgr.start_workspace_asset_delete(&ws, &p).unwrap_err();
            assert!(
                matches!(err, FileError::InvalidName(_)),
                "expected InvalidName for `{bad}`; got {err:?}",
            );
        }
    }

    /// Ordering pin: shape `InvalidName` (400) must surface BEFORE the
    /// producer-active `JobConflict` (409); else a malformed-path-during-train
    /// would wrongly 409.
    #[test]
    fn start_workspace_asset_delete_log_shape_check_runs_before_producer_check() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        // Train job so the producer-active check would fire if reached.
        let _train = mgr
            .jobs
            .try_acquire(
                JobType::Train,
                vec![JobReference::Workspace { workspace_id: ws }],
                None,
            )
            .expect("train admission");
        let bad = AssetPath::parse("training_logs/nested/job.jsonl").unwrap();
        let err = mgr.start_workspace_asset_delete(&ws, &bad).unwrap_err();
        assert!(
            matches!(err, FileError::InvalidName(_)),
            "shape error must surface before producer-active 409; got {err:?}",
        );
    }

    /// Whole-tree datasets wipe stages the tree dir and recreates it empty.
    #[test]
    fn start_workspace_asset_delete_datasets_whole_tree_returns_async_and_recreates_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        // Stage content so the rename has bytes to drain.
        let p = AssetPath::parse("datasets/cls/sample.bin").unwrap();
        let scratch = tmp.path().join("scratch.bin");
        std::fs::write(&scratch, b"x").unwrap();
        mgr.upload_workspace_file(&ws, &p, &scratch, "00", 1)
            .unwrap();

        let p = AssetPath::parse("datasets").unwrap();
        let _job = mgr
            .start_workspace_asset_delete(&ws, &p)
            .expect("whole-tree datasets delete");
        let datasets_dir = mgr.workspace_dir(&ws).join("datasets");
        assert!(datasets_dir.exists(), "empty datasets/ recreated");
        assert!(datasets_dir.is_dir());
        assert_eq!(
            std::fs::read_dir(&datasets_dir).unwrap().count(),
            0,
            "datasets/ is empty post-wipe",
        );
    }

    #[test]
    fn upload_blocked_by_active_workspace_delete() {
        // Uploads are gated only by `WorkspaceDelete`; train/convert/
        // dataset-delete don't block them.
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let _del = mgr
            .jobs
            .try_acquire(
                JobType::WorkspaceDelete,
                vec![JobReference::Workspace { workspace_id: ws }],
                None,
            )
            .expect("workspace-delete admitted");
        let p = AssetPath::parse("datasets/audio/cat/sample.wav").unwrap();
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        let err = mgr
            .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
            .unwrap_err();
        assert!(matches!(err, FileError::JobConflict { .. }));
    }

    #[test]
    fn upload_coexists_with_active_train_in_same_workspace() {
        // Train + upload in the same workspace overlap; no path-overlap conflict.
        let tmp = tempfile::tempdir().unwrap();
        let mgr = fresh_mgr(tmp.path().to_path_buf());
        let ws = new_workspace(&mgr, "main");
        let _train = mgr
            .jobs
            .try_acquire(
                JobType::Train,
                vec![JobReference::Workspace { workspace_id: ws }],
                None,
            )
            .expect("train admitted");
        let p = AssetPath::parse("datasets/audio/cat/sample.wav").unwrap();
        let staged = stage_test_tempfile(&mgr.workspace_dir(&ws), b"x").unwrap();
        let receipt = mgr
            .upload_workspace_file(&ws, &p, staged.path(), "deadbeef", 1)
            .expect("upload during train succeeds");
        assert_eq!(receipt.workspace_revision_id, 1);
    }
}
