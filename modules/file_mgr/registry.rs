//! Workspace-lifecycle surface of [`WorkspaceMgr`]. Workspaces nest under
//! `<root>/workspaces/<id>/`; creation writes `workspace.json` +
//! `heads.json`, deletion stages the tree under
//! `<root>/.tmp/delete-workspace-<job_id>/payload` and drains it off-mutex
//! via [`crate::file_mgr::staging`].

use std::path::PathBuf;
use std::sync::Arc;

use crate::common::ids::{JobId, WorkspaceId};
use crate::common::workspace::{HeadIndex, WorkspaceCore, WorkspaceRevision};

use crate::file_mgr::WorkspaceMgr;
use crate::file_mgr::cache::WorkspaceCacheCell;
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::schema::{
    root_tmp_dir, workspace_dir_for, workspaces_dir, write_head_index, write_workspace_core,
};
use crate::file_mgr::staging::{
    DEFAULT_DELETE_BATCH_ENTRIES, DeleteTombstone, DrainResult, StagedDelete, drain_staged_payload,
    finalize_staged_delete, stage_payload, write_tombstone,
};
use crate::file_mgr::time_util::now_rfc3339;
use crate::file_mgr::validate::fsync_dir;

pub(crate) struct WorkspaceDirEntry {
    pub id: WorkspaceId,
    pub path: PathBuf,
    /// `workspace.json` exists (the publish point); false = residue.
    pub has_core: bool,
}

pub(crate) struct WalkWorkspaceDirsOutcome {
    pub entries: Vec<WorkspaceDirEntry>,
    /// Dirents that failed to read/stat (EIO/EACCES), warn-skipped; folds into
    /// recovery's filesystem-level `workspace_enumeration_failures`, kept
    /// distinct from workspace-level `workspace_recovery_failures` for triage.
    pub dirent_errors: usize,
}

/// Yield every direct child of `workspaces_root` named as a strict
/// [`WorkspaceId`] (UUID-v4). Non-dir/non-UUID entries `debug!`-skip; per-dirent
/// I/O errors `warn!`-skip AND count into `dirent_errors`. Absent root => empty.
pub(crate) fn walk_workspace_dirs(
    workspaces_root: &std::path::Path,
) -> Result<WalkWorkspaceDirsOutcome, FileError> {
    if !workspaces_root.exists() {
        return Ok(WalkWorkspaceDirsOutcome {
            entries: Vec::new(),
            dirent_errors: 0,
        });
    }
    let entries =
        std::fs::read_dir(workspaces_root).map_err(|e| io_err(workspaces_root.display(), e))?;
    let mut out = Vec::new();
    let mut dirent_errors = 0usize;
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                // Warn (not silent-drop): a dropped dirent vanishes from both list and recovery.
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %workspaces_root.display(),
                    "walk_workspace_dirs: skipping unreadable dirent",
                );
                dirent_errors = dirent_errors.saturating_add(1);
                continue;
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                // Surface EACCES/EIO rather than conflate with "not a dir".
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %path.display(),
                    "walk_workspace_dirs: skipping entry with unreadable file_type",
                );
                dirent_errors = dirent_errors.saturating_add(1);
                continue;
            }
        };
        if !file_type.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let id = match WorkspaceId::parse(&name) {
            Ok(id) => id,
            Err(_) => {
                tracing::debug!(
                    target: "file_mgr",
                    name = %name,
                    "walk_workspace_dirs: skipping entry whose name is not a valid UUID-v4",
                );
                continue;
            }
        };
        // lstat (not `Path::exists`): a symlinked workspace.json could point OUT
        // of the tree, making recovery read unrelated metadata. Non-regular shape
        // is corruption -> skip + bump dirent_errors (not auto-delete); only
        // genuine NotFound (incomplete-create residue) yields `has_core = false`.
        let core_path = crate::file_mgr::schema::workspace_core_path(&path);
        let has_core = match std::fs::symlink_metadata(&core_path) {
            Ok(m) if m.is_file() => true,
            Ok(m) => {
                tracing::warn!(
                    target: "file_mgr",
                    path = %core_path.display(),
                    file_type = ?m.file_type(),
                    "walk_workspace_dirs: workspace.json is not a regular file; skipping entry",
                );
                dirent_errors = dirent_errors.saturating_add(1);
                continue;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => {
                tracing::warn!(
                    target: "file_mgr",
                    err = %e,
                    path = %core_path.display(),
                    "walk_workspace_dirs: stat workspace.json failed; skipping entry",
                );
                dirent_errors = dirent_errors.saturating_add(1);
                continue;
            }
        };
        out.push(WorkspaceDirEntry { id, path, has_core });
    }
    Ok(WalkWorkspaceDirsOutcome {
        entries: out,
        dirent_errors,
    })
}

fn validate_workspace_name(name: &str) -> Result<(), FileError> {
    let invalid = name.is_empty()
        || name.len() > 128
        || name.contains('\0')
        || name.contains('/')
        || name.contains('\\')
        || name.chars().next().is_some_and(char::is_whitespace)
        || name.chars().last().is_some_and(char::is_whitespace)
        || name.chars().any(char::is_control);
    if invalid {
        return Err(FileError::InvalidName(name.to_string()));
    }
    Ok(())
}

const MAX_WORKSPACE_TAGS: usize = 32;
/// Per-tag byte cap, measured post ASCII-trim.
const MAX_TAG_BYTES: usize = 64;

/// ASCII-trims each tag and rejects duplicates by `to_lowercase` fold (no NFC,
/// so composed != decomposed); returns the trimmed set in input order.
fn validate_workspace_tags(tags: &[String]) -> Result<Vec<String>, FileError> {
    if tags.len() > MAX_WORKSPACE_TAGS {
        return Err(FileError::InvalidName(format!(
            "tags exceed max {MAX_WORKSPACE_TAGS}: got {}",
            tags.len()
        )));
    }
    let mut out: Vec<String> = Vec::with_capacity(tags.len());
    let mut seen_lower: std::collections::HashSet<String> =
        std::collections::HashSet::with_capacity(tags.len());
    for raw in tags {
        let trimmed = raw.trim_matches(|c: char| c.is_ascii_whitespace());
        if trimmed.is_empty() {
            return Err(FileError::InvalidName(format!(
                "tag empty after ASCII whitespace trim: {raw:?}"
            )));
        }
        if trimmed.len() > MAX_TAG_BYTES {
            return Err(FileError::InvalidName(format!(
                "tag exceeds {MAX_TAG_BYTES} bytes: {trimmed:?}"
            )));
        }
        if trimmed.contains('\0') || trimmed.contains('/') || trimmed.contains('\\') {
            return Err(FileError::InvalidName(format!(
                "tag contains NUL or path separator: {trimmed:?}"
            )));
        }
        if trimmed.chars().any(char::is_control) {
            return Err(FileError::InvalidName(format!(
                "tag contains control character: {trimmed:?}"
            )));
        }
        let lower = trimmed.to_lowercase();
        if !seen_lower.insert(lower) {
            return Err(FileError::InvalidName(format!(
                "duplicate tag (case-insensitive): {trimmed:?}"
            )));
        }
        out.push(trimmed.to_string());
    }
    Ok(out)
}

impl WorkspaceMgr {
    /// Ensure `self.root` exists; subdirectories are materialized lazily by
    /// whichever writer owns each path.
    pub fn ensure_root_layout(&self) -> Result<(), FileError> {
        std::fs::create_dir_all(&self.root).map_err(|e| io_err(self.root.display(), e))?;
        Ok(())
    }

    /// Create a new workspace under `<root>/workspaces/<id>/`.
    ///
    /// `registry_lock` (sync, no `.await`) serializes list-check-create so
    /// two concurrent `create("main")` can't both win with distinct UUIDs.
    pub fn create(&self, name: &str) -> Result<WorkspaceId, FileError> {
        self.create_with_tags(name, &[])
    }

    pub fn create_with_tags(&self, name: &str, tags: &[String]) -> Result<WorkspaceId, FileError> {
        validate_workspace_name(name)?;
        let normalized_tags = validate_workspace_tags(tags)?;
        // Sync, no `.await`, so cancellation can't strand the guard.
        let _registry_guard = self.registry_lock.lock();
        self.ensure_workspace_name_available(name, None)?;
        let workspaces_root = workspaces_dir(&self.root);
        std::fs::create_dir_all(&workspaces_root)
            .map_err(|e| io_err(workspaces_root.display(), e))?;
        let id = WorkspaceId::new();
        let ws = self.workspace_dir(&id);
        std::fs::create_dir_all(&ws).map_err(|e| io_err(ws.display(), e))?;
        let now = now_rfc3339();
        // Head index BEFORE core: a crash between leaves a dir without
        // `workspace.json`, which `list_workspaces` filters as residue.
        write_head_index(&ws, &HeadIndex::default())?;
        let core = WorkspaceCore {
            id,
            name: name.to_string(),
            tags: normalized_tags,
            created_at: now.clone(),
            workspace_revision: WorkspaceRevision { id: 0, at: now },
            head_count: 0,
        };
        write_workspace_core(&ws, &core)?;
        // `write_workspace_core` fsynced `ws`; fsync the `workspaces/` and
        // `<root>/` parents too so a crash+remount sees the new dir entry.
        if workspaces_root.exists() {
            fsync_dir(&workspaces_root).map_err(|e| io_err(workspaces_root.display(), e))?;
        }
        if self.root.exists() {
            fsync_dir(&self.root).map_err(|e| io_err(self.root.display(), e))?;
        }
        self.caches.insert(
            id,
            Arc::new(WorkspaceCacheCell::new(core, HeadIndex::default())),
        );
        Ok(id)
    }

    /// Atomically update `name` and/or `tags` on `workspace.json` (both `None`
    /// => [`FileError::InvalidName`]). Same validation as `create`, uniqueness
    /// scanned across the registry excluding self. `workspace_revision` /
    /// `head_count` / `created_at` are preserved since name/tag edits are
    /// operator metadata, not mutations.
    pub fn patch_workspace(
        &self,
        id: &WorkspaceId,
        name: Option<&str>,
        tags: Option<&[String]>,
    ) -> Result<Arc<crate::common::workspace::WorkspaceCore>, FileError> {
        if name.is_none() && tags.is_none() {
            return Err(FileError::InvalidName(
                "requires at least one of `name` or `tags`".into(),
            ));
        }
        // Validate before locking so a malformed request fails cheaply.
        if let Some(n) = name {
            validate_workspace_name(n)?;
        }
        let normalized_tags = match tags {
            Some(t) => Some(validate_workspace_tags(t)?),
            None => None,
        };

        // Hold across uniqueness scan + publish so a concurrent `create` can't
        // slip a name in between. Sync, no `.await`.
        let _registry_guard = self.registry_lock.lock();

        if let Some(new_name) = name {
            self.ensure_workspace_name_available(new_name, Some(id))?;
        }

        let workspace_dir = self.workspace_dir(id);
        let core_path = crate::file_mgr::schema::workspace_core_path(&workspace_dir);
        if !core_path.exists() {
            return Err(FileError::NotFound(id.to_string()));
        }

        // Per-workspace mutation mutex serializes against the dataset/head
        // publish paths AND the delete staging primitive. Sync.
        let lock = self.metadata_lock(id);
        let _guard = lock.lock();

        // Re-check under the lock: a delete admitted after the pre-lock check
        // could have renamed the tree away, so the core write would ENOENT or
        // re-materialize a split tree.
        if !core_path.exists() {
            // `metadata_lock(id)` above may have re-inserted a fresh Arc the
            // delete path won't reclaim; remove it to keep `metadata_locks` in
            // lockstep with the live set, dropping the guard first.
            drop(_guard);
            self.metadata_locks.remove(id);
            return Err(FileError::NotFound(id.to_string()));
        }

        let cell = self.cache_cell(id)?;
        let prev_core = cell.core();
        let mut next_core = (*prev_core).clone();
        if let Some(new_name) = name {
            next_core.name = new_name.to_string();
        }
        if let Some(new_tags) = normalized_tags {
            next_core.tags = new_tags;
        }
        crate::file_mgr::schema::write_workspace_core(&workspace_dir, &next_core)?;
        cell.publish_core(next_core.clone());
        Ok(Arc::new(next_core))
    }

    /// Ensure `name` is case-insensitively unique across published workspaces.
    /// Callers MUST hold `registry_lock` across this scan AND their commit so
    /// create/rename races can't slip through. Fold is `to_lowercase` (no NFC;
    /// composed != decomposed).
    fn ensure_workspace_name_available(
        &self,
        name: &str,
        except: Option<&WorkspaceId>,
    ) -> Result<(), FileError> {
        let lower = name.to_lowercase();
        for existing in self.list_workspaces()? {
            if except.is_some_and(|id| existing == *id) {
                continue;
            }
            let core = match self.read_cached_core(&existing) {
                Ok(c) => c,
                // Vanished peer (half-create or concurrent delete): skip, don't
                // abort. Both NotFound shapes must be caught, else the typed
                // NotFound escapes as a spurious 404 carrying the deleted peer's
                // id on a valid create/patch.
                Err(FileError::Io { source, .. })
                    if source.kind() == std::io::ErrorKind::NotFound =>
                {
                    continue;
                }
                Err(FileError::NotFound(_)) => continue,
                Err(e) => return Err(e),
            };
            if core.name.to_lowercase() == lower {
                return Err(FileError::NameConflict(name.to_string()));
            }
        }
        Ok(())
    }

    /// Synchronous fallback for [`Self::start_delete_workspace`]: stage, drain,
    /// and finalize inline so tests expecting "delete returns when the bytes are
    /// gone" pass. Production takes the async path via
    /// [`crate::file_mgr::FsService::start_delete_workspace`].
    pub(crate) fn delete(&self, id: &WorkspaceId) -> Result<(), FileError> {
        let (staged, _job) = self.start_delete_workspace_inner(id)?;
        loop {
            match drain_staged_payload(&staged, DEFAULT_DELETE_BATCH_ENTRIES)? {
                DrainResult::Done => break,
                DrainResult::More => continue,
            }
        }
        finalize_staged_delete(&staged)?;
        Ok(())
    }

    /// Begin an async workspace delete: stage the tree under
    /// `<root>/.tmp/delete-workspace-<job_id>/payload` and return the `JobId`
    /// immediately; a drain task (or boot recovery on crash) runs
    /// [`drain_staged_payload`] + [`finalize_staged_delete`]. Registers a job so
    /// follow-up upload/delete requests get 409 `JobConflict` until the handle
    /// drops; `&Arc<Self>` because admission bumps the Arc.
    pub(crate) fn start_delete_workspace(
        self: &Arc<Self>,
        id: &WorkspaceId,
    ) -> Result<JobId, FileError> {
        // Existence gate first so a second delete on an already-deleted
        // workspace returns 404 even mid-drain.
        let ws = self.workspace_dir(id);
        let core_path = crate::file_mgr::schema::workspace_core_path(&ws);
        if !core_path.exists() {
            return Err(FileError::NotFound(id.to_string()));
        }
        // JobHandle before any disk mutation so overlap conflicts fire before
        // staging; its job_id supersedes the tombstone's so the JSONL log and
        // `/jobs/{job_id}` snapshot share one identity.
        let job_handle = self.register_workspace_delete(*id)?;
        let job_id = job_handle.job_id();
        // Snapshot active-source status before the rename (caller holds
        // `active_mutex`, so `POST /active` can't race it); surfaced via
        // `active_source_deleted` (runtime survives, but `source_workspace_alive`
        // flips false on `GET /active`).
        let was_active_source =
            crate::file_mgr::active_source_head_in_workspace(&self.root, *id).is_some();
        let staged = self.start_delete_workspace_inner_with_id(id, job_id)?;
        // Drain off the request path; spawn-blocking only when a runtime exists
        // (absent in sync tests). Not registered with `DrainRegistry`, so SIGTERM
        // mid-drain loses the in-memory Running->Completed transition -- but data
        // is safe: the rename moved the tree atomically and boot recovery's
        // `recover_root_staging` (via `drain_workspace_staging_dir`) re-drains
        // the tombstone.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let staged_for_task = staged.clone();
            handle.spawn_blocking(move || {
                let budget = DEFAULT_DELETE_BATCH_ENTRIES;
                loop {
                    match drain_staged_payload(&staged_for_task, budget) {
                        Ok(DrainResult::Done) => break,
                        Ok(DrainResult::More) => continue,
                        Err(e) => {
                            tracing::warn!(
                                target: "file_mgr",
                                err = %e,
                                "workspace delete drain failed; boot recovery will resume",
                            );
                            job_handle.fail(format!("workspace delete drain failed: {e}"));
                            return;
                        }
                    }
                }
                if let Err(e) = finalize_staged_delete(&staged_for_task) {
                    tracing::warn!(
                        target: "file_mgr",
                        err = %e,
                        "workspace delete finalize failed; boot recovery will resume",
                    );
                    job_handle.fail(format!("workspace delete finalize failed: {e}"));
                    return;
                }
                job_handle.succeed(Some(
                    crate::file_mgr::job_registry::JobResult::WorkspaceDelete {
                        active_source_deleted: was_active_source,
                    },
                ));
            });
        } else {
            // No runtime (sync tests): run inline. Loop is bounded by tree size
            // (each `More` removes >=1 entry); a cap could exit with stage_dir +
            // tombstone on disk -> recovery loops forever.
            loop {
                match drain_staged_payload(&staged, DEFAULT_DELETE_BATCH_ENTRIES)? {
                    DrainResult::Done => break,
                    DrainResult::More => continue,
                }
            }
            finalize_staged_delete(&staged)?;
            job_handle.succeed(Some(
                crate::file_mgr::job_registry::JobResult::WorkspaceDelete {
                    active_source_deleted: was_active_source,
                },
            ));
        }
        Ok(job_id)
    }

    /// Stage a delete (rename tree under `.tmp/`, write tombstone, eject
    /// cache + lock); caller drains + finalizes the returned handle.
    fn start_delete_workspace_inner(
        &self,
        id: &WorkspaceId,
    ) -> Result<(StagedDelete, JobId), FileError> {
        let job_id = JobId::new();
        let staged = self.start_delete_workspace_inner_with_id(id, job_id)?;
        Ok((staged, job_id))
    }

    /// Like [`Self::start_delete_workspace_inner`] but reuses an external
    /// `job_id` for one shared identity. Holds `metadata_lock(id)` across the
    /// `stage_payload` rename so same-lock writers (rotation, `patch_workspace`,
    /// dataset) can't land mutations into a dir being renamed away; released
    /// AFTER the per-id entry is removed from `metadata_locks`, so later callers
    /// get a fresh `Mutex` while writers on a stale Arc clone re-check existence
    /// under their lock and 404.
    fn start_delete_workspace_inner_with_id(
        &self,
        id: &WorkspaceId,
        job_id: JobId,
    ) -> Result<StagedDelete, FileError> {
        let ws = self.workspace_dir(id);
        let core_path = crate::file_mgr::schema::workspace_core_path(&ws);
        if !core_path.exists() {
            return Err(FileError::NotFound(id.to_string()));
        }
        let lock = self.metadata_lock(id);
        let _guard = lock.lock();
        // Re-check under the lock: a racing writer could have evicted.
        if !core_path.exists() {
            return Err(FileError::NotFound(id.to_string()));
        }
        let tombstone = DeleteTombstone::Workspace {
            job_id,
            workspace_id: *id,
            created_at: now_rfc3339(),
        };
        let staging_dir = root_tmp_dir(&self.root);
        let staged = StagedDelete::for_tombstone(&staging_dir, &tombstone);
        // Stage-before-tombstone INVARIANT: a crash between must never leave a
        // consumable tombstone beside a still-live workspace. Tombstone-first
        // would let recovery (staging sweep precedes workspace recovery) read a
        // missing stage dir as "already drained", unlink the tombstone, and
        // resurrect the intact tree -- silently reviving destroyed data. Staging
        // first makes the residue an orphan stage dir the sweep reclaims.
        stage_payload(&ws, &staged)?;
        // Durable resumable intent the drain resumes from after restart; both
        // parents already fsynced in `stage_payload`.
        write_tombstone(&staging_dir, &tombstone)?;
        // Eject runtime refs so subsequent reads fail closed.
        self.caches.remove(id);
        self.metadata_locks.remove(id);
        Ok(staged)
    }

    /// List workspace UUIDs under `<root>/workspaces/`. Entries without a
    /// UUID-v4 name or `workspace.json` (half-creates/stray files) are
    /// `debug!`-skipped.
    pub fn list_workspaces(&self) -> Result<Vec<WorkspaceId>, FileError> {
        // `dirent_errors` discarded: the public list returns a partial result,
        // not 5xx; recovery's report owns those counts.
        let mut out = Vec::new();
        let outcome = walk_workspace_dirs(&workspaces_dir(&self.root))?;
        for entry in outcome.entries {
            if entry.has_core {
                out.push(entry.id);
            } else {
                tracing::debug!(
                    target: "file_mgr",
                    workspace_id = %entry.id,
                    "list_workspaces: skipping workspace dir without workspace.json (incomplete create)",
                );
            }
        }
        Ok(out)
    }

    pub(crate) fn read_cached_core(
        &self,
        id: &WorkspaceId,
    ) -> Result<Arc<WorkspaceCore>, FileError> {
        let cell = self.cache_cell(id)?;
        Ok(cell.core())
    }

    pub(crate) fn cache_cell(
        &self,
        id: &WorkspaceId,
    ) -> Result<Arc<WorkspaceCacheCell>, FileError> {
        if let Some(cell) = self.caches.get(id) {
            return Ok(cell.clone());
        }
        let ws = self.workspace_dir(id);
        let cell = Arc::new(WorkspaceCacheCell::load_from_disk(&ws)?);
        // Re-check existence under the shard write lock (`Entry`) so a concurrent
        // delete's `caches.remove(id)` (same shard lock) can't interleave
        // load+insert to resurrect a cell, stranding a `caches` entry with no
        // `metadata_locks` peer and breaking the moved-in-lockstep invariant. Do
        // NOT consult `metadata_lock(id)`: it lazily inserts, trading one orphan
        // map for the other.
        use dashmap::mapref::entry::Entry;
        match self.caches.entry(*id) {
            Entry::Occupied(e) => Ok(e.get().clone()),
            Entry::Vacant(e) => {
                if !crate::file_mgr::schema::workspace_core_path(&ws).is_file() {
                    // Deleted between load and lock: fail closed.
                    return Err(FileError::NotFound(id.to_string()));
                }
                Ok(e.insert(cell).clone())
            }
        }
    }

    /// Hot-path summary: cached `WorkspaceCore` + `HeadIndex` plus the
    /// derived per-head [`crate::common::workspace::HeadStatus`] list.
    /// Never walks `datasets/`.
    pub fn summary(
        &self,
        id: &WorkspaceId,
    ) -> Result<crate::file_mgr::WorkspaceSummary, FileError> {
        let cell = self.cache_cell(id)?;
        // One `snapshot` (single ArcSwap load) so `head_status`, derived from
        // `(core.workspace_revision, head.workspace_revision)`, can't tear
        // against a concurrent head-rotation publish landing between separate
        // `core()` and `heads()` loads.
        let (core, heads) = cell.snapshot();
        let head_statuses = heads
            .heads
            .iter()
            .map(|h| {
                crate::common::workspace::HeadStatus::from_revisions(
                    &h.workspace_revision,
                    &core.workspace_revision,
                )
            })
            .collect();
        Ok(crate::file_mgr::WorkspaceSummary {
            core,
            heads,
            head_statuses,
        })
    }

    pub(crate) fn workspace_dir(&self, id: &WorkspaceId) -> PathBuf {
        workspace_dir_for(&self.root, id)
    }
}
