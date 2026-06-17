//! Object-safe `TrainingRegistry` over [`JobRegistry`], held as a trait object
//! so tests can substitute mocks.

use crate::common::ids::{JobId, WorkspaceId};
use crate::file_mgr::{FsService, JobHandle};
use crate::training::{JobRegistry, JobView, TrainingError, TrainingJob};
use std::sync::Arc;

/// Submit + observe + cancel in-process training jobs; `max_train_jobs = 1`
/// admission is enforced upstream by [`crate::file_mgr::JobRegistry`].
pub trait TrainingRegistry: Send + Sync + std::fmt::Debug {
    /// Re-validates the wire `TrainingCfg` then runs on a `spawn_blocking`
    /// worker. `Some(job_handle)` bridges events to `/jobs` + SSE; `None` is
    /// test-only (JSONL backstop, no snapshot).
    fn spawn(
        &self,
        files: Arc<dyn FsService>,
        job: TrainingJob,
        job_handle: Option<JobHandle>,
    ) -> Result<JobId, TrainingError>;

    /// Sets the cancel flag; the blocking task observes it at its next progress
    /// emit, exiting as [`crate::training::JobState::Cancelled`].
    fn cancel(&self, workspace_id: &WorkspaceId, job_id: JobId) -> Result<(), TrainingError>;

    fn status(&self, workspace_id: &WorkspaceId, job_id: JobId) -> Result<JobView, TrainingError>;

    fn list_for_workspace(&self, workspace_id: &WorkspaceId) -> Vec<JobView>;

    /// Pre-drain hook: set the cancel flag on every running job so blocking
    /// trainers observe shutdown at once; returns count newly set.
    fn cancel_all_for_shutdown(&self) -> usize;

    fn active_count(&self) -> usize;
}

impl TrainingRegistry for JobRegistry {
    fn spawn(
        &self,
        files: Arc<dyn FsService>,
        job: TrainingJob,
        job_handle: Option<JobHandle>,
    ) -> Result<JobId, TrainingError> {
        JobRegistry::spawn(self, files, job, job_handle)
    }
    fn cancel(&self, workspace_id: &WorkspaceId, job_id: JobId) -> Result<(), TrainingError> {
        JobRegistry::cancel(self, workspace_id, job_id)
    }
    fn status(&self, workspace_id: &WorkspaceId, job_id: JobId) -> Result<JobView, TrainingError> {
        JobRegistry::status(self, workspace_id, job_id)
    }
    fn list_for_workspace(&self, workspace_id: &WorkspaceId) -> Vec<JobView> {
        JobRegistry::list_for_workspace(self, workspace_id)
    }
    fn cancel_all_for_shutdown(&self) -> usize {
        JobRegistry::cancel_all_for_shutdown(self)
    }
    fn active_count(&self) -> usize {
        JobRegistry::active_count(self)
    }
}

#[cfg(test)]
const _: fn() = || {
    fn assert_obj_safe<T: ?Sized>() {}
    assert_obj_safe::<dyn TrainingRegistry>();
};
