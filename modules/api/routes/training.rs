//! Training endpoints. Body is the flattened [`TrainingCfg`]; the trainer always walks
//! `<workspace>/datasets/`. Backbone candidates come from
//! [`crate::api::AppState::training_backbones`] (the launch catalogue filtered to supported
//! kinds at boot); the job resolves the first loadable one -- NPU first on device, Burn
//! `.mpk` fallback -- matching the serving resolution rule.

use std::sync::Arc;

use crate::common::ids::{HeadId, JobId, WorkspaceId};
use crate::common::workspace::{JobReference, JobType};
use crate::file_mgr::{FsService, JobRegistry, TrainingCfg, validate_training_cfg};
use crate::training::{TrainingJob, TrainingRegistry};
use axum::Router;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::response::Json;
use axum::routing::{get, post};
use serde::Serialize;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::{ApiJson, CONTROL_JSON_BODY_LIMIT};

/// Daemon-allocated identifiers the caller uses to observe progress.
#[derive(Debug, Serialize)]
struct TrainStartResp {
    /// Pre-allocated head id; the index entry is committed only when publish succeeds.
    head_id: String,
    job_id: String,
}

#[derive(Serialize)]
struct TrainingListResp {
    jobs: Vec<crate::training::JobView>,
}

async fn start_training(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(cfg): ApiJson<TrainingCfg>,
) -> Result<Json<TrainStartResp>, ApiError> {
    let workspace_id = WorkspaceId::parse(&id)?;
    // `ApiJson` enforced wire shape; this gates numeric ranges only.
    validate_training_cfg(&cfg).map_err(|e| ApiError::Bad(e.to_string()))?;

    // Resolve backbone candidates before the workspace check so an empty catalogue
    // (deployment misconfig) surfaces immediately, not behind a valid workspace lookup.
    let backbone_candidates = state.training_backbones.clone();
    if backbone_candidates.is_empty() {
        return Err(ApiError::File(crate::file_mgr::io_err(
            "<launch.backbone.candidates>",
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no training backbone candidate in launch TOML that this build can load",
            ),
        )));
    }

    // Workspace existence/revision snapshot + backbone file probe run on the
    // blocking pool to keep the runtime free under eMMC pressure.
    let workspace_id_for_check = workspace_id;
    let files_for_check = state.files.clone();
    let backbones_for_check = backbone_candidates.clone();
    let workspace_revision: crate::common::workspace::WorkspaceRevision =
        tokio::task::spawn_blocking(move || -> Result<_, ApiError> {
            // Classify so a missing workspace is 404 not 500: `summary` wraps
            // every IO error (incl. NotFound) as `FsError::Internal`.
            let summary = files_for_check
                .summary(&workspace_id_for_check)
                .map_err(|e| {
                    crate::api::routes::workspace::classify_workspace_existence_error(
                        &workspace_id_for_check,
                        e,
                    )
                })?;
            // Admit if ANY candidate file is present (the job-side resolver
            // skips unusable ones); else report the first candidate's failure
            // (declaration order = operator preference). `metadata` not
            // `symlink_metadata`: the probe follows symlinks like the loaders
            // do; NotFound still distinguished from EACCES below.
            let mut first_failure: Option<ApiError> = None;
            let mut any_usable = false;
            for cand in &backbones_for_check.candidates {
                let failure = match std::fs::metadata(&cand.path) {
                    Ok(md) if md.is_file() => {
                        any_usable = true;
                        break;
                    }
                    Ok(_) => crate::file_mgr::io_err(
                        cand.path.display(),
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "deployment backbone path is not a regular file",
                        ),
                    ),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => crate::file_mgr::io_err(
                        cand.path.display(),
                        std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "deployment backbone not found",
                        ),
                    ),
                    Err(e) => crate::file_mgr::io_err(cand.path.display(), e),
                };
                first_failure.get_or_insert(ApiError::File(failure));
            }
            if !any_usable && let Some(failure) = first_failure {
                return Err(failure);
            }
            Ok(summary.core.workspace_revision.clone())
        })
        .await??;

    // Allocate head id before spawn so the response returns it; publish reuses it verbatim.
    let head_id = HeadId::new();
    let job = TrainingJob {
        workspace_id,
        head_id,
        workspace_revision,
        training_cfg: cfg,
        backbone_candidates,
        serving_backbone: state.serving_backbone.clone(),
    };

    // Admission gate: takes the global `max_train_jobs = 1` slot and stamps a workspace
    // reference for `WorkspaceDelete` exclusion (2nd train -> `AnotherTrainRunning` 409);
    // handle released at terminal.
    let jobs: Arc<JobRegistry> = state.jobs.clone();
    let job_handle = jobs
        .try_acquire(
            JobType::Train,
            vec![JobReference::Workspace { workspace_id }],
            None,
        )
        .map_err(|c| ApiError::File(crate::file_mgr::FileError::from(c)))?;
    let job_id = job_handle.job_id();

    let training_for_spawn = state.training.clone();
    let files_for_spawn = state.files.clone();
    tokio::task::spawn_blocking(move || {
        training_for_spawn.spawn(files_for_spawn, job, Some(job_handle))
    })
    .await??;
    Ok(Json(TrainStartResp {
        head_id: head_id.to_string(),
        job_id: job_id.to_string(),
    }))
}

async fn list_training(
    State(training): State<Arc<dyn TrainingRegistry>>,
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
) -> Result<Json<TrainingListResp>, ApiError> {
    let workspace_id = WorkspaceId::parse(&id)?;
    // Existence proof so a missing workspace is 404, not `200 { jobs: [] }`.
    crate::api::routes::workspace::summary_or_404(&files, workspace_id).await?;
    Ok(Json(TrainingListResp {
        jobs: training.list_for_workspace(&workspace_id),
    }))
}

async fn get_training(
    State(training): State<Arc<dyn TrainingRegistry>>,
    Path((id, job)): Path<(String, String)>,
) -> Result<Json<crate::training::JobView>, ApiError> {
    let workspace_id = WorkspaceId::parse(&id)?;
    let job_id = JobId::parse(&job)?;
    Ok(Json(training.status(&workspace_id, job_id)?))
}

async fn cancel_training(
    State(training): State<Arc<dyn TrainingRegistry>>,
    Path((id, job)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let workspace_id = WorkspaceId::parse(&id)?;
    let job_id = JobId::parse(&job)?;
    training.cancel(&workspace_id, job_id)?;
    Ok(Json(serde_json::json!({"ok": true})))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/workspaces/{id}/train", post(start_training))
        .route("/workspaces/{id}/training", get(list_training))
        .route(
            "/workspaces/{id}/training/{job}",
            get(get_training).delete(cancel_training),
        )
        // Router-wide, but only the train POST carries a body.
        .layer(DefaultBodyLimit::max(CONTROL_JSON_BODY_LIMIT))
}
