//! Training endpoints. Body is the flattened [`TrainingCfg`]; the trainer always walks
//! `<workspace>/datasets/`. Backbone comes from [`crate::api::AppState::training_backbone_path`]
//! (resolved at boot from the first `kind = "burn"` launch candidate).

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

    // Resolve backbone before the workspace check so a missing Burn candidate
    // (deployment misconfig) surfaces immediately, not behind a valid workspace lookup.
    let backbone_path = state.training_backbone_path.ok_or_else(|| {
        ApiError::File(crate::file_mgr::io_err(
            "<launch.backbone.candidates[kind=\"burn\"]>",
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no Burn backbone configured in launch TOML",
            ),
        ))
    })?;

    // Workspace existence/revision snapshot + backbone file probe run on the
    // blocking pool to keep the runtime free under eMMC pressure.
    let workspace_id_for_check = workspace_id;
    let files_for_check = state.files.clone();
    let backbone_for_check = backbone_path.clone();
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
            // `metadata` not `symlink_metadata` so the probe follows symlinks like the
            // loaders do (a symlink-to-.mpk inference uses must not 400); NotFound still
            // distinguished from EACCES below.
            match std::fs::metadata(&backbone_for_check) {
                Ok(md) if md.is_file() => {}
                Ok(_) => {
                    return Err(ApiError::File(crate::file_mgr::io_err(
                        backbone_for_check.display(),
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "deployment backbone path is not a regular file",
                        ),
                    )));
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    return Err(ApiError::File(crate::file_mgr::io_err(
                        backbone_for_check.display(),
                        std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "deployment backbone not found",
                        ),
                    )));
                }
                Err(e) => {
                    return Err(ApiError::File(crate::file_mgr::io_err(
                        backbone_for_check.display(),
                        e,
                    )));
                }
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
        backbone_path,
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
