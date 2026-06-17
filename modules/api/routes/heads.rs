//! Trained-head routes under `/workspaces/{id}/heads/...`: list/get are
//! cache-only (never walk disk); delete is synchronous, 409 on overlapping running jobs.

use std::sync::Arc;

use crate::common::ids::{HeadId, WorkspaceId};
use crate::common::workspace::HeadManifest;
use crate::file_mgr::FsService;
use axum::Router;
use axum::extract::{Path, State};
use axum::response::Json;
use axum::routing::get;
use serde::Serialize;
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::routes::workspace::{
    HeadResponseEntry, classify_workspace_existence_error, head_response_entry, summary_or_404,
};

#[derive(Serialize)]
struct ListHeadsResp {
    heads: Vec<HeadResponseEntry>,
}

/// Cache-only: status derived from `workspace_revision`, no disk walk.
async fn list_heads(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
) -> Result<Json<ListHeadsResp>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let summary = summary_or_404(&files, id).await?;
    let heads = summary
        .heads
        .heads
        .iter()
        .zip(summary.head_statuses.iter())
        .map(|(rec, status)| head_response_entry(rec, *status))
        .collect();
    Ok(Json(ListHeadsResp { heads }))
}

/// Gate on the cached index before reading the manifest so orphan `.json` residue
/// 404s instead of leaking its contents.
async fn get_head_manifest(
    State(files): State<Arc<dyn FsService>>,
    Path((id, head_id)): Path<(String, String)>,
) -> Result<Json<HeadManifest>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let head_id = HeadId::parse(&head_id)?;
    let summary = summary_or_404(&files, id).await?;
    let in_index = summary.heads.heads.iter().any(|rec| rec.head_id == head_id);
    if !in_index {
        return Err(ApiError::NotFound(format!(
            "head {head_id} not in workspace {id} (heads.json index)"
        )));
    }
    let workspace_dir = crate::file_mgr::schema::workspace_dir_for(files.root(), &id);
    let manifest = task::spawn_blocking(move || {
        crate::file_mgr::schema::read_head_manifest(&workspace_dir, head_id)
    })
    .await?
    .map_err(|e| classify_workspace_existence_error(&id, crate::file_mgr::FsError::new(e)))?;
    Ok(Json(manifest))
}

#[derive(Serialize)]
struct DeleteHeadResp {
    deleted_head_id: String,
}

/// Synchronous single-head delete; the handler omits `AppState::active_mutex` because
/// `FsService::delete_head` takes it itself (active -> workspace lock order). Failures
/// classify as 404 (missing) or 409 (running job overlap).
async fn delete_head(
    State(files): State<Arc<dyn FsService>>,
    Path((id, head_id)): Path<(String, String)>,
) -> Result<Json<DeleteHeadResp>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let head_id = HeadId::parse(&head_id)?;
    task::spawn_blocking(move || files.delete_head(&id, head_id))
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok(Json(DeleteHeadResp {
        deleted_head_id: head_id.to_string(),
    }))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/workspaces/{id}/heads", get(list_heads))
        .route(
            "/workspaces/{id}/heads/{head_id}",
            get(get_head_manifest).delete(delete_head),
        )
}
