//! Workspace lifecycle (`/workspaces`, `/workspaces/{id}`).
//!
//! Handlers take `Path<String>` then `WorkspaceId::parse`, not `Path<WorkspaceId>`:
//! axum's `PathRejection` returns plain-text 400s that bypass the `{error, code}` envelope.

use std::sync::Arc;

use crate::common::ids::WorkspaceId;
use crate::common::workspace::{HeadStatus, WorkspaceCore, WorkspaceRevision};
use crate::file_mgr::FsService;
use axum::Router;
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Json};
use axum::routing::get;
use serde::{Deserialize, Serialize};
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiJson;

#[derive(Serialize)]
struct WorkspaceListResp {
    workspaces: Vec<WorkspaceListEntry>,
    /// Ids enumerated but whose `summary` failed to load, surfaced so a corrupt
    /// workspace cannot silently vanish. Omitted when empty (unchanged healthy wire shape).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    unreadable: Vec<String>,
}

#[derive(Serialize)]
struct WorkspaceListEntry {
    id: String,
    name: String,
    created_at: String,
}

async fn list_workspaces(
    State(files): State<Arc<dyn FsService>>,
) -> Result<Json<WorkspaceListResp>, ApiError> {
    // Sequential by design: parallelizing the ~1ms/id cold reads is not worth
    // pulling rayon onto the api surface at the workspace counts we target.
    let resp = task::spawn_blocking(move || -> Result<_, ApiError> {
        let ids = files.list_workspaces()?;
        let mut out = Vec::with_capacity(ids.len());
        let mut unreadable: Vec<String> = Vec::new();
        // Per-id failure records into `unreadable` rather than short-circuiting so
        // one corrupt `workspace.json` cannot hide every workspace; the enumerate
        // `?` above still propagates (a root read failure is a genuine 500).
        for id in ids {
            match files.summary(&id) {
                Ok(summary) => {
                    out.push(WorkspaceListEntry {
                        id: summary.core.id.to_string(),
                        name: summary.core.name.clone(),
                        created_at: summary.core.created_at.clone(),
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        target: "api",
                        id = %id,
                        err = %e,
                        "list_workspaces: skipping workspace whose summary failed to load",
                    );
                    unreadable.push(id.to_string());
                }
            }
        }
        Ok(WorkspaceListResp {
            workspaces: out,
            unreadable,
        })
    })
    .await??;
    Ok(Json(resp))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateWorkspaceReq {
    name: String,
    /// Per-tag validation runs daemon-side.
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Serialize)]
struct CreateWorkspaceResp {
    id: String,
    name: String,
    tags: Vec<String>,
    created_at: String,
    workspace_revision: WorkspaceRevision,
}

fn workspace_lifecycle_resp(core: &WorkspaceCore) -> CreateWorkspaceResp {
    CreateWorkspaceResp {
        id: core.id.to_string(),
        name: core.name.clone(),
        tags: core.tags.clone(),
        created_at: core.created_at.clone(),
        workspace_revision: core.workspace_revision.clone(),
    }
}

async fn create_workspace(
    State(files): State<Arc<dyn FsService>>,
    ApiJson(req): ApiJson<CreateWorkspaceReq>,
) -> Result<Json<CreateWorkspaceResp>, ApiError> {
    if req.name.is_empty() {
        return Err(ApiError::Bad("name must be non-empty".into()));
    }
    let CreateWorkspaceReq { name, tags } = req;
    // Create + first-summary in ONE `spawn_blocking`: no await window for a
    // concurrent DELETE to race the fresh id into a 404.
    let summary = task::spawn_blocking(move || -> Result<_, crate::file_mgr::FsError> {
        let id = files.create_with_tags(&name, &tags)?;
        files.summary(&id)
    })
    .await??;
    Ok(Json(workspace_lifecycle_resp(&summary.core)))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PatchWorkspaceReq {
    /// New display name; uniqueness/charset enforced daemon-side.
    #[serde(default)]
    name: Option<String>,
    /// New tag set; replaces (does not merge with) the existing tags.
    #[serde(default)]
    tags: Option<Vec<String>>,
}

/// Atomic update of one/both metadata fields; operator metadata only, so it does
/// NOT advance `workspace_revision` or head freshness.
async fn patch_workspace(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
    ApiJson(req): ApiJson<PatchWorkspaceReq>,
) -> Result<Json<CreateWorkspaceResp>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    if req.name.is_none() && req.tags.is_none() {
        return Err(ApiError::Bad(
            "PATCH /workspaces requires at least one of `name` or `tags`".into(),
        ));
    }
    let PatchWorkspaceReq { name, tags } = req;
    let core =
        task::spawn_blocking(move || files.patch_workspace(&id, name.as_deref(), tags.as_deref()))
            .await?
            .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok(Json(workspace_lifecycle_resp(&core)))
}

/// Per-head DTO shared by `GET /workspaces/{id}` and `.../heads`; field
/// names/order/types pin the on-wire JSON.
#[derive(Serialize)]
pub(crate) struct HeadResponseEntry {
    pub(crate) head_id: String,
    pub(crate) workspace_revision: WorkspaceRevision,
    pub(crate) sha256: String,
    pub(crate) n_classes: u32,
    pub(crate) size_bytes: u64,
    pub(crate) created_at: String,
    pub(crate) status: HeadStatus,
}

pub(crate) fn head_response_entry(
    rec: &crate::common::workspace::HeadRecord,
    status: HeadStatus,
) -> HeadResponseEntry {
    HeadResponseEntry {
        head_id: rec.head_id.to_string(),
        workspace_revision: rec.workspace_revision.clone(),
        sha256: rec.sha256.clone(),
        n_classes: rec.n_classes,
        size_bytes: rec.size_bytes,
        created_at: rec.created_at.clone(),
        status,
    }
}

#[derive(Serialize)]
struct WorkspaceSummaryResp {
    id: String,
    name: String,
    created_at: String,
    workspace_revision: WorkspaceRevision,
    heads: Vec<HeadResponseEntry>,
}

/// Summary read; does NOT walk workspace files (`.../assets` owns that).
async fn get_workspace(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
) -> Result<Json<WorkspaceSummaryResp>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let summary = summary_or_404(&files, id).await?;
    let heads = summary
        .heads
        .heads
        .iter()
        .zip(summary.head_statuses.iter())
        .map(|(rec, status)| head_response_entry(rec, *status))
        .collect();
    Ok(Json(WorkspaceSummaryResp {
        id: summary.core.id.to_string(),
        name: summary.core.name.clone(),
        created_at: summary.core.created_at.clone(),
        workspace_revision: summary.core.workspace_revision.clone(),
        heads,
    }))
}

#[derive(Serialize)]
struct DeleteWorkspaceResp {
    job_id: String,
}

/// Async delete: stage the tree, return the `JobId`, drain off the hot path.
/// `202 Accepted` = tree durable but byte drain still running (matches assets-delete).
async fn delete_workspace(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
) -> Result<axum::response::Response, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let job_id = task::spawn_blocking(move || files.start_delete_workspace(&id))
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok((
        axum::http::StatusCode::ACCEPTED,
        Json(DeleteWorkspaceResp {
            job_id: job_id.to_string(),
        }),
    )
        .into_response())
}

/// Re-classify a `summary`/`read_metadata` failure at the API boundary: `file_mgr`
/// wraps every IO failure (incl. missing `workspace.json`) as `Internal`, so promote
/// a genuine ENOENT to `NotFound` while other shapes (EACCES, ENOSPC) stay `Internal`
/// to avoid a misleading 404.
pub(crate) fn classify_workspace_existence_error(
    id: &WorkspaceId,
    err: crate::file_mgr::FsError,
) -> ApiError {
    use crate::common::error::{Categorized, ErrorKind};
    let kind = err.kind();
    if matches!(kind, ErrorKind::NotFound) {
        return ApiError::NotFound(format!("workspace {id} not found"));
    }
    if matches!(kind, ErrorKind::Internal) && first_io_error_is_not_found(&err) {
        // `err`'s Display leaks the absolute path and 4xx bodies are unscrubbed
        // (only 5xx are), so log the full chain and return a path-free 404 body.
        tracing::debug!(
            target: "api",
            %id,
            err = %err,
            "workspace existence probe: ENOENT promoted to 404",
        );
        return ApiError::NotFound(format!("workspace {id} not found"));
    }
    err.into()
}

/// True iff the *first* `io::Error` in `err`'s source chain is `NotFound`.
pub(crate) fn first_io_error_is_not_found(err: &crate::file_mgr::FsError) -> bool {
    let mut src = std::error::Error::source(err);
    while let Some(s) = src {
        if let Some(io_err) = s.downcast_ref::<std::io::Error>() {
            return io_err.kind() == std::io::ErrorKind::NotFound;
        }
        src = s.source();
    }
    false
}

/// `files.summary(id)` on the blocking pool with existence-error reclassification;
/// clones `files` so the caller keeps it after the await.
pub(crate) async fn summary_or_404(
    files: &Arc<dyn FsService>,
    id: WorkspaceId,
) -> Result<crate::file_mgr::WorkspaceSummary, ApiError> {
    let files_for_spawn = files.clone();
    task::spawn_blocking(move || files_for_spawn.summary(&id))
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/workspaces", get(list_workspaces).post(create_workspace))
        .route(
            "/workspaces/{id}",
            get(get_workspace)
                .patch(patch_workspace)
                .delete(delete_workspace),
        )
}
