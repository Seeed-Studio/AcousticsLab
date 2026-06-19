//! `POST`/`GET /active` -- active-head activation and read. `POST` stages a
//! generation, pre-loads+validates, atomically publishes `current.json`, then
//! installs the candidate into `HotHead` (on-disk and runtime stay in sync);
//! activations serialize through `active_mutex` and staging fails closed if its
//! source disappears mid-copy.

use std::path::Path;

use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::post;
use serde::{Deserialize, Serialize};
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiJson;
use crate::common::ids::{HeadId, WorkspaceId};
use crate::common::workspace::{ActiveHeadManifest, ActiveOrigin, WorkspaceRevision};
use crate::file_mgr::active_head_writer::{
    ActivationError, ActivationOriginInput, ActivationResult, DefaultHeadSource, HeadInnerLoader,
    PendingActivation, prune_old_generations, publish_active_generation,
    stage_and_validate_activation, staging_path_for,
};

/// `{workspace_id, head_id}` or `{default: true}`. `untagged` so per-variant
/// `deny_unknown_fields` (which does not compose at the parent) surfaces stray
/// keys as a 400 bad-request (`ApiError::Bad` -> `UserInput`).
#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum ActivateRequest {
    Head(HeadActivate),
    Default(DefaultActivate),
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct HeadActivate {
    workspace_id: WorkspaceId,
    head_id: HeadId,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct DefaultActivate {
    default: bool,
}

/// Wire shape mirroring [`ActiveHeadManifest`] with `origin` flattened.
#[derive(Serialize, Debug)]
struct ActiveResp {
    sha256: String,
    labels_sha256: String,
    n_classes: u32,
    labels: Vec<String>,
    runtime_head_id: String,
    activated_at: String,
    /// `"head"` | `"default"`.
    origin: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_workspace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_head_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspace_revision: Option<WorkspaceRevision>,
    activation_id: String,
    /// GET-only, head-origin: whether the source workspace dir still exists;
    /// `false` does not stop inference (the generation owns its bytes).
    #[serde(skip_serializing_if = "Option::is_none")]
    source_workspace_alive: Option<bool>,
}

impl ActiveResp {
    fn from_manifest(
        manifest: &ActiveHeadManifest,
        activation_id: &str,
        source_workspace_alive: Option<bool>,
    ) -> Self {
        let (origin, src_ws, src_head, src_rev) = match &manifest.origin {
            ActiveOrigin::Default => ("default", None, None, None),
            ActiveOrigin::Head {
                source_workspace_id,
                source_head_id,
                workspace_revision,
            } => (
                "head",
                Some(source_workspace_id.to_string()),
                Some(source_head_id.to_string()),
                Some(workspace_revision.clone()),
            ),
        };
        Self {
            sha256: manifest.sha256.clone(),
            labels_sha256: manifest.labels_sha256.clone(),
            n_classes: manifest.n_classes,
            labels: manifest.labels.clone(),
            runtime_head_id: manifest.runtime_head_id.to_string(),
            activated_at: manifest.activated_at.clone(),
            origin,
            source_workspace_id: src_ws,
            source_head_id: src_head,
            workspace_revision: src_rev,
            activation_id: activation_id.to_string(),
            source_workspace_alive,
        }
    }
}

/// `POST /active`: under `active_mutex`, snapshot prior `current.json`, stage +
/// validate + pre-load, atomically publish, install, then prune.
async fn post_active(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<ActivateRequest>,
) -> Result<Json<ActiveResp>, ApiError> {
    // `untagged` accepts `default: false` as the Default arm; reject it.
    if let ActivateRequest::Default(DefaultActivate { default }) = &req
        && !*default
    {
        return Err(ApiError::Bad(
            "`default` must be true; use `{workspace_id, head_id}` to activate a workspace head"
                .into(),
        ));
    }

    // Head-origin fast-fail OUTSIDE active_mutex so a typo'd id costs nothing
    // under contention; staging re-checks and fails closed if the source dies.
    if let ActivateRequest::Head(HeadActivate {
        workspace_id,
        head_id,
    }) = &req
    {
        let workspace_id = *workspace_id;
        let head_id = *head_id;
        let summary =
            crate::api::routes::workspace::summary_or_404(&state.files, workspace_id).await?;
        if !summary.heads.heads.iter().any(|r| r.head_id == head_id) {
            return Err(ApiError::NotFound(format!(
                "head {head_id} not in workspace {workspace_id} (heads.json index)"
            )));
        }
    }

    // Whole read+stage+publish+install+prune chain runs on one spawn_blocking
    // worker holding active_mutex end-to-end (sync parking_lot::Mutex, never
    // crosses .await) so publish+install can't reorder (current.json vs runtime
    // HotHead would diverge) and prune can't race a peer publish (peer dir falls
    // outside `keep`, gets deleted, leaving current.json dangling). DO NOT call
    // any lock-taking FsService mutator here (delete, start_delete_workspace,
    // delete_head, publish_trained_head, publish_imported_head): the
    // active_mutex is non-reentrant and deadlocks. Free fns in
    // active_head_writer/schema, files.root(), state.head.* take no lock.
    let active_mutex = state.active_mutex.clone();
    let files = state.files.clone();
    let default_head = state.default_head.clone();
    let head = state.head.clone();
    let (activation_id, manifest) =
        task::spawn_blocking(move || -> Result<(String, ActiveHeadManifest), ApiError> {
            let _guard = active_mutex.lock();

            // Prior activation id for the prune keep-list; absent = first-boot/wiped.
            let previous_activation_id =
                match crate::file_mgr::schema::read_active_current(files.root()) {
                    Ok(p) => Some(p.activation_id),
                    Err(crate::file_mgr::FileError::Io { source, .. })
                        if source.kind() == std::io::ErrorKind::NotFound =>
                    {
                        None
                    }
                    Err(e) => return Err(ApiError::File(e)),
                };

            // Stage+validate+publish (writes current.json): lacking a lock-only
            // FsService handle, staging fails closed (NotFound/HashMismatch) if
            // the source dies mid-flight; a published generation owns
            // independent bytes, immune to later deletes.
            let result = match req {
                ActivateRequest::Head(HeadActivate {
                    workspace_id,
                    head_id,
                }) => {
                    let workspace_dir =
                        crate::file_mgr::schema::workspace_dir_for(files.root(), &workspace_id);
                    stage_and_publish_activation(
                        files.root(),
                        ActivationOriginInput::Head {
                            workspace_dir: &workspace_dir,
                            workspace_id,
                            head_id,
                        },
                    )?
                }
                ActivateRequest::Default(_) => {
                    let default_head = default_head
                        .as_ref()
                        .ok_or_else(|| ApiError::Bad("head.default not configured".into()))?;
                    stage_and_publish_activation(
                        files.root(),
                        ActivationOriginInput::Default {
                            source: DefaultHeadSource {
                                path: &default_head.path,
                                labels_path: &default_head.labels_path,
                            },
                        },
                    )?
                }
            };

            // Install AFTER current.json is durable: a failure leaves on-disk
            // (current.json NEW) diverged from runtime (HotHead OLD), so abort
            // rather than 500-with-daemon-up -- the supervisor restarts and boot
            // recovery re-derives from current.json. Receipt dropped because
            // read-your-write goes through the next GET /active.
            match head.install_prevalidated(result.candidate) {
                Ok(_swap_receipt) => {}
                Err(e) => {
                    tracing::error!(
                        target: "acoustics",
                        err = %e,
                        activation_id = %result.activation_id,
                        "install_prevalidated failed after current.json publish; \
                         on-disk active generation diverges from runtime HotHead",
                    );
                    // abort() skips the appender's WorkerGuard::drop, so write
                    // stderr synchronously (journald-visible); 50ms sleep lets
                    // the appender thread drain the tracing::error above first.
                    eprintln!(
                        "acousticslabd: ABORT -- install_prevalidated divergence \
                         (activation_id={}, err={}); supervisor must restart",
                        result.activation_id, e,
                    );
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    std::process::abort();
                }
            }

            // Best-effort prune; failure logs and continues -- boot recovery
            // sweeps residue.
            let keep: Vec<&str> = std::iter::once(result.activation_id.as_str())
                .chain(previous_activation_id.as_deref())
                .collect();
            if let Err(e) = prune_old_generations(files.root(), &keep) {
                tracing::warn!(
                    target: "acoustics",
                    err = %e,
                    activation_id = %result.activation_id,
                    "active generation prune failed; residue will be swept on next boot",
                );
            }

            Ok((result.activation_id, result.manifest))
        })
        .await??;

    Ok(Json(ActiveResp::from_manifest(
        &manifest,
        &activation_id,
        None,
    )))
}

fn stage_and_publish_activation(
    root: &Path,
    origin_input: ActivationOriginInput<'_>,
) -> Result<ActivationResult, ActivationError> {
    let staged = stage_and_validate_activation(
        PendingActivation {
            root,
            origin_input,
            now_rfc3339: crate::file_mgr::now_rfc3339(),
        },
        &runtime_head_loader(),
    )?;
    publish_active_generation(
        root,
        &staging_path_for(root, &staged.activation_id),
        &staged.manifest,
        &staged.activation_id,
    )?;
    Ok(staged)
}

/// Pre-load closure for the activation primitive; boxed as `dyn Any + Send` to
/// keep the file_mgr primitive decoupled from the inference crate.
fn runtime_head_loader() -> Box<HeadInnerLoader> {
    Box::new(|head_mpk, labels, head_id| {
        let head = crate::inference::HotHead::load(head_mpk, labels, head_id)
            .map_err(|e| format!("{e}"))?;
        // Clone the VersionedSwap inner for the install-step downcast.
        let inner = (*head.snapshot()).clone();
        Ok(Box::new(inner) as Box<dyn std::any::Any + Send>)
    })
}

/// `GET /active`: read `current.json`, load+validate the pointed manifest, add
/// the GET-only `source_workspace_alive` field. Wait-free; no active_mutex.
async fn get_active(State(state): State<AppState>) -> Result<Json<ActiveResp>, ApiError> {
    let root = state.files.root().to_path_buf();
    // All FS touches in one spawn_blocking off the async hot path -- the alive
    // stat can take several ms under eMMC pressure.
    let (pointer, manifest, source_workspace_alive) = task::spawn_blocking(
        move || -> Result<
            (
                crate::file_mgr::ActiveCurrentPointer,
                ActiveHeadManifest,
                Option<bool>,
            ),
            ApiError,
        > {
            let pointer =
                crate::file_mgr::schema::read_active_current(&root).map_err(|e| match e {
                    crate::file_mgr::FileError::Io { source, .. }
                        if source.kind() == std::io::ErrorKind::NotFound =>
                    {
                        ApiError::NotFound("no active head: current.json absent".into())
                    }
                    other => ApiError::File(other),
                })?;
            let manifest =
                crate::file_mgr::schema::read_active_manifest(&root, &pointer.activation_id)?;
            // Validate before trusting fields: the daemon wrote this through
            // validating paths, so failure means on-disk drift -- 500 (not 400,
            // which blames the bodyless GET) hits the operator 5xx hook.
            manifest.validate().map_err(|e| {
                ApiError::File(crate::file_mgr::io_err(
                    format!("active manifest {}", pointer.activation_id),
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("active manifest validation: {e}"),
                    ),
                ))
            })?;
            let source_workspace_alive = match &manifest.origin {
                ActiveOrigin::Default => None,
                ActiveOrigin::Head {
                    source_workspace_id,
                    ..
                } => {
                    let ws_dir =
                        crate::file_mgr::schema::workspace_dir_for(&root, source_workspace_id);
                    // is_dir() collapses every io::Error to false; fine on this
                    // single-tenant tree where the only trigger is deletion.
                    Some(ws_dir.is_dir())
                }
            };
            Ok((pointer, manifest, source_workspace_alive))
        },
    )
    .await??;

    Ok(Json(ActiveResp::from_manifest(
        &manifest,
        &pointer.activation_id,
        source_workspace_alive,
    )))
}

pub fn router() -> Router<AppState> {
    Router::new().route("/active", post(post_active).get(get_active))
}
