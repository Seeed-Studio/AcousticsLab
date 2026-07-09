//! REST API for the daemon; all routes under `/api/v1/*`. Asset upload
//! (`PUT .../assets/{*path}`) carries a 256 MiB body limit for full classifier bundles.

#![warn(missing_debug_implementations)]

mod error;
pub(crate) mod extract;
mod routes;
#[cfg(test)]
mod tests;

pub use error::ApiError;

use std::sync::Arc;

use crate::common::traits::head_store::HeadStore;
use crate::common::traits::lag_source::LagSource;
use crate::config::{ConfigHandle, MicSettingsHandle};
use crate::file_mgr::{FsService, JobRegistry};
use crate::inference::InferenceCfg;
use crate::status::StatusReporter;
use crate::training::TrainingRegistry;
use arc_swap::ArcSwap;
use axum::Router;
use axum::extract::FromRef;
use serde::Deserialize;

/// Threaded into every handler; cheap to clone (all-Arc). Trait-object fields let tests substitute mocks.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<dyn ConfigHandle>,
    pub head: Arc<dyn HeadStore>,
    /// Immutable boot catalogue + policy hot-swapped atomically on `POST /mic/policy`.
    pub mic_settings: Arc<dyn MicSettingsHandle>,
    pub inference_cfg: Arc<ArcSwap<InferenceCfg>>,
    pub files: Arc<dyn FsService>,
    pub monitor: Arc<dyn StatusReporter>,
    pub training: Arc<dyn TrainingRegistry>,
    pub broadcast_lag_reader: Arc<dyn LagSource>,
    /// Serializes `POST /active` read+stage+publish+install+prune so publish/install stay atomic and prune
    /// never races a peer publishing outside this request's keep-list. Lock order active -> per-workspace;
    /// sync (one `spawn_blocking` worker, no `.await`). Non-reentrant: `FsServiceImpl::{delete,
    /// start_delete_workspace, delete_head, publish_trained_head, publish_imported_head}` take this same Arc,
    /// so a holder must not call them (deadlock).
    pub active_mutex: Arc<parking_lot::Mutex<()>>,
    /// Bundled default head pair resolved at boot; sourced on `POST /active {default: true}`.
    pub default_head: Option<crate::config::DefaultHeadRef>,
    /// Trainer's ordered backbone candidates (launch catalogue filtered to kinds this build can
    /// load); the job resolves the first usable one, mirroring serving. Empty makes `POST /train`
    /// fail before admission.
    pub training_backbones: crate::inference::BackboneCatalogue,
    /// Serving's boot-loaded candidate (`None` = inference not running); passed to training jobs
    /// for the train/serve feature-basis warning.
    pub serving_backbone: Option<crate::inference::BackboneRef>,
    /// In-process job registry. Must be the same instance passed to `WorkspaceMgr::with_admission_and_jobs`
    /// so admission and the routes agree.
    pub jobs: Arc<JobRegistry>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("config_path", &self.config.path())
            .field("inference_cfg", &self.inference_cfg.load())
            .finish_non_exhaustive()
    }
}

// Hand-written `FromRef` impls (no `axum_macros` dep) so a handler extracts only the trait it needs.
impl FromRef<AppState> for Arc<dyn ConfigHandle> {
    fn from_ref(state: &AppState) -> Self {
        state.config.clone()
    }
}

impl FromRef<AppState> for Arc<dyn HeadStore> {
    fn from_ref(state: &AppState) -> Self {
        state.head.clone()
    }
}

impl FromRef<AppState> for Arc<dyn MicSettingsHandle> {
    fn from_ref(state: &AppState) -> Self {
        state.mic_settings.clone()
    }
}

impl FromRef<AppState> for Arc<ArcSwap<InferenceCfg>> {
    fn from_ref(state: &AppState) -> Self {
        state.inference_cfg.clone()
    }
}

impl FromRef<AppState> for Arc<dyn FsService> {
    fn from_ref(state: &AppState) -> Self {
        state.files.clone()
    }
}

impl FromRef<AppState> for Arc<dyn StatusReporter> {
    fn from_ref(state: &AppState) -> Self {
        state.monitor.clone()
    }
}

impl FromRef<AppState> for Arc<dyn TrainingRegistry> {
    fn from_ref(state: &AppState) -> Self {
        state.training.clone()
    }
}

impl FromRef<AppState> for Arc<dyn LagSource> {
    fn from_ref(state: &AppState) -> Self {
        state.broadcast_lag_reader.clone()
    }
}

impl FromRef<AppState> for Arc<JobRegistry> {
    fn from_ref(state: &AppState) -> Self {
        state.jobs.clone()
    }
}

/// Build the API router. The 404/405 fallbacks wrap unmatched-route/method-mismatch in the `{error, code}`
/// envelope so the wire-shape is uniform on every path. Trust posture: the daemon terminates no auth and
/// trusts every request reaching it; production fronts it with a reverse proxy owning TLS/auth/IP allow-listing.
pub fn router(state: AppState) -> Router {
    // Warn if `file.max_upload_bytes` exceeds axum's `DefaultBodyLimit`, else such uploads truncate silently.
    routes::dataset::boot_warn_if_misconfigured(state.files.as_ref());
    Router::new()
        .merge(routes::health::router())
        .merge(routes::mic::router())
        .merge(routes::inference::router())
        .merge(routes::active::router())
        .merge(routes::workspace::router())
        .merge(routes::dataset::router())
        .merge(routes::heads::router())
        .merge(routes::training::router())
        .merge(routes::status::router())
        .merge(routes::converter::router())
        .merge(routes::jobs::router())
        .fallback(fallback_404)
        .method_not_allowed_fallback(fallback_405)
        .with_state(state)
}

async fn fallback_404(req: axum::http::Request<axum::body::Body>) -> error::ApiError {
    error::ApiError::NotFound(format!("no route matched: {} {}", req.method(), req.uri()))
}

async fn fallback_405(req: axum::http::Request<axum::body::Body>) -> error::ApiError {
    error::ApiError::MethodNotAllowed {
        method: req.method().to_string(),
        path: req.uri().path().to_string(),
    }
}

pub fn router_v1_nested(state: AppState) -> Router {
    Router::new().nest("/api/v1", router(state))
}

// CORS intentionally unhandled: the fronting reverse proxy owns cross-origin.

/// Read-your-writes query (`?min_version=N`); `None` accepts the current snapshot at any version.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct VersionQuery {
    #[serde(default)]
    pub(crate) min_version: Option<u64>,
}

/// Gate the snapshot on `min_version`: `Ok(())` when met or omitted, else non-blocking
/// [`ApiError::TooEarly`] (HTTP 425) so callers retry once their write's
/// [`crate::common::version::SwapReceipt`] settles.
pub(crate) fn check_min_version(
    current: crate::common::version::ResourceVersion,
    requested: Option<u64>,
) -> Result<(), ApiError> {
    let cur: u64 = current.get();
    if let Some(req) = requested
        && cur < req
    {
        return Err(ApiError::TooEarly {
            requested: req,
            current: cur,
        });
    }
    Ok(())
}
