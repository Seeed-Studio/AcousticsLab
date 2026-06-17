// Each test binary uses only some helpers, so the unused rest would flag dead.
#![allow(dead_code)]

//! Shared `AppState` fixture for the integration-test binaries: one definition
//! so an `AppState` shape change is a single-site edit. Pull in via
//! `mod api_fixtures;` and call `fresh_app_state(...)`; `pub` fields override after.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Method, Request, header};
use axum::response::Response;
use tower::ServiceExt;

use acoustics_lab::api::AppState;
use acoustics_lab::common::traits::head_store::HeadStore;
use acoustics_lab::common::traits::lag_source::{BroadcastLagSnapshot, LagSource};
use acoustics_lab::config::{
    Config, ConfigCell, DefaultHeadRef, LaunchConfig, MicSettingsCell, MicSettingsHandle,
};
use acoustics_lab::file_mgr::{FsService, FsServiceImpl};
use acoustics_lab::inference::{HeadInner, HotHead};
use acoustics_lab::status::{StatusMonitor, StatusReporter};
use acoustics_lab::training::{JobRegistry as TrainingJobRegistry, TrainingRegistry};
use arc_swap::ArcSwap;

/// Drain `resp` and parse it as JSON into `T`. The 4 MiB cap exceeds any
/// suite response; truncation tests must call `to_bytes` with their own bound.
pub async fn json_body<T: serde::de::DeserializeOwned>(resp: Response) -> T {
    let bytes = to_bytes(resp.into_body(), 1 << 22).await.expect("body");
    serde_json::from_slice(&bytes).expect("parse json")
}

/// One `oneshot` HTTP request against a `clone()`d `router` (so the original
/// stays reusable). `Some(s)` sends `s` as JSON; `None` sends an empty body.
pub async fn call(router: &Router, method: Method, path: &str, body: Option<&str>) -> Response {
    let mut req = Request::builder().method(method).uri(path);
    if body.is_some() {
        req = req.header("content-type", "application/json");
    }
    let body = body.map(|s| Body::from(s.to_string())).unwrap_or_default();
    let req = req.body(body).expect("build req");
    router.clone().oneshot(req).await.expect("oneshot")
}

/// `PUT /api/v1/workspaces/{ws}/assets/{*path}` with raw bytes. Each `path`
/// segment is percent-encoded so URI-invalid chars (backslash, NUL, CR/LF,
/// non-ASCII) survive the `uri()` parse and reach `AssetPath::parse` for the
/// path-traversal tests; `/` is kept verbatim to feed the wildcard tail.
pub async fn upload(router: &Router, ws: &str, path: &str, payload: &[u8]) -> Response {
    let encoded: String = path
        .split('/')
        .map(|seg| urlencoding::encode(seg).into_owned())
        .collect::<Vec<_>>()
        .join("/");
    let req = Request::builder()
        .method(Method::PUT)
        .uri(format!("/api/v1/workspaces/{ws}/assets/{encoded}"))
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .body(Body::from(payload.to_vec()))
        .expect("build req");
    router.clone().oneshot(req).await.expect("oneshot")
}

/// `POST /api/v1/workspaces` with name; asserts 200 and returns the new id.
/// Exercise 4xx create surfaces via [`call`] directly.
pub async fn create_workspace(router: &Router, name: &str) -> String {
    let resp = call(
        router,
        Method::POST,
        "/api/v1/workspaces",
        Some(&format!("{{\"name\":\"{name}\"}}")),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    v["id"].as_str().expect("id").to_string()
}

/// `LagSource` stub reporting zero drops, replacing the production WS fan-out
/// counters; tests asserting on lag counts plug in their own source.
#[derive(Debug, Clone, Copy, Default)]
pub struct StubLagSource(pub BroadcastLagSnapshot);

impl LagSource for StubLagSource {
    fn snapshot(&self) -> BroadcastLagSnapshot {
        self.0
    }
}

/// On-disk per-workspace dir. Doubled `workspaces`: [`fresh_app_state`] roots
/// the FsService at `<dir>/workspaces/`, then `WORKSPACES_DIR_NAME` nests a
/// second inside, landing at `<dir>/workspaces/workspaces/<id>/`.
pub fn fixture_workspace_dir(dir: &Path, ws_id: impl AsRef<Path>) -> PathBuf {
    dir.join("workspaces").join("workspaces").join(ws_id)
}

/// Canonical test `AppState` rooted at `dir`: persisted default `Config`, the
/// launch catalogue's `default-mock` candidate, a 2-class synthetic `HotHead`,
/// FsService over `dir/workspaces/`, fresh registries, and a stub `LagSource`.
/// `default_head` points at an absent pair under `dir/bundled_default` so an
/// accidental `POST /active {default: true}` fails closed; override for a real one.
pub fn fresh_app_state(dir: &Path) -> AppState {
    let cfg_path = dir.join("config.toml");
    let workspace_root = dir.join("workspaces");
    std::fs::create_dir_all(&workspace_root).expect("workspace root");
    let cfg = Config::default_for();
    let config = Arc::new(ConfigCell::from_value(cfg.clone(), cfg_path).expect("validate"));
    config.persist().expect("persist initial");
    // Ships a single `default-mock` candidate so `Fixed { id: "default-mock" }`
    // passes the config cell's cross-check, mirroring daemon boot.
    let launch = LaunchConfig::default_for();
    let mic_settings: Arc<dyn MicSettingsHandle> = Arc::new(MicSettingsCell::new(
        Arc::new(launch.mic),
        cfg.mic.clone(),
        config.clone(),
    ));
    let inference_cfg = Arc::new(ArcSwap::from_pointee(cfg.inference));
    let head: Arc<dyn HeadStore> = Arc::new(HotHead::from_inner(HeadInner {
        weight: vec![0.0; acoustics_lab::common::dims::BACKBONE_FEATURE_DIM * 2],
        bias: vec![0.0; 2],
        labels: vec!["a".into(), "b".into()],
        head_id: acoustics_lab::common::ids::HeadId::new(),
        n_classes: 2,
    }));
    let jobs = Arc::new(acoustics_lab::file_mgr::JobRegistry::new(
        acoustics_lab::file_mgr::JobRegistryCfg::default(),
    ));
    let active_mutex: Arc<parking_lot::Mutex<()>> = Arc::new(parking_lot::Mutex::new(()));
    let files: Arc<dyn FsService> = Arc::new(FsServiceImpl::with_admission_jobs_and_active_mutex(
        workspace_root,
        Default::default(),
        jobs.clone(),
        active_mutex.clone(),
    ));
    let monitor: Arc<dyn StatusReporter> = Arc::new(StatusMonitor::new());
    let training: Arc<dyn TrainingRegistry> = Arc::new(TrainingJobRegistry::new());
    AppState {
        config,
        head,
        mic_settings,
        inference_cfg,
        files,
        monitor,
        training,
        broadcast_lag_reader: Arc::new(StubLagSource::default()),
        active_mutex,
        default_head: Some(DefaultHeadRef {
            path: dir.join("bundled_default/head.mpk"),
            labels_path: dir.join("bundled_default/labels.txt"),
        }),
        // None: no fixture consumer exercises `POST /train`; the train pipeline
        // has its own router that stubs a backbone and wires it through here.
        training_backbone_path: None,
        jobs,
    }
}
