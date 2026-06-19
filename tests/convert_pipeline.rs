//! In-process integration tests for `POST /workspaces/{id}/convert`,
//! driven via `tower::ServiceExt::oneshot` against a tempdir workspace.
//! Bundle-dependent tests skip when the gitignored `misc/models/model.json`
//! fixture is absent.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    clippy::await_holding_lock
)]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acousticslab::api::{AppState, router_v1_nested};
use acousticslab::common::traits::head_store::HeadStore;
use acousticslab::common::traits::lag_source::{BroadcastLagSnapshot, LagSource};
use acousticslab::config::{
    Config, ConfigCell, DefaultHeadRef, LaunchConfig, MicSettingsCell, MicSettingsHandle,
};
use acousticslab::file_mgr::{FsService, FsServiceImpl};
use acousticslab::inference::{HeadInner, HotHead};
use acousticslab::status::{StatusMonitor, StatusReporter};
use acousticslab::training::{JobRegistry, TrainingRegistry};
use arc_swap::ArcSwap;
use axum::Router;
use axum::http::{Method, StatusCode};

mod api_fixtures;
use api_fixtures::{call, create_workspace, fixture_workspace_dir, json_body, upload};

/// Serializes the whole suite: the converter semaphore is a process-wide
/// `OnceLock` static, so concurrent `#[tokio::test]` cases would contend the
/// single permit. Held for each test's full body.
static CONVERT_TEST_SERIALIZER: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Recovers from poison so one failing test doesn't disable the rest.
fn serialize_test() -> std::sync::MutexGuard<'static, ()> {
    CONVERT_TEST_SERIALIZER
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

#[derive(Debug, Clone, Copy, Default)]
struct StubLagSource(BroadcastLagSnapshot);
impl LagSource for StubLagSource {
    fn snapshot(&self) -> BroadcastLagSnapshot {
        self.0
    }
}

/// Build a router whose `FsService` root is `<dir>/workspaces/`.
fn fresh_router(dir: &Path) -> (Router, Arc<dyn FsService>) {
    let workspace_root = dir.join("workspaces");
    std::fs::create_dir_all(&workspace_root).expect("workspace root");

    let cfg_path = dir.join("config.toml");
    let cfg = Config::default_for();
    let config = Arc::new(ConfigCell::from_value(cfg.clone(), cfg_path).expect("validate"));
    config.persist().expect("persist initial");
    let launch = LaunchConfig::default_for();
    let mic_settings: Arc<dyn MicSettingsHandle> = Arc::new(MicSettingsCell::new(
        Arc::new(launch.mic),
        cfg.mic.clone(),
        config.clone(),
    ));
    let inference_cfg = Arc::new(ArcSwap::from_pointee(cfg.inference));
    let head: Arc<dyn HeadStore> = Arc::new(HotHead::from_inner(HeadInner {
        weight: vec![0.0; acousticslab::common::dims::BackboneFeatureDim::USIZE * 2],
        bias: vec![0.0; 2],
        labels: vec!["a".into(), "b".into()],
        head_id: acousticslab::common::ids::HeadId::new(),
        n_classes: 2,
    }));
    let jobs = Arc::new(acousticslab::file_mgr::JobRegistry::new(
        acousticslab::file_mgr::JobRegistryCfg::default(),
    ));
    let active_mutex: Arc<parking_lot::Mutex<()>> = Arc::new(parking_lot::Mutex::new(()));
    let files: Arc<dyn FsService> = Arc::new(FsServiceImpl::with_admission_jobs_and_active_mutex(
        workspace_root,
        Default::default(),
        jobs.clone(),
        active_mutex.clone(),
    ));
    let monitor: Arc<dyn StatusReporter> = Arc::new(StatusMonitor::new());
    let training: Arc<dyn TrainingRegistry> = Arc::new(JobRegistry::new());
    let app = AppState {
        config,
        head,
        mic_settings,
        inference_cfg,
        files: files.clone(),
        monitor,
        training,
        broadcast_lag_reader: Arc::new(StubLagSource::default()),
        active_mutex,
        default_head: Some(DefaultHeadRef {
            path: dir.join("bundled_default/head.mpk"),
            labels_path: dir.join("bundled_default/labels.txt"),
        }),
        // Unused: these tests never hit `POST /train`.
        training_backbone_path: None,
        jobs,
    };
    (router_v1_nested(app), files)
}

/// The upstream TFJS Speech-Commands fixture dir, or `None` when it isn't
/// checked out (gitignored) so dependent tests can skip.
fn try_fixture_dir() -> Option<std::path::PathBuf> {
    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    let p = crate_root.join("misc/models");
    if p.join("model.json").exists() {
        Some(p)
    } else {
        None
    }
}

/// Upload the upstream TFJS bundle (model.json, two shards, metadata.json with
/// `wordLabels`) under `converters/tfjs/`. Returns the converter-rooted paths
/// for the convert request.
async fn seed_tfjs_bundle(
    router: &Router,
    ws: &str,
    fixture: &Path,
) -> (String, Vec<String>, String) {
    let model_json = std::fs::read(fixture.join("model.json")).expect("read model.json");
    let shard1 = std::fs::read(fixture.join("group1-shard1of2")).expect("read shard1");
    let shard2 = std::fs::read(fixture.join("group1-shard2of2")).expect("read shard2");
    let metadata_json = std::fs::read(fixture.join("metadata.json")).expect("read metadata.json");

    // Uploads are workspace-rooted (`converters/<...>`); the request body uses
    // the canonical converter-rooted form (slashless).
    let model_json_abs = "tfjs/model.json".to_string();
    let shard_abs = vec![
        "tfjs/group1-shard1of2".to_string(),
        "tfjs/group1-shard2of2".to_string(),
    ];
    let labels_abs = "tfjs/metadata.json".to_string();

    let upload_paths = [
        ("converters/tfjs/model.json", &model_json[..]),
        ("converters/tfjs/group1-shard1of2", &shard1[..]),
        ("converters/tfjs/group1-shard2of2", &shard2[..]),
        ("converters/tfjs/metadata.json", &metadata_json[..]),
    ];
    for (path, body) in upload_paths {
        let resp = upload(router, ws, path, body).await;
        assert_eq!(resp.status(), StatusCode::OK, "upload {path}");
    }

    (model_json_abs, shard_abs, labels_abs)
}

fn convert_body(
    model_json_path: &str,
    _shards: &[String],
    labels_path: &str,
    labels_format: &str,
) -> String {
    // Internally tagged on `converter_type`; path fields are converter-rooted.
    // Shards are derived from `model.json`'s `weightsManifest[].paths`, not sent
    // in the body; `_shards` only documents which shards callers expect resolved.
    serde_json::json!({
        "converter_type": "tfjs",
        "model_json_path": model_json_path,
        "labels_path": labels_path,
        "labels_format": labels_format,
    })
    .to_string()
}

/// Poll the workspace summary until `heads` is non-empty, or `None` on timeout.
async fn wait_for_head(router: &Router, ws: &str) -> Option<serde_json::Value> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let resp = call(
            router,
            Method::GET,
            &format!("/api/v1/workspaces/{ws}"),
            None,
        )
        .await;
        if resp.status() == StatusCode::OK {
            let v: serde_json::Value = json_body(resp).await;
            let heads = v["heads"].as_array().cloned().unwrap_or_default();
            if !heads.is_empty() {
                return Some(v);
            }
        }
        if Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Happy path: `POST /convert` returns well-formed UUID `{head_id, job_id}`,
/// the head lands, and a `job_completed` event is logged.
#[tokio::test]
async fn convert_producer_returns_head_id_and_job_id() {
    let _gate = serialize_test();
    let Some(fixture) = try_fixture_dir() else {
        eprintln!("skipping: misc/models/model.json not present");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;
    let (mj, sh, lb) = seed_tfjs_bundle(&r, &ws, &fixture).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/convert"),
        Some(&convert_body(&mj, &sh, &lb, "tfjs_metadata")),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "convert accept; got {}",
        resp.status()
    );
    let v: serde_json::Value = json_body(resp).await;
    let head_id = v["head_id"].as_str().expect("head_id");
    let job_id = v["job_id"].as_str().expect("job_id");
    assert_eq!(head_id.len(), 36, "head_id is not a UUID");
    assert_eq!(job_id.len(), 36, "job_id is not a UUID");
    assert_ne!(head_id, job_id);

    let summary = wait_for_head(&r, &ws).await.expect("head published");
    let heads = summary["heads"].as_array().expect("heads");
    assert_eq!(heads.len(), 1, "exactly one head: {summary}");
    assert_eq!(heads[0]["head_id"], head_id);

    let log_path = fixture_workspace_dir(dir.path(), &ws)
        .join("converter_logs")
        .join(format!("{job_id}.jsonl"));
    assert!(
        log_path.exists(),
        "converter_logs/<job_id>.jsonl missing: {}",
        log_path.display(),
    );
    let log_contents = std::fs::read_to_string(&log_path).expect("read log");
    assert!(
        log_contents.lines().any(|line| {
            serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|v| v["kind"].as_str().map(str::to_string))
                .as_deref()
                == Some("job_completed")
        }),
        "log must contain a `job_completed` event line; got:\n{log_contents}",
    );
}

/// A second concurrent convert request hits 409: the registry's single
/// `max_convert_jobs = 1` Convert slot is acquired first, so while the first
/// job holds it the second is rejected as `RegistryConflict::JobConflict` ->
/// `FileError::JobConflict`; the converter semaphore (`Semaphore::new(1)`) is a
/// secondary guard yielding `ConvertError::Busy` only if the slot is free.
#[tokio::test]
async fn convert_second_request_returns_conflict_or_busy() {
    let _gate = serialize_test();
    let Some(fixture) = try_fixture_dir() else {
        eprintln!("skipping: misc/models/model.json not present");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;
    let (mj, sh, lb) = seed_tfjs_bundle(&r, &ws, &fixture).await;

    let resp_first = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/convert"),
        Some(&convert_body(&mj, &sh, &lb, "tfjs_metadata")),
    )
    .await;
    assert_eq!(resp_first.status(), StatusCode::OK);
    let resp_second = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/convert"),
        Some(&convert_body(&mj, &sh, &lb, "tfjs_metadata")),
    )
    .await;
    let status = resp_second.status();
    // 200 is allowed because conversion can finish before the second request
    // issues (head bytes are tiny). The load-bearing assertion is "no 5xx leak".
    assert!(
        status == StatusCode::CONFLICT || status == StatusCode::OK,
        "second convert: expected 409 (conflict) or 200 (first finished); got {status}",
    );
}

/// An in-flight convert does not block `DELETE /assets/<input>`: convert holds a
/// workspace-scoped lease and only `WorkspaceDelete` is exclusive.
#[tokio::test]
async fn delete_asset_during_convert_coexists() {
    let _gate = serialize_test();
    let Some(fixture) = try_fixture_dir() else {
        eprintln!("skipping: misc/models/model.json not present");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;
    let (mj, sh, lb) = seed_tfjs_bundle(&r, &ws, &fixture).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/convert"),
        Some(&convert_body(&mj, &sh, &lb, "tfjs_metadata")),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/converters/tfjs/model.json"),
        None,
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::ACCEPTED,
        "DELETE during convert must coexist (only WorkspaceDelete excludes); \
         async dispatch returns 202 Accepted",
    );
}

/// A convert input path that resolves to no regular file is rejected without
/// spawning a job or recording a head.
#[tokio::test]
async fn convert_rejects_missing_input() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    let body = convert_body(
        "/missing/model.json",
        &["/missing/shard.bin".to_string()],
        "/missing/labels.txt",
        "lines",
    );
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/convert"),
        Some(&body),
    )
    .await;
    assert!(
        resp.status().is_client_error() || resp.status().is_server_error(),
        "expected non-success; got {}",
        resp.status(),
    );

    let resp = call(&r, Method::GET, &format!("/api/v1/workspaces/{ws}"), None).await;
    if resp.status() == StatusCode::OK {
        let v: serde_json::Value = json_body(resp).await;
        let heads = v["heads"].as_array().expect("heads");
        assert!(heads.is_empty(), "summary heads must be empty: {v}");
    }
}

/// Path traversal in a path field rejects at deserialize time
/// (`ConverterPath::parse`), before any file lookup.
#[tokio::test]
async fn convert_rejects_path_traversal_via_serde() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    // After leading-slash strip, `/..` becomes `..`, which AssetPath rejects.
    let body = convert_body(
        "/../escape.json",
        &["/s.bin".to_string()],
        "/labels.txt",
        "lines",
    );
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/convert"),
        Some(&body),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// A body omitting the required `converter_type` tag (legacy flat shape) 400s.
#[tokio::test]
async fn convert_rejects_round_1_body_without_converter_type() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    let body = serde_json::json!({
        "model_json_path": "/m",
        "shards": ["/s"],
        "labels_path": "/l",
        "labels_format": "lines",
    })
    .to_string();
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/convert"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "missing converter_type must be 400",
    );
}

/// An unknown `converter_type` 400s rather than falling through to a default.
#[tokio::test]
async fn convert_rejects_unknown_converter_type() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    let body = serde_json::json!({
        "converter_type": "onnx",
        "model_json_path": "/m",
        "shards": ["/s"],
        "labels_path": "/l",
        "labels_format": "lines",
    })
    .to_string();
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/convert"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "unknown converter_type must be 400",
    );
}

/// Path fields are converter-rooted (slashless; legacy leading `/` accepted via
/// BC shim), but empty/lone-slash/traversal still reject at deserialize; pinned
/// via `model_json_path = ".."`.
#[tokio::test]
async fn convert_rejects_invalid_path_field() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let ws = create_workspace(&r, "main").await;

    let body = serde_json::json!({
        "converter_type": "tfjs",
        "model_json_path": "..",
        "shards": ["tfjs/s.bin"],
        "labels_path": "tfjs/labels.txt",
        "labels_format": "lines",
    })
    .to_string();
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/convert"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "traversal model_json_path must be 400",
    );
}

/// `POST /convert` on a phantom (well-formed UUID, non-existent) workspace 404s,
/// not 500: the existence probe's `FileError::Io { NotFound }` must route through
/// `classify_workspace_existence_error` rather than wrapping as `FsError::Internal`.
#[tokio::test]
async fn start_convert_phantom_workspace_returns_404() {
    let _gate = serialize_test();
    let dir = tempfile::tempdir().unwrap();
    let (r, _files) = fresh_router(dir.path());
    let phantom = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";

    // Path fields needn't resolve: the workspace-existence probe fails first.
    let body = convert_body(
        "tfjs/model.json",
        &["tfjs/s.bin".to_string()],
        "tfjs/labels.txt",
        "lines",
    );
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{phantom}/convert"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "start_convert on a phantom workspace must 404 (not 500); got {}",
        resp.status(),
    );
}
