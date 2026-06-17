//! Integration tests for `POST /workspaces/{id}/train` admission plus the
//! trainer's terminal-state and JSONL-event surface (`job_submitted` + a
//! terminal event); the end-to-end head-publish path needs a Burn-trainable
//! wav fixture (tens of seconds/run) and lives in
//! `tests/trained_head_rotation.rs`.

#![allow(clippy::disallowed_methods)]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acoustics_lab::api::{AppState, router_v1_nested};
use acoustics_lab::common::traits::head_store::HeadStore;
use acoustics_lab::common::traits::lag_source::{BroadcastLagSnapshot, LagSource};
use acoustics_lab::config::{
    Config, ConfigCell, DefaultHeadRef, LaunchConfig, MicSettingsCell, MicSettingsHandle,
};
use acoustics_lab::file_mgr::{FsService, FsServiceImpl};
use acoustics_lab::inference::{HeadInner, HotHead};
use acoustics_lab::status::{StatusMonitor, StatusReporter};
use acoustics_lab::training::{JobRegistry, TrainingRegistry};
use arc_swap::ArcSwap;
use axum::Router;
use axum::http::{Method, StatusCode};

mod api_fixtures;
use api_fixtures::{call, create_workspace, json_body, upload};

#[derive(Debug, Clone, Copy, Default)]
struct StubLagSource(BroadcastLagSnapshot);
impl LagSource for StubLagSource {
    fn snapshot(&self) -> BroadcastLagSnapshot {
        self.0
    }
}

/// Router rooted at `dir` with an opaque-bytes backbone stub wired through
/// `AppState::training_backbone_path` (production's channel); the trainer
/// fails at load time (`JobState::Failed`, no head committed), which is
/// sufficient since admission is the unit under test.
fn fresh_router(dir: &Path) -> Router {
    let backbone_path = dir.join("backbone").join("backbone.mpk");
    std::fs::create_dir_all(backbone_path.parent().unwrap()).expect("backbone dir");
    std::fs::write(&backbone_path, b"stub-backbone-bytes").expect("write backbone stub");
    fresh_router_with_backbone(dir, backbone_path)
}

/// Like [`fresh_router`] but the backbone is a caller-supplied path (e.g. a
/// symlink the admission gate stats); the opaque stub still fails `load_mpk`.
fn fresh_router_with_backbone(dir: &Path, backbone_path: std::path::PathBuf) -> Router {
    std::fs::create_dir_all(dir.join("workspaces")).expect("workspace root");

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
        weight: vec![0.0; acoustics_lab::common::dims::BackboneFeatureDim::USIZE * 2],
        bias: vec![0.0; 2],
        labels: vec!["a".into(), "b".into()],
        head_id: acoustics_lab::common::ids::HeadId::new(),
        n_classes: 2,
    }));
    // FsServiceImpl roots at `dir` (workspaces at `dir/workspaces/<id>/`), but
    // the backbone comes from `training_backbone_path`, so the FsService root
    // does not constrain where the stub lives.
    let jobs = Arc::new(acoustics_lab::file_mgr::JobRegistry::new(
        acoustics_lab::file_mgr::JobRegistryCfg::default(),
    ));
    let active_mutex: Arc<parking_lot::Mutex<()>> = Arc::new(parking_lot::Mutex::new(()));
    let files: Arc<dyn FsService> = Arc::new(FsServiceImpl::with_admission_jobs_and_active_mutex(
        dir.to_path_buf(),
        Default::default(),
        jobs.clone(),
        active_mutex.clone(),
    ));
    let monitor: Arc<dyn StatusReporter> = Arc::new(StatusMonitor::new());
    let training: Arc<dyn TrainingRegistry> = Arc::new(JobRegistry::new());
    router_v1_nested(AppState {
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
        training_backbone_path: Some(backbone_path),
        jobs,
    })
}

/// Seed a two-class dataset; the trainer walks `<workspace>/datasets/`, so
/// class folders are immediate children of `datasets/`.
async fn seed_dataset(router: &Router, ws: &str) {
    for cls in ["cat", "dog"] {
        for i in 0..2 {
            let resp = upload(
                router,
                ws,
                &format!("datasets/{cls}/sample{i}.bin"),
                b"stub-audio-bytes",
            )
            .await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "upload {cls}/sample{i}.bin failed: {}",
                resp.status(),
            );
        }
    }
}

/// Flattened [`TrainingCfg`] body: no wrapper, no `dataset_path`.
fn train_body() -> String {
    serde_json::json!({
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.001,
    })
    .to_string()
}

/// Happy-path admission returns `{head_id, job_id}` distinct well-formed UUIDs.
#[tokio::test]
async fn train_producer_returns_head_id_and_job_id() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "train accept; got {}",
        resp.status()
    );
    let v: serde_json::Value = json_body(resp).await;
    let head_id = v["head_id"].as_str().expect("head_id");
    let job_id = v["job_id"].as_str().expect("job_id");
    assert_eq!(head_id.len(), 36, "head_id is not a UUID");
    assert_eq!(job_id.len(), 36, "job_id is not a UUID");
    assert_ne!(head_id, job_id, "head_id and job_id must differ");
}

/// `max_train_jobs = 1`: a second train request while the first is unfinished
/// returns 409 `another_train_running`.
#[tokio::test]
async fn train_second_request_rejects_another_train_running() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp_first = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(
        resp_first.status(),
        StatusCode::OK,
        "first train accept; got {}",
        resp_first.status(),
    );

    // The permit is held for the whole job lifetime (even before the first
    // job's `spawn_blocking` wakes), so a prompt second request hits the gate.
    let resp_second = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(resp_second.status(), StatusCode::CONFLICT);
    let v: serde_json::Value = json_body(resp_second).await;
    assert_eq!(
        v["code"], "another_train_running",
        "expected `another_train_running`; body={v}",
    );
}

/// An in-flight train job does NOT block a dataset `DELETE /assets/<path>` in
/// the same workspace: both take only a workspace-scoped reference and
/// `WorkspaceDelete` is the only exclusive admission shape.
#[tokio::test]
async fn train_does_not_block_dataset_delete_in_same_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let _ = v["head_id"].as_str().expect("head_id");

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/cat"),
        None,
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::ACCEPTED,
        "DELETE during train must coexist (only WorkspaceDelete excludes); \
         async dispatch returns 202 Accepted",
    );
    let v: serde_json::Value = json_body(resp).await;
    assert!(
        v["job_id"].as_str().is_some(),
        "DELETE response must carry job_id; body={v}"
    );
}

/// A wrapper-shape body (`dataset_path`/`training_cfg`) parse-fails before any
/// spawn; the flattened [`TrainingCfg`] is the only accepted shape.
#[tokio::test]
async fn train_rejects_round_1_wrapper_body() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let body = serde_json::json!({
        "dataset_path": "audio",
        "training_cfg": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
        },
    })
    .to_string();
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/train"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "wrapper body must reject as 400",
    );

    let resp = call(&r, Method::GET, &format!("/api/v1/workspaces/{ws}"), None).await;
    if resp.status() == StatusCode::OK {
        let v: serde_json::Value = json_body(resp).await;
        let heads = v["heads"].as_array().expect("heads array in summary");
        assert!(heads.is_empty(), "summary heads must be empty: {v}");
    }
}

/// An out-of-range `learning_rate` rejects at the route boundary (400) before
/// any spawn.
#[tokio::test]
async fn train_rejects_invalid_learning_rate() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let body = serde_json::json!({
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.0,
    })
    .to_string();
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/train"),
        Some(&body),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// Permit auto-releases on terminal state: once the in-flight job terminates,
/// a follow-up train request is admitted again.
#[tokio::test]
async fn train_admission_recycles_after_terminal_state() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let job_id = v["job_id"].as_str().unwrap().to_string();

    // Poll to termination; bound at 30 s for slow hosts (stub fails in ~ms).
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspaces/{ws}/training/{job_id}"),
            None,
        )
        .await;
        if resp.status() == StatusCode::OK {
            let v: serde_json::Value = json_body(resp).await;
            let state = v["state"].as_str().unwrap_or("");
            if state == "failed" || state == "completed" || state == "cancelled" {
                break;
            }
        }
        if Instant::now() >= deadline {
            panic!("training job did not terminate within 30 s");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK, "permit not recycled");
}

/// The trainer writes a submit event then a terminal event into
/// `<workspace>/training_logs/<job_id>.jsonl`, readable via
/// `GET /assets/training_logs/<job_id>.jsonl`.
#[tokio::test]
async fn train_emits_started_then_terminal_jsonl_event() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());

    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let job_id = v["job_id"].as_str().unwrap().to_string();

    // Wait for terminal state so the terminal event is on disk.
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspaces/{ws}/training/{job_id}"),
            None,
        )
        .await;
        if resp.status() == StatusCode::OK {
            let v: serde_json::Value = json_body(resp).await;
            let state = v["state"].as_str().unwrap_or("");
            if state == "failed" || state == "completed" || state == "cancelled" {
                break;
            }
        }
        if Instant::now() >= deadline {
            panic!("training job did not terminate within 30 s");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Both a `job_submitted` and a terminal event must land (snake_case `kind`
    // per JSONL line); writing only one or neither is the regression.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/training_logs/{job_id}.jsonl?limit=20",),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let events = v["events"].as_array().expect("events array");
    let kinds: Vec<&str> = events.iter().filter_map(|e| e["kind"].as_str()).collect();
    let log_path = dir
        .path()
        .join("workspaces")
        .join(
            acoustics_lab::common::ids::WorkspaceId::parse(&ws)
                .unwrap()
                .to_string(),
        )
        .join("training_logs")
        .join(format!("{job_id}.jsonl"));
    let body = std::fs::read_to_string(&log_path).unwrap_or_default();
    assert!(
        kinds.contains(&"job_submitted"),
        "expected `job_submitted` event; got {kinds:?}; on-disk body=\n{body}",
    );
    assert!(
        kinds
            .iter()
            .any(|k| matches!(*k, "job_completed" | "job_failed" | "job_cancelled")),
        "expected a terminal event; got {kinds:?}; on-disk body=\n{body}",
    );
}

/// `GET /workspaces/{id}/training` on a phantom (well-formed but non-existent)
/// workspace id returns 404, NOT `200 { jobs: [] }`; guards `list_training`'s
/// `summary_or_404` existence probe.
#[tokio::test]
async fn list_training_phantom_workspace_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());
    // Well-formed v4 UUID that no `create_workspace` ever staged.
    let phantom = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{phantom}/training"),
        None,
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "list_training on a phantom workspace must 404; got {}",
        resp.status(),
    );
}

/// `POST /workspaces/{id}/train` on a phantom workspace returns 404, not 500:
/// the route must pipe the probe `FsError` through
/// `classify_workspace_existence_error`; raw `ApiError::from` would wrap
/// `NotFound` as `FsError::Internal` -> 500.
#[tokio::test]
async fn start_training_phantom_workspace_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let r = fresh_router(dir.path());
    let phantom = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{phantom}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "start_training on a phantom workspace must 404 (not 500); got {}",
        resp.status(),
    );
}

/// The backbone existence probe must follow symlinks (`fs::metadata`, not
/// `symlink_metadata`), so a symlink-to-`.mpk` admits; regression for
/// `symlink_metadata().is_file()` being `false` on a symlink and 400'ing a
/// working release-management symlink.
#[tokio::test]
async fn start_training_accepts_symlinked_backbone() {
    let dir = tempfile::tempdir().unwrap();
    // Wire the probe at the SYMLINK, not the real bytes.
    let real = dir.path().join("backbone").join("real-backbone.mpk");
    std::fs::create_dir_all(real.parent().unwrap()).unwrap();
    std::fs::write(&real, b"stub-backbone-bytes").unwrap();
    let link = dir.path().join("backbone").join("backbone-link.mpk");
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let r = fresh_router_with_backbone(dir.path(), link);
    let ws = create_workspace(&r, "main").await;
    seed_dataset(&r, &ws).await;

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/train"),
        Some(&train_body()),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "training must admit a symlinked backbone (probe follows symlinks); got {}",
        resp.status(),
    );
}
