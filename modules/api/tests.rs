//! API integration + unit tests.

#![cfg(test)]
// Test fixtures use `std::fs::write` directly; the file_mgr-only clippy.toml constraint is setup-exempt.
#![allow(clippy::disallowed_methods)]
use super::*;
use crate::common::traits::lag_source::{BroadcastLagSnapshot, LagSource};
use crate::config::Config;
use axum::body::{Body, to_bytes};
use axum::http::{Method, Request, StatusCode};
use axum::response::Response;
use tower::ServiceExt;

/// `LagSource` stub standing in for the production WS fan-out counters.
#[derive(Debug, Clone, Copy, Default)]
struct StubLagSource(BroadcastLagSnapshot);
impl LagSource for StubLagSource {
    fn snapshot(&self) -> BroadcastLagSnapshot {
        self.0
    }
}

fn fresh_state(dir: &std::path::Path) -> AppState {
    use crate::config::{ConfigCell, DefaultHeadRef, LaunchConfig, MicSettingsCell};
    let cfg_path = dir.join("config.toml");
    let workspace_root = dir.join("workspaces");
    std::fs::create_dir_all(&workspace_root).expect("workspace root");
    let cfg = Config::default_for();
    let config = Arc::new(ConfigCell::from_value(cfg.clone(), cfg_path).expect("validate"));
    config.persist().expect("persist initial");
    // Launch catalogue ships one `default-mock` candidate so `Fixed { id }` requests pass the cell cross-check.
    let launch = LaunchConfig::default_for();
    let mic_settings: Arc<dyn crate::config::MicSettingsHandle> = Arc::new(MicSettingsCell::new(
        Arc::new(launch.mic),
        cfg.mic.clone(),
        config.clone(),
    ));
    let inference_cfg = Arc::new(ArcSwap::from_pointee(cfg.inference));
    // Synthetic 2-class head so the active-head extractor wiring resolves.
    let head: Arc<dyn crate::common::traits::head_store::HeadStore> = Arc::new(
        crate::inference::HotHead::from_inner(crate::inference::HeadInner {
            weight: vec![0.0; crate::common::dims::BackboneFeatureDim::USIZE * 2],
            bias: vec![0.0; 2],
            labels: vec!["a".into(), "b".into()],
            head_id: crate::common::ids::HeadId::new(),
            n_classes: 2,
        }),
    );
    let jobs = Arc::new(crate::file_mgr::JobRegistry::new(
        crate::file_mgr::JobRegistryCfg::default(),
    ));
    let active_mutex: Arc<parking_lot::Mutex<()>> = Arc::new(parking_lot::Mutex::new(()));
    let files: Arc<dyn FsService> = Arc::new(
        crate::file_mgr::FsServiceImpl::with_admission_jobs_and_active_mutex(
            workspace_root,
            Default::default(),
            jobs.clone(),
            active_mutex.clone(),
        ),
    );
    let monitor: Arc<dyn StatusReporter> = Arc::new(crate::status::StatusMonitor::new());
    let training: Arc<dyn TrainingRegistry> = Arc::new(crate::training::JobRegistry::new());
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
        // Default points at an absent tempdir path so an accidental bundled-default activation fails closed.
        default_head: Some(DefaultHeadRef {
            path: dir.join("bundled_default/head.mpk"),
            labels_path: dir.join("bundled_default/labels.txt"),
        }),
        training_backbone_path: None,
        jobs,
    }
}

async fn json_body<T: serde::de::DeserializeOwned>(resp: Response) -> T {
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.expect("body");
    serde_json::from_slice(&bytes).expect("parse json")
}

async fn call(router: &Router, method: Method, path: &str, body: Option<&str>) -> Response {
    let mut req = Request::builder().method(method).uri(path);
    if body.is_some() {
        req = req.header("content-type", "application/json");
    }
    let req = req
        .body(
            body.map(|s| Body::from(s.to_string()))
                .unwrap_or(Body::empty()),
        )
        .expect("build req");
    router.clone().oneshot(req).await.expect("oneshot")
}

/// `PUT` an `application/octet-stream` body to `path`; the asset-upload counterpart to [`call`].
async fn put_octet(router: &Router, path: &str, body: impl Into<Body>) -> Response {
    let req = Request::builder()
        .method(Method::PUT)
        .uri(path)
        .header(axum::http::header::CONTENT_TYPE, "application/octet-stream")
        .body(body.into())
        .expect("build req");
    router.clone().oneshot(req).await.expect("oneshot")
}

#[tokio::test]
async fn health_endpoint_ok() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::GET, "/health", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["status"], "ok");
}

/// 404 must return the `{error, code}` envelope, not axum's plain-text default.
#[tokio::test]
async fn fallback_404_uses_envelope() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::GET, "/this_path_does_not_exist", None).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "not_found");
    assert!(
        v["error"].as_str().unwrap().contains("no route matched"),
        "envelope error must surface the unmatched method+path; got {v}",
    );
}

/// 405 path-method mismatch must use the same `{error, code}` envelope.
#[tokio::test]
async fn fallback_405_uses_envelope() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    // `/inference` is GET/POST only; PUT should 405.
    let resp = call(&r, Method::PUT, "/inference", None).await;
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "method_not_allowed");
    assert!(
        v["error"].as_str().unwrap().contains("method not allowed"),
        "envelope error must surface the verb mismatch; got {v}",
    );
}

#[tokio::test]
async fn get_mic_returns_catalogue_and_policy_separately() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::GET, "/mic", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["policy"]["mic"]["kind"], "first_available");
    assert_eq!(v["policy"]["channel"]["kind"], "auto");
    assert!(
        v["catalogue"]["candidates"].is_array(),
        "catalogue.candidates should be present in /mic response",
    );
    assert_eq!(
        v["catalogue"]["candidates"][0]["id"], "default-mock",
        "first-boot launch catalogue includes the synthetic mock candidate",
    );
}

#[tokio::test]
async fn post_mic_policy_persists_and_swaps() {
    use crate::audio_io::mic_arbitrator::{ChannelSelection, MicSelection};

    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let mic_settings_view = state.mic_settings.clone();
    let cfg_view = state.config.clone();
    let r = router(state);

    // Pin the `default-mock` candidate so the request passes the catalogue cross-check.
    let body = r#"{
            "policy": {
                "mic": { "kind": "fixed", "id": "default-mock" },
                "channel": { "kind": "fixed", "channel": 0 }
            }
        }"#;
    let resp = call(&r, Method::POST, "/mic/policy", Some(body)).await;
    if resp.status() != StatusCode::OK {
        let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        panic!("status not ok; body={}", String::from_utf8_lossy(&bytes));
    }
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["policy"]["mic"]["kind"], "fixed");
    assert_eq!(v["policy"]["mic"]["id"], "default-mock");
    assert_eq!(v["policy"]["channel"]["kind"], "fixed");
    assert_eq!(v["policy"]["channel"]["channel"], 0);
    assert_eq!(v["catalogue"]["candidates"][0]["id"], "default-mock");

    let in_mem = (*mic_settings_view.snapshot()).clone();
    match &in_mem.policy.mic {
        MicSelection::Fixed { id } => assert_eq!(id.as_str(), "default-mock"),
        other => panic!("expected Fixed, got {other:?}"),
    }
    match &in_mem.policy.channel {
        ChannelSelection::Fixed { channel } => assert_eq!(*channel, 0),
        other => panic!("expected Fixed channel, got {other:?}"),
    }
    let on_disk = std::fs::read_to_string(cfg_view.path()).unwrap();
    assert!(
        on_disk.contains("default-mock"),
        "config not persisted: {on_disk}"
    );
}

/// Read-your-writes: the POST version fed into `GET /mic?min_version=N` confirms a prior write settled.
#[tokio::test]
async fn post_mic_policy_surfaces_version_and_get_honours_min_version() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);
    let body = r#"{
            "policy": {
                "mic": { "kind": "fixed", "id": "default-mock" },
                "channel": { "kind": "auto" }
            }
        }"#;
    let resp = call(&r, Method::POST, "/mic/policy", Some(body)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let post_version = v["version"].as_u64().expect("version field on post");
    assert!(
        post_version >= 1,
        "first successful mutation must yield version >= 1, got {post_version}",
    );

    // min_version == post_version succeeds (current >= requested).
    let get = call(
        &r,
        Method::GET,
        &format!("/mic?min_version={post_version}"),
        None,
    )
    .await;
    assert_eq!(get.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(get).await;
    assert_eq!(v["version"].as_u64(), Some(post_version));

    // min_version one past current returns 425 Too Early.
    let get = call(
        &r,
        Method::GET,
        &format!("/mic?min_version={}", post_version + 1),
        None,
    )
    .await;
    assert_eq!(get.status(), StatusCode::TOO_EARLY);
    let v: serde_json::Value = json_body(get).await;
    assert_eq!(v["code"], "too_early");
}

/// Without the catalogue cross-check, an unknown `Fixed { id }` leaves the arbitrator silently inert (no audio).
#[tokio::test]
async fn post_mic_policy_rejects_unknown_fixed_id() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let mic_settings_view = state.mic_settings.clone();
    let r = router(state);

    let body = r#"{
            "policy": {
                "mic": { "kind": "fixed", "id": "no-such-mic" },
                "channel": { "kind": "auto" }
            }
        }"#;
    let resp = call(&r, Method::POST, "/mic/policy", Some(body)).await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "unknown Fixed id must yield 400",
    );
    // Rejected POST must leave the default FirstAvailable policy unswapped.
    let in_mem = (*mic_settings_view.snapshot()).clone();
    assert!(
        matches!(
            in_mem.policy.mic,
            crate::audio_io::mic_arbitrator::MicSelection::FirstAvailable,
        ),
        "live policy was mutated despite rejection: {:?}",
        in_mem.policy.mic,
    );
}

#[tokio::test]
async fn post_inference_validates_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    let resp = call(&r, Method::POST, "/inference", Some(r#"{"top_k":0}"#)).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");

    let resp = call(&r, Method::POST, "/inference", Some(r#"{"hop_samples":0}"#)).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    // Below MIN_HOP_SAMPLES (11_025) rejects: the floor caps overlap at 75% so the engine doesn't re-run on near-identical audio.
    let resp = call(
        &r,
        Method::POST,
        "/inference",
        Some(r#"{"hop_samples":1024}"#),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "hop_samples=1024 is below MIN_HOP_SAMPLES=11_025; must reject",
    );

    // Above MAX_HOP_SAMPLES (44_100) rejects: the ceiling holds cadence at one inference per wall-clock second.
    let resp = call(
        &r,
        Method::POST,
        "/inference",
        Some(r#"{"hop_samples":44101}"#),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "hop_samples=44101 exceeds MAX_HOP_SAMPLES=44_100; must reject",
    );

    // Both bounds are inclusive: MIN 11_025 (4 Hz busy end), MAX 44_100 (default, 1 Hz Speech-Commands convention).
    let resp = call(
        &r,
        Method::POST,
        "/inference",
        Some(r#"{"hop_samples":11025}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK, "MIN_HOP_SAMPLES must accept");
    let resp = call(
        &r,
        Method::POST,
        "/inference",
        Some(r#"{"hop_samples":44100}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK, "MAX_HOP_SAMPLES must accept");
}

#[tokio::test]
async fn post_inference_swaps_cadence() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let cfg_view = state.inference_cfg.clone();
    let r = router(state);

    let body = r#"{"hop_samples":22050,"top_k":5}"#;
    let resp = call(&r, Method::POST, "/inference", Some(body)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["cfg"]["hop_samples"], 22050);
    assert_eq!(v["cfg"]["top_k"], 5);

    let live = **cfg_view.load();
    assert_eq!(live.hop_samples, 22050);
    assert_eq!(live.top_k, 5);
}

#[tokio::test]
async fn training_start_round_1_wrapper_body_returns_400() {
    // Train body is the flattened `TrainingCfg`; a wrapper body parse-fails at deserialize before any spawn.
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    let resp = call(&r, Method::POST, "/workspaces", Some(r#"{"name":"train"}"#)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let ws = v["id"].as_str().unwrap();

    let body = serde_json::json!({
        "dataset_path": "missing",
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
        &format!("/workspaces/{ws}/train"),
        Some(&body),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "wrapper body must parse-fail",
    );
}

#[tokio::test]
async fn training_status_rejects_bad_job_id() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    let resp = call(&r, Method::POST, "/workspaces", Some(r#"{"name":"train"}"#)).await;
    let v: serde_json::Value = json_body(resp).await;
    let ws = v["id"].as_str().unwrap();

    let resp = call(
        &r,
        Method::GET,
        &format!("/workspaces/{ws}/training/not-a-uuid"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
}

#[tokio::test]
async fn status_endpoint_returns_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let monitor = crate::status::StatusMonitor::new();
    // 50 ms cadence + 150 ms wait = >=2 ticks to publish a real RSS sample.
    monitor.start_sampler(None, std::time::Duration::from_millis(50));
    let tx = monitor.register("test_alive").expect("register");
    tx.send(crate::status::Heartbeat::ok("running"))
        .expect("send");
    let mut state = fresh_state(dir.path());
    state.monitor = Arc::new(monitor);
    let r = router(state);
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    let resp = call(&r, Method::GET, "/status", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    // 0 would mean the sampler task path is broken.
    assert!(v["mem_rss_kb"].as_u64().unwrap_or(0) > 0);
    assert_eq!(v["subsystems"]["test_alive"]["healthy"], true);
    assert_eq!(v["subsystems"]["test_alive"]["detail"], "running");
    // Pin field presence + numeric type against a silent schema drop.
    assert_eq!(v["broadcast_audio_messages_dropped"].as_u64(), Some(0));
    assert_eq!(v["broadcast_inference_messages_dropped"].as_u64(), Some(0));
}

/// `/status` reflects whatever the `broadcast_lag_reader` returns.
#[tokio::test]
async fn status_endpoint_surfaces_broadcast_lags() {
    let dir = tempfile::tempdir().unwrap();
    let mut state = fresh_state(dir.path());
    state.broadcast_lag_reader = Arc::new(StubLagSource(BroadcastLagSnapshot {
        audio_messages_dropped: 17,
        inference_messages_dropped: 42,
    }));
    let r = router(state);
    let resp = call(&r, Method::GET, "/status", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["broadcast_audio_messages_dropped"].as_u64(), Some(17));
    assert_eq!(v["broadcast_inference_messages_dropped"].as_u64(), Some(42));
}

/// `metrics_age_ms`/`metrics_stale` distinguish a wedged sampler from real zero metrics: pre-sampler reads `{0, true}`, post-tick `{small, false}`.
#[tokio::test]
async fn status_endpoint_surfaces_metrics_freshness() {
    // Case 1: pre-sampler -- captured_at = None.
    {
        let dir = tempfile::tempdir().unwrap();
        let r = router(fresh_state(dir.path()));
        let resp = call(&r, Method::GET, "/status", None).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = json_body(resp).await;
        assert_eq!(
            v["metrics_age_ms"].as_u64(),
            Some(0),
            "pre-sampler: metrics_age_ms must be 0; body={v}",
        );
        assert_eq!(
            v["metrics_stale"].as_bool(),
            Some(true),
            "pre-sampler: metrics_stale must be true (no sample yet); body={v}",
        );
    }

    // Case 2: sampler running -- captured_at = Some(recent).
    {
        let dir = tempfile::tempdir().unwrap();
        let monitor = crate::status::StatusMonitor::new();
        monitor.start_sampler(None, std::time::Duration::from_millis(50));
        let mut state = fresh_state(dir.path());
        state.monitor = Arc::new(monitor);
        let r = router(state);
        // ~3 ticks at 50 ms cadence; non-flaky.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        let resp = call(&r, Method::GET, "/status", None).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value = json_body(resp).await;
        assert_eq!(
            v["metrics_stale"].as_bool(),
            Some(false),
            "live sampler: metrics_stale must be false; body={v}",
        );
        // ~1 s slack for scheduler jitter on busy CI.
        let age = v["metrics_age_ms"]
            .as_u64()
            .expect("metrics_age_ms must be u64");
        assert!(
            age < 1_000,
            "live sampler: metrics_age_ms must be < 1 s, got {age}",
        );
    }
}

/// Workspace-id path-traversal must be rejected before the filesystem layer; `WorkspaceId::parse` enforces strict UUID-v4.
#[tokio::test]
async fn workspace_id_rejects_path_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    for nasty in [
        "../etc",
        "/etc/passwd",
        "etc/passwd",
        "abc",                                     // not a UUID
        "00000000-0000-0000-0000-00000000000",     // 35 chars
        "../00000000-0000-4000-8000-000000000000", // UUID-v4 with traversal prefix
        "00000000_0000_4000_8000_000000000000",    // wrong separators
    ] {
        let resp = call(
            &r,
            Method::DELETE,
            &format!("/workspaces/{}", urlencoding::encode(nasty)),
            None,
        )
        .await;
        // 400/404/405 acceptable; 200 OK and any 5xx are not.
        assert_ne!(
            resp.status(),
            StatusCode::OK,
            "path traversal accepted for {nasty:?}"
        );
        assert!(
            resp.status() == StatusCode::BAD_REQUEST
                || resp.status() == StatusCode::NOT_FOUND
                || resp.status() == StatusCode::METHOD_NOT_ALLOWED,
            "unexpected status {} for {nasty:?}",
            resp.status()
        );
    }
}

#[tokio::test]
async fn workspace_create_list_delete_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    let resp = call(&r, Method::GET, "/workspaces", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert!(v["workspaces"].as_array().unwrap().is_empty());

    let resp = call(&r, Method::POST, "/workspaces", Some(r#"{"name":"first"}"#)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();
    assert_eq!(v["name"], "first");

    let resp = call(&r, Method::GET, "/workspaces", None).await;
    let v: serde_json::Value = json_body(resp).await;
    let ws = v["workspaces"].as_array().unwrap();
    assert_eq!(ws.len(), 1);
    assert_eq!(ws[0]["id"], id);

    let resp = call(&r, Method::POST, "/workspaces", Some(r#"{"name":"first"}"#)).await;
    assert_eq!(resp.status(), StatusCode::CONFLICT);

    let resp = call(&r, Method::DELETE, &format!("/workspaces/{id}"), None).await;
    // 202 Accepted: async drain on the blocking pool, matching the assets-delete surface.
    assert_eq!(resp.status(), StatusCode::ACCEPTED);

    let resp = call(&r, Method::DELETE, &format!("/workspaces/{id}"), None).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Unified asset surface: GET/PUT/DELETE on `/assets/{*path}` with the method picking the op, raw body carries the file.
#[tokio::test]
async fn workspace_upload_happy_path() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    let resp = call(&r, Method::POST, "/workspaces", Some(r#"{"name":"u"}"#)).await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();

    let payload = b"DEMO-MPK-PAYLOAD";
    let resp = put_octet(
        &r,
        &format!("/workspaces/{id}/assets/datasets/cls/demo.bin"),
        payload.to_vec(),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK, "upload status");
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["path"], "datasets/cls/demo.bin");
    assert_eq!(v["size_bytes"], payload.len());
    assert_eq!(v["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(v["workspace_revision_id"], 1);

    let resp = call(&r, Method::GET, &format!("/workspaces/{id}/assets"), None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let names: Vec<&str> = v["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| e["name"].as_str())
        .collect();
    assert!(names.contains(&"datasets"), "datasets/ in {names:?}");
    assert!(!names.contains(&".tmp"), ".tmp excluded: {names:?}");
    let resp = call(
        &r,
        Method::GET,
        &format!("/workspaces/{id}/assets/datasets"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    // First-level child of `datasets/` is the class folder (trainer keys class labels off this dir), not the leaf file.
    assert_eq!(entries[0]["name"], "cls");
}

/// `GET /assets/workspace.json` serves the daemon-owned `WorkspaceCore` verbatim: the alpkg-export contract that every archive carries provenance an importer can recover without `GET /workspaces/{id}`.
#[tokio::test]
async fn workspace_core_asset_get_serves_verbatim_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    // Create a workspace so `workspace.json` lands on disk via the lifecycle handler's atomic write.
    let resp = call(
        &r,
        Method::POST,
        "/workspaces",
        Some(r#"{"name":"prov","tags":["env-dev","unit"]}"#),
    )
    .await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();

    let resp = call(
        &r,
        Method::GET,
        &format!("/workspaces/{id}/assets/workspace.json"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
        .await
        .expect("read body");
    let parsed: serde_json::Value = serde_json::from_slice(&bytes).expect("workspace.json is JSON");
    assert_eq!(parsed["id"], id, "embedded id matches the workspace id");
    assert_eq!(parsed["name"], "prov");
    assert_eq!(parsed["tags"][0], "env-dev");
    assert_eq!(parsed["tags"][1], "unit");
    assert!(
        parsed["created_at"].is_string(),
        "created_at present: {parsed:?}"
    );
    assert!(
        parsed["workspace_revision"]["id"].is_number(),
        "workspace_revision.id present: {parsed:?}"
    );
    assert!(
        parsed["head_count"].is_number(),
        "head_count present: {parsed:?}"
    );
}

/// 5 MiB body pins the chunked `Body::into_data_stream()` reader so a regression to a buffered read fails here (peak RSS unobservable in a unit test).
#[tokio::test]
async fn workspace_upload_streams_large_payload() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::POST, "/workspaces", Some(r#"{"name":"big"}"#)).await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();

    // Deterministic 5 MiB payload (cycled bytes of `id`) for independent sha256 without a constant pattern.
    let mut payload = Vec::with_capacity(5 * 1024 * 1024);
    let pat = id.as_bytes();
    for i in 0..(5 * 1024 * 1024) {
        payload.push(pat[i % pat.len()]);
    }
    let expected_sha = crate::api::routes::converter::sha256_hex(&payload);
    let payload_len = payload.len();

    let resp = put_octet(
        &r,
        &format!("/workspaces/{id}/assets/datasets/cls/big.bin"),
        payload,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["size_bytes"].as_u64(), Some(payload_len as u64));
    assert_eq!(v["sha256"].as_str().unwrap(), expected_sha);
}

/// `PUT /assets` (no wildcard tail) hits the GET-only root listing; method dispatch must return 405, not 404.
#[tokio::test]
async fn workspace_upload_without_path_returns_405() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));
    let resp = call(&r, Method::POST, "/workspaces", Some(r#"{"name":"q"}"#)).await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();

    let resp = put_octet(&r, &format!("/workspaces/{id}/assets"), "DATA").await;
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn router_v1_nested_mounts_under_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_state(dir.path()));
    let resp = call(&r, Method::GET, "/api/v1/health", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    // Without the prefix the unnested route is 404.
    let resp = call(&r, Method::GET, "/health", None).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Upload to a valid-UUID non-workspace returns 404 and leaves no orphan dir: existence is checked before any `.tmp/` staging dir is created.
#[tokio::test]
async fn upload_to_nonexistent_workspace_is_404_no_orphan_dir() {
    let dir = tempfile::tempdir().unwrap();
    // `fresh_state` passes `dir/workspaces` as the FsService root; per-id dirs nest under `<root>/workspaces/<id>/`.
    let workspace_root = dir.path().join("workspaces").join("workspaces");
    let r = router(fresh_state(dir.path()));

    const PHANTOM_ID: &str = "00000000-0000-4000-8000-000000000777";
    let phantom_dir = workspace_root.join(PHANTOM_ID);
    assert!(
        !phantom_dir.exists(),
        "phantom workspace dir must not exist before the upload",
    );

    let resp = put_octet(
        &r,
        &format!("/workspaces/{PHANTOM_ID}/assets/datasets/cls/x.bin"),
        "DATA",
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "not_found");

    assert!(
        !phantom_dir.exists(),
        "upload to nonexistent workspace left an orphan dir at {}",
        phantom_dir.display(),
    );
}

/// Malformed JSON must surface as `{error, code: "bad_request"}`; wrapper extractors map deser rejections to `ApiError::Bad`.
#[tokio::test]
async fn bad_json_body_returns_envelope_400() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    let resp = call(&r, Method::POST, "/inference", Some("not-json {")).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
    assert!(
        v["error"].as_str().unwrap().contains("invalid JSON body"),
        "envelope must surface the diagnosis; got {v}",
    );
}

/// Bad query-parameter types also envelope-wrap; `ApiQuery` maps `Query<T>` rejections to `ApiError::Bad`.
#[tokio::test]
async fn bad_query_string_returns_envelope_400() {
    let dir = tempfile::tempdir().unwrap();
    let r = router(fresh_state(dir.path()));

    // `min_version` is `Option<u64>`; "abc" can't deserialize.
    let resp = call(&r, Method::GET, "/mic?min_version=abc", None).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
}
