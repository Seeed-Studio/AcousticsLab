//! Job-inspection integration tests for the `/jobs` and
//! `/workspaces/{id}/assets/*_logs` surfaces plus the JobConflict
//! admission contracts.
//!
//! Jobs are admitted directly through the [`JobRegistry`] handle so the
//! test surface is independent of the per-domain producer pipelines
//! (training / converter / dataset-delete).

#![allow(clippy::disallowed_methods)]

use std::sync::Arc;

use acousticslab::api::router_v1_nested;
use acousticslab::common::asset_path::AssetPath;
use acousticslab::common::ids::HeadId;
use acousticslab::common::workspace::{JobReference, JobType};
use acousticslab::file_mgr::{FsService, JobRegistry, RegistryJobResult};
use axum::Router;
use axum::body::to_bytes;
use axum::http::{Method, StatusCode};

mod api_fixtures;
use api_fixtures::{call, create_workspace, fresh_app_state, json_body};

struct Harness {
    router: Router,
    jobs: Arc<JobRegistry>,
    files: Arc<dyn FsService>,
}

fn fresh_harness(dir: &std::path::Path) -> Harness {
    let app_state = fresh_app_state(dir);
    let jobs = app_state.jobs.clone();
    let files = app_state.files.clone();
    Harness {
        router: router_v1_nested(app_state),
        jobs,
        files,
    }
}

#[tokio::test(flavor = "current_thread")]
async fn get_jobs_empty_initially() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let resp = call(&h.router, Method::GET, "/api/v1/jobs", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let arr = v.as_array().expect("jobs is array");
    assert!(arr.is_empty(), "expected empty list, got {arr:?}");
}

#[tokio::test(flavor = "current_thread")]
async fn get_jobs_includes_running_job() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    let handle = h
        .jobs
        .try_acquire(
            JobType::Train,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .expect("admission cleared");
    let job_id = handle.job_id();

    let resp = call(&h.router, Method::GET, "/api/v1/jobs", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let arr: serde_json::Value = json_body(resp).await;
    let arr = arr.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["state"].as_str(), Some("running"));
    assert_eq!(arr[0]["job_type"].as_str(), Some("train"));

    let resp = call(
        &h.router,
        Method::GET,
        &format!("/api/v1/jobs/{job_id}"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["state"].as_str(), Some("running"));

    // Dropping the handle without a terminal call abandons the job -> failed.
    drop(handle);
    let resp = call(
        &h.router,
        Method::GET,
        &format!("/api/v1/jobs/{job_id}"),
        None,
    )
    .await;
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["state"].as_str(), Some("failed"));
}

#[tokio::test(flavor = "current_thread")]
async fn get_job_returns_404_for_unknown_id() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let phantom = acousticslab::common::ids::JobId::new();
    let resp = call(
        &h.router,
        Method::GET,
        &format!("/api/v1/jobs/{phantom}"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test(flavor = "current_thread")]
async fn job_events_returns_409_event_gap_on_stale_after_seq() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    let handle = h
        .jobs
        .try_acquire(
            JobType::Train,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .unwrap();
    let job_id = handle.job_id();
    // The 128-entry ring won't overflow from one log line, so force the
    // gap with an after_seq far past last_seq.
    handle.append_log("hello");
    let resp = call(
        &h.router,
        Method::GET,
        &format!("/api/v1/jobs/{job_id}/events?after_seq=99999999"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::CONFLICT);
    let body: serde_json::Value = json_body(resp).await;
    assert_eq!(body["code"].as_str(), Some("event_gap"));
    assert!(body["oldest_seq"].is_number());
    assert!(body["latest_seq"].is_number());
    drop(handle);
}

#[tokio::test(flavor = "current_thread")]
async fn job_conflict_on_overlapping_train_admission() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let _h1 = h
        .jobs
        .try_acquire(
            JobType::Train,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .unwrap();
    // A second Train is rejected regardless of workspace (the Train cap is
    // global, not per-workspace).
    let err = h
        .jobs
        .try_acquire(
            JobType::Train,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        acousticslab::file_mgr::RegistryConflict::AnotherTrainRunning
    ));
    // Dataset-delete coexists with an active train in the same workspace,
    // grabbing the single delete slot.
    let _del = h
        .jobs
        .try_acquire(
            JobType::DatasetDelete,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            Some(AssetPath::parse("audio").unwrap()),
        )
        .expect("dataset-delete coexists with train");
}

#[tokio::test(flavor = "current_thread")]
async fn training_log_page_reads_jsonl_file() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    let job_id = acousticslab::common::ids::JobId::new();
    let workspace_dir = h
        .files
        .workspace_tmpdir(&ws_id)
        .parent()
        .unwrap()
        .to_path_buf();
    let log_dir = workspace_dir.join("training_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let log_path = log_dir.join(format!("{job_id}.jsonl"));
    let lines = [
        r#"{"seq":1,"at":"2026-05-07T12:00:00Z","message":"first"}"#,
        r#"{"seq":2,"at":"2026-05-07T12:00:01Z","message":"second"}"#,
        r#"{"seq":3,"at":"2026-05-07T12:00:02Z","message":"third"}"#,
    ];
    std::fs::write(&log_path, lines.join("\n")).unwrap();
    // JSONL paging is served from `/assets/{*path}`, gated to `.jsonl` files.
    let resp = call(
        &h.router,
        Method::GET,
        &format!("/api/v1/workspaces/{ws_id_str}/assets/training_logs/{job_id}.jsonl?limit=2",),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let events = v["events"].as_array().unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(v["next_after_seq"].as_u64(), Some(2));
    let resp = call(
        &h.router,
        Method::GET,
        &format!(
            "/api/v1/workspaces/{ws_id_str}/assets/training_logs/{job_id}.jsonl?after_seq=2&limit=10",
        ),
        None,
    )
    .await;
    let v: serde_json::Value = json_body(resp).await;
    let events = v["events"].as_array().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(v["next_after_seq"].as_u64(), Some(3));
}

#[tokio::test(flavor = "current_thread")]
async fn delete_training_logs_refuses_while_train_active_and_succeeds_after() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    let workspace_dir = h
        .files
        .workspace_tmpdir(&ws_id)
        .parent()
        .unwrap()
        .to_path_buf();
    let log_dir = workspace_dir.join("training_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let job_id = acousticslab::common::ids::JobId::new();
    std::fs::write(
        log_dir.join(format!("{job_id}.jsonl")),
        r#"{"seq":1,"at":"2026-05-07T12:00:00Z","message":"hi"}"#,
    )
    .unwrap();
    let handle = h
        .jobs
        .try_acquire(
            JobType::Train,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .unwrap();
    // DELETE refuses with 409 while a producer holds the workspace.
    let resp = call(
        &h.router,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws_id_str}/assets/training_logs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::CONFLICT);
    // A terminal call releases the workspace reference.
    handle.succeed(None);
    // DELETE now returns 202 + job_id; the staged drain runs off-mutex and
    // unlinks the jsonl eventually.
    let resp = call(
        &h.router,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws_id_str}/assets/training_logs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    let returned_job = v["job_id"]
        .as_str()
        .expect("async log delete returns job_id");

    // log_dir is recreated empty before the 202 returns (under the workspace
    // mutex), so polling it is a no-op; the per-job tombstone, unlinked by
    // drain+finalize, is the canonical "drain still in flight" marker.
    let tombstone_path = workspace_dir
        .join(".tmp")
        .join(format!("delete-training-logs-{returned_job}.json"));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while tombstone_path.exists() {
        if std::time::Instant::now() > deadline {
            panic!(
                "training-logs delete tombstone {} still present 5 s after delete",
                tombstone_path.display(),
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    // The drain wipes children but recreates `training_logs/` empty to keep
    // the canonical workspace structural shape.
    assert!(
        log_dir.is_dir(),
        "empty training_logs/ recreated after whole-tree wipe",
    );
    let entries: Vec<_> = std::fs::read_dir(&log_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert!(
        entries.is_empty(),
        "drained training_logs/ has no children; got {entries:?}",
    );
}

#[tokio::test(flavor = "current_thread")]
async fn job_events_sse_replays_then_terminates() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    let handle = h
        .jobs
        .try_acquire(
            JobType::Convert,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .unwrap();
    let job_id = handle.job_id();
    handle.append_log("started");
    handle.append_log("midway");
    handle.succeed(None);

    let resp = call(
        &h.router,
        Method::GET,
        &format!("/api/v1/jobs/{job_id}/events"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), 1 << 16).await.unwrap();
    let body_str = String::from_utf8_lossy(&body);
    // SSE replay (lines `event: job` / `data: {...}`) emits both log lines
    // and the terminal event before the stream closes.
    assert!(
        body_str.contains("event: job"),
        "missing event marker: {body_str}"
    );
    assert!(
        body_str.contains("started"),
        "missing started log: {body_str}"
    );
    assert!(
        body_str.contains("succeeded"),
        "missing terminal: {body_str}"
    );
}

/// Pins `JobType::ConverterDelete` to its wire string `"converter_delete"`.
#[tokio::test(flavor = "current_thread")]
async fn get_jobs_surfaces_converter_delete_job_type() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    // ConverterDelete shares the `max_delete_jobs` slot with the other delete
    // subtypes.
    let handle = h
        .jobs
        .try_acquire(
            JobType::ConverterDelete,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            Some(AssetPath::parse("tfjs/model.json").unwrap()),
        )
        .expect("admission cleared");
    let _job_id = handle.job_id();

    let resp = call(&h.router, Method::GET, "/api/v1/jobs", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let arr: serde_json::Value = json_body(resp).await;
    let arr = arr.as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["job_type"].as_str(), Some("converter_delete"));
    assert_eq!(arr[0]["state"].as_str(), Some("running"));
}

/// A job admitted with a target path surfaces it as `target_path`, never the
/// legacy `dataset_path` alias.
#[tokio::test(flavor = "current_thread")]
async fn job_snapshot_carries_target_path_not_legacy_dataset_path() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    let target = AssetPath::parse("audio/cat").unwrap();
    let handle = h
        .jobs
        .try_acquire(
            JobType::DatasetDelete,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            Some(target.clone()),
        )
        .expect("admission cleared");
    let job_id = handle.job_id();

    let resp = call(
        &h.router,
        Method::GET,
        &format!("/api/v1/jobs/{job_id}"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(
        v["target_path"].as_str(),
        Some("audio/cat"),
        "`target_path` must surface the admission path; body={v}",
    );
    assert!(
        v.get("dataset_path").is_none(),
        "legacy `dataset_path` alias must not appear; body={v}",
    );
}

/// Converter-side mirror of the training-logs delete: 409 while a convert
/// job for the workspace runs, 202 once it terminates.
#[tokio::test(flavor = "current_thread")]
async fn delete_converter_logs_refuses_while_convert_active_and_succeeds_after() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    // The job id here is arbitrary: the DELETE 409 fires from the
    // active-convert gate before any I/O on the log dir.
    let workspace_dir = h
        .files
        .workspace_tmpdir(&ws_id)
        .parent()
        .unwrap()
        .to_path_buf();
    let log_dir = workspace_dir.join("converter_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let job_id = acousticslab::common::ids::JobId::new();
    std::fs::write(
        log_dir.join(format!("{job_id}.jsonl")),
        r#"{"seq":1,"at":"2026-05-08T12:00:00Z","message":"hi"}"#,
    )
    .unwrap();

    let handle = h
        .jobs
        .try_acquire(
            JobType::Convert,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .unwrap();
    // DELETE refuses with 409 while a producer holds the workspace.
    let resp = call(
        &h.router,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws_id_str}/assets/converter_logs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::CONFLICT);
    let v: serde_json::Value = json_body(resp).await;
    assert!(
        v["error"].as_str().unwrap_or("").contains("converter_logs"),
        "409 message must mention converter_logs; body={v}",
    );

    // A terminal call releases the workspace reference.
    handle.succeed(None);
    // DELETE now returns 202 + job_id.
    let resp = call(
        &h.router,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws_id_str}/assets/converter_logs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    let returned_job = v["job_id"]
        .as_str()
        .expect("async log delete returns job_id");

    // log_dir is recreated empty before the 202 returns, so poll the per-job
    // tombstone (unlinked by drain+finalize) for "drain in flight" instead.
    let tombstone_path = workspace_dir
        .join(".tmp")
        .join(format!("delete-converter-logs-{returned_job}.json"));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while tombstone_path.exists() {
        if std::time::Instant::now() > deadline {
            panic!(
                "converter-logs delete tombstone {} still present 5 s after delete",
                tombstone_path.display(),
            );
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    assert!(log_dir.is_dir(), "empty converter_logs/ recreated");
    let entries: Vec<_> = std::fs::read_dir(&log_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert!(
        entries.is_empty(),
        "drained converter_logs/ has no children; got {entries:?}",
    );
}

/// A succeeded convert surfaces its typed `JobResult::Convert` (head_id,
/// sha256, n_classes) on the snapshot, so tooling can chain `POST /active`
/// off `result.head_id` without parsing the JSONL log.
#[tokio::test(flavor = "current_thread")]
async fn convert_job_terminal_carries_typed_result_with_head_id() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    let handle = h
        .jobs
        .try_acquire(
            JobType::Convert,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .expect("admission cleared");
    let job_id = handle.job_id();
    let head_id = HeadId::new();
    let sha256 = "deadbeef".repeat(8);
    let n_classes: u32 = 7;

    handle.succeed(Some(RegistryJobResult::Convert {
        head_id,
        sha256: sha256.clone(),
        n_classes,
    }));

    let resp = call(
        &h.router,
        Method::GET,
        &format!("/api/v1/jobs/{job_id}"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["state"].as_str(), Some("succeeded"));
    let result = v["result"].as_object().expect("result is an object");
    assert_eq!(result["kind"].as_str(), Some("convert"));
    assert_eq!(
        result["head_id"].as_str(),
        Some(head_id.to_string().as_str())
    );
    assert_eq!(result["sha256"].as_str(), Some(sha256.as_str()));
    assert_eq!(result["n_classes"].as_u64(), Some(n_classes as u64));
}

/// Pins the train variant's wire shape (`kind:"train"` + head_id/sha256/
/// n_classes), guarding against a regression dropping or renaming the variant.
#[tokio::test(flavor = "current_thread")]
async fn train_typed_result_variant_round_trips_on_wire() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    let handle = h
        .jobs
        .try_acquire(
            JobType::Train,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .expect("admission cleared");
    let job_id = handle.job_id();
    let head_id = HeadId::new();
    let sha256 = "abcd".repeat(16);
    let n_classes: u32 = 12;

    handle.succeed(Some(RegistryJobResult::Train {
        head_id,
        sha256: sha256.clone(),
        n_classes,
    }));

    let resp = call(
        &h.router,
        Method::GET,
        &format!("/api/v1/jobs/{job_id}"),
        None,
    )
    .await;
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["state"].as_str(), Some("succeeded"));
    let result = v["result"].as_object().expect("result is an object");
    assert_eq!(result["kind"].as_str(), Some("train"));
    assert_eq!(
        result["head_id"].as_str(),
        Some(head_id.to_string().as_str())
    );
    assert_eq!(result["sha256"].as_str(), Some(sha256.as_str()));
    assert_eq!(result["n_classes"].as_u64(), Some(n_classes as u64));
}

/// Jobs coexist in one workspace because only `WorkspaceDelete` is exclusive:
/// the cap admits one of each non-delete type, and the delete family shares a
/// single slot.
#[tokio::test(flavor = "current_thread")]
async fn get_jobs_lists_coexisting_jobs() {
    let dir = tempfile::tempdir().unwrap();
    let h = fresh_harness(dir.path());
    let ws_id_str = create_workspace(&h.router, "main").await;
    let ws_id = acousticslab::common::ids::WorkspaceId::parse(&ws_id_str).unwrap();
    let _train = h
        .jobs
        .try_acquire(
            JobType::Train,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .expect("train admits");
    let _convert = h
        .jobs
        .try_acquire(
            JobType::Convert,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            None,
        )
        .expect("convert admits");
    let _delete = h
        .jobs
        .try_acquire(
            JobType::DatasetDelete,
            vec![JobReference::Workspace {
                workspace_id: ws_id,
            }],
            Some(AssetPath::parse("audio/cat").unwrap()),
        )
        .expect("dataset-delete admits");

    let resp = call(&h.router, Method::GET, "/api/v1/jobs", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let arr: serde_json::Value = json_body(resp).await;
    let arr = arr.as_array().unwrap();
    let job_types: std::collections::HashSet<&str> =
        arr.iter().filter_map(|j| j["job_type"].as_str()).collect();
    assert!(job_types.contains("train"), "missing train; got {arr:?}");
    assert!(
        job_types.contains("convert"),
        "missing convert; got {arr:?}"
    );
    assert!(
        job_types.contains("dataset_delete"),
        "missing dataset_delete; got {arr:?}",
    );
    assert_eq!(arr.len(), 3, "exactly three concurrent jobs");
}
