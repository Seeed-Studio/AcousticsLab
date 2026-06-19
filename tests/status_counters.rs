//! `workspace.*` counters on `GET /api/v1/status`. The metrics global is
//! process-wide, so tests measure before/after deltas under `METRICS_TEST_LOCK`.

#![allow(clippy::disallowed_methods)]

use std::sync::Arc;
use std::time::Duration;

use acousticslab::api::router_v1_nested;
use acousticslab::status::WorkspaceMetrics;
use axum::http::{Method, StatusCode};

mod api_fixtures;
use api_fixtures::{call, create_workspace, fresh_app_state, json_body, upload};

/// Serializes the before/after delta window: the `WorkspaceMetrics` global and
/// `metrics_hooks` slots are process-wide `OnceLock`s (first install wins), so
/// concurrent tests would interleave increments and break `after - before`.
/// `.await`-safe across router calls.
static METRICS_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Wires file_mgr metrics hooks to `metrics`; the `OnceLock`s are single-shot, so re-calls no-op.
fn install_metrics_hooks(metrics: Arc<WorkspaceMetrics>) {
    let m = Arc::clone(&metrics);
    acousticslab::file_mgr::metrics_hooks::install_workspace_core_write_hook(move |d| {
        m.record_workspace_core_write(d);
    });
    let m = Arc::clone(&metrics);
    acousticslab::file_mgr::metrics_hooks::install_head_index_write_hook(move |d| {
        m.record_head_index_write(d);
    });
    let m = Arc::clone(&metrics);
    acousticslab::file_mgr::metrics_hooks::install_upload_hook(move |bytes| {
        m.record_upload(bytes);
    });
    let m = Arc::clone(&metrics);
    acousticslab::file_mgr::metrics_hooks::install_dataset_mutation_rejected_hook(move || {
        m.record_dataset_mutation_rejected();
    });
    let m = Arc::clone(&metrics);
    acousticslab::file_mgr::metrics_hooks::install_job_events_dropped_hook(move |n| {
        m.record_job_events_dropped(n);
    });
}

/// Every `workspace.*` counter is present and numeric on the status wire;
/// asserts shape not value, since the process-shared global may be nonzero.
#[tokio::test]
async fn status_includes_workspace_counter_block() {
    let _g = METRICS_TEST_LOCK.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));

    let resp = call(&r, Method::GET, "/api/v1/status", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let workspace = v.get("workspace").expect("workspace counter block present");
    for key in [
        "assets_uploaded_total",
        "bytes_uploaded_total",
        "workspace_core_writes_total",
        "head_index_writes_total",
        "dataset_mutations_rejected_total",
        "converter_mutations_rejected_total",
        "workspace_core_write_p99_us",
        "head_index_write_p99_us",
        "job_events_dropped_total",
        "sse_clients_current",
        "boot_orphans_swept_total",
        "boot_workspace_recovery_failures_total",
        // Dirent enumeration failures (EIO/EACCES), split from recovery failures for triage.
        "boot_workspace_enumeration_failures_total",
    ] {
        let val = workspace
            .get(key)
            .unwrap_or_else(|| panic!("workspace.{key} present"));
        // sse_clients_current is i64, the rest u64; accept either numeric shape.
        assert!(
            val.as_u64().is_some() || val.as_i64().is_some(),
            "workspace.{key} must serialize as a number; got {val:?}",
        );
    }
    // Pin the config_reload pair's wire shape so a stray #[serde(skip)] can't silently drop it.
    let config_reload = v
        .get("config_reload")
        .expect("config_reload counter block present");
    for key in ["reloads_succeeded_total", "reloads_rejected_total"] {
        let val = config_reload
            .get(key)
            .unwrap_or_else(|| panic!("config_reload.{key} present"));
        assert!(
            val.as_u64().is_some(),
            "config_reload.{key} must serialize as u64; got {val:?}",
        );
    }
}

/// A dataset upload bumps the upload + core-write counters, reflected end-to-end
/// on the status route (which reads the same global).
#[tokio::test]
async fn upload_increments_counters_on_status_wire() {
    let _g = METRICS_TEST_LOCK.lock().await;
    let dir = tempfile::tempdir().unwrap();
    let metrics = Arc::new(WorkspaceMetrics::new());
    acousticslab::status::workspace_metrics::install_for_tests(Arc::clone(&metrics));
    // A sibling test may already own the OnceLock global; assert against the winner.
    let metrics_for_assert: Arc<WorkspaceMetrics> =
        match acousticslab::status::workspace_metrics::global() {
            Some(g) => Arc::clone(g),
            None => metrics,
        };
    install_metrics_hooks(Arc::clone(&metrics_for_assert));
    let before = metrics_for_assert.snapshot();

    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let bytes = b"audio body of known length";
    let resp = upload(&r, &ws, "datasets/audio_dataset/cat/sample.wav", bytes).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Cheap defence against scheduler skew; no async work is actually pending.
    tokio::task::yield_now().await;

    let after = metrics_for_assert.snapshot();
    assert_eq!(
        after.assets_uploaded_total - before.assets_uploaded_total,
        1,
        "assets_uploaded_total bumped by exactly one upload",
    );
    assert_eq!(
        after.bytes_uploaded_total - before.bytes_uploaded_total,
        bytes.len() as u64,
        "bytes_uploaded_total bumped by the upload size",
    );
    assert!(
        after.workspace_core_writes_total > before.workspace_core_writes_total,
        "workspace_core_writes_total advances on the dataset_revision bump (before={} after={})",
        before.workspace_core_writes_total,
        after.workspace_core_writes_total,
    );

    let resp = call(&r, Method::GET, "/api/v1/status", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let ws_block = v.get("workspace").expect("workspace block");
    assert!(
        ws_block["assets_uploaded_total"].as_u64().expect("u64") >= after.assets_uploaded_total,
        "wire counter >= snapshot counter (counters are cumulative): {ws_block}",
    );
}

/// `record_boot_orphans_swept` accumulates onto `boot_orphans_swept_total`.
#[tokio::test]
async fn boot_orphans_swept_records_on_metrics_global() {
    let _g = METRICS_TEST_LOCK.lock().await;
    let metrics = Arc::new(WorkspaceMetrics::new());
    acousticslab::status::workspace_metrics::install_for_tests(Arc::clone(&metrics));
    let pinned: Arc<WorkspaceMetrics> = match acousticslab::status::workspace_metrics::global() {
        Some(g) => Arc::clone(g),
        None => metrics,
    };
    let before = pinned.snapshot().boot_orphans_swept_total;

    pinned.record_boot_orphans_swept(5);
    pinned.record_boot_orphans_swept(2);

    let after = pinned.snapshot().boot_orphans_swept_total;
    assert_eq!(after - before, 7);

    pinned.record_workspace_core_write(Duration::from_millis(2));
    pinned.record_head_index_write(Duration::from_micros(500));
    let snap = pinned.snapshot();
    assert!(snap.workspace_core_write_p99_us > 0);
    assert!(snap.head_index_write_p99_us > 0);

    // sse_client_guard increments on acquire, decrements on Drop.
    let before_clients = pinned.snapshot().sse_clients_current;
    {
        let _g = pinned.sse_client_guard();
        let _g2 = pinned.sse_client_guard();
        assert_eq!(pinned.snapshot().sse_clients_current - before_clients, 2,);
    }
    assert_eq!(pinned.snapshot().sse_clients_current, before_clients);
}

/// Dataset and converter rejections land on separate per-`AssetTree` counters.
#[tokio::test]
async fn mutation_rejections_dispatch_per_tree() {
    let _g = METRICS_TEST_LOCK.lock().await;
    let metrics = Arc::new(WorkspaceMetrics::new());
    acousticslab::status::workspace_metrics::install_for_tests(Arc::clone(&metrics));
    let pinned: Arc<WorkspaceMetrics> = match acousticslab::status::workspace_metrics::global() {
        Some(g) => Arc::clone(g),
        None => metrics,
    };
    install_metrics_hooks(Arc::clone(&pinned));

    let before = pinned.snapshot();

    // Dispatch helper is private; exercise the per-tree counters via public record_* directly.
    pinned.record_dataset_mutation_rejected();
    pinned.record_dataset_mutation_rejected();
    pinned.record_converter_mutation_rejected();

    let after = pinned.snapshot();
    assert_eq!(
        after.dataset_mutations_rejected_total - before.dataset_mutations_rejected_total,
        2,
        "dataset rejections increment dataset counter only",
    );
    assert_eq!(
        after.converter_mutations_rejected_total - before.converter_mutations_rejected_total,
        1,
        "converter rejections increment converter counter only",
    );
}
