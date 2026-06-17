//! In-process integration tests for `POST /active` / `GET /active`. Most tests
//! hand-roll a bundled-default `head.mpk` (ACSTHEAD header) + `labels.txt` via
//! `fresh_state` so `default_head` drives the full pipeline without the
//! checked-in fixture.

#![allow(clippy::disallowed_methods)]

use std::path::Path;

use acoustics_lab::api::{AppState, router};
use acoustics_lab::common::ids::{HeadId, WorkspaceId};
use acoustics_lab::common::workspace::{
    HeadIndex, HeadManifest, HeadRecord, WorkspaceCore, WorkspaceRevision,
};
use acoustics_lab::config::DefaultHeadRef;
use axum::http::{Method, StatusCode};

mod api_fixtures;
use api_fixtures::{call, fresh_app_state, json_body};

/// [`fresh_app_state`] with `default_head` staged as a real `head.mpk` so
/// `POST /active {default: true}` exercises the full preload pipeline.
fn fresh_state(dir: &Path) -> AppState {
    let mut state = fresh_app_state(dir);
    state.default_head = Some(stage_bundled_default(dir));
    state
}

/// Stage a real bundled-default `head.mpk` under `<dir>/bundled_default/` so
/// runtime preload via `HotHead::load` succeeds.
fn stage_bundled_default(dir: &Path) -> DefaultHeadRef {
    use acoustics_lab::common::head_header::write_with_payload;
    use acoustics_lab::model::Head;
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

    let bundled = dir.join("bundled_default");
    std::fs::create_dir_all(&bundled).unwrap();
    // Burn's recorder appends `.mpk` to the path stem.
    let raw_stem = bundled.join("raw");
    let raw_mpk = bundled.join("raw.mpk");
    let device: burn::tensor::Device<NdArray<f32>> = Default::default();
    let head = Head::<NdArray<f32>>::new(2, &device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    recorder.record(head.into_record(), raw_stem).unwrap();
    let payload = std::fs::read(&raw_mpk).unwrap();
    let mut composed = std::fs::File::create(bundled.join("head.mpk")).unwrap();
    write_with_payload(
        &mut composed,
        acoustics_lab::common::dims::BackboneFeatureDim::USIZE as u32,
        2,
        &payload,
    )
    .unwrap();
    drop(composed);
    std::fs::write(bundled.join("labels.txt"), "alpha\nbeta\n").unwrap();
    let _ = std::fs::remove_file(&raw_mpk);
    DefaultHeadRef {
        path: bundled.join("head.mpk"),
        labels_path: bundled.join("labels.txt"),
    }
}

/// Stage a real trained head by writing the workspace dir directly (not via
/// `FsService::create`) so the route's first `summary` lazy-loads fresh from
/// disk; create-then-publish would pin a stale cell, since
/// `publish_trained_head` updates a caller-supplied [`WorkspaceCacheCell`]
/// separate from FsService's internal map.
fn publish_one_trained_head(
    state: &AppState,
    workspace_id: acoustics_lab::common::ids::WorkspaceId,
) -> HeadId {
    use acoustics_lab::common::head_header::write_with_payload;
    use acoustics_lab::file_mgr::schema::{
        head_artifact_path, head_manifest_path, heads_dir, workspace_dir_for, write_head_index,
        write_head_manifest, write_workspace_core,
    };
    use acoustics_lab::model::Head;
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};

    let workspace_dir = workspace_dir_for(state.files.root(), &workspace_id);
    std::fs::create_dir_all(&workspace_dir).unwrap();
    std::fs::create_dir_all(workspace_dir.join(".tmp")).unwrap();
    std::fs::create_dir_all(workspace_dir.join("datasets")).unwrap();
    std::fs::create_dir_all(workspace_dir.join("training_logs")).unwrap();
    std::fs::create_dir_all(workspace_dir.join("converter_logs")).unwrap();
    std::fs::create_dir_all(heads_dir(&workspace_dir)).unwrap();

    let raw_stem = workspace_dir.join(".tmp").join("raw");
    let raw_mpk = workspace_dir.join(".tmp").join("raw.mpk");
    let device: burn::tensor::Device<NdArray<f32>> = Default::default();
    let head = Head::<NdArray<f32>>::new(3, &device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    recorder.record(head.into_record(), raw_stem).unwrap();
    let payload = std::fs::read(&raw_mpk).unwrap();
    std::fs::remove_file(&raw_mpk).ok();

    let head_id: HeadId = HeadId::new();
    let mpk_path = head_artifact_path(&workspace_dir, head_id);
    let mut composed = std::fs::File::create(&mpk_path).unwrap();
    write_with_payload(
        &mut composed,
        acoustics_lab::common::dims::BackboneFeatureDim::USIZE as u32,
        3,
        &payload,
    )
    .unwrap();
    drop(composed);
    let mpk_bytes = std::fs::read(&mpk_path).unwrap();
    let sha256 = sha256_hex(&mpk_bytes);

    let workspace_revision = WorkspaceRevision {
        id: 0,
        at: "2026-05-07T12:00:00Z".to_string(),
    };
    let manifest = HeadManifest {
        head_id,
        workspace_id,
        workspace_revision: workspace_revision.clone(),
        sha256: sha256.clone(),
        n_classes: 3,
        size_bytes: mpk_bytes.len() as u64,
        created_at: "2026-05-07T12:34:56Z".to_string(),
        labels: vec!["cat".to_string(), "dog".to_string(), "bird".to_string()],
    };
    write_head_manifest(&workspace_dir, &manifest).unwrap();
    assert!(head_manifest_path(&workspace_dir, head_id).is_file());

    let mut idx = HeadIndex::default();
    idx.heads.push(HeadRecord {
        head_id,
        workspace_revision: workspace_revision.clone(),
        sha256,
        n_classes: 3,
        size_bytes: mpk_bytes.len() as u64,
        created_at: "2026-05-07T12:34:56Z".to_string(),
    });
    write_head_index(&workspace_dir, &idx).unwrap();

    let core = WorkspaceCore {
        id: workspace_id,
        name: "main".to_string(),
        tags: Vec::new(),
        created_at: "2026-05-07T12:34:56Z".to_string(),
        workspace_revision,
        head_count: 1,
    };
    write_workspace_core(&workspace_dir, &core).unwrap();
    head_id
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    static HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = vec![0u8; d.len() * 2];
    for (i, &b) in d.iter().enumerate() {
        out[2 * i] = HEX[(b >> 4) as usize];
        out[2 * i + 1] = HEX[(b & 0x0f) as usize];
    }
    String::from_utf8(out).unwrap()
}

#[tokio::test]
async fn post_active_default_writes_bundled_generation() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "status was {}",
        resp.status()
    );
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "default");
    assert_eq!(
        v["runtime_head_id"],
        acoustics_lab::common::ids::DEFAULT_RUNTIME_HEAD_ID_STR,
    );
    assert_eq!(v["n_classes"], 2);
    assert!(v["sha256"].as_str().unwrap().len() == 64);
    assert!(v["activation_id"].is_string());
}

/// `POST /active {default: true}` force-sets: re-issuing mints a fresh
/// `activation_id` but keeps origin / runtime_head_id / n_classes consistent.
#[tokio::test]
async fn post_active_default_is_idempotent_force_set() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let first: serde_json::Value = {
        let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
        json_body(resp).await
    };
    let second: serde_json::Value = {
        let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
        json_body(resp).await
    };
    assert_ne!(first["activation_id"], second["activation_id"]);
    assert_eq!(first["runtime_head_id"], second["runtime_head_id"]);
    assert_eq!(first["origin"], second["origin"]);
    assert_eq!(first["n_classes"], second["n_classes"]);
}

#[tokio::test]
async fn post_active_head_then_get_active_returns_head_origin() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555540").unwrap();
    let head_id = publish_one_trained_head(&state, ws_id);
    let r = router(state);

    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": head_id.to_string(),
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "status was {}",
        resp.status()
    );
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "head");
    assert_eq!(v["runtime_head_id"], head_id.to_string());
    assert_eq!(v["source_head_id"], head_id.to_string());
    assert_eq!(v["source_workspace_id"], ws_id.to_string());
    assert_eq!(v["n_classes"], 3);

    let resp = call(&r, Method::GET, "/active", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "head");
    assert_eq!(v["runtime_head_id"], head_id.to_string());
    assert_eq!(v["source_workspace_alive"], true);
}

/// After the source workspace is deleted, `GET /active` still serves the
/// head (the generation owns independent bytes) and reports
/// `source_workspace_alive: false`.
#[tokio::test]
async fn get_active_reports_source_workspace_alive_false_after_workspace_delete() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555541").unwrap();
    let head_id = publish_one_trained_head(&state, ws_id);
    let workspace_dir =
        acoustics_lab::file_mgr::schema::workspace_dir_for(state.files.root(), &ws_id);
    let r = router(state);

    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": head_id.to_string(),
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Simulate workspace delete: the active generation owns
    // independent bytes, so the runtime is unaffected.
    std::fs::remove_dir_all(&workspace_dir).unwrap();

    let resp = call(&r, Method::GET, "/active", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "head");
    assert_eq!(v["runtime_head_id"], head_id.to_string());
    assert_eq!(v["source_workspace_alive"], false);
}

#[tokio::test]
async fn post_active_default_after_head_resets_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555540").unwrap();
    let head_id = publish_one_trained_head(&state, ws_id);
    let r = router(state);

    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": head_id.to_string(),
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "default");

    let resp = call(&r, Method::GET, "/active", None).await;
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["origin"], "default");
    assert_eq!(
        v["runtime_head_id"],
        acoustics_lab::common::ids::DEFAULT_RUNTIME_HEAD_ID_STR,
    );
}

/// Activating a head id absent from the workspace's `heads.json` returns 404.
#[tokio::test]
async fn post_active_head_404_when_head_missing_from_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555540").unwrap();
    let _head_id = publish_one_trained_head(&state, ws_id);
    let r = router(state);

    let unknown = "00000000-0000-4000-8000-000000000099";
    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": unknown,
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "not_found");
}

#[tokio::test]
async fn post_active_prunes_to_two_generations() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    for _ in 0..4 {
        let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
    let generations_dir = dir
        .path()
        .join("workspaces")
        .join("active")
        .join("generations");
    let count = std::fs::read_dir(&generations_dir).unwrap().count();
    assert!(
        count <= 2,
        "expected at most 2 retained generations, observed {count} under {}",
        generations_dir.display()
    );
}

/// `POST /active` rejects an unknown body shape with 4xx: the `untagged` request
/// enum matches no arm and `ApiJson` surfaces the deny-unknown-fields error.
#[tokio::test]
async fn post_active_rejects_unknown_body_shape() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let resp = call(
        &r,
        Method::POST,
        "/active",
        Some(r#"{"unexpected_field": "x"}"#),
    )
    .await;
    assert!(
        resp.status() == StatusCode::UNPROCESSABLE_ENTITY
            || resp.status() == StatusCode::BAD_REQUEST,
        "expected 4xx for unknown body shape, got {}",
        resp.status()
    );
}

/// `POST /active {default: false}` parses but the route explicitly rejects the
/// false value with 400.
#[tokio::test]
async fn post_active_rejects_default_false() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let resp = call(&r, Method::POST, "/active", Some(r#"{"default": false}"#)).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
}

// Shape pins lock the wire shape end-to-end (on-disk manifest + POST/GET
// responses) so a future re-add of the legacy `source_dataset_revision` alias
// fails at the test boundary.

/// The on-disk Head-origin manifest carries exactly the expected field set,
/// never the legacy `source_dataset_revision`.
#[tokio::test]
async fn active_head_manifest_on_disk_carries_field_set() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555548").unwrap();
    let head_id = publish_one_trained_head(&state, ws_id);
    let r = router(state);

    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": head_id.to_string(),
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let activation_id = v["activation_id"].as_str().expect("activation_id");

    // Read raw bytes as an opaque map: the typed `read_active_manifest` would
    // tautologically re-derive the struct that wrote them; the on-disk wire
    // shape is the actual contract.
    let manifest_path = dir
        .path()
        .join("workspaces")
        .join("active")
        .join("generations")
        .join(activation_id)
        .join("manifest.json");
    assert!(
        manifest_path.is_file(),
        "active generation manifest must land on disk at {}",
        manifest_path.display(),
    );
    let bytes = std::fs::read(&manifest_path).unwrap();
    let m: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let obj = m.as_object().expect("manifest is a JSON object");
    let actual: std::collections::BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    let expected: std::collections::BTreeSet<&str> = [
        "origin",
        "source_workspace_id",
        "source_head_id",
        "workspace_revision",
        "runtime_head_id",
        "sha256",
        "labels_sha256",
        "n_classes",
        "labels",
        "activated_at",
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "active Head-origin manifest must carry exactly the field set; got {actual:?}",
    );
    for forbidden in [
        "source_dataset_revision",
        "dataset_revision_at_train",
        "dataset_path",
        "training_cfg",
        "training_cfg_sha256",
    ] {
        assert!(
            !obj.contains_key(forbidden),
            "legacy field {forbidden:?} must not appear",
        );
    }
    let rev = obj["workspace_revision"]
        .as_object()
        .expect("workspace_revision is a sub-object");
    let rev_keys: std::collections::BTreeSet<&str> = rev.keys().map(String::as_str).collect();
    let expected_rev: std::collections::BTreeSet<&str> = ["id", "at"].into_iter().collect();
    assert_eq!(
        rev_keys, expected_rev,
        "active manifest's workspace_revision sub-object must carry exactly id + at",
    );
    assert_eq!(obj["origin"].as_str(), Some("head"));
    assert_eq!(
        obj["source_workspace_id"].as_str(),
        Some(ws_id.to_string().as_str())
    );
    assert_eq!(
        obj["source_head_id"].as_str(),
        Some(head_id.to_string().as_str())
    );
    assert_eq!(
        obj["runtime_head_id"].as_str(),
        Some(head_id.to_string().as_str())
    );
}

/// The on-disk active manifest for a Default-origin activation
/// carries no `source_*` fields and no `workspace_revision`.
#[tokio::test]
async fn active_default_manifest_on_disk_carries_field_subset() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let activation_id = v["activation_id"].as_str().expect("activation_id");

    let manifest_path = dir
        .path()
        .join("workspaces")
        .join("active")
        .join("generations")
        .join(activation_id)
        .join("manifest.json");
    let bytes = std::fs::read(&manifest_path).unwrap();
    let m: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let obj = m.as_object().expect("manifest is a JSON object");
    let actual: std::collections::BTreeSet<&str> = obj.keys().map(String::as_str).collect();
    let expected: std::collections::BTreeSet<&str> = [
        "origin",
        "runtime_head_id",
        "sha256",
        "labels_sha256",
        "n_classes",
        "labels",
        "activated_at",
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "active Default-origin manifest must carry exactly the field subset; got {actual:?}",
    );
    for forbidden in [
        "source_workspace_id",
        "source_head_id",
        "workspace_revision",
        "source_dataset_revision",
        "dataset_revision_at_train",
    ] {
        assert!(
            !obj.contains_key(forbidden),
            "Default-origin manifest must not carry {forbidden:?}",
        );
    }
    assert_eq!(obj["origin"].as_str(), Some("default"));
    assert_eq!(
        obj["runtime_head_id"].as_str(),
        Some(acoustics_lab::common::ids::DEFAULT_RUNTIME_HEAD_ID_STR),
    );
}

/// `POST /active` and `GET /active` Head-origin bodies carry
/// `workspace_revision`, never the legacy `source_dataset_revision` alias.
#[tokio::test]
async fn active_response_carries_workspace_revision_field() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let ws_id = WorkspaceId::parse("11111111-2222-4333-8444-555555555549").unwrap();
    let head_id = publish_one_trained_head(&state, ws_id);
    let r = router(state);

    let body = serde_json::json!({
        "workspace_id": ws_id.to_string(),
        "head_id": head_id.to_string(),
    });
    let resp = call(&r, Method::POST, "/active", Some(&body.to_string())).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert!(
        v["workspace_revision"].is_object(),
        "POST /active Head-origin response must carry `workspace_revision`; body={v}",
    );
    let rev = v["workspace_revision"].as_object().unwrap();
    assert!(rev.contains_key("id"), "workspace_revision must carry `id`");
    assert!(rev.contains_key("at"), "workspace_revision must carry `at`");
    assert!(
        v.get("source_dataset_revision").is_none(),
        "legacy `source_dataset_revision` must not appear; body={v}",
    );

    let resp = call(&r, Method::GET, "/active", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert!(v["workspace_revision"].is_object());
    assert!(
        v.get("source_dataset_revision").is_none(),
        "GET /active must not surface legacy alias; body={v}",
    );
    assert_eq!(v["source_workspace_alive"], true);
}

/// A legacy `source_dataset_revision` manifest parse-fails through
/// `read_active_manifest`: the alias drops as unknown and the now-required
/// `workspace_revision` is absent, so serde reports the missing field.
#[tokio::test]
async fn legacy_active_manifest_shape_parse_fails_on_read() {
    use acoustics_lab::file_mgr::schema::read_active_manifest;
    let dir = tempfile::tempdir().unwrap();
    // Stage a legacy-shape manifest directly and feed it to the schema reader
    // shared by `GET /active` and boot recovery: both must fail closed.
    let root = dir.path().join("workspaces");
    std::fs::create_dir_all(&root).unwrap();
    let activation_id = "11111111-2222-4333-8444-555555555560";
    let gen_dir = root.join("active").join("generations").join(activation_id);
    std::fs::create_dir_all(&gen_dir).unwrap();

    let round_1_body = serde_json::json!({
        "origin": "head",
        "source_workspace_id": "11111111-2222-4333-8444-555555555548",
        "source_head_id": "11111111-2222-4333-8444-555555555540",
        // Legacy alias in place of the now-required `workspace_revision`.
        "source_dataset_revision": { "id": 5, "at": "2026-05-07T13:00:00Z" },
        "runtime_head_id": "11111111-2222-4333-8444-555555555540",
        "sha256": "deadbeef",
        "labels_sha256": "cafef00d",
        "n_classes": 3,
        "labels": ["a", "b", "c"],
        "activated_at": "2026-05-07T12:34:56Z",
    });
    std::fs::write(
        gen_dir.join("manifest.json"),
        serde_json::to_vec(&round_1_body).unwrap(),
    )
    .unwrap();

    let res = read_active_manifest(&root, activation_id);
    assert!(
        res.is_err(),
        "legacy active manifest body must parse-fail (missing required `workspace_revision`); got {res:?}",
    );
}

#[tokio::test]
async fn get_active_returns_404_on_fresh_root() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let r = router(state);

    let resp = call(&r, Method::GET, "/active", None).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Concurrent `POST /active` must hold two invariants across the activation
/// worker (read -> publish -> install -> prune): publish+install must not
/// reorder (else `current.json` and the runtime `HotHead` diverge), and prune
/// must not run while a peer can publish (else the peer's dir falls outside this
/// request's `keep` and is deleted, leaving `current.json` dangling).
///
/// Defensive guard, not deterministic: the `oneshot` race window is narrow
/// enough that a buggy build can pass, so a failing run is a real regression.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn post_active_default_concurrent_requests_all_succeed() {
    let dir = tempfile::tempdir().unwrap();
    let state = fresh_state(dir.path());
    let root = state.files.root().to_path_buf();
    let r = router(state);

    const N: usize = 16;
    let mut handles = Vec::with_capacity(N);
    for _ in 0..N {
        let r = r.clone();
        handles.push(tokio::spawn(async move {
            let resp = call(&r, Method::POST, "/active", Some(r#"{"default": true}"#)).await;
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "concurrent activation rejected (status {})",
                resp.status(),
            );
            let v: serde_json::Value = json_body(resp).await;
            v["activation_id"].as_str().unwrap().to_string()
        }));
    }
    let mut activation_ids = Vec::with_capacity(N);
    for h in handles {
        activation_ids.push(h.await.expect("join"));
    }

    // Unique activation_ids: staging mints a fresh UUID per request.
    let mut sorted = activation_ids.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        activation_ids.len(),
        "duplicate activation_ids returned across concurrent requests: {activation_ids:?}",
    );

    // `GET /active` succeeds iff a peer prune did not delete the dir
    // `current.json` points at.
    let resp = call(&r, Method::GET, "/active", None).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /active 404'd after concurrent activations -- current generation \
         directory missing on disk",
    );
    let v: serde_json::Value = json_body(resp).await;
    let final_id = v["activation_id"].as_str().unwrap().to_string();
    assert!(
        activation_ids.contains(&final_id),
        "final current.json activation_id {final_id:?} not in any of the {} \
         concurrent responses {activation_ids:?}",
        activation_ids.len(),
    );

    // Prune retains {current, previous}: >2 means prune never ran,
    // missing-current means a dangling pointer.
    let generations = root.join("active").join("generations");
    let surviving: Vec<String> = std::fs::read_dir(&generations)
        .expect("read_dir generations")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    assert!(
        surviving.len() <= 2,
        "after {N} concurrent activations the generations dir has {} entries; \
         expected <=2 (current + previous): {surviving:?}",
        surviving.len(),
    );
    assert!(
        surviving.contains(&final_id),
        "current generation {final_id:?} missing from disk after concurrent prune; \
         on-disk surviving = {surviving:?}",
    );
}
