//! Integration tests for the workspace lifecycle routes, driving the
//! in-process router (no daemon binary) over a tempdir via `oneshot`.

#![allow(clippy::disallowed_methods)]

use std::time::{Duration, Instant};

use acousticslab::api::router_v1_nested;
use axum::http::{Method, StatusCode};

mod api_fixtures;
use api_fixtures::{call, fresh_app_state, json_body};

/// End-to-end happy path: create -> list -> get -> delete -> 404 on
/// follow-up GET -> staging dir swept clean.
#[tokio::test]
async fn workspace_lifecycle_round_trip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let r = router_v1_nested(fresh_app_state(dir.path()));

    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspaces",
        Some(r#"{"name":"test"}"#),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "POST /workspaces must succeed"
    );
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().expect("id is string").to_string();
    assert!(!id.is_empty(), "create response carries an id");
    assert_eq!(v["name"], "test");
    assert_eq!(
        v["workspace_revision"]["id"], 0,
        "fresh workspace has revision 0; body={v}"
    );

    let resp = call(&r, Method::GET, "/api/v1/workspaces", None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["workspaces"].as_array().expect("workspaces array");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["id"], id);
    assert_eq!(entries[0]["name"], "test");

    let resp = call(&r, Method::GET, &format!("/api/v1/workspaces/{id}"), None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["id"], id);
    assert_eq!(v["name"], "test");
    assert!(v["created_at"].as_str().is_some(), "created_at present");
    assert_eq!(v["workspace_revision"]["id"], 0);
    let heads = v["heads"].as_array().expect("heads array");
    assert!(heads.is_empty(), "fresh workspace has no heads");
    // Summary must not leak the assets key.
    assert!(v.get("assets").is_none(), "summary must not echo assets");

    // DELETE is async: 202 Accepted with a job_id, not 200.
    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{id}"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    let job_id = v["job_id"].as_str().expect("job_id is string");
    assert!(!job_id.is_empty(), "delete response carries job_id");

    // Cache and dir entry vanish immediately even though the drain
    // runs in the background, so the follow-up GET is 404 at once.
    let resp = call(&r, Method::GET, &format!("/api/v1/workspaces/{id}"), None).await;
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "deleted workspace must surface 404 on follow-up GET",
    );

    // Drain runs on the blocking pool, so poll for residue; a
    // dataset-less workspace converges in milliseconds. The fixture
    // nests the FsService root at `<dir>/workspaces`, so staging is
    // `workspaces/.tmp`, not the tempdir's top-level `.tmp`.
    let staging = root.join("workspaces").join(".tmp");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let any_residue = if staging.exists() {
            std::fs::read_dir(&staging)
                .map(|d| {
                    d.filter_map(Result::ok).any(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .starts_with("delete-workspace-")
                    })
                })
                .unwrap_or(false)
        } else {
            false
        };
        if !any_residue {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "workspace delete drain did not converge within 5 s; staging={}",
                staging.display()
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn create_workspace_accepts_optional_tags() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspaces",
        Some(r#"{"name":"scoped","tags":["  field-recordings ","pet-noises"]}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    // Tags are trimmed before persist.
    assert_eq!(v["tags"][0], "field-recordings");
    assert_eq!(v["tags"][1], "pet-noises");
    assert_eq!(v["tags"].as_array().map(|a| a.len()), Some(2));
}

#[tokio::test]
async fn create_workspace_rejects_invalid_tags() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspaces",
        Some(r#"{"name":"scoped","tags":["a/b"]}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn patch_workspace_renames_and_retags() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspaces",
        Some(r#"{"name":"orig","tags":["a"]}"#),
    )
    .await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();
    let prev_revision = v["workspace_revision"]["id"].as_u64().unwrap();
    let resp = call(
        &r,
        Method::PATCH,
        &format!("/api/v1/workspaces/{id}"),
        Some(r#"{"name":"renamed","tags":["b","c"]}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["name"], "renamed");
    assert_eq!(v["tags"][0], "b");
    assert_eq!(v["tags"][1], "c");
    // Revision unchanged: name + tag edits do not bump it.
    assert_eq!(v["workspace_revision"]["id"].as_u64(), Some(prev_revision));
    let resp = call(&r, Method::GET, &format!("/api/v1/workspaces/{id}"), None).await;
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["name"], "renamed");
}

#[tokio::test]
async fn patch_workspace_rejects_empty_body() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspaces",
        Some(r#"{"name":"x"}"#),
    )
    .await;
    let id = json_body::<serde_json::Value>(resp).await["id"]
        .as_str()
        .unwrap()
        .to_string();
    let resp = call(
        &r,
        Method::PATCH,
        &format!("/api/v1/workspaces/{id}"),
        Some(r#"{}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn patch_workspace_rejects_name_collision() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspaces",
        Some(r#"{"name":"taken"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspaces",
        Some(r#"{"name":"free"}"#),
    )
    .await;
    let v: serde_json::Value = json_body(resp).await;
    let id = v["id"].as_str().unwrap().to_string();
    // `Taken` collides with `taken` case-insensitively -> 409.
    let resp = call(
        &r,
        Method::PATCH,
        &format!("/api/v1/workspaces/{id}"),
        Some(r#"{"name":"Taken"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn patch_workspace_returns_404_for_unknown_id() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    // Well-formed UUID-v4 so routing parses it, but no such workspace.
    let phantom = "11111111-2222-4333-8444-555555555556";
    let resp = call(
        &r,
        Method::PATCH,
        &format!("/api/v1/workspaces/{phantom}"),
        Some(r#"{"name":"ghost"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Name uniqueness folds case via full Unicode `to_lowercase`, not ASCII-only.
/// The `café`/`cafÉ` pair differs only in the non-ASCII `é`(U+00E9)/`É`(U+00C9),
/// so it collides ONLY under Unicode folding; a regression to `to_ascii_lowercase`
/// (which leaves `É` untouched) would keep them distinct and fail this test loudly.
#[tokio::test]
async fn create_workspace_rejects_unicode_case_collision() {
    let dir = tempfile::tempdir().expect("tempdir");
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspaces",
        Some(r#"{"name":"café"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = call(
        &r,
        Method::POST,
        "/api/v1/workspaces",
        Some(r#"{"name":"cafÉ"}"#),
    )
    .await;
    assert_eq!(
        resp.status(),
        StatusCode::CONFLICT,
        "`cafÉ` must collide with the existing `café` under Unicode case folding \
         (`É`->`é`); ASCII-only folding would leave `É` distinct",
    );
}
