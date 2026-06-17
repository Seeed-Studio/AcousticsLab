//! Integration tests for the `/workspaces/{id}/assets` dataset routes (GET
//! list/stream/JSONL-page/byte-slice, PUT upload, always-async DELETE), driven
//! in-process against a tempdir workspace to pin wire shape + traversal negatives.

#![allow(clippy::disallowed_methods)]

use std::time::{Duration, Instant};

use acoustics_lab::api::router_v1_nested;
use axum::body::to_bytes;
use axum::http::{Method, StatusCode, header};
use axum::response::Response;

mod api_fixtures;
use api_fixtures::{
    call, create_workspace, fixture_workspace_dir, fresh_app_state, json_body, upload,
};

async fn body_bytes(resp: Response) -> Vec<u8> {
    to_bytes(resp.into_body(), 1 << 22)
        .await
        .expect("body")
        .to_vec()
}

#[tokio::test]
async fn dataset_happy_path_upload_list_get_delete() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));

    let ws = create_workspace(&r, "main").await;
    let bytes = b"audio body";
    let resp = upload(&r, &ws, "datasets/audio_dataset/cat/sample.wav", bytes).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["path"], "datasets/audio_dataset/cat/sample.wav");
    assert_eq!(v["size_bytes"], bytes.len());
    assert_eq!(v["workspace_revision_id"], 1);
    assert_eq!(v["sha256"].as_str().unwrap().len(), 64);

    // Leaf subdirs are created lazily by the writer that touches them, so only
    // the just-uploaded `datasets/` exists; converter/log dirs stay absent until
    // a producer runs, and `.tmp/` is always excluded from listings.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["entries"].as_array().unwrap();
    let names: Vec<&str> = entries.iter().filter_map(|e| e["name"].as_str()).collect();
    assert!(names.contains(&"datasets"), "datasets/ in {names:?}");
    assert!(
        !names.contains(&"converters"),
        "converters/ must be absent until a converter has run: {names:?}",
    );
    assert!(
        !names.contains(&".tmp"),
        ".tmp/ must be excluded from listings: {names:?}"
    );

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/audio_dataset"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["name"], "cat");

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/audio_dataset/cat/sample.wav"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert_eq!(ct, "audio/wav");
    let received = body_bytes(resp).await;
    assert_eq!(received, bytes);

    // Async dispatch: 202 + {job_id}; rename + tombstone are durable but the
    // staged drain runs in the background (hence the poll below).
    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/audio_dataset/cat"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    assert!(v["job_id"].as_str().is_some(), "job_id present; body={v}");

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspaces/{ws}/assets/datasets/audio_dataset/cat/sample.wav"),
            None,
        )
        .await;
        if resp.status() == StatusCode::NOT_FOUND {
            break;
        }
        if Instant::now() >= deadline {
            panic!(
                "deleted asset still observable after 5 s; status={}",
                resp.status(),
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// `converters/` shares the upload + delete dispatcher with `datasets/`; only
/// the tombstone/`JobType` variant and the on-disk subdir differ.
#[tokio::test]
async fn converters_happy_path_upload_get_delete() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let bytes = br#"{"format":"tfjs","weights":[]}"#;
    let resp = upload(&r, &ws, "converters/tfjs/model.json", bytes).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["path"], "converters/tfjs/model.json");
    assert_eq!(v["size_bytes"], bytes.len());

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/converters/tfjs/model.json"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert_eq!(ct, "application/json");
    let received = body_bytes(resp).await;
    assert_eq!(received, bytes);

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/converters/tfjs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspaces/{ws}/assets/converters/tfjs/model.json"),
            None,
        )
        .await;
        if resp.status() == StatusCode::NOT_FOUND {
            break;
        }
        if Instant::now() >= deadline {
            panic!("converter delete drain did not converge");
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Paths whose top-level isn't `datasets/`/`converters/` reject with 400.
#[tokio::test]
async fn upload_rejects_non_mutable_top_level() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    for bad in [
        "heads/x.mpk",
        "training_logs/log.jsonl",
        "scratch/file.bin",
        "datasets",
        "converters",
    ] {
        let resp = upload(&r, &ws, bad, b"x").await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "upload `{bad}` must reject; got {}",
            resp.status()
        );
    }
}

#[tokio::test]
async fn upload_rejects_path_traversal_variants() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    // Empty path `""` is omitted: it collapses the URL to `/assets`, which has
    // only GET registered, so the wire surface is 405 not 400.
    for (label, bad_path) in [
        (".. literal", ".."),
        (".. with subpath", "../etc/passwd"),
        ("absolute", "/abs"),
        ("interior dotdot", "foo/../bar"),
        ("url-encoded ..", "%2E%2E%2Fetc"),
        ("backslash", "foo\\bar"),
        ("trailing slash", "foo/"),
        ("double slash", "foo//bar"),
        ("nul byte", "foo\0bar"),
        ("control byte", "foo\nbar"),
        ("non-ascii", "caf\u{00e9}/foo"),
    ] {
        let resp = upload(&r, &ws, bad_path, b"x").await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "{label} ({bad_path:?}) must reject with 400; got {}",
            resp.status(),
        );
        let v: serde_json::Value = json_body(resp).await;
        assert_eq!(v["code"], "bad_request", "{label} envelope code; body={v}");
    }
}

#[tokio::test]
async fn get_and_delete_reject_path_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    // axum 0.8 URL-decodes the wildcard before the handler, so the parser catches
    // the literal `..` via LeadingDot.
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/.."),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/.."),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn upload_bumps_workspace_revision() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let resp = call(&r, Method::GET, &format!("/api/v1/workspaces/{ws}"), None).await;
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["workspace_revision"]["id"], 0);
    let resp = upload(&r, &ws, "datasets/cls/a.json", br#"{"k":1}"#).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let resp = call(&r, Method::GET, &format!("/api/v1/workspaces/{ws}"), None).await;
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["workspace_revision"]["id"], 1);
}

#[tokio::test]
async fn mime_types_match_redesign_section_7() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    for (filename, expected_ct) in [
        ("audio.wav", "audio/wav"),
        ("manifest.json", "application/json"),
        ("labels.txt", "text/plain; charset=utf-8"),
        ("blob.bin", "application/octet-stream"),
    ] {
        let upload_path = format!("datasets/cls/{filename}");
        let resp = upload(&r, &ws, &upload_path, b"x").await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "upload {upload_path} succeeded"
        );
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspaces/{ws}/assets/{upload_path}"),
            None,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert_eq!(ct, expected_ct, "MIME for {filename}");
    }
}

/// Every listed entry (file and directory) carries `mtime` (RFC3339 UTC); pinned
/// because clients sort JSONL log files by mtime recency, so a refactor that strips
/// or stops populating the field is a regression.
#[tokio::test]
async fn list_assets_carries_mtime_per_entry() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    // One upload so both the file branch and a non-empty subtree below exist.
    let resp = upload(&r, &ws, "datasets/cls/sample.bin", b"x").await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["entries"].as_array().expect("entries array");
    assert!(!entries.is_empty(), "fresh workspace has direct children");
    for entry in entries {
        let name = entry["name"].as_str().unwrap_or("<no-name>");
        let mtime = entry["mtime"]
            .as_str()
            .unwrap_or_else(|| panic!("entry {name:?} missing mtime; entry={entry}"));
        // Loose RFC3339 check (year + `T` + `Z`) catches an epoch-millis or
        // locale-formatted regression.
        assert!(
            mtime.len() >= 20 && mtime.chars().nth(4) == Some('-') && mtime.ends_with('Z'),
            "entry {name:?} mtime {mtime:?} is not RFC3339 UTC",
        );
    }

    // Sub-listing via the wildcard form (there is no `?path=` query variant).
    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/cls"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let entries = v["entries"].as_array().expect("entries array");
    let sample = entries
        .iter()
        .find(|e| e["name"].as_str() == Some("sample.bin"))
        .expect("sample.bin in listing");
    assert!(
        sample["mtime"].as_str().is_some(),
        "file entry must carry mtime; got {sample}",
    );
}

/// `?after_seq=&limit=` returns a JSONL page (`{ events, next_after_seq }`) when
/// the resolved file ends in `.jsonl`; pins that log reads reached via `/assets`
/// keep the dedicated-route page shape.
#[tokio::test]
async fn assets_jsonl_page_round_trips_on_jsonl_file() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;

    // Synthesise the converter_logs JSONL on disk; a wire-shape pin needs no real
    // producer run.
    let ws_id = acoustics_lab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let workspace_dir = fixture_workspace_dir(dir.path(), ws_id.to_string());
    let log_dir = workspace_dir.join("converter_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let job_id = acoustics_lab::common::ids::JobId::new();
    let log_path = log_dir.join(format!("{job_id}.jsonl"));
    let lines = [
        r#"{"seq":1,"at":"2026-05-07T12:00:00Z","message":"first"}"#,
        r#"{"seq":2,"at":"2026-05-07T12:00:01Z","message":"second"}"#,
        r#"{"seq":3,"at":"2026-05-07T12:00:02Z","message":"third"}"#,
    ];
    std::fs::write(&log_path, lines.join("\n")).unwrap();

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/converter_logs/{job_id}.jsonl?limit=2"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let events = v["events"].as_array().expect("events array");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["seq"], 1);
    assert_eq!(events[1]["seq"], 2);
    assert_eq!(v["next_after_seq"].as_u64(), Some(2));

    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspaces/{ws}/assets/converter_logs/{job_id}.jsonl?after_seq=2&limit=10",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    let events = v["events"].as_array().unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["seq"], 3);
    assert_eq!(v["next_after_seq"].as_u64(), Some(3));
}

/// `?after_seq=/?limit=` on a non-`.jsonl` file 400s: the JSON-page response is
/// reachable only by paging a `.jsonl` file, keeping the byte-stream surface
/// unambiguous.
#[tokio::test]
async fn assets_jsonl_page_rejects_on_binary_file() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let resp = upload(&r, &ws, "datasets/cls/sample.bin", b"audio").await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/cls/sample.bin?after_seq=0&limit=5"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
    let err = v["error"].as_str().unwrap_or_default();
    assert!(
        err.contains(".jsonl"),
        "diagnostic must name the .jsonl gate; got {err}",
    );
}

/// With no query params, `GET /assets/{*path}` keeps streaming bytes; guards the
/// JSONL paging branch from silently changing the wire for every binary asset.
#[tokio::test]
async fn assets_byte_stream_unchanged_when_no_query_params() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body = b"raw audio";
    let resp = upload(&r, &ws, "datasets/cls/clip.wav", body).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/cls/clip.wav"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert_eq!(ct, "audio/wav", "byte-stream branch keeps MIME mapping");
    let received = body_bytes(resp).await;
    assert_eq!(received, body);
}

/// `?byte_offset=/?byte_limit=` slice a file to the requested range; pins the
/// random-access surface that fragment-loading binary clients (e.g. WAV seek) rely on.
#[tokio::test]
async fn assets_byte_range_returns_slice_with_offset_and_limit() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body: Vec<u8> = (0..256u32).map(|i| (i & 0xff) as u8).collect();
    let resp = upload(&r, &ws, "datasets/cls/payload.bin", &body).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspaces/{ws}/assets/datasets/cls/payload.bin?byte_offset=10&byte_limit=20",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let cl = resp
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    assert_eq!(cl, Some(20), "Content-Length reflects slice size");
    let received = body_bytes(resp).await;
    assert_eq!(received, body[10..30]);
}

/// `?byte_offset=` alone streams from the offset to EOF (no partner param required).
#[tokio::test]
async fn assets_byte_range_offset_only_streams_to_eof() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body: Vec<u8> = (0..100u32).map(|i| (i & 0xff) as u8).collect();
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", &body).await;

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/cls/payload.bin?byte_offset=70"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let received = body_bytes(resp).await;
    assert_eq!(received, body[70..]);
}

/// `?byte_limit=` alone streams the first N bytes (offset
/// defaults to 0).
#[tokio::test]
async fn assets_byte_range_limit_only_streams_from_zero() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body: Vec<u8> = (0..100u32).map(|i| (i & 0xff) as u8).collect();
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", &body).await;

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/cls/payload.bin?byte_limit=15"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let received = body_bytes(resp).await;
    assert_eq!(received, body[..15]);
}

/// `?byte_limit=0` is a valid degenerate slice -> 200 + `Content-Length: 0`; the
/// `take(0)` boundary must terminate cleanly, not 400 (over-strict) or stream the
/// whole file (under-strict).
#[tokio::test]
async fn assets_byte_range_limit_zero_returns_empty_slice() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body: Vec<u8> = (0..50u32).map(|i| (i & 0xff) as u8).collect();
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", &body).await;

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/cls/payload.bin?byte_limit=0"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let cl = resp
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    assert_eq!(cl, Some(0), "byte_limit=0 advertises Content-Length: 0");
    let received = body_bytes(resp).await;
    assert!(received.is_empty(), "byte_limit=0 yields zero bytes");
}

/// `byte_offset + byte_limit` past EOF clamps silently to the remainder; the slice
/// is best-effort (fewer bytes, never an error).
#[tokio::test]
async fn assets_byte_range_clamps_oversized_limit_to_eof() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body: Vec<u8> = (0..50u32).map(|i| (i & 0xff) as u8).collect();
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", &body).await;

    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspaces/{ws}/assets/datasets/cls/payload.bin?byte_offset=40&byte_limit=1000",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let cl = resp
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());
    assert_eq!(cl, Some(10), "clamped Content-Length");
    let received = body_bytes(resp).await;
    assert_eq!(received, body[40..]);
}

/// `byte_offset == file_size` is a valid zero-byte slice (whereas `> file_size`
/// 400s), so offsets that land exactly on EOF don't spuriously error.
#[tokio::test]
async fn assets_byte_range_offset_at_eof_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let body = b"hello";
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", body).await;

    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspaces/{ws}/assets/datasets/cls/payload.bin?byte_offset={}",
            body.len(),
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let received = body_bytes(resp).await;
    assert!(received.is_empty(), "offset at EOF yields zero bytes");
}

/// `byte_offset > file_size` 400s rather than silently yielding nothing for a
/// mis-computed offset.
#[tokio::test]
async fn assets_byte_range_offset_past_eof_returns_400() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let _ = upload(&r, &ws, "datasets/cls/payload.bin", b"hi").await;

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/cls/payload.bin?byte_offset=999"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
    let err = v["error"].as_str().unwrap_or_default();
    assert!(
        err.contains("byte_offset"),
        "diagnostic must name the gate; got {err}",
    );
}

/// Byte-range slicing works on `.jsonl` files too: the gate is the query namespace
/// (`?byte_offset=` raw bytes vs `?after_seq=` parsed events), not the extension.
#[tokio::test]
async fn assets_byte_range_works_on_jsonl_file() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let ws_id = acoustics_lab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let workspace_dir = fixture_workspace_dir(dir.path(), ws_id.to_string());
    let log_dir = workspace_dir.join("converter_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let job_id = acoustics_lab::common::ids::JobId::new();
    let log_path = log_dir.join(format!("{job_id}.jsonl"));
    let raw = b"{\"seq\":1}\n{\"seq\":2}\n{\"seq\":3}\n";
    std::fs::write(&log_path, raw).unwrap();

    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspaces/{ws}/assets/converter_logs/{job_id}.jsonl?byte_offset=10&byte_limit=10",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let received = body_bytes(resp).await;
    assert_eq!(received, &raw[10..20]);
}

/// Mixing the byte-range and JSONL-page query namespaces 400s, with a diagnostic
/// naming both axes so the client sees which combination it triggered.
#[tokio::test]
async fn assets_byte_range_rejects_with_jsonl_paging_params() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let ws_id = acoustics_lab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let workspace_dir = fixture_workspace_dir(dir.path(), ws_id.to_string());
    let log_dir = workspace_dir.join("converter_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let job_id = acoustics_lab::common::ids::JobId::new();
    std::fs::write(
        log_dir.join(format!("{job_id}.jsonl")),
        b"{\"seq\":1}\n{\"seq\":2}\n",
    )
    .unwrap();

    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspaces/{ws}/assets/converter_logs/{job_id}.jsonl?byte_offset=0&after_seq=0",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    let err = v["error"].as_str().unwrap_or_default();
    assert!(
        err.contains("byte_offset") && err.contains("after_seq"),
        "diagnostic must name both axes; got {err}",
    );

    // `limit`'s JSONL-paging meaning collides with the byte-slice ceiling -> 400.
    let resp = call(
        &r,
        Method::GET,
        &format!(
            "/api/v1/workspaces/{ws}/assets/converter_logs/{job_id}.jsonl?byte_limit=4&limit=2",
        ),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// Byte-range params on a directory 400 (slice is file-only) so a dir-listing
/// request with an accidental byte param fails loudly instead of falling through.
#[tokio::test]
async fn assets_byte_range_rejects_on_directory() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let _ = upload(&r, &ws, "datasets/cls/sample.bin", b"x").await;

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/cls?byte_offset=0&byte_limit=1"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
}

/// `?after_seq=` on a directory 400s; without this branch an accidental
/// `?after_seq=` falls through to the listing instead of the expected page.
#[tokio::test]
async fn assets_jsonl_page_rejects_on_directory() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let resp = upload(&r, &ws, "datasets/cls/sample.bin", b"x").await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/cls?after_seq=0"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["code"], "bad_request");
}

/// Single-file `.jsonl` delete routes through the async tombstone+stage+drain path
/// like dataset/converter: 202 + `{ job_id }`, file staged under `.tmp/`, background
/// drain unlinks it (no sync `removed` field).
#[tokio::test]
async fn delete_assets_training_log_file_returns_async_job_id() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let ws_id = acoustics_lab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let workspace_dir = fixture_workspace_dir(dir.path(), ws_id.to_string());
    let log_dir = workspace_dir.join("training_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let job_id = acoustics_lab::common::ids::JobId::new();
    let log_path = log_dir.join(format!("{job_id}.jsonl"));
    std::fs::write(
        &log_path,
        r#"{"seq":1,"at":"2026-05-07T12:00:00Z","message":"hi"}"#,
    )
    .unwrap();

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/training_logs/{job_id}.jsonl"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    let returned_job = v["job_id"].as_str().expect("job_id present on async wipe");
    assert!(!returned_job.is_empty());
    assert!(
        v.get("removed").is_none(),
        "async log delete carries no `removed` field",
    );

    // Poll the per-job `.tmp/` tombstone, cleared only after the drain completes.
    // The synchronous rename under the workspace mutex removes `log_path` BEFORE the
    // 202 returns, so polling it would be a no-op; only the tombstone tracks drain.
    let tombstone_path = workspace_dir
        .join(".tmp")
        .join(format!("delete-training-logs-{returned_job}.json"));
    let deadline = Instant::now() + Duration::from_secs(5);
    while tombstone_path.exists() {
        if Instant::now() > deadline {
            panic!(
                "training-logs delete tombstone {} still present 5 s after delete",
                tombstone_path.display(),
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        !log_path.exists(),
        "single-file rename moved the jsonl out of log_dir",
    );
}

/// Whole-dir `converter_logs` delete: 202 + job_id + background drain; the empty
/// dir is recreated so later producer runs find the canonical shape.
#[tokio::test]
async fn delete_assets_converter_logs_whole_dir_returns_async_job_id() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let ws_id = acoustics_lab::common::ids::WorkspaceId::parse(&ws).unwrap();
    let workspace_dir = fixture_workspace_dir(dir.path(), ws_id.to_string());
    let log_dir = workspace_dir.join("converter_logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let mut jsonl_paths = Vec::new();
    for n in 0..3 {
        let job_id = acoustics_lab::common::ids::JobId::new();
        let p = log_dir.join(format!("{job_id}.jsonl"));
        std::fs::write(&p, format!(r#"{{"seq":{n},"at":"2026-05-07T12:00:00Z"}}"#)).unwrap();
        jsonl_paths.push(p);
    }

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/converter_logs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    let returned_job = v["job_id"].as_str().expect("job_id on whole-tree wipe");
    assert!(!returned_job.is_empty());

    // Poll the per-job tombstone, cleared only after the drain completes. The
    // whole-tree rename moves the log dir under staging BEFORE the 202 returns, so
    // the original child paths are already absent; only the tombstone tracks drain.
    let tombstone_path = workspace_dir
        .join(".tmp")
        .join(format!("delete-converter-logs-{returned_job}.json"));
    let deadline = Instant::now() + Duration::from_secs(5);
    while tombstone_path.exists() {
        if Instant::now() > deadline {
            panic!(
                "converter-logs delete tombstone {} still present 5 s after delete",
                tombstone_path.display(),
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(log_dir.exists(), "empty converter_logs/ recreated");
    assert!(log_dir.is_dir());
    for p in &jsonl_paths {
        assert!(!p.exists(), "{} drained", p.display());
    }
}

/// Whole-tree wipe of an existing-but-empty `training_logs/` returns 202 + job_id;
/// materializes the lazily-created dir to exercise the exists-but-empty idempotent
/// clear (the never-created 404 case is the absent test).
#[tokio::test]
async fn delete_assets_training_logs_whole_dir_succeeds_on_empty() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;

    let ws_path = fixture_workspace_dir(dir.path(), &ws).join("training_logs");
    std::fs::create_dir_all(&ws_path).expect("mkdir training_logs");

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/training_logs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    assert!(v["job_id"].as_str().is_some());
}

/// Whole-tree wipe of a never-created (lazy) `training_logs/` returns 404, so an
/// idempotent clear distinguishes "no logs to clear" from a successful purge.
#[tokio::test]
async fn delete_assets_training_logs_absent_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/training_logs"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Single-file delete of a missing `.jsonl` returns 404; with no sync `removed: 0`
/// fast-path, idempotent "remove this log file" must check 404 explicitly.
#[tokio::test]
async fn delete_assets_training_log_file_missing_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let phantom = acoustics_lab::common::ids::JobId::new();

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/training_logs/{phantom}.jsonl"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

/// Whole-tree `datasets/` wipe returns 202 + JobId: rename durable under the
/// per-workspace mutex, drain in the background (queued, not done); the empty dir is
/// recreated for the canonical shape.
#[tokio::test]
async fn delete_assets_datasets_whole_tree_returns_job_id() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let _ = upload(&r, &ws, "datasets/cls/sample.bin", b"x").await;

    let resp = call(
        &r,
        Method::DELETE,
        &format!("/api/v1/workspaces/{ws}/assets/datasets"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v: serde_json::Value = json_body(resp).await;
    let job_id = v["job_id"].as_str().expect("job_id present on async wipe");
    assert!(!job_id.is_empty());
    assert!(v.get("removed").is_none(), "async wipe carries no removed");
}

/// Uploads to the log trees are rejected at the validator (logs are daemon-produced).
#[tokio::test]
async fn upload_rejects_log_tree_paths() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    for path in ["training_logs/manual.jsonl", "converter_logs/manual.jsonl"] {
        let resp = upload(&r, &ws, path, b"forbidden").await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "upload to {path} must be rejected; got {}",
            resp.status(),
        );
    }
}

// Conflict semantics: uploads/file-deletes take no path leases and conflict only
// with an in-flight WorkspaceDelete for the same workspace. That contract is pinned
// at the lib level; no HTTP integration test lives here because the workspace-delete
// drain is too short-lived on real filesystems to catch without a synthetic hook.

/// `id` field of the `GET /workspaces/{id}` summary's `workspace_revision`.
async fn summary_revision(r: &axum::Router, ws: &str) -> u64 {
    let resp = call(r, Method::GET, &format!("/api/v1/workspaces/{ws}"), None).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    v["workspace_revision"]["id"].as_u64().expect("revision id")
}

#[tokio::test]
async fn rename_category_moves_slices_and_bumps_revision() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;

    // Two uploads bump the revision to 1 then 2.
    assert_eq!(
        upload(&r, &ws, "datasets/dog/sample.wav", b"audio body")
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        upload(&r, &ws, "datasets/dog/clip.wav", b"second clip")
            .await
            .status(),
        StatusCode::OK
    );

    // Rename dog -> puppy: 200 + new revision 3.
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/datasets/dog/rename"),
        Some(r#"{"to_name":"puppy"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    assert_eq!(v["workspace_revision_id"], 3);

    for (file, want) in [
        ("sample.wav", &b"audio body"[..]),
        ("clip.wav", &b"second clip"[..]),
    ] {
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspaces/{ws}/assets/datasets/puppy/{file}"),
            None,
        )
        .await;
        assert_eq!(resp.status(), StatusCode::OK, "{file} under puppy");
        assert_eq!(body_bytes(resp).await, want);
    }

    let resp = call(
        &r,
        Method::GET,
        &format!("/api/v1/workspaces/{ws}/assets/datasets/dog"),
        None,
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    assert_eq!(summary_revision(&r, &ws).await, 3);
}

#[tokio::test]
async fn rename_category_self_rename_is_noop_200() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let resp = upload(&r, &ws, "datasets/dog/sample.wav", b"x").await;
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/datasets/dog/rename"),
        Some(r#"{"to_name":"dog"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v: serde_json::Value = json_body(resp).await;
    // Self-rename is a no-op: revision stays at 1.
    assert_eq!(v["workspace_revision_id"], 1);
    assert_eq!(summary_revision(&r, &ws).await, 1);
}

#[tokio::test]
async fn rename_category_missing_source_404() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/datasets/ghost/rename"),
        Some(r#"{"to_name":"x"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn rename_category_existing_target_409() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    assert_eq!(
        upload(&r, &ws, "datasets/dog/a.wav", b"a").await.status(),
        StatusCode::OK
    );
    assert_eq!(
        upload(&r, &ws, "datasets/cat/b.wav", b"b").await.status(),
        StatusCode::OK
    );

    let resp = call(
        &r,
        Method::POST,
        &format!("/api/v1/workspaces/{ws}/datasets/dog/rename"),
        Some(r#"{"to_name":"cat"}"#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::CONFLICT);
    // Rejected rename leaves state untouched: revision stays at 2, both dirs survive.
    assert_eq!(summary_revision(&r, &ws).await, 2);
    for cat in ["dog", "cat"] {
        let resp = call(
            &r,
            Method::GET,
            &format!("/api/v1/workspaces/{ws}/assets/datasets/{cat}"),
            None,
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "{cat} must survive a rejected rename"
        );
    }
}

#[tokio::test]
async fn rename_category_invalid_name_400() {
    let dir = tempfile::tempdir().unwrap();
    let r = router_v1_nested(fresh_app_state(dir.path()));
    let ws = create_workspace(&r, "main").await;
    assert_eq!(
        upload(&r, &ws, "datasets/dog/a.wav", b"a").await.status(),
        StatusCode::OK
    );

    // AssetPath::parse rejects leading-'.' (LeadingDot) and empty (EmptyComponent via
    // the trailing '/') names -> 400.
    for bad in [r#"{"to_name":".hidden"}"#, r#"{"to_name":""}"#] {
        let resp = call(
            &r,
            Method::POST,
            &format!("/api/v1/workspaces/{ws}/datasets/dog/rename"),
            Some(bad),
        )
        .await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "rename body {bad} must 400"
        );
    }
}
