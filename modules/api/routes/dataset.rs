//! Workspace asset routes on `/workspaces/{id}/assets/{*path}`; HTTP method picks
//! the op (GET stream/list/JSONL-page/byte-slice, PUT upload, DELETE async). PUT
//! is idempotent on disk (atomic rename-into-tree) yet bumps `workspace_revision`.

use std::path::PathBuf;
use std::sync::Arc;

use crate::common::asset_path::AssetPath;
use crate::common::ids::WorkspaceId;
use crate::file_mgr::log_page::{DEFAULT_LOG_PAGE_LIMIT, read_jsonl_page};
use crate::file_mgr::{
    DATASETS_DIR_NAME, DEFAULT_DATASET_LIST_LIMIT, DatasetListing, DatasetUploadReceipt, FsService,
    MAX_DATASET_LIST_LIMIT, RenameCategoryReceipt, content_type_from_path, hex_lowercase,
};
use axum::Router;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::header;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::{ApiJson, ApiQuery, CONTROL_JSON_BODY_LIMIT};
use crate::api::routes::workspace::{
    classify_workspace_existence_error, first_io_error_is_not_found,
};

/// axum URL-decodes wildcard captures, so `%2E%2E%2F` arrives as `../` and is rejected via `LeadingDot`.
fn parse_asset_path(raw: &str) -> Result<AssetPath, ApiError> {
    AssetPath::parse(raw).map_err(|e| ApiError::Bad(format!("invalid asset path: {e}")))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListAssetsQuery {
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

async fn list_assets(
    State(files): State<Arc<dyn FsService>>,
    Path(id): Path<String>,
    ApiQuery(q): ApiQuery<ListAssetsQuery>,
) -> Result<Json<DatasetListing>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let offset = q.offset.unwrap_or(0);
    let limit = q
        .limit
        .unwrap_or(DEFAULT_DATASET_LIST_LIMIT)
        .min(MAX_DATASET_LIST_LIMIT);
    // `None` lists the workspace root; `.tmp/` staging is excluded (unaddressable).
    let listing =
        task::spawn_blocking(move || files.list_workspace_children(&id, None, offset, limit))
            .await?
            .map_err(|e| classify_workspace_existence_error(&id, e))?;
    Ok(Json(listing))
}

/// `GET /assets/{*path}` query dispatch: dir listing (`offset`/`limit`), `.jsonl`
/// page (`after_seq`/`limit`), or byte slice (`byte_offset`/`byte_limit`). The
/// byte-range and JSONL-page namespaces are mutually exclusive (mixing is 400); a
/// file with no query streams full bytes.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GetAssetQuery {
    /// JSONL cursor: yields rows with `seq > after_seq` (exclusive).
    #[serde(default)]
    after_seq: Option<u64>,
    #[serde(default)]
    offset: Option<usize>,
    /// Ceiling: dir entries ([`MAX_DATASET_LIST_LIMIT`]) or JSONL lines
    /// ([`crate::file_mgr::log_page::MAX_LOG_PAGE_LIMIT`]).
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    byte_offset: Option<u64>,
    #[serde(default)]
    byte_limit: Option<u64>,
}

async fn get_asset(
    State(files): State<Arc<dyn FsService>>,
    Path((id, raw_path)): Path<(String, String)>,
    ApiQuery(q): ApiQuery<GetAssetQuery>,
) -> Result<Response, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let asset_path = parse_asset_path(&raw_path)?;

    // Resolve + stat off-mutex (read-only) to branch file-vs-dir.
    let files_for_resolve = files.clone();
    let asset_path_for_resolve = asset_path.clone();
    let resolved: PathBuf = task::spawn_blocking(move || {
        files_for_resolve.workspace_asset_path(&id, &asset_path_for_resolve)
    })
    .await?
    .map_err(|e| classify_workspace_existence_error(&id, e))?;

    let stat_path = resolved.clone();
    let md = match task::spawn_blocking(move || std::fs::symlink_metadata(&stat_path)).await? {
        Ok(md) => md,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(ApiError::NotFound(format!(
                "asset {} in workspace {id}",
                asset_path.as_str(),
            )));
        }
        Err(source) => {
            return Err(ApiError::File(crate::file_mgr::io_err(
                resolved.display(),
                source,
            )));
        }
    };

    if md.is_dir() {
        // Reject file-only params, not silently swallow a client typo.
        if q.after_seq.is_some() {
            return Err(ApiError::Bad(format!(
                "?after_seq= applies only to .jsonl files, not directory {:?}",
                asset_path.as_str(),
            )));
        }
        if q.byte_offset.is_some() || q.byte_limit.is_some() {
            return Err(ApiError::Bad(format!(
                "?byte_offset= / ?byte_limit= apply only to regular files, not directory {:?}",
                asset_path.as_str(),
            )));
        }
        let offset = q.offset.unwrap_or(0);
        let limit = q
            .limit
            .unwrap_or(DEFAULT_DATASET_LIST_LIMIT)
            .min(MAX_DATASET_LIST_LIMIT);
        let files_for_dir = files.clone();
        let asset_for_dir = asset_path.clone();
        let listing = task::spawn_blocking(move || {
            files_for_dir.list_workspace_children(&id, Some(&asset_for_dir), offset, limit)
        })
        .await?
        .map_err(|e| classify_workspace_existence_error(&id, e))?;
        return Ok(Json(listing).into_response());
    }
    if !md.is_file() {
        return Err(ApiError::Bad(format!(
            "asset {} is not a regular file or directory",
            asset_path.as_str(),
        )));
    }

    if q.offset.is_some() {
        return Err(ApiError::Bad(format!(
            "?offset= applies only to directory listings, not file {:?}",
            asset_path.as_str(),
        )));
    }

    let wants_byte_range = q.byte_offset.is_some() || q.byte_limit.is_some();
    let wants_jsonl_page = q.after_seq.is_some() || q.limit.is_some();

    if wants_byte_range && wants_jsonl_page {
        return Err(ApiError::Bad(format!(
            "?byte_offset= / ?byte_limit= cannot combine with ?after_seq= / ?limit= for {:?}",
            asset_path.as_str(),
        )));
    }

    // Paging is `.jsonl`-only so the byte-stream surface stays unambiguous; raw
    // `.jsonl` bytes use `?byte_offset=`.
    if wants_jsonl_page {
        let is_jsonl = resolved
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("jsonl"));
        if !is_jsonl {
            return Err(ApiError::Bad(format!(
                "?after_seq= / ?limit= apply only to .jsonl files, not {:?}",
                asset_path.as_str(),
            )));
        }
        let after_seq = q.after_seq.unwrap_or(0);
        let limit = q.limit.unwrap_or(DEFAULT_LOG_PAGE_LIMIT);
        let path_for_page = resolved.clone();
        let page = task::spawn_blocking(move || read_jsonl_page(&path_for_page, after_seq, limit))
            .await?
            .map_err(|source| {
                ApiError::File(crate::file_mgr::io_err(resolved.display(), source))
            })?;
        return Ok(Json(page).into_response());
    }

    // `Content-Length` is the slice carried: OOR `byte_offset` is 400, oversized `byte_limit` clamps to the remainder.
    let total = md.len();
    let (byte_offset, take_limit) = if wants_byte_range {
        let off = q.byte_offset.unwrap_or(0);
        if off > total {
            return Err(ApiError::Bad(format!(
                "?byte_offset={off} exceeds file size {total} for {:?}",
                asset_path.as_str(),
            )));
        }
        let remaining = total - off;
        let take = q.byte_limit.unwrap_or(remaining).min(remaining);
        (off, take)
    } else {
        (0, total)
    };

    // No workspace mutex while streaming (resolve already validated the path), so concurrent mutations are not blocked.
    use tokio::io::{AsyncReadExt as _, AsyncSeekExt as _};
    let mut f = tokio::fs::File::open(&resolved)
        .await
        .map_err(|source| ApiError::File(crate::file_mgr::io_err(resolved.display(), source)))?;
    // Re-stat the OPEN handle: a mutex-free delete-recreate can swap in a SHORTER
    // inode mid-flight, so pre-open `total`/`take_limit` would over-promise
    // `Content-Length` and desync HTTP framing; clamping to the opened inode's len
    // leaves a grown file unaffected (smaller `total`'s `take_limit` wins).
    let real_len = f
        .metadata()
        .await
        .map_err(|source| ApiError::File(crate::file_mgr::io_err(resolved.display(), source)))?
        .len();
    let effective_take = take_limit.min(real_len.saturating_sub(byte_offset));
    if byte_offset > 0 {
        f.seek(std::io::SeekFrom::Start(byte_offset))
            .await
            .map_err(|source| {
                ApiError::File(crate::file_mgr::io_err(resolved.display(), source))
            })?;
    }
    let stream = tokio_util::io::ReaderStream::new(f.take(effective_take));
    let body = Body::from_stream(stream);
    let content_type = content_type_from_path(&resolved);
    let resp = Response::builder()
        .header(header::CONTENT_TYPE, content_type)
        .header(header::CONTENT_LENGTH, effective_take)
        .body(body)
        .map_err(|e| {
            ApiError::File(crate::file_mgr::io_err(
                resolved.display(),
                std::io::Error::other(e),
            ))
        })?;
    Ok(resp)
}

/// Always `202 Accepted`: rename + tombstone are durable, but the staged drain runs in the background (queued, not done).
#[derive(Serialize)]
struct AsyncDeleteResp {
    job_id: String,
}

/// A missing ASSET surfaces as `FileError::Io { NotFound }` (kind `Internal`), not
/// a missing WORKSPACE (`FileError::NotFound`); blame the asset so the 404 matches `GET`.
fn classify_asset_existence_error(
    id: &WorkspaceId,
    asset_path: &AssetPath,
    err: crate::file_mgr::FsError,
) -> ApiError {
    use crate::common::error::{Categorized, ErrorKind};
    if matches!(err.kind(), ErrorKind::Internal) && first_io_error_is_not_found(&err) {
        return ApiError::NotFound(format!("asset {} in workspace {id}", asset_path.as_str(),));
    }
    classify_workspace_existence_error(id, err)
}

async fn delete_asset(
    State(files): State<Arc<dyn FsService>>,
    Path((id, raw_path)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let asset_path = parse_asset_path(&raw_path)?;
    let asset_path_for_err = asset_path.clone();
    let job_id = task::spawn_blocking(move || files.start_workspace_asset_delete(&id, &asset_path))
        .await?
        .map_err(|e| classify_asset_existence_error(&id, &asset_path_for_err, e))?;
    Ok((
        axum::http::StatusCode::ACCEPTED,
        Json(AsyncDeleteResp {
            job_id: job_id.to_string(),
        }),
    )
        .into_response())
}

/// Upload raw body bytes (no multipart) via atomic rename-into-tree + revision
/// bump. Size is bounded twice: the route's `DefaultBodyLimit::max(256 MiB)` (413
/// on `Content-Length`, else mid-stream error -> 500) and the handler's running
/// check against operator-tunable `max_upload_bytes()` (`PayloadTooLarge` -> 400,
/// expected <= the hard ceiling).
async fn upload_asset(
    State(files): State<Arc<dyn FsService>>,
    Path((id, raw_path)): Path<(String, String)>,
    body: Body,
) -> Result<Json<DatasetUploadReceipt>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    let asset_path = parse_asset_path(&raw_path)?;

    // Existence-check before the tempfile, else a phantom workspace orphans `.tmp/`.
    crate::api::routes::workspace::summary_or_404(&files, id).await?;

    // Acquire before consuming any body bytes; drops at fn exit.
    let _permit = files.acquire_upload_permit()?;

    let tmp_dir = files.workspace_tmpdir(&id);
    tokio::fs::create_dir_all(&tmp_dir)
        .await
        .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_dir.display(), e)))?;
    let tmp_dir_for_spawn = tmp_dir.clone();
    let tmp_file =
        task::spawn_blocking(move || tempfile::NamedTempFile::new_in(&tmp_dir_for_spawn))
            .await?
            .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_dir.display(), e)))?;

    // Reject once cumulative bytes cross the cap; tempfile drops on Err = no partial commit.
    let max_upload_bytes = files.max_upload_bytes();
    let tmp_reopened = tmp_file
        .reopen()
        .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e)))?;
    let mut writer = tokio::fs::File::from_std(tmp_reopened);
    use futures_util::TryStreamExt as _;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let stream = body.into_data_stream().map_err(std::io::Error::other);
    let mut reader = tokio_util::io::StreamReader::new(stream);
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    let mut hasher = Sha256::new();
    loop {
        let n = reader
            .read(&mut buf)
            .await
            .map_err(|e| ApiError::File(crate::file_mgr::io_err("<upload-stream>", e)))?;
        if n == 0 {
            break;
        }
        total = total.saturating_add(n as u64);
        if total > max_upload_bytes {
            return Err(ApiError::Fs(crate::file_mgr::FsError::new(
                crate::file_mgr::FileError::PayloadTooLarge {
                    observed: total,
                    max: max_upload_bytes,
                },
            )));
        }
        hasher.update(&buf[..n]);
        writer
            .write_all(&buf[..n])
            .await
            .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e)))?;
    }
    writer
        .flush()
        .await
        .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e)))?;
    writer
        .sync_all()
        .await
        .map_err(|e| ApiError::File(crate::file_mgr::io_err(tmp_file.path().display(), e)))?;
    drop(writer);

    let digest = hex_lowercase(&hasher.finalize());

    let r = task::spawn_blocking(move || {
        files.upload_workspace_file(&id, &asset_path, tmp_file.path(), &digest, total)
    })
    .await??;
    Ok(Json(r))
}

/// Hard ceiling enforced by the upload route's axum `DefaultBodyLimit`; equals
/// default `max_upload_bytes()` (256 MiB). `max_upload_bytes` ABOVE this silently
/// truncates uploads, hence `boot_warn_if_misconfigured`.
pub const UPLOAD_BODY_HARD_CEILING_BYTES: usize = 256 * 1024 * 1024;

/// Warn at boot if `max_upload_bytes()` exceeds [`UPLOAD_BODY_HARD_CEILING_BYTES`] (else the middleware truncates silently).
pub(crate) fn boot_warn_if_misconfigured(files: &dyn crate::file_mgr::FsService) {
    let configured = files.max_upload_bytes();
    if configured > UPLOAD_BODY_HARD_CEILING_BYTES as u64 {
        tracing::warn!(
            target: "api",
            configured_max_upload_bytes = configured,
            hard_ceiling = UPLOAD_BODY_HARD_CEILING_BYTES as u64,
            "file.max_upload_bytes exceeds the api router's hard ceiling; \
             axum's DefaultBodyLimit will short-circuit uploads at the lower value -- \
             reduce `file.max_upload_bytes` in config or rebuild the daemon \
             with a higher `UPLOAD_BODY_HARD_CEILING_BYTES`",
        );
    }
}

/// Rename body; `to_name` is the destination class dir, server-validated as a single `AssetPath` component.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RenameCategoryReq {
    to_name: String,
}

/// Synchronously rename a dataset category dir (one atomic `rename(2)`, no bytes
/// moved). 200 + new `workspace_revision_id`; 404 missing source, 409 target
/// collision or active Train job, 400 invalid name. The single `datasets/<name>`
/// component arrives as one segment (no wildcard).
async fn rename_category(
    State(files): State<Arc<dyn FsService>>,
    Path((id, from_name)): Path<(String, String)>,
    ApiJson(req): ApiJson<RenameCategoryReq>,
) -> Result<Json<RenameCategoryReceipt>, ApiError> {
    let id = WorkspaceId::parse(&id)?;
    // 400s bad bytes / leading `.`/`-` / a body-smuggled `/`.
    let from = parse_asset_path(&format!("{DATASETS_DIR_NAME}/{from_name}"))?;
    let to = parse_asset_path(&format!("{DATASETS_DIR_NAME}/{}", req.to_name))?;
    // 404 a phantom workspace before the blocking hop (matches upload).
    crate::api::routes::workspace::summary_or_404(&files, id).await?;
    let from_for_err = from.clone();
    let receipt = task::spawn_blocking(move || files.rename_dataset_category(&id, &from, &to))
        .await?
        .map_err(|e| classify_asset_existence_error(&id, &from_for_err, e))?;
    Ok(Json(receipt))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/workspaces/{id}/assets", get(list_assets))
        .route(
            "/workspaces/{id}/assets/{*path}",
            get(get_asset)
                .put(upload_asset)
                .delete(delete_asset)
                // Applies to PUT only; GET/DELETE consume no body.
                .layer(DefaultBodyLimit::max(UPLOAD_BODY_HARD_CEILING_BYTES)),
        )
        .route(
            "/workspaces/{id}/datasets/{name}/rename",
            // Scope the JSON cap to THIS MethodRouter; a router-wide layer would
            // clobber the upload route's 256 MiB limit.
            post(rename_category).layer(DefaultBodyLimit::max(CONTROL_JSON_BODY_LIMIT)),
        )
}
