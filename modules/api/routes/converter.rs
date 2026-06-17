//! `POST /workspaces/{id}/convert` producer: validates, resolves inputs on the
//! blocking pool, takes a `Workspace` job-reference (excludes `WorkspaceDelete`;
//! uploads/file-deletes overlap freely), then spawns under `max_convert_jobs`.

use std::path::PathBuf;
use std::sync::Arc;

use crate::common::asset_path::AssetPath;
use crate::common::ids::{HeadId, WorkspaceId};
use crate::common::workspace::{JobReference, JobType, WorkspaceRevision};
use crate::file_mgr::{
    ConvertRequest, ConverterPath, FsService, JobRegistry, validate_convert_request,
};
use axum::Router;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::response::Json;
use axum::routing::post;
use serde::Serialize;
use tokio::task;

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::{ApiJson, CONTROL_JSON_BODY_LIMIT};

/// SHA-256 hex; `pub(crate)` for streaming-upload tests. Test-only today.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    let d = sha2::Sha256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Debug, Serialize)]
struct ConvertStartResp {
    /// Pre-allocated head id; index entry committed only on successful publish.
    head_id: String,
    job_id: String,
}

/// Resolve a converter-rooted input to its on-disk path, promoting
/// ENOENT to 404 and rejecting non-regular files (dir/symlink) as 400.
fn resolve_converter_input(
    files: &Arc<dyn FsService>,
    workspace_id: &WorkspaceId,
    path: &AssetPath,
) -> Result<PathBuf, ApiError> {
    let (resolved, md) = files.open_workspace_file(workspace_id, path).map_err(|e| {
        use std::error::Error as _;
        // ENOENT/non-regular-file are operator-input faults forced to 4xx; the
        // catch-all From<FsError> (carrying the inner FileError::Io's Internal
        // kind) would otherwise make them 500.
        let io_kind = e
            .source()
            .and_then(|s| s.downcast_ref::<crate::file_mgr::FileError>())
            .and_then(|fe| match fe {
                crate::file_mgr::FileError::Io { source, .. } => Some(source.kind()),
                _ => None,
            });
        match io_kind {
            Some(std::io::ErrorKind::NotFound) => ApiError::NotFound(format!(
                "convert input not found: /{}",
                strip_converter_prefix(path)
            )),
            Some(std::io::ErrorKind::InvalidInput) => ApiError::Bad(format!(
                "convert input /{} is not a regular file",
                strip_converter_prefix(path)
            )),
            _ => ApiError::from(e),
        }
    })?;
    // Unreachable today (open_workspace_file filters non-regular files); guards
    // against a future weakening of that invariant.
    if !md.is_file() {
        return Err(ApiError::Bad(format!(
            "convert input /{} is not a regular file",
            strip_converter_prefix(path),
        )));
    }
    Ok(resolved)
}

fn strip_converter_prefix(path: &AssetPath) -> &str {
    path.as_str()
        .strip_prefix("converters/")
        .expect("converter input path starts with converters/")
}

/// Derive each TFJS shard's [`AssetPath`] from the `model.json` path + parsed
/// `weightsManifest[].paths` (sibling-relative per the TFJS contract).
///
/// Traversal-safe: parent is AssetPath-clean, shards passed `validate_shard_path`
/// (no absolute/`..`/`\`/NUL), and the result is re-parsed via [`AssetPath::parse`]
/// so any breach surfaces as 400.
fn derive_tfjs_shard_asset_paths(
    model_json_path: &ConverterPath,
    relative_shards: &[String],
) -> Result<Vec<AssetPath>, ApiError> {
    let manifest_str = model_json_path.workspace_path().as_str();
    // ConverterPath guarantees >= 2 components, so the None arm is unreachable.
    let parent = manifest_str
        .rsplit_once('/')
        .map(|(p, _)| p)
        .ok_or_else(|| {
            ApiError::Bad(format!(
                "TFJS model_json path has no parent directory: {manifest_str}"
            ))
        })?;
    let mut out = Vec::with_capacity(relative_shards.len());
    for shard in relative_shards {
        let derived = format!("{parent}/{shard}");
        let asset = AssetPath::parse(&derived).map_err(|e| {
            ApiError::Bad(format!(
                "TFJS derived shard path is malformed ({derived}): {e}"
            ))
        })?;
        out.push(asset);
    }
    Ok(out)
}

/// Derive the alpkg `.mpk` [`AssetPath`] via the `<parent>/<head_id>.mpk`
/// convention (sibling of its manifest).
///
/// Traversal-safe: parent is AssetPath-clean, `head_id` Display is a UUID
/// (`[0-9a-f-]`, no leading `.`), and the result is re-parsed via
/// [`AssetPath::parse`] so any breach surfaces as 400.
fn derive_alpkg_mpk_asset_path(
    manifest_path: &ConverterPath,
    head_id: HeadId,
) -> Result<AssetPath, ApiError> {
    let manifest_str = manifest_path.workspace_path().as_str();
    // ConverterPath guarantees >= 2 components, so the None arm is unreachable.
    let parent = manifest_str
        .rsplit_once('/')
        .map(|(p, _)| p)
        .ok_or_else(|| {
            ApiError::Bad(format!(
                "alpkg manifest path has no parent directory: {manifest_str}"
            ))
        })?;
    let derived = format!("{parent}/{head_id}.mpk");
    AssetPath::parse(&derived).map_err(|e| {
        ApiError::Bad(format!(
            "alpkg derived weights path is malformed ({derived}): {e}"
        ))
    })
}

/// Alpkg manifest read cap; matches the worker's re-read cap so a manifest
/// accepted here isn't later rejected by the worker.
const ALPKG_MANIFEST_READ_CAP: u64 = 8 * 1024 * 1024;

/// TFJS `model.json` read cap: the downstream parser's 1 MiB limit fires only
/// after reading, so without this gate the route slurps up to the 256 MiB body
/// ceiling. Sized above that cap so oversize yields the parser's actionable
/// `LimitExceeded`, not this bare gate.
const TFJS_MODEL_JSON_READ_CAP: u64 = 8 * 1024 * 1024;

/// Capped admission read. Returns `io::Error::Other` ("exceeds N-byte cap") on
/// the `metadata.len()` precheck or the post-read `+1` torn-write recheck; stays
/// out of FileError so the admission path doesn't drag in the 500 class.
fn read_capped_for_api(path: &std::path::Path, cap: u64) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let f = std::fs::File::open(path)?;
    let stat = f.metadata()?;
    // Cap-exceeded messages OMIT the absolute path: as unsanitized 400 bodies
    // (only 5xx is scrubbed) they would leak the FS root + workspace UUID; caller
    // appends the wire path instead.
    if stat.len() > cap {
        return Err(std::io::Error::other(format!(
            "file exceeds {cap}-byte cap (observed {} bytes)",
            stat.len(),
        )));
    }
    let cap_for_hint = std::cmp::min(stat.len(), cap);
    let mut buf = Vec::with_capacity(cap_for_hint as usize);
    let mut limited = f.take(cap + 1);
    limited.read_to_end(&mut buf)?;
    if buf.len() as u64 > cap {
        return Err(std::io::Error::other(format!(
            "file exceeds {cap}-byte cap (observed {} bytes)",
            buf.len(),
        )));
    }
    Ok(buf)
}

/// Extract the [`HeadId`] from the alpkg manifest; the synchronous response
/// echoes it, so it must resolve before the worker spawns (IO/parse failure is
/// 400). Capped to [`ALPKG_MANIFEST_READ_CAP`].
fn read_alpkg_head_id_from_manifest_file(
    manifest_path: &std::path::Path,
    wire: &str,
) -> Result<HeadId, ApiError> {
    let manifest_bytes = read_capped_or_bad(
        manifest_path,
        ALPKG_MANIFEST_READ_CAP,
        "alpkg manifest read",
        wire,
    )?;
    let manifest: crate::common::workspace::HeadManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| ApiError::Bad(format!("alpkg manifest JSON parse failed: /{wire}: {e}")))?;
    Ok(manifest.head_id)
}

/// `read_capped_for_api` with a per-call prefix. Cap-exceeded (the only
/// `ErrorKind::Other`) -> 400 with the wire path; other IO faults -> 500 via
/// `ApiError::File` whose resolved path the 5xx sanitizer scrubs.
fn read_capped_or_bad(
    path: &std::path::Path,
    cap: u64,
    what: &'static str,
    wire: &str,
) -> Result<Vec<u8>, ApiError> {
    read_capped_for_api(path, cap).map_err(|e| match e.kind() {
        std::io::ErrorKind::Other => ApiError::Bad(format!("{what} failed: /{wire}: {e}")),
        _ => ApiError::File(crate::file_mgr::io_err(path.display(), e)),
    })
}

async fn start_convert(
    State(files): State<Arc<dyn FsService>>,
    State(jobs): State<Arc<JobRegistry>>,
    Path(id): Path<String>,
    ApiJson(req): ApiJson<ConvertRequest>,
) -> Result<Json<ConvertStartResp>, ApiError> {
    let workspace_id = WorkspaceId::parse(&id)?;
    // Traversal already rejected at deserialize via ConverterPath; here a no-op
    // for TFJS, enforces only `.json` extension hygiene for Alpkg.
    validate_convert_request(&req).map_err(|e| ApiError::Bad(e.to_string()))?;

    // Existence + revision snapshot + per-file resolution (and Alpkg's manifest
    // head_id read) run on the blocking pool to keep the runtime free under eMMC
    // pressure.
    let files_for_resolve = files.clone();
    let req_for_resolve = req.clone();
    let (workspace_revision, job_kind, head_id): (
        WorkspaceRevision,
        crate::converter::ConvertJobKind,
        HeadId,
    ) = task::spawn_blocking(move || -> Result<_, ApiError> {
        // Missing workspace -> 404 (else summary's FsError::Internal is a raw 500).
        let summary = files_for_resolve.summary(&workspace_id).map_err(|e| {
            crate::api::routes::workspace::classify_workspace_existence_error(&workspace_id, e)
        })?;
        let rev = summary.core.workspace_revision.clone();
        match &req_for_resolve {
            ConvertRequest::Tfjs(params) => {
                let model_json = resolve_converter_input(
                    &files_for_resolve,
                    &workspace_id,
                    params.model_json_path.workspace_path(),
                )?;
                let labels = resolve_converter_input(
                    &files_for_resolve,
                    &workspace_id,
                    params.labels_path.workspace_path(),
                )?;
                // Parsing the shard list here makes a bad manifest a synchronous
                // 400, not a deferred SSE end.
                let model_json_bytes = read_capped_or_bad(
                    &model_json,
                    TFJS_MODEL_JSON_READ_CAP,
                    "TFJS model.json read",
                    &params.model_json_path.wire_form(),
                )?;
                let parsed_manifest = crate::converter::parse_tfjs_manifest_with_limits(
                    &model_json_bytes,
                    &crate::converter::ConvertLimits::default(),
                )
                .map_err(|e| {
                    ApiError::Bad(format!(
                        "TFJS model.json parse failed: /{}: {e}",
                        params.model_json_path.wire_form()
                    ))
                })?;
                let shard_asset_paths = derive_tfjs_shard_asset_paths(
                    &params.model_json_path,
                    &parsed_manifest.shards,
                )?;
                let mut shards: Vec<PathBuf> = Vec::with_capacity(shard_asset_paths.len());
                for shard in &shard_asset_paths {
                    shards.push(resolve_converter_input(
                        &files_for_resolve,
                        &workspace_id,
                        shard,
                    )?);
                }
                let kind =
                    crate::converter::ConvertJobKind::Tfjs(crate::converter::ConvertJobTfjs {
                        model_json_path: model_json,
                        shard_paths: shards,
                        // Snapshot lets the worker detect a mid-job re-upload that
                        // reorders `paths` without changing cardinality.
                        shard_names: parsed_manifest.shards.clone(),
                        labels_path: labels,
                        labels_format: params.labels_format,
                    });
                Ok((rev, kind, HeadId::new()))
            }
            ConvertRequest::Alpkg(params) => {
                // Operator names only the `.json` manifest; read its `head_id`,
                // then derive the sibling `<parent>/<head_id>.mpk`.
                let manifest_path = resolve_converter_input(
                    &files_for_resolve,
                    &workspace_id,
                    params.manifest_path.workspace_path(),
                )?;
                let head_id = read_alpkg_head_id_from_manifest_file(
                    &manifest_path,
                    &params.manifest_path.wire_form(),
                )?;
                // Resolving the `.mpk` enforces exists + regular-file (manifest's
                // 404/400 mapping), so a missing/misnamed one errors.
                let mpk_asset_path = derive_alpkg_mpk_asset_path(&params.manifest_path, head_id)?;
                let mpk_path =
                    resolve_converter_input(&files_for_resolve, &workspace_id, &mpk_asset_path)?;
                let kind =
                    crate::converter::ConvertJobKind::Alpkg(crate::converter::ConvertJobAlpkg {
                        mpk_path,
                        manifest_path,
                    });
                Ok((rev, kind, head_id))
            }
        }
    })
    .await??;

    // Registry `try_acquire` FIRST so the diagnostic `RegistryConflict` -> 409
    // (e.g. WorkspaceDelete in progress) wins over the generic semaphore-Busy 409
    // when both reject; both releases are RAII so order is safe. The registry job
    // id doubles as the JSONL log filename, correlating `GET /jobs`.
    let job_handle = jobs
        .try_acquire(
            JobType::Convert,
            vec![JobReference::Workspace { workspace_id }],
            None,
        )
        .map_err(|c| ApiError::File(crate::file_mgr::FileError::from(c)))?;

    // Single-tenant (`max_convert_jobs = 1`); a concurrent request gets
    // `ConvertError::Busy` -> 409, `job_handle` drops via RAII on error.
    let convert_permit = crate::converter::acquire_convert_permit()?;
    let job_id = job_handle.job_id();

    let files_for_worker = files.clone();
    let job = crate::converter::ConvertJob {
        job_id,
        workspace_id,
        head_id,
        workspace_revision,
        kind: job_kind,
    };
    // TODO(convert-drain): the JoinHandle is dropped, so shutdown leans on the
    // process-exit budget to bound the worker. Crash-safe regardless: a killed
    // convert leaves only `.tmp/` residue + a transient `Convert` JobReference,
    // both swept by boot recovery. Proper fix mirrors training's `cancel_all_for_shutdown`.
    tokio::task::spawn_blocking(move || {
        // Permit moves in so the slot stays held until the job terminates.
        // `JobHandle` owns the registry transition + JSONL trace inside
        // `run_convert_job`; the route only logs a terminal breadcrumb.
        let _convert_permit = convert_permit;
        if let Err(e) = crate::converter::run_convert_job(files_for_worker, job, Some(job_handle)) {
            tracing::warn!(
                target: "converter",
                job_id = %job_id,
                workspace_id = %workspace_id,
                err = %e,
                "convert job failed",
            );
        }
        // Return convert blobs (tens of MiB at high class counts) eagerly instead
        // of leaving them resident under the allocator's lazy purge; safe inline
        // on the blocking pool.
        crate::allocator::release_to_os();
    });

    Ok(Json(ConvertStartResp {
        head_id: head_id.to_string(),
        job_id: job_id.to_string(),
    }))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/workspaces/{id}/convert", post(start_convert))
        .layer(DefaultBodyLimit::max(CONTROL_JSON_BODY_LIMIT))
}
