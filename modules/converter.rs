//! Extracts an operator-uploaded TFJS Layers-Model head (`Linear` = Keras
//! `Dense` kernel + bias) into Burn's `[in_dim, n_classes]` orientation,
//! writing `head.mpk` + `labels.txt` + `metadata.json`. TFJS is the only
//! source format. Only the head is replaced; the pre-built
//! `backbone.rknn` is trusted to match (a mismatch yields garbage
//! classification, not an error).
//!
//! Manifest tensor offsets are implicit: each entry starts where the
//! previous ended, shards concatenated in manifest order. Keras Dense
//! `[in_dim, out_dim]` row-major matches Burn's `Linear` (no transpose).
//! Head ID'd by `NewHeadDense/kernel`+`/bias` names, else the unique 2-D
//! `[BACKBONE_FEATURE_DIM, N]` paired with the unique 1-D `[N]`;
//! non-unique fails with a manifest-listing diagnostic.

#![warn(missing_debug_implementations)]

pub(crate) mod pipeline;
pub(crate) mod sink;
pub(crate) mod source;

pub use pipeline::Pipeline;
pub use sink::{ArtifactSink, MpkSink};
pub use source::{LoadedSource, SourceModel, TfjsSource, TfjsSourceLimited};

use std::path::{Path, PathBuf};

use crate::common::dims::BACKBONE_FEATURE_DIM;
use crate::common::error::{Categorized, Severity};
use crate::common::hex::hex_lowercase;
use crate::common::ids::HeadId;
use crate::common::workspace::{ConverterType, MAX_LABEL_BYTES, WorkspaceRevision};
use crate::file_mgr::{
    CONVERTER_LOGS_DIR_NAME, FsService, JobHandle, JsonlEventLog, RegistryJobResult,
};
use crate::model::{self, Head};
use burn::backend::NdArray;
use burn::module::{Module, Param};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
use burn::tensor::TensorData;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use time::OffsetDateTime;

type B = NdArray<f32>;

/// Daemon-wide single-slot cap: concurrent runs multiply per-run peaks
/// (one shard + in-memory `head.mpk` + `max_kernel_bytes` Param copy) and
/// risk OOM.
static CONVERT_SEMAPHORE: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
    std::sync::OnceLock::new();

/// Returns [`ConvertError::Busy`] (HTTP 409) instead of blocking so
/// operators get a progress signal.
pub fn acquire_convert_permit() -> Result<tokio::sync::OwnedSemaphorePermit, ConvertError> {
    CONVERT_SEMAPHORE
        .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(1)))
        .clone()
        .try_acquire_owned()
        .map_err(|_| ConvertError::Busy)
}

/// Resource caps rejecting an operator-supplied manifest BEFORE the
/// per-tensor allocation its declared shapes would justify (shard data is
/// not yet proven). `max_n_classes` is sized so `max_kernel_bytes /
/// (BACKBONE_FEATURE_DIM * 4)` lands on the same boundary; `max_bias_bytes`
/// is loose (n_classes already bounds it).
#[derive(Clone, Copy, Debug)]
pub struct ConvertLimits {
    pub max_n_classes: usize,
    pub max_kernel_bytes: usize,
    pub max_bias_bytes: usize,
    pub max_manifest_bytes: usize,
    pub max_shards: usize,
}

impl Default for ConvertLimits {
    fn default() -> Self {
        Self {
            // 80 MiB / (BACKBONE_FEATURE_DIM * 4) ~= 10_485, rounded down.
            max_n_classes: 10_000,
            max_kernel_bytes: 80 * 1024 * 1024,
            max_bias_bytes: 1024 * 1024,
            max_manifest_bytes: 1024 * 1024,
            max_shards: 64,
        }
    }
}

/// Source format the operator uploaded. `#[non_exhaustive]`: a snapshot,
/// not a closed set.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case", tag = "kind")]
#[non_exhaustive]
pub enum SourceKind {
    Tfjs,
}

/// Failures from the TFJS-to-Burn conversion pipeline; the
/// [`Categorized`] impl maps them to HTTP statuses.
#[derive(Debug, Error)]
pub enum ConvertError {
    #[error("read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("write {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("burn record: {0}")]
    Record(String),
    #[error("burn tensor: {0}")]
    Tensor(String),
    #[error("metadata serialize: {0}")]
    MetadataSerialize(#[from] serde_json::Error),
    #[error("labels file: {0}")]
    Labels(String),
    #[error("tfjs parse {what}: {msg}")]
    TfjsParse { what: &'static str, msg: String },
    #[error("tfjs head locator: {0}")]
    TfjsLocator(String),
    #[error("tfjs weights.bin too short: have {have} bytes, manifest needs {need}")]
    TfjsShortRead { have: usize, need: usize },
    #[error("tfjs weight `{name}` has unsupported dtype `{dtype}`, expected float32")]
    TfjsDtype { name: String, dtype: String },
    #[error("tfjs unsafe shard path `{0}`: must be a relative path with no parent traversal")]
    TfjsUnsafePath(String),
    #[error(
        "tfjs manifest declares dimensions that overflow usize on weight `{name}` (shape {shape:?})"
    )]
    TfjsShapeOverflow { name: String, shape: Vec<usize> },
    #[error("tfjs blob length mismatch: have {have} bytes, manifest declares {declared}")]
    TfjsBlobLength { have: usize, declared: usize },
    /// `ConvertLimits` cap tripped before allocation. `what` is a stable
    /// cap identifier matching a `ConvertLimits` field.
    #[error("tfjs limit `{what}` exceeded: {value} > max {max}")]
    LimitExceeded {
        what: &'static str,
        value: u64,
        max: u64,
    },
    /// Zero dimension; rejected because Burn's `LinearConfig::init` panics
    /// on zero out-features.
    #[error("tfjs zero dimension on weight `{name}` (shape {shape:?})")]
    TfjsZeroDimension { name: String, shape: Vec<usize> },
    /// Decoded fp32 weight non-finite (NaN, +/-Inf); caught here, not just
    /// on the inference cold path. `index` locates it in the source f32 seq.
    #[error("tfjs non-finite weight in `{tensor}` at index {index}: {value}")]
    NonFiniteWeight {
        tensor: String,
        index: usize,
        value: f32,
    },
    /// `n_classes` outside `1..=max_n_classes`. Raised here so operators get
    /// a `UserInput` (400), not a daemon-internal `Head::try_new`.
    #[error("n_classes {got} out of range, must be in 1..={max}")]
    BadClassCount { got: usize, max: usize },
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),
    /// Another converter job is in flight (concurrent extracts risk OOM).
    #[error("converter job already running")]
    Busy,
    /// `.alpkg` import: IO failure re-reading the manifest mid-job (route's
    /// startup read already verified it exists+parses).
    #[error("alpkg manifest read {path}: {source}")]
    AlpkgManifestRead {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// `.alpkg` import: manifest parsed at start-time but failed the
    /// worker's full structural validation; `reason` names the field.
    #[error("alpkg manifest schema: {reason}")]
    AlpkgManifestSchema { reason: String },
    #[error("alpkg .mpk size mismatch: manifest declares {expected} bytes, file holds {observed}")]
    AlpkgSizeMismatch { expected: u64, observed: u64 },
    #[error("alpkg .mpk hash mismatch: manifest declares {expected}, file hashes to {observed}")]
    AlpkgHashMismatch { expected: String, observed: String },
    /// `.alpkg` refused: same `head_id` exists with a different `sha256`;
    /// the rotation primitive won't overwrite (409, `head_id_collision`
    /// discriminator).
    #[error("{0}")]
    HeadIdCollision(#[source] crate::file_mgr::FileError),
}

pub(crate) fn convert_read_err(
    path: impl std::fmt::Display,
    source: std::io::Error,
) -> ConvertError {
    ConvertError::Read {
        path: path.to_string(),
        source,
    }
}

/// `ConvertError::Write`; some call sites wrap a `FileError` via
/// `io::Error::other(format!("{e}"))`.
pub(crate) fn convert_write_err(
    path: impl std::fmt::Display,
    source: std::io::Error,
) -> ConvertError {
    ConvertError::Write {
        path: path.to_string(),
        source,
    }
}

fn alpkg_schema_err(reason: impl Into<String>) -> ConvertError {
    ConvertError::AlpkgManifestSchema {
        reason: reason.into(),
    }
}

/// Read a manifest with a hard size cap: `metadata().len()` precheck
/// rejects oversized payloads before allocating; `take(cap + 1)` covers a
/// torn write where the file grew between stat and read.
fn read_capped_for_convert(path: &Path, cap: u64) -> Result<Vec<u8>, ConvertError> {
    use std::io::Read;
    let manifest_read_err = |source| ConvertError::AlpkgManifestRead {
        path: path.display().to_string(),
        source,
    };
    let f = std::fs::File::open(path).map_err(manifest_read_err)?;
    let stat = f.metadata().map_err(manifest_read_err)?;
    if stat.len() > cap {
        // Stable `manifest.json` token, NOT `path.display()`: this UserInput
        // error lifts unsanitized into the job's SSE/JSONL stream and the
        // path is absolute. IO-error arms keep the path (AlpkgManifestRead =
        // Internal, scrubbed).
        return Err(alpkg_schema_err(format!(
            "manifest.json: exceeds {cap}-byte cap (observed {} bytes)",
            stat.len()
        )));
    }
    let cap_for_hint = std::cmp::min(stat.len(), cap);
    let mut buf = Vec::with_capacity(cap_for_hint as usize);
    let mut limited = f.take(cap + 1);
    limited.read_to_end(&mut buf).map_err(manifest_read_err)?;
    if buf.len() as u64 > cap {
        return Err(alpkg_schema_err(format!(
            "manifest.json: exceeds {cap}-byte cap (read {} bytes after torn-write)",
            buf.len()
        )));
    }
    Ok(buf)
}

/// Cap on an operator-supplied labels source (`labels.txt` /
/// `metadata.json`); else an operator could slurp a 256 MiB upload onto the
/// worker (stacked on live shard state) before any structural check.
const LABELS_READ_CAP: u64 = 8 * 1024 * 1024;

/// Read a labels source with a hard size cap. A breach maps to
/// [`ConvertError::Labels`] with a STABLE `token` (never the absolute
/// staging path, which would lift unsanitized into the job SSE/JSONL
/// stream); IO failures stay in `Read` (Internal, scrubbed).
fn read_capped_labels(path: &Path, token: &str) -> Result<Vec<u8>, ConvertError> {
    use std::io::Read;
    let cap = LABELS_READ_CAP;
    let f = std::fs::File::open(path).map_err(|e| convert_read_err(path.display(), e))?;
    let stat = f
        .metadata()
        .map_err(|e| convert_read_err(path.display(), e))?;
    if stat.len() > cap {
        return Err(ConvertError::Labels(format!(
            "{token}: exceeds {cap}-byte cap (observed {} bytes)",
            stat.len()
        )));
    }
    let mut buf = Vec::with_capacity(std::cmp::min(stat.len(), cap) as usize);
    f.take(cap + 1)
        .read_to_end(&mut buf)
        .map_err(|e| convert_read_err(path.display(), e))?;
    if buf.len() as u64 > cap {
        return Err(ConvertError::Labels(format!(
            "{token}: exceeds {cap}-byte cap (read {} bytes after torn-write)",
            buf.len()
        )));
    }
    Ok(buf)
}

impl crate::common::error::Categorized for ConvertError {
    /// Exhaustive `match` so a new variant forces a classification.
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            // Source model malformed / unsupported; operator re-exports.
            ConvertError::Labels(_)
            | ConvertError::TfjsParse { .. }
            | ConvertError::TfjsLocator(_)
            | ConvertError::TfjsShortRead { .. }
            | ConvertError::TfjsDtype { .. }
            | ConvertError::TfjsUnsafePath(_)
            | ConvertError::TfjsShapeOverflow { .. }
            | ConvertError::TfjsBlobLength { .. }
            | ConvertError::LimitExceeded { .. }
            | ConvertError::TfjsZeroDimension { .. }
            | ConvertError::NonFiniteWeight { .. }
            | ConvertError::BadClassCount { .. }
            // `.alpkg` operator-input: manifest fails structural checks or
            // `.mpk` bytes mismatch declared hash/size; recovery is re-export.
            | ConvertError::AlpkgManifestSchema { .. }
            | ConvertError::AlpkgSizeMismatch { .. }
            | ConvertError::AlpkgHashMismatch { .. } => UserInput,

            ConvertError::NotImplemented(_) => NotImplemented,

            ConvertError::Busy => Conflict,

            ConvertError::HeadIdCollision(_) => Conflict,

            // Daemon-internal, not the uploader's fault. `AlpkgManifestRead`
            // lives here: a file vanishing between the route's start-time
            // parse and the worker re-read is transient FS (500+retry), not a
            // 400 that misleads re-upload.
            ConvertError::Read { .. }
            | ConvertError::Write { .. }
            | ConvertError::Record(_)
            | ConvertError::Tensor(_)
            | ConvertError::MetadataSerialize(_)
            | ConvertError::AlpkgManifestRead { .. } => Internal,
        }
    }
}

/// Extracted head weights; kernel in Burn's `[in_dim, n_classes]` layout.
#[derive(Clone, Debug)]
pub struct HeadWeights {
    pub kernel: Vec<f32>,
    pub bias: Vec<f32>,
    pub n_classes: usize,
    pub in_dim: usize,
}

/// On-disk artifacts produced by [`write_head_artifacts`].
#[derive(Clone, Debug, Serialize)]
pub struct HeadArtifacts {
    pub head_mpk: PathBuf,
    pub labels_txt: PathBuf,
    pub metadata_json: PathBuf,
    pub head_id: HeadId,
    pub n_classes: usize,
    /// SHA-256 of the source model bytes (lowercase hex).
    pub source_sha256: String,
}

/// Per-conversion metadata persisted alongside the head; `schema_version`
/// bumps on breaking change.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversionMetadata {
    pub schema_version: u32,
    pub source_kind: SourceKind,
    pub source_sha256: String,
    pub n_classes: usize,
    pub labels: Vec<String>,
    pub created_at: String,
    pub head_id: HeadId,
}

/// Extract the head from a TFJS Layers-Model directory (`model.json` +
/// shards from `weightsManifest[*].paths`, resolved relative to it).
/// Weights in Burn's `[in_dim, n_classes]` orientation.
pub fn extract_head_from_tfjs_dir(tfjs_dir: &Path) -> Result<HeadWeights, ConvertError> {
    let limits = ConvertLimits::default();
    let model_json = tfjs_dir.join("model.json");
    let json_bytes =
        std::fs::read(&model_json).map_err(|e| convert_read_err(model_json.display(), e))?;
    let manifest = parse_tfjs_manifest_with_limits(&json_bytes, &limits)?;

    // No SHA contract on this entry point: hasher is `None`.
    let (k_entry, b_entry) = pick_tfjs_head_entries(&manifest)?;
    let (kernel_bytes, bias_bytes) =
        source::read_head_bytes_streaming(tfjs_dir, &manifest, k_entry, b_entry, None)?;
    head_weights_from_head_byte_ranges(&manifest, k_entry, b_entry, &kernel_bytes, &bias_bytes)
}

/// Single-shard convenience: parse `model_json`, read the head from
/// `weights_bin`. Errors if the manifest references >1 shard.
pub fn extract_head_from_tfjs(
    model_json: &Path,
    weights_bin: &Path,
) -> Result<HeadWeights, ConvertError> {
    let limits = ConvertLimits::default();
    let json_bytes =
        std::fs::read(model_json).map_err(|e| convert_read_err(model_json.display(), e))?;
    let manifest = parse_tfjs_manifest_with_limits(&json_bytes, &limits)?;
    if manifest.shards.len() != 1 {
        return Err(ConvertError::TfjsParse {
            what: "weightsManifest",
            msg: format!(
                "expected exactly 1 shard path, got {}: {:?}",
                manifest.shards.len(),
                manifest.shards
            ),
        });
    }
    let weights_blob =
        std::fs::read(weights_bin).map_err(|e| convert_read_err(weights_bin.display(), e))?;
    extract_head_from_tfjs_buffers(&manifest, &weights_blob)
}

/// Extract + persist a TFJS upload in one call. `labels` is an argument
/// (not auto-discovered) so callers normalize/validate. `source_sha256` =
/// SHA-256 of `model.json` bytes ++ each shard's bytes in manifest order
/// (deterministic for a given layout).
pub fn convert_tfjs(
    tfjs_dir: &Path,
    labels: &[String],
    dst_dir: &Path,
    fs: &dyn FsService,
) -> Result<HeadArtifacts, ConvertError> {
    let limits = ConvertLimits::default();
    let model_json = tfjs_dir.join("model.json");
    let json_bytes =
        std::fs::read(&model_json).map_err(|e| convert_read_err(model_json.display(), e))?;
    let manifest = parse_tfjs_manifest_with_limits(&json_bytes, &limits)?;

    // Stream each shard once (hasher + kernel/bias copy-out): peak heap is
    // one shard's Vec, not the concatenated payload.
    let (k_entry, b_entry) = pick_tfjs_head_entries(&manifest)?;
    let mut hasher = Sha256::new();
    hasher.update(&json_bytes);
    let (kernel_bytes, bias_bytes) = source::read_head_bytes_streaming(
        tfjs_dir,
        &manifest,
        k_entry,
        b_entry,
        Some(&mut hasher),
    )?;
    let source_sha256 = hex_lowercase(&hasher.finalize());

    let weights = head_weights_from_head_byte_ranges(
        &manifest,
        k_entry,
        b_entry,
        &kernel_bytes,
        &bias_bytes,
    )?;
    write_head_artifacts(
        &weights,
        labels,
        dst_dir,
        SourceKind::Tfjs,
        source_sha256,
        fs,
    )
}

/// Parse the labels array from a TFJS `metadata.json`, accepting
/// `wordLabels` (Teachable Machine) or `words` (Speech-Commands),
/// `wordLabels` winning when both present. Empty, non-string, or
/// missing-both is an error.
pub fn read_tfjs_labels(metadata_json: &Path) -> Result<Vec<String>, ConvertError> {
    // Stable token, not the staging path: `Labels` is UserInput, streams
    // unsanitized to the operator.
    let bytes = read_capped_labels(metadata_json, "metadata.json")?;
    let v: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| ConvertError::Labels(format!("metadata.json: invalid JSON: {e}")))?;
    let (key, arr) = if let Some(arr) = v.get("wordLabels").and_then(serde_json::Value::as_array) {
        ("wordLabels", arr)
    } else if let Some(arr) = v.get("words").and_then(serde_json::Value::as_array) {
        ("words", arr)
    } else {
        return Err(ConvertError::Labels(
            "metadata.json: missing `wordLabels` or `words` array".to_string(),
        ));
    };
    let mut out = Vec::with_capacity(arr.len());
    for (i, e) in arr.iter().enumerate() {
        let s = e.as_str().ok_or_else(|| {
            ConvertError::Labels(format!("metadata.json: {key}[{i}] is not a string"))
        })?;
        out.push(s.to_string());
    }
    if out.is_empty() {
        return Err(ConvertError::Labels(format!(
            "metadata.json: {key} is empty"
        )));
    }
    Ok(out)
}

/// Materialize a Burn `Head<NdArray>` and write `head.mpk`, `labels.txt`,
/// `metadata.json` into `dst_dir`; returns paths + fresh head_id.
/// `labels.len()` must equal `weights.n_classes`; labels written verbatim
/// (caller normalizes).
///
/// Crash safety: all three go through [`FsService::put_atomic`] and
/// `head.mpk` builds fully in memory, so no partial siblings appear.
/// `metadata.json` is written LAST as the consistency marker: a crash
/// before it leaves a workspace loaders treat as "not yet converted", not
/// "converted with unknown n_classes".
pub fn write_head_artifacts(
    weights: &HeadWeights,
    labels: &[String],
    dst_dir: &Path,
    source_kind: SourceKind,
    source_sha256: String,
    fs: &dyn FsService,
) -> Result<HeadArtifacts, ConvertError> {
    // Re-check: a hand-built `HeadWeights { n_classes: 0 }` through this
    // `pub` entry would otherwise panic in `Head::new`.
    validate_head_class_count(weights.n_classes, &ConvertLimits::default())?;

    if labels.len() != weights.n_classes {
        return Err(ConvertError::Labels(format!(
            "labels.len() = {}, but n_classes = {}",
            labels.len(),
            weights.n_classes
        )));
    }
    if weights.in_dim != BACKBONE_FEATURE_DIM {
        return Err(ConvertError::Tensor(format!(
            "kernel in_dim {} != BACKBONE_FEATURE_DIM {BACKBONE_FEATURE_DIM}",
            weights.in_dim
        )));
    }

    std::fs::create_dir_all(dst_dir).map_err(|e| convert_write_err(dst_dir.display(), e))?;

    let head_id = HeadId::new();
    let head_mpk = dst_dir.join("head.mpk");
    let labels_txt = dst_dir.join("labels.txt");
    let metadata_json = dst_dir.join("metadata.json");

    let head_blob = build_head_mpk_blob(weights)?;
    fs.put_atomic(&head_mpk, &head_blob).map_err(|e| {
        convert_write_err(head_mpk.display(), std::io::Error::other(format!("{e}")))
    })?;

    let labels_blob = labels.join("\n") + "\n";
    fs.put_atomic(&labels_txt, labels_blob.as_bytes())
        .map_err(|e| {
            convert_write_err(labels_txt.display(), std::io::Error::other(format!("{e}")))
        })?;

    // metadata.json LAST: its presence is the commit marker.
    let meta = ConversionMetadata {
        schema_version: 1,
        source_kind,
        source_sha256: source_sha256.clone(),
        n_classes: weights.n_classes,
        labels: labels.to_vec(),
        created_at: OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            // Unreachable (now_utc always formats); Internal-kinded Tensor.
            .map_err(|e| ConvertError::Tensor(format!("rfc3339 created_at: {e}")))?,
        head_id,
    };
    let bytes = serde_json::to_vec_pretty(&meta)?;
    fs.put_atomic(&metadata_json, &bytes).map_err(|e| {
        convert_write_err(
            metadata_json.display(),
            std::io::Error::other(format!("{e}")),
        )
    })?;

    Ok(HeadArtifacts {
        head_mpk,
        labels_txt,
        metadata_json,
        head_id,
        n_classes: weights.n_classes,
        source_sha256,
    })
}

// MARK: TFJS manifest helpers

/// One TFJS weights-manifest entry. `offset_bytes` is the absolute offset
/// into the concatenated shard buffer (manifest order).
#[derive(Clone, Debug)]
pub(crate) struct TfjsManifestEntry {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) offset_bytes: usize,
    pub(crate) len_bytes: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct TfjsManifest {
    pub(crate) entries: Vec<TfjsManifestEntry>,
    /// Shard paths (relative to model.json dir) in concatenation order.
    pub(crate) shards: Vec<String>,
}

/// `ConvertLimits::default` wrapper preserving the
/// `parse_tfjs_manifest(bytes)` shape used by sibling-crate tests.
#[allow(dead_code)] // sibling-crate tests only; lib build sees no callers.
pub(crate) fn parse_tfjs_manifest(model_json_bytes: &[u8]) -> Result<TfjsManifest, ConvertError> {
    parse_tfjs_manifest_with_limits(model_json_bytes, &ConvertLimits::default())
}

pub(crate) fn parse_tfjs_manifest_with_limits(
    model_json_bytes: &[u8],
    limits: &ConvertLimits,
) -> Result<TfjsManifest, ConvertError> {
    // Cap raw payload before parsing, or an adversarial `model.json` OOMs
    // the parser itself.
    if model_json_bytes.len() > limits.max_manifest_bytes {
        return Err(ConvertError::LimitExceeded {
            what: "manifest_bytes",
            value: model_json_bytes.len() as u64,
            max: limits.max_manifest_bytes as u64,
        });
    }
    let v: serde_json::Value =
        serde_json::from_slice(model_json_bytes).map_err(|e| ConvertError::TfjsParse {
            what: "model.json",
            msg: format!("invalid JSON: {e}"),
        })?;
    let manifest = v
        .get("weightsManifest")
        .and_then(serde_json::Value::as_array)
        .ok_or(ConvertError::TfjsParse {
            what: "weightsManifest",
            msg: "missing or not an array".to_string(),
        })?;
    if manifest.is_empty() {
        return Err(ConvertError::TfjsParse {
            what: "weightsManifest",
            msg: "empty array".to_string(),
        });
    }

    let mut entries: Vec<TfjsManifestEntry> = Vec::new();
    let mut shards: Vec<String> = Vec::new();
    let mut offset: usize = 0;

    for (gi, group) in manifest.iter().enumerate() {
        let paths = group
            .get("paths")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| ConvertError::TfjsParse {
                what: "weightsManifest[i].paths",
                msg: format!("group {gi}: missing or not an array"),
            })?;
        for p in paths {
            let s = p.as_str().ok_or_else(|| ConvertError::TfjsParse {
                what: "weightsManifest[i].paths[j]",
                msg: format!("group {gi}: path is not a string"),
            })?;
            validate_shard_path(s)?;
            // Cap BEFORE pushing, or a manifest declaring millions of paths
            // balloons the `shards` Vec.
            if shards.len() >= limits.max_shards {
                return Err(ConvertError::LimitExceeded {
                    what: "shards",
                    value: (shards.len() as u64).saturating_add(1),
                    max: limits.max_shards as u64,
                });
            }
            shards.push(s.to_string());
        }
        let weights = group
            .get("weights")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| ConvertError::TfjsParse {
                what: "weightsManifest[i].weights",
                msg: format!("group {gi}: missing or not an array"),
            })?;
        for (wi, w) in weights.iter().enumerate() {
            let name = w
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| ConvertError::TfjsParse {
                    what: "weight.name",
                    msg: format!("group {gi} entry {wi}: missing or non-string"),
                })?
                .to_string();
            let dtype = w
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("float32");
            if dtype != "float32" {
                return Err(ConvertError::TfjsDtype {
                    name,
                    dtype: dtype.to_string(),
                });
            }
            let shape_arr = w
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| ConvertError::TfjsParse {
                    what: "weight.shape",
                    msg: format!("`{name}`: missing or not an array"),
                })?;
            let mut shape: Vec<usize> = Vec::with_capacity(shape_arr.len());
            // Takes args (not capture) so the loop can keep mutating `shape`;
            // the error carries the partial shape.
            let overflow = |name: &str, shape: &[usize]| ConvertError::TfjsShapeOverflow {
                name: name.to_string(),
                shape: shape.to_vec(),
            };
            for d in shape_arr {
                let n = d.as_u64().ok_or_else(|| ConvertError::TfjsParse {
                    what: "weight.shape[]",
                    msg: format!("`{name}`: shape dim is not a non-negative integer"),
                })?;
                // On 32-bit hosts a u64 dim may not fit in usize.
                let n_us = usize::try_from(n).map_err(|_| overflow(&name, &shape))?;
                shape.push(n_us);
            }
            // Reject zero dims before the product: a `[D, 0]` kernel / `[0]`
            // bias passes overflow but crashes `LinearConfig::init` or loads
            // n_classes = 0.
            if shape.contains(&0) {
                return Err(ConvertError::TfjsZeroDimension {
                    name: name.clone(),
                    shape: shape.clone(),
                });
            }
            let mut count: usize = 1;
            for &d in &shape {
                count = count
                    .checked_mul(d)
                    .ok_or_else(|| overflow(&name, &shape))?;
            }
            let len_bytes = count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| overflow(&name, &shape))?;
            // Gate the class-count dim FIRST (specific diagnostic, keeps
            // pathological N out of the allocators): a 2-D
            // `[BACKBONE_FEATURE_DIM, N]` is the head kernel.
            if shape.len() == 2
                && shape[0] == BACKBONE_FEATURE_DIM
                && shape[1] > limits.max_n_classes
            {
                return Err(ConvertError::LimitExceeded {
                    what: "n_classes",
                    value: shape[1] as u64,
                    max: limits.max_n_classes as u64,
                });
            }
            if shape.len() == 1 && shape[0] > limits.max_n_classes {
                // May be a non-head bias (BN beta/gamma): label
                // `1d_tensor_dim` not `n_classes` to avoid mis-directing the
                // "shrink your head" hint.
                return Err(ConvertError::LimitExceeded {
                    what: "1d_tensor_dim",
                    value: shape[0] as u64,
                    max: limits.max_n_classes as u64,
                });
            }
            // Per-tensor byte cap (upper envelope across ALL tensors;
            // tighter `max_bias_bytes` fires later) gates allocations before
            // shard data is read.
            if len_bytes > limits.max_kernel_bytes {
                return Err(ConvertError::LimitExceeded {
                    what: "tensor_bytes",
                    value: len_bytes as u64,
                    max: limits.max_kernel_bytes as u64,
                });
            }
            // Cumulative-offset overflow checked BEFORE pushing so the
            // diagnostic can name the entry.
            let next_offset =
                offset
                    .checked_add(len_bytes)
                    .ok_or_else(|| ConvertError::TfjsShapeOverflow {
                        name: format!("<cumulative offset after `{name}`>"),
                        shape: shape.clone(),
                    })?;
            entries.push(TfjsManifestEntry {
                name,
                shape,
                offset_bytes: offset,
                len_bytes,
            });
            offset = next_offset;
        }
    }

    if entries.is_empty() {
        return Err(ConvertError::TfjsParse {
            what: "weightsManifest",
            msg: "no weight entries declared".to_string(),
        });
    }
    if shards.is_empty() {
        return Err(ConvertError::TfjsParse {
            what: "weightsManifest",
            msg: "no shard paths declared".to_string(),
        });
    }
    Ok(TfjsManifest { entries, shards })
}

/// Locate the head kernel + bias: name match, then shape fallback.
pub(crate) fn pick_tfjs_head_entries(
    manifest: &TfjsManifest,
) -> Result<(&TfjsManifestEntry, &TfjsManifestEntry), ConvertError> {
    let kernel = manifest
        .entries
        .iter()
        .find(|e| e.name.ends_with("NewHeadDense/kernel"));
    let bias = manifest
        .entries
        .iter()
        .find(|e| e.name.ends_with("NewHeadDense/bias"));
    if let (Some(k), Some(b)) = (kernel, bias) {
        return Ok((k, b));
    }

    // Shape fallback requires shape[0] == BACKBONE_FEATURE_DIM: collides
    // only with a 2000->2000 layer (then we error, demanding canonical
    // naming).
    let kernel_candidates: Vec<&TfjsManifestEntry> = manifest
        .entries
        .iter()
        .filter(|e| e.shape.len() == 2 && e.shape[0] == BACKBONE_FEATURE_DIM)
        .collect();
    if kernel_candidates.len() != 1 {
        return Err(ConvertError::TfjsLocator(format!(
            "expected exactly one 2-D weight with shape [{BACKBONE_FEATURE_DIM}, N], \
             got {} candidates. manifest:\n{}",
            kernel_candidates.len(),
            format_manifest(manifest)
        )));
    }
    let k = kernel_candidates[0];
    let n = k.shape[1];
    let bias_candidates: Vec<&TfjsManifestEntry> =
        manifest.entries.iter().filter(|e| e.shape == [n]).collect();
    if bias_candidates.len() != 1 {
        return Err(ConvertError::TfjsLocator(format!(
            "expected exactly one 1-D weight with shape [{n}] to pair with `{}`, got {}. manifest:\n{}",
            k.name,
            bias_candidates.len(),
            format_manifest(manifest)
        )));
    }
    Ok((k, bias_candidates[0]))
}

fn format_manifest(manifest: &TfjsManifest) -> String {
    manifest
        .entries
        .iter()
        .map(|e| format!("  {} {:?}", e.name, e.shape))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn extract_head_from_tfjs_buffers(
    manifest: &TfjsManifest,
    weights_blob: &[u8],
) -> Result<HeadWeights, ConvertError> {
    // Total declared size = last entry's offset + len. `checked_add` keeps
    // the failure typed for a hand-built `TfjsManifest` (pub(crate),
    // test-constructible) that bypasses the parse-path checks.
    let declared: usize = match manifest.entries.last() {
        Some(e) => e.offset_bytes.checked_add(e.len_bytes).ok_or_else(|| {
            // Decorate `name` to distinguish cumulative-size from
            // shape-dimension overflow (both TfjsShapeOverflow).
            ConvertError::TfjsShapeOverflow {
                name: format!("<cumulative size for `{}`>", e.name),
                shape: e.shape.clone(),
            }
        })?,
        None => 0,
    };
    if weights_blob.len() != declared {
        return Err(ConvertError::TfjsBlobLength {
            have: weights_blob.len(),
            declared,
        });
    }

    let (k_entry, b_entry) = pick_tfjs_head_entries(manifest)?;
    let n_classes = check_head_entry_shapes(k_entry, b_entry)?;
    validate_head_class_count(n_classes, &ConvertLimits::default())?;

    let need = k_entry
        .offset_bytes
        .saturating_add(k_entry.len_bytes)
        .max(b_entry.offset_bytes.saturating_add(b_entry.len_bytes));
    // Redundant under the strict total-length check above; kept in case it
    // is ever loosened (e.g. trailing pad).
    if weights_blob.len() < need {
        return Err(ConvertError::TfjsShortRead {
            have: weights_blob.len(),
            need,
        });
    }

    let kernel_count = BACKBONE_FEATURE_DIM.checked_mul(n_classes).ok_or_else(|| {
        ConvertError::TfjsShapeOverflow {
            name: k_entry.name.clone(),
            shape: k_entry.shape.clone(),
        }
    })?;
    // Keras Dense `[in, out]` matches Burn's `Linear.weight`, no transpose.
    let kernel = read_f32_at_checked(
        weights_blob,
        k_entry.offset_bytes,
        kernel_count,
        &k_entry.name,
    )?;
    let bias = read_f32_at_checked(weights_blob, b_entry.offset_bytes, n_classes, &b_entry.name)?;

    Ok(HeadWeights {
        kernel,
        bias,
        n_classes,
        in_dim: BACKBONE_FEATURE_DIM,
    })
}

/// Decode `count` little-endian f32 from `bytes` at `offset`, rejecting
/// non-finite values (else they'd trip the inference finite validator at
/// activation). Callers pre-validate `offset + count*4 <= bytes.len()`.
fn read_f32_at_checked(
    bytes: &[u8],
    offset: usize,
    count: usize,
    tensor: &str,
) -> Result<Vec<f32>, ConvertError> {
    // Checked arithmetic so a hand-built `TfjsManifestEntry` with extreme
    // values can't panic on the slice index.
    let byte_count = count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| ConvertError::TfjsShapeOverflow {
            name: tensor.to_string(),
            shape: vec![count],
        })?;
    let end = offset
        .checked_add(byte_count)
        .ok_or_else(|| ConvertError::TfjsShapeOverflow {
            name: tensor.to_string(),
            shape: vec![offset, byte_count],
        })?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| ConvertError::TfjsBlobLength {
            have: bytes.len(),
            declared: end,
        })?;
    let mut out = vec![0.0f32; count];
    for (i, chunk) in slice.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !v.is_finite() {
            return Err(ConvertError::NonFiniteWeight {
                tensor: tensor.to_string(),
                index: i,
                value: v,
            });
        }
        out[i] = v;
    }
    Ok(out)
}

/// Streaming-read sibling of [`extract_head_from_tfjs_buffers`]: takes the
/// kernel + bias byte ranges (one slice each, length = `entry.len_bytes`)
/// and validates the same shape invariants.
pub(crate) fn head_weights_from_head_byte_ranges(
    _manifest: &TfjsManifest,
    k_entry: &TfjsManifestEntry,
    b_entry: &TfjsManifestEntry,
    kernel_bytes: &[u8],
    bias_bytes: &[u8],
) -> Result<HeadWeights, ConvertError> {
    let n_classes = check_head_entry_shapes(k_entry, b_entry)?;
    let limits = ConvertLimits::default();
    validate_head_class_count(n_classes, &limits)?;
    let kernel_count = BACKBONE_FEATURE_DIM.checked_mul(n_classes).ok_or_else(|| {
        ConvertError::TfjsShapeOverflow {
            name: k_entry.name.clone(),
            shape: k_entry.shape.clone(),
        }
    })?;
    let kernel_need = kernel_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| ConvertError::TfjsShapeOverflow {
            name: k_entry.name.clone(),
            shape: k_entry.shape.clone(),
        })?;
    // Re-check the byte cap (parse already did): closes a future parse-cap
    // loosening for this streaming-reader path.
    if kernel_need > limits.max_kernel_bytes {
        return Err(ConvertError::LimitExceeded {
            what: "kernel_bytes",
            value: kernel_need as u64,
            max: limits.max_kernel_bytes as u64,
        });
    }
    if kernel_bytes.len() != kernel_need {
        return Err(ConvertError::TfjsShortRead {
            have: kernel_bytes.len(),
            need: kernel_need,
        });
    }
    let bias_need = n_classes
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| ConvertError::TfjsShapeOverflow {
            name: b_entry.name.clone(),
            shape: b_entry.shape.clone(),
        })?;
    if bias_need > limits.max_bias_bytes {
        return Err(ConvertError::LimitExceeded {
            what: "bias_bytes",
            value: bias_need as u64,
            max: limits.max_bias_bytes as u64,
        });
    }
    if bias_bytes.len() != bias_need {
        return Err(ConvertError::TfjsShortRead {
            have: bias_bytes.len(),
            need: bias_need,
        });
    }
    // Keras Dense `[in, out]` matches Burn's `Linear.weight`, no transpose.
    let kernel = read_f32_at_checked(kernel_bytes, 0, kernel_count, &k_entry.name)?;
    let bias = read_f32_at_checked(bias_bytes, 0, n_classes, &b_entry.name)?;
    Ok(HeadWeights {
        kernel,
        bias,
        n_classes,
        in_dim: BACKBONE_FEATURE_DIM,
    })
}

// MARK: typed events
//
// Wire shape for the durable JSONL log + SSE bridge: each line is
// `{seq, at, ...flattened ConvertEvent}`, mirroring the `TrainEvent`
// taxonomy so SSE clients render convert progress like training.

/// Per-converter stage selector, stamped on
/// [`ConvertEvent::StageStarted`] (progress) and
/// [`ConvertEvent::JobFailed`] (which step failed). Variants flat across
/// both converters (`Prepare`/`PublishHead` shared);
/// `#[non_exhaustive]` for future converters.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ConvertStage {
    /// Shared: pre-dispatch initial value so a pre-`StageStarted`
    /// `JobFailed` still has a label.
    Prepare,
    /// Alpkg: read + parse the manifest.
    ReadManifest,
    /// Alpkg: structural validation (head_id, label count, sha256).
    ValidateManifest,
    /// Alpkg: stream the `.mpk` once to verify size + sha256.
    VerifyMpk,
    /// Alpkg: hard-link (or copy) the `.mpk` into `.tmp/`.
    StageMpk,
    /// Tfjs: read + parse `model.json`.
    ReadModelJson,
    /// Tfjs: stage declared shards into `.tmp/convert-<job_id>/`.
    StageShards,
    /// Tfjs: extract kernel + bias into [`HeadWeights`].
    ExtractWeights,
    /// Tfjs: read labels, cross-check count vs `n_classes`.
    ReadLabels,
    /// Tfjs: build the ACSTHEAD `.mpk` blob, stage under `.tmp/`.
    StageHeadMpk,
    /// Shared: `publish_{trained,imported}_head`; terminal happy path.
    PublishHead,
}

/// Typed failure payload for [`ConvertEvent::JobFailed`]: `category`
/// discriminates; per-variant fields let the frontend build hint copy
/// without re-parsing `error`. Many-to-one from [`ConvertError`] (variants
/// share a category when hint copy is identical); `#[non_exhaustive]` so
/// consumers fall back to the `error` string.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "category", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ConvertFailPayload {
    /// TFJS source failed parse/validation (catch-all: re-export).
    SourceMalformed {
        detail: String,
    },
    LimitExceeded {
        /// Stable cap identifier (matches a [`ConvertLimits`] field).
        what: String,
        value: u64,
        max: u64,
    },
    /// Distinct from `LimitExceeded` so the frontend renders zero-class copy.
    BadClassCount {
        got: u32,
        max: u32,
    },
    /// Labels source failed parse, was empty, or count != `n_classes`.
    Labels {
        detail: String,
    },
    AlpkgManifestSchema {
        reason: String,
    },
    AlpkgSizeMismatch {
        expected: u64,
        observed: u64,
    },
    AlpkgHashMismatch {
        expected: String,
        observed: String,
    },
    /// Same `head_id`, different `sha256`; operator deletes existing.
    HeadIdCollision {
        head_id: String,
        got_sha256: String,
        stored_sha256: String,
    },
    /// Daemon-internal (FS, recorder, tensor, serializer); operator retries.
    Internal {
        detail: String,
    },
}

/// Terminal-success summary for [`ConvertEvent::JobCompleted`] +
/// [`run_convert_job`]'s return, enough to render the published-head card
/// without a follow-up fetch. `RegistryJobResult::Convert` takes a subset;
/// the full list lives here so JSONL replay renders the live SSE shape.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ConvertResult {
    pub head_id: HeadId,
    /// Lowercase-hex SHA-256 of the published `.mpk` bytes.
    pub head_sha256: String,
    pub n_classes: u32,
    /// Full label list in n_classes order.
    pub classes: Vec<String>,
}

/// One lifecycle event to the JSONL log AND SSE broadcast (`kind`
/// discriminator). Wrapper events come from [`run_convert_job`]; stage
/// transitions / outcomes from the workers. `#[non_exhaustive]`: external
/// matches handle the unknown case.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ConvertEvent {
    /// Admission cleared, log opened, before any pipeline work.
    JobSubmitted {
        head_id: HeadId,
        converter: ConverterType,
    },
    /// Worker started on the blocking pool; distinct from `JobSubmitted` so
    /// SSE renders a sub-second queued->running.
    JobRunning,
    /// Stage boundary; updates the stage tracker so a later `JobFailed`
    /// carries the label.
    StageStarted { stage: ConvertStage },
    /// Alpkg `ValidateManifest` outcome; `sha256` is the manifest's declared
    /// `.mpk` hash (`VerifyMpk` proves agreement).
    ManifestValidated { n_classes: u32, sha256: String },
    /// Alpkg `VerifyMpk` outcome. Hashes the STAGED inode (not the source
    /// path) to close a concurrent re-upload TOCTOU.
    MpkVerified { size_bytes: u64, sha256: String },
    /// Tfjs `ExtractWeights` outcome: parsed head shape.
    WeightsExtracted { n_classes: u32, in_dim: u32 },
    /// Tfjs `ReadLabels` outcome (`n_labels` == head's n_classes).
    LabelsLoaded { n_labels: u32 },
    /// Head landed via the rotation primitive; emitted only on success
    /// (absence means publish was not reached). `idempotent_skip = true`
    /// for the Alpkg already-exists no-op.
    HeadPublished {
        head_id: HeadId,
        head_sha256: String,
        size_bytes: u64,
        n_classes: u32,
        classes: Vec<String>,
        workspace_revision: WorkspaceRevision,
        idempotent_skip: bool,
    },
    /// Terminal success; full [`ConvertResult`] for tab-refresh replay.
    JobCompleted { result: ConvertResult },
    /// Terminal failure. `stage` is the last `StageStarted`; `severity`
    /// from the error's `ErrorKind`; `payload` the typed hint-copy enum.
    JobFailed {
        stage: ConvertStage,
        severity: Severity,
        error: String,
        #[serde(flatten)]
        payload: ConvertFailPayload,
    },
}

/// Map [`ConvertError`] to the typed [`ConvertFailPayload`] for frontend
/// hint copy: the source-malformation tail collapses to `SourceMalformed`
/// (same advice), structurally-rich variants keep their own payload.
fn fail_payload_from_convert_error(err: &ConvertError) -> ConvertFailPayload {
    match err {
        ConvertError::LimitExceeded { what, value, max } => ConvertFailPayload::LimitExceeded {
            what: what.to_string(),
            value: *value,
            max: *max,
        },
        ConvertError::BadClassCount { got, max } => ConvertFailPayload::BadClassCount {
            got: u32::try_from(*got).unwrap_or(u32::MAX),
            max: u32::try_from(*max).unwrap_or(u32::MAX),
        },
        ConvertError::Labels(detail) => ConvertFailPayload::Labels {
            detail: detail.clone(),
        },
        ConvertError::AlpkgManifestSchema { reason } => ConvertFailPayload::AlpkgManifestSchema {
            reason: reason.clone(),
        },
        ConvertError::AlpkgSizeMismatch { expected, observed } => {
            ConvertFailPayload::AlpkgSizeMismatch {
                expected: *expected,
                observed: *observed,
            }
        }
        ConvertError::AlpkgHashMismatch { expected, observed } => {
            ConvertFailPayload::AlpkgHashMismatch {
                expected: expected.clone(),
                observed: observed.clone(),
            }
        }
        ConvertError::HeadIdCollision(file_err) => match file_err {
            crate::file_mgr::FileError::HeadIdCollision {
                head_id,
                got_sha256,
                stored_sha256,
            } => ConvertFailPayload::HeadIdCollision {
                head_id: head_id.to_string(),
                got_sha256: got_sha256.clone(),
                stored_sha256: stored_sha256.clone(),
            },
            // Any other FileError is a violated invariant (the only
            // constructor pulls the typed collision from the source chain).
            other => ConvertFailPayload::Internal {
                detail: format!("head-id-collision wraps unexpected FileError: {other}"),
            },
        },
        // TFJS source-malformed catch-all (advice is identical).
        ConvertError::TfjsParse { .. }
        | ConvertError::TfjsLocator(_)
        | ConvertError::TfjsShortRead { .. }
        | ConvertError::TfjsDtype { .. }
        | ConvertError::TfjsUnsafePath(_)
        | ConvertError::TfjsShapeOverflow { .. }
        | ConvertError::TfjsBlobLength { .. }
        | ConvertError::TfjsZeroDimension { .. }
        | ConvertError::NonFiniteWeight { .. } => ConvertFailPayload::SourceMalformed {
            detail: err.to_string(),
        },
        // Daemon-internal; operator retries.
        ConvertError::Read { .. }
        | ConvertError::Write { .. }
        | ConvertError::Record(_)
        | ConvertError::Tensor(_)
        | ConvertError::MetadataSerialize(_)
        | ConvertError::NotImplemented(_)
        | ConvertError::Busy
        | ConvertError::AlpkgManifestRead { .. } => ConvertFailPayload::Internal {
            detail: err.to_string(),
        },
    }
}

/// [`Severity`] via `Severity::from(err.kind())` (single source of truth,
/// shared with the training producer).
fn severity_from_convert_error(err: &ConvertError) -> Severity {
    Severity::from(err.kind())
}

/// Fan one [`ConvertEvent`] to three sinks: stage tracker, JSONL log, SSE
/// broadcast (via the optional [`JobHandle`]). Failures are tracing-warned,
/// never returned - a failed log write must not promote a successful run to
/// failed.
fn emit_convert_event(
    log: &mut JsonlEventLog<ConvertEvent>,
    handle: Option<&JobHandle>,
    stage: &mut ConvertStage,
    event: ConvertEvent,
) {
    if let ConvertEvent::StageStarted { stage: new_stage } = &event {
        *stage = *new_stage;
    }
    if let Err(err) = log.emit(&event) {
        tracing::warn!(target: "converter", err = %err, "converter: log emit failed");
    }
    if let Some(h) = handle {
        match serde_json::to_string(&event) {
            Ok(s) => h.append_log(s),
            Err(e) => {
                tracing::warn!(target: "converter", err = %e, "converter: SSE event serialize failed");
            }
        }
    }
}

fn begin_stage(
    log: &mut JsonlEventLog<ConvertEvent>,
    handle: Option<&JobHandle>,
    stage: &mut ConvertStage,
    new_stage: ConvertStage,
) {
    emit_convert_event(
        log,
        handle,
        stage,
        ConvertEvent::StageStarted { stage: new_stage },
    );
}

// MARK: convert pipeline

/// Convert-worker inputs, built by the api producer; all paths
/// pre-resolved under `<workspace_dir>/converters/`. Shared admission
/// fields are flat; variant-specific fields live on [`ConvertJobKind`].
#[derive(Clone, Debug)]
pub struct ConvertJob {
    /// Drives the JSONL log filename `converter_logs/<job_id>.jsonl`.
    pub job_id: crate::common::ids::JobId,
    pub workspace_id: crate::common::ids::WorkspaceId,
    /// TFJS: route-allocated via `HeadId::new()`. Alpkg: extracted from the
    /// manifest at start-time so the sync response can return it.
    pub head_id: crate::common::ids::HeadId,
    /// Producer-snapshotted for stale detection. For Alpkg this is the
    /// destination's revision at convert-start (the source manifest's is
    /// workspace-local and discarded).
    pub workspace_revision: crate::common::workspace::WorkspaceRevision,
    pub kind: ConvertJobKind,
}

/// Variant-specific convert payload, one arm per [`ConverterType`].
#[derive(Clone, Debug)]
pub enum ConvertJobKind {
    /// TFJS bundle -> head; pre-resolved absolute paths.
    Tfjs(ConvertJobTfjs),
    /// `.alpkg` archive -> head import; pre-resolved `.mpk` + `.json`.
    Alpkg(ConvertJobAlpkg),
}

impl ConvertJobKind {
    /// Operator-facing [`ConverterType`] so [`run_convert_job`] stamps
    /// `JobSubmitted` before any stage.
    pub fn converter_type(&self) -> ConverterType {
        match self {
            ConvertJobKind::Tfjs(_) => ConverterType::Tfjs,
            ConvertJobKind::Alpkg(_) => ConverterType::Alpkg,
        }
    }
}

/// TFJS-specific worker inputs.
#[derive(Clone, Debug)]
pub struct ConvertJobTfjs {
    pub model_json_path: PathBuf,
    /// Manifest-declared order, paired 1:1 with `shard_names`.
    pub shard_paths: Vec<PathBuf>,
    /// Route's snapshot of `manifest.shards` (declared NAMES, manifest
    /// order). The worker re-parses and compares element-wise; without it a
    /// same-cardinality reorder (re-uploaded `model.json`) pairs wrong bytes
    /// with declared names and publishes a corrupt head.
    pub shard_names: Vec<String>,
    pub labels_path: PathBuf,
    pub labels_format: crate::file_mgr::LabelsFormat,
}

/// `.alpkg`-specific worker inputs, uploaded to
/// `<workspace>/converters/alpkg/<head_id>/`.
#[derive(Clone, Debug)]
pub struct ConvertJobAlpkg {
    pub mpk_path: PathBuf,
    pub manifest_path: PathBuf,
}

/// Run a convert job end-to-end: open the JSONL log, fan [`ConvertEvent`]s
/// through it + the SSE bridge, dispatch to the per-converter worker, sweep
/// per-job tempfiles, emit the terminal event, then consume `job_handle` to
/// flip registry state.
///
/// This fn's terminal `succeed`/`fail` consumes `job_handle`; the handle moves
/// on the first call, so a literal repeat is a compile error and a duplicate
/// registry transition is a silent no-op (idempotent `terminate`). `None` is
/// test-only (no SSE, no registry transition). On failure no head record is
/// committed.
pub fn run_convert_job(
    files: std::sync::Arc<dyn crate::file_mgr::FsService>,
    job: ConvertJob,
    job_handle: Option<JobHandle>,
) -> Result<ConvertResult, ConvertError> {
    let workspace_dir = crate::file_mgr::schema::workspace_dir_for(files.root(), &job.workspace_id);

    // Open the per-job JSONL log; an unwritable `converter_logs/` is a typed
    // failure. The handle rides the failure path so a shutdown drain still
    // sees the registry transition; with no JSONL line possible the copy
    // goes only through `fail`.
    let mut log = match JsonlEventLog::<ConvertEvent>::open(
        &workspace_dir,
        CONVERTER_LOGS_DIR_NAME,
        job.job_id,
    ) {
        Ok(l) => l,
        Err(source) => {
            let log_path = workspace_dir
                .join(CONVERTER_LOGS_DIR_NAME)
                .join(format!("{}.jsonl", job.job_id));
            let e = convert_write_err(log_path.display(), source);
            if let Some(h) = job_handle {
                h.fail(e.to_string());
            }
            return Err(e);
        }
    };

    let mut stage = ConvertStage::Prepare;
    let converter = job.kind.converter_type();

    emit_convert_event(
        &mut log,
        job_handle.as_ref(),
        &mut stage,
        ConvertEvent::JobSubmitted {
            head_id: job.head_id,
            converter,
        },
    );
    emit_convert_event(
        &mut log,
        job_handle.as_ref(),
        &mut stage,
        ConvertEvent::JobRunning,
    );

    let result = run_convert_job_inner(
        &files,
        &job,
        &workspace_dir,
        &mut log,
        job_handle.as_ref(),
        &mut stage,
    );

    // Unconditional NotFound-tolerant sweep of per-job tempfiles (success:
    // `.mpk` already renamed out, only scratch remains; failure: whatever
    // the inner left behind).
    let result_ok = result.is_ok();
    let staging_dir = convert_staging_dir(&workspace_dir, job.job_id);
    let mpk_tempfile = convert_mpk_tempfile(&workspace_dir, job.job_id, job.head_id);
    if let Err(e) = std::fs::remove_dir_all(&staging_dir)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            target: "converter",
            err = %e,
            path = %staging_dir.display(),
            published = result_ok,
            "convert: failed to remove staging dir; storage_reaper will sweep",
        );
    }
    if let Err(e) = std::fs::remove_file(&mpk_tempfile)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        tracing::warn!(
            target: "converter",
            err = %e,
            path = %mpk_tempfile.display(),
            "convert: failed to remove .mpk tempfile; boot recovery should sweep",
        );
    }

    // Emit the rich typed payload FIRST, THEN the registry transition
    // (whose flat `state` event consumers ignore once they have the typed
    // payload).
    let final_event = match &result {
        Ok(r) => ConvertEvent::JobCompleted { result: r.clone() },
        Err(e) => ConvertEvent::JobFailed {
            stage,
            severity: severity_from_convert_error(e),
            error: e.to_string(),
            payload: fail_payload_from_convert_error(e),
        },
    };
    emit_convert_event(&mut log, job_handle.as_ref(), &mut stage, final_event);

    // Registry transition; consumes the JobHandle. Best-effort (broadcast
    // can lag); the typed payload above is the durable record.
    if let Some(handle) = job_handle {
        match &result {
            Ok(r) => handle.succeed(Some(RegistryJobResult::Convert {
                head_id: r.head_id,
                sha256: r.head_sha256.clone(),
                n_classes: r.n_classes,
            })),
            Err(e) => handle.fail(format!("{e}")),
        }
    }
    result
}

/// Per-job staging dir under `.tmp/` (inner pipeline + cleanup wrapper must
/// agree on this path).
fn convert_staging_dir(workspace_dir: &Path, job_id: crate::common::ids::JobId) -> PathBuf {
    workspace_dir.join(".tmp").join(format!("convert-{job_id}"))
}

/// Per-job `.mpk` tempfile under `.tmp/`; renamed out on success by the
/// rotation primitive (`publish_trained_head` for TFJS, `publish_imported_head`
/// for `.alpkg`), swept by the wrapper otherwise.
fn convert_mpk_tempfile(
    workspace_dir: &Path,
    job_id: crate::common::ids::JobId,
    head_id: crate::common::ids::HeadId,
) -> PathBuf {
    workspace_dir
        .join(".tmp")
        .join(format!("convert-{job_id}-{head_id}.mpk"))
}

/// Dispatches on [`ConvertJobKind`] to the variant-specific worker; the
/// [`run_convert_job`] wrapper owns the log + terminal fan-out.
fn run_convert_job_inner(
    files: &std::sync::Arc<dyn crate::file_mgr::FsService>,
    job: &ConvertJob,
    workspace_dir: &Path,
    log: &mut JsonlEventLog<ConvertEvent>,
    handle: Option<&JobHandle>,
    stage: &mut ConvertStage,
) -> Result<ConvertResult, ConvertError> {
    match &job.kind {
        ConvertJobKind::Tfjs(tfjs) => {
            run_tfjs_convert(files, job, tfjs, workspace_dir, log, handle, stage)
        }
        ConvertJobKind::Alpkg(alpkg) => {
            run_alpkg_convert(files, job, alpkg, workspace_dir, log, handle, stage)
        }
    }
}

/// `.alpkg` head-import worker: verify the `.mpk` + `.json` pair,
/// idempotency-check, stage the `.mpk` into `.tmp/`, publish via the
/// rotation primitive (steps map to `ConvertStage`s; wrapper sweeps
/// tempfiles on both paths). Stage BEFORE hash so the hash covers the
/// stable staged inode (closing a concurrent re-upload TOCTOU) and
/// VerifyMpk verifies exactly the bytes publish renames.
#[allow(clippy::too_many_arguments)]
fn run_alpkg_convert(
    files: &std::sync::Arc<dyn crate::file_mgr::FsService>,
    job: &ConvertJob,
    alpkg: &ConvertJobAlpkg,
    workspace_dir: &Path,
    log: &mut JsonlEventLog<ConvertEvent>,
    handle: Option<&JobHandle>,
    stage: &mut ConvertStage,
) -> Result<ConvertResult, ConvertError> {
    begin_stage(log, handle, stage, ConvertStage::ReadManifest);
    // Cap the re-read: uploads overlap convert jobs freely (the lease
    // excludes only WorkspaceDelete), so a race-replaced manifest could
    // otherwise slurp the 256 MiB upload ceiling for this re-pass.
    let manifest_bytes = read_capped_for_convert(
        &alpkg.manifest_path,
        crate::file_mgr::schema::MAX_HEAD_MANIFEST_BYTES,
    )?;
    let manifest: crate::common::workspace::HeadManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| ConvertError::AlpkgManifestSchema {
            // Stable token, not the staging path: `AlpkgManifestSchema`
            // (UserInput) streams unsanitized to the operator.
            reason: format!("manifest.json: {e}"),
        })?;

    // Structural validation of value-shape invariants serde
    // (deny_unknown_fields) can't express.
    begin_stage(log, handle, stage, ConvertStage::ValidateManifest);
    validate_alpkg_head_manifest(&manifest, job.head_id)?;
    emit_convert_event(
        log,
        handle,
        stage,
        ConvertEvent::ManifestValidated {
            n_classes: manifest.n_classes,
            sha256: manifest.sha256.clone(),
        },
    );

    // Stage the .mpk into `.tmp/` and hash the STAGED inode at VerifyMpk so
    // verified bytes == published bytes: hashing the source then re-opening
    // to stage is a TOCTOU where a concurrent re-upload renames a different
    // inode over the source between hash and hard-link, publishing bytes
    // whose sha256 != manifest = a permanently un-activatable head. Once
    // `hard_link`/`copy` returns, `mpk_tempfile` is a stable inode immune to
    // the swap. Hard-link (intra-FS) is needed for the rotation primitive's
    // atomic rename; `fs::copy` fallback if links rejected. Cleanup unlinks
    // only the staging dirent, never the operator's source.
    begin_stage(log, handle, stage, ConvertStage::StageMpk);
    let tmp_root = workspace_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_root).map_err(|e| convert_write_err(tmp_root.display(), e))?;
    let mpk_tempfile = convert_mpk_tempfile(workspace_dir, job.job_id, job.head_id);
    // Defensive remove: a prior failed-and-retried run could occupy this
    // path (NotFound is the happy case).
    if let Err(e) = std::fs::remove_file(&mpk_tempfile)
        && e.kind() != std::io::ErrorKind::NotFound
    {
        return Err(convert_write_err(mpk_tempfile.display(), e));
    }
    if let Err(e) = std::fs::hard_link(&alpkg.mpk_path, &mpk_tempfile) {
        // EXDEV / EPERM / unsupported: byte-copy. NotFound on the source is
        // a real read error (uploaded file vanished).
        if e.kind() == std::io::ErrorKind::NotFound {
            return Err(convert_read_err(alpkg.mpk_path.display(), e));
        }
        std::fs::copy(&alpkg.mpk_path, &mpk_tempfile)
            .map_err(|source| convert_read_err(alpkg.mpk_path.display(), source))?;
    }
    // No caller-side fsync: the rotation primitive's defensive `sync_all`
    // before its rename already covers both paths.

    // Stream the STAGED .mpk once to verify size + hash vs the manifest:
    // authoritative gate proving the bytes publish renames into `heads/`
    // hash to `manifest.sha256`. Streaming caps RSS at the 64 KiB buffer.
    begin_stage(log, handle, stage, ConvertStage::VerifyMpk);
    let (observed_size, observed_sha256) = stream_sha256_file(&mpk_tempfile)?;
    if observed_size != manifest.size_bytes {
        return Err(ConvertError::AlpkgSizeMismatch {
            expected: manifest.size_bytes,
            observed: observed_size,
        });
    }
    if observed_sha256 != manifest.sha256 {
        return Err(ConvertError::AlpkgHashMismatch {
            expected: manifest.sha256.clone(),
            observed: observed_sha256,
        });
    }
    emit_convert_event(
        log,
        handle,
        stage,
        ConvertEvent::MpkVerified {
            size_bytes: observed_size,
            sha256: observed_sha256.clone(),
        },
    );

    // Publish via imported-head rotation. Rewrite `workspace_id` /
    // `workspace_revision` / `created_at` to the destination (source values
    // are workspace-local to the exporting install);
    // `head_id`/`sha256`/`n_classes`/`size_bytes`/`labels` survive verbatim
    // (they describe the head, not the workspace).
    begin_stage(log, handle, stage, ConvertStage::PublishHead);
    let published_manifest = crate::common::workspace::HeadManifest {
        head_id: job.head_id,
        workspace_id: job.workspace_id,
        workspace_revision: job.workspace_revision.clone(),
        sha256: manifest.sha256.clone(),
        n_classes: manifest.n_classes,
        size_bytes: manifest.size_bytes,
        created_at: crate::file_mgr::now_rfc3339(),
        labels: manifest.labels.clone(),
    };
    let pending = crate::file_mgr::PendingHead {
        head_id: job.head_id,
        mpk_tempfile: mpk_tempfile.clone(),
        manifest: published_manifest,
    };
    let idempotent_skip = match files.publish_imported_head(&job.workspace_id, pending) {
        Ok(crate::file_mgr::HeadImportResult::Published(_)) => false,
        Ok(crate::file_mgr::HeadImportResult::AlreadyExists) => {
            // Idempotent no-op (matching sha256); `HeadPublished` still fires
            // with `idempotent_skip = true`.
            true
        }
        Err(fs_err) => {
            // Distinguish head-id-collision (operator 409) from generic FS
            // failures; destructuring the source chain extracts the strings
            // so the re-wrap preserves them.
            use std::error::Error as _;
            if let Some(crate::file_mgr::FileError::HeadIdCollision {
                head_id,
                got_sha256,
                stored_sha256,
            }) = fs_err
                .source()
                .and_then(|s| s.downcast_ref::<crate::file_mgr::FileError>())
            {
                return Err(ConvertError::HeadIdCollision(
                    crate::file_mgr::FileError::HeadIdCollision {
                        head_id: head_id.clone(),
                        got_sha256: got_sha256.clone(),
                        stored_sha256: stored_sha256.clone(),
                    },
                ));
            }
            return Err(convert_write_err(
                workspace_dir.display(),
                std::io::Error::other(fs_err.to_string()),
            ));
        }
    };

    emit_convert_event(
        log,
        handle,
        stage,
        ConvertEvent::HeadPublished {
            head_id: job.head_id,
            head_sha256: manifest.sha256.clone(),
            size_bytes: manifest.size_bytes,
            n_classes: manifest.n_classes,
            classes: manifest.labels.clone(),
            workspace_revision: job.workspace_revision.clone(),
            idempotent_skip,
        },
    );

    Ok(ConvertResult {
        head_id: job.head_id,
        head_sha256: manifest.sha256,
        n_classes: manifest.n_classes,
        classes: manifest.labels,
    })
}

/// Per-label content gate (TFJS + `.alpkg`): reject empty/whitespace,
/// TRIMMED length > [`MAX_LABEL_BYTES`], or control chars. Mirrors the
/// runtime `head::load_inner` loader (trim, drop empties, cap) so a label
/// accepted here never fails activation. `mk_err` selects the
/// operator-facing variant.
fn validate_convert_labels(
    labels: &[String],
    mk_err: impl Fn(String) -> ConvertError,
) -> Result<(), ConvertError> {
    for (i, lbl) in labels.iter().enumerate() {
        if lbl.trim().is_empty() {
            return Err(mk_err(format!("labels[{i}] is empty")));
        }
        let trimmed_len = lbl.trim().len();
        if trimmed_len > MAX_LABEL_BYTES {
            return Err(mk_err(format!(
                "labels[{i}] is {trimmed_len} bytes, over the {MAX_LABEL_BYTES}-byte cap"
            )));
        }
        if let Some(ch) = lbl.chars().find(|c| c.is_control()) {
            return Err(mk_err(format!(
                "labels[{i}] contains a control character (U+{:04X})",
                ch as u32
            )));
        }
    }
    Ok(())
}

/// Structural validation for the operator-uploaded `HeadManifest`,
/// mirroring the frontend's `validateManifestStructure`. Agreement with
/// `expected_head_id` (route's start-time snapshot) trip-wires a manifest
/// swap between the route and worker parses.
fn validate_alpkg_head_manifest(
    manifest: &crate::common::workspace::HeadManifest,
    expected_head_id: crate::common::ids::HeadId,
) -> Result<(), ConvertError> {
    if manifest.head_id != expected_head_id {
        return Err(alpkg_schema_err(format!(
            "manifest head_id {} does not match route-extracted head_id {}",
            manifest.head_id, expected_head_id
        )));
    }
    if manifest.n_classes == 0 {
        return Err(alpkg_schema_err("n_classes must be >= 1"));
    }
    let limits = ConvertLimits::default();
    let max_n = u32::try_from(limits.max_n_classes).unwrap_or(u32::MAX);
    if manifest.n_classes > max_n {
        return Err(alpkg_schema_err(format!(
            "n_classes = {} exceeds cap {}",
            manifest.n_classes, max_n
        )));
    }
    if manifest.labels.len() != manifest.n_classes as usize {
        return Err(alpkg_schema_err(format!(
            "labels.len() = {} does not match n_classes = {}",
            manifest.labels.len(),
            manifest.n_classes
        )));
    }
    // Reject bad labels at the import boundary (UserInput 400) with the
    // runtime loader's verdict; else the head imports but is permanently
    // un-activatable (activation fires the same gates as a confusing
    // internal error).
    validate_convert_labels(&manifest.labels, alpkg_schema_err)?;
    // 64 lowercase hex chars. A placeholder ("00"*32) passes here and is
    // caught later as AlpkgHashMismatch (real bytes hashed); placeholder
    // detection is out of scope.
    if manifest.sha256.len() != 64
        || !manifest
            .sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(alpkg_schema_err(format!(
            "sha256 must be 64 lowercase hex chars; got `{}`",
            manifest.sha256
        )));
    }
    // Defense-in-depth cap on declared `.mpk` size: guards a future caller
    // that pre-allocates from `manifest.size_bytes` against a multi-GiB
    // attacker integer (1 GiB is well above real heads). Typed
    // `LimitExceeded` (not `AlpkgManifestSchema`) so the frontend picks hint
    // copy without parsing a reason string.
    const MAX_HEAD_SIZE_BYTES: u64 = 1024 * 1024 * 1024;
    if manifest.size_bytes > MAX_HEAD_SIZE_BYTES {
        return Err(ConvertError::LimitExceeded {
            what: "alpkg_size_bytes",
            value: manifest.size_bytes,
            max: MAX_HEAD_SIZE_BYTES,
        });
    }
    Ok(())
}

/// Stream SHA-256 + byte count in one pass (64 KiB buffer), to verify the
/// `.alpkg` `.mpk` against the manifest.
fn stream_sha256_file(path: &Path) -> Result<(u64, String), ConvertError> {
    use std::io::Read as _;
    let mut f = std::fs::File::open(path).map_err(|e| convert_read_err(path.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut total: u64 = 0;
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| convert_read_err(path.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        // `checked_add` (overflow unreachable on a real file): a `+=` wrap
        // would corrupt the count.
        total = match total.checked_add(n as u64) {
            Some(t) => t,
            None => {
                return Err(convert_read_err(
                    path.display(),
                    std::io::Error::other(format!(
                        "streaming SHA-256 byte-count overflow: {total} + {n} > u64::MAX",
                    )),
                ));
            }
        };
    }
    Ok((total, hex_lowercase(&hasher.finalize())))
}

/// TFJS bundle -> head worker: read `model.json` + shards + labels, extract
/// the head `Linear` weights into a Burn `.mpk`, publish via the rotation
/// primitive (steps map to `ConvertStage`s). Never idempotent (fresh
/// `head_id` per job).
#[allow(clippy::too_many_arguments)]
fn run_tfjs_convert(
    files: &std::sync::Arc<dyn crate::file_mgr::FsService>,
    job: &ConvertJob,
    tfjs: &ConvertJobTfjs,
    workspace_dir: &Path,
    log: &mut JsonlEventLog<ConvertEvent>,
    handle: Option<&JobHandle>,
    stage: &mut ConvertStage,
) -> Result<ConvertResult, ConvertError> {
    begin_stage(log, handle, stage, ConvertStage::ReadModelJson);
    let json_bytes = std::fs::read(&tfjs.model_json_path)
        .map_err(|e| convert_read_err(tfjs.model_json_path.display(), e))?;
    let limits = ConvertLimits::default();
    let manifest = parse_tfjs_manifest_with_limits(&json_bytes, &limits)?;

    // The re-read MUST match the route's snapshot ELEMENT-WISE: a
    // same-cardinality reorder (re-uploaded `model.json` between route parse
    // and worker re-read, which cross-tab uploads can interleave past the
    // single-job gate) would zip declared names against resolved paths in
    // the wrong order = silently corrupt head.
    if manifest.shards.len() != tfjs.shard_names.len()
        || manifest.shards.len() != tfjs.shard_paths.len()
    {
        return Err(ConvertError::TfjsParse {
            what: "shards",
            msg: format!(
                "model.json shard count changed mid-job: route saw {}, worker sees {}",
                tfjs.shard_paths.len(),
                manifest.shards.len(),
            ),
        });
    }
    if manifest.shards != tfjs.shard_names {
        return Err(ConvertError::TfjsParse {
            what: "shards",
            msg: format!(
                "model.json shard ORDER / names changed mid-job: route snapshot {:?}, \
                 worker re-read {:?} (likely re-uploaded mid-job)",
                tfjs.shard_names, manifest.shards,
            ),
        });
    }

    // Stage shards into a per-job `.tmp/` dir so the streaming reader
    // resolves the manifest's sibling-relative paths.
    begin_stage(log, handle, stage, ConvertStage::StageShards);
    let tmp_root = workspace_dir.join(".tmp");
    std::fs::create_dir_all(&tmp_root).map_err(|e| convert_write_err(tmp_root.display(), e))?;
    let staging_dir = convert_staging_dir(workspace_dir, job.job_id);
    std::fs::create_dir_all(&staging_dir)
        .map_err(|e| convert_write_err(staging_dir.display(), e))?;
    // `model.json` via `put_atomic` (1 MiB bound); shards hard-linked below.
    let model_json_staged = staging_dir.join("model.json");
    files
        .put_atomic(&model_json_staged, &json_bytes)
        .map_err(|e| {
            convert_write_err(
                model_json_staged.display(),
                std::io::Error::other(e.to_string()),
            )
        })?;
    for (declared, src) in manifest.shards.iter().zip(tfjs.shard_paths.iter()) {
        let dst = staging_dir.join(declared);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(|e| convert_write_err(parent.display(), e))?;
        }
        // Hard-link to skip the per-shard heap copy; `fs::copy` fallback if
        // links rejected. NotFound on the source is a Read.
        if let Err(e) = std::fs::hard_link(src, &dst) {
            if e.kind() == std::io::ErrorKind::NotFound {
                return Err(convert_read_err(src.display(), e));
            }
            std::fs::copy(src, &dst).map_err(|source| convert_read_err(src.display(), source))?;
        }
    }

    begin_stage(log, handle, stage, ConvertStage::ExtractWeights);
    let (k_entry, b_entry) = pick_tfjs_head_entries(&manifest)?;
    // Source-bundle sha not persisted; `None` skips the hasher.
    let (kernel_bytes, bias_bytes) =
        source::read_head_bytes_streaming(&staging_dir, &manifest, k_entry, b_entry, None)?;
    let weights = head_weights_from_head_byte_ranges(
        &manifest,
        k_entry,
        b_entry,
        &kernel_bytes,
        &bias_bytes,
    )?;
    // Coerce to u32 once and reuse across every event + the manifest so the
    // wire value stays identical. `try_from` always succeeds under the cap;
    // `BadClassCount` is defensive against future loosening.
    let n_classes_u32 =
        u32::try_from(weights.n_classes).map_err(|_| ConvertError::BadClassCount {
            got: weights.n_classes,
            max: limits.max_n_classes,
        })?;
    let in_dim_u32 = u32::try_from(weights.in_dim).unwrap_or(u32::MAX);
    emit_convert_event(
        log,
        handle,
        stage,
        ConvertEvent::WeightsExtracted {
            n_classes: n_classes_u32,
            in_dim: in_dim_u32,
        },
    );

    begin_stage(log, handle, stage, ConvertStage::ReadLabels);
    let labels = read_labels_from_path(&tfjs.labels_path, tfjs.labels_format, weights.n_classes)?;
    let n_labels = u32::try_from(labels.len()).unwrap_or(u32::MAX);
    emit_convert_event(log, handle, stage, ConvertEvent::LabelsLoaded { n_labels });

    begin_stage(log, handle, stage, ConvertStage::StageHeadMpk);
    // Build the ACSTHEAD `.mpk` in memory, stage under `.tmp/` for the
    // rotation primitive's intra-FS atomic rename into `heads/`.
    let head_blob = build_head_mpk_blob(&weights)?;
    let head_sha256 = hex_lowercase(&Sha256::digest(&head_blob));
    let head_size = head_blob.len() as u64;
    let mpk_tempfile = convert_mpk_tempfile(workspace_dir, job.job_id, job.head_id);
    files.put_atomic(&mpk_tempfile, &head_blob).map_err(|e| {
        convert_write_err(mpk_tempfile.display(), std::io::Error::other(e.to_string()))
    })?;

    // Heads carry only sha256 / n_classes / size_bytes / labels /
    // workspace_revision; the JSONL log is the durable input record.
    let manifest_struct = crate::common::workspace::HeadManifest {
        head_id: job.head_id,
        workspace_id: job.workspace_id,
        workspace_revision: job.workspace_revision.clone(),
        sha256: head_sha256.clone(),
        n_classes: n_classes_u32,
        size_bytes: head_size,
        created_at: crate::file_mgr::now_rfc3339(),
        labels: labels.clone(),
    };
    let pending = crate::file_mgr::PendingHead {
        head_id: job.head_id,
        mpk_tempfile: mpk_tempfile.clone(),
        manifest: manifest_struct,
    };

    begin_stage(log, handle, stage, ConvertStage::PublishHead);
    files
        .publish_trained_head(&job.workspace_id, pending)
        .map_err(|e| {
            convert_write_err(
                workspace_dir.display(),
                std::io::Error::other(e.to_string()),
            )
        })?;
    emit_convert_event(
        log,
        handle,
        stage,
        ConvertEvent::HeadPublished {
            head_id: job.head_id,
            head_sha256: head_sha256.clone(),
            size_bytes: head_size,
            n_classes: n_classes_u32,
            classes: labels.clone(),
            workspace_revision: job.workspace_revision.clone(),
            idempotent_skip: false,
        },
    );

    // Tempfile cleanup is unconditional in `run_convert_job`.
    Ok(ConvertResult {
        head_id: job.head_id,
        head_sha256,
        n_classes: n_classes_u32,
        classes: labels,
    })
}

/// Read labels per `labels_format`, then cross-validate count vs
/// `expected_n_classes` (the head kernel's second dim).
fn read_labels_from_path(
    labels_path: &Path,
    labels_format: crate::file_mgr::LabelsFormat,
    expected_n_classes: usize,
) -> Result<Vec<String>, ConvertError> {
    let labels = match labels_format {
        crate::file_mgr::LabelsFormat::Lines => {
            let bytes = read_capped_labels(labels_path, "labels.txt")?;
            let text = String::from_utf8(bytes)
                .map_err(|_| ConvertError::Labels("labels.txt: not valid UTF-8".to_string()))?;
            text.lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        }
        crate::file_mgr::LabelsFormat::TfjsMetadata => read_tfjs_labels(labels_path)?,
    };
    // Validate CONTENT before the count check: control chars corrupt the
    // `\n`-joined labels.txt round-trip; empties inflate n_classes then get
    // dropped by the loader => late LabelCountMismatch at activation.
    // Surfaces here as UserInput (400) at the owning boundary.
    validate_convert_labels(&labels, ConvertError::Labels)?;
    if labels.len() != expected_n_classes {
        return Err(ConvertError::Labels(format!(
            "labels source declares {} entries; head kernel declares {} classes",
            labels.len(),
            expected_n_classes,
        )));
    }
    Ok(labels)
}

/// Build the ACSTHEAD-wrapped `.mpk` payload as one in-memory blob (callers
/// pair it with [`FsService::put_atomic`]).
fn build_head_mpk_blob(weights: &HeadWeights) -> Result<Vec<u8>, ConvertError> {
    validate_head_class_count(weights.n_classes, &ConvertLimits::default())?;
    if weights.in_dim != BACKBONE_FEATURE_DIM {
        return Err(ConvertError::Tensor(format!(
            "kernel in_dim {} != BACKBONE_FEATURE_DIM {BACKBONE_FEATURE_DIM}",
            weights.in_dim
        )));
    }
    let device: burn::tensor::Device<B> = Default::default();
    let mut head = Head::<B>::try_new(weights.n_classes, &device).map_err(|e| match e {
        model::Error::BadClassCount { got, max } => ConvertError::BadClassCount { got, max },
        other => ConvertError::Tensor(format!("head construct: {other}")),
    })?;
    let kernel_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(weights.kernel.clone(), [weights.in_dim, weights.n_classes]),
        &device,
    );
    let bias_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(weights.bias.clone(), [weights.n_classes]),
        &device,
    );
    head.linear.weight = Param::from_tensor(kernel_tensor);
    head.linear.bias = Some(Param::from_tensor(bias_tensor));
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::new();
    let payload = recorder
        .record(head.into_record(), ())
        .map_err(|e| ConvertError::Record(format!("{e}")))?;
    let header = crate::common::head_header::serialize_header(
        weights.in_dim as u32,
        weights.n_classes as u32,
        payload.len() as u32,
    );
    let mut blob: Vec<u8> = Vec::with_capacity(header.len() + payload.len());
    blob.extend_from_slice(&header);
    blob.extend_from_slice(&payload);
    Ok(blob)
}

/// Validate `n_classes` against `1..=max_n_classes`, mirroring
/// `Head::try_new` so the operator gets a structured `ConvertError`, not an
/// Internal `model::Error`.
fn validate_head_class_count(n_classes: usize, limits: &ConvertLimits) -> Result<(), ConvertError> {
    if n_classes == 0 || n_classes > limits.max_n_classes {
        return Err(ConvertError::BadClassCount {
            got: n_classes,
            max: limits.max_n_classes,
        });
    }
    Ok(())
}

/// Shape-sanity gate for a TFJS kernel + bias pair; returns derived
/// `n_classes` (the kernel's second dim).
fn check_head_entry_shapes(
    k_entry: &TfjsManifestEntry,
    b_entry: &TfjsManifestEntry,
) -> Result<usize, ConvertError> {
    if k_entry.shape.len() != 2 || k_entry.shape[0] != BACKBONE_FEATURE_DIM {
        return Err(ConvertError::TfjsLocator(format!(
            "head kernel `{}` has shape {:?}; expected 2-D [{BACKBONE_FEATURE_DIM}, N]",
            k_entry.name, k_entry.shape
        )));
    }
    let n_classes = k_entry.shape[1];
    if b_entry.shape != [n_classes] {
        return Err(ConvertError::TfjsLocator(format!(
            "head bias `{}` has shape {:?}; expected [{n_classes}]",
            b_entry.name, b_entry.shape
        )));
    }
    Ok(n_classes)
}

/// Reject TFJS shard `paths` that could escape the model dir: only strictly
/// relative, all-`Normal` components. Backslashes (cross-platform
/// ambiguity) and NUL bytes (path-as-key safety) also rejected.
fn validate_shard_path(s: &str) -> Result<(), ConvertError> {
    if s.is_empty() || s.contains('\\') || s.contains('\0') {
        return Err(ConvertError::TfjsUnsafePath(s.to_string()));
    }
    let p = std::path::Path::new(s);
    if p.is_absolute() {
        return Err(ConvertError::TfjsUnsafePath(s.to_string()));
    }
    for comp in p.components() {
        match comp {
            std::path::Component::Normal(_) => {}
            // `.`, `..`, root, prefix all rejected.
            _ => return Err(ConvertError::TfjsUnsafePath(s.to_string())),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // Scaffolding writes via `std::fs::write`, exempt from the clippy.toml
    // atomic-writer constraint.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use crate::model::HeadRecord;
    use burn::record::NamedMpkFileRecorder;

    fn crate_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    /// Strip the ACSTHEAD header to a sibling tempfile (production goes
    /// through `inference::HotHead::load`, undepended here).
    fn strip_head_header(mpk: &Path, dir: &Path) -> PathBuf {
        let bytes = std::fs::read(mpk).expect("read mpk");
        let payload = &bytes[crate::common::head_header::HEAD_HEADER_SIZE..];
        let stripped = dir.join("payload_only.mpk");
        std::fs::write(&stripped, payload).expect("write stripped");
        stripped
    }

    #[test]
    fn convert_tfjs_writes_artifacts() {
        let dir = crate_root().join("misc/models");
        if !dir.join("model.json").exists() {
            return;
        }
        let labels = read_tfjs_labels(&dir.join("metadata.json")).expect("labels");
        let dst = tempfile::tempdir().expect("tempdir");
        // Throwaway `FsServiceImpl` rooted anywhere (converter only uses the
        // workspace-agnostic `put_atomic`).
        let fs_root = tempfile::tempdir().expect("fs root");
        let fs = crate::file_mgr::FsServiceImpl::new(fs_root.path().to_path_buf());
        let arts = convert_tfjs(&dir, &labels, dst.path(), &fs).expect("convert");
        assert!(arts.head_mpk.exists());
        assert!(arts.labels_txt.exists());
        assert!(arts.metadata_json.exists());
        assert_eq!(arts.n_classes, labels.len());
        assert_eq!(arts.source_sha256.len(), 64);

        // No partial siblings linger; dst_dir holds exactly the triple.
        let mut entries: Vec<String> = std::fs::read_dir(dst.path())
            .expect("read dst")
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        entries.sort();
        assert_eq!(
            entries,
            vec![
                "head.mpk".to_string(),
                "labels.txt".to_string(),
                "metadata.json".to_string(),
            ],
            "dst dir should contain exactly the published triple",
        );

        let meta_bytes = std::fs::read(&arts.metadata_json).unwrap();
        let meta: ConversionMetadata = serde_json::from_slice(&meta_bytes).unwrap();
        assert_eq!(meta.source_kind, SourceKind::Tfjs);
        assert_eq!(meta.n_classes, labels.len());
        assert_eq!(meta.labels, labels);

        // Strip the ACSTHEAD header before Burn's recorder.
        let payload_only = strip_head_header(&arts.head_mpk, dst.path());
        let device: burn::tensor::Device<B> = Default::default();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let head_rec: HeadRecord<B> = recorder
            .load(payload_only.clone(), &device)
            .expect("load head.mpk");
        let head = Head::<B>::new(1, &device).load_record(head_rec);
        assert_eq!(
            head.linear.weight.val().dims(),
            [BACKBONE_FEATURE_DIM, labels.len()]
        );
    }

    /// Labels and head agree on class count for the upstream bundle
    /// (multi-shard read + `words`-schema fallback); skipped if absent.
    #[test]
    fn tfjs_labels_match_head_models_dir() {
        let dir = crate_root().join("misc/models");
        if !dir.join("model.json").exists() {
            return;
        }
        let labels = read_tfjs_labels(&dir.join("metadata.json")).expect("labels");
        let weights = extract_head_from_tfjs_dir(&dir).expect("weights");
        assert_eq!(labels.len(), weights.n_classes);
    }

    /// `read_tfjs_labels` accepts both `wordLabels` and `words`,
    /// `wordLabels` winning when both present.
    #[test]
    fn read_tfjs_labels_accepts_words_and_word_labels() {
        let dir = tempfile::tempdir().expect("tempdir");

        let tm_path = dir.path().join("tm.json");
        std::fs::write(
            &tm_path,
            br#"{"wordLabels":["a","b","c"],"modelName":"tm"}"#,
        )
        .unwrap();
        assert_eq!(read_tfjs_labels(&tm_path).unwrap(), ["a", "b", "c"]);

        let sc_path = dir.path().join("sc.json");
        std::fs::write(&sc_path, br#"{"words":["go","stop"],"frameSize":232}"#).unwrap();
        assert_eq!(read_tfjs_labels(&sc_path).unwrap(), ["go", "stop"]);

        // Both keys present: `wordLabels` wins.
        let both_path = dir.path().join("both.json");
        std::fs::write(&both_path, br#"{"wordLabels":["x"],"words":["y","z"]}"#).unwrap();
        assert_eq!(read_tfjs_labels(&both_path).unwrap(), ["x"]);
    }

    /// Neither key present: `Labels` error names both schemas.
    #[test]
    fn read_tfjs_labels_errors_when_neither_key_present() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("missing.json");
        std::fs::write(&p, br#"{"otherField":42}"#).unwrap();
        let err = read_tfjs_labels(&p).unwrap_err();
        let msg = format!("{err}");
        assert!(matches!(err, ConvertError::Labels(_)), "{err:?}");
        assert!(
            msg.contains("wordLabels") && msg.contains("words"),
            "diagnostic must list both schemas: {msg}",
        );
    }

    /// Empty array errors, naming the detected key.
    #[test]
    fn read_tfjs_labels_errors_on_empty_array() {
        let dir = tempfile::tempdir().expect("tempdir");

        let tm_path = dir.path().join("tm_empty.json");
        std::fs::write(&tm_path, br#"{"wordLabels":[]}"#).unwrap();
        let err = read_tfjs_labels(&tm_path).unwrap_err();
        assert!(format!("{err}").contains("wordLabels"));

        let sc_path = dir.path().join("sc_empty.json");
        std::fs::write(&sc_path, br#"{"words":[]}"#).unwrap();
        let err = read_tfjs_labels(&sc_path).unwrap_err();
        assert!(format!("{err}").contains("words"));
    }

    /// Extraction smoke test on the upstream bundle (multi-shard read +
    /// shape-based head match); skipped if absent.
    #[test]
    fn extract_from_shipped_tfjs_models_dir() {
        let dir = crate_root().join("misc/models");
        if !dir.join("model.json").exists() {
            eprintln!("skipping: {} not present", dir.display());
            return;
        }
        let weights = extract_head_from_tfjs_dir(&dir).expect("extract");
        assert_eq!(weights.in_dim, BACKBONE_FEATURE_DIM);
        assert_eq!(weights.n_classes, 20);
        assert_eq!(weights.kernel.len(), BACKBONE_FEATURE_DIM * 20);
        assert_eq!(weights.bias.len(), 20);
        let nonzero = weights.kernel.iter().filter(|x| **x != 0.0).count();
        assert!(
            nonzero > weights.kernel.len() / 100,
            "implausibly sparse kernel: {nonzero}/{}",
            weights.kernel.len()
        );
    }

    #[test]
    fn tfjs_invalid_json_errors_cleanly() {
        let err = parse_tfjs_manifest(b"{not json").unwrap_err();
        assert!(matches!(err, ConvertError::TfjsParse { .. }));
    }

    #[test]
    fn tfjs_validate_shard_path_rejects_unsafe() {
        assert!(validate_shard_path("weights.bin").is_ok());
        assert!(validate_shard_path("group1-shard1of2.bin").is_ok());
        assert!(validate_shard_path("sub/dir/file.bin").is_ok());

        for bad in [
            "",
            "../escape",
            "../../etc/passwd",
            "/abs/path",
            "a\\b",
            ".",
            "weights\0.bin",
            "\0",
        ] {
            let r = validate_shard_path(bad);
            assert!(
                matches!(r, Err(ConvertError::TfjsUnsafePath(_))),
                "expected unsafe-path rejection for {bad:?}, got {r:?}",
            );
        }
    }

    #[test]
    fn tfjs_manifest_path_traversal_rejected() {
        let json = br#"{
          "weightsManifest": [{
            "paths": ["../etc/passwd"],
            "weights": [
              {"name": "k", "shape": [2000, 3], "dtype": "float32"},
              {"name": "b", "shape": [3], "dtype": "float32"}
            ]
          }]
        }"#;
        let err = parse_tfjs_manifest(json).unwrap_err();
        assert!(matches!(err, ConvertError::TfjsUnsafePath(_)), "{err:?}");
    }

    /// A shape that would overflow `usize::product()` is caught, not UB.
    #[test]
    fn tfjs_manifest_shape_overflow_rejected() {
        let huge = u64::MAX / 2; // product of two will overflow usize
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "huge", "shape": [{huge}, {huge}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let err = parse_tfjs_manifest(json.as_bytes()).unwrap_err();
        assert!(
            matches!(err, ConvertError::TfjsShapeOverflow { .. }),
            "{err:?}"
        );
    }

    /// `n_classes > max_n_classes` rejected before allocation; uses a
    /// usize-fitting count so the overflow arm can't fire.
    #[test]
    fn tfjs_huge_n_classes_rejected_before_allocation() {
        let huge_n = ConvertLimits::default().max_n_classes + 1;
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "k", "shape": [{BACKBONE_FEATURE_DIM}, {huge_n}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let err = parse_tfjs_manifest(json.as_bytes()).unwrap_err();
        match err {
            ConvertError::LimitExceeded { what, value, max } => {
                assert_eq!(what, "n_classes", "{what:?}");
                assert_eq!(value, huge_n as u64);
                assert_eq!(max, ConvertLimits::default().max_n_classes as u64);
            }
            other => panic!("expected LimitExceeded n_classes, got {other:?}"),
        }
    }

    /// 1-D tensor over `max_n_classes` fires the `1d_tensor_dim` cap (not
    /// `n_classes`: it may be a non-head bias); pins the wire label.
    #[test]
    fn tfjs_one_dim_cap_uses_1d_tensor_dim_label() {
        // Above the dim cap but under the byte cap.
        let huge_n = ConvertLimits::default().max_n_classes + 1;
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "b", "shape": [{huge_n}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let err = parse_tfjs_manifest(json.as_bytes()).unwrap_err();
        match err {
            ConvertError::LimitExceeded { what, value, max } => {
                assert_eq!(
                    what, "1d_tensor_dim",
                    "the 1-D-tensor cap label is `1d_tensor_dim`, not \
                     `n_classes`; relabel must not regress"
                );
                assert_eq!(value, huge_n as u64);
                assert_eq!(max, ConvertLimits::default().max_n_classes as u64);
            }
            other => panic!("expected LimitExceeded 1d_tensor_dim, got {other:?}"),
        }
    }

    /// Over-cap `size_bytes` rejected via typed `LimitExceeded { what:
    /// "alpkg_size_bytes" }`, not the catch-all schema string; pins the
    /// wire label.
    #[test]
    fn validate_alpkg_head_manifest_rejects_oversize() {
        use crate::common::ids::{HeadId, WorkspaceId};
        use crate::common::workspace::{HeadManifest, WorkspaceRevision};
        let head_id = HeadId::new();
        let manifest = HeadManifest {
            head_id,
            workspace_id: WorkspaceId::new(),
            workspace_revision: WorkspaceRevision {
                id: 1,
                at: "2026-01-01T00:00:00Z".to_string(),
            },
            // 64 hex chars to pass the sha256 shape gate.
            sha256: "0".repeat(64),
            n_classes: 2,
            size_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB > 1 GiB cap.
            created_at: "2026-01-01T00:00:00Z".to_string(),
            labels: vec!["cat".to_string(), "dog".to_string()],
        };
        let err = validate_alpkg_head_manifest(&manifest, head_id).unwrap_err();
        match err {
            ConvertError::LimitExceeded { what, value, max } => {
                assert_eq!(
                    what, "alpkg_size_bytes",
                    "the size_bytes cap is typed LimitExceeded, not \
                     AlpkgManifestSchema; retype must not regress"
                );
                assert_eq!(value, 2 * 1024 * 1024 * 1024);
                assert_eq!(max, 1024 * 1024 * 1024);
            }
            other => panic!("expected LimitExceeded alpkg_size_bytes, got {other:?}"),
        }
    }

    /// A control char in any label is rejected at the import boundary
    /// (`AlpkgManifestSchema` -> 400), mirroring the runtime gate.
    #[test]
    fn validate_alpkg_head_manifest_rejects_label_control_char() {
        use crate::common::ids::{HeadId, WorkspaceId};
        use crate::common::workspace::{HeadManifest, WorkspaceRevision};
        let head_id = HeadId::new();
        let mk = |labels: Vec<String>| HeadManifest {
            head_id,
            workspace_id: WorkspaceId::new(),
            workspace_revision: WorkspaceRevision {
                id: 1,
                at: "2026-01-01T00:00:00Z".to_string(),
            },
            sha256: "0".repeat(64),
            n_classes: 2,
            size_bytes: 1024,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            labels,
        };
        // Embedded newline (U+000A) corrupts the labels.txt round-trip.
        let bad = mk(vec!["good".to_string(), "bad\nlabel".to_string()]);
        match validate_alpkg_head_manifest(&bad, head_id).unwrap_err() {
            ConvertError::AlpkgManifestSchema { reason } => {
                assert!(
                    reason.contains("control character") && reason.contains("U+000A"),
                    "reason should name the control char + codepoint; got {reason:?}"
                );
            }
            other => panic!("expected AlpkgManifestSchema, got {other:?}"),
        }
        // A clean label set passes the gate (fails later at hash-verify).
        let ok = mk(vec!["good".to_string(), "fine".to_string()]);
        assert!(
            validate_alpkg_head_manifest(&ok, head_id).is_ok(),
            "clean labels must pass the import-time structural gate",
        );
    }

    /// Per-tensor byte cap fires when `len_bytes > max_kernel_bytes`
    /// even though the dim count passes the per-dimension cap.
    #[test]
    fn tfjs_per_tensor_byte_cap_rejected() {
        // 1 KiB cap so a 600-element f32 (2400 B) trips it.
        let limits = ConvertLimits {
            max_kernel_bytes: 1024,
            ..ConvertLimits::default()
        };
        let json = br#"{
          "weightsManifest": [{
            "paths": ["weights.bin"],
            "weights": [
              {"name": "k", "shape": [600], "dtype": "float32"}
            ]
          }]
        }"#;
        let err = parse_tfjs_manifest_with_limits(json, &limits).unwrap_err();
        match err {
            ConvertError::LimitExceeded { what, .. } => {
                assert_eq!(what, "tensor_bytes", "{what:?}");
            }
            other => panic!("expected LimitExceeded tensor_bytes, got {other:?}"),
        }
    }

    /// Manifest size cap is enforced before the JSON parser runs.
    #[test]
    fn tfjs_manifest_payload_cap_rejected() {
        let limits = ConvertLimits {
            max_manifest_bytes: 64,
            ..ConvertLimits::default()
        };
        // Cap fires before parsing, so any 128 bytes suffice.
        let json = vec![b' '; 128];
        let err = parse_tfjs_manifest_with_limits(&json, &limits).unwrap_err();
        match err {
            ConvertError::LimitExceeded { what, value, max } => {
                assert_eq!(what, "manifest_bytes", "{what:?}");
                assert_eq!(value, 128);
                assert_eq!(max, 64);
            }
            other => panic!("expected LimitExceeded manifest_bytes, got {other:?}"),
        }
    }

    /// Shard-count cap fires when `paths` exceeds `max_shards`.
    #[test]
    fn tfjs_shard_count_cap_rejected() {
        let limits = ConvertLimits {
            max_shards: 2,
            ..ConvertLimits::default()
        };
        let json = br#"{
          "weightsManifest": [{
            "paths": ["a.bin", "b.bin", "c.bin"],
            "weights": [
              {"name": "k", "shape": [2000, 3], "dtype": "float32"}
            ]
          }]
        }"#;
        let err = parse_tfjs_manifest_with_limits(json, &limits).unwrap_err();
        match err {
            ConvertError::LimitExceeded { what, .. } => {
                assert_eq!(what, "shards", "{what:?}");
            }
            other => panic!("expected LimitExceeded shards, got {other:?}"),
        }
    }

    /// Zero-dim kernel rejected with `TfjsZeroDimension`, preventing a
    /// downstream `Head::new(0, ...)` panic.
    #[test]
    fn tfjs_zero_dimension_kernel_rejected() {
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "head/kernel", "shape": [{BACKBONE_FEATURE_DIM}, 0], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let err = parse_tfjs_manifest(json.as_bytes()).unwrap_err();
        match err {
            ConvertError::TfjsZeroDimension { name, shape } => {
                assert_eq!(name, "head/kernel");
                assert_eq!(shape, vec![BACKBONE_FEATURE_DIM, 0]);
            }
            other => panic!("expected TfjsZeroDimension, got {other:?}"),
        }
    }

    /// Zero-dim bias `[0]` is similarly rejected.
    #[test]
    fn tfjs_zero_dimension_bias_rejected() {
        let json = br#"{
          "weightsManifest": [{
            "paths": ["weights.bin"],
            "weights": [
              {"name": "head/bias", "shape": [0], "dtype": "float32"}
            ]
          }]
        }"#;
        let err = parse_tfjs_manifest(json).unwrap_err();
        assert!(
            matches!(err, ConvertError::TfjsZeroDimension { .. }),
            "{err:?}"
        );
    }

    /// `write_head_artifacts` rejects `n_classes = 0` at the public boundary
    /// with a hand-built `HeadWeights`.
    #[test]
    fn write_head_artifacts_rejects_zero_classes() {
        let weights = HeadWeights {
            kernel: vec![],
            bias: vec![],
            n_classes: 0,
            in_dim: BACKBONE_FEATURE_DIM,
        };
        let dst = tempfile::tempdir().expect("tempdir");
        let fs_root = tempfile::tempdir().expect("fs root");
        let fs = crate::file_mgr::FsServiceImpl::new(fs_root.path().to_path_buf());
        let err = write_head_artifacts(
            &weights,
            &[],
            dst.path(),
            SourceKind::Tfjs,
            "deadbeef".to_string(),
            &fs,
        )
        .unwrap_err();
        match err {
            ConvertError::BadClassCount { got, max } => {
                assert_eq!(got, 0);
                assert_eq!(max, ConvertLimits::default().max_n_classes);
            }
            other => panic!("expected BadClassCount, got {other:?}"),
        }
    }

    /// NaN / +-Inf in the TFJS source rejected at the converter boundary
    /// with `NonFiniteWeight` (tensor + index).
    #[test]
    fn tfjs_non_finite_kernel_rejected() {
        let n = 3usize;
        let kernel_count = BACKBONE_FEATURE_DIM * n;
        let bias_count = n;
        let mut blob = Vec::<u8>::with_capacity((kernel_count + bias_count) * 4);
        // Finite kernel, poison index 17 with NaN.
        for i in 0..kernel_count {
            let v = if i == 17 { f32::NAN } else { 0.5_f32 };
            blob.extend_from_slice(&v.to_le_bytes());
        }
        for _ in 0..bias_count {
            blob.extend_from_slice(&0.25_f32.to_le_bytes());
        }
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "head/kernel", "shape": [{BACKBONE_FEATURE_DIM}, {n}], "dtype": "float32"}},
                  {{"name": "head/bias", "shape": [{n}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let manifest = parse_tfjs_manifest(json.as_bytes()).unwrap();
        let err = extract_head_from_tfjs_buffers(&manifest, &blob).unwrap_err();
        match err {
            ConvertError::NonFiniteWeight {
                tensor,
                index,
                value,
            } => {
                assert_eq!(tensor, "head/kernel");
                assert_eq!(index, 17);
                assert!(value.is_nan(), "value should be NaN, got {value}");
            }
            other => panic!("expected NonFiniteWeight, got {other:?}"),
        }
    }

    /// +Inf in the bias is similarly caught (tensor name + index).
    #[test]
    fn tfjs_non_finite_bias_rejected() {
        let n = 4usize;
        let kernel_count = BACKBONE_FEATURE_DIM * n;
        let mut blob = Vec::<u8>::with_capacity((kernel_count + n) * 4);
        for _ in 0..kernel_count {
            blob.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        // Poison bias index 2 with +Inf.
        for i in 0..n {
            let v = if i == 2 { f32::INFINITY } else { 0.0_f32 };
            blob.extend_from_slice(&v.to_le_bytes());
        }
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "head/kernel", "shape": [{BACKBONE_FEATURE_DIM}, {n}], "dtype": "float32"}},
                  {{"name": "head/bias", "shape": [{n}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let manifest = parse_tfjs_manifest(json.as_bytes()).unwrap();
        let err = extract_head_from_tfjs_buffers(&manifest, &blob).unwrap_err();
        match err {
            ConvertError::NonFiniteWeight {
                tensor,
                index,
                value,
            } => {
                assert_eq!(tensor, "head/bias");
                assert_eq!(index, 2);
                assert!(value.is_infinite() && value.is_sign_positive());
            }
            other => panic!("expected NonFiniteWeight on bias, got {other:?}"),
        }
    }

    /// Truncated blob fails with `TfjsBlobLength`, not a panic (length check
    /// fires before the head locator).
    #[test]
    fn tfjs_truncated_blob_rejected() {
        let json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "head/kernel", "shape": [{BACKBONE_FEATURE_DIM}, 3], "dtype": "float32"}},
                  {{"name": "head/bias", "shape": [3], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        let manifest = parse_tfjs_manifest(json.as_bytes()).unwrap();
        let truncated = vec![0u8; 32];
        let err = extract_head_from_tfjs_buffers(&manifest, &truncated).unwrap_err();
        assert!(
            matches!(err, ConvertError::TfjsBlobLength { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn tfjs_unsupported_dtype_rejected() {
        let json = br#"{
          "weightsManifest": [{
            "paths": ["weights.bin"],
            "weights": [
              {"name": "k", "shape": [2000, 3], "dtype": "int32"}
            ]
          }]
        }"#;
        let err = parse_tfjs_manifest(json).unwrap_err();
        assert!(matches!(err, ConvertError::TfjsDtype { .. }), "{err:?}");
    }

    /// `Categorized::kind` classifies every variant.
    #[test]
    fn convert_error_classification() {
        use crate::common::error::{Categorized, ErrorKind};
        use std::io;

        fn assert_kind(err: ConvertError, expected: ErrorKind) {
            assert_eq!(err.kind(), expected, "{err:?}");
        }

        assert_kind(ConvertError::Labels("x".into()), ErrorKind::UserInput);
        assert_kind(
            ConvertError::TfjsParse {
                what: "x",
                msg: "y".into(),
            },
            ErrorKind::UserInput,
        );
        assert_kind(ConvertError::TfjsLocator("x".into()), ErrorKind::UserInput);
        assert_kind(
            ConvertError::TfjsShortRead { have: 0, need: 1 },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::TfjsDtype {
                name: "x".into(),
                dtype: "y".into(),
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::TfjsUnsafePath("x".into()),
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::TfjsShapeOverflow {
                name: "x".into(),
                shape: vec![],
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::TfjsBlobLength {
                have: 0,
                declared: 1,
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::LimitExceeded {
                what: "n_classes",
                value: 1_000_000,
                max: 100_000,
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::TfjsZeroDimension {
                name: "x".into(),
                shape: vec![0],
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::NonFiniteWeight {
                tensor: "x".into(),
                index: 0,
                value: f32::NAN,
            },
            ErrorKind::UserInput,
        );
        assert_kind(
            ConvertError::BadClassCount { got: 0, max: 100 },
            ErrorKind::UserInput,
        );

        assert_kind(
            ConvertError::Read {
                path: "x".into(),
                source: io::Error::other("y"),
            },
            ErrorKind::Internal,
        );
        assert_kind(
            ConvertError::Write {
                path: "x".into(),
                source: io::Error::other("y"),
            },
            ErrorKind::Internal,
        );
        assert_kind(ConvertError::Record("x".into()), ErrorKind::Internal);
        assert_kind(ConvertError::Tensor("x".into()), ErrorKind::Internal);

        assert_kind(ConvertError::NotImplemented("x"), ErrorKind::NotImplemented);

        // `#[from]`-only variants, via their conversions.
        let serde_err: serde_json::Error =
            serde_json::from_slice::<serde_json::Value>(b"{").unwrap_err();
        assert_kind(ConvertError::from(serde_err), ErrorKind::Internal);
    }

    /// Each [`ConvertError`] lifts to the documented [`ConvertFailPayload`]
    /// category; pins the frontend hint-copy mapping against a future variant
    /// addition.
    #[test]
    fn fail_payload_lifts_each_convert_error_variant() {
        fn assert_category(err: ConvertError, expected: &str) {
            let payload = fail_payload_from_convert_error(&err);
            let v = serde_json::to_value(&payload).expect("payload serializes");
            assert_eq!(
                v["category"].as_str(),
                Some(expected),
                "{err:?} -> {payload:?}",
            );
        }

        assert_category(
            ConvertError::LimitExceeded {
                what: "n_classes",
                value: 99,
                max: 10,
            },
            "limit_exceeded",
        );
        assert_category(
            ConvertError::BadClassCount { got: 0, max: 10 },
            "bad_class_count",
        );
        assert_category(ConvertError::Labels("bad".into()), "labels");
        assert_category(
            ConvertError::AlpkgManifestSchema { reason: "x".into() },
            "alpkg_manifest_schema",
        );
        assert_category(
            ConvertError::AlpkgSizeMismatch {
                expected: 1,
                observed: 2,
            },
            "alpkg_size_mismatch",
        );
        assert_category(
            ConvertError::AlpkgHashMismatch {
                expected: "a".into(),
                observed: "b".into(),
            },
            "alpkg_hash_mismatch",
        );
        assert_category(
            ConvertError::HeadIdCollision(crate::file_mgr::FileError::HeadIdCollision {
                head_id: crate::common::ids::HeadId::new().to_string(),
                got_sha256: "a".into(),
                stored_sha256: "b".into(),
            }),
            "head_id_collision",
        );

        // Source-malformed catch-all.
        assert_category(
            ConvertError::TfjsParse {
                what: "x",
                msg: "y".into(),
            },
            "source_malformed",
        );
        assert_category(ConvertError::TfjsLocator("x".into()), "source_malformed");
        assert_category(
            ConvertError::TfjsShortRead { have: 0, need: 1 },
            "source_malformed",
        );
        assert_category(
            ConvertError::TfjsUnsafePath("..".into()),
            "source_malformed",
        );
        assert_category(
            ConvertError::TfjsZeroDimension {
                name: "k".into(),
                shape: vec![0],
            },
            "source_malformed",
        );
        assert_category(
            ConvertError::NonFiniteWeight {
                tensor: "k".into(),
                index: 0,
                value: f32::NAN,
            },
            "source_malformed",
        );

        // Internal catch-all.
        assert_category(
            ConvertError::Read {
                path: "/tmp/x".into(),
                source: std::io::Error::other("io"),
            },
            "internal",
        );
        assert_category(
            ConvertError::Write {
                path: "/tmp/x".into(),
                source: std::io::Error::other("io"),
            },
            "internal",
        );
        assert_category(ConvertError::Record("x".into()), "internal");
        assert_category(ConvertError::Tensor("x".into()), "internal");
        assert_category(ConvertError::Busy, "internal");
        assert_category(ConvertError::NotImplemented("x"), "internal");
        assert_category(
            ConvertError::AlpkgManifestRead {
                path: "/tmp/x".into(),
                source: std::io::Error::other("io"),
            },
            "internal",
        );
        let serde_err: serde_json::Error =
            serde_json::from_slice::<serde_json::Value>(b"{").unwrap_err();
        assert_category(ConvertError::from(serde_err), "internal");
    }

    /// Serializes tests touching the global [`CONVERT_SEMAPHORE`] so
    /// permit-acquiring cases don't collide under the parallel runner.
    static CONVERT_TEST_SERIALIZER: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    fn serialize_convert_test() -> parking_lot::MutexGuard<'static, ()> {
        CONVERT_TEST_SERIALIZER.lock()
    }

    /// Concurrency cap: first acquire succeeds, a second returns `Busy` (409)
    /// while held, and dropping frees the next.
    #[test]
    fn convert_permit_caps_at_one_in_flight() {
        let _gate = serialize_convert_test();
        let p1 = acquire_convert_permit().expect("first acquire");
        let err = acquire_convert_permit()
            .expect_err("second acquire must reject while first permit held");
        assert!(
            matches!(err, ConvertError::Busy),
            "expected ConvertError::Busy, got {err:?}",
        );

        use crate::common::error::{Categorized, ErrorKind};
        assert_eq!(err.kind(), ErrorKind::Conflict);

        drop(p1);
        let p2 = acquire_convert_permit().expect("acquire after drop must succeed");
        drop(p2);
    }

    // MARK: convert pipeline -- unit tests

    use crate::common::ids::{HeadId, JobId, WorkspaceId};
    use crate::common::workspace::WorkspaceRevision;

    /// Stage a workspace so the publish primitive can land a head.
    fn stage_test_workspace(
        root: &Path,
    ) -> (WorkspaceId, std::sync::Arc<dyn crate::file_mgr::FsService>) {
        let fs = std::sync::Arc::new(crate::file_mgr::FsServiceImpl::new(root.to_path_buf()));
        fs.ensure_root_layout().expect("layout");
        let id = fs.create("convert-test").expect("create workspace");
        (id, fs as std::sync::Arc<dyn crate::file_mgr::FsService>)
    }

    /// Build a minimal TFJS bundle (one `model.json` + one `weights.bin`)
    /// and return the staged job-input paths.
    fn stage_minimal_tfjs(
        workspace_dir: &Path,
        n_classes: usize,
    ) -> (PathBuf, Vec<PathBuf>, PathBuf) {
        let datasets_dir = workspace_dir.join("converters/tfjs");
        std::fs::create_dir_all(&datasets_dir).unwrap();

        let model_json = datasets_dir.join("model.json");
        let manifest_json = format!(
            r#"{{
              "weightsManifest": [{{
                "paths": ["weights.bin"],
                "weights": [
                  {{"name": "NewHeadDense/kernel", "shape": [{BACKBONE_FEATURE_DIM}, {n_classes}], "dtype": "float32"}},
                  {{"name": "NewHeadDense/bias", "shape": [{n_classes}], "dtype": "float32"}}
                ]
              }}]
            }}"#
        );
        std::fs::write(&model_json, manifest_json.as_bytes()).unwrap();

        let kernel_count = BACKBONE_FEATURE_DIM * n_classes;
        let mut weights_blob = Vec::with_capacity((kernel_count + n_classes) * 4);
        for _ in 0..kernel_count {
            weights_blob.extend_from_slice(&0.5_f32.to_le_bytes());
        }
        for _ in 0..n_classes {
            weights_blob.extend_from_slice(&0.25_f32.to_le_bytes());
        }
        let shard = datasets_dir.join("weights.bin");
        std::fs::write(&shard, &weights_blob).unwrap();

        let labels = datasets_dir.join("labels.txt");
        let labels_text = (0..n_classes)
            .map(|i| format!("class_{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&labels, &labels_text).unwrap();

        (model_json, vec![shard], labels)
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    /// `run_convert_job` on a minimal TFJS bundle publishes the head
    /// into the rotation and the `heads.json` index reflects it.
    #[test]
    fn convert_with_minimal_tfjs_fixture_publishes_head() {
        let _gate = serialize_convert_test();

        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 3);

        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(0),
            kind: ConvertJobKind::Tfjs(ConvertJobTfjs {
                model_json_path: model_json,
                shard_paths: shards,
                // Route derives this name from the parsed manifest.
                shard_names: vec!["weights.bin".to_string()],
                labels_path: labels,
                labels_format: crate::file_mgr::LabelsFormat::Lines,
            }),
        };

        // `run_convert_job` does not re-acquire the permit.
        run_convert_job(fs.clone(), job, None).expect("convert publishes");

        let summary = fs.summary(&ws).expect("summary");
        assert_eq!(summary.heads.heads.len(), 1, "exactly one head");
        let h = &summary.heads.heads[0];
        assert_eq!(h.head_id, head_id);
        assert_eq!(h.n_classes, 3);
        let mpk = crate::file_mgr::schema::head_artifact_path(&workspace_dir, head_id);
        assert!(mpk.is_file(), "head .mpk missing at {}", mpk.display());
        let manifest = crate::file_mgr::schema::read_head_manifest(&workspace_dir, head_id)
            .expect("read manifest");
        assert_eq!(manifest.head_id, head_id);
        assert_eq!(manifest.workspace_id, ws);
        assert_eq!(manifest.workspace_revision.id, 0);
        assert_eq!(manifest.labels.len(), 3);

        // Wrapper sweep clears the staging dir + `.mpk` tempfile.
        let staging_dir = workspace_dir.join(".tmp").join(format!("convert-{job_id}"));
        let mpk_tempfile = workspace_dir
            .join(".tmp")
            .join(format!("convert-{job_id}-{head_id}.mpk"));
        assert!(
            !staging_dir.exists(),
            "staging dir {} must be swept on success",
            staging_dir.display(),
        );
        assert!(
            !mpk_tempfile.exists(),
            ".mpk tempfile {} must be swept on success",
            mpk_tempfile.display(),
        );
    }

    /// A failed `run_convert_job` (invalid manifest) commits no head
    /// record; `heads.json` stays empty and a `JobFailed` line lands.
    #[test]
    fn convert_failure_releases_references_and_no_head_committed() {
        let _gate = serialize_convert_test();

        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let datasets_dir = workspace_dir.join("datasets");
        std::fs::create_dir_all(&datasets_dir).unwrap();

        let bad_model = datasets_dir.join("model.json");
        std::fs::write(&bad_model, b"not json").unwrap();
        let bad_shard = datasets_dir.join("weights.bin");
        std::fs::write(&bad_shard, b"any").unwrap();
        let bad_labels = datasets_dir.join("labels.txt");
        std::fs::write(&bad_labels, b"x\n").unwrap();

        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(0),
            kind: ConvertJobKind::Tfjs(ConvertJobTfjs {
                model_json_path: bad_model,
                shard_paths: vec![bad_shard],
                // Parse fails before the shard-name check.
                shard_names: vec!["weights.bin".to_string()],
                labels_path: bad_labels,
                labels_format: crate::file_mgr::LabelsFormat::Lines,
            }),
        };
        let err = run_convert_job(fs.clone(), job, None).expect_err("invalid manifest must fail");
        assert!(
            matches!(err, ConvertError::TfjsParse { .. }),
            "expected TfjsParse, got {err:?}",
        );

        // No record committed; sweep cleared the staging dir + tempfile.
        let summary = fs.summary(&ws).expect("summary");
        assert!(summary.heads.heads.is_empty(), "no head record committed");
        assert_eq!(summary.core.head_count, 0);
        let staging_dir = workspace_dir.join(".tmp").join(format!("convert-{job_id}"));
        let mpk_tempfile = workspace_dir
            .join(".tmp")
            .join(format!("convert-{job_id}-{head_id}.mpk"));
        assert!(
            !staging_dir.exists(),
            "staging dir {} must be swept on failure",
            staging_dir.display(),
        );
        assert!(
            !mpk_tempfile.exists(),
            ".mpk tempfile {} must be swept on failure",
            mpk_tempfile.display(),
        );

        let log_path = workspace_dir
            .join(CONVERTER_LOGS_DIR_NAME)
            .join(format!("{job_id}.jsonl"));
        assert!(log_path.is_file(), "log missing");
        let log = std::fs::read_to_string(&log_path).unwrap();
        assert!(
            log.lines().any(|line| {
                serde_json::from_str::<serde_json::Value>(line)
                    .ok()
                    .and_then(|v| v["kind"].as_str().map(str::to_string))
                    .as_deref()
                    == Some("job_failed")
            }),
            "log must contain a `job_failed` event line; got:\n{log}",
        );
        // The same line carries the typed payload category + severity.
        let job_failed_line = log
            .lines()
            .find_map(|line| {
                let v: serde_json::Value = serde_json::from_str(line).ok()?;
                (v["kind"].as_str()? == "job_failed").then_some(v)
            })
            .expect("at least one job_failed line");
        assert_eq!(
            job_failed_line["category"].as_str(),
            Some("source_malformed"),
            "TfjsParse maps to source_malformed payload; got line {job_failed_line}",
        );
        assert_eq!(
            job_failed_line["severity"].as_str(),
            Some("operator_fixable"),
            "UserInput-kinded errors carry operator_fixable severity",
        );
        assert_eq!(
            job_failed_line["stage"].as_str(),
            Some("read_model_json"),
            "failure stage stamped from last StageStarted; got line {job_failed_line}",
        );
    }

    /// Rejects a job whose worker re-read of `model.json` reorders
    /// shard names vs the route's snapshot (same cardinality): the
    /// element-wise check prevents wrong-order byte pairing.
    #[test]
    fn convert_rejects_shard_name_mismatch() {
        let _gate = serialize_convert_test();

        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 2);

        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(0),
            kind: ConvertJobKind::Tfjs(ConvertJobTfjs {
                model_json_path: model_json,
                shard_paths: shards,
                // Fixture declares `weights.bin`; a different snapshot
                // name simulates the mid-job re-upload race.
                shard_names: vec!["renamed.bin".to_string()],
                labels_path: labels,
                labels_format: crate::file_mgr::LabelsFormat::Lines,
            }),
        };

        let err =
            run_convert_job(fs.clone(), job, None).expect_err("shard-name mismatch must reject");
        match err {
            ConvertError::TfjsParse { what, msg } => {
                assert_eq!(what, "shards");
                assert!(
                    msg.contains("ORDER") || msg.contains("names") || msg.contains("re-uploaded"),
                    "diagnostic must name the race; got: {msg}",
                );
            }
            other => panic!("expected TfjsParse(shards), got {other:?}"),
        }
    }

    /// `open` enforces `LOG_RETENTION_KEEP_COUNT` on `converter_logs/`:
    /// the just-opened file survives, older siblings are unlinked
    /// oldest-first.
    #[test]
    fn convert_job_log_open_enforces_keep_last_n() {
        // No gate: this test doesn't touch `CONVERT_SEMAPHORE`.
        let tmp = tempfile::tempdir().unwrap();
        let workspace_dir = tmp.path();
        let dir = workspace_dir.join(CONVERTER_LOGS_DIR_NAME);
        std::fs::create_dir_all(&dir).unwrap();
        let cap = crate::file_mgr::LOG_RETENTION_KEEP_COUNT;
        let mut stale_paths = Vec::with_capacity(cap + 1);
        for i in 0..=cap {
            let p = dir.join(format!("00000000-0000-4000-8000-{i:012x}.jsonl"));
            std::fs::write(&p, b"{}\n").unwrap();
            let backdate = std::time::SystemTime::now()
                .checked_sub(std::time::Duration::from_secs((1000 - i as u64) * 60))
                .expect("backdate");
            let secs = backdate
                .duration_since(std::time::UNIX_EPOCH)
                .expect("post-epoch")
                .as_secs();
            let ft = filetime::FileTime::from_unix_time(secs as i64, 0);
            filetime::set_file_mtime(&p, ft).expect("set mtime");
            stale_paths.push(p);
        }
        let job_id = crate::common::ids::JobId::new();
        let _log =
            JsonlEventLog::<ConvertEvent>::open(workspace_dir, CONVERTER_LOGS_DIR_NAME, job_id)
                .expect("open log");

        let new_path = dir.join(format!("{job_id}.jsonl"));
        assert!(new_path.is_file(), "new log survived");
        // (cap+1) stale + 1 new = cap+2; 2 oldest unlinked.
        assert!(!stale_paths[0].exists(), "oldest stale log unlinked");
        assert!(!stale_paths[1].exists(), "second-oldest stale log unlinked");
        assert!(
            stale_paths[2].exists(),
            "third-oldest stale log must survive (inside top-cap)",
        );
        let remaining: usize = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .ok()
                    .and_then(|e| e.file_name().into_string().ok())
                    .is_some_and(|n| n.ends_with(".jsonl"))
            })
            .count();
        assert_eq!(remaining, cap, "after open, dir holds exactly cap .jsonl");
    }

    /// Every JSONL line is well-formed JSON with the fixed-shape
    /// envelope fields. Pins the JSONL contract.
    #[test]
    fn convert_log_events_are_jsonl() {
        let _gate = serialize_convert_test();

        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 2);

        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(0),
            kind: ConvertJobKind::Tfjs(ConvertJobTfjs {
                model_json_path: model_json,
                shard_paths: shards,
                // Route derives this name from the parsed manifest.
                shard_names: vec!["weights.bin".to_string()],
                labels_path: labels,
                labels_format: crate::file_mgr::LabelsFormat::Lines,
            }),
        };
        run_convert_job(fs.clone(), job, None).expect("convert publishes");

        let log_path = workspace_dir
            .join(CONVERTER_LOGS_DIR_NAME)
            .join(format!("{job_id}.jsonl"));
        let log = std::fs::read_to_string(&log_path).expect("read log");
        let mut prev_seq: u64 = 0;
        let mut saw_submitted = false;
        let mut saw_running = false;
        let mut saw_completed = false;
        for line in log.lines() {
            let v: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("non-JSON log line {line:?}: {e}"));
            assert!(v["seq"].is_u64(), "seq must be u64; line={line}");
            assert!(v["at"].is_string(), "at must be string");
            assert!(v["kind"].is_string(), "kind must be string");
            let seq = v["seq"].as_u64().unwrap();
            assert!(seq > prev_seq, "seq not monotonic: {prev_seq} -> {seq}");
            prev_seq = seq;
            match v["kind"].as_str().unwrap() {
                "job_submitted" => saw_submitted = true,
                "job_running" => saw_running = true,
                "job_completed" => saw_completed = true,
                _ => {}
            }
        }
        // Match by kind to sidestep per-stage ordering.
        assert!(saw_submitted, "log must contain a `job_submitted` event");
        assert!(saw_running, "log must contain a `job_running` event");
        assert!(saw_completed, "log must contain a `job_completed` event");
    }

    // MARK: convert-pipeline manifest-shape pins
    //
    // Pin the worker's on-disk `HeadManifest` field set so a quiet
    // re-addition of a legacy field surfaces here first.

    /// Published `<head_id>.json` field set is exactly the minimized contract.
    #[test]
    fn convert_publishes_minimized_manifest_field_set() {
        let _gate = serialize_convert_test();
        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 2);
        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(7),
            kind: ConvertJobKind::Tfjs(ConvertJobTfjs {
                model_json_path: model_json,
                shard_paths: shards,
                // Route derives this name from the parsed manifest.
                shard_names: vec!["weights.bin".to_string()],
                labels_path: labels,
                labels_format: crate::file_mgr::LabelsFormat::Lines,
            }),
        };
        run_convert_job(fs.clone(), job, None).expect("convert publishes");

        let manifest_bytes = std::fs::read(crate::file_mgr::schema::head_manifest_path(
            &workspace_dir,
            head_id,
        ))
        .expect("read on-disk manifest");
        let v: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
        let obj = v.as_object().expect("manifest is a JSON object");
        let actual: std::collections::BTreeSet<&str> = obj.keys().map(String::as_str).collect();
        let expected: std::collections::BTreeSet<&str> = [
            "head_id",
            "workspace_id",
            "workspace_revision",
            "sha256",
            "n_classes",
            "size_bytes",
            "created_at",
            "labels",
        ]
        .into_iter()
        .collect();
        assert_eq!(
            actual, expected,
            "convert-published manifest must carry exactly the minimized field set; got {actual:?}",
        );
        for forbidden in [
            "dataset_path",
            "training_cfg",
            "training_cfg_sha256",
            "dataset_revision",
            "dataset_revision_at_train",
            "convert_provenance",
            "input_paths",
        ] {
            assert!(
                !obj.contains_key(forbidden),
                "legacy field {forbidden:?} must not appear in published manifest",
            );
        }
    }

    /// Published `heads.json` `HeadRecord` field set is exactly the minimized
    /// contract.
    #[test]
    fn convert_publishes_minimized_head_record_field_set() {
        let _gate = serialize_convert_test();
        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 2);
        let head_id = HeadId::new();
        let job_id = JobId::new();
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: rev(7),
            kind: ConvertJobKind::Tfjs(ConvertJobTfjs {
                model_json_path: model_json,
                shard_paths: shards,
                // Route derives this name from the parsed manifest.
                shard_names: vec!["weights.bin".to_string()],
                labels_path: labels,
                labels_format: crate::file_mgr::LabelsFormat::Lines,
            }),
        };
        run_convert_job(fs.clone(), job, None).expect("convert publishes");

        let index_bytes = std::fs::read(crate::file_mgr::schema::head_index_path(&workspace_dir))
            .expect("read on-disk heads.json");
        let v: serde_json::Value = serde_json::from_slice(&index_bytes).unwrap();
        let entries = v["heads"].as_array().expect("heads is an array");
        assert_eq!(entries.len(), 1, "exactly one published head");
        let rec = entries[0].as_object().expect("HeadRecord is a JSON object");
        let actual: std::collections::BTreeSet<&str> = rec.keys().map(String::as_str).collect();
        let expected: std::collections::BTreeSet<&str> = [
            "head_id",
            "workspace_revision",
            "sha256",
            "n_classes",
            "size_bytes",
            "created_at",
        ]
        .into_iter()
        .collect();
        assert_eq!(
            actual, expected,
            "convert-published HeadRecord must carry exactly the minimized field set; got {actual:?}",
        );
        for forbidden in [
            "dataset_path",
            "training_cfg_sha256",
            "dataset_revision_at_train",
            "labels",
            "workspace_id",
        ] {
            assert!(
                !rec.contains_key(forbidden),
                "legacy / non-record field {forbidden:?} must not appear in HeadRecord",
            );
        }
    }

    /// The converter publishes with the producer-snapshotted
    /// `workspace_revision` (a re-fetch would break stale detection).
    #[test]
    fn convert_manifest_workspace_revision_matches_producer_snapshot() {
        let _gate = serialize_convert_test();
        let tmp = tempfile::tempdir().unwrap();
        let (ws, fs) = stage_test_workspace(tmp.path());
        let workspace_dir = crate::file_mgr::schema::workspace_dir_for(fs.root(), &ws);
        let (model_json, shards, labels) = stage_minimal_tfjs(&workspace_dir, 2);
        let head_id = HeadId::new();
        let job_id = JobId::new();
        // A distinct revision id makes the assertion load-bearing (a
        // publish-time re-read would surface id = 0).
        let snapshot = rev(42);
        let job = ConvertJob {
            job_id,
            workspace_id: ws,
            head_id,
            workspace_revision: snapshot.clone(),
            kind: ConvertJobKind::Tfjs(ConvertJobTfjs {
                model_json_path: model_json,
                shard_paths: shards,
                // Route derives this name from the parsed manifest.
                shard_names: vec!["weights.bin".to_string()],
                labels_path: labels,
                labels_format: crate::file_mgr::LabelsFormat::Lines,
            }),
        };
        run_convert_job(fs.clone(), job, None).expect("convert publishes");

        let manifest = crate::file_mgr::schema::read_head_manifest(&workspace_dir, head_id)
            .expect("read manifest");
        assert_eq!(manifest.workspace_revision.id, snapshot.id);
        assert_eq!(manifest.workspace_revision.at, snapshot.at);
        let summary = fs.summary(&ws).expect("summary");
        assert_eq!(summary.heads.heads[0].workspace_revision.id, snapshot.id);
    }

    /// The constructed `HeadManifest` has no slot for legacy provenance
    /// fields.
    #[test]
    fn convert_constructed_manifest_has_no_legacy_provenance() {
        let manifest = crate::common::workspace::HeadManifest {
            head_id: HeadId::new(),
            workspace_id: WorkspaceId::new(),
            workspace_revision: rev(3),
            sha256: "abc".into(),
            n_classes: 2,
            size_bytes: 1024,
            created_at: "2026-05-08T12:00:00Z".into(),
            labels: vec!["a".into(), "b".into()],
        };
        let v = serde_json::to_value(&manifest).unwrap();
        let obj = v.as_object().expect("manifest serializes as JSON object");
        for forbidden in [
            "dataset_path",
            "training_cfg",
            "training_cfg_sha256",
            "convert_provenance",
            "input_paths",
        ] {
            assert!(
                !obj.contains_key(forbidden),
                "HeadManifest must not carry legacy field {forbidden:?}",
            );
        }
    }
}
