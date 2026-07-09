//! In-process training job registry.
//!
//! Dataset root is fixed at `<workspace_dir>/datasets/`; at most one
//! unfinished train job daemon-wide (409 otherwise). `finetune::run`
//! opens/closes per batch, so worst-case FDs are
//! `batch_size * parallel_loaders` independent of dataset size. On
//! success the `.mpk` is staged under `<workspace_dir>/.tmp/` and
//! published via the head-rotation primitive; no head record commits
//! on failure.

#![warn(missing_debug_implementations)]

mod finetune;
pub(crate) mod registry;
pub use finetune::{ClassCount, EpochMetrics, Stage};
pub use registry::TrainingRegistry;

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::common::ids::{HeadId, JobId, WorkspaceId};
use crate::common::workspace::{HeadManifest, WorkspaceRevision};
use crate::file_mgr::{
    DATASETS_DIR_NAME, FsService, JobHandle, JsonlEventLog, PendingHead, RegistryJobResult,
    TRAINING_LOGS_DIR_NAME, TrainingCfg, now_rfc3339, sha256_file_streaming, validate_training_cfg,
};
use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Serialize;
use thiserror::Error;
use tokio::sync::watch;

// `AtomicU8` (not bool) so the terminal `JobCancelled` event can carry
// a typed reason. Value transitions monotonically NONE->OPERATOR/SHUTDOWN,
// never reversing, so Acquire/Release would suffice; SeqCst for uniformity.
const CANCEL_NONE: u8 = 0;
const CANCEL_OPERATOR: u8 = 1;
const CANCEL_SHUTDOWN: u8 = 2;

fn cancel_requested(cancel: &AtomicU8) -> bool {
    cancel.load(Ordering::SeqCst) != CANCEL_NONE
}

/// Daemon-internal job descriptor from a validated `TrainRequest`.
#[derive(Clone, Debug)]
pub struct TrainingJob {
    pub workspace_id: WorkspaceId,
    /// Allocated by the api producer so the response carries the published
    /// id before the job spawns; published verbatim on success.
    pub head_id: HeadId,
    /// Recorded in the head manifest for stale detection.
    pub workspace_revision: WorkspaceRevision,
    pub training_cfg: TrainingCfg,
    /// Ordered extractor candidates (supported kinds only); semantics on
    /// `finetune::FinetuneConfig::backbones`.
    pub backbone_candidates: crate::inference::BackboneCatalogue,
    /// Serving's boot-loaded candidate, for the resolver's basis-divergence
    /// warning; semantics on `finetune::FinetuneConfig::serving_backbone`.
    pub serving_backbone: Option<crate::inference::BackboneRef>,
}

pub use crate::common::error::Severity;

/// Why a job ended in [`JobState::Cancelled`]; the frontend renders
/// distinct copy for operator-cancel vs shutdown-drain.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CancelReason {
    Operator,
    Shutdown,
}

/// Tagged failure payload for [`TrainEvent::JobFailed`]: per-variant
/// fields let the frontend build hint copy without re-parsing free-form
/// error strings.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "category", rename_all = "snake_case")]
pub enum FailPayload {
    /// Malformed dataset layout, detected by `scan_dataset` at admission.
    BadDataset { path: String, reason: String },
    /// A discovered dataset file disappeared/became unreadable mid-walk.
    DatasetRead { path: String, reason: String },
    /// Post-extract a class had zero kept examples.
    EmptyClass {
        class: String,
        per_class_kept: Vec<(String, usize)>,
    },
    /// Post-extract sibling of [`Self::EmptyClass`]: class survived scan
    /// but every example failed decode/resample/non-finite-spectrogram.
    EmptyClassAfterExtract {
        class: String,
        per_class_kept: Vec<(String, usize)>,
        per_class_dropped: Vec<(String, usize)>,
    },
    /// Aggregate drop ratio exceeded `MAX_DROP_RATIO`, so the published
    /// head's metrics would describe a degraded subset.
    DropRatioExceeded {
        dropped: usize,
        total: usize,
        threshold: f32,
        per_class_kept: Vec<(String, usize)>,
        per_class_dropped: Vec<(String, usize)>,
    },
    /// One class crossed the per-class cap while the aggregate stayed
    /// under [`Self::DropRatioExceeded`]; distinct so the operator isn't
    /// told an aggregate ceiling tripped when one class did.
    PerClassDropExceeded {
        class: String,
        dropped: usize,
        total: usize,
        threshold: f32,
        per_class_kept: Vec<(String, usize)>,
        per_class_dropped: Vec<(String, usize)>,
    },
    /// A class has too few samples for `validation_split` to leave one
    /// example in each of train + val (e.g. singleton class, split > 0).
    StratifiedSplitImpossible {
        class: String,
        per_class_kept: Vec<(String, usize)>,
        val_split: f32,
    },
    /// Post-spawn numeric/shape validation failure on the derived config.
    InvalidConfig { detail: String },
    /// Model-artifact failure: Burn `.mpk` load/save, or the feature-extractor
    /// backbone (candidate resolution / NPU inference fault).
    ModelError { detail: String },
    /// Mid-training NaN/+Inf. Pre-step abort means the live head is NOT
    /// poisoned; the prior best-epoch snapshot remains usable.
    NumericFailure {
        epoch: u32,
        batch_index: u32,
        kind: String,
        value: f64,
    },
    /// IO failure on a daemon-owned file.
    Io { path: String, detail: String },
    /// Training loop panicked; the unwind was caught at the worker boundary.
    Panic { detail: String },
    /// Catch-all for daemon-internal errors with no specific variant.
    Internal { detail: String },
}

/// One lifecycle event, emitted to both the durable JSONL log and the
/// cross-cutting SSE broadcast as `{seq, at, ...flattened}` so a
/// tab-refresh hydrates from JSONL with no shape divergence from the
/// live stream. Wrapper-only variants are emitted by `run_job`; the
/// algorithmic ones are lifted from `finetune::Event` via the [`From`]
/// impl below. `#[non_exhaustive]`: external matches must handle the
/// unknown `kind`.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TrainEvent {
    /// Admission cleared, log opened, before any pipeline work; carries
    /// the cfg snapshot so a JSONL replay describes what was attempted.
    JobSubmitted {
        head_id: HeadId,
        cfg: TrainingCfg,
        /// Basename only; full path withheld from operator logs.
        backbone: String,
    },
    /// Worker started on the blocking pool; distinct from `JobSubmitted`
    /// so the SSE consumer can render a queued->running transition.
    JobRunning,
    PhaseStarted {
        phase: Stage,
    },
    DatasetScanned {
        n_classes: u32,
        classes: Vec<ClassCount>,
        n_examples_total: u64,
    },
    FeatureExtractCompleted {
        kept: u64,
        dropped_nan: u64,
        dropped_io: u64,
        elapsed_ms: u64,
    },
    TrainSplit {
        train_n: u64,
        val_n: u64,
    },
    EpochCompleted {
        epoch: u32,
        epochs: u32,
        train_loss: f64,
        train_acc: f32,
        #[serde(serialize_with = "serialize_finite_or_null")]
        val_acc: f32,
        #[serde(serialize_with = "serialize_finite_or_null")]
        best_val_acc: f32,
        /// Mean validation cross-entropy; tiebreaker among equal-`val_acc`
        /// epochs. `null` on the `val_split == 0.0` path.
        #[serde(serialize_with = "serialize_finite_or_null_f64")]
        val_loss: f64,
        /// Validation loss of the currently-published best snapshot.
        #[serde(serialize_with = "serialize_finite_or_null_f64")]
        best_val_loss: f64,
        lr: f32,
        elapsed_ms: u64,
    },
    TrainCompleted {
        epochs_run: u32,
        total_elapsed_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        best_val_epoch: Option<u32>,
        #[serde(
            skip_serializing_if = "Option::is_none",
            serialize_with = "serialize_finite_or_null_opt"
        )]
        best_val_acc: Option<f32>,
        #[serde(
            skip_serializing_if = "Option::is_none",
            serialize_with = "serialize_finite_or_null_opt_f64"
        )]
        best_val_loss: Option<f64>,
    },
    /// Emitted only on success; its absence in a JSONL transcript means
    /// the publish was not reached (`JobFailed` will follow).
    HeadPublished {
        head_id: HeadId,
        head_sha256: String,
        size_bytes: u64,
        n_classes: u32,
        classes: Vec<String>,
        workspace_revision: WorkspaceRevision,
    },
    /// Terminal success; carries the full `TrainingResult` so a JSONL
    /// hydration surfaces the verdict without a separate fetch.
    JobCompleted {
        result: TrainingResult,
    },
    /// Terminal failure. `stage` is the last `PhaseStarted` observed.
    JobFailed {
        stage: Stage,
        severity: Severity,
        error: String,
        #[serde(flatten)]
        payload: FailPayload,
    },
    JobCancelled {
        stage: Stage,
        reason: CancelReason,
    },
}

/// Lift `finetune::Event` variants into the wrapper's wire shape, 1:1.
impl From<finetune::Event> for TrainEvent {
    fn from(e: finetune::Event) -> Self {
        match e {
            finetune::Event::PhaseStarted { phase } => TrainEvent::PhaseStarted { phase },
            finetune::Event::DatasetScanned {
                n_classes,
                classes,
                n_examples_total,
            } => TrainEvent::DatasetScanned {
                n_classes,
                classes,
                n_examples_total,
            },
            finetune::Event::FeatureExtractCompleted {
                kept,
                dropped_nan,
                dropped_io,
                elapsed_ms,
            } => TrainEvent::FeatureExtractCompleted {
                kept,
                dropped_nan,
                dropped_io,
                elapsed_ms,
            },
            finetune::Event::TrainSplit { train_n, val_n } => {
                TrainEvent::TrainSplit { train_n, val_n }
            }
            finetune::Event::EpochCompleted {
                epoch,
                epochs,
                train_loss,
                train_acc,
                val_acc,
                best_val_acc,
                val_loss,
                best_val_loss,
                lr,
                elapsed_ms,
            } => TrainEvent::EpochCompleted {
                epoch,
                epochs,
                train_loss,
                train_acc,
                val_acc,
                best_val_acc,
                val_loss,
                best_val_loss,
                lr,
                elapsed_ms,
            },
            finetune::Event::TrainCompleted {
                epochs_run,
                total_elapsed_ms,
                best_val_epoch,
                best_val_acc,
                best_val_loss,
            } => TrainEvent::TrainCompleted {
                epochs_run,
                total_elapsed_ms,
                best_val_epoch,
                best_val_acc,
                best_val_loss,
            },
        }
    }
}

/// Strip the workspace_dir prefix so failure cards render
/// workspace-relative paths; non-matching paths pass through, `None` is
/// the test-only no-strip mode.
fn strip_workspace_prefix(path: &str, workspace_dir: Option<&std::path::Path>) -> String {
    let Some(wd) = workspace_dir else {
        return path.to_string();
    };
    let prefix = wd.display().to_string();
    if let Some(rest) = path.strip_prefix(&prefix) {
        rest.trim_start_matches('/').to_string()
    } else {
        path.to_string()
    }
}

/// Map a [`TrainingError`] to the typed [`FailPayload`] at the terminal
/// transition; `workspace_dir` strips path-bearing variants (tests pass `None`).
fn fail_payload_from_error(
    err: &TrainingError,
    workspace_dir: Option<&std::path::Path>,
) -> FailPayload {
    use finetune::FinetuneError as F;
    let strip = |p: &str| strip_workspace_prefix(p, workspace_dir);
    match err {
        TrainingError::BadDataset { path, reason } => FailPayload::BadDataset {
            path: strip(path),
            reason: reason.clone(),
        },
        TrainingError::DatasetRead { path, reason } => FailPayload::DatasetRead {
            path: strip(path),
            reason: reason.clone(),
        },
        TrainingError::Finetune(F::EmptyClassAfterScan {
            class,
            per_class_kept,
        }) => FailPayload::EmptyClass {
            class: class.clone(),
            per_class_kept: per_class_kept.clone(),
        },
        TrainingError::Finetune(F::EmptyClassAfterExtract {
            class,
            per_class_kept,
            per_class_dropped,
        }) => FailPayload::EmptyClassAfterExtract {
            class: class.clone(),
            per_class_kept: per_class_kept.clone(),
            per_class_dropped: per_class_dropped.clone(),
        },
        TrainingError::Finetune(F::DropRatioExceeded {
            dropped,
            total,
            ratio: _,
            max_ratio,
            per_class_kept,
            per_class_dropped,
        }) => FailPayload::DropRatioExceeded {
            dropped: *dropped,
            total: *total,
            threshold: *max_ratio,
            per_class_kept: per_class_kept.clone(),
            per_class_dropped: per_class_dropped.clone(),
        },
        TrainingError::Finetune(F::PerClassDropExceeded {
            class,
            dropped,
            total,
            ratio: _,
            max_ratio,
            per_class_kept,
            per_class_dropped,
        }) => FailPayload::PerClassDropExceeded {
            class: class.clone(),
            dropped: *dropped,
            total: *total,
            threshold: *max_ratio,
            per_class_kept: per_class_kept.clone(),
            per_class_dropped: per_class_dropped.clone(),
        },
        TrainingError::Finetune(F::StratifiedSplitImpossible {
            class,
            per_class_kept,
            val_split,
        }) => FailPayload::StratifiedSplitImpossible {
            class: class.clone(),
            per_class_kept: per_class_kept.clone(),
            val_split: *val_split,
        },
        TrainingError::Finetune(F::InvalidConfig(detail))
        | TrainingError::InvalidConfig(detail) => FailPayload::InvalidConfig {
            detail: detail.clone(),
        },
        TrainingError::Finetune(F::Model(e)) => FailPayload::ModelError {
            detail: e.to_string(),
        },
        // Backbone failures (no usable candidate, NPU fault) are
        // model-artifact problems on the wire.
        TrainingError::Finetune(F::Backbone(detail)) => FailPayload::ModelError {
            detail: detail.clone(),
        },
        TrainingError::Io { path, source } => FailPayload::Io {
            path: strip(path),
            detail: source.to_string(),
        },
        TrainingError::Finetune(F::Io { path, source }) => FailPayload::Io {
            path: strip(path),
            detail: source.to_string(),
        },
        TrainingError::Finetune(F::Panic(detail)) => FailPayload::Panic {
            detail: detail.clone(),
        },
        TrainingError::Finetune(F::NumericFailure {
            epoch,
            batch_index,
            kind,
            value,
        }) => FailPayload::NumericFailure {
            epoch: *epoch,
            batch_index: *batch_index,
            kind: (*kind).to_string(),
            value: *value,
        },
        // Catch-alls surface as `Internal`, diagnostic preserved.
        TrainingError::Finetune(other) => internal_payload(other),
        TrainingError::File(e) => internal_payload(e),
        TrainingError::Fs(e) => internal_payload(e),
        TrainingError::Join(e) => internal_payload(e),
        // Literal pins the wire shape independent of Cancelled's Display.
        TrainingError::Cancelled => FailPayload::Internal {
            detail: "cancelled".into(),
        },
        // Outer Display carries the id/state/workspace operators need;
        // the inner error would strip them.
        TrainingError::JobNotCancellable { .. }
        | TrainingError::JobNotFound(_)
        | TrainingError::WrongWorkspace { .. } => internal_payload(err),
    }
}

fn internal_payload(e: impl std::fmt::Display) -> FailPayload {
    FailPayload::Internal {
        detail: e.to_string(),
    }
}

fn severity_from_error(err: &TrainingError) -> Severity {
    use crate::common::error::Categorized;
    Severity::from(err.kind())
}

fn serialize_finite_or_null<S: serde::Serializer>(v: &f32, s: S) -> Result<S::Ok, S::Error> {
    if v.is_finite() {
        s.serialize_f32(*v)
    } else {
        s.serialize_none()
    }
}

fn serialize_finite_or_null_opt<S: serde::Serializer>(
    v: &Option<f32>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match v {
        Some(f) if f.is_finite() => s.serialize_f32(*f),
        _ => s.serialize_none(),
    }
}

/// `f64` sibling of [`serialize_finite_or_null`]: val-loss NaN
/// (`val_split == 0.0`) and +Inf (no-best-yet sentinel) both become JSON
/// `null`, since serde_json refuses non-finite floats.
fn serialize_finite_or_null_f64<S: serde::Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
    if v.is_finite() {
        s.serialize_f64(*v)
    } else {
        s.serialize_none()
    }
}

fn serialize_finite_or_null_opt_f64<S: serde::Serializer>(
    v: &Option<f64>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match v {
        Some(f) if f.is_finite() => s.serialize_f64(*f),
        _ => s.serialize_none(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Final artefacts published by a successful training run.
#[derive(Clone, Debug, Serialize)]
pub struct TrainingResult {
    pub head_id: HeadId,
    /// Lowercase-hex SHA-256 of the published `<head_id>.mpk`.
    pub head_sha256: String,
    pub n_classes: u32,
    /// Class labels in inference order.
    pub classes: Vec<String>,
    pub final_train_acc: f32,
    /// NaN when `validation_split == 0.0` (no holdout); serialises to
    /// JSON `null` so the `JobView` endpoint round-trips.
    #[serde(serialize_with = "serialize_finite_or_null")]
    pub final_val_acc: f32,
}

/// Read-shape returned by `/api/v1/training/{id}`.
#[derive(Clone, Debug, Serialize)]
pub struct JobView {
    pub job_id: String,
    pub workspace_id: String,
    pub state: JobState,
    pub progress: finetune::Progress,
    /// Present only once `state == Completed`.
    pub result: Option<TrainingResult>,
    /// Failure message once `Failed`, literal `"cancelled"` once
    /// `Cancelled`; `None` while Running and on success.
    pub error: Option<String>,
    /// RFC3339 wall-clock at job spawn.
    pub started_at: String,
    /// RFC3339 wall-clock at terminal state, if any.
    pub finished_at: Option<String>,
}

/// Failure shapes for the training pipeline; mapped to HTTP statuses by
/// the [`crate::common::error::Categorized`] impl below.
#[derive(Debug, Error)]
pub enum TrainingError {
    /// Config issue surfaced after the request-boundary
    /// `validate_training_cfg` (e.g. a missing backbone artefact).
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("job not found: {0}")]
    JobNotFound(String),
    #[error("job {job} does not belong to workspace {workspace}")]
    WrongWorkspace { job: String, workspace: String },
    /// Operator cancelled; the worker exited at the next checkpoint.
    #[error("cancelled")]
    Cancelled,
    /// Cancel rejected on an already-terminal job; distinct from
    /// `Cancelled` so the api returns 409 not a misleading 200.
    #[error("job {job} is not cancellable (state: {state:?})")]
    JobNotCancellable { job: String, state: JobState },
    #[error("file: {0}")]
    File(#[from] crate::file_mgr::FileError),
    #[error("fs: {0}")]
    Fs(#[from] crate::file_mgr::FsError),
    /// Underlying fine-tune failure; inner `BadDataset`/`DatasetRead`
    /// are lifted to [`Self::BadDataset`]/[`Self::DatasetRead`] so
    /// tooling pattern-matches once at this boundary.
    #[error("finetune: {0}")]
    Finetune(finetune::FinetuneError),
    /// Typed dataset-shape rejection at scan time (400).
    #[error("bad dataset {path}: {reason}")]
    BadDataset { path: String, reason: String },
    /// IO failure on a daemon-owned file.
    #[error("io {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("spawn_blocking join: {0}")]
    Join(#[from] tokio::task::JoinError),
    /// A dataset file disappeared mid-job; `Internal` because
    /// `datasets/` is daemon-owned and the lease blocks legit mutations.
    #[error("dataset read failure {path}: {reason}")]
    DatasetRead { path: String, reason: String },
}

impl crate::common::error::Categorized for TrainingError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            TrainingError::InvalidConfig(_) => UserInput,
            TrainingError::Cancelled => Conflict,
            TrainingError::JobNotCancellable { .. } => Conflict,
            TrainingError::JobNotFound(_) | TrainingError::WrongWorkspace { .. } => NotFound,
            TrainingError::File(e) => e.kind(),
            TrainingError::Fs(e) => e.kind(),
            TrainingError::BadDataset { .. } => UserInput,
            // Delegate so dataset-quality variants keep their 400 while
            // panic/Io/Model stay Internal.
            TrainingError::Finetune(e) => e.kind(),
            // Daemon-owned tree: a missing mid-walk file is not operator-fixable.
            TrainingError::Io { .. }
            | TrainingError::Join(_)
            | TrainingError::DatasetRead { .. } => Internal,
        }
    }
}

/// Lift `BadDataset`/`DatasetRead` to the wrapper's typed shapes; other
/// variants flow through `Finetune(_)` unchanged.
impl From<finetune::FinetuneError> for TrainingError {
    fn from(value: finetune::FinetuneError) -> Self {
        match value {
            finetune::FinetuneError::BadDataset { path, reason } => {
                TrainingError::BadDataset { path, reason }
            }
            finetune::FinetuneError::DatasetRead { path, reason } => {
                TrainingError::DatasetRead { path, reason }
            }
            other => TrainingError::Finetune(other),
        }
    }
}

/// In-process registry of training jobs (cheaply cloneable), holding
/// the rich per-job state the cross-cutting `file_mgr::JobRegistry`'s
/// flat shape cannot. When bridged both registries key on the same
/// [`JobId`] (flowing in via the [`JobHandle`]); the test-only
/// `job_handle: None` path mints a local id.
#[derive(Clone, Debug)]
pub struct JobRegistry {
    jobs: Arc<DashMap<JobId, Arc<JobEntry>>>,
}

impl JobRegistry {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(DashMap::new()),
        }
    }

    /// Spawn a new training job, returning the [`JobId`]. Training runs
    /// on a `spawn_blocking` worker that transitions the entry to
    /// terminal on completion. `Some(job_handle)` fans typed events to
    /// the SSE broadcast; `None` (test-only) still writes the JSONL.
    /// Admission against `max_train_jobs = 1` happens at the api boundary.
    pub fn spawn(
        &self,
        files: Arc<dyn FsService>,
        job: TrainingJob,
        job_handle: Option<JobHandle>,
    ) -> Result<JobId, TrainingError> {
        // Re-validate here: a hand-built TrainingJob (test/replay) must
        // hit the same gate as the api route.
        validate_training_cfg(&job.training_cfg).map_err(|e| {
            TrainingError::File(crate::file_mgr::FileError::InvalidName(e.to_string()))
        })?;

        // Reuse the file_mgr-allocated id when bridged so `/jobs/{id}`
        // and `/workspaces/{id}/training/{job}` agree on one id.
        let job_id = job_handle
            .as_ref()
            .map(|h| h.job_id())
            .unwrap_or_else(JobId::new);
        let initial = finetune::Progress {
            phase: Stage::Prepare,
            current: 0,
            total: 0,
            message: "training job accepted".into(),
            metrics: None,
        };
        let (progress_tx, progress_rx) = watch::channel(initial);
        let cancel = Arc::new(AtomicU8::new(CANCEL_NONE));
        let entry = Arc::new(JobEntry {
            job_id,
            workspace_id: job.workspace_id,
            started_at: now_rfc3339(),
            progress: progress_rx,
            cancel: cancel.clone(),
            core: Mutex::new(JobCore {
                state: JobState::Running,
                result: None,
                error: None,
                finished_at: None,
                finished_at_instant: None,
            }),
        });
        self.jobs.insert(job_id, entry.clone());

        // Detached: shutdown reaches the worker via `cancel`; the
        // blocking pool can't abort mid-batch, so cancel latency is
        // bounded by one BACKBONE_BATCH drain (~hundreds of ms on the
        // Burn path; the NPU path polls per window). `JobHandle`
        // is consumed at terminal; if the closure panics first,
        // `JobHandle::Drop` records `Failed`.
        tokio::spawn(async move {
            // Without this guard a panic inside the closure leaves the
            // entry `Running` forever (phantom job); force `Failed` on
            // unwind. Repairs only the training-local view; `JobHandle::Drop`
            // handles the cross-cutting slot.
            struct FinishGuard {
                entry: Arc<JobEntry>,
                job_id: JobId,
                committed: bool,
            }
            impl Drop for FinishGuard {
                fn drop(&mut self) {
                    if self.committed {
                        return;
                    }
                    let mut core = self.entry.core.lock();
                    // Don't overwrite a pre-stamped terminal state.
                    if core.state == JobState::Running {
                        core.state = JobState::Failed;
                        core.error = Some("training task panicked".into());
                        core.finished_at = Some(now_rfc3339());
                        core.finished_at_instant = Some(std::time::Instant::now());
                        tracing::error!(
                            target: "training",
                            job_id = %self.job_id,
                            "training task panicked",
                        );
                    }
                }
            }
            let mut guard = FinishGuard {
                entry: entry.clone(),
                job_id,
                committed: false,
            };
            let outcome = run_job(files, job, job_id, progress_tx, cancel, job_handle).await;
            let mut core = entry.core.lock();
            core.finished_at = Some(now_rfc3339());
            core.finished_at_instant = Some(std::time::Instant::now());
            match outcome {
                Ok(result) => {
                    core.state = JobState::Completed;
                    core.result = Some(result);
                }
                Err(TrainingError::Cancelled)
                | Err(TrainingError::Finetune(finetune::FinetuneError::Cancelled)) => {
                    core.state = JobState::Cancelled;
                    core.error = Some("cancelled".into());
                }
                Err(e) => {
                    core.state = JobState::Failed;
                    core.error = Some(e.to_string());
                    tracing::warn!(target: "training", job_id = %job_id, err = %e, "training job failed");
                }
            }
            drop(core);
            guard.committed = true;
        });

        Ok(job_id)
    }

    /// Look up `job_id` and confirm it belongs to `workspace_id`. The
    /// returned `Arc<JobEntry>` outlives the dashmap shard guard, so the
    /// caller may take per-entry locks without holding the shard ref.
    fn lookup_for_workspace(
        &self,
        workspace_id: &WorkspaceId,
        job_id: JobId,
    ) -> Result<Arc<JobEntry>, TrainingError> {
        let entry = self
            .jobs
            .get(&job_id)
            .ok_or_else(|| TrainingError::JobNotFound(job_id.to_string()))?;
        if entry.workspace_id != *workspace_id {
            return Err(TrainingError::WrongWorkspace {
                job: job_id.to_string(),
                workspace: workspace_id.to_string(),
            });
        }
        Ok(entry.clone())
    }

    /// Request cancellation: stores `CANCEL_OPERATOR` unconditionally
    /// (idempotent, overrides a prior `CANCEL_SHUTDOWN`); the job exits
    /// at the next checkpoint. Rejects with `JobNotCancellable` (409) if
    /// already terminal under the entry lock, so the frontend never shows
    /// a stuck "cancelling...". The remaining Running->terminal TOCTOU
    /// after lock release is harmless: the worker has exited so no reader
    /// observes the flag.
    pub fn cancel(&self, workspace_id: &WorkspaceId, job_id: JobId) -> Result<(), TrainingError> {
        let entry = self.lookup_for_workspace(workspace_id, job_id)?;
        {
            let core = entry.core.lock();
            if core.state != JobState::Running {
                return Err(TrainingError::JobNotCancellable {
                    job: job_id.to_string(),
                    state: core.state.clone(),
                });
            }
        }
        entry.cancel.store(CANCEL_OPERATOR, Ordering::SeqCst);
        Ok(())
    }

    pub fn status(
        &self,
        workspace_id: &WorkspaceId,
        job_id: JobId,
    ) -> Result<JobView, TrainingError> {
        Ok(self.lookup_for_workspace(workspace_id, job_id)?.view())
    }

    /// Jobs scoped to `workspace_id`, oldest-first by parsed instant:
    /// `started_at` is variable-width RFC3339 whose lexical order is not
    /// chronological within a second.
    pub fn list_for_workspace(&self, workspace_id: &WorkspaceId) -> Vec<JobView> {
        use crate::file_mgr::time_util::parse_rfc3339_or_epoch;
        let mut out: Vec<_> = self
            .jobs
            .iter()
            .filter(|entry| entry.value().workspace_id == *workspace_id)
            .map(|entry| entry.value().view())
            .collect();
        out.sort_by(|a, b| {
            parse_rfc3339_or_epoch(&a.started_at).cmp(&parse_rfc3339_or_epoch(&b.started_at))
        });
        out
    }

    /// Pre-drain hook: stamp `CANCEL_SHUTDOWN` on every running job,
    /// returning the count that transitioned from `CANCEL_NONE`;
    /// already-operator-cancelled jobs are preserved.
    pub fn cancel_all_for_shutdown(&self) -> usize {
        let mut n = 0usize;
        for entry in self.jobs.iter() {
            let core = entry.value().core.lock();
            if core.state != JobState::Running {
                continue;
            }
            drop(core);
            if entry
                .value()
                .cancel
                .compare_exchange(
                    CANCEL_NONE,
                    CANCEL_SHUTDOWN,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                n = n.saturating_add(1);
            }
        }
        n
    }

    pub fn active_count(&self) -> usize {
        self.jobs
            .iter()
            .filter(|entry| entry.value().core.lock().state == JobState::Running)
            .count()
    }

    /// Drop finished entries older than `max_age`; returns the count
    /// reaped. Running jobs and entries with no finish time are kept.
    pub fn reap_finished(&self, max_age: std::time::Duration) -> usize {
        let now = std::time::Instant::now();
        let to_remove: Vec<JobId> = self
            .jobs
            .iter()
            .filter_map(|entry| {
                let core = entry.value().core.lock();
                let finished_at = core.finished_at_instant?;
                if now.duration_since(finished_at) >= max_age {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect();
        let n = to_remove.len();
        for id in to_remove {
            self.jobs.remove(&id);
        }
        n
    }
}

impl Default for JobRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct JobEntry {
    job_id: JobId,
    workspace_id: WorkspaceId,
    started_at: String,
    progress: watch::Receiver<finetune::Progress>,
    /// Polled at every chunk/epoch boundary; read at the terminal
    /// transition to attach a [`CancelReason`].
    cancel: Arc<AtomicU8>,
    core: Mutex<JobCore>,
}

impl JobEntry {
    fn view(&self) -> JobView {
        let progress = self.progress.borrow().clone();
        let core = self.core.lock();
        JobView {
            job_id: self.job_id.to_string(),
            workspace_id: self.workspace_id.to_string(),
            state: core.state.clone(),
            progress,
            result: core.result.clone(),
            error: core.error.clone(),
            started_at: self.started_at.clone(),
            finished_at: core.finished_at.clone(),
        }
    }
}

#[derive(Debug)]
struct JobCore {
    state: JobState,
    result: Option<TrainingResult>,
    error: Option<String>,
    finished_at: Option<String>,
    finished_at_instant: Option<std::time::Instant>,
}

/// Run a single training job end-to-end: open the JSONL log, emit
/// `JobSubmitted`+`JobRunning`, run [`run_job_inner`], then emit a typed
/// terminal event and consume the [`JobHandle`] (`succeed` carries
/// [`RegistryJobResult::Train`] so `GET /jobs/{id}` surfaces the new head
/// id). An unwritable `training_logs/` refuses the run loudly; any
/// failure before publish leaves no head record on disk.
async fn run_job(
    files: Arc<dyn FsService>,
    job: TrainingJob,
    job_id: JobId,
    progress_tx: watch::Sender<finetune::Progress>,
    cancel: Arc<AtomicU8>,
    job_handle: Option<JobHandle>,
) -> Result<TrainingResult, TrainingError> {
    let workspace_dir = crate::file_mgr::schema::workspace_dir_for(files.root(), &job.workspace_id);
    // First candidate = planned extractor; the finetune resolver logs any fallback.
    let backbone_basename = job
        .backbone_candidates
        .candidates
        .first()
        .and_then(|c| c.path.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<unknown>".to_string());

    // Open under TrainingError so an unwritable training_logs surfaces
    // as a typed failure; re-wrap the writer's io::Error with the jsonl path.
    let log =
        match JsonlEventLog::<TrainEvent>::open(&workspace_dir, TRAINING_LOGS_DIR_NAME, job_id) {
            Ok(l) => Arc::new(Mutex::new(l)),
            Err(source) => {
                let log_path = workspace_dir
                    .join(TRAINING_LOGS_DIR_NAME)
                    .join(format!("{job_id}.jsonl"))
                    .display()
                    .to_string();
                let e = TrainingError::Io {
                    path: log_path,
                    source,
                };
                // No JSONL line possible yet; surface on the SSE bridge.
                if let Some(h) = job_handle {
                    h.fail(e.to_string());
                }
                return Err(e);
            }
        };
    let stage = Arc::new(Mutex::new(Stage::Prepare));
    // Cloned into the spawn_blocking closures; consumed at terminal via
    // `Arc::try_unwrap` once all child clones have dropped.
    let handle_arc: Option<Arc<JobHandle>> = job_handle.map(Arc::new);

    let head_id = job.head_id;
    let cfg_snapshot = job.training_cfg.clone();
    emit_train_event(
        &log,
        handle_arc.as_deref(),
        &stage,
        TrainEvent::JobSubmitted {
            head_id,
            cfg: cfg_snapshot,
            backbone: backbone_basename,
        },
    );
    emit_train_event(&log, handle_arc.as_deref(), &stage, TrainEvent::JobRunning);

    let result = run_job_inner(
        files,
        job,
        progress_tx,
        cancel.clone(),
        log.clone(),
        stage.clone(),
        handle_arc.clone(),
    )
    .await;

    // Build the terminal event before tearing down the JobHandle so the
    // SSE consumer sees the rich payload before the flat transition.
    let final_event = match &result {
        Ok(r) => TrainEvent::JobCompleted { result: r.clone() },
        Err(TrainingError::Cancelled)
        | Err(TrainingError::Finetune(finetune::FinetuneError::Cancelled)) => {
            let stage_now = *stage.lock();
            let reason = match cancel.load(Ordering::SeqCst) {
                CANCEL_SHUTDOWN => CancelReason::Shutdown,
                // CANCEL_NONE (finetune surfaced its own Cancelled) biases
                // to operator, avoiding mis-attribution to shutdown.
                _ => CancelReason::Operator,
            };
            TrainEvent::JobCancelled {
                stage: stage_now,
                reason,
            }
        }
        Err(e) => {
            let stage_now = *stage.lock();
            TrainEvent::JobFailed {
                stage: stage_now,
                severity: severity_from_error(e),
                error: e.to_string(),
                payload: fail_payload_from_error(e, Some(&workspace_dir)),
            }
        }
    };
    emit_train_event(&log, handle_arc.as_deref(), &stage, final_event);

    // All clones were held by the (already-awaited) spawn_blocking
    // closures, so `try_unwrap` should succeed; an `Err` means a clone
    // leaked into a detached future, so fall back to `JobHandle::Drop`
    // marking `Failed`.
    if let Some(handle_arc) = handle_arc {
        match Arc::try_unwrap(handle_arc) {
            Ok(handle) => match &result {
                Ok(r) => handle.succeed(Some(RegistryJobResult::Train {
                    head_id: r.head_id,
                    sha256: r.head_sha256.clone(),
                    n_classes: r.n_classes,
                })),
                Err(TrainingError::Cancelled)
                | Err(TrainingError::Finetune(finetune::FinetuneError::Cancelled)) => {
                    handle.cancel()
                }
                Err(e) => handle.fail(e.to_string()),
            },
            Err(arc) => {
                tracing::warn!(
                    target: "training",
                    job_id = %job_id,
                    "JobHandle still shared at terminal",
                );
                drop(arc);
            }
        }
    }

    // Force mimalloc to release the just-freed training pages: the lazy
    // purge timer is allocation-driven and would otherwise keep the
    // 256 MiB feature buffer resident on an idle daemon for minutes.
    // Non-fatal RSS hygiene; no-op without `mimalloc`.
    if let Err(err) = tokio::task::spawn_blocking(crate::allocator::release_to_os).await {
        tracing::debug!(
            target: "training",
            job_id = %job_id,
            err = %err,
            "post-training allocator release_to_os did not complete",
        );
    }

    result
}

/// Fan one [`TrainEvent`] out to the stage tracker, the durable JSONL
/// log, and the SSE broadcast. Failures are warned, never returned: a
/// failed log write must not promote a successful run to failed.
fn emit_train_event(
    log: &Arc<Mutex<JsonlEventLog<TrainEvent>>>,
    handle: Option<&JobHandle>,
    stage: &Arc<Mutex<Stage>>,
    event: TrainEvent,
) {
    if let TrainEvent::PhaseStarted { phase } = &event {
        *stage.lock() = *phase;
    }
    if let (Some(h), TrainEvent::EpochCompleted { epoch, epochs, .. }) = (handle, &event) {
        h.update_progress(crate::file_mgr::JobProgress {
            done: u64::from(*epoch),
            total: Some(u64::from(*epochs)),
        });
    }
    if let Err(err) = log.lock().emit(&event) {
        tracing::warn!(target: "training", err = %err, "training: log emit failed");
    }
    if let Some(h) = handle {
        match serde_json::to_string(&event) {
            Ok(s) => h.append_log(s),
            Err(e) => {
                tracing::warn!(target: "training", err = %e, "training: SSE event serialize failed")
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_job_inner(
    files: Arc<dyn FsService>,
    job: TrainingJob,
    progress_tx: watch::Sender<finetune::Progress>,
    cancel: Arc<AtomicU8>,
    log: Arc<Mutex<JsonlEventLog<TrainEvent>>>,
    stage: Arc<Mutex<Stage>>,
    handle: Option<Arc<JobHandle>>,
) -> Result<TrainingResult, TrainingError> {
    let workspace = job.workspace_id;

    emit_train_event(
        &log,
        handle.as_deref(),
        &stage,
        TrainEvent::PhaseStarted {
            phase: Stage::Prepare,
        },
    );

    // Cached `summary` re-confirms the workspace exists without walking
    // `datasets/`; the scan inside `finetune::run` enforces directory
    // shape and surfaces bad entries as `DatasetRead`.
    let files_for_summary = files.clone();
    tokio::task::spawn_blocking(move || files_for_summary.summary(&workspace))
        .await?
        .map_err(TrainingError::Fs)?;
    let dataset_root = crate::file_mgr::schema::workspace_dir_for(files.root(), &workspace)
        .join(DATASETS_DIR_NAME);
    let stat_root = dataset_root.clone();
    let md = tokio::task::spawn_blocking(move || std::fs::symlink_metadata(&stat_root))
        .await?
        .map_err(|source| TrainingError::Io {
            path: dataset_root.display().to_string(),
            source,
        })?;
    if !md.is_dir() {
        return Err(TrainingError::InvalidConfig(format!(
            "dataset root {} is not a directory",
            dataset_root.display(),
        )));
    }

    // Stage output under `<workspace_dir>/.tmp/`: same FS as `heads/` so
    // `publish_trained_head`'s rename is intra-FS POSIX-atomic. The
    // tempdir guard auto-removes on exit; the rename moves the `.mpk`
    // out, so an empty tempdir is the success state.
    let workspace_tmpdir = files.workspace_tmpdir(&workspace);
    let workspace_tmpdir_for_create = workspace_tmpdir.clone();
    tokio::task::spawn_blocking(move || std::fs::create_dir_all(&workspace_tmpdir_for_create))
        .await?
        .map_err(|source| TrainingError::Io {
            path: workspace_tmpdir.display().to_string(),
            source,
        })?;
    let workspace_tmpdir_for_temp = workspace_tmpdir.clone();
    let output_temp =
        tokio::task::spawn_blocking(move || tempfile::tempdir_in(&workspace_tmpdir_for_temp))
            .await?
            .map_err(|source| TrainingError::Io {
                path: workspace_tmpdir.display().to_string(),
                source,
            })?;
    let output_head = output_temp.path().join(format!("{}.mpk", job.head_id));

    if cancel_requested(&cancel) {
        return Err(TrainingError::Cancelled);
    }

    let ft_cfg = finetune::FinetuneConfig {
        data: dataset_root.clone(),
        backbones: job.backbone_candidates.clone(),
        serving_backbone: job.serving_backbone.clone(),
        init_head: None,
        out: output_head.clone(),
        epochs: job.training_cfg.epochs as usize,
        batch: job.training_cfg.batch_size as usize,
        lr: job.training_cfg.learning_rate,
        // 0.0 disables the split; `(0.0, 1.0)` runs validation and
        // publishes the best-val-acc epoch (val_loss breaks ties).
        val_split: job.training_cfg.validation_split,
        seed: job.training_cfg.seed.unwrap_or(42),
    };
    let cancel_for_run = cancel.clone();
    let progress_for_run = progress_tx.clone();
    let log_for_run = log.clone();
    let handle_for_run = handle.clone();
    let stage_for_run = stage.clone();
    let output = tokio::task::spawn_blocking(move || {
        let progress = |p: &finetune::Progress| {
            let _ = progress_for_run.send(p.clone());
        };
        let event_cb = |e: finetune::Event| {
            emit_train_event(
                &log_for_run,
                handle_for_run.as_deref(),
                &stage_for_run,
                e.into(),
            );
        };
        let cancel_fn = || cancel_requested(&cancel_for_run);
        finetune::run(&ft_cfg, &progress, &event_cb, &cancel_fn)
    })
    .await??;

    if cancel_requested(&cancel) {
        return Err(TrainingError::Cancelled);
    }

    emit_train_event(
        &log,
        handle.as_deref(),
        &stage,
        TrainEvent::PhaseStarted {
            phase: Stage::Publish,
        },
    );

    let mpk_path_for_sha = output.head_mpk.clone();
    let head_sha256 =
        tokio::task::spawn_blocking(move || sha256_file_streaming(&mpk_path_for_sha)).await??;
    let mpk_path_for_meta = output.head_mpk.clone();
    let size_bytes = tokio::task::spawn_blocking(move || std::fs::metadata(&mpk_path_for_meta))
        .await?
        .map_err(|source| TrainingError::Io {
            path: output.head_mpk.display().to_string(),
            source,
        })?
        .len();

    let n_classes_u32 = u32::try_from(output.classes.len()).map_err(|_| {
        TrainingError::InvalidConfig(format!(
            "trained head has {} classes; exceeds u32 cap",
            output.classes.len(),
        ))
    })?;

    let manifest = HeadManifest {
        head_id: job.head_id,
        workspace_id: workspace,
        workspace_revision: job.workspace_revision.clone(),
        sha256: head_sha256.clone(),
        n_classes: n_classes_u32,
        size_bytes,
        created_at: now_rfc3339(),
        labels: output.classes.clone(),
    };
    let pending = PendingHead {
        head_id: job.head_id,
        mpk_tempfile: output.head_mpk.clone(),
        manifest,
    };

    // A cancel landing in the sha256+metadata await window would
    // otherwise still publish + Completed.
    if cancel_requested(&cancel) {
        return Err(TrainingError::Cancelled);
    }

    // Publish into the sliding-window rotation; the primitive holds the
    // per-workspace mutation mutex.
    let files_for_publish = files.clone();
    tokio::task::spawn_blocking(move || {
        files_for_publish.publish_trained_head(&workspace, pending)
    })
    .await??;

    // Emit only after the rotation returned: its absence in a JSONL
    // transcript signals that publish did not commit.
    emit_train_event(
        &log,
        handle.as_deref(),
        &stage,
        TrainEvent::HeadPublished {
            head_id: job.head_id,
            head_sha256: head_sha256.clone(),
            size_bytes,
            n_classes: n_classes_u32,
            classes: output.classes.clone(),
            workspace_revision: job.workspace_revision.clone(),
        },
    );

    let result = TrainingResult {
        head_id: job.head_id,
        head_sha256,
        n_classes: n_classes_u32,
        classes: output.classes,
        final_train_acc: output.final_train_acc,
        final_val_acc: output.final_val_acc,
    };

    // `output_temp` drops here; the `.mpk` was already renamed out, so
    // remaining residue (empty dir + sibling labels.txt) is safe to drop.
    Ok(result)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixtures use direct file writes

    use super::*;
    use crate::common::ids::{HeadId, WorkspaceId};
    use std::fs;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    fn synthetic_entry(
        workspace: &WorkspaceId,
        finished_state: Option<(JobState, Option<Instant>)>,
    ) -> Arc<JobEntry> {
        let job_id = JobId::new();
        let initial = finetune::Progress {
            phase: Stage::Prepare,
            current: 0,
            total: 0,
            message: "synthetic".into(),
            metrics: None,
        };
        let (_tx, rx) = watch::channel(initial);
        let (state, finished_at_instant) = finished_state.unwrap_or((JobState::Running, None));
        Arc::new(JobEntry {
            job_id,
            workspace_id: *workspace,
            started_at: now_rfc3339(),
            progress: rx,
            cancel: Arc::new(AtomicU8::new(CANCEL_NONE)),
            core: Mutex::new(JobCore {
                state,
                result: None,
                error: None,
                finished_at: finished_at_instant.map(|_| now_rfc3339()),
                finished_at_instant,
            }),
        })
    }

    #[test]
    fn training_result_serialization_carries_no_filesystem_path() {
        let result = TrainingResult {
            head_id: HeadId::new(),
            head_sha256: "0".repeat(64),
            n_classes: 2,
            classes: vec!["a".into(), "b".into()],
            final_train_acc: 0.9,
            final_val_acc: 0.85,
        };
        let v = serde_json::to_value(&result).expect("serialize TrainingResult");
        let body = v
            .as_object()
            .expect("TrainingResult serializes as a JSON object");
        let allowed: std::collections::BTreeSet<&str> = [
            "head_id",
            "head_sha256",
            "n_classes",
            "classes",
            "final_train_acc",
            "final_val_acc",
        ]
        .into_iter()
        .collect();
        let actual: std::collections::BTreeSet<&str> = body.keys().map(String::as_str).collect();
        assert_eq!(
            actual, allowed,
            "TrainingResult must serialize exactly {allowed:?}; got {actual:?}",
        );
        for forbidden in [
            "head_mpk_path",
            "labels_path",
            "head_path",
            "weights_path",
            "path",
            "dataset_path",
        ] {
            assert!(
                body.get(forbidden).is_none(),
                "TrainingResult must not carry filesystem path field `{forbidden}`; body={v}",
            );
        }
    }

    #[test]
    fn reap_finished_drops_stale_keeps_fresh_and_running() {
        let reg = JobRegistry::new();
        let workspace = WorkspaceId::new();

        let now = Instant::now();
        let stale = synthetic_entry(
            &workspace,
            Some((JobState::Completed, Some(now - Duration::from_secs(7200)))),
        );
        let fresh = synthetic_entry(
            &workspace,
            Some((JobState::Completed, Some(now - Duration::from_secs(60)))),
        );
        let running = synthetic_entry(&workspace, None);

        reg.jobs.insert(stale.job_id, stale.clone());
        reg.jobs.insert(fresh.job_id, fresh.clone());
        reg.jobs.insert(running.job_id, running.clone());

        let n = reg.reap_finished(Duration::from_secs(3600));
        assert_eq!(n, 1, "exactly one stale entry expected");
        assert!(!reg.jobs.contains_key(&stale.job_id));
        assert!(reg.jobs.contains_key(&fresh.job_id));
        assert!(reg.jobs.contains_key(&running.job_id));
    }

    /// Guards chronological-instant ordering: within a second a lexical
    /// RFC3339 sort would invert `"...:00Z"` and `"...:00.5Z"`.
    #[test]
    fn list_for_workspace_orders_by_instant_not_lexical_string() {
        let reg = JobRegistry::new();
        let workspace = WorkspaceId::new();
        let mut senders = Vec::new();
        let mut entry_at = |started_at: &str| {
            let (tx, rx) = watch::channel(finetune::Progress {
                phase: Stage::Prepare,
                current: 0,
                total: 0,
                message: "synthetic".into(),
                metrics: None,
            });
            senders.push(tx);
            Arc::new(JobEntry {
                job_id: JobId::new(),
                workspace_id: workspace,
                started_at: started_at.to_string(),
                progress: rx,
                cancel: Arc::new(AtomicU8::new(CANCEL_NONE)),
                core: Mutex::new(JobCore {
                    state: JobState::Running,
                    result: None,
                    error: None,
                    finished_at: None,
                    finished_at_instant: None,
                }),
            })
        };
        // Earlier instant but lexically greater (trailing 'Z').
        let earlier = entry_at("2026-01-01T00:00:00Z");
        // 0.5s later but lexically smaller (fraction starts with '.').
        let later = entry_at("2026-01-01T00:00:00.5Z");
        // Insert later-first so a passing sort can't be insertion order.
        reg.jobs.insert(later.job_id, later.clone());
        reg.jobs.insert(earlier.job_id, earlier.clone());

        let out = reg.list_for_workspace(&workspace);
        let order: Vec<&str> = out.iter().map(|v| v.started_at.as_str()).collect();
        assert_eq!(
            order,
            ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00.5Z"],
            "oldest-first by instant; a lexical sort would invert these",
        );
    }

    #[test]
    fn cancel_all_for_shutdown_sets_flag_on_running_only() {
        let reg = JobRegistry::new();
        let workspace = WorkspaceId::new();

        let running1 = synthetic_entry(&workspace, None);
        let running2 = synthetic_entry(&workspace, None);
        let completed = synthetic_entry(
            &workspace,
            Some((JobState::Completed, Some(Instant::now()))),
        );
        let cancelled = synthetic_entry(
            &workspace,
            Some((JobState::Cancelled, Some(Instant::now()))),
        );
        // Operator reason pre-set: shutdown drain must not overwrite it.
        let running_pre_cancelled = synthetic_entry(&workspace, None);
        running_pre_cancelled
            .cancel
            .store(CANCEL_OPERATOR, Ordering::SeqCst);

        reg.jobs.insert(running1.job_id, running1.clone());
        reg.jobs.insert(running2.job_id, running2.clone());
        reg.jobs.insert(completed.job_id, completed.clone());
        reg.jobs.insert(cancelled.job_id, cancelled.clone());
        reg.jobs
            .insert(running_pre_cancelled.job_id, running_pre_cancelled.clone());

        let n = reg.cancel_all_for_shutdown();
        assert_eq!(n, 2, "exactly two newly-signalled jobs expected");
        assert_eq!(
            running1.cancel.load(Ordering::SeqCst),
            CANCEL_SHUTDOWN,
            "drain stamps SHUTDOWN reason on freshly-cancelled jobs",
        );
        assert_eq!(running2.cancel.load(Ordering::SeqCst), CANCEL_SHUTDOWN);
        assert_eq!(
            completed.cancel.load(Ordering::SeqCst),
            CANCEL_NONE,
            "terminal jobs are skipped by the drain",
        );
        assert_eq!(cancelled.cancel.load(Ordering::SeqCst), CANCEL_NONE);
        assert_eq!(
            running_pre_cancelled.cancel.load(Ordering::SeqCst),
            CANCEL_OPERATOR,
            "operator-cancelled jobs keep their OPERATOR reason; shutdown does not overwrite",
        );

        let n2 = reg.cancel_all_for_shutdown();
        assert_eq!(n2, 0, "idempotent on repeat shutdown drain");
    }

    #[test]
    fn active_count_only_counts_running_jobs() {
        let reg = JobRegistry::new();
        let workspace = WorkspaceId::new();
        assert_eq!(reg.active_count(), 0);

        let r1 = synthetic_entry(&workspace, None);
        let r2 = synthetic_entry(&workspace, None);
        let f = synthetic_entry(
            &workspace,
            Some((JobState::Completed, Some(Instant::now()))),
        );
        reg.jobs.insert(r1.job_id, r1.clone());
        reg.jobs.insert(r2.job_id, r2.clone());
        reg.jobs.insert(f.job_id, f.clone());
        assert_eq!(reg.active_count(), 2);
    }

    /// `scan_dataset` walks each class folder recursively (non-hidden
    /// `.wav` only); hidden entries are skipped.
    #[test]
    fn class_file_discovery_walks_recursively() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        for cls in ["cat", "dog"] {
            fs::create_dir_all(root.join(cls)).unwrap();
            for i in 0..3 {
                fs::write(root.join(cls).join(format!("s{i}.wav")), b"stub").unwrap();
            }
        }
        // Hidden root entry must be ignored.
        fs::write(root.join(".README"), b"meta").unwrap();
        fs::create_dir_all(root.join("cat").join("nested")).unwrap();
        fs::write(root.join("cat").join("nested").join("x.wav"), b"deeper").unwrap();

        let (classes, examples) = finetune::scan_dataset_for_test(root);
        assert_eq!(classes, vec!["cat".to_string(), "dog".to_string()]);
        // 3 direct + 1 nested = 4 for cat; 3 for dog.
        assert_eq!(examples.len(), 7);
        for (path, _label) in &examples {
            assert!(path.extension().is_some_and(|e| e == "wav"));
        }
        // Nested wav associated with `cat` (label 0, cat sorts first).
        let cat_count = examples.iter().filter(|(_, l)| *l == 0).count();
        let dog_count = examples.iter().filter(|(_, l)| *l == 1).count();
        assert_eq!(cat_count, 4);
        assert_eq!(dog_count, 3);
    }

    /// Mid-walk file disappearance surfaces as `DatasetRead`.
    #[test]
    fn dataset_read_failure_surfaces_typed_error() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("cat")).unwrap();
        fs::write(root.join("cat").join("a.wav"), b"stub").unwrap();
        let (_classes, examples) = finetune::scan_dataset_for_test(root);
        assert_eq!(examples.len(), 1);
        fs::remove_file(root.join("cat").join("a.wav")).unwrap();
        let err = match std::fs::File::open(&examples[0].0) {
            Ok(_) => panic!("expected ENOENT"),
            Err(e) => TrainingError::DatasetRead {
                path: examples[0].0.display().to_string(),
                reason: e.to_string(),
            },
        };
        match err {
            TrainingError::DatasetRead { path, .. } => assert!(path.ends_with("a.wav")),
            other => panic!("expected DatasetRead; got {other:?}"),
        }
    }

    /// `From<FinetuneError>` lifts inner `BadDataset` to the typed
    /// `TrainingError::BadDataset`.
    #[test]
    fn finetune_bad_dataset_translates_to_training_bad_dataset() {
        let inner = finetune::FinetuneError::BadDataset {
            path: "/ws/datasets/empty".into(),
            reason: "no class folders".into(),
        };
        let outer: TrainingError = inner.into();
        match outer {
            TrainingError::BadDataset { path, reason } => {
                assert_eq!(path, "/ws/datasets/empty");
                assert_eq!(reason, "no class folders");
            }
            other => panic!("expected TrainingError::BadDataset, got {other:?}"),
        }
    }

    #[test]
    fn finetune_dataset_read_translates_to_training_dataset_read() {
        let inner = finetune::FinetuneError::DatasetRead {
            path: "/ws/datasets/cat/a.wav".into(),
            reason: "ENOENT".into(),
        };
        let outer: TrainingError = inner.into();
        match outer {
            TrainingError::DatasetRead { path, reason } => {
                assert_eq!(path, "/ws/datasets/cat/a.wav");
                assert_eq!(reason, "ENOENT");
            }
            other => panic!("expected TrainingError::DatasetRead, got {other:?}"),
        }
    }

    /// `BadDataset` -> 400, `DatasetRead` -> 500; `Finetune(_)`
    /// delegates so dataset-quality variants keep their 400.
    #[test]
    fn training_error_kinds_classify_correctly() {
        use crate::common::error::{Categorized, ErrorKind};
        let bad = TrainingError::BadDataset {
            path: "/x".into(),
            reason: "y".into(),
        };
        assert_eq!(bad.kind(), ErrorKind::UserInput);
        let read = TrainingError::DatasetRead {
            path: "/x".into(),
            reason: "y".into(),
        };
        assert_eq!(read.kind(), ErrorKind::Internal);

        let wrapped_bad = TrainingError::Finetune(finetune::FinetuneError::EmptyClassAfterScan {
            class: "cat".into(),
            per_class_kept: vec![],
        });
        assert_eq!(wrapped_bad.kind(), ErrorKind::UserInput);
        let wrapped_panic = TrainingError::Finetune(finetune::FinetuneError::Panic("oops".into()));
        assert_eq!(wrapped_panic.kind(), ErrorKind::Internal);
    }

    /// FD usage is bounded by `batch_size * parallel_loaders`: scan
    /// returns `(PathBuf, label)` pairs without opening any file.
    #[test]
    fn lazy_fd_bounded_no_open_during_scan() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("cat")).unwrap();
        fs::create_dir_all(root.join("dog")).unwrap();
        for cls in ["cat", "dog"] {
            for i in 0..100 {
                fs::write(root.join(cls).join(format!("s{i}.wav")), b"x").unwrap();
            }
        }
        let (classes, examples) = finetune::scan_dataset_for_test(root);
        assert_eq!(classes.len(), 2);
        assert_eq!(examples.len(), 200);
        for (p, _) in &examples {
            assert!(p.is_file(), "scan returned a non-file path: {p:?}");
        }
    }

    /// One JSONL line per `emit`, each carrying `seq`, `at`, `kind`
    /// plus event-specific fields (the page reader parses this shape).
    #[test]
    fn train_job_log_writes_one_jsonl_line_per_event() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace_dir = tmp.path();
        let job_id = JobId::new();
        let mut log =
            JsonlEventLog::<TrainEvent>::open(workspace_dir, TRAINING_LOGS_DIR_NAME, job_id)
                .expect("open log");
        log.emit(&TrainEvent::JobRunning).expect("running");
        log.emit(&TrainEvent::EpochCompleted {
            epoch: 3,
            epochs: 5,
            train_loss: 0.42,
            train_acc: 0.91,
            val_acc: 0.88,
            best_val_acc: 0.89,
            val_loss: 0.31,
            best_val_loss: 0.29,
            lr: 0.01,
            elapsed_ms: 750,
        })
        .expect("epoch");
        log.emit(&TrainEvent::JobCancelled {
            stage: Stage::Train,
            reason: CancelReason::Operator,
        })
        .expect("cancelled");
        drop(log);

        let path = workspace_dir
            .join(TRAINING_LOGS_DIR_NAME)
            .join(format!("{job_id}.jsonl"));
        let body = std::fs::read_to_string(&path).expect("read log");
        let lines: Vec<_> = body.lines().collect();
        assert_eq!(lines.len(), 3, "one JSONL line per emit() call");

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["seq"], 1);
        assert_eq!(first["kind"], "job_running");
        assert!(first["at"].as_str().unwrap().ends_with('Z'));

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["seq"], 2);
        assert_eq!(second["kind"], "epoch_completed");
        assert_eq!(second["epoch"], 3);
        assert_eq!(second["epochs"], 5);
        assert_eq!(second["train_acc"], 0.91);
        assert_eq!(second["val_acc"], 0.88);
        // Pin the finite branch of `serialize_finite_or_null_f64`.
        assert_eq!(second["val_loss"], 0.31);
        assert_eq!(second["best_val_loss"], 0.29);
        assert_eq!(second["lr"], 0.01);
        assert_eq!(second["elapsed_ms"], 750);

        let third: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(third["seq"], 3);
        assert_eq!(third["kind"], "job_cancelled");
        assert_eq!(third["stage"], "train");
        assert_eq!(third["reason"], "operator");
    }

    /// `open` enforces `LOG_RETENTION_KEEP_COUNT` on `training_logs/`:
    /// the just-opened file survives; older siblings are unlinked
    /// oldest-first.
    #[test]
    fn train_job_log_open_enforces_keep_last_n() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace_dir = tmp.path();
        let dir = workspace_dir.join(TRAINING_LOGS_DIR_NAME);
        std::fs::create_dir_all(&dir).unwrap();
        // One more pre-existing log than the keep cap.
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
        // New log is freshest (no backdate) so it survives.
        let job_id = JobId::new();
        let _log = JsonlEventLog::<TrainEvent>::open(workspace_dir, TRAINING_LOGS_DIR_NAME, job_id)
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

    /// `val_split == 0.0` lands NaN into `final_val_acc`/metrics; the
    /// `serialize_finite_or_null` shims must emit `null` so the page
    /// doesn't 500 (serde_json refuses non-finite floats).
    #[test]
    fn jobview_serializes_with_nan_val_acc() {
        let metrics = finetune::EpochMetrics {
            epoch: 1,
            epochs: 1,
            train_loss: 0.5,
            train_acc: 0.9,
            val_acc: f32::NAN,
            best_val_acc: f32::NAN,
            val_loss: f64::NAN,
            best_val_loss: f64::INFINITY,
        };
        let v = serde_json::to_value(metrics).expect("EpochMetrics serialises");
        assert!(v["val_acc"].is_null(), "EpochMetrics::val_acc NaN -> null");
        assert!(v["best_val_acc"].is_null());
        assert!(
            v["val_loss"].is_null(),
            "EpochMetrics::val_loss NaN -> null"
        );
        assert!(
            v["best_val_loss"].is_null(),
            "EpochMetrics::best_val_loss +inf (no-best-yet sentinel) -> null",
        );

        let result = TrainingResult {
            head_id: HeadId::new(),
            head_sha256: "0".repeat(64),
            n_classes: 2,
            classes: vec!["a".into(), "b".into()],
            final_train_acc: 0.9,
            final_val_acc: f32::NAN,
        };
        let v = serde_json::to_value(&result).expect("TrainingResult serialises");
        assert!(
            v["final_val_acc"].is_null(),
            "TrainingResult::final_val_acc NaN -> null",
        );
    }

    /// NaN `val_acc`/`best_val_acc` serialise to JSON `null`, not the
    /// `NaN` literal serde_json refuses.
    #[test]
    fn epoch_completed_nan_val_acc_serializes_to_null() {
        let event = TrainEvent::EpochCompleted {
            epoch: 1,
            epochs: 1,
            train_loss: 0.5,
            train_acc: 0.6,
            val_acc: f32::NAN,
            best_val_acc: f32::NAN,
            val_loss: f64::NAN,
            best_val_loss: f64::INFINITY,
            lr: 0.01,
            elapsed_ms: 100,
        };
        let v = serde_json::to_value(&event).expect("serialize");
        assert_eq!(v["kind"], "epoch_completed");
        assert!(v["val_acc"].is_null(), "NaN val_acc must be JSON null");
        assert!(
            v["best_val_acc"].is_null(),
            "NaN best_val_acc must be JSON null",
        );
        assert!(v["val_loss"].is_null(), "NaN val_loss must be JSON null");
        assert!(
            v["best_val_loss"].is_null(),
            "+inf best_val_loss (no-best sentinel) must be JSON null",
        );
    }

    /// `From<finetune::Event>` lifts each algorithmic variant 1:1.
    #[test]
    fn finetune_event_lifts_to_train_event() {
        let inner = finetune::Event::TrainSplit {
            train_n: 80,
            val_n: 20,
        };
        let outer: TrainEvent = inner.into();
        match outer {
            TrainEvent::TrainSplit { train_n, val_n } => {
                assert_eq!(train_n, 80);
                assert_eq!(val_n, 20);
            }
            other => panic!("expected TrainSplit, got {other:?}"),
        }

        let inner = finetune::Event::EpochCompleted {
            epoch: 2,
            epochs: 4,
            train_loss: 0.1,
            train_acc: 0.95,
            val_acc: 0.9,
            best_val_acc: 0.92,
            val_loss: 0.12,
            best_val_loss: 0.10,
            lr: 0.005,
            elapsed_ms: 333,
        };
        let outer: TrainEvent = inner.into();
        match outer {
            TrainEvent::EpochCompleted {
                epoch,
                epochs,
                lr,
                val_loss,
                best_val_loss,
                ..
            } => {
                assert_eq!(epoch, 2);
                assert_eq!(epochs, 4);
                assert!((lr - 0.005).abs() < f32::EPSILON);
                assert!((val_loss - 0.12).abs() < 1e-9);
                assert!((best_val_loss - 0.10).abs() < 1e-9);
            }
            other => panic!("expected EpochCompleted, got {other:?}"),
        }

        let inner = finetune::Event::TrainCompleted {
            epochs_run: 4,
            total_elapsed_ms: 1234,
            best_val_epoch: Some(2),
            best_val_acc: Some(0.92),
            best_val_loss: Some(0.10),
        };
        let outer: TrainEvent = inner.into();
        match outer {
            TrainEvent::TrainCompleted {
                epochs_run,
                best_val_epoch,
                best_val_acc,
                best_val_loss,
                ..
            } => {
                assert_eq!(epochs_run, 4);
                assert_eq!(best_val_epoch, Some(2));
                assert_eq!(best_val_acc, Some(0.92));
                assert_eq!(best_val_loss, Some(0.10));
            }
            other => panic!("expected TrainCompleted, got {other:?}"),
        }
    }

    /// `fail_payload_from_error` discriminates each error variant onto
    /// its typed `FailPayload` shape.
    #[test]
    fn fail_payload_lifts_each_error_variant() {
        let err = TrainingError::BadDataset {
            path: "/ws/datasets".into(),
            reason: "no class folders".into(),
        };
        match fail_payload_from_error(&err, None) {
            FailPayload::BadDataset { path, reason } => {
                assert_eq!(path, "/ws/datasets");
                assert_eq!(reason, "no class folders");
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }

        let err = TrainingError::Finetune(finetune::FinetuneError::EmptyClassAfterScan {
            class: "cat".into(),
            per_class_kept: vec![("cat".into(), 0), ("dog".into(), 5)],
        });
        match fail_payload_from_error(&err, None) {
            FailPayload::EmptyClass {
                class,
                per_class_kept,
            } => {
                assert_eq!(class, "cat");
                assert_eq!(per_class_kept.len(), 2);
            }
            other => panic!("expected EmptyClass, got {other:?}"),
        }

        let err = TrainingError::Finetune(finetune::FinetuneError::DropRatioExceeded {
            dropped: 30,
            total: 100,
            ratio: 0.30,
            max_ratio: 0.1,
            per_class_kept: vec![("cat".into(), 35), ("dog".into(), 35)],
            per_class_dropped: vec![("cat".into(), 15), ("dog".into(), 15)],
        });
        match fail_payload_from_error(&err, None) {
            FailPayload::DropRatioExceeded {
                dropped,
                total,
                threshold,
                per_class_kept,
                per_class_dropped,
            } => {
                assert_eq!(dropped, 30);
                assert_eq!(total, 100);
                assert!((threshold - 0.1).abs() < f32::EPSILON);
                assert_eq!(per_class_kept.len(), 2);
                assert_eq!(per_class_dropped.len(), 2);
            }
            other => panic!("expected DropRatioExceeded, got {other:?}"),
        }

        // Operator_fixable: must keep class+tables, NOT collapse to Internal.
        let err = TrainingError::Finetune(finetune::FinetuneError::PerClassDropExceeded {
            class: "dog".into(),
            dropped: 8,
            total: 10,
            ratio: 0.80,
            max_ratio: 0.5,
            per_class_kept: vec![("cat".into(), 50), ("dog".into(), 2)],
            per_class_dropped: vec![("cat".into(), 0), ("dog".into(), 8)],
        });
        assert_eq!(severity_from_error(&err), Severity::OperatorFixable);
        match fail_payload_from_error(&err, None) {
            FailPayload::PerClassDropExceeded {
                class,
                dropped,
                total,
                threshold,
                per_class_kept,
                per_class_dropped,
            } => {
                assert_eq!(class, "dog");
                assert_eq!(dropped, 8);
                assert_eq!(total, 10);
                assert!((threshold - 0.5).abs() < f32::EPSILON);
                assert_eq!(per_class_kept.len(), 2);
                assert_eq!(per_class_dropped.len(), 2);
            }
            other => panic!("expected PerClassDropExceeded, got {other:?}"),
        }

        let err = TrainingError::Io {
            path: "/ws/.tmp".into(),
            source: std::io::Error::other("disk full"),
        };
        match fail_payload_from_error(&err, None) {
            FailPayload::Io { path, detail } => {
                assert_eq!(path, "/ws/.tmp");
                assert!(detail.contains("disk full"));
            }
            other => panic!("expected Io, got {other:?}"),
        }

        // Severity discrimination: BadDataset operator_fixable, Io internal.
        let bad = TrainingError::BadDataset {
            path: "x".into(),
            reason: "y".into(),
        };
        assert_eq!(severity_from_error(&bad), Severity::OperatorFixable);
        let io = TrainingError::Io {
            path: "x".into(),
            source: std::io::Error::other("y"),
        };
        assert_eq!(severity_from_error(&io), Severity::Internal);

        // Pins the wire payload of Internal-classified arms against
        // changes to `internal_payload` or `Cancelled`'s Display.

        // Cancelled is the inline outlier: literal `"cancelled"`.
        let cancelled = TrainingError::Cancelled;
        match fail_payload_from_error(&cancelled, None) {
            FailPayload::Internal { detail } => assert_eq!(detail, "cancelled"),
            other => panic!("expected Internal{{cancelled}}, got {other:?}"),
        }

        // File arm uses inner Display, which carries the workspace id.
        let file_err = TrainingError::File(crate::file_mgr::FileError::NotFound("abc-123".into()));
        match fail_payload_from_error(&file_err, None) {
            FailPayload::Internal { detail } => {
                assert!(
                    detail.contains("abc-123"),
                    "File arm should carry inner FileError diagnostic; got {detail:?}",
                );
            }
            other => panic!("expected Internal from File arm, got {other:?}"),
        }

        // JobNotFound/WrongWorkspace/JobNotCancellable use the OUTER
        // Display so the operator sees id/state/workspace context.
        let not_found = TrainingError::JobNotFound("job-xyz".into());
        match fail_payload_from_error(&not_found, None) {
            FailPayload::Internal { detail } => {
                assert!(
                    detail.contains("job-xyz"),
                    "JobNotFound arm should carry outer Display; got {detail:?}",
                );
                assert!(
                    detail.starts_with("job not found"),
                    "JobNotFound arm should keep outer 'job not found:' prefix; got {detail:?}",
                );
            }
            other => panic!("expected Internal from JobNotFound, got {other:?}"),
        }
    }

    /// `fail_payload_from_error` strips the workspace_dir prefix from
    /// path-bearing variants; non-matching paths pass through and `None`
    /// disables stripping.
    #[test]
    fn fail_payload_strips_workspace_prefix() {
        use std::path::PathBuf;
        let workspace_dir = PathBuf::from("/var/data/workspaces/abc-123");

        // Under the workspace: stripped to relative.
        let err = TrainingError::BadDataset {
            path: "/var/data/workspaces/abc-123/datasets/cat".into(),
            reason: "empty".into(),
        };
        match fail_payload_from_error(&err, Some(&workspace_dir)) {
            FailPayload::BadDataset { path, .. } => assert_eq!(path, "datasets/cat"),
            other => panic!("expected BadDataset, got {other:?}"),
        }

        let err = TrainingError::Io {
            path: "/var/data/workspaces/abc-123/.tmp/blob".into(),
            source: std::io::Error::other("ENOSPC"),
        };
        match fail_payload_from_error(&err, Some(&workspace_dir)) {
            FailPayload::Io { path, .. } => assert_eq!(path, ".tmp/blob"),
            other => panic!("expected Io, got {other:?}"),
        }

        // Outside the workspace tree: verbatim.
        let err = TrainingError::Io {
            path: "/opt/acoustics/backbone.mpk".into(),
            source: std::io::Error::other("permission denied"),
        };
        match fail_payload_from_error(&err, Some(&workspace_dir)) {
            FailPayload::Io { path, .. } => assert_eq!(path, "/opt/acoustics/backbone.mpk"),
            other => panic!("expected Io, got {other:?}"),
        }

        // `None` disables stripping.
        let err = TrainingError::BadDataset {
            path: "/var/data/workspaces/abc-123/datasets/cat".into(),
            reason: "empty".into(),
        };
        match fail_payload_from_error(&err, None) {
            FailPayload::BadDataset { path, .. } => {
                assert_eq!(path, "/var/data/workspaces/abc-123/datasets/cat");
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }
}
