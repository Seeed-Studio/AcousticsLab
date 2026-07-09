//! On-device fine-tuning of the classifier head: scan a
//! Speech-Commands-style dataset, compute frozen-backbone features once,
//! train only the head, save `head.mpk` + sibling `labels.txt`.
//!
//! Feature extraction resolves the backbone from the SAME ordered candidate
//! catalogue serving uses (first supported+loadable wins): frozen-backbone
//! forwards run on the NPU where serving does (RKNN fp16, ~ms/window), so the
//! head is fit on the exact feature basis it is served against; hosts fall
//! back to batched Burn fp32 forwards.

use crate::common::dims::{BACKBONE_FEATURE_DIM as FEATURE_DIM, NBins, NFrames};
use crate::inference::{BackboneCatalogue, BackbonePipeline, BackboneRef};
use crate::model::{Backbone, Head};
use crate::preproc::Preproc;
use crate::preproc::wav_io::{self, ResamplerCache};
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use thiserror::Error;

const DEFAULT_PROGRESS_EVERY: usize = 500;

/// Momentum recovers most of a larger (divergence-prone) LR's accuracy
/// without its `NumericFailure` risk on the unnormalized 2000-d features.
const HEAD_MOMENTUM: f64 = 0.9;

static RAYON_POOL_INIT: OnceLock<()> = OnceLock::new();

type InnerB = NdArray<f32>;
type AutoB = Autodiff<InnerB>;
type Example = (PathBuf, usize);
type DatasetScan = (Vec<String>, Vec<Example>);
/// Boxed row-major spectrogram, the shape `Preproc::spectrogram` returns.
type Spectrogram = Box<[[f32; NBins::USIZE]; NFrames::USIZE]>;

/// Configuration for a single fine-tune run.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FinetuneConfig {
    pub data: PathBuf,
    /// Ordered extractor candidates, non-empty; first supported+loadable
    /// wins (the serving resolution rule).
    pub backbones: BackboneCatalogue,
    /// Serving's boot-loaded candidate (`None` = inference not running); a
    /// resolution mismatch warns of a train/serve feature-basis divergence.
    pub serving_backbone: Option<BackboneRef>,
    pub init_head: Option<PathBuf>,
    pub out: PathBuf,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f32,
    pub val_split: f32,
    pub seed: u64,
}

impl FinetuneConfig {
    /// Lexical range validation duplicating the api-boundary
    /// [`crate::file_mgr::request_payload::MAX_LEARNING_RATE`] cap so a
    /// caller bypassing the request validator (tests, recovery replay)
    /// can't land a pathological lr that drives weights to ±Inf on the
    /// first SGD step, before any best-epoch snapshot exists.
    pub fn validate(&self) -> Result<(), FinetuneError> {
        use crate::file_mgr::request_payload::{MAX_BATCH_SIZE, MAX_EPOCHS, MAX_LEARNING_RATE};
        if !(1..=MAX_EPOCHS as usize).contains(&self.epochs) {
            return Err(FinetuneError::InvalidConfig(format!(
                "epochs must be in 1..={MAX_EPOCHS}; got {}",
                self.epochs
            )));
        }
        if !(1..=MAX_BATCH_SIZE as usize).contains(&self.batch) {
            return Err(FinetuneError::InvalidConfig(format!(
                "batch must be in 1..={MAX_BATCH_SIZE}; got {}",
                self.batch
            )));
        }
        if !self.lr.is_finite() || self.lr <= 0.0 || self.lr > MAX_LEARNING_RATE {
            return Err(FinetuneError::InvalidConfig(format!(
                "lr must be finite and in (0, {MAX_LEARNING_RATE}]; got {}",
                self.lr
            )));
        }
        if !(self.val_split.is_finite() && self.val_split >= 0.0 && self.val_split < 1.0) {
            return Err(FinetuneError::InvalidConfig(format!(
                "val_split must be finite and in [0, 1); got {}",
                self.val_split
            )));
        }
        if self.backbones.is_empty() {
            return Err(FinetuneError::InvalidConfig(
                "backbones: at least one candidate is required".into(),
            ));
        }
        if let Err((idx, err)) = self.backbones.validate() {
            return Err(FinetuneError::InvalidConfig(format!(
                "backbones: candidate {idx}: {err}"
            )));
        }
        Ok(())
    }
}

/// Pipeline stage tagging `Event::PhaseStarted` + terminal failures so
/// tooling attributes a failure to its step.  Wire snake_case.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Workspace + tempdir setup; runs in the wrapper before `run`.
    Prepare,
    DatasetScan,
    FeatureExtract,
    Train,
    /// `save_mpk_atomic` of head + sibling labels.txt under `.tmp/`.
    Save,
    /// Wrapper rotation publishes the staged `.mpk` into `heads/` after
    /// `run` returns.
    Publish,
}

/// Per-epoch metrics on `Stage::Train` progress + durable
/// `Event::EpochCompleted` lines.  `val_acc`/`best_val_acc` are
/// `f32::NAN` when `val_split == 0.0`, serialising to JSON `null`.
///
/// `best_val_acc` = best val_acc whose snapshot was *successfully saved*,
/// not "best observed": a failed save (ENOSPC/EIO) keeps the prior best,
/// mirroring what `Head::load_mpk(best_path)` resolves to at end-of-run.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub epochs: usize,
    pub train_loss: f64,
    pub train_acc: f32,
    #[serde(serialize_with = "serialize_finite_or_null")]
    pub val_acc: f32,
    #[serde(serialize_with = "serialize_finite_or_null")]
    pub best_val_acc: f32,
    /// Mean validation cross-entropy (same reduction as `train_loss`); the
    /// loss-second tiebreaker among epochs tied on `val_acc`.
    #[serde(serialize_with = "serialize_finite_or_null_f64")]
    pub val_loss: f64,
    /// Val loss of the published best snapshot; same successfully-saved
    /// semantics as `best_val_acc`.
    #[serde(serialize_with = "serialize_finite_or_null_f64")]
    pub best_val_loss: f64,
}

/// Training progress snapshot, forwarded through a `tokio::sync::watch`
/// channel by the daemon-side training crate.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Progress {
    pub phase: Stage,
    pub current: usize,
    pub total: usize,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<EpochMetrics>,
}

impl Progress {
    fn new(phase: Stage, current: usize, total: usize, message: impl Into<String>) -> Self {
        Self {
            phase,
            current,
            total,
            message: message.into(),
            metrics: None,
        }
    }

    fn with_metrics(message: impl Into<String>, metrics: EpochMetrics) -> Self {
        Self {
            phase: Stage::Train,
            current: metrics.epoch,
            total: metrics.epochs,
            message: message.into(),
            metrics: Some(metrics),
        }
    }
}

/// One class entry surfaced in `Event::DatasetScanned`.
#[derive(Clone, Debug, Serialize)]
pub struct ClassCount {
    pub name: String,
    pub n_samples: u64,
}

/// Lossless transcript of `run` milestones, persisted to JSONL +
/// broadcast over SSE; the wrapper (`crate::training`) lifts each variant
/// into a `TrainEvent` superset adding job-lifecycle events.  Distinct
/// from [`Progress`] (latest-snapshot only); both originate at the same
/// emit point so they can never disagree.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    /// Stage transition, just before each stage's work; the wrapper sets
    /// the `current_stage` tracker on terminal payloads from it.
    PhaseStarted {
        phase: Stage,
    },
    DatasetScanned {
        n_classes: u32,
        classes: Vec<ClassCount>,
        n_examples_total: u64,
    },
    /// `kept` + `dropped_*` sum to the input window count only when the
    /// feature-buffer cap is not hit; windows chopped off past the cap are
    /// neither kept nor counted as dropped.  Large `dropped_io` indicates
    /// corrupt `.wav` files.
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
    /// `val_acc`/`best_val_acc` are `null` on the wire when
    /// `val_split == 0.0`.
    EpochCompleted {
        epoch: u32,
        epochs: u32,
        train_loss: f64,
        train_acc: f32,
        #[serde(serialize_with = "serialize_finite_or_null")]
        val_acc: f32,
        #[serde(serialize_with = "serialize_finite_or_null")]
        best_val_acc: f32,
        #[serde(serialize_with = "serialize_finite_or_null_f64")]
        val_loss: f64,
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
}

use super::{
    serialize_finite_or_null, serialize_finite_or_null_f64, serialize_finite_or_null_opt,
    serialize_finite_or_null_opt_f64,
};

/// Result of one successful run.  `final_*_acc` describe the *published*
/// head (highest-val_acc epoch, or last-epoch head when `val_split == 0`).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FinetuneOutput {
    pub head_mpk: PathBuf,
    pub labels_txt: PathBuf,
    pub final_train_acc: f32,
    pub final_val_acc: f32,
    pub classes: Vec<String>,
}

/// Failure shapes from the head fine-tune algorithm.
#[derive(Debug, Error)]
pub enum FinetuneError {
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("cancelled")]
    Cancelled,
    #[error("io {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// Burn `.mpk` load/save error; its `Display` carries path + message.
    #[error(transparent)]
    Model(#[from] crate::model::Error),
    /// Backbone resolution or NPU inference failure; a stringified
    /// [`crate::inference::BackboneError`], whose cfg-gated variants would
    /// make a typed wrap's match surface build-dependent.
    #[error("backbone: {0}")]
    Backbone(String),
    #[error("training panicked: {0}")]
    Panic(String),
    /// Dataset shape rejection at scan time (no class folders,
    /// case-insensitive duplicate labels, unreadable/empty class dirs,
    /// stray non-dir root entries).  400: operator can fix the layout.
    #[error("bad dataset {path}: {reason}")]
    BadDataset {
        /// Root for "no classes"/"duplicate"; offending child otherwise.
        path: String,
        reason: String,
    },
    /// A discovered dataset file vanished/became unreadable mid-scan
    /// (uploads/deletes are allowed during training).  500: unrecoverable.
    #[error("dataset read failure {path}: {reason}")]
    DatasetRead { path: String, reason: String },
    /// Class with zero usable examples after scan (dir has no `.wav` --
    /// distinct from [`Self::EmptyClassAfterExtract`] decode failure).
    /// Belt-and-suspenders; [`Self::BadDataset`] is the canonical path.
    #[error("class {class:?} has no usable .wav examples (per-class kept: {per_class_kept:?})")]
    EmptyClassAfterScan {
        class: String,
        per_class_kept: Vec<(String, usize)>,
    },
    /// One or more classes lost every example to preproc failures
    /// (decode/resample/non-finite spectrogram).
    #[error(
        "class {class:?} lost every example to preproc failures \
         (per-class kept: {per_class_kept:?}; per-class dropped: {per_class_dropped:?})"
    )]
    EmptyClassAfterExtract {
        class: String,
        per_class_kept: Vec<(String, usize)>,
        per_class_dropped: Vec<(String, usize)>,
    },
    /// Aggregate GENUINE-failure drop ratio crossed [`MAX_DROP_RATIO`].
    /// `dropped` = decode/format/resample + backbone-NaN, EXCLUDING benign
    /// silent-clip filtering (~10-15% of speech); `per_class_dropped`
    /// shows TOTAL drops incl. silence.
    #[error(
        "dataset processing-failure ratio {dropped}/{total} = {ratio:.3} exceeds \
         max {max_ratio:.3} (per-class kept: {per_class_kept:?}; \
         per-class dropped: {per_class_dropped:?})"
    )]
    DropRatioExceeded {
        dropped: usize,
        total: usize,
        ratio: f32,
        max_ratio: f32,
        per_class_kept: Vec<(String, usize)>,
        per_class_dropped: Vec<(String, usize)>,
    },
    /// One class crossed [`MAX_PER_CLASS_DROP_RATIO`] while the aggregate
    /// [`MAX_DROP_RATIO`] stayed under cap (clean siblings diluted it) --
    /// fail loud, not skewed-model.
    #[error(
        "class {class:?} lost {dropped}/{total} = {ratio:.3} of its examples to preproc, \
         exceeding per-class cap {max_ratio:.3} \
         (per-class kept: {per_class_kept:?}; per-class dropped: {per_class_dropped:?})"
    )]
    PerClassDropExceeded {
        class: String,
        dropped: usize,
        total: usize,
        ratio: f32,
        max_ratio: f32,
        per_class_kept: Vec<(String, usize)>,
        per_class_dropped: Vec<(String, usize)>,
    },
    /// Stratified split could not give every class a non-empty training
    /// partition (e.g. a singleton class landed entirely in val).
    #[error(
        "stratified split would leave class {class:?} with zero \
         training examples (kept per class: {per_class_kept:?}, \
         val_split={val_split})"
    )]
    StratifiedSplitImpossible {
        class: String,
        per_class_kept: Vec<(String, usize)>,
        val_split: f32,
    },
    /// A `file_mgr` commit op failed; carries the original
    /// [`crate::file_mgr::FileError`] so the api layer keeps its
    /// `FileError::kind()` classifier (path + kind).
    #[error("file_mgr: {0}")]
    File(#[from] crate::file_mgr::FileError),
    /// Non-finite forward/loss scalar, emitted BEFORE `loss.backward()` +
    /// `optim.step()` so the head's `Param` is NOT poisoned; the job
    /// abandons to a cleanly-stale snapshot.  Recovery = retry (different
    /// seed / smaller lr).
    #[error("non-finite {kind} at epoch {epoch}, batch {batch_index}: value={value:?}")]
    NumericFailure {
        /// 1-based, matching `EpochMetrics::epoch`.
        epoch: u32,
        /// 0-based within the epoch.
        batch_index: u32,
        /// Today only `"loss"`; open for future finiteness probes.
        kind: &'static str,
        value: f64,
    },
}

impl crate::common::error::Categorized for FinetuneError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            FinetuneError::InvalidConfig(_) => UserInput,
            FinetuneError::Cancelled => Conflict,
            // Dataset-quality rejections are operator-fixable -> 4xx.
            FinetuneError::BadDataset { .. }
            | FinetuneError::EmptyClassAfterScan { .. }
            | FinetuneError::EmptyClassAfterExtract { .. }
            | FinetuneError::DropRatioExceeded { .. }
            | FinetuneError::PerClassDropExceeded { .. }
            | FinetuneError::StratifiedSplitImpossible { .. } => UserInput,
            // `datasets/` is daemon-owned; mid-job read failure = external
            // tamper / storage fault, not retryable.
            FinetuneError::DatasetRead { .. } => Internal,
            // Delegate so a wrapped NotFound/Permission keeps its class.
            FinetuneError::File(e) => e.kind(),
            FinetuneError::Io { .. }
            | FinetuneError::Model(_)
            | FinetuneError::Backbone(_)
            | FinetuneError::Panic(_)
            | FinetuneError::NumericFailure { .. } => Internal,
        }
    }
}

/// `FinetuneError::Io` shorthand; `path: impl Display` takes
/// `Path::display()` directly.
fn finetune_io_err(path: impl std::fmt::Display, source: std::io::Error) -> FinetuneError {
    FinetuneError::Io {
        path: path.to_string(),
        source,
    }
}

/// Fraction of examples allowed to drop during preproc before the job
/// fails loud (above it the head's metrics no longer describe the
/// submitted dataset).
pub const MAX_DROP_RATIO: f32 = 0.10;

/// Per-class drop ceiling catching ONE class losing a majority while clean
/// siblings keep the aggregate [`MAX_DROP_RATIO`] under threshold.  50% =
/// "this class is statistically unusable".
pub const MAX_PER_CLASS_DROP_RATIO: f32 = 0.50;

/// Run a complete fine-tune job.
///
/// All three closures are `Sync` so rayon workers can poll `cancel`
/// between expensive feature-extraction units; `progress` and `event`
/// are invoked only on the main thread.  `cancel` cannot preempt a
/// single wav/resample/spectrogram op.
pub fn run(
    cfg: &FinetuneConfig,
    progress: &(dyn Fn(&Progress) + Sync),
    event: &(dyn Fn(Event) + Sync),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<FinetuneOutput, FinetuneError> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_inner(cfg, progress, event, cancel)
    }));
    match result {
        Ok(r) => r,
        Err(payload) => Err(FinetuneError::Panic(panic_payload_to_string(payload))),
    }
}

fn run_inner(
    cfg: &FinetuneConfig,
    progress: &(dyn Fn(&Progress) + Sync),
    event: &(dyn Fn(Event) + Sync),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<FinetuneOutput, FinetuneError> {
    cfg.validate()?;
    install_rayon_pool_cap();
    check_cancel(cancel)?;

    let device_inner: burn::tensor::Device<InnerB> = Default::default();
    let device_auto: burn::tensor::Device<AutoB> = Default::default();

    // Seed the backend RNG so `Head::new` weight init draws
    // deterministically from `cfg.seed` (shuffles use it directly).
    // KNOWN-LIMIT: on a `multi-threads` Burn build (production default)
    // `LinearConfig::init` runs on rayon workers whose RNG slots
    // `Backend::seed` (main thread only) doesn't touch, so identical seeds
    // publish bit-different heads with >=2 threads -- harmless for this
    // single-tenant workload (no cross-host bit-reproducibility needed).
    <InnerB as Backend>::seed(&device_inner, cfg.seed);
    <AutoB as Backend>::seed(&device_auto, cfg.seed);

    event(Event::PhaseStarted {
        phase: Stage::DatasetScan,
    });
    // Progress messages never carry server filesystem paths; the typed
    // DatasetScanned event carries the structured payload.
    progress(&Progress::new(Stage::DatasetScan, 0, 0, "scan dataset"));
    let (classes, examples) = scan_dataset(&cfg.data)?;
    let n_classes = classes.len();
    if n_classes < 2 {
        return Err(FinetuneError::InvalidConfig(format!(
            "need at least 2 classes; found {n_classes} under datasets/"
        )));
    }
    if examples.is_empty() {
        return Err(FinetuneError::InvalidConfig(
            "datasets/ contains no .wav examples".into(),
        ));
    }
    // Reject post-scan empty classes before any extraction/optim: a
    // zero-wav class would become a label with no examples, silently
    // degrading the head.
    let pre_scan_counts = per_class_counts_from_examples(&classes, &examples);
    if let Some((idx, _)) = pre_scan_counts.iter().enumerate().find(|(_, n)| **n == 0) {
        return Err(FinetuneError::EmptyClassAfterScan {
            class: classes[idx].clone(),
            per_class_kept: classes
                .iter()
                .cloned()
                .zip(pre_scan_counts.iter().copied())
                .collect(),
        });
    }
    event(Event::DatasetScanned {
        n_classes: n_classes as u32,
        classes: classes
            .iter()
            .zip(pre_scan_counts.iter())
            .map(|(name, n)| ClassCount {
                name: name.clone(),
                n_samples: *n as u64,
            })
            .collect(),
        n_examples_total: examples.len() as u64,
    });
    progress(&Progress::new(
        Stage::DatasetScan,
        examples.len(),
        examples.len(),
        format!("classes: {:?}  total examples: {}", classes, examples.len()),
    ));
    check_cancel(cancel)?;

    // Gate the feature-buffer cap BEFORE loading the backbone so an
    // over-cap dataset errors instead of risking the OOM-killer; re-run
    // defensively inside `extract_features`.
    check_feature_buffer_cap(examples.len())?;

    progress(&Progress::new(Stage::DatasetScan, 0, 1, "load backbone"));
    let (mut extractor, extractor_label) =
        resolve_feature_extractor(&cfg.backbones, cfg.serving_backbone.as_ref())?;
    check_cancel(cancel)?;

    event(Event::PhaseStarted {
        phase: Stage::FeatureExtract,
    });
    // Label is kind + basename (progress messages carry no server paths).
    progress(&Progress::new(
        Stage::FeatureExtract,
        0,
        examples.len(),
        format!("extract features ({extractor_label})..."),
    ));
    let preproc = Preproc::new();
    let total_in = examples.len();
    let t_extract = Instant::now();
    let (feats, labels, drop_counts) =
        extract_features(&mut extractor, &preproc, &examples, progress, cancel)?;
    event(Event::FeatureExtractCompleted {
        kept: feats.len() as u64,
        dropped_nan: drop_counts.dropped_nan,
        dropped_io: drop_counts.dropped_io,
        elapsed_ms: t_extract.elapsed().as_millis() as u64,
    });
    if feats.is_empty() {
        return Err(FinetuneError::InvalidConfig(
            "all examples were dropped during feature extraction".into(),
        ));
    }
    assert_eq!(feats.len(), labels.len());
    check_cancel(cancel)?;

    // Free the extractor (Burn weights, or the RKNN session + NPU context)
    // and the preproc FFT plan; train reads only feats.
    drop(extractor);
    drop(preproc);

    validate_post_extract_quality(
        &classes,
        &pre_scan_counts,
        n_classes,
        &labels,
        total_in,
        drop_counts.dropped_silence as usize,
    )?;

    // Stratified split lands every class in both partitions even when
    // small/imbalanced; deterministic via `cfg.seed`.
    let split =
        stratified_split_indices(&labels, n_classes, cfg.val_split, cfg.seed, Some(&classes))?;
    let train_idx = split.train;
    let val_idx = split.val;
    if train_idx.is_empty() {
        // Defensive: `stratified_split_indices` already errors first.
        return Err(FinetuneError::InvalidConfig(format!(
            "empty training split after val_split={} over {} examples",
            cfg.val_split,
            labels.len(),
        )));
    }
    event(Event::PhaseStarted {
        phase: Stage::Train,
    });
    event(Event::TrainSplit {
        train_n: train_idx.len() as u64,
        val_n: val_idx.len() as u64,
    });
    progress(&Progress::new(
        Stage::Train,
        0,
        cfg.epochs,
        format!(
            "split: train={} val={} (stratified per-class)",
            train_idx.len(),
            val_idx.len()
        ),
    ));

    let head_auto: Head<AutoB> = if let Some(p) = &cfg.init_head {
        // Reject a class-count mismatch early, else the loaded shape
        // computes loss against the wrong target dim.
        let loaded = Head::<AutoB>::load_mpk(p, &device_auto)?;
        let loaded_n = loaded.linear.weight.val().dims()[1];
        if loaded_n != n_classes {
            return Err(FinetuneError::InvalidConfig(format!(
                "init_head has {loaded_n} classes; dataset has {n_classes}"
            )));
        }
        loaded
    } else {
        // `try_new` not `new`: operator-controlled `n_classes` must pass
        // the `0 || > MAX_N_CLASSES` gate `new` skips (else panics inside
        // `LinearConfig::init`).
        Head::<AutoB>::try_new(n_classes, &device_auto)?
    };

    let t_train = Instant::now();
    let train_data = TrainData {
        n_classes,
        feats: &feats,
        labels: &labels,
        train_idx: &train_idx,
        val_idx: &val_idx,
    };
    let train_settings = TrainSettings {
        epochs: cfg.epochs,
        batch: cfg.batch,
        lr: cfg.lr,
        seed: cfg.seed,
    };
    // Stage under `cfg.out.parent()` so the best-epoch tempdir is intra-FS
    // with the publish target (atomic rename) and safe from
    // systemd-tmpfiles eviction.
    let snapshot_parent = cfg.out.parent();
    let train_outcome = train_head(
        head_auto,
        train_data,
        train_settings,
        snapshot_parent,
        progress,
        event,
        cancel,
    )?;
    let head_auto = train_outcome.head;
    event(Event::TrainCompleted {
        epochs_run: train_outcome.epochs_run as u32,
        total_elapsed_ms: t_train.elapsed().as_millis() as u64,
        best_val_epoch: train_outcome.best_val_epoch.map(|e| e as u32),
        best_val_acc: train_outcome.best_val_acc,
        best_val_loss: train_outcome.best_val_loss,
    });
    progress(&Progress::new(
        Stage::Train,
        cfg.epochs,
        cfg.epochs,
        format!("train wall: {:.2?}", t_train.elapsed()),
    ));
    check_cancel(cancel)?;

    event(Event::PhaseStarted { phase: Stage::Save });
    progress(&Progress::new(Stage::Save, 0, 2, "save head"));
    // Final metrics from the in-memory head: `save_mpk_atomic` consumes
    // `head_inner` and the saved bytes are payload-identical, so re-loading
    // would only add I/O.
    let head_inner = head_auto.valid();
    // Only accuracy reaches `FinetuneOutput`; per-epoch val_loss rides the
    // EpochCompleted events.
    let (final_train, _) = evaluate(&head_inner, n_classes, &feats, &labels, &train_idx, cancel)?;
    check_cancel(cancel)?;
    let (final_val, _) = evaluate(&head_inner, n_classes, &feats, &labels, &val_idx, cancel)?;

    if let Some(parent) = cfg.out.parent() {
        std::fs::create_dir_all(parent).map_err(|e| finetune_io_err(parent.display(), e))?;
    }

    // `with_file_name` not `with_extension`: the latter replaces only the
    // final extension, so `v1.2.mpk` -> `v1.2.labels.txt` (drops `.mpk`).
    let labels_path = {
        let stem = cfg
            .out
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "head".to_string());
        cfg.out.with_file_name(format!("{stem}.labels.txt"))
    };

    // Labels FIRST, then head, each crash-consistent via `put_atomic`: a
    // crash between leaves a harmless orphan labels file; the reverse would
    // leave a head whose class indices can't map back to labels.
    let labels_blob = format!("{}\n", classes.join("\n"));
    crate::file_mgr::fs_atomic::put_atomic(&labels_path, labels_blob.as_bytes())?;
    head_inner.save_mpk_atomic(&cfg.out)?;

    let out = FinetuneOutput {
        head_mpk: cfg.out.clone(),
        labels_txt: labels_path,
        final_train_acc: final_train,
        final_val_acc: final_val,
        classes,
    };
    // No `Phase::Done` tick: the wrapper emits the terminal
    // JobCompleted/HeadPublished events; the last `Stage::Save` snapshot
    // pins "what happened last" until then.
    Ok(out)
}

fn install_rayon_pool_cap() {
    RAYON_POOL_INIT.get_or_init(|| {
        // `available_parallelism` honors cgroup/cpuset limits; reserve one
        // core for the daemon's mic/inference/tokio threads.
        let total = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        let n = total.saturating_sub(1).max(1);
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
        {
            // Pool already installed elsewhere; training stays correct, the
            // cap is just lost.
            tracing::warn!(
                target: "training",
                err = %e,
                "rayon global pool already configured; finetune will use the existing pool",
            );
        }
    });
}

fn check_cancel(cancel: &(dyn Fn() -> bool + Sync)) -> Result<(), FinetuneError> {
    if cancel() {
        Err(FinetuneError::Cancelled)
    } else {
        Ok(())
    }
}

/// Test-only `scan_dataset` that unwraps.
#[cfg(test)]
pub(crate) fn scan_dataset_for_test(data_dir: &Path) -> (Vec<String>, Vec<(PathBuf, usize)>) {
    scan_dataset(data_dir).expect("scan_dataset")
}

/// Walk `<workspace>/datasets/` into `(classes, examples)`: non-hidden
/// direct child dirs are class folders, hidden root entries ignored; stray
/// non-dir root entries, unreadable/empty class dirs, and
/// case-insensitive duplicate labels reject as
/// [`FinetuneError::BadDataset`].
///
/// INVARIANT: classes sorted by ASCII byte order (uppercase before
/// lowercase: `Ant/ Zebra/ bear/` -> `[Ant, Zebra, bear]`) for cross-host
/// determinism; reordering re-indexes every previously-trained head.
fn scan_dataset(data_dir: &Path) -> Result<DatasetScan, FinetuneError> {
    use std::collections::BTreeMap;

    let entries = std::fs::read_dir(data_dir).map_err(|source| FinetuneError::BadDataset {
        path: data_dir.display().to_string(),
        reason: format!("read datasets root: {source}"),
    })?;

    // Keyed by lowercase so case-insensitive duplicate detection happens
    // at insert; value is (original-case name, path).
    let mut classes_by_lower: BTreeMap<String, (String, PathBuf)> = BTreeMap::new();
    for entry in entries {
        let entry = entry.map_err(|source| FinetuneError::BadDataset {
            path: data_dir.display().to_string(),
            reason: format!("read entry: {source}"),
        })?;
        let raw_name = entry.file_name();
        let name = raw_name.to_string_lossy();
        // Skip hidden entries (matches the upload leading-dot rule).
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        // `symlink_metadata` doesn't follow, so a symlink-to-dir rejects
        // below.
        let md = std::fs::symlink_metadata(&path).map_err(|source| FinetuneError::BadDataset {
            path: path.display().to_string(),
            reason: format!("stat: {source}"),
        })?;
        if !md.is_dir() {
            return Err(FinetuneError::BadDataset {
                path: path.display().to_string(),
                reason: "non-hidden root entry is not a directory \
                        (only class folders may live under datasets/)"
                    .into(),
            });
        }
        let original = name.into_owned();
        let lower = original.to_ascii_lowercase();
        if let Some((existing_name, existing_path)) =
            classes_by_lower.insert(lower.clone(), (original.clone(), path.clone()))
        {
            return Err(FinetuneError::BadDataset {
                path: data_dir.display().to_string(),
                reason: format!(
                    "duplicate class label under ASCII case-insensitive comparison: \
                     {original:?} (at {}) and {existing_name:?} (at {})",
                    path.display(),
                    existing_path.display(),
                ),
            });
        }
    }

    if classes_by_lower.is_empty() {
        return Err(FinetuneError::BadDataset {
            path: data_dir.display().to_string(),
            reason: "no class folders under datasets/ \
                    (each non-hidden direct subdirectory is a class)"
                .into(),
        });
    }

    // BTreeMap iterates in lowercase order; mixed-case originals differ,
    // so re-sort by original-case below.
    let mut classes: Vec<String> = Vec::with_capacity(classes_by_lower.len());
    let mut by_class_idx: Vec<Vec<PathBuf>> = Vec::with_capacity(classes_by_lower.len());
    for (original, class_dir) in classes_by_lower.values() {
        let mut wavs = Vec::<PathBuf>::new();
        collect_wavs_recursive(class_dir, &mut wavs)?;
        if wavs.is_empty() {
            return Err(FinetuneError::BadDataset {
                path: class_dir.display().to_string(),
                reason: format!("class folder {original:?} has no non-hidden .wav sample files"),
            });
        }
        wavs.sort();
        classes.push(original.clone());
        by_class_idx.push(wavs);
    }

    // Re-sort by original-case byte order, reordering `by_class_idx` in
    // lockstep to keep each class paired with its examples (the INVARIANT).
    let mut zipped: Vec<(String, Vec<PathBuf>)> = classes.into_iter().zip(by_class_idx).collect();
    zipped.sort_by(|a, b| a.0.cmp(&b.0));
    let classes: Vec<String> = zipped.iter().map(|(name, _)| name.clone()).collect();
    let mut examples: Vec<(PathBuf, usize)> = Vec::new();
    for (i, (_name, wavs)) in zipped.into_iter().enumerate() {
        for p in wavs {
            examples.push((p, i));
        }
    }
    Ok((classes, examples))
}

/// Recursively collect non-hidden `.wav` files; hidden + non-`.wav` are
/// skipped, an unreadable subdir rejects as [`FinetuneError::BadDataset`].
fn collect_wavs_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), FinetuneError> {
    let entries = std::fs::read_dir(dir).map_err(|source| FinetuneError::BadDataset {
        path: dir.display().to_string(),
        reason: format!("unreadable: {source}"),
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| FinetuneError::BadDataset {
            path: dir.display().to_string(),
            reason: format!("read entry: {source}"),
        })?;
        let raw_name = entry.file_name();
        let name = raw_name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let md = std::fs::symlink_metadata(&path).map_err(|source| FinetuneError::BadDataset {
            path: path.display().to_string(),
            reason: format!("stat: {source}"),
        })?;
        if md.is_dir() {
            // Recursion depth bounded by `AssetPath::MAX_DEPTH` at upload.
            collect_wavs_recursive(&path, out)?;
        } else if md.is_file() {
            if path.extension().is_some_and(|e| e == "wav") {
                out.push(path);
            }
        } else {
            // Symlinks/devices/FIFOs reject.
            return Err(FinetuneError::BadDataset {
                path: path.display().to_string(),
                reason: "unsupported file type (only regular .wav files are accepted)".into(),
            });
        }
    }
    Ok(())
}

/// Examples per class from the post-scan `examples` vec, indexed parallel
/// to `classes`; feeds the empty-class precondition.
fn per_class_counts_from_examples(classes: &[String], examples: &[(PathBuf, usize)]) -> Vec<usize> {
    let mut counts = vec![0usize; classes.len()];
    for (_, label) in examples {
        if *label < counts.len() {
            counts[*label] += 1;
        }
    }
    counts
}

/// Survivors per class from a flat `labels` vec; feeds post-extract drop
/// totals + empty-class detection.
fn per_class_counts_from_labels(n_classes: usize, labels: &[usize]) -> Vec<usize> {
    let mut counts = vec![0usize; n_classes];
    for &label in labels {
        if label < counts.len() {
            counts[label] += 1;
        }
    }
    counts
}

/// Post-extract quality gates: empty-class detection + per-class and
/// aggregate drop-ratio caps.  `total_in` is the pre-drop example count.
fn validate_post_extract_quality(
    classes: &[String],
    pre_scan_counts: &[usize],
    n_classes: usize,
    labels: &[usize],
    total_in: usize,
    dropped_silence: usize,
) -> Result<(), FinetuneError> {
    let post_extract_counts = per_class_counts_from_labels(n_classes, labels);
    let per_class_kept: Vec<(String, usize)> = classes
        .iter()
        .cloned()
        .zip(post_extract_counts.iter().copied())
        .collect();
    let per_class_dropped: Vec<(String, usize)> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let dropped = pre_scan_counts
                .get(i)
                .copied()
                .unwrap_or(0)
                .saturating_sub(post_extract_counts[i]);
            (c.clone(), dropped)
        })
        .collect();

    // A class with zero survivors: every wav for it failed preproc.
    if let Some((idx, _)) = post_extract_counts
        .iter()
        .enumerate()
        .find(|(_, n)| **n == 0)
    {
        return Err(FinetuneError::EmptyClassAfterExtract {
            class: classes[idx].clone(),
            per_class_kept,
            per_class_dropped,
        });
    }

    // Per-class gate BEFORE the aggregate so the diagnostic names the
    // class.
    for (idx, class_name) in classes.iter().enumerate() {
        let pre = pre_scan_counts.get(idx).copied().unwrap_or(0);
        let post = post_extract_counts[idx];
        let dropped = pre.saturating_sub(post);
        if pre == 0 {
            // Structural empty, caught upstream by EmptyClassAfterScan.
            continue;
        }
        let ratio = dropped as f32 / pre as f32;
        if ratio > MAX_PER_CLASS_DROP_RATIO {
            return Err(FinetuneError::PerClassDropExceeded {
                class: class_name.clone(),
                dropped,
                total: pre,
                ratio,
                max_ratio: MAX_PER_CLASS_DROP_RATIO,
                per_class_kept,
                per_class_dropped,
            });
        }
    }

    // Aggregate gate over GENUINE failures only (decode/format/resample +
    // backbone-NaN); benign all-NaN silent clips (~12% of Speech-Commands)
    // are excluded so a quiet dataset doesn't abort, but backbone-NaN stays
    // in `genuine_dropped` so a sustained backbone failure still trips it.
    let total_dropped = total_in.saturating_sub(labels.len());
    let genuine_dropped = total_dropped.saturating_sub(dropped_silence);
    if total_in > 0 {
        let ratio = genuine_dropped as f32 / total_in as f32;
        if ratio > MAX_DROP_RATIO {
            return Err(FinetuneError::DropRatioExceeded {
                dropped: genuine_dropped,
                total: total_in,
                ratio,
                max_ratio: MAX_DROP_RATIO,
                per_class_kept,
                per_class_dropped,
            });
        }
    }
    Ok(())
}

/// Disjoint train/val index lists into `feats`/`labels` covering every
/// kept example; both deterministically pre-shuffled from the seed.
#[derive(Debug)]
struct SplitIndices {
    train: Vec<usize>,
    val: Vec<usize>,
}

/// Stratified train/val split.  Per class: shuffle the kept indices
/// (per-class seed so classes don't share a permutation), take
/// `n_val = round(class_n * val_split)` clamped to `[v_min, class_n-1]`
/// (v_min = 1 when `val_split > 0 && class_n >= 2`, else 0) so every class
/// lands >=1 in train and (validation on) >=1 in val.  A singleton class
/// with `val_split > 0` can't satisfy both ->
/// [`FinetuneError::StratifiedSplitImpossible`].
fn stratified_split_indices(
    labels: &[usize],
    n_classes: usize,
    val_split: f32,
    seed: u64,
    classes: Option<&[String]>,
) -> Result<SplitIndices, FinetuneError> {
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &label) in labels.iter().enumerate() {
        if label < n_classes {
            by_class[label].push(i);
        }
    }

    // Operator-facing class label; synthetic `class#i` fallback keeps
    // raw-labels-vec unit tests self-contained.
    let label_for = |i: usize| -> String {
        classes
            .and_then(|cs| cs.get(i).cloned())
            .unwrap_or_else(|| format!("class#{i}"))
    };
    let per_class_kept_pairs: Vec<(String, usize)> = by_class
        .iter()
        .enumerate()
        .map(|(i, v)| (label_for(i), v.len()))
        .collect();

    let val_enabled = val_split > 0.0;
    let mut train: Vec<usize> = Vec::with_capacity(labels.len());
    let mut val: Vec<usize> = Vec::with_capacity(labels.len());
    for (class_idx, mut bucket) in by_class.into_iter().enumerate() {
        let class_n = bucket.len();
        // Distinct deterministic stream per class.  `(class_idx + 1)` not
        // `class_idx`: index 0 gives `seed ^ 0 = seed`, colliding with the
        // train/val shuffle seeds below.  Golden-ratio multiplier shared
        // with `shuffle_in_place` (one constant for both sites).
        let class_seed = seed
            ^ ((class_idx as u64)
                .wrapping_add(1)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15));
        shuffle_in_place(&mut bucket, class_seed);

        // Per-class val count, clamped so train keeps >=1 and val gets >=1
        // when validation is on and class_n >= 2.  f64 math for cross-host
        // boundary determinism.
        let raw_val = (class_n as f64 * val_split as f64).round() as usize;
        let v_min = if val_enabled && class_n >= 2 { 1 } else { 0 };
        let v_max = class_n.saturating_sub(1);
        let n_val = raw_val.clamp(v_min, v_max.max(v_min));
        // class_n == 1 + validation can't keep >=1 in both partitions.
        if val_enabled && class_n == 1 {
            return Err(FinetuneError::StratifiedSplitImpossible {
                class: label_for(class_idx),
                per_class_kept: per_class_kept_pairs,
                val_split,
            });
        }
        let (val_part, train_part) = bucket.split_at(n_val);
        val.extend_from_slice(val_part);
        train.extend_from_slice(train_part);
    }

    // Final shuffle so per-class blocks don't leak into ordering; distinct
    // sub-seeds so `train`/`val` permutations don't collide.
    shuffle_in_place(&mut train, seed.wrapping_add(0xC0FFEE));
    shuffle_in_place(&mut val, seed.wrapping_add(0xBADF00D));
    Ok(SplitIndices { train, val })
}

/// Windows per drain of the `pending` accumulator (NOT the preproc file
/// batch [`PREPROC_FILE_BATCH`]).  On the Burn path this is also the windows
/// per forward: 256 sits at the throughput knee (amortizes Burn per-call
/// overhead + the `multi-threads` BLAS path); its forward-activation
/// transient (~74 MiB) is dataset-independent and coexists with the
/// up-to-256 MiB feature buffer.  The batch=1 RKNN path infers per window
/// inside the drain, so here the constant only sets drain cadence.
///
/// Cancel-latency contract: `extract_features` polls `cancel()` at
/// file-batch boundaries, in each worker's per-file fast path, and at the
/// top of every sub-batch forward (the RKNN arm also per window); worst
/// case is ~one `PREPROC_FILE_BATCH` pass (~48 ms, 4-worker SBC) or one
/// Burn sub-batch forward, whichever is larger.
const BACKBONE_BATCH: usize = 256;

/// Files preprocessed in parallel per `map_init` pass -- BOUNDS resident
/// spectrogram memory, decoupled from [`BACKBONE_BATCH`].  Snippet-chopping
/// yields up to [`MAX_WINDOWS_PER_FILE`] (64) windows/file, so a whole
/// 512-file chunk could hold `512*64*~40 KB ~= 1.2 GiB` -- uncapped by
/// [`check_feature_buffer_cap`] (counts FILES) and able to OOM a 2 GB box.
/// 32 caps resident `pending` to `~(BACKBONE_BATCH + PREPROC_FILE_BATCH *
/// MAX_WINDOWS_PER_FILE)` windows (~88 MiB) while the common
/// 1-window-per-file case still fills 256-window forwards.  Trade-off:
/// smaller passes re-run the rubato `SincResampler` init (~6 ms) more often
/// (~+20-30 s on a 33 k resampled dataset), worth it to bound peak RSS.
const PREPROC_FILE_BATCH: usize = 32;

/// Upper bound on 1 s windows a recording is snippet-chopped into.
/// `read_wav_mono` caps duration at 60 s (~60 windows), so 64 covers a 60 s
/// `_background_noise_` recording without truncation; the dataset-wide
/// total is capped by [`MAX_EXAMPLES_FEATURE_BUFFER`].
const MAX_WINDOWS_PER_FILE: usize = 64;

/// Hard ceiling on feature-buffer resident bytes (the dense
/// `Vec<[f32; FEATURE_DIM]>` is `examples.len() * 8 KB`).  256 MiB (~33 k
/// examples) fits the training-job envelope; not operator-tunable -- the
/// failure mode is OOM, not a preference.
const MAX_FEATURE_BYTES: usize = 256 * 1024 * 1024;

/// Derived ceiling on `examples.len()` for [`extract_features`].
const MAX_EXAMPLES_FEATURE_BUFFER: usize =
    MAX_FEATURE_BYTES / std::mem::size_of::<[f32; FEATURE_DIM]>();

/// `Err(InvalidConfig)` if `total` would overflow [`MAX_FEATURE_BYTES`].
fn check_feature_buffer_cap(total: usize) -> Result<(), FinetuneError> {
    if total > MAX_EXAMPLES_FEATURE_BUFFER {
        return Err(FinetuneError::InvalidConfig(format!(
            "dataset has {} examples; on-device cap is {} \
             (~= {} MB feature buffer); use a developer host \
             for larger datasets",
            total,
            MAX_EXAMPLES_FEATURE_BUFFER,
            MAX_FEATURE_BYTES / (1024 * 1024),
        )));
    }
    Ok(())
}

/// Resolved frozen-backbone feature extractor. Burn keeps the raw network so
/// forwards stay batched ([`BACKBONE_BATCH`] amortizes Burn per-call
/// overhead); RKNN reuses the MAE-verified inference wrapper, whose session
/// is compiled for batch=1, and infers per window (~ms each on the NPU).
enum FeatureExtractor {
    Burn {
        /// Boxed: KBs of `Param`/pool state next to the pointer-sized Rknn
        /// variant.
        net: Box<Backbone<InnerB>>,
        device: burn::tensor::Device<InnerB>,
    },
    #[cfg(all(target_os = "linux", feature = "rknpu"))]
    Rknn {
        backbone: Box<crate::inference::RknnBackbone>,
        /// Reused per-window output scratch.
        scratch: Box<[f32; FEATURE_DIM]>,
    },
}

impl std::fmt::Debug for FeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Burn { .. } => f
                .debug_struct("FeatureExtractor::Burn")
                .finish_non_exhaustive(),
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            Self::Rknn { backbone, .. } => f
                .debug_struct("FeatureExtractor::Rknn")
                .field("backbone", backbone)
                .finish_non_exhaustive(),
        }
    }
}

impl FeatureExtractor {
    /// Forward one `sub` of (window, label) pairs, appending kept rows to
    /// `feats`/`labels`. Non-finite outputs count into `dropped_nan` (GENUINE
    /// drops; a sustained fault trips `DropRatioExceeded`): Burn drops the
    /// WHOLE sub-batch (a batched forward can't attribute the fault), RKNN
    /// per window (fp16 can overflow where fp32 would not). The RKNN arm
    /// polls `cancel` per window, not per batch.
    fn forward_sub_batch(
        &mut self,
        sub: &[(Spectrogram, usize)],
        feats: &mut Vec<[f32; FEATURE_DIM]>,
        labels: &mut Vec<usize>,
        dropped_nan: &AtomicUsize,
        cancel: &(dyn Fn() -> bool + Sync),
        chunk_start: usize,
    ) -> Result<(), FinetuneError> {
        check_cancel(cancel)?;
        let n = sub.len();
        match self {
            FeatureExtractor::Burn { net, device } => {
                let mut batched: Vec<f32> = Vec::with_capacity(n * NFrames::USIZE * NBins::USIZE);
                for (spec, _) in sub {
                    batched.extend_from_slice(spec[..].as_flattened());
                }
                let x = Tensor::<InnerB, 4>::from_data(
                    TensorData::new(batched, [n, 1, NFrames::USIZE, NBins::USIZE]),
                    &*device,
                );
                let f = net.forward(x);
                let out_data = f.into_data().to_vec::<f32>().unwrap();
                // Hard assert (release too): a wrong output size would
                // corrupt training silently.
                assert_eq!(
                    out_data.len(),
                    n * FEATURE_DIM,
                    "backbone returned {} floats for batch {n}; expected {}",
                    out_data.len(),
                    n * FEATURE_DIM,
                );
                if out_data.iter().any(|v| !v.is_finite()) {
                    tracing::warn!(
                        target: "training",
                        chunk_start,
                        batch_size = n,
                        "backbone produced non-finite features; dropping the sub-batch \
                         and counting toward dropped_nan (preproc spec was finite -- \
                         suspect backbone .mpk integrity or numerical degeneracy)",
                    );
                    dropped_nan.fetch_add(n, Ordering::Relaxed);
                } else {
                    for ((_, label), feat_chunk) in
                        sub.iter().zip(out_data.chunks_exact(FEATURE_DIM))
                    {
                        let mut arr = [0f32; FEATURE_DIM];
                        arr.copy_from_slice(feat_chunk);
                        feats.push(arr);
                        labels.push(*label);
                    }
                }
            }
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            FeatureExtractor::Rknn { backbone, scratch } => {
                for (spec, label) in sub {
                    check_cancel(cancel)?;
                    // An NPU error is a device/runtime fault, not a data
                    // fault: fail the job typed instead of silently dropping.
                    backbone
                        .infer(spec, scratch)
                        .map_err(|e| FinetuneError::Backbone(e.to_string()))?;
                    if scratch.iter().any(|v| !v.is_finite()) {
                        tracing::warn!(
                            target: "training",
                            chunk_start,
                            "NPU backbone produced non-finite features; dropping the \
                             window and counting toward dropped_nan (fp16 range \
                             overflow or model fault)",
                        );
                        dropped_nan.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    feats.push(**scratch);
                    labels.push(*label);
                }
            }
        }
        Ok(())
    }
}

/// Resolve the extractor from the ordered catalogue (first supported+loadable
/// candidate, exactly like serving), returning it with an operator-facing
/// label (kind + basename, no server paths). A resolved candidate differing
/// from what serving loaded means the head is fit on a basis it won't be
/// classified against -- warn, not fail: a faithful conversion of the same
/// model (the verified `.mpk` <-> `.rknn` pair) is still valid.
fn resolve_feature_extractor(
    catalogue: &BackboneCatalogue,
    serving: Option<&BackboneRef>,
) -> Result<(FeatureExtractor, String), FinetuneError> {
    let (pipeline, idx) = catalogue
        .load_first_supported_indexed()
        .map_err(|e| FinetuneError::Backbone(e.to_string()))?;
    let resolved = &catalogue.candidates[idx];
    tracing::info!(
        target: "training",
        kind = ?resolved.kind,
        path = %resolved.path.display(),
        "training feature extractor resolved",
    );
    if let Some(serving) = serving
        && serving != resolved
    {
        tracing::warn!(
            target: "training",
            training_kind = ?resolved.kind,
            training_path = %resolved.path.display(),
            serving_kind = ?serving.kind,
            serving_path = %serving.path.display(),
            "training feature extractor differs from the serving backbone; \
             unless both artifacts represent the same model (a faithful \
             conversion), the trained head will classify silently wrong",
        );
    }
    let basename = resolved
        .path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<unnamed>".into());
    let label = format!("{:?}: {basename}", resolved.kind);
    let extractor = match pipeline {
        #[cfg(all(target_os = "linux", feature = "rknpu"))]
        BackbonePipeline::Rknn(backbone) => FeatureExtractor::Rknn {
            backbone,
            scratch: Box::new([0.0; FEATURE_DIM]),
        },
        BackbonePipeline::Burn(b) => FeatureExtractor::Burn {
            net: Box::new(b.into_network()),
            device: Default::default(),
        },
    };
    Ok((extractor, label))
}

/// Drop accounting carried alongside `(feats, labels)` so the wrapper emits
/// `Event::FeatureExtractCompleted` without re-scanning.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ExtractDropCounts {
    pub dropped_nan: u64,
    pub dropped_io: u64,
    /// Subset of `dropped_nan` from benign all-NaN (silent-frame)
    /// spectrograms; tracked separately so the aggregate drop-ratio gate
    /// excludes the ~10-15% quiet clips a speech dataset has.  Not on wire.
    pub dropped_silence: u64,
}

/// Return shape of [`extract_features`], aliased for the complexity lint.
type ExtractOutput = (Vec<[f32; FEATURE_DIM]>, Vec<usize>, ExtractDropCounts);

fn extract_features(
    extractor: &mut FeatureExtractor,
    preproc: &Preproc,
    examples: &[(PathBuf, usize)],
    progress_cb: &(dyn Fn(&Progress) + Sync),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<ExtractOutput, FinetuneError> {
    let t0 = Instant::now();
    let dropped_nan = AtomicUsize::new(0);
    // Read/resample/spectrogram failures drop the example, not abort.
    let dropped_io = AtomicUsize::new(0);
    // Subset of `dropped_nan` (benign silent-frame clips); excluded from
    // the aggregate drop-ratio gate.
    let dropped_silence = AtomicUsize::new(0);
    let total = examples.len();

    // Fail fast before the `with_capacity(total)` alloc (`total * 8 KB`)
    // so an over-cap dataset yields InvalidConfig instead of OOM-killing
    // the daemon mid-training.
    check_feature_buffer_cap(total)?;
    let mut feats: Vec<[f32; FEATURE_DIM]> = Vec::with_capacity(total);
    let mut labels: Vec<usize> = Vec::with_capacity(total);
    let mut processed: usize = 0;
    // Set when snippet-chopping fills the feature buffer and later windows
    // go unextracted; surfaced once after the loop.
    let mut buffer_capped = false;
    // Windows awaiting a forward, accumulated ACROSS preproc file-batches
    // and drained in fixed `BACKBONE_BATCH` forwards -- bounds the resident
    // spectrogram set (see `PREPROC_FILE_BATCH`).
    let mut pending: Vec<(Spectrogram, usize)> = Vec::new();

    for file_batch in examples.chunks(PREPROC_FILE_BATCH) {
        check_cancel(cancel)?;

        // Parallel preproc within the file-batch.  `Preproc::clone` shares
        // the FFT plan via `Arc` with fresh scratch; the resampler slot
        // lazy-inits on first non-44.1 kHz file.
        let specs: Vec<Vec<(Spectrogram, usize)>> = file_batch
            .par_iter()
            .map_init(
                || (ResamplerCache::empty(), preproc.clone()),
                |(resampler, worker_preproc), (path, label)| {
                    // Once `cancel()` flips, remaining files drop to empty
                    // rather than paying for I/O; the next `check_cancel`
                    // errors, bounding cancel latency.
                    if cancel() {
                        return Vec::new();
                    }
                    // Read + resample, snippet-chopping a >1 s recording into
                    // windows.  A read/resample failure drops the whole file
                    // (counted in `dropped_io`) without unwinding; a
                    // `Resample(_)` error may leave partial FIR history, so
                    // clear the resampler to re-init fresh next file.
                    let windows = match wav_io::read_wav_mono(path).and_then(|(sr, mono)| {
                        wav_io::to_waveform_windows(sr, mono, resampler, MAX_WINDOWS_PER_FILE)
                    }) {
                        Ok(w) => w,
                        Err(e) => {
                            if matches!(e, wav_io::PreprocError::Resample(_)) {
                                resampler.clear();
                            }
                            dropped_io.fetch_add(1, Ordering::Relaxed);
                            return Vec::new();
                        }
                    };
                    // Each window is an independent example; an all-NaN
                    // (silent-frame) window is benign -> both `dropped_nan`
                    // and `dropped_silence`.
                    let mut kept = Vec::with_capacity(windows.len());
                    for pcm in &windows {
                        let spec = worker_preproc.spectrogram(pcm);
                        if spec[..].as_flattened().iter().any(|v| !v.is_finite()) {
                            dropped_nan.fetch_add(1, Ordering::Relaxed);
                            dropped_silence.fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        kept.push((spec, *label));
                    }
                    kept
                },
            )
            .collect();
        check_cancel(cancel)?;

        pending.extend(specs.into_iter().flatten());
        let prev_processed = processed;
        processed += file_batch.len();
        // On the final file-batch the drain flushes the partial remainder
        // (< BACKBONE_BATCH).
        let is_last = processed == total;

        // Drain in fixed BACKBONE_BATCH forwards so no forward exceeds its
        // sized activation memory.  The dataset-wide feature buffer is
        // capped HERE (`room`): the file-count `check_feature_buffer_cap`
        // can't see snippet-chop window counts.
        while pending.len() >= BACKBONE_BATCH || (is_last && !pending.is_empty()) {
            let room = MAX_EXAMPLES_FEATURE_BUFFER.saturating_sub(feats.len());
            if room == 0 {
                buffer_capped = true;
                break;
            }
            // Windows past the cap are left unextracted (via `buffer_capped`).
            let take = pending.len().min(BACKBONE_BATCH).min(room);
            // First time snippet-chopping exceeds the `with_capacity`
            // reserve, grow straight to the hard cap in ONE realloc so
            // `push` doublings don't transiently hold old+new (~384 MB near
            // the cap); no-op in the common 1-window-per-file case.
            if feats.len() + take > feats.capacity()
                && feats.capacity() < MAX_EXAMPLES_FEATURE_BUFFER
            {
                let want = MAX_EXAMPLES_FEATURE_BUFFER - feats.len();
                feats.reserve_exact(want);
                labels.reserve_exact(want);
            }
            let sub = &pending[..take];
            extractor.forward_sub_batch(
                sub,
                &mut feats,
                &mut labels,
                &dropped_nan,
                cancel,
                processed,
            )?;
            pending.drain(..take);
        }

        // Emit once per ~DEFAULT_PROGRESS_EVERY files via milestone-index
        // comparison; the closing summary follows the loop.
        let prev_milestone = prev_processed / DEFAULT_PROGRESS_EVERY;
        let cur_milestone = processed / DEFAULT_PROGRESS_EVERY;
        if cur_milestone > prev_milestone && processed != total {
            progress_cb(&Progress::new(
                Stage::FeatureExtract,
                processed,
                total,
                format!(
                    "feature extract [{processed:>5}/{total:>5}] kept={} dropped_nan={} dropped_io={} elapsed={:.1?}",
                    feats.len(),
                    dropped_nan.load(Ordering::Relaxed),
                    dropped_io.load(Ordering::Relaxed),
                    t0.elapsed(),
                ),
            ));
        }

        if buffer_capped {
            break;
        }
    }

    let dropped_nan_total = dropped_nan.load(Ordering::Relaxed);
    let dropped_io_total = dropped_io.load(Ordering::Relaxed);
    let dropped_silence_total = dropped_silence.load(Ordering::Relaxed);
    if buffer_capped {
        tracing::warn!(
            target: "training",
            kept = feats.len(),
            cap = MAX_EXAMPLES_FEATURE_BUFFER,
            "feature buffer cap reached while snippet-chopping long recordings; \
             remaining files/windows were not extracted (use shorter recordings, \
             fewer files, or a developer host for larger datasets)",
        );
    }
    // Report `current = total` so a `current/total` consumer sees 100%
    // even when drops mean `feats.len() < total`; counts ride the message
    // + the typed event.
    progress_cb(&Progress::new(
        Stage::FeatureExtract,
        total,
        total,
        format!(
            "feature extraction: {} kept, {dropped_nan_total} dropped (NaN), {dropped_io_total} dropped (IO); total={:.1?}",
            feats.len(),
            t0.elapsed()
        ),
    ));
    // High drop rates leave the `with_capacity(total)` reserve mostly
    // unused for the training phase; shrink only below 50% utilisation (the
    // realloc copy isn't worth it above that).
    if feats.len() * 2 < feats.capacity() {
        feats.shrink_to_fit();
        labels.shrink_to_fit();
    }
    Ok((
        feats,
        labels,
        ExtractDropCounts {
            dropped_nan: dropped_nan_total as u64,
            dropped_io: dropped_io_total as u64,
            dropped_silence: dropped_silence_total as u64,
        },
    ))
}

/// Index of the max element, NaN treated as -Inf (skipped): largest finite
/// value's index, or 0 if all NaN.  Differs from
/// [`crate::inference::kernel::top_k_indices_into`] (NaN-as-Equal), but
/// both sit downstream of the NaN-drop gates so the divergence is
/// unreachable on the hot path.
#[inline]
fn argmax(xs: &[f32]) -> usize {
    debug_assert!(!xs.is_empty(), "argmax called on empty slice");
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    best_i
}

/// Deterministic Fisher-Yates shuffle, reused across partitioning + the
/// per-epoch training shuffle so order is reproducible from the seed.
fn shuffle_in_place<T>(slice: &mut [T], seed: u64) {
    let n = slice.len();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        slice.swap(i, j);
    }
}

/// Best-epoch predicate, accuracy-first then loss (lexicographic): adopt
/// iff no incumbent (`best_val` NaN), strictly-higher accuracy, or equal
/// accuracy with strictly-lower loss; non-finite candidates never win
/// (empty val split -> NaN -> last-epoch fallback).
///
/// Exact `==` on accuracy (not epsilon) because constant-N `correct/N` is
/// bit-identical; an epsilon would fold a real `1/N` gap into a tie and
/// hand it to the loss key, violating accuracy-first.  Strict `<` on the
/// tie keeps the earliest equal-acc-equal-loss epoch.
fn is_new_best(val_acc: f32, val_loss: f64, best_val: f32, best_loss: f64) -> bool {
    if !(val_acc.is_finite() && val_loss.is_finite()) {
        return false;
    }
    best_val.is_nan() || val_acc > best_val || (val_acc == best_val && val_loss < best_loss)
}

fn evaluate(
    head: &Head<InnerB>,
    n_classes: usize,
    feats: &[[f32; FEATURE_DIM]],
    labels: &[usize],
    indices: &[usize],
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<(f32, f64), FinetuneError> {
    if indices.is_empty() {
        // Empty split (`val_split == 0.0`): NaN propagates to
        // `best_val`/`best_loss` (never update) and serialises to null.
        return Ok((f32::NAN, f64::NAN));
    }
    let device: burn::tensor::Device<InnerB> = Default::default();
    // Same mean reduction as the train loop so `val_loss` is comparable
    // across epochs (the tiebreaker key).
    let ce = CrossEntropyLossConfig::new().init(&device);
    let batch = 256;
    let mut correct = 0usize;
    let mut running_loss = 0.0_f64;
    for chunk in indices.chunks(batch) {
        // Per-chunk cancel check keeps bounded cancel latency = one forward;
        // without it a cancel waits for the full pass.
        check_cancel(cancel)?;
        let mut flat = Vec::with_capacity(chunk.len() * FEATURE_DIM);
        for &i in chunk {
            flat.extend_from_slice(&feats[i]);
        }
        let x = Tensor::<InnerB, 2>::from_data(
            TensorData::new(flat, [chunk.len(), FEATURE_DIM]),
            &device,
        );
        let targets: Vec<i32> = chunk.iter().map(|&i| labels[i] as i32).collect();
        let y =
            Tensor::<InnerB, 1, Int>::from_data(TensorData::new(targets, [chunk.len()]), &device);
        #[cfg(test)]
        evaluate_forward_observer::bump();
        let logits = head.forward(x);
        // Weight the mean-reduced scalar by chunk size so the running sum is
        // a true per-example mean even on a short final chunk.
        let chunk_loss = ce.forward(logits.clone(), y).into_scalar() as f64;
        running_loss += chunk_loss * chunk.len() as f64;
        let pred: Vec<f32> = logits.into_data().to_vec::<f32>().unwrap();
        // Hard assert (release too): a wrong head output size would index
        // out of `pred`.
        assert_eq!(pred.len(), chunk.len() * n_classes);
        for (row, &i) in chunk.iter().enumerate() {
            let start = row * n_classes;
            if argmax(&pred[start..start + n_classes]) == labels[i] {
                correct += 1;
            }
        }
    }
    let n = indices.len();
    Ok((correct as f32 / n as f32, running_loss / n as f64))
}

/// Test-only forward-pass counter bumped just before `head.forward(x)` in
/// [`evaluate`], so cancel tests distinguish a top-of-chunk cancel (count
/// 0) from a post-forward one (count >= 1).  Thread-local to avoid
/// cross-test contamination under the parallel runner.
#[cfg(test)]
mod evaluate_forward_observer {
    use std::cell::Cell;

    thread_local! {
        static FORWARDS: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn bump() {
        FORWARDS.with(|c| c.set(c.get() + 1));
    }

    pub(crate) fn reset() {
        FORWARDS.with(|c| c.set(0));
    }

    pub(crate) fn observed() -> usize {
        FORWARDS.with(|c| c.get())
    }
}

struct TrainData<'a> {
    n_classes: usize,
    feats: &'a [[f32; FEATURE_DIM]],
    labels: &'a [usize],
    train_idx: &'a [usize],
    val_idx: &'a [usize],
}

#[derive(Clone, Copy, Debug)]
struct TrainSettings {
    epochs: usize,
    batch: usize,
    lr: f32,
    seed: u64,
}

/// Output of [`train_head`]: the published `head` (best-val-acc epoch if
/// validation ran, last epoch otherwise) plus `Event::TrainCompleted`
/// bookkeeping.
struct TrainOutcome {
    head: Head<AutoB>,
    /// Epochs actually run; < `settings.epochs` only on mid-loop cancel.
    epochs_run: usize,
    /// 1-based best-`val_acc` epoch; `None` when `val_split == 0.0` or
    /// every observed `val_acc` was non-finite.
    best_val_epoch: Option<usize>,
    /// `None` for the same reasons as `best_val_epoch`.
    best_val_acc: Option<f32>,
    /// Val loss of the published best snapshot (the tiebreaker); `None` as
    /// above.
    best_val_loss: Option<f64>,
}

/// `snapshot_parent` stages the best-epoch tempdir: production passes
/// `Some(cfg.out.parent())` for an intra-FS atomic rename onto the
/// daemon-managed `.tmp/` (safe from systemd-tmpfiles); tests pass `None`
/// for the system `$TMPDIR`.
fn train_head(
    mut head: Head<AutoB>,
    data: TrainData<'_>,
    settings: TrainSettings,
    snapshot_parent: Option<&std::path::Path>,
    progress: &(dyn Fn(&Progress) + Sync),
    event: &(dyn Fn(Event) + Sync),
    cancel: &(dyn Fn() -> bool + Sync),
) -> Result<TrainOutcome, FinetuneError> {
    let TrainData {
        n_classes,
        feats,
        labels,
        train_idx,
        val_idx,
    } = data;
    let TrainSettings {
        epochs,
        batch,
        lr,
        seed,
    } = settings;
    let device_auto: burn::tensor::Device<AutoB> = Default::default();
    let ce = CrossEntropyLossConfig::new().init(&device_auto);
    // Classic-momentum SGD, no weight_decay, no LR schedule: momentum
    // recovers most of a larger plain-SGD LR's accuracy without its
    // divergence risk on unnormalized features; no decay because the
    // reference recipe regularizes via dropout.  With `val_split == 0` an
    // empty `val_idx` leaves `best_val` never updating, publishing the
    // last-epoch head with only the `epochs` cap as overfit protection
    // (momentum is not a regularizer); operators SHOULD set
    // `validation_split > 0` but the wrapper doesn't enforce it.
    let mut optim = SgdConfig::new()
        .with_momentum(Some(
            MomentumConfig::new()
                .with_momentum(HEAD_MOMENTUM)
                .with_dampening(0.0)
                .with_nesterov(false),
        ))
        .init();

    // NaN so the metric is well-defined when `val_idx` is empty; the update
    // below ignores non-finite values.
    let mut best_val = f32::NAN;
    // Tiebreaker key seeded `+inf` (min-mode sentinel) so the first
    // accuracy-tie has a finite incumbent; only read on the tie branch.
    let mut best_loss = f64::INFINITY;
    let mut best_val_epoch: Option<usize> = None;
    // Snapshot each strictly-better epoch via `.mpk` round-trip, NOT an
    // in-memory clone: `Param::clone` Arc-shares the Tensor buffer with the
    // running head, so subsequent SGD steps would corrupt the snapshot;
    // only the recorder produces an owning immutable copy (~272 KB).
    let snapshot_dir = match snapshot_parent {
        Some(parent) => {
            // Workspace `.tmp/` is materialized lazily by its first writer.
            std::fs::create_dir_all(parent).map_err(|e| finetune_io_err(parent.display(), e))?;
            tempfile::Builder::new()
                .prefix("train-snapshot-")
                .tempdir_in(parent)
                .map_err(|e| finetune_io_err(parent.display(), e))?
        }
        None => {
            tempfile::tempdir().map_err(|e| finetune_io_err("<finetune snapshot tempdir>", e))?
        }
    };
    let snapshot_path = snapshot_dir.path().join("best.mpk");
    let mut best_path: Option<PathBuf> = None;
    // Per-epoch shuffle scratch, allocated once.
    let mut order: Vec<usize> = Vec::with_capacity(train_idx.len());
    let mut epochs_run = 0usize;
    for epoch in 0..epochs {
        check_cancel(cancel)?;
        let t_epoch = Instant::now();
        order.clear();
        order.extend_from_slice(train_idx);
        shuffle_in_place(&mut order, seed.wrapping_add(epoch as u64));

        let mut running_loss = 0.0_f64;
        let mut running_correct = 0usize;
        let mut running_count = 0usize;

        for (batch_index, chunk) in order.chunks(batch).enumerate() {
            check_cancel(cancel)?;
            let mut flat = Vec::with_capacity(chunk.len() * FEATURE_DIM);
            for &i in chunk {
                flat.extend_from_slice(&feats[i]);
            }
            let x = Tensor::<AutoB, 2>::from_data(
                TensorData::new(flat, [chunk.len(), FEATURE_DIM]),
                &device_auto,
            );
            let targets: Vec<i32> = chunk.iter().map(|&i| labels[i] as i32).collect();
            let y = Tensor::<AutoB, 1, Int>::from_data(
                TensorData::new(targets, [chunk.len()]),
                &device_auto,
            );

            let logits = head.forward(x);
            let loss = ce.forward(logits.clone(), y.clone());

            // Weight the mean-reduced scalar by chunk size for a true
            // per-example epoch average on a short final chunk.
            let loss_scalar = loss.clone().into_scalar() as f64;

            // A non-finite loss yields a non-finite gradient `optim.step`
            // would land on the live head; abort BEFORE the step so the
            // snapshot stays clean (Param NOT poisoned).
            if !loss_scalar.is_finite() {
                return Err(FinetuneError::NumericFailure {
                    epoch: (epoch + 1) as u32,
                    batch_index: batch_index as u32,
                    kind: "loss",
                    value: loss_scalar,
                });
            }

            running_loss += loss_scalar * chunk.len() as f64;
            let pred: Vec<f32> = logits.detach().into_data().to_vec::<f32>().unwrap();
            for (row, &i) in chunk.iter().enumerate() {
                let start = row * n_classes;
                if argmax(&pred[start..start + n_classes]) == labels[i] {
                    running_correct += 1;
                }
            }
            running_count += chunk.len();

            // Cancel check between forward and backward+step: else a cancel
            // during the forward completes one SGD step past the last clean
            // state.  Bounded latency = one forward.
            check_cancel(cancel)?;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &head);
            head = optim.step(lr as f64, head, grads);
        }

        // Zero-guard: `running_count == 0` needs an empty `train_idx`
        // (already rejected at `run_inner`), but defends a regression from
        // surfacing as a NaN `train_acc` that breaks `EpochCompleted`
        // serialisation.
        let train_acc = if running_count > 0 {
            running_correct as f32 / running_count as f32
        } else {
            0.0
        };
        let avg_loss = if running_count > 0 {
            running_loss / running_count as f64
        } else {
            0.0
        };
        // `?` only propagates `Cancelled` (`evaluate` does no other
        // fallible work).  A future transient-fallible step MUST instead
        // log + set `(val_acc, val_loss) = (NaN, NaN)` and continue (like
        // the `save_mpk` arm below) -- never discard the run.
        let (val_acc, val_loss) =
            evaluate(&head.valid(), n_classes, feats, labels, val_idx, cancel)?;
        if is_new_best(val_acc, val_loss, best_val, best_loss) {
            // Snapshot stays raw Burn (the ACSTHEAD schema is enforced only
            // at `cfg.out`).  A transient save failure must NOT abort: the
            // atomic-rename leaves any prior best byte-identical, so log +
            // continue WITHOUT advancing `best_*` (next better epoch
            // retries); if every save fails, `best_path` stays None and the
            // final branch publishes the in-memory head.
            match head.clone().save_mpk(&snapshot_path) {
                Ok(()) => {
                    // Advance all best-* together so a later accuracy-tie
                    // compares against a loss an on-disk snapshot backs.
                    best_val = val_acc;
                    best_loss = val_loss;
                    best_val_epoch = Some(epoch + 1);
                    best_path = Some(snapshot_path.clone());
                }
                Err(e) => {
                    tracing::warn!(
                        target: "training",
                        path = %snapshot_path.display(),
                        err = %e,
                        "best-epoch snapshot save failed; keeping prior best (if any) and continuing training",
                    );
                }
            }
        }
        let metrics = EpochMetrics {
            epoch: epoch + 1,
            epochs,
            train_loss: avg_loss,
            train_acc,
            val_acc,
            best_val_acc: best_val,
            val_loss,
            best_val_loss: best_loss,
        };
        progress(&Progress::with_metrics(
            format!(
                "epoch {:>3}/{}:  train_loss={:.4}  train_acc={:.4}  \
                 val_acc={:.4}  val_loss={:.4}  (best_val={:.4} @ loss {:.4})",
                epoch + 1,
                epochs,
                avg_loss,
                train_acc,
                val_acc,
                val_loss,
                best_val,
                best_loss
            ),
            metrics,
        ));
        event(Event::EpochCompleted {
            epoch: (epoch + 1) as u32,
            epochs: epochs as u32,
            train_loss: avg_loss,
            train_acc,
            val_acc,
            best_val_acc: best_val,
            val_loss,
            best_val_loss: best_loss,
            lr,
            elapsed_ms: t_epoch.elapsed().as_millis() as u64,
        });
        epochs_run = epoch + 1;
    }
    // A reload fault here (EIO/decode/systemd-tmpfiles eviction) would
    // discard every epoch via `?` despite the last-epoch `head` being live;
    // fall back to it but CLEAR the best-epoch metrics so they don't
    // misdescribe the published (last, not best) head.
    let mut best_snapshot_published = best_path.is_some();
    let head_out = match best_path {
        Some(p) => match Head::<AutoB>::load_mpk(&p, &device_auto) {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!(
                    target: "training",
                    path = %p.display(),
                    err = %e,
                    "best-epoch snapshot reload failed at end-of-train; \
                     publishing the last in-memory epoch head instead \
                     (best-epoch metrics cleared)"
                );
                best_snapshot_published = false;
                head
            }
        },
        None => head,
    };
    // All three best-* fields present iff the snapshot was published (else
    // they'd misdescribe the fallback last-epoch head).
    let (best_val_epoch, best_val_acc, best_val_loss) = if best_snapshot_published {
        (
            best_val_epoch,
            best_val_epoch.map(|_| best_val),
            best_val_epoch.map(|_| best_loss),
        )
    } else {
        (None, None, None)
    };
    Ok(TrainOutcome {
        head: head_out,
        epochs_run,
        best_val_epoch,
        best_val_acc,
        best_val_loss,
    })
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    crate::common::error::panic_payload_to_string(payload.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::BackboneKind;
    use parking_lot::Mutex;

    /// Serializes Burn's per-backend RNG slot for re-seeding tests.
    static RNG_LOCK: Mutex<()> = Mutex::new(());

    /// Single-Burn-candidate catalogue for configs whose backbone is never
    /// reached (validation/scan-time rejection fixtures).
    fn burn_only_catalogue(path: &str) -> BackboneCatalogue {
        BackboneCatalogue {
            candidates: vec![BackboneRef {
                kind: BackboneKind::Burn,
                path: PathBuf::from(path),
                hash: None,
            }],
        }
    }

    /// An empty candidate list must reject at `validate`, before any
    /// dataset walk or backbone load.
    #[test]
    fn validate_rejects_empty_backbone_catalogue() {
        let cfg = FinetuneConfig {
            data: PathBuf::from("/nonexistent/data"),
            backbones: BackboneCatalogue::default(),
            serving_backbone: None,
            init_head: None,
            out: PathBuf::from("/nonexistent/out.mpk"),
            epochs: 1,
            batch: 1,
            lr: 0.01,
            val_split: 0.2,
            seed: 1,
        };
        match cfg.validate() {
            Err(FinetuneError::InvalidConfig(msg)) => {
                assert!(
                    msg.contains("backbones"),
                    "diagnostic must name the backbones field; got {msg:?}",
                );
            }
            other => panic!("expected InvalidConfig for empty catalogue, got {other:?}"),
        }
    }

    /// The resolver walks candidates in declaration order and surfaces a
    /// typed `Backbone` error naming EVERY skipped candidate when none loads.
    #[test]
    fn resolve_feature_extractor_reports_every_skipped_candidate() {
        let cat = BackboneCatalogue {
            candidates: vec![
                BackboneRef {
                    kind: BackboneKind::Rknn,
                    path: PathBuf::from("/nonexistent/backbone.rknn"),
                    hash: None,
                },
                BackboneRef {
                    kind: BackboneKind::Burn,
                    path: PathBuf::from("/nonexistent/backbone.mpk"),
                    hash: None,
                },
            ],
        };
        let err = resolve_feature_extractor(&cat, None).expect_err("no candidate can load");
        match err {
            FinetuneError::Backbone(msg) => {
                assert!(
                    msg.contains("backbone.rknn") && msg.contains("backbone.mpk"),
                    "summary must name every skipped candidate; got {msg:?}",
                );
            }
            other => panic!("expected FinetuneError::Backbone, got {other:?}"),
        }
    }

    /// `Backend::seed` makes `Head::new` weight init deterministic.
    /// Ignored: `multi-threads` dispatches init onto rayon workers whose
    /// RNG slot this Mutex can't reach, so parallel runs perturb the seed.
    #[test]
    #[ignore = "requires process isolation; rayon-thread RNG races parallel runs"]
    fn backend_seed_determines_head_init() {
        let _rng_guard = RNG_LOCK.lock();
        let device: burn::tensor::Device<AutoB> = Default::default();
        let n_classes = 4;

        <AutoB as Backend>::seed(&device, 0xDEAD_BEEF);
        let h1 = Head::<AutoB>::new(n_classes, &device);
        let w1: Vec<f32> = h1.linear.weight.val().into_data().to_vec().expect("to_vec");

        <AutoB as Backend>::seed(&device, 0xDEAD_BEEF);
        let h2 = Head::<AutoB>::new(n_classes, &device);
        let w2: Vec<f32> = h2.linear.weight.val().into_data().to_vec().expect("to_vec");

        assert_eq!(w1.len(), w2.len(), "weight length drift");
        for (i, (a, b)) in w1.iter().zip(w2.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "weight drift at idx {i}: a={a}, b={b} -- Backend::seed didn't produce determinism",
            );
        }

        // Different seed -> different weights (determinism above isn't a
        // no-op via some other state source).
        <AutoB as Backend>::seed(&device, 0x1234_5678);
        let h3 = Head::<AutoB>::new(n_classes, &device);
        let w3: Vec<f32> = h3.linear.weight.val().into_data().to_vec().expect("to_vec");
        let any_diff = w1
            .iter()
            .zip(w3.iter())
            .any(|(a, b)| (a - b).abs() > f32::EPSILON);
        assert!(
            any_diff,
            "different seeds must produce different weights; the seed plumbing is a no-op",
        );
    }

    /// `check_feature_buffer_cap` accepts up to
    /// `MAX_EXAMPLES_FEATURE_BUFFER` and rejects strictly more; pins the
    /// cap derivation and the "developer host" diagnostic steer.
    #[test]
    fn feature_buffer_cap_rejects_overflow() {
        // Pin 256 MiB / 8000 B-per-row ([f32; 2000]) = 33 554 rows so a cap
        // bump fails here, not by silently raising the OOM ceiling.
        let expected_cap = (256 * 1024 * 1024) / std::mem::size_of::<[f32; FEATURE_DIM]>();
        assert_eq!(
            MAX_EXAMPLES_FEATURE_BUFFER, expected_cap,
            "MAX_EXAMPLES_FEATURE_BUFFER drifted from the documented \
             256 MiB / FEATURE_DIM*4 derivation",
        );

        check_feature_buffer_cap(MAX_EXAMPLES_FEATURE_BUFFER).expect("cap-equal must pass");
        let err = check_feature_buffer_cap(MAX_EXAMPLES_FEATURE_BUFFER + 1)
            .expect_err("cap+1 must reject");
        match err {
            FinetuneError::InvalidConfig(msg) => {
                assert!(
                    msg.contains("on-device cap"),
                    "diagnostic missing on-device cap context: {msg}",
                );
                assert!(
                    msg.contains("developer host"),
                    "diagnostic missing operator-actionable steer: {msg}",
                );
            }
            other => panic!("expected FinetuneError::InvalidConfig, got {other:?}"),
        }
        assert!(
            matches!(
                check_feature_buffer_cap(MAX_EXAMPLES_FEATURE_BUFFER * 4),
                Err(FinetuneError::InvalidConfig(_)),
            ),
            "4x-cap must reject",
        );
    }

    /// Pins the resident spectrogram set (`pending`) peak,
    /// `(BACKBONE_BATCH-1) + PREPROC_FILE_BATCH*MAX_WINDOWS_PER_FILE`,
    /// below the 256 MiB feature-buffer ceiling, so a constant bump that
    /// re-opens the long-recording OOM cliff fails HERE not in production.
    #[test]
    fn pending_window_set_stays_under_feature_buffer_ceiling() {
        let spec_bytes = std::mem::size_of::<[[f32; NBins::USIZE]; NFrames::USIZE]>();
        assert_eq!(spec_bytes, NFrames::USIZE * NBins::USIZE * 4);

        // A pass must progress yet stay within one forward batch so
        // accumulated passes still fill BACKBONE_BATCH forwards.
        assert!(
            (1..=BACKBONE_BATCH).contains(&PREPROC_FILE_BATCH),
            "PREPROC_FILE_BATCH ({PREPROC_FILE_BATCH}) must be in 1..=BACKBONE_BATCH ({BACKBONE_BATCH})",
        );

        let peak_windows = (BACKBONE_BATCH - 1) + PREPROC_FILE_BATCH * MAX_WINDOWS_PER_FILE;
        let per_window = spec_bytes + std::mem::size_of::<(Spectrogram, usize)>();
        let peak_bytes = peak_windows * per_window;
        assert!(
            peak_bytes < MAX_FEATURE_BYTES,
            "peak in-flight spectrogram set {peak_bytes} B ({peak_windows} windows) must stay \
             below the 256 MiB feature-buffer ceiling {MAX_FEATURE_BYTES} B; a \
             PREPROC_FILE_BATCH / MAX_WINDOWS_PER_FILE bump re-opened the long-recording \
             OOM cliff",
        );

        // The unbounded 512-file-chunk design dwarfs the ceiling, so the
        // bound above isn't trivially satisfied.
        const PRE_REFACTOR_CHUNK_FILES: usize = 512;
        let old_worst_bytes = PRE_REFACTOR_CHUNK_FILES * MAX_WINDOWS_PER_FILE * spec_bytes;
        assert!(
            old_worst_bytes > MAX_FEATURE_BYTES * 4,
            "expected the unbounded design to exceed 4x the ceiling ({old_worst_bytes} B)",
        );
    }

    /// Mono 16-bit broadband NOISE at TARGET_SR (identity resample, so
    /// `secs` s -> `secs` windows); noise keeps every FFT bin non-zero so
    /// the spectrogram stays finite (a sine would NaN -> drop).
    fn write_noise(path: &Path, secs: u32, seed: u64) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: wav_io::TARGET_SR,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).expect("wav create");
        let mut state = seed | 1;
        for _ in 0..(secs * wav_io::TARGET_SR) {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let sample = (state >> 48) as i16;
            w.write_sample(sample / 3).expect("write sample");
        }
        w.finalize().expect("finalize");
    }

    /// End-to-end `run` against the BUNDLED production backbone
    /// (`misc/backbones/backbone.mpk`) through the catalogue resolver: the
    /// unloadable rknn candidate falls back to Burn, features extract, the
    /// head trains, and `head.mpk` + `labels.txt` land at `out`. Guards the
    /// resolver->extractor->train->save seam with the real artifact.
    #[test]
    #[ignore = "depends on repo-root reference assets; --include-ignored"]
    fn run_end_to_end_with_bundled_backbone_via_catalogue_fallback() {
        let _g = RNG_LOCK.lock();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mpk = root.join("misc/backbones/backbone.mpk");
        assert!(mpk.exists(), "missing test asset: {}", mpk.display());

        let dir = tempfile::tempdir().expect("tempdir");
        let data = dir.path().join("datasets");
        for (class, seed_base) in [("alpha", 100u64), ("beta", 200u64)] {
            let class_dir = data.join(class);
            std::fs::create_dir_all(&class_dir).expect("class dir");
            for i in 0..3u64 {
                write_noise(&class_dir.join(format!("s{i}.wav")), 1, seed_base + i);
            }
        }

        let out = dir.path().join("out").join("head.mpk");
        let cfg = FinetuneConfig {
            data,
            backbones: BackboneCatalogue {
                candidates: vec![
                    // Missing on host (and unsupported without rknpu): must be
                    // skipped, not fatal.
                    BackboneRef {
                        kind: BackboneKind::Rknn,
                        path: PathBuf::from("/nonexistent/backbone.rknn"),
                        hash: None,
                    },
                    BackboneRef {
                        kind: BackboneKind::Burn,
                        path: mpk.clone(),
                        hash: None,
                    },
                ],
            },
            serving_backbone: Some(BackboneRef {
                kind: BackboneKind::Burn,
                path: mpk,
                hash: None,
            }),
            init_head: None,
            out: out.clone(),
            epochs: 2,
            batch: 4,
            lr: 0.01,
            val_split: 0.34,
            seed: 42,
        };

        let events: Mutex<Vec<Event>> = Mutex::new(Vec::new());
        let output = run(&cfg, &|_| {}, &|e| events.lock().push(e), &|| false)
            .expect("end-to-end finetune must succeed with the bundled backbone");

        assert_eq!(output.classes, vec!["alpha", "beta"]);
        assert!(output.head_mpk.exists(), "head.mpk must be written");
        assert!(output.labels_txt.exists(), "labels.txt must be written");
        let labels = std::fs::read_to_string(&output.labels_txt).expect("read labels");
        assert_eq!(labels, "alpha\nbeta\n");

        let events = events.lock();
        let extract_completed = events
            .iter()
            .find_map(|e| match e {
                Event::FeatureExtractCompleted {
                    kept, dropped_io, ..
                } => Some((*kept, *dropped_io)),
                _ => None,
            })
            .expect("FeatureExtractCompleted event must be emitted");
        assert_eq!(
            extract_completed,
            (6, 0),
            "6 one-window noise files must all survive extraction",
        );
    }

    /// `extract_features` forwards EVERY finite window across multiple
    /// preproc passes when the total isn't a multiple of `BACKBONE_BATCH`
    /// -- guards the `is_last` trailing flush and the cross-pass `pending`
    /// accumulator.  Random backbone (only kept-row COUNT matters);
    /// broadband noise so no window drops as silence.
    #[test]
    fn extract_features_forwards_every_window_across_passes() {
        let _g = RNG_LOCK.lock();
        let dir = tempfile::tempdir().expect("tempdir");

        // 66 files > PREPROC_FILE_BATCH (3 passes); 70 windows <
        // BACKBONE_BATCH so the dataset flushes via `is_last`.  Two 3 s
        // files exercise snippet chopping.
        let mut examples: Vec<(PathBuf, usize)> = Vec::new();
        for i in 0..33u32 {
            let p = dir.path().join(format!("a{i}.wav"));
            write_noise(&p, 1, i as u64 + 1);
            examples.push((p, 0));
        }
        for i in 0..31u32 {
            let p = dir.path().join(format!("b{i}.wav"));
            write_noise(&p, 1, i as u64 + 1000);
            examples.push((p, 1));
        }
        for i in 0..2u32 {
            let p = dir.path().join(format!("blong{i}.wav"));
            write_noise(&p, 3, i as u64 + 2000);
            examples.push((p, 1));
        }
        assert!(
            examples.len() > PREPROC_FILE_BATCH,
            "fixture must span more than one preproc pass",
        );

        // Reference finite-window count via the same preproc path,
        // independent of the drain/flush plumbing under test.
        let mut reference = 0usize;
        let mut ref_preproc = Preproc::new();
        let mut cache = ResamplerCache::empty();
        for (path, _) in &examples {
            let windows = wav_io::read_wav_mono(path)
                .and_then(|(sr, mono)| {
                    wav_io::to_waveform_windows(sr, mono, &mut cache, MAX_WINDOWS_PER_FILE)
                })
                .expect("reference preproc");
            for pcm in &windows {
                let spec = ref_preproc.spectrogram(pcm);
                if spec[..].as_flattened().iter().all(|v| v.is_finite()) {
                    reference += 1;
                }
            }
        }
        assert!(
            reference >= examples.len(),
            "broadband wavs must yield >= 1 finite window each (got {reference})",
        );

        // Seed the random backbone so features can't go non-finite and drop
        // a sub-batch, making the kept-row count flaky.
        let device: burn::tensor::Device<InnerB> = Default::default();
        <InnerB as Backend>::seed(&device, 7);
        let mut extractor = FeatureExtractor::Burn {
            net: Box::new(Backbone::<InnerB>::new(&device)),
            device,
        };
        let preproc = Preproc::new();

        let no_progress = |_: &Progress| {};
        let never_cancel = || false;
        let (feats, labels, drops) = extract_features(
            &mut extractor,
            &preproc,
            &examples,
            &no_progress,
            &never_cancel,
        )
        .expect("extract_features");

        assert_eq!(feats.len(), labels.len(), "feats/labels length mismatch");
        assert_eq!(
            feats.len(),
            reference,
            "drain/flush must forward every finite window across passes (got {} feats, \
             reference {reference}); a dropped trailing batch or a per-pass-reset `pending` \
             accumulator would short this count",
            feats.len(),
        );
        assert_eq!(drops.dropped_io, 0, "no IO drops expected on clean wavs");
        assert_eq!(
            labels.iter().filter(|&&l| l == 0).count(),
            33,
            "class-0 window count must equal the 33 single-window class-a files",
        );
    }

    /// Linearly separable 2-class features (class 0: dim 0 = 1.0, class 1:
    /// dim 1 = 1.0); SGD reaches val_acc 1.0 within one epoch.
    fn synthetic_separable_data() -> (Vec<[f32; FEATURE_DIM]>, Vec<usize>) {
        let mut feats = Vec::with_capacity(8);
        let mut labels = Vec::with_capacity(8);
        for _ in 0..4 {
            let mut a = [0.0f32; FEATURE_DIM];
            a[0] = 1.0;
            feats.push(a);
            labels.push(0);
            let mut b = [0.0f32; FEATURE_DIM];
            b[1] = 1.0;
            feats.push(b);
            labels.push(1);
        }
        (feats, labels)
    }

    fn weights_of(head: &Head<AutoB>) -> Vec<f32> {
        head.linear
            .weight
            .val()
            .into_data()
            .to_vec()
            .expect("to_vec")
    }

    /// Save a fresh head and return two byte-identical loads.  Two
    /// `Head::new` calls would draw different RNG weights and diverge
    /// post-SGD for reasons unrelated to the best-snapshot tracker; the
    /// `.mpk` round-trip materializes one immutable snapshot to share.
    fn paired_initial_heads(
        n_classes: usize,
        device: &burn::tensor::Device<AutoB>,
    ) -> (tempfile::TempDir, Head<AutoB>, Head<AutoB>) {
        // RNG_LOCK serialises this `Head::new` against the seed-determinism
        // test, whose draw would otherwise race into two different initial
        // heads.
        let _rng_guard = RNG_LOCK.lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("init.mpk");
        Head::<AutoB>::new(n_classes, device)
            .save_mpk(&path)
            .expect("save init head");
        let a = Head::<AutoB>::load_mpk(&path, device).expect("load a");
        let b = Head::<AutoB>::load_mpk(&path, device).expect("load b");
        (dir, a, b)
    }

    /// Exhaustively pins `is_new_best`'s accuracy-first, loss-second
    /// contract.
    #[test]
    fn is_new_best_accuracy_first_loss_second() {
        // No incumbent (best_val NaN): any finite candidate wins.
        assert!(is_new_best(0.5, 9.9, f32::NAN, f64::INFINITY));

        // Higher accuracy wins despite worse loss; lower accuracy loses
        // despite better loss.
        assert!(is_new_best(0.90, 5.0, 0.85, 0.01));
        assert!(!is_new_best(0.85, 0.001, 0.90, 9.0));

        // Tiebreaker: equal accuracy, strictly-lower loss wins; equal on
        // both is not an improvement (strict `<`).
        assert!(is_new_best(0.90, 0.20, 0.90, 0.30));
        assert!(!is_new_best(0.90, 0.40, 0.90, 0.30));
        assert!(!is_new_best(0.90, 0.30, 0.90, 0.30));

        // Non-finite candidates never win.
        assert!(!is_new_best(f32::NAN, 0.1, 0.5, 0.5));
        assert!(!is_new_best(0.99, f64::NAN, 0.5, 0.5));
        assert!(!is_new_best(f32::INFINITY, 0.1, 0.5, 0.5));
        assert!(!is_new_best(0.99, f64::INFINITY, 0.5, 0.5));

        // Equal correct/N is bit-identical, so the tie (loss) branch fires,
        // not `>`.
        let acc_a = 19.0f32 / 20.0;
        let acc_b = 19.0f32 / 20.0;
        assert_eq!(
            acc_a.to_bits(),
            acc_b.to_bits(),
            "equal correct/N is bit-identical"
        );
        assert!(
            is_new_best(acc_b, 0.10, acc_a, 0.20),
            "tie on acc -> loss decides"
        );
    }

    /// On an accuracy plateau (val_acc 1.0 every epoch, val_loss strictly
    /// decreasing) the loss tiebreaker publishes the last (lowest-loss)
    /// epoch, not epoch 1 that accuracy-only selection kept; pins both the
    /// epoch-3 pick and that the head DIFFERS from a 1-epoch head.
    #[test]
    fn train_head_publishes_lowest_loss_epoch_on_accuracy_plateau() {
        let device: burn::tensor::Device<AutoB> = Default::default();
        let (feats, labels) = synthetic_separable_data();
        let train_idx: Vec<usize> = (0..feats.len()).collect();
        let val_idx: Vec<usize> = train_idx.clone();
        let n_classes = 2;
        // Byte-identical inits: 3-epoch run + 1-epoch stand-in for the
        // earliest-plateau epoch accuracy-only would publish.
        let (_init_dir, init_3, init_1) = paired_initial_heads(n_classes, &device);

        let captured: Mutex<Vec<EpochMetrics>> = Mutex::new(Vec::new());
        let progress = |p: &Progress| {
            if let Some(m) = p.metrics.as_ref() {
                captured.lock().push(*m);
            }
        };
        let cancel = || false;
        let data_3 = TrainData {
            n_classes,
            feats: &feats,
            labels: &labels,
            train_idx: &train_idx,
            val_idx: &val_idx,
        };
        let settings_3 = TrainSettings {
            epochs: 3,
            batch: 4,
            lr: 0.5,
            seed: 7,
        };
        let head_3 = train_head(
            init_3,
            data_3,
            settings_3,
            None,
            &progress,
            &|_| {},
            &cancel,
        )
        .expect("3-epoch train")
        .head;

        let metrics = captured.into_inner();
        assert_eq!(metrics.len(), 3, "one EpochMetrics per epoch");
        // Preconditions: accuracy plateaus at 1.0 while val_loss strictly
        // decreases, giving the tiebreaker a clear last-epoch winner.
        assert!(
            metrics.iter().all(|m| m.val_acc == 1.0),
            "val_acc must plateau at 1.0 for this scenario; got {metrics:?}",
        );
        assert!(
            metrics[0].val_loss > metrics[1].val_loss && metrics[1].val_loss > metrics[2].val_loss,
            "val_loss must strictly decrease so the tiebreaker selects the last epoch; got {metrics:?}",
        );

        let mut best_epoch = 0;
        let mut best_val = f32::NAN;
        let mut best_loss = f64::INFINITY;
        for (i, m) in metrics.iter().enumerate() {
            if is_new_best(m.val_acc, m.val_loss, best_val, best_loss) {
                best_val = m.val_acc;
                best_loss = m.val_loss;
                best_epoch = i + 1;
            }
        }
        assert_eq!(
            best_epoch, 3,
            "loss tiebreaker must select the lowest-loss (last) epoch on a decreasing-loss plateau",
        );

        // Published head must be the epoch-3 snapshot, DIFFERING from the
        // epoch-1 head it would equal under accuracy-only selection.
        let data_1 = TrainData {
            n_classes,
            feats: &feats,
            labels: &labels,
            train_idx: &train_idx,
            val_idx: &val_idx,
        };
        let settings_1 = TrainSettings {
            epochs: 1,
            batch: 4,
            lr: 0.5,
            seed: 7,
        };
        let head_1 = train_head(init_1, data_1, settings_1, None, &|_| {}, &|_| {}, &cancel)
            .expect("1-epoch train")
            .head;
        let w3 = weights_of(&head_3);
        let w1 = weights_of(&head_1);
        assert_eq!(w3.len(), w1.len(), "weight buffer length drift");
        assert!(
            w3.iter()
                .zip(w1.iter())
                .any(|(a, b)| (a - b).abs() >= f32::EPSILON),
            "published 3-epoch head must DIFFER from the epoch-1 head: the loss tiebreaker selects \
             the later, lower-loss epoch, not the earliest-plateau epoch accuracy-only would keep",
        );
    }

    /// Empty val set => `best_val` never updates => fallback to the
    /// last-epoch head; pinned by a 2-epoch run differing from a 1-epoch one
    /// (epoch-2 steps took effect).
    #[test]
    fn train_head_falls_back_to_last_when_no_val() {
        let device: burn::tensor::Device<AutoB> = Default::default();
        let (feats, labels) = synthetic_separable_data();
        let train_idx: Vec<usize> = (0..feats.len()).collect();
        let val_idx: Vec<usize> = Vec::new();
        let n_classes = 2;
        let progress = |_: &Progress| {};
        let cancel = || false;
        let (_init_dir, init_2, init_1) = paired_initial_heads(n_classes, &device);

        let data2 = TrainData {
            n_classes,
            feats: &feats,
            labels: &labels,
            train_idx: &train_idx,
            val_idx: &val_idx,
        };
        let head_2 = train_head(
            init_2,
            data2,
            TrainSettings {
                epochs: 2,
                batch: 4,
                lr: 0.5,
                seed: 7,
            },
            None,
            &progress,
            &|_| {},
            &cancel,
        )
        .expect("2-epoch train (no val)")
        .head;

        let data1 = TrainData {
            n_classes,
            feats: &feats,
            labels: &labels,
            train_idx: &train_idx,
            val_idx: &val_idx,
        };
        let head_1 = train_head(
            init_1,
            data1,
            TrainSettings {
                epochs: 1,
                batch: 4,
                lr: 0.5,
                seed: 7,
            },
            None,
            &progress,
            &|_| {},
            &cancel,
        )
        .expect("1-epoch train (no val)")
        .head;

        let w2 = weights_of(&head_2);
        let w1 = weights_of(&head_1);
        let differs = w2
            .iter()
            .zip(w1.iter())
            .any(|(a, b)| (a - b).abs() > f32::EPSILON);
        assert!(
            differs,
            "no-val path must publish the last-epoch head; weights are identical to a 1-epoch run, \
             which means epoch-2 SGD steps were silently discarded",
        );
    }

    /// `train_head` aborts on a non-finite per-batch loss with the right
    /// diagnostic payload; trigger is an `f32::INFINITY` feature row.
    #[test]
    fn train_head_aborts_on_non_finite_loss() {
        let device: burn::tensor::Device<AutoB> = Default::default();
        let n_classes = 2;
        let (_init_dir, head, _spare) = paired_initial_heads(n_classes, &device);

        // One INFINITY feature: logits -> Inf -> CE -> NaN.
        let mut row_inf = [1.0_f32; FEATURE_DIM];
        row_inf[0] = f32::INFINITY;
        let row_zero = [0.0_f32; FEATURE_DIM];
        let feats: Vec<[f32; FEATURE_DIM]> = vec![row_inf, row_zero];
        let labels = vec![0_usize, 1];
        let train_idx: Vec<usize> = (0..feats.len()).collect();
        let val_idx: Vec<usize> = train_idx.clone();

        let data = TrainData {
            n_classes,
            feats: &feats,
            labels: &labels,
            train_idx: &train_idx,
            val_idx: &val_idx,
        };
        let settings = TrainSettings {
            epochs: 1,
            batch: 2,
            lr: 0.5,
            seed: 7,
        };
        let err = match train_head(head, data, settings, None, &|_| {}, &|_| {}, &|| false) {
            Ok(_) => panic!("non-finite loss must abort training"),
            Err(e) => e,
        };
        match err {
            FinetuneError::NumericFailure {
                epoch,
                batch_index,
                kind,
                value,
            } => {
                assert_eq!(epoch, 1, "non-finite landed in epoch 1");
                assert_eq!(batch_index, 0, "non-finite landed in the first batch");
                assert_eq!(kind, "loss");
                assert!(
                    !value.is_finite(),
                    "captured value must itself be non-finite; got {value}",
                );
            }
            other => panic!("expected FinetuneError::NumericFailure, got {other:?}"),
        }
    }

    /// `Head::save_mpk` materializes synchronously: on-disk bytes are an
    /// immutable snapshot of pre-call weights regardless of a later SGD
    /// step on the source head.  Defends the best-epoch tracker's "save
    /// then mutate source" interleaving that a deferred-write recorder
    /// upgrade would silently break.
    #[test]
    fn save_mpk_snapshot_is_independent_of_subsequent_sgd_step() {
        let device: burn::tensor::Device<AutoB> = Default::default();
        let (feats, labels) = synthetic_separable_data();
        let n_classes = 2;
        let (_init_dir, mut head, _spare) = paired_initial_heads(n_classes, &device);

        // Ground-truth pre-step weights the recorder must serialize.
        let pre_step_weights = weights_of(&head);

        // Save through the same path `train_head` uses; `head` is owned and
        // about to be stepped.
        let dir = tempfile::tempdir().expect("snapshot tempdir");
        let snapshot_path = dir.path().join("snapshot.mpk");
        head.clone()
            .save_mpk(&snapshot_path)
            .expect("save_mpk synchronous");

        // Eager-materialization gate: non-empty bytes by the time
        // `save_mpk` returned (a deferred write would leave 0 bytes).
        let bytes_before_step = std::fs::read(&snapshot_path).expect("snapshot readable post-save");
        assert!(
            !bytes_before_step.is_empty(),
            "snapshot is empty after save_mpk returned -- recorder may be deferring \
             materialization, which would break the best-epoch tracker",
        );

        // One SGD step mirroring `train_head`.
        let ce = CrossEntropyLossConfig::new().init(&device);
        let mut optim = SgdConfig::new().init();
        let chunk_n = feats.len().min(8);
        let mut flat = Vec::with_capacity(chunk_n * FEATURE_DIM);
        for row in feats.iter().take(chunk_n) {
            flat.extend_from_slice(row);
        }
        let x =
            Tensor::<AutoB, 2>::from_data(TensorData::new(flat, [chunk_n, FEATURE_DIM]), &device);
        let targets: Vec<i32> = labels.iter().take(chunk_n).map(|&i| i as i32).collect();
        let y = Tensor::<AutoB, 1, Int>::from_data(TensorData::new(targets, [chunk_n]), &device);
        let logits = head.forward(x);
        let loss = ce.forward(logits, y);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &head);
        head = optim.step(0.5_f64, head, grads);

        // Post-step head must move, else the test is vacuous.
        let post_step_weights = weights_of(&head);
        let moved = pre_step_weights
            .iter()
            .zip(post_step_weights.iter())
            .any(|(a, b)| (a - b).abs() > f32::EPSILON);
        assert!(
            moved,
            "SGD step did not change weights; test cannot prove the snapshot is independent",
        );

        let bytes_after_step = std::fs::read(&snapshot_path).expect("snapshot readable post-step");
        assert_eq!(
            bytes_before_step, bytes_after_step,
            "snapshot file bytes changed after the SGD step -- snapshot is not immutable",
        );

        // Loaded weights must match pre-step, not post-step: the invariant
        // behind `train_head`'s end-of-train reload.
        let loaded =
            Head::<AutoB>::load_mpk(&snapshot_path, &device).expect("load_mpk round-trips");
        let loaded_weights = weights_of(&loaded);
        assert_eq!(
            pre_step_weights.len(),
            loaded_weights.len(),
            "weight buffer length drift across save/load",
        );
        for (i, (a, b)) in pre_step_weights
            .iter()
            .zip(loaded_weights.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "loaded weight at idx {i} differs from pre-step snapshot \
                 (loaded={b}, pre-step={a}) -- snapshot captured post-step state",
            );
        }
    }

    /// An empty class directory fails closed at scan time with
    /// `BadDataset`, before any extraction/optim step.
    #[test]
    fn run_rejects_empty_class_dir_before_training() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Scan rejects `no` (alphabetically first) before `yes`.
        let yes = dir.path().join("yes");
        let no = dir.path().join("no");
        std::fs::create_dir(&yes).unwrap();
        std::fs::create_dir(&no).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(yes.join("a.wav"), b"placeholder").unwrap();

        let cfg = FinetuneConfig {
            data: dir.path().to_path_buf(),
            backbones: burn_only_catalogue("/nonexistent/backbone.mpk"),
            serving_backbone: None,
            init_head: None,
            out: dir.path().join("out.mpk"),
            epochs: 1,
            batch: 1,
            lr: 0.01,
            val_split: 0.2,
            seed: 1,
        };
        let err = run(&cfg, &|_| {}, &|_| {}, &|| false).expect_err("empty class must reject");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert!(
                    path.ends_with("no"),
                    "BadDataset path must point at the offending class folder; got {path:?}"
                );
                assert!(
                    reason.contains("no non-hidden .wav sample files"),
                    "reason must explain the empty-class rejection; got {reason:?}"
                );
            }
            other => panic!("expected FinetuneError::BadDataset, got {other:?}"),
        }
    }

    /// `stratified_split_indices` lands every `class_n >= 2` class in BOTH
    /// partitions when `val_split > 0`.  Imbalanced 3-class fixture (8/3/2)
    /// where a naive `round(n * 0.25)` would give the 2-example class zero
    /// val rows that the per-class clamp pulls to 1.
    #[test]
    fn stratified_split_represents_every_class_in_both_partitions() {
        // classes 0/1/2 = 8/3/2 examples (imbalanced).
        let labels: Vec<usize> = (0..8)
            .map(|_| 0)
            .chain((0..3).map(|_| 1))
            .chain((0..2).map(|_| 2))
            .collect();
        let n_classes = 3;
        let val_split = 0.25_f32;
        let split = stratified_split_indices(&labels, n_classes, val_split, 12345, None)
            .expect("stratified split must succeed for class_n >= 2");

        for class in 0..n_classes {
            let train_n = split.train.iter().filter(|&&i| labels[i] == class).count();
            let val_n = split.val.iter().filter(|&&i| labels[i] == class).count();
            assert!(
                train_n >= 1,
                "class {class} missing from train partition (train={train_n}, val={val_n}); \
                 stratification regressed",
            );
            assert!(
                val_n >= 1,
                "class {class} missing from val partition (train={train_n}, val={val_n}); \
                 stratification regressed",
            );
        }

        let split2 = stratified_split_indices(&labels, n_classes, val_split, 12345, None).unwrap();
        assert_eq!(
            split.train, split2.train,
            "train indices must be deterministic for a given seed"
        );
        assert_eq!(
            split.val, split2.val,
            "val indices must be deterministic for a given seed"
        );

        // Distinct seeds shuffle differently (sanity; determinism above is
        // the load-bearing check) -- guards a no-op RNG.
        let split_other =
            stratified_split_indices(&labels, n_classes, val_split, 67890, None).unwrap();
        let train_set: std::collections::HashSet<_> = split.train.iter().collect();
        let train_other: std::collections::HashSet<_> = split_other.train.iter().collect();
        assert!(
            train_set != train_other,
            "two distinct seeds yielded byte-identical train partitions; \
             RNG plumbing may be a no-op",
        );
    }

    /// A singleton class with validation enabled can't satisfy >=1 in
    /// both partitions; the error names the class.
    #[test]
    fn stratified_split_rejects_singleton_class_when_val_enabled() {
        let labels: Vec<usize> = (0..5).map(|_| 0).chain(std::iter::once(1)).collect();
        let err = stratified_split_indices(&labels, 2, 0.2, 7, None)
            .expect_err("singleton class + val must reject");
        match err {
            FinetuneError::StratifiedSplitImpossible {
                class,
                per_class_kept,
                val_split,
            } => {
                assert_eq!(class, "class#1", "diagnostic must name the singleton class");
                assert_eq!(
                    per_class_kept,
                    vec![("class#0".into(), 5), ("class#1".into(), 1)]
                );
                assert!((val_split - 0.2).abs() < f32::EPSILON);
            }
            other => panic!("expected StratifiedSplitImpossible, got {other:?}"),
        }

        // val_split == 0.0 accepts the same shape (no val minimum).
        let ok = stratified_split_indices(&labels, 2, 0.0, 7, None)
            .expect("val_split=0 must accept singleton classes");
        assert_eq!(ok.train.len(), 6);
        assert_eq!(ok.val.len(), 0);
    }

    /// Defense-in-depth: `FinetuneConfig::validate` rejects `lr` above
    /// `MAX_LEARNING_RATE` even for callers bypassing the api validator.
    #[test]
    fn validate_rejects_lr_above_max_learning_rate() {
        let mut cfg = FinetuneConfig {
            data: PathBuf::from("/nonexistent/data"),
            backbones: burn_only_catalogue("/nonexistent/backbone.mpk"),
            serving_backbone: None,
            init_head: None,
            out: PathBuf::from("/nonexistent/out.mpk"),
            epochs: 1,
            batch: 1,
            lr: crate::file_mgr::request_payload::MAX_LEARNING_RATE,
            val_split: 0.2,
            seed: 1,
        };
        cfg.validate().expect("lr at cap must validate");
        cfg.lr = crate::file_mgr::request_payload::MAX_LEARNING_RATE * 1.1;
        match cfg.validate() {
            Err(FinetuneError::InvalidConfig(msg)) => {
                assert!(
                    msg.contains("lr"),
                    "InvalidConfig message must mention lr; got {msg:?}"
                );
            }
            other => panic!("expected InvalidConfig for lr > MAX_LEARNING_RATE, got {other:?}"),
        }
        // Pathological: would diverge to ±Inf before NumericFailure.
        cfg.lr = 1e30;
        cfg.validate()
            .expect_err("lr = 1e30 must be rejected by the algorithm-side validator");
    }

    /// `evaluate` propagates `Cancelled` from the top-of-chunk cancel
    /// check, keeping "bounded cancel latency = one forward".  600 indices
    /// / batch 256 = 3 chunks; exactly 1 cancel call proves the check is at
    /// the chunk top, before the forward.
    #[test]
    fn evaluate_cancel_propagates_on_non_empty_indices() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let device: burn::tensor::Device<InnerB> = Default::default();
        let head = Head::<InnerB>::new(2, &device);
        let feats: Vec<[f32; FEATURE_DIM]> = vec![[0.0; FEATURE_DIM]; 600];
        let labels: Vec<usize> = (0..600).map(|i| i % 2).collect();
        let indices: Vec<usize> = (0..600).collect();

        // Reset so a prior test's bumps don't leak.
        evaluate_forward_observer::reset();
        let calls = AtomicUsize::new(0);
        let cancel = || {
            calls.fetch_add(1, Ordering::Relaxed);
            true
        };
        let err =
            evaluate(&head, 2, &feats, &labels, &indices, &cancel).expect_err("must propagate");
        assert!(
            matches!(err, FinetuneError::Cancelled),
            "expected Cancelled, got {err:?}",
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "cancel must be polled exactly once -- at the top of the first chunk, \
             before any forward pass.  More than 1 call means the check moved \
             past the per-chunk top and would no longer bound cancel latency.",
        );
        // No forward may run between the cancel poll and the Err return (a
        // post-forward check would still satisfy `calls == 1`).
        assert_eq!(
            evaluate_forward_observer::observed(),
            0,
            "cancel must short-circuit BEFORE `head.forward(x)` runs; \
             a non-zero observer count means the cancel check was \
             moved past the forward, blowing the documented bounded \
             cancel latency contract.",
        );
    }

    /// Happy path: `evaluate` returns finite accuracy in `[0.0, 1.0]` for
    /// non-empty indices with a never-cancel closure; catches a flipped
    /// cancel sense the Err-propagation test wouldn't.
    #[test]
    fn evaluate_returns_finite_accuracy_when_not_cancelled() {
        let device: burn::tensor::Device<InnerB> = Default::default();
        let head = Head::<InnerB>::new(2, &device);
        let feats: Vec<[f32; FEATURE_DIM]> = vec![[0.0; FEATURE_DIM]; 600];
        let labels: Vec<usize> = (0..600).map(|i| i % 2).collect();
        let indices: Vec<usize> = (0..600).collect();

        evaluate_forward_observer::reset();
        let (acc, loss) = evaluate(&head, 2, &feats, &labels, &indices, &|| false)
            .expect("non-cancelled evaluate must succeed");
        assert!(
            acc.is_finite(),
            "non-empty-indices accuracy must be finite; got {acc}",
        );
        assert!(
            (0.0..=1.0).contains(&acc),
            "accuracy must be in [0.0, 1.0]; got {acc}",
        );
        assert!(
            loss.is_finite() && loss >= 0.0,
            "non-empty-indices cross-entropy loss must be finite and non-negative; got {loss}",
        );
        assert_eq!(
            evaluate_forward_observer::observed(),
            3,
            "expected one forward pass per chunk (600 / 256 = 3 chunks)",
        );
    }

    /// Empty-indices fast path returns `Ok(NaN)` regardless of cancel
    /// (the `is_empty` guard short-circuits before any poll); callers
    /// rely on the NaN sentinel for "no val partition".
    #[test]
    fn evaluate_empty_indices_fast_path_ignores_cancel() {
        let device: burn::tensor::Device<InnerB> = Default::default();
        let head = Head::<InnerB>::new(2, &device);
        let feats: Vec<[f32; FEATURE_DIM]> = vec![[0.0; FEATURE_DIM]; 3];
        let labels = vec![0usize, 1, 0];
        let empty: Vec<usize> = Vec::new();
        let (nan_acc, nan_loss) = evaluate(&head, 2, &feats, &labels, &empty, &|| true)
            .expect("empty-indices fast-path must NOT consult cancel");
        assert!(
            nan_acc.is_nan() && nan_loss.is_nan(),
            "empty-indices result must be (NaN, NaN); got ({nan_acc}, {nan_loss})",
        );
    }

    /// `validate_post_extract_quality` rejects when a class loses every
    /// example to preproc, naming the offender + kept/dropped counts.
    #[test]
    fn validate_post_extract_rejects_class_with_zero_survivors() {
        // class 1 ("no") loses everything.
        let classes = vec!["yes".to_string(), "no".to_string()];
        let pre_scan = vec![3usize, 3];
        let labels = vec![0usize, 0, 0];
        let err = validate_post_extract_quality(&classes, &pre_scan, 2, &labels, 6, 0)
            .expect_err("class with zero survivors must reject");
        match err {
            FinetuneError::EmptyClassAfterExtract {
                class,
                per_class_kept,
                per_class_dropped,
            } => {
                assert_eq!(class, "no", "diagnostic must name the offending class");
                assert_eq!(
                    per_class_kept,
                    vec![("yes".into(), 3), ("no".into(), 0)],
                    "per-class kept counts must surface in the error",
                );
                assert_eq!(
                    per_class_dropped,
                    vec![("yes".into(), 0), ("no".into(), 3)],
                    "per-class dropped counts must surface in the error",
                );
            }
            other => panic!("expected EmptyClassAfterExtract, got {other:?}"),
        }
    }

    /// `validate_post_extract_quality` rejects when the aggregate drop
    /// ratio crosses [`MAX_DROP_RATIO`] (100 in, 80 out = 20%).
    #[test]
    fn validate_post_extract_rejects_high_drop_ratio() {
        // Both classes clear the empty-class gate, so the drop-ratio gate is
        // the load-bearing rejection.
        let classes = vec!["a".to_string(), "b".to_string()];
        let pre_scan = vec![50usize, 50];
        let labels: Vec<usize> = (0..40).map(|_| 0).chain((0..40).map(|_| 1)).collect();
        let err = validate_post_extract_quality(&classes, &pre_scan, 2, &labels, 100, 0)
            .expect_err("20% drop ratio must reject");
        match err {
            FinetuneError::DropRatioExceeded {
                dropped,
                total,
                ratio,
                max_ratio,
                per_class_kept,
                per_class_dropped,
            } => {
                assert_eq!(dropped, 20);
                assert_eq!(total, 100);
                assert!(
                    (ratio - 0.20).abs() < 1e-4,
                    "ratio should be ~0.20: {ratio}"
                );
                assert!((max_ratio - MAX_DROP_RATIO).abs() < f32::EPSILON);
                assert_eq!(per_class_kept, vec![("a".into(), 40), ("b".into(), 40)]);
                assert_eq!(per_class_dropped, vec![("a".into(), 10), ("b".into(), 10)]);
            }
            other => panic!("expected DropRatioExceeded, got {other:?}"),
        }
    }

    /// Benign silence drops don't trip the aggregate gate (20% silence,
    /// 0 genuine must PASS); the genuine path stays gated and reports the
    /// genuine count, not the total.
    #[test]
    fn validate_post_extract_tolerates_silence_drops() {
        let classes = vec!["a".to_string(), "b".to_string()];
        let pre_scan = vec![50usize, 50];
        // 100 in, 80 kept -> 20% total drop.
        let labels: Vec<usize> = (0..40).map(|_| 0).chain((0..40).map(|_| 1)).collect();

        // All 20 drops silence -> genuine = 0 -> PASS.
        validate_post_extract_quality(&classes, &pre_scan, 2, &labels, 100, 20)
            .expect("20% silence drop must NOT abort (benign filtering)");

        // 8 silence + 12 genuine -> 12% > 10% -> reject, reporting genuine.
        let err = validate_post_extract_quality(&classes, &pre_scan, 2, &labels, 100, 8)
            .expect_err("12% genuine-failure drop must reject");
        match err {
            FinetuneError::DropRatioExceeded {
                dropped,
                total,
                ratio,
                ..
            } => {
                assert_eq!(
                    dropped, 12,
                    "reported drop must be the genuine-failure count"
                );
                assert_eq!(total, 100);
                assert!(
                    (ratio - 0.12).abs() < 1e-4,
                    "ratio should be ~0.12: {ratio}"
                );
            }
            other => panic!("expected DropRatioExceeded, got {other:?}"),
        }
    }

    /// `validate_post_extract_quality` accepts a clean dataset (0% drop;
    /// gate is `>` not `>=`, no empty-class false positive).
    #[test]
    fn validate_post_extract_accepts_clean_dataset() {
        let classes = vec!["yes".to_string(), "no".to_string()];
        let pre_scan = vec![5usize, 5];
        let labels: Vec<usize> = (0..5).map(|_| 0).chain((0..5).map(|_| 1)).collect();
        validate_post_extract_quality(&classes, &pre_scan, 2, &labels, 10, 0)
            .expect("clean dataset must pass post-extract validation");
    }

    /// `validate_post_extract_quality` accepts exactly at the cap (10%
    /// drop); confirms the predicate is `>` not `>=`.
    #[test]
    fn validate_post_extract_accepts_at_cap_drop_ratio() {
        // 100 in, 90 out = exactly 10%.
        let classes = vec!["a".to_string(), "b".to_string()];
        let pre_scan = vec![50usize, 50];
        let labels: Vec<usize> = (0..45).map(|_| 0).chain((0..45).map(|_| 1)).collect();
        validate_post_extract_quality(&classes, &pre_scan, 2, &labels, 100, 0)
            .expect("at-cap drop ratio must pass (predicate is strict >)");
    }

    /// `validate_post_extract_quality` rejects when ONE class drops past
    /// [`MAX_PER_CLASS_DROP_RATIO`] while the aggregate stays under cap,
    /// so the per-class gate is the only thing catching it.
    #[test]
    fn validate_post_extract_rejects_per_class_drop_exceeded() {
        // Class 0 drops 12/20 = 60% (> per-class cap) while the aggregate
        // stays 12/200 = 6% (under cap), so only the per-class gate catches
        // it.
        let classes: Vec<String> = (0..10).map(|i| format!("class_{i}")).collect();
        let pre_scan: Vec<usize> = std::iter::repeat_n(20usize, 10).collect();
        let mut labels: Vec<usize> = Vec::with_capacity(8 + 9 * 20);
        labels.extend(std::iter::repeat_n(0usize, 8));
        for c in 1..10usize {
            labels.extend(std::iter::repeat_n(c, 20));
        }
        let err = validate_post_extract_quality(&classes, &pre_scan, 10, &labels, 200, 0)
            .expect_err("per-class drop must reject");
        match err {
            FinetuneError::PerClassDropExceeded {
                class,
                dropped,
                total,
                ratio,
                max_ratio,
                ..
            } => {
                assert_eq!(class, "class_0");
                assert_eq!(dropped, 12);
                assert_eq!(total, 20);
                assert!((ratio - 0.60).abs() < 1e-6, "ratio mismatch: {ratio}");
                assert!(
                    (max_ratio - MAX_PER_CLASS_DROP_RATIO).abs() < 1e-6,
                    "max_ratio mismatch: {max_ratio}",
                );
            }
            other => panic!("expected PerClassDropExceeded, got {other:?}"),
        }
    }

    /// `per_class_counts_from_labels` buckets a flat labels vec.
    #[test]
    fn per_class_counts_from_labels_buckets_correctly() {
        let labels = vec![0usize, 1, 0, 2, 1, 0];
        let counts = per_class_counts_from_labels(3, &labels);
        assert_eq!(counts, vec![3, 2, 1]);
        assert_eq!(per_class_counts_from_labels(4, &[]), vec![0; 4]);
        // Out-of-range labels dropped, not panicked.
        let counts = per_class_counts_from_labels(2, &[0, 99, 1]);
        assert_eq!(counts, vec![1, 1]);
    }

    /// An empty datasets root rejects with `BadDataset`.
    #[test]
    fn scan_rejects_no_class_folders() {
        let dir = tempfile::tempdir().expect("tempdir");
        let err = scan_dataset(dir.path()).expect_err("empty root rejects");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert_eq!(path, dir.path().display().to_string());
                assert!(
                    reason.contains("no class folders"),
                    "reason must mention `no class folders`; got {reason:?}",
                );
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// Stray non-hidden non-dir root entries reject with `BadDataset`.
    #[test]
    fn scan_rejects_stray_root_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(dir.path().join("cat").join("a.wav"), b"x").unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(dir.path().join("README.txt"), b"meta").unwrap();
        let err = scan_dataset(dir.path()).expect_err("stray root file rejects");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert!(path.ends_with("README.txt"));
                assert!(
                    reason.contains("not a directory"),
                    "reason must mention non-directory rejection; got {reason:?}",
                );
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// Hidden root entries (leading `.`) are silently ignored.
    #[test]
    fn scan_skips_hidden_root_entries() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        std::fs::create_dir(dir.path().join("dog")).unwrap();
        for cls in ["cat", "dog"] {
            #[allow(clippy::disallowed_methods)]
            std::fs::write(dir.path().join(cls).join("s.wav"), b"x").unwrap();
        }
        std::fs::create_dir(dir.path().join(".cache")).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(dir.path().join(".DS_Store"), b"").unwrap();
        let (classes, examples) = scan_dataset(dir.path()).expect("hidden ignored");
        assert_eq!(classes, vec!["cat".to_string(), "dog".to_string()]);
        assert_eq!(examples.len(), 2);
    }

    /// Case-insensitive duplicate class labels reject with `BadDataset`.
    /// Skipped on case-insensitive filesystems (macOS APFS) that can't
    /// stage `Cat/` + `cat/` as siblings.
    #[test]
    fn scan_rejects_case_insensitive_duplicate_labels() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Probe case-sensitivity: a second create at differing case returns
        // AlreadyExists on a case-insensitive FS.
        let probe_a = dir.path().join("case-probe-A");
        let probe_b = dir.path().join("case-probe-a");
        std::fs::create_dir(&probe_a).expect("probe a");
        let case_sensitive = std::fs::create_dir(&probe_b).is_ok();
        std::fs::remove_dir_all(&probe_a).ok();
        std::fs::remove_dir_all(&probe_b).ok();
        if !case_sensitive {
            eprintln!(
                "skipping scan_rejects_case_insensitive_duplicate_labels: \
                 host filesystem is case-insensitive (cannot stage Cat/ + cat/ as siblings)"
            );
            return;
        }

        std::fs::create_dir(dir.path().join("Cat")).unwrap();
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        for cls in ["Cat", "cat"] {
            #[allow(clippy::disallowed_methods)]
            std::fs::write(dir.path().join(cls).join("s.wav"), b"x").unwrap();
        }
        let err = scan_dataset(dir.path()).expect_err("case-insensitive duplicate rejects");
        match err {
            FinetuneError::BadDataset { reason, .. } => {
                assert!(
                    reason.contains("duplicate class label") && reason.contains("case-insensitive"),
                    "reason must mention case-insensitive duplicate; got {reason:?}",
                );
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// A class folder with no non-hidden `.wav` files anywhere
    /// under its subtree rejects with `BadDataset`.
    #[test]
    fn scan_rejects_empty_class_folder() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        std::fs::create_dir(dir.path().join("dog")).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(dir.path().join("dog").join("s.wav"), b"x").unwrap();
        let err = scan_dataset(dir.path()).expect_err("empty class rejects");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert!(path.ends_with("cat"));
                assert!(reason.contains("no non-hidden .wav"));
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// Classes are sorted by canonical byte order so the published
    /// head's label order is deterministic across hosts.
    #[test]
    fn scan_sorts_classes_by_byte_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        for cls in ["zebra", "ant", "manatee"] {
            std::fs::create_dir(dir.path().join(cls)).unwrap();
            #[allow(clippy::disallowed_methods)]
            std::fs::write(dir.path().join(cls).join("s.wav"), b"x").unwrap();
        }
        let (classes, _) = scan_dataset(dir.path()).expect("scan");
        assert_eq!(
            classes,
            vec![
                "ant".to_string(),
                "manatee".to_string(),
                "zebra".to_string()
            ],
        );
    }

    /// Nested non-hidden `.wav` files count as samples; hidden subdirs
    /// and non-`.wav` regular files are skipped.
    #[test]
    fn scan_recursive_discovery_picks_up_nested_wavs() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        std::fs::create_dir(dir.path().join("cat").join("subdir")).unwrap();
        std::fs::create_dir(dir.path().join("cat").join(".cache")).unwrap();
        std::fs::create_dir(dir.path().join("dog")).unwrap();
        for path in [
            "cat/a.wav",
            "cat/subdir/b.wav",
            "cat/subdir/c.txt",       // non-wav, skipped
            "cat/.cache/skipped.wav", // hidden, skipped
            "dog/d.wav",
        ] {
            #[allow(clippy::disallowed_methods)]
            std::fs::write(dir.path().join(path), b"x").unwrap();
        }
        let (classes, examples) = scan_dataset(dir.path()).expect("scan");
        assert_eq!(classes, vec!["cat".to_string(), "dog".to_string()]);
        assert_eq!(examples.len(), 3);
        let cat_count = examples.iter().filter(|(_, l)| *l == 0).count();
        assert_eq!(cat_count, 2);
    }

    /// Unsupported file types inside a class folder reject with
    /// `BadDataset`; uses a UDS (`std::fs` creates one without root).
    #[cfg(unix)]
    #[test]
    fn scan_rejects_unsupported_file_type_inside_class() {
        use std::os::unix::net::UnixListener;
        let dir = tempfile::tempdir().expect("tempdir");
        let cls = dir.path().join("cat");
        std::fs::create_dir(&cls).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(cls.join("a.wav"), b"x").unwrap();
        // Tempdir paths fit `sockaddr_un.sun_path` (104/108).
        let sock_path = cls.join("strange.sock");
        let _listener = UnixListener::bind(&sock_path).expect("bind unix socket");

        let err = scan_dataset(dir.path()).expect_err("unsupported file rejects");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert!(path.ends_with("strange.sock"));
                assert!(reason.contains("unsupported file type"));
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// Root-level symlinks reject as non-directories;
    /// `symlink_metadata` doesn't follow, so a symlink-to-dir rejects.
    #[cfg(unix)]
    #[test]
    fn scan_rejects_symlink_at_root_level() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("cat")).unwrap();
        #[allow(clippy::disallowed_methods)]
        std::fs::write(dir.path().join("cat").join("a.wav"), b"x").unwrap();
        symlink(dir.path().join("cat"), dir.path().join("cat-link")).unwrap();

        let err = scan_dataset(dir.path()).expect_err("symlink rejects");
        match err {
            FinetuneError::BadDataset { path, reason } => {
                assert!(path.ends_with("cat-link"));
                assert!(reason.contains("not a directory"));
            }
            other => panic!("expected BadDataset, got {other:?}"),
        }
    }

    /// `BadDataset` -> 400, `DatasetRead` -> 500; pins the
    /// operator-vs-daemon-internal split.
    #[test]
    fn finetune_error_kinds_classify_correctly() {
        use crate::common::error::{Categorized, ErrorKind};
        let bad = FinetuneError::BadDataset {
            path: "/x".into(),
            reason: "y".into(),
        };
        assert_eq!(bad.kind(), ErrorKind::UserInput);
        let read = FinetuneError::DatasetRead {
            path: "/x".into(),
            reason: "y".into(),
        };
        assert_eq!(read.kind(), ErrorKind::Internal);
    }
}
