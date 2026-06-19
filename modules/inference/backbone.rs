//! Backbone abstraction: NPU (RKNN) or CPU (Burn). 9976-element spectrogram per
//! window in, 2000-element feature vector out.
//!
//! Burn fp32 is the canonical reference; RKNN runs fp16 internally (~1e-3 drift).
//! Both hold per-call scratch, so concurrent calls need external sync (engine
//! owns via `&mut`).
//!
//! Layout: `Preproc::spectrogram` returns frame-major row-major. RKNN wants
//! bin-major flat (dims `[1,232,1,43]`) so `RknnBackbone` transposes via
//! [`crate::inference::kernel::transpose_frame_major_to_bin_major`]; Burn's
//! `forward` wants NCHW `[1,1,43,232]`, which the row-major flatten of
//! `spec[h][w]` matches exactly (C=1), so no transpose.

#![allow(missing_debug_implementations)]

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::model::Backbone as BurnNet;
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

use crate::common::dims::{BackboneFeatureDim, NBins, NFrames};
use crate::common::hex::hex_lowercase;
#[cfg(all(target_os = "linux", feature = "rknpu"))]
use crate::rknn_runtime::{InputSlice, OutputSlice, Session, TensorFormat};

#[cfg(all(target_os = "linux", feature = "rknpu"))]
use crate::inference::kernel::transpose_frame_major_to_bin_major;

/// Burn backend pinned to NdArray fp32 -- the single CPU choice.
type B = NdArray<f32>;

#[derive(Debug, Error)]
pub enum BackboneError {
    #[error("read backbone {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[cfg(all(target_os = "linux", feature = "rknpu"))]
    #[error("rknn: {0}")]
    Rknn(#[from] crate::rknn_runtime::Error),
    #[error("burn: {0}")]
    Burn(String),
    #[error(
        "backbone i/o counts: must be 1 input + 1 output; \
         got {n_input} inputs / {n_output} outputs"
    )]
    IoCount { n_input: u32, n_output: u32 },
    #[error("backbone input has {got} elements; expected {expected}")]
    InputDim { got: usize, expected: usize },
    #[error("backbone output has {got} elements; expected {expected}")]
    OutputDim { got: usize, expected: usize },
    /// Host buffer is f32; an int8/int32 RKNN model would mis-interpret the bytes.
    #[error("backbone {tensor} dtype unsupported: got {got:?}, expected Float32")]
    Dtype { tensor: &'static str, got: String },
    /// The bin-major transpose is orientation-correct only for NHWC input;
    /// Nchw/Nc1Hwc2 would silently mis-interpret the bytes.
    #[error("backbone {tensor} layout unsupported: got {got:?}, expected Nhwc")]
    Layout { tensor: &'static str, got: String },
    /// The count/dtype/format gates all pass a frequency<->time axis swap that
    /// the unconditional bin-major transpose then silently scrambles (no NaN/Err
    /// for the finite gate); only `[NBins,NFrames]`-ordered non-unit extents work.
    #[error(
        "backbone {tensor} dims orientation unsupported: got {got}, \
         expected non-unit extents [232, 43] (bins, frames), e.g. dims [1, 232, 1, 43]"
    )]
    DimsOrientation { tensor: &'static str, got: String },
    /// `librknnrt.so` (or `RKNN_LIB` alternative) not found; distinct from `Read`
    /// so the log names the missing runtime dep, not a backbone file.
    #[error("rknn runtime library not found: {detail}")]
    LibraryNotFound { detail: String },
    #[error("no usable backbone candidate; {summary}")]
    NoUsableCandidate { summary: String },
}

impl crate::common::error::Categorized for BackboneError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            BackboneError::Read { .. } => Internal,
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            BackboneError::Rknn(_) => Internal,
            BackboneError::Burn(_) => Internal,
            // Malformed / wrong-shape file => UserInput.
            BackboneError::IoCount { .. }
            | BackboneError::InputDim { .. }
            | BackboneError::OutputDim { .. }
            | BackboneError::Dtype { .. }
            | BackboneError::DimsOrientation { .. }
            | BackboneError::Layout { .. } => UserInput,
            BackboneError::LibraryNotFound { .. } => Internal,
            // Transient: re-plug device + retry.
            BackboneError::NoUsableCandidate { .. } => Unavailable,
        }
    }
}

/// `path` is `impl Display` to take either a `Path::display()` adapter or a
/// literal sentinel like `"librknnrt.so"`.
#[cfg(all(target_os = "linux", feature = "rknpu"))]
fn read_err(path: impl std::fmt::Display, source: std::io::Error) -> BackboneError {
    BackboneError::Read {
        path: path.to_string(),
        source,
    }
}

/// Basename of `p`, else full `display()`; avoids leaking server-side fs layout
/// in operator-facing display fields.
#[cfg(all(target_os = "linux", feature = "rknpu"))]
fn basename_or_full(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

/// True iff `dims` non-unit extents are exactly `[NBins,NFrames]`=`[232,43]` in
/// order -- the only orientation [`transpose_frame_major_to_bin_major`]'s
/// `out[bin*NFrames+frame]` is correct for; a frames-before-bins swap or
/// collapsed shape is rejected.
#[cfg(any(all(target_os = "linux", feature = "rknpu"), test))]
fn input_dims_orientation_ok(dims: &[u32]) -> bool {
    let non_unit: Vec<u32> = dims.iter().copied().filter(|&d| d != 1).collect();
    non_unit == [NBins::VALUE, NFrames::VALUE]
}

/// Rockchip NPU backbone. Built only with `--features rknpu` on Linux; else
/// cfg-gated out and `BackboneKind::Rknn::is_supported()` is `false`.
#[cfg(all(target_os = "linux", feature = "rknpu"))]
pub struct RknnBackbone {
    session: Session,
    /// Bin-major scratch (`spec_flat[bin*NFrames+frame]`); allocated once at load.
    spec_flat: Vec<f32>,
    description: String,
}

#[cfg(all(target_os = "linux", feature = "rknpu"))]
impl std::fmt::Debug for RknnBackbone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RknnBackbone")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

#[cfg(all(target_os = "linux", feature = "rknpu"))]
impl RknnBackbone {
    /// Resolve the runtime lib (`RKNN_LIB` override else `find_library_candidates`),
    /// then validate 1 input + 1 output and the `NFrames x NBins ->
    /// BackboneFeatureDim` shapes.
    pub fn load(backbone_rknn: &Path) -> Result<Self, BackboneError> {
        let lib = resolve_rknn_library()?;
        let mut bytes =
            std::fs::read(backbone_rknn).map_err(|e| read_err(backbone_rknn.display(), e))?;
        // SAFETY: trusted lib (immutable rootfs + restricted RKNN_LIB pin the
        // vendored librknnrt.so; no post-boot swap) with ABI matched by the
        // generated bindings.rs.
        let session = unsafe { Session::load(&lib, &mut bytes)? };

        let io = session.io_count()?;
        if io.n_input != 1 || io.n_output != 1 {
            return Err(BackboneError::IoCount {
                n_input: io.n_input,
                n_output: io.n_output,
            });
        }
        let in_attr = session.input_attr(0)?;
        let in_elems = in_attr.n_elems as usize;
        if in_elems != NFrames::USIZE * NBins::USIZE {
            return Err(BackboneError::InputDim {
                got: in_elems,
                expected: NFrames::USIZE * NBins::USIZE,
            });
        }
        // `pass_through=false` lets librknnrt convert the f32 host buffer to the
        // model's native FLOAT dtype, so accept any float (rv1126b declares
        // Float16); integer/quantized would mis-read the bytes. Widening this set
        // needs the production .rknn's reported input dtype confirmed on-device.
        use crate::rknn_runtime::DataType;
        if !matches!(
            in_attr.dtype,
            DataType::Float32 | DataType::Float16 | DataType::Bfloat16
        ) {
            return Err(BackboneError::Dtype {
                tensor: "input",
                got: format!("{:?}", in_attr.dtype),
            });
        }
        // Bin-major transpose is correct only for NHWC input; NCHW would
        // silently mis-read bytes. `Undefined` allowed (channel-collapsed exports).
        use crate::rknn_runtime::TensorFormat;
        if !matches!(in_attr.format, TensorFormat::Nhwc | TensorFormat::Undefined) {
            return Err(BackboneError::Layout {
                tensor: "input",
                got: format!("{:?}", in_attr.format),
            });
        }
        if !input_dims_orientation_ok(&in_attr.dims) {
            return Err(BackboneError::DimsOrientation {
                tensor: "input",
                got: format!("{:?}", in_attr.dims),
            });
        }
        let out_attr = session.output_attr(0)?;
        let out_elems = out_attr.n_elems as usize;
        if out_elems != BackboneFeatureDim::USIZE {
            return Err(BackboneError::OutputDim {
                got: out_elems,
                expected: BackboneFeatureDim::USIZE,
            });
        }
        // `want_float=true` dequantizes output to f32 (native fp16/bf16 OK);
        // reject only integer/quantized native output dtypes.
        if !matches!(
            out_attr.dtype,
            DataType::Float32 | DataType::Float16 | DataType::Bfloat16
        ) {
            return Err(BackboneError::Dtype {
                tensor: "output",
                got: format!("{:?}", out_attr.dtype),
            });
        }
        // Output is 1D today (layout moot), but an NCHW re-export would scramble
        // features; mirror the input format gate.
        if !matches!(
            out_attr.format,
            TensorFormat::Nhwc | TensorFormat::Undefined
        ) {
            return Err(BackboneError::Layout {
                tensor: "output",
                got: format!("{:?}", out_attr.format),
            });
        }
        Ok(Self {
            session,
            spec_flat: vec![0.0; NFrames::USIZE * NBins::USIZE],
            description: format!(
                "RKNN: {} (lib: {})",
                basename_or_full(backbone_rknn),
                basename_or_full(&lib),
            ),
        })
    }

    /// One inference, writing features in place (bin-major transpose first).
    pub fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError> {
        transpose_frame_major_to_bin_major::<{ NFrames::USIZE }, { NBins::USIZE }>(
            spec,
            &mut self.spec_flat,
        );
        self.session.infer(
            InputSlice::f32(0, &mut self.spec_flat).with_format(TensorFormat::Nhwc),
            OutputSlice::f32_preallocated(0, features),
        )?;
        Ok(())
    }

    pub fn description(&self) -> &str {
        &self.description
    }
}

/// `RKNN_LIB` override, else first `find_library_candidates` hit; emits
/// `LibraryNotFound` (not `Read`) when neither yields a path.
#[cfg(all(target_os = "linux", feature = "rknpu"))]
fn resolve_rknn_library() -> Result<std::path::PathBuf, BackboneError> {
    if let Some(p) = std::env::var_os("RKNN_LIB") {
        let pb = std::path::PathBuf::from(p);
        if !pb.exists() {
            return Err(BackboneError::LibraryNotFound {
                detail: format!("RKNN_LIB={} points at a non-existent file", pb.display()),
            });
        }
        return Ok(pb);
    }
    crate::rknn_runtime::utils::find_library_candidates()
        .into_iter()
        .next()
        .ok_or_else(|| BackboneError::LibraryNotFound {
            detail: format!(
                "librknnrt.so / librknnmrt.so not found; searched: {}; \
                 set RKNN_LIB=/path/to/librknnrt.so or install it to a search dir",
                crate::rknn_runtime::utils::library_search_dir_descriptions().join(", "),
            ),
        })
}

/// CPU (NdArray fp32) backbone; host-dev fallback and canonical fp32 parity
/// reference.
pub struct BurnBackbone {
    backbone: BurnNet<B>,
    device: burn::tensor::Device<B>,
    description: String,
}

impl std::fmt::Debug for BurnBackbone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BurnBackbone")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl BurnBackbone {
    /// Allocates one `BurnNet<B>` (~5.7 MB) on the CPU device.
    pub fn load(backbone_mpk: &Path) -> Result<Self, BackboneError> {
        let device: burn::tensor::Device<B> = Default::default();
        let backbone = BurnNet::<B>::load_mpk(backbone_mpk, &device)
            .map_err(|e| BackboneError::Burn(format!("{e}")))?;
        Ok(Self {
            backbone,
            device,
            description: format!("Burn (NdArray fp32): {}", backbone_mpk.display()),
        })
    }

    pub fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError> {
        // Row-major flatten of spec[h][w] is exactly NCHW [1,1,43,232] (C=1), no
        // transpose. The Vec is unavoidable: TensorData::new consumes it.
        let flat: Vec<f32> = spec.as_slice().as_flattened().to_vec();
        let input = Tensor::<B, 4>::from_data(
            TensorData::new(flat, [1, 1, NFrames::USIZE, NBins::USIZE]),
            &self.device,
        );
        let output = self.backbone.forward(input);
        // Defensive: won't fire for a correct backbone.mpk.
        let dims = output.dims();
        if dims != [1, BackboneFeatureDim::USIZE] {
            return Err(BackboneError::OutputDim {
                got: dims.iter().product::<usize>(),
                expected: BackboneFeatureDim::USIZE,
            });
        }
        // `let data` binding required: `as_slice` borrows it zero-alloc, so the
        // owner must outlive the borrow.
        let data = output.into_data();
        let slice = data
            .as_slice::<f32>()
            .map_err(|e| BackboneError::Burn(format!("as_slice: {e:?}")))?;
        if slice.len() != BackboneFeatureDim::USIZE {
            // Defensive: dims already checked; surface a Burn bug rather than
            // panic in copy_from_slice.
            return Err(BackboneError::OutputDim {
                got: slice.len(),
                expected: BackboneFeatureDim::USIZE,
            });
        }
        features.copy_from_slice(slice);
        Ok(())
    }

    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Enum-dispatch wrapper so the engine holds one concrete type (no hot-path
/// vtable). Variants boxed because [`BurnBackbone`]'s `Param`/pool/relu state is
/// tens of KB; boxing keeps the enum one pointer per variant.
pub enum BackbonePipeline {
    #[cfg(all(target_os = "linux", feature = "rknpu"))]
    Rknn(Box<RknnBackbone>),
    Burn(Box<BurnBackbone>),
}

impl std::fmt::Debug for BackbonePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            Self::Rknn(r) => r.fmt(f),
            Self::Burn(b) => b.fmt(f),
        }
    }
}

impl BackbonePipeline {
    pub fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError> {
        match self {
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            Self::Rknn(r) => r.infer(spec, features),
            Self::Burn(b) => b.infer(spec, features),
        }
    }

    pub fn description(&self) -> &str {
        match self {
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            Self::Rknn(r) => r.description(),
            Self::Burn(b) => b.description(),
        }
    }

    /// Erase the cfg-gated enum into the engine's `Box<dyn Backbone>` (lets tests
    /// substitute a mock).
    pub fn into_boxed(self) -> Box<dyn Backbone> {
        match self {
            #[cfg(all(target_os = "linux", feature = "rknpu"))]
            Self::Rknn(r) => r,
            Self::Burn(b) => b,
        }
    }
}

/// Produces a fixed-size feature vector from one preprocessed spectrogram. The
/// engine holds one as `Box<dyn Backbone>` so tests can mock without the
/// cfg-gated [`BackbonePipeline`] enum. Lives here (not `common`) to avoid
/// forcing `common` to depend on `preproc`/`BackboneError`.
///
/// `&mut self` is mandatory -- RKNN sessions are stateful and Burn impls hold
/// per-call scratch, so concurrent inference needs per-thread sessions; `infer`
/// writes in-place to an engine-preallocated buffer (avoids ~8 KB/frame churn).
pub trait Backbone: Send + std::fmt::Debug + 'static {
    /// Output feature dim; a method (not assoc const) for object safety.
    fn feature_dim(&self) -> BackboneFeatureDim {
        BackboneFeatureDim::default()
    }

    /// Operator-readable description (heartbeat + failure log lines).
    fn description(&self) -> &str;

    fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError>;
}

#[cfg(all(target_os = "linux", feature = "rknpu"))]
impl Backbone for RknnBackbone {
    fn description(&self) -> &str {
        RknnBackbone::description(self)
    }
    fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError> {
        RknnBackbone::infer(self, spec, features)
    }
}

impl Backbone for BurnBackbone {
    fn description(&self) -> &str {
        BurnBackbone::description(self)
    }
    fn infer(
        &mut self,
        spec: &[[f32; NBins::USIZE]; NFrames::USIZE],
        features: &mut [f32; BackboneFeatureDim::USIZE],
    ) -> Result<(), BackboneError> {
        BurnBackbone::infer(self, spec, features)
    }
}

// Object-safety smoke: a future non-dyn-compatible method fails here, not as a
// confusing trait-object error in the engine.
#[cfg(test)]
const _: fn() = || {
    fn assert_obj_safe<T: ?Sized>() {}
    assert_obj_safe::<dyn Backbone>();
};

/// On-disk type tag for a backbone candidate; drives per-kind dispatch and the
/// cfg-gated support filter.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackboneKind {
    /// Rockchip NPU (`*.rknn`).
    Rknn,
    /// Burn fp32 (`*.mpk`), CPU.
    Burn,
}

impl BackboneKind {
    /// True when this build can load this kind; the loader skips unsupported
    /// kinds during catalogue traversal rather than failing boot.
    pub const fn is_supported(self) -> bool {
        match self {
            BackboneKind::Burn => true,
            BackboneKind::Rknn => cfg!(all(target_os = "linux", feature = "rknpu")),
        }
    }
}

/// A single backbone candidate as declared in the launch config.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BackboneRef {
    pub kind: BackboneKind,
    pub path: PathBuf,
    /// Optional sha256 digest, bare hex (64 chars, case-insensitive); checked
    /// before instantiation, mismatch is non-fatal (candidate skipped, next tried).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

impl BackboneRef {
    /// Static well-formedness check at `LaunchConfig::load`, so operators see
    /// catalogue typos in systemd logs rather than at runtime.
    pub fn validate(&self) -> Result<(), String> {
        if self.path.as_os_str().is_empty() {
            return Err("path must not be empty".into());
        }
        if let Some(h) = &self.hash {
            let h = h.trim();
            if h.len() != 64 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(format!(
                    "hash must be 64 hex chars, case-insensitive (got {} chars: {h:?})",
                    h.len(),
                ));
            }
        }
        Ok(())
    }
}

/// Ordered backbone candidates; the loader picks the first usable one in
/// declaration order ([`Self::load_first_supported`]). Empty is legal.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BackboneCatalogue {
    #[serde(default)]
    pub candidates: Vec<BackboneRef>,
}

impl BackboneCatalogue {
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Validate every candidate; returns the first failure's index + diagnostic.
    pub fn validate(&self) -> Result<(), (usize, String)> {
        for (i, c) in self.candidates.iter().enumerate() {
            if let Err(e) = c.validate() {
                return Err((i, e));
            }
        }
        Ok(())
    }

    /// First successfully loaded backbone in declaration order, skipping
    /// unsupported kinds and hash mismatches via `tracing::warn!`; else
    /// [`BackboneError::NoUsableCandidate`] summarizing every attempt.
    ///
    /// Blocking (file I/O + RKNN C-FFI session-create); call off the async path
    /// (daemon uses `spawn_blocking`).
    pub fn load_first_supported(&self) -> Result<BackbonePipeline, BackboneError> {
        let mut summaries: Vec<String> = Vec::with_capacity(self.candidates.len());
        for cand in &self.candidates {
            match try_load_candidate(cand) {
                Ok(pipeline) => {
                    tracing::info!(
                        target: "inference",
                        kind = ?cand.kind,
                        path = %cand.path.display(),
                        "backbone candidate loaded",
                    );
                    // Heads are fit to the first `kind="burn"` candidate, but
                    // serving resolves the first *supported* one (a different
                    // artifact on RKNN devices) and heads carry no backbone
                    // identity, so a basis mismatch classifies silently wrong.
                    // Warn, not fail-closed: a faithful conversion is still valid.
                    match basis_relation(&self.candidates, cand) {
                        BasisRelation::Consistent => {}
                        BasisRelation::DivergesFromTrainingBurn { training_path } => {
                            tracing::warn!(
                                target: "inference",
                                serving_kind = ?cand.kind,
                                serving_path = %cand.path.display(),
                                training_path = %training_path.display(),
                                "serving backbone diverges from the head-training (Burn) basis; \
                                 unless it represents the same basis (e.g. a faithful conversion), \
                                 classification is silently wrong -- verify via the converter",
                            );
                        }
                        BasisRelation::NoTrainingBurn => {
                            tracing::warn!(
                                target: "inference",
                                serving_kind = ?cand.kind,
                                serving_path = %cand.path.display(),
                                "no `kind = \"burn\"` backbone candidate configured; cannot \
                                 verify the served backbone matches the head-training basis",
                            );
                        }
                    }
                    return Ok(pipeline);
                }
                Err(reason) => {
                    tracing::warn!(
                        target: "inference",
                        kind = ?cand.kind,
                        path = %cand.path.display(),
                        reason = %reason,
                        "backbone candidate skipped",
                    );
                    summaries.push(format!(
                        "[{:?} {}]: {}",
                        cand.kind,
                        cand.path.display(),
                        reason,
                    ));
                }
            }
        }
        let summary = if summaries.is_empty() {
            "catalogue is empty".to_string()
        } else {
            summaries.join("; ")
        };
        Err(BackboneError::NoUsableCandidate { summary })
    }
}

/// How the served backbone relates to the head-training (first Burn) feature
/// basis; drives the boot warning about a silent train/serve basis divergence.
#[derive(Debug, PartialEq, Eq)]
enum BasisRelation {
    /// Served IS the first `kind="burn"` candidate heads are fit to: same basis.
    Consistent,
    /// Served a different artifact (RKNN, or a non-first Burn); `training_path`
    /// is the first Burn `.mpk` heads were fit to.
    DivergesFromTrainingBurn { training_path: PathBuf },
    /// No `kind="burn"` candidate, so no training basis to compare.
    NoTrainingBurn,
}

/// Classify `served` against the first Burn candidate (the path training
/// resolves), so `Consistent` means train and serve provably load the same `.mpk`.
fn basis_relation(candidates: &[BackboneRef], served: &BackboneRef) -> BasisRelation {
    match candidates.iter().find(|c| c.kind == BackboneKind::Burn) {
        None => BasisRelation::NoTrainingBurn,
        Some(burn) if burn.path == served.path => BasisRelation::Consistent,
        Some(burn) => BasisRelation::DivergesFromTrainingBurn {
            training_path: burn.path.clone(),
        },
    }
}

fn try_load_candidate(cand: &BackboneRef) -> Result<BackbonePipeline, String> {
    if !cand.kind.is_supported() {
        return Err("feature not supported in this build".into());
    }
    if !cand.path.exists() {
        return Err(format!("file does not exist: {}", cand.path.display()));
    }
    if let Some(expected) = &cand.hash {
        verify_sha256(&cand.path, expected)?;
    }
    match cand.kind {
        BackboneKind::Rknn => load_rknn(&cand.path),
        BackboneKind::Burn => BurnBackbone::load(&cand.path)
            .map(|b| BackbonePipeline::Burn(Box::new(b)))
            .map_err(|e| format!("{e}")),
    }
}

#[cfg(all(target_os = "linux", feature = "rknpu"))]
fn load_rknn(path: &Path) -> Result<BackbonePipeline, String> {
    RknnBackbone::load(path)
        .map(|b| BackbonePipeline::Rknn(Box::new(b)))
        .map_err(|e| format!("{e}"))
}

#[cfg(not(all(target_os = "linux", feature = "rknpu")))]
fn load_rknn(_path: &Path) -> Result<BackbonePipeline, String> {
    // Unreachable via the is_supported short-circuit; guards a future direct caller.
    Err("rknn backbone not supported in this build".into())
}

/// Streaming SHA-256 of `path` vs `expected_hex` (case-insensitive, trimmed);
/// 64 KiB chunks so a multi-MB backbone doesn't pin memory.
fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let expected = expected_hex.trim();
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("open {} for hash: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {} for hash: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got_hex = hex_lowercase(&hasher.finalize());
    if got_hex.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(format!(
            "sha256 mismatch: expected {expected}, got {got_hex}"
        ))
    }
}

#[cfg(test)]
mod tests {
    // Fixtures use `std::fs::write`; the production clippy.toml ban doesn't apply.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use std::path::PathBuf;

    fn crate_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    fn bref(kind: BackboneKind, path: &str) -> BackboneRef {
        BackboneRef {
            kind,
            path: PathBuf::from(path),
            hash: None,
        }
    }

    /// Guards the boot train/serve basis divergence check: served vs first Burn.
    #[test]
    fn basis_relation_classifies_train_vs_serve() {
        // Device hybrid: RKNN served, training fits the Burn .mpk -> divergence.
        let cands = vec![
            bref(BackboneKind::Rknn, "/b/backbone.rknn"),
            bref(BackboneKind::Burn, "/b/backbone.mpk"),
        ];
        assert_eq!(
            basis_relation(&cands, &cands[0]),
            BasisRelation::DivergesFromTrainingBurn {
                training_path: PathBuf::from("/b/backbone.mpk"),
            },
        );
        assert_eq!(basis_relation(&cands, &cands[1]), BasisRelation::Consistent);

        // Host single-Burn config: serve == first (only) Burn.
        let cands = vec![bref(BackboneKind::Burn, "/b/backbone.mpk")];
        assert_eq!(basis_relation(&cands, &cands[0]), BasisRelation::Consistent);

        // A later Burn than the training first-Burn -> diverges to the first.
        let cands = vec![
            bref(BackboneKind::Burn, "/b/first.mpk"),
            bref(BackboneKind::Burn, "/b/second.mpk"),
        ];
        assert_eq!(
            basis_relation(&cands, &cands[1]),
            BasisRelation::DivergesFromTrainingBurn {
                training_path: PathBuf::from("/b/first.mpk"),
            },
        );

        // No Burn candidate -> nothing to compare against.
        let cands = vec![bref(BackboneKind::Rknn, "/b/backbone.rknn")];
        assert_eq!(
            basis_relation(&cands, &cands[0]),
            BasisRelation::NoTrainingBurn,
        );
    }

    /// The dims-orientation gate accepts the production shape (+ unit-padded
    /// equivalents) but rejects a frequency<->time axis swap.
    #[test]
    fn input_dims_orientation_gate_accepts_canonical_rejects_swap() {
        let b = NBins::VALUE; // 232 bins
        let f = NFrames::VALUE; // 43 frames

        assert!(
            input_dims_orientation_ok(&[1, b, 1, f]),
            "production [1,232,1,43] must pass",
        );
        assert!(
            input_dims_orientation_ok(&[b, f]),
            "bare [232,43] must pass"
        );
        assert!(input_dims_orientation_ok(&[1, b, f, 1]));
        assert!(input_dims_orientation_ok(&[b, 1, 1, f]));

        // Swap keeps n_elems=9976 and is NHWC-taggable, so only this gate catches it.
        assert!(
            !input_dims_orientation_ok(&[1, f, b, 1]),
            "swapped [1,43,232,1] must reject",
        );
        assert!(
            !input_dims_orientation_ok(&[f, b]),
            "swapped [43,232] must reject",
        );

        // Collapsed / ambiguous / empty: orientation undefined.
        assert!(
            !input_dims_orientation_ok(&[b * f]),
            "collapsed [9976] must reject",
        );
        assert!(!input_dims_orientation_ok(&[1, 1, 1, 1]));
        assert!(!input_dims_orientation_ok(&[]));
    }

    /// Burn backbone loads the shipped .mpk and yields finite features on zero
    /// input (biases dominate).
    #[test]
    #[ignore = "depends on repo-root reference assets"]
    fn burn_backbone_loads_and_runs_on_zero_input() {
        let path = crate_root().join("misc/backbones/backbone.mpk");
        assert!(path.exists(), "missing test asset: {}", path.display());
        let mut bb = BurnBackbone::load(&path).expect("load");
        let spec = Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
        let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
        bb.infer(&spec, &mut features).expect("infer");
        assert!(features.iter().all(|v| v.is_finite()));
    }

    /// Bundled `backbone.mpk` carries the same conv + dense_1 weights as the
    /// upstream Speech-Commands TFJS model (skips if not fetched). TFJS conv
    /// kernels are HWIO (Keras), Burn `Conv2d` is OIHW (PyTorch), so re-index
    /// before an exact 1e-7 compare; `dense_1` is `[d_in,d_out]` in both.
    #[test]
    #[ignore = "depends on bundled fixtures + upstream TFJS model; --include-ignored"]
    fn backbone_mpk_matches_speech_commands_tfjs() {
        use crate::model::Backbone as BurnNet;

        let root = crate_root();
        let model_json = root.join("misc/models/model.json");
        if !model_json.exists() {
            eprintln!(
                "skipping: {} not present (run misc/models/get_tfjs_sc_model.sh)",
                model_json.display(),
            );
            return;
        }

        let manifest_bytes = std::fs::read(&model_json).expect("read model.json");
        let manifest =
            crate::converter::parse_tfjs_manifest(&manifest_bytes).expect("parse manifest");

        let model_dir = model_json.parent().unwrap();
        let mut blob: Vec<u8> = Vec::new();
        for shard in &manifest.shards {
            let p = model_dir.join(shard);
            let mut bytes =
                std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {}", p.display(), e));
            blob.append(&mut bytes);
        }

        let entry = |suffix: &str| -> &crate::converter::TfjsManifestEntry {
            manifest
                .entries
                .iter()
                .find(|e| e.name.ends_with(suffix))
                .unwrap_or_else(|| panic!("manifest missing entry ending in {suffix:?}"))
        };
        let tfjs_f32 = |suffix: &str| -> Vec<f32> {
            let e = entry(suffix);
            blob[e.offset_bytes..e.offset_bytes + e.len_bytes]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };

        let device: burn::tensor::Device<B> = Default::default();
        let backbone_path = root.join("misc/backbones/backbone.mpk");
        let backbone: BurnNet<B> =
            BurnNet::load_mpk(&backbone_path, &device).expect("load backbone");

        fn assert_close(name: &str, got: &[f32], expected: &[f32], tol: f32) {
            assert_eq!(
                got.len(),
                expected.len(),
                "{name}: length mismatch -- burn={}, tfjs={}",
                got.len(),
                expected.len(),
            );
            let mut max_diff = 0.0f32;
            let mut max_at = 0usize;
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                let d = (g - e).abs();
                if d > max_diff {
                    max_diff = d;
                    max_at = i;
                }
            }
            assert!(
                max_diff <= tol,
                "{name}: max |D|={max_diff} at flat idx {max_at} exceeds tol={tol}; \
                 burn={}, tfjs={}",
                got[max_at],
                expected[max_at],
            );
        }

        // Permute TFJS HWIO [kH, kW, in, out] -> Burn OIHW [out, in, kH, kW].
        fn hwio_to_oihw(hwio: &[f32], kh: usize, kw: usize, ic: usize, oc: usize) -> Vec<f32> {
            assert_eq!(hwio.len(), kh * kw * ic * oc);
            let mut oihw = vec![0.0f32; hwio.len()];
            for h in 0..kh {
                for w in 0..kw {
                    for i in 0..ic {
                        for o in 0..oc {
                            let src = ((h * kw + w) * ic + i) * oc + o;
                            let dst = ((o * ic + i) * kh + h) * kw + w;
                            oihw[dst] = hwio[src];
                        }
                    }
                }
            }
            oihw
        }

        // (Burn name, TFJS kernel suffix, TFJS bias suffix, [kH,kW,in,out]).
        let conv_layers: [(&str, &str, &str, [usize; 4]); 4] = [
            ("conv1", "conv2d_1/kernel", "conv2d_1/bias", [2, 8, 1, 8]),
            ("conv2", "conv2d_2/kernel", "conv2d_2/bias", [2, 4, 8, 32]),
            ("conv3", "conv2d_3/kernel", "conv2d_3/bias", [2, 4, 32, 32]),
            ("conv4", "conv2d_4/kernel", "conv2d_4/bias", [2, 4, 32, 32]),
        ];

        for (idx, (name, k_suffix, b_suffix, [kh, kw, ic, oc])) in conv_layers.iter().enumerate() {
            let conv = match idx {
                0 => &backbone.conv1,
                1 => &backbone.conv2,
                2 => &backbone.conv3,
                3 => &backbone.conv4,
                _ => unreachable!(),
            };
            let burn_w: Vec<f32> = conv
                .weight
                .val()
                .clone()
                .into_data()
                .to_vec()
                .expect("to_vec weight");
            let tfjs_w_oihw = hwio_to_oihw(&tfjs_f32(k_suffix), *kh, *kw, *ic, *oc);
            assert_close(&format!("{name}.weight"), &burn_w, &tfjs_w_oihw, 1e-7);

            let burn_b: Vec<f32> = conv
                .bias
                .as_ref()
                .expect("conv has bias")
                .val()
                .clone()
                .into_data()
                .to_vec()
                .expect("to_vec bias");
            let tfjs_b = tfjs_f32(b_suffix);
            assert_close(&format!("{name}.bias"), &burn_b, &tfjs_b, 1e-7);
        }

        let dense1_w_burn: Vec<f32> = backbone
            .dense1
            .weight
            .val()
            .clone()
            .into_data()
            .to_vec()
            .expect("to_vec dense1.weight");
        let dense1_w_tfjs: Vec<f32> = tfjs_f32("dense_1/kernel");
        assert_close("dense1.weight", &dense1_w_burn, &dense1_w_tfjs, 1e-7);

        let dense1_b_burn: Vec<f32> = backbone
            .dense1
            .bias
            .as_ref()
            .expect("dense1 has bias")
            .val()
            .clone()
            .into_data()
            .to_vec()
            .expect("to_vec dense1.bias");
        let dense1_b_tfjs: Vec<f32> = tfjs_f32("dense_1/bias");
        assert_close("dense1.bias", &dense1_b_burn, &dense1_b_tfjs, 1e-7);
    }

    /// Missing backbone.mpk surfaces `BackboneError::Burn`, not a panic.
    #[test]
    fn burn_backbone_load_missing_file_returns_err() {
        let path = std::path::Path::new("/nonexistent/.acousticslab/missing-backbone.mpk");
        let res = BurnBackbone::load(path);
        let err = res.unwrap_err();
        match err {
            BackboneError::Burn(msg) => {
                assert!(
                    !msg.is_empty(),
                    "Burn error should carry a non-empty message",
                );
            }
            other => panic!("expected BackboneError::Burn, got {other:?}"),
        }
    }

    /// Missing backbone.rknn surfaces `BackboneError::Read`.
    #[test]
    #[cfg(all(target_os = "linux", feature = "rknpu"))]
    fn rknn_backbone_load_missing_file_returns_err() {
        // RKNN_LIB at a non-existent path so library resolution fails with the
        // expected Read variant before the backbone read runs.
        // SAFETY: tests are single-threaded by default; env mutation is fine.
        unsafe {
            std::env::set_var(
                "RKNN_LIB",
                "/nonexistent/.acousticslab/missing-librknnrt.so",
            );
        }
        let bb = std::path::Path::new("/nonexistent/.acousticslab/missing-backbone.rknn");
        let res = RknnBackbone::load(bb);
        unsafe {
            std::env::remove_var("RKNN_LIB");
        }
        let err = res.unwrap_err();
        match err {
            BackboneError::Read { path, .. } => {
                assert!(
                    path.contains("missing-backbone") || path.contains("missing-librknnrt"),
                    "path mismatch: {path}",
                );
            }
            other => panic!("expected BackboneError::Read, got {other:?}"),
        }
    }

    /// `BackboneRef::validate` accepts a 64-char hex hash and rejects malformed shapes.
    #[test]
    fn backbone_ref_validate_hash_format() {
        let mut r = BackboneRef {
            kind: BackboneKind::Burn,
            path: PathBuf::from("/tmp/x.mpk"),
            hash: None,
        };
        assert!(r.validate().is_ok(), "no-hash candidate must validate");

        r.hash = Some("a".repeat(64));
        assert!(r.validate().is_ok(), "64 hex chars must validate");

        r.hash = Some("a".repeat(63)); // too short
        assert!(r.validate().is_err());

        r.hash = Some("g".repeat(64)); // non-hex
        assert!(r.validate().is_err());

        r.hash = None;
        r.path = PathBuf::new(); // empty path
        assert!(r.validate().is_err());
    }

    /// An empty catalogue surfaces `NoUsableCandidate` with an "empty" summary,
    /// not a panic-shaped error.
    #[test]
    fn empty_catalogue_returns_no_usable_candidate() {
        let cat = BackboneCatalogue::default();
        let err = cat.load_first_supported().expect_err("must be err");
        match err {
            BackboneError::NoUsableCandidate { summary } => {
                assert!(
                    summary.contains("empty"),
                    "summary should mention emptiness, got: {summary}"
                );
            }
            other => panic!("expected NoUsableCandidate, got {other:?}"),
        }
    }

    /// A non-existent file is named in the catalogue summary.
    #[test]
    fn missing_file_reported_in_summary() {
        let cat = BackboneCatalogue {
            candidates: vec![BackboneRef {
                kind: BackboneKind::Burn,
                path: PathBuf::from("/nonexistent/.acousticslab/missing.mpk"),
                hash: None,
            }],
        };
        let err = cat.load_first_supported().expect_err("must be err");
        let msg = err.to_string();
        assert!(
            msg.contains("missing.mpk") && msg.contains("does not exist"),
            "summary should name the missing file: {msg}",
        );
    }

    /// Hash mismatch is named in the candidate summary (fatal as the only candidate).
    #[test]
    fn hash_mismatch_reported_in_summary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("dummy.mpk");
        std::fs::write(&p, b"hello world").expect("write");
        let cat = BackboneCatalogue {
            candidates: vec![BackboneRef {
                kind: BackboneKind::Burn,
                path: p,
                hash: Some("0".repeat(64)),
            }],
        };
        let err = cat.load_first_supported().expect_err("must be err");
        let msg = err.to_string();
        assert!(
            msg.contains("sha256 mismatch"),
            "summary should call out hash mismatch: {msg}",
        );
    }

    /// On a non-rknpu build an `Rknn` candidate is reported unsupported without
    /// reaching load, exercising the `is_supported` short-circuit.
    #[test]
    #[cfg(not(all(target_os = "linux", feature = "rknpu")))]
    fn rknn_candidate_unsupported_on_non_rknpu_build() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("backbone.rknn");
        std::fs::write(&p, b"fake rknn bytes").expect("write");
        let cat = BackboneCatalogue {
            candidates: vec![BackboneRef {
                kind: BackboneKind::Rknn,
                path: p,
                hash: None,
            }],
        };
        let err = cat.load_first_supported().expect_err("must be err");
        let msg = err.to_string();
        assert!(
            msg.contains("not supported"),
            "summary should call out unsupported kind: {msg}",
        );
    }
}
