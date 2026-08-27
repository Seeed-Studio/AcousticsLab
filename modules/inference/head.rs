//! Hot-swappable classifier head over `Arc<VersionedSwap<HeadInner>>`: monotonic
//! [`ResourceVersion`] + writer mutex linearises swaps, reads stay wait-free, and
//! [`HeadInner`] bundles all swap-coupled fields so a mid-flight swap can never publish
//! new labels against old weights. Load/swap block; call from a non-async context.

use crate::model::Head;
use burn::backend::NdArray;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

use crate::common::dims::BackboneFeatureDim;
use crate::common::ids::HeadId;
use crate::common::traits::head_store::{HeadCandidate, HeadStore, HeadStoreError, HeadView};
use crate::common::version::{ResourceVersion, SwapReceipt, VersionedSwap};

// Bare ident because thiserror `#[error]` format strings interpolate idents, not
// `Type::CONST` paths.
const BACKBONE_FEATURE_DIM: usize = BackboneFeatureDim::USIZE;

/// Cold-path backend only (head load + bias hoist); the engine bypasses Burn in
/// steady state via `kernel::head_forward`.
type B = NdArray<f32>;

/// One self-consistent snapshot of head weights + metadata. `Clone` is DEEP
/// (copies the Vecs); hot-path code uses [`HotHead::snapshot`].
#[derive(Debug, Clone)]
pub struct HeadInner {
    /// Flat fp32 in Burn's `[in_features, out_features]` orientation
    /// (`[BACKBONE_FEATURE_DIM x n_classes]`); `head_forward` walks
    /// `weight.chunks_exact(n_classes)`.
    pub weight: Vec<f32>,

    /// Per-class bias (`len == n_classes`); a bias-free head gets zeros synthesised
    /// at load so `head_forward` stays uniform.
    pub bias: Vec<f32>,

    /// Class labels (`len == n_classes`); blank lines stripped and count mismatch
    /// rejected at load so the hot path indexes unconditionally.
    pub labels: Vec<String>,

    pub head_id: HeadId,

    pub n_classes: usize,
}

/// Shareable head handle (clone = one [`Arc`] bump); all clones share one
/// `VersionedSwap`, whose [`ResourceVersion`] backs `?min_version=N` read-your-write.
#[derive(Clone, Debug)]
pub struct HotHead {
    inner: Arc<VersionedSwap<HeadInner>>,
}

/// Failure shapes from [`HotHead::load`] and [`HotHead::swap`].
#[derive(Debug, Error)]
pub enum HeadError {
    #[error("read head .mpk {path}: {message}")]
    LoadMpk {
        path: String,
        // Rendered String, not `#[source]`: Burn's error isn't reliably `'static`.
        message: String,
    },
    #[error(
        "head .mpk weight has {got} elements, expected {expected} ({BACKBONE_FEATURE_DIM} x n_classes)"
    )]
    WeightShape { got: usize, expected: usize },
    #[error("head .mpk bias has {got} elements, expected {expected} (= n_classes)")]
    BiasShape { got: usize, expected: usize },
    #[error("head .mpk linear input dim is {got}, expected {BACKBONE_FEATURE_DIM}")]
    InputDim { got: usize },
    #[error("head .mpk produced n_classes = {got}; refusing (must be > 0 and <= {max})")]
    BadClassCount { got: usize, max: usize },
    #[error("read labels file {path}: {source}")]
    ReadLabels {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("read head .mpk {path}: {source}")]
    ReadHeadMpk {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("labels file {path} has {got} entries but head has {n_classes} classes")]
    LabelCountMismatch {
        path: String,
        got: usize,
        n_classes: usize,
    },
    /// In-memory counterpart to `LabelCountMismatch` (no on-disk path).
    #[error("HeadInner labels has {got} entries but n_classes is {n_classes}")]
    LabelShape { got: usize, n_classes: usize },
    /// Rejection keeps NaN/Inf out of emitted `InferenceFrame`s.
    #[error("head weight[{idx}] = {value} is not finite")]
    NonFiniteWeight { idx: usize, value: f32 },
    #[error("head bias[{idx}] = {value} is not finite")]
    NonFiniteBias { idx: usize, value: f32 },
    #[error("internal: tensor.into_data().to_vec failed: {0}")]
    TensorIntoVec(String),
    /// No `ACSTHEAD` magic (headerless artifact); distinct from `HeaderCorrupt`
    /// because remediation is "regenerate via converter", not "restore from backup".
    #[error(
        "head .mpk {path} missing ACSTHEAD header; regenerate via the converter (POST /api/v1/converter)"
    )]
    SchemaTooOld { path: String },
    /// Header present but malformed (bad CRC, schema too new, truncated).
    #[error("head .mpk {path} header corrupt: {reason}")]
    HeaderCorrupt { path: String, reason: String },
    /// Header `feature_dim` disagrees with this build's backbone (wrong topology
    /// or corrupt header past CRC).
    #[error("head .mpk {path} feature_dim {got} != build's {expected}")]
    HeaderFeatureDimMismatch {
        path: String,
        got: u32,
        expected: u32,
    },
}

impl crate::common::error::Categorized for HeadError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            HeadError::LoadMpk { .. }
            | HeadError::WeightShape { .. }
            | HeadError::BiasShape { .. }
            | HeadError::InputDim { .. }
            | HeadError::BadClassCount { .. }
            | HeadError::LabelCountMismatch { .. }
            | HeadError::LabelShape { .. }
            | HeadError::NonFiniteWeight { .. }
            | HeadError::NonFiniteBias { .. }
            | HeadError::SchemaTooOld { .. }
            | HeadError::HeaderCorrupt { .. }
            | HeadError::HeaderFeatureDimMismatch { .. } => UserInput,
            // Opaque IO reading labels/head .mpk, or Burn decode drift: not user's fault.
            HeadError::ReadLabels { .. } | HeadError::ReadHeadMpk { .. } => Internal,
            HeadError::TensorIntoVec(_) => Internal,
        }
    }
}

/// Re-export; canonical in `common::dims`.
pub use crate::common::dims::MAX_N_CLASSES;

/// Re-export; canonical in `common::workspace` so producer/import validators and
/// this `load_inner` consumer share one source of truth.
pub use crate::common::workspace::MAX_LABEL_BYTES;

impl HeadInner {
    /// Assert the hot-path invariants before inputs reach `head_forward`: `n_classes in
    /// [1, MAX_N_CLASSES]`, `weight.len() == n_classes * BACKBONE_FEATURE_DIM`,
    /// `bias.len() == labels.len() == n_classes`, all entries finite. Cold path.
    pub fn validate(&self) -> Result<(), HeadError> {
        if self.n_classes == 0 || self.n_classes > MAX_N_CLASSES {
            return Err(HeadError::BadClassCount {
                got: self.n_classes,
                max: MAX_N_CLASSES,
            });
        }
        let expected_weight = self.n_classes * BACKBONE_FEATURE_DIM;
        if self.weight.len() != expected_weight {
            return Err(HeadError::WeightShape {
                got: self.weight.len(),
                expected: expected_weight,
            });
        }
        if self.bias.len() != self.n_classes {
            return Err(HeadError::BiasShape {
                got: self.bias.len(),
                expected: self.n_classes,
            });
        }
        if self.labels.len() != self.n_classes {
            return Err(HeadError::LabelShape {
                got: self.labels.len(),
                n_classes: self.n_classes,
            });
        }
        self.validate_finite()
    }

    /// Scan `weight` then `bias` for the first non-finite entry. `O(n)`; cold path.
    pub fn validate_finite(&self) -> Result<(), HeadError> {
        for (idx, &v) in self.weight.iter().enumerate() {
            if !v.is_finite() {
                return Err(HeadError::NonFiniteWeight { idx, value: v });
            }
        }
        for (idx, &v) in self.bias.iter().enumerate() {
            if !v.is_finite() {
                return Err(HeadError::NonFiniteBias { idx, value: v });
            }
        }
        Ok(())
    }
}

impl HotHead {
    /// Build a head from disk. Blocking -- call from spawn_blocking.
    pub fn load(head_mpk: &Path, labels: &Path, head_id: HeadId) -> Result<Self, HeadError> {
        let inner = load_inner(head_mpk, labels, head_id)?;
        // try_from_inner re-validates: defence in depth if a revision drops an invariant.
        Self::try_from_inner(inner)
    }

    /// Atomic in-place swap: load + validate, then rotate the `VersionedSwap`. The old
    /// `Arc` drops only after the last snapshot guard releases, so no UAF mid-frame.
    pub fn swap(
        &self,
        head_mpk: &Path,
        labels: &Path,
        head_id: HeadId,
    ) -> Result<SwapReceipt, HeadError> {
        let new = load_inner(head_mpk, labels, head_id)?;
        let (receipt, _) = self
            .inner
            // `Infallible` pins the infallible-mutator contract at the type level.
            .try_mutate::<(), std::convert::Infallible>(|_cur| Ok((Arc::new(new), ())))
            .expect("infallible mutator");
        Ok(receipt)
    }

    /// Validating constructor: validates before publishing so a malformed inner surfaces
    /// here, not as a hot-loop panic.
    pub fn try_from_inner(inner: HeadInner) -> Result<Self, HeadError> {
        inner.validate()?;
        Ok(Self {
            inner: Arc::new(VersionedSwap::new(inner)),
        })
    }

    /// Panics on failure; prefer [`Self::try_from_inner`] for non-hand-constructed input.
    pub fn from_inner(inner: HeadInner) -> Self {
        Self::try_from_inner(inner)
            .unwrap_or_else(|e| panic!("HotHead::from_inner: {e}; use try_from_inner"))
    }

    /// Replace the inner directly (mirrors `swap` without file I/O). Validates so a
    /// shape-inconsistent inner surfaces here, not as a hot-path panic.
    pub fn store_inner(&self, inner: HeadInner) -> Result<SwapReceipt, HeadError> {
        inner.validate()?;
        let (receipt, _) = self
            .inner
            .try_mutate::<(), std::convert::Infallible>(|_cur| Ok((Arc::new(inner), ())))
            .expect("infallible mutator");
        Ok(receipt)
    }

    /// One `ArcSwap::load_full` aliasing the current snapshot (~5 ns). Hold for one frame.
    pub fn snapshot(&self) -> Arc<HeadInner> {
        self.inner.snapshot()
    }

    /// Current resource version; bumps on every successful swap.
    pub fn version(&self) -> ResourceVersion {
        self.inner.version()
    }

    /// Atomic `(snapshot, version)`. Reading them separately would race a swap landing
    /// between the loads, stamping version-N logits as N+1.
    pub fn snapshot_with_version(&self) -> (Arc<HeadInner>, ResourceVersion) {
        self.inner.snapshot_with_version()
    }

    pub fn n_classes(&self) -> usize {
        self.snapshot().n_classes
    }
}

/// `HeadStore` backing `api::AppState::head`; reads delegate to `HotHead`, load+install
/// runs under the `VersionedSwap` writer mutex via `HotHead::swap`.
impl HeadStore for HotHead {
    fn snapshot(&self) -> Arc<HeadView> {
        let inner = self.snapshot();
        Arc::new(HeadView {
            head_id: inner.head_id,
            feature_dim: BackboneFeatureDim::default(),
            num_classes: inner.n_classes as u32,
        })
    }

    fn version(&self) -> ResourceVersion {
        self.version()
    }

    /// Override the race-prone default with the atomic `snapshot_with_version`.
    fn snapshot_with_version(&self) -> (Arc<HeadView>, ResourceVersion) {
        let (inner, version) = self.snapshot_with_version();
        let view = Arc::new(HeadView {
            head_id: inner.head_id,
            feature_dim: BackboneFeatureDim::default(),
            num_classes: inner.n_classes as u32,
        });
        (view, version)
    }

    fn try_swap(&self, candidate: HeadCandidate) -> Result<SwapReceipt, HeadStoreError> {
        self.swap(&candidate.head_mpk, &candidate.labels, candidate.head_id)
            .map_err(classify_head_error)
    }

    /// Install a prevalidated [`HeadInner`] for the activation flow (installs only AFTER
    /// `current.json` is durable). The `dyn Any` downcast keeps the `common` trait crate
    /// layered above inference; a type mismatch surfaces as `Unsupported` so caller bugs
    /// don't silently no-op.
    fn install_prevalidated(
        &self,
        candidate: Box<dyn std::any::Any + Send>,
    ) -> Result<SwapReceipt, HeadStoreError> {
        let inner = candidate
            .downcast::<HeadInner>()
            .map_err(|_| HeadStoreError::Unsupported)?;
        // store_inner re-validates: defence in depth against a partially-constructed inner.
        self.store_inner(*inner).map_err(classify_head_error)
    }
}

/// Map `HeadError` to `HeadStoreError`: `NotFound` only for `io NotFound` on read paths,
/// `Internal` for Burn drift, else `InvalidContent` (validation) or `LoadFailed` (I/O).
fn classify_head_error(err: HeadError) -> HeadStoreError {
    use std::io::ErrorKind as Io;
    match err {
        HeadError::ReadHeadMpk { path, source } | HeadError::ReadLabels { path, source }
            if source.kind() == Io::NotFound =>
        {
            HeadStoreError::NotFound { path }
        }
        e @ (HeadError::ReadHeadMpk { .. } | HeadError::ReadLabels { .. }) => {
            HeadStoreError::LoadFailed {
                source: Box::new(e),
            }
        }
        e @ HeadError::TensorIntoVec(_) => HeadStoreError::Internal {
            source: Box::new(e),
        },
        e => HeadStoreError::InvalidContent {
            source: Box::new(e),
        },
    }
}

/// Read `path` under a `cap`-byte ceiling enforced by both a stat precheck and
/// `take(cap + 1)` (closes torn-write races). IO failures map through `mk_io_err`; a cap
/// miss surfaces as `HeaderCorrupt`.
fn read_capped_or_corrupt(
    path: &Path,
    cap: u64,
    kind: &'static str,
    mk_io_err: impl Fn(std::io::Error) -> HeadError,
) -> Result<Vec<u8>, HeadError> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(&mk_io_err)?;
    let metadata = f.metadata().map_err(&mk_io_err)?;
    if metadata.len() > cap {
        return Err(HeadError::HeaderCorrupt {
            path: path.display().to_string(),
            reason: format!(
                "{kind} exceeds {cap}-byte cap (observed {} bytes)",
                metadata.len()
            ),
        });
    }
    // Cap the pre-alloc hint at 64 KiB so a tampered stat can't drive a huge
    // `Vec::with_capacity` and OOM the daemon on a 1.5 GiB SBC; `take(cap+1)` still rejects.
    const INITIAL_CAPACITY_HINT: u64 = 64 * 1024;
    let cap_for_hint = std::cmp::min(metadata.len(), cap).min(INITIAL_CAPACITY_HINT);
    let mut bytes = Vec::with_capacity(cap_for_hint as usize);
    f.take(cap + 1)
        .read_to_end(&mut bytes)
        .map_err(&mk_io_err)?;
    if bytes.len() as u64 > cap {
        return Err(HeadError::HeaderCorrupt {
            path: path.display().to_string(),
            reason: format!(
                "{kind} exceeds {cap}-byte cap (read {} bytes after torn-write)",
                bytes.len()
            ),
        });
    }
    Ok(bytes)
}

fn load_inner(
    head_mpk: &Path,
    labels_path: &Path,
    head_id: HeadId,
) -> Result<HeadInner, HeadError> {
    let device: burn::tensor::Device<B> = Default::default();

    // Cap before slurp (tamper-OOM): worst legitimate weights ~800 MiB
    // (MAX_N_CLASSES * BACKBONE_FEATURE_DIM * 4 B), so 1 GiB gives headroom.
    const MAX_HEAD_MPK_BYTES: u64 = 1024 * 1024 * 1024;
    let bytes = read_capped_or_corrupt(head_mpk, MAX_HEAD_MPK_BYTES, "head .mpk", |e| {
        HeadError::ReadHeadMpk {
            path: head_mpk.display().to_string(),
            source: e,
        }
    })?;
    if bytes.len() < crate::common::head_header::HEAD_HEADER_SIZE {
        // Too short to hold the header = truncated/corrupt -> `HeaderCorrupt`, not
        // `SchemaTooOld`, so the operator gets "restore from backup" not "regenerate".
        return Err(HeadError::HeaderCorrupt {
            path: head_mpk.display().to_string(),
            reason: format!(
                "head .mpk too short: got {} bytes, need >= {}",
                bytes.len(),
                crate::common::head_header::HEAD_HEADER_SIZE
            ),
        });
    }
    let header = crate::common::head_header::parse_header(
        &bytes[..crate::common::head_header::HEAD_HEADER_SIZE],
    )
    .map_err(|e| match e {
        crate::common::head_header::HeadHeaderError::BadMagic { .. } => HeadError::SchemaTooOld {
            path: head_mpk.display().to_string(),
        },
        other => HeadError::HeaderCorrupt {
            path: head_mpk.display().to_string(),
            reason: format!("{other}"),
        },
    })?;
    if header.feature_dim as usize != BACKBONE_FEATURE_DIM {
        return Err(HeadError::HeaderFeatureDimMismatch {
            path: head_mpk.display().to_string(),
            got: header.feature_dim,
            expected: BACKBONE_FEATURE_DIM as u32,
        });
    }
    if bytes.len() - crate::common::head_header::HEAD_HEADER_SIZE != header.payload_len as usize {
        return Err(HeadError::HeaderCorrupt {
            path: head_mpk.display().to_string(),
            reason: format!(
                "header.payload_len={}, file_len-header_size={}",
                header.payload_len,
                bytes.len() - crate::common::head_header::HEAD_HEADER_SIZE,
            ),
        });
    }

    // Burn's recorder must own the payload; `drain` strips the header in place,
    // reusing the allocation (O(payload_len) memmove).
    let mut payload = bytes;
    payload.drain(..crate::common::head_header::HEAD_HEADER_SIZE);

    let head = Head::<B>::load_mpk_bytes(payload, &device).map_err(|e| HeadError::LoadMpk {
        path: head_mpk.display().to_string(),
        message: format!("{e}"),
    })?;
    let weight_dims = head.linear.weight.val().dims();
    if weight_dims[0] != BACKBONE_FEATURE_DIM {
        return Err(HeadError::InputDim {
            got: weight_dims[0],
        });
    }
    let n_classes = weight_dims[1];
    if n_classes == 0 || n_classes > MAX_N_CLASSES {
        return Err(HeadError::BadClassCount {
            got: n_classes,
            max: MAX_N_CLASSES,
        });
    }
    // Cross-check the write-time header `num_classes` against the payload-decoded value;
    // a mismatch means a tampered/swapped payload -- fail closed.
    if header.num_classes as usize != n_classes {
        return Err(HeadError::HeaderCorrupt {
            path: head_mpk.display().to_string(),
            reason: format!(
                "header.num_classes={} disagrees with payload-decoded n_classes={n_classes}",
                header.num_classes,
            ),
        });
    }

    // Hoist out of Burn into flat `Vec<f32>` for the hot path.
    let weight: Vec<f32> = head
        .linear
        .weight
        .val()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| HeadError::TensorIntoVec(format!("{e:?}")))?;
    if weight.len() != BACKBONE_FEATURE_DIM * n_classes {
        return Err(HeadError::WeightShape {
            got: weight.len(),
            expected: BACKBONE_FEATURE_DIM * n_classes,
        });
    }

    let bias: Vec<f32> = match head.linear.bias.as_ref() {
        Some(b) => b
            .val()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| HeadError::TensorIntoVec(format!("{e:?}")))?,
        None => vec![0.0; n_classes],
    };
    // Else a length mismatch panics at the first frame in `head_forward`'s
    // `logits.copy_from_slice(bias)`.
    if bias.len() != n_classes {
        return Err(HeadError::BiasShape {
            got: bias.len(),
            expected: n_classes,
        });
    }

    // Cap before slurp (tamper-OOM): worst legitimate size ~26 MiB
    // (MAX_N_CLASSES * (MAX_LABEL_BYTES + 1)), so 32 MiB gives headroom.
    const MAX_LABELS_FILE_BYTES: u64 = 32 * 1024 * 1024;
    let labels_bytes =
        read_capped_or_corrupt(labels_path, MAX_LABELS_FILE_BYTES, "labels.txt", |e| {
            HeadError::ReadLabels {
                path: labels_path.display().to_string(),
                source: e,
            }
        })?;
    let labels_text = String::from_utf8(labels_bytes).map_err(|e| HeadError::ReadLabels {
        path: labels_path.display().to_string(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
    })?;
    // Per-label cap: each label rides every `InferenceFrame` (~4 Hz), so a multi-MB label
    // could DoS the SSE consumer. Filter the borrowed `&str` first to skip blank-line allocs.
    let labels: Vec<String> = labels_text
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    if let Some(over) = labels.iter().find(|s| s.len() > MAX_LABEL_BYTES) {
        // Walk back to a char boundary so truncating multi-byte UTF-8 (CJK class
        // names) in the diagnostic can't panic the formatter.
        let mut end = over.len().min(64);
        while end > 0 && !over.is_char_boundary(end) {
            end -= 1;
        }
        return Err(HeadError::HeaderCorrupt {
            path: labels_path.display().to_string(),
            reason: format!(
                "label {:?} exceeds {MAX_LABEL_BYTES}-byte cap ({} bytes); trim labels.txt and re-publish",
                &over[..end],
                over.len(),
            ),
        });
    }
    if labels.len() != n_classes {
        return Err(HeadError::LabelCountMismatch {
            path: labels_path.display().to_string(),
            got: labels.len(),
            n_classes,
        });
    }

    let inner = HeadInner {
        weight,
        bias,
        labels,
        head_id,
        n_classes,
    };
    // Shapes enforced inline above; this scan catches NaN/Inf that would otherwise
    // reach `head_forward` -> `softmax_into` and surface as NaN probs.
    inner.validate_finite()?;
    Ok(inner)
}

/// Canonical synthetic test head (0.01 weights, zero bias, `class_{i}` labels);
/// shared with the engine-level stream tests so the suites can't drift.
#[cfg(test)]
pub(crate) fn synth_inner(n_classes: usize, head_id: HeadId) -> HeadInner {
    HeadInner {
        weight: vec![0.01; BACKBONE_FEATURE_DIM * n_classes],
        bias: vec![0.0; n_classes],
        labels: (0..n_classes).map(|i| format!("class_{i}")).collect(),
        head_id,
        n_classes,
    }
}

#[cfg(test)]
mod tests {
    // Fixtures use raw `std::fs::write`; the file_mgr atomic-write discipline is N/A here.
    #![allow(clippy::disallowed_methods)]
    use super::*;

    #[test]
    fn from_inner_round_trip() {
        let id = HeadId::new();
        let h = HotHead::from_inner(synth_inner(3, id));
        let snap = h.snapshot();
        assert_eq!(snap.n_classes, 3);
        assert_eq!(snap.weight.len(), BACKBONE_FEATURE_DIM * 3);
        assert_eq!(snap.labels.len(), 3);
        assert_eq!(snap.head_id, id);
        assert_eq!(h.n_classes(), 3);
    }

    #[test]
    fn store_inner_validates_shape() {
        let id = HeadId::new();
        let h = HotHead::from_inner(synth_inner(3, id));

        let bad_zero = HeadInner {
            weight: vec![],
            bias: vec![],
            labels: vec![],
            head_id: id,
            n_classes: 0,
        };
        let err = h
            .store_inner(bad_zero)
            .expect_err("n_classes=0 must reject");
        assert!(
            matches!(err, HeadError::BadClassCount { got: 0, .. }),
            "expected BadClassCount, got {err:?}",
        );

        // n_classes > MAX is checked before weight shape, so empty weight is fine here.
        let bad_huge = HeadInner {
            weight: vec![],
            bias: vec![],
            labels: vec![],
            head_id: id,
            n_classes: MAX_N_CLASSES + 1,
        };
        let err = h
            .store_inner(bad_huge)
            .expect_err("n_classes > MAX must reject");
        assert!(
            matches!(err, HeadError::BadClassCount { got, max } if got == MAX_N_CLASSES + 1 && max == MAX_N_CLASSES),
            "expected BadClassCount, got {err:?}",
        );

        let bad_weight = HeadInner {
            weight: vec![0.0; 99],
            bias: vec![0.0; 3],
            labels: vec!["a".into(), "b".into(), "c".into()],
            head_id: id,
            n_classes: 3,
        };
        let err = h
            .store_inner(bad_weight)
            .expect_err("weight shape must reject");
        assert!(
            matches!(err, HeadError::WeightShape { got: 99, expected }
                     if expected == 3 * BACKBONE_FEATURE_DIM),
            "expected WeightShape, got {err:?}",
        );

        let bad_bias = HeadInner {
            weight: vec![0.0; 3 * BACKBONE_FEATURE_DIM],
            bias: vec![0.0; 5],
            labels: vec!["a".into(), "b".into(), "c".into()],
            head_id: id,
            n_classes: 3,
        };
        let err = h.store_inner(bad_bias).expect_err("bias shape must reject");
        assert!(
            matches!(
                err,
                HeadError::BiasShape {
                    got: 5,
                    expected: 3
                }
            ),
            "expected BiasShape, got {err:?}",
        );

        let bad_labels = HeadInner {
            weight: vec![0.0; 3 * BACKBONE_FEATURE_DIM],
            bias: vec![0.0; 3],
            labels: vec!["only_one".into()],
            head_id: id,
            n_classes: 3,
        };
        let err = h
            .store_inner(bad_labels)
            .expect_err("label shape must reject");
        assert!(
            matches!(
                err,
                HeadError::LabelShape {
                    got: 1,
                    n_classes: 3
                }
            ),
            "expected LabelShape, got {err:?}",
        );

        let v_before = h.version();
        let receipt = h
            .store_inner(synth_inner(7, HeadId::new()))
            .expect("well-shaped store must succeed");
        assert!(receipt.version > v_before, "version did not advance");
        assert_eq!(h.snapshot().n_classes, 7);
    }

    /// No snapshot tears across a swap-storm: each snapshot's weight/bias/labels align to
    /// one (n_classes, head_id) bundle; the `labels[i]`-tag check fails if a swap publishes
    /// new labels against old weights.
    #[test]
    fn store_inner_no_torn_snapshot() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        // Each label embeds head_id's string form so a labels-vs-head_id tear is visible
        // at read time; n_classes is swept independently of head_id.
        fn tagged(n_classes: usize, head_id: HeadId) -> HeadInner {
            let v = head_id.to_string();
            HeadInner {
                weight: vec![0.01; BACKBONE_FEATURE_DIM * n_classes],
                bias: vec![0.0; n_classes],
                labels: (0..n_classes).map(|i| format!("{v}/cls_{i}")).collect(),
                head_id,
                n_classes,
            }
        }

        // Strict UUID-v4 sentinels: init seeds the head, even/odd alternate the storm.
        const TEST_V4_NIL: &str = "00000000-0000-4000-8000-000000000000";
        const TEST_V4_NIL_1: &str = "00000000-0000-4000-8000-000000000001";
        const TEST_V4_NIL_2: &str = "00000000-0000-4000-8000-000000000002";
        let v_init = HeadId::parse(TEST_V4_NIL).unwrap();
        let v_even = HeadId::parse(TEST_V4_NIL_1).unwrap();
        let v_odd = HeadId::parse(TEST_V4_NIL_2).unwrap();

        let h = HotHead::from_inner(tagged(3, v_init));
        let stop = Arc::new(AtomicBool::new(false));

        // Storm alternates (n=3, even)/(n=5, odd) so the bundle changes every swap,
        // making a torn snapshot statistically observable.
        let h_writer = h.clone();
        let stop_w = stop.clone();
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !stop_w.load(Ordering::Relaxed) {
                let (n, v) = if i.is_multiple_of(2) {
                    (3, v_even)
                } else {
                    (5, v_odd)
                };
                let _ = h_writer
                    .store_inner(tagged(n, v))
                    .expect("tagged() always produces a valid HeadInner");
                i = i.wrapping_add(1);
            }
            i
        });

        let h_reader = h.clone();
        let reader = thread::spawn(move || {
            for _ in 0..10_000 {
                let s = h_reader.snapshot();
                let v = s.head_id.to_string();
                assert_eq!(
                    s.weight.len(),
                    BACKBONE_FEATURE_DIM * s.n_classes,
                    "weight/n_classes torn: v={v} n={}",
                    s.n_classes
                );
                assert_eq!(
                    s.bias.len(),
                    s.n_classes,
                    "bias/n_classes torn: v={v} n={}",
                    s.n_classes
                );
                assert_eq!(
                    s.labels.len(),
                    s.n_classes,
                    "labels/n_classes torn: v={v} n={}",
                    s.n_classes
                );
                for (i, lbl) in s.labels.iter().enumerate() {
                    assert!(
                        lbl.starts_with(&format!("{v}/")),
                        "labels/head_id torn: head_id={v}, labels[{i}]={lbl}",
                    );
                }
            }
        });

        reader.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        let swaps = writer.join().unwrap();
        // Reject a degenerate run where the writer never swapped.
        assert!(
            swaps > 0,
            "writer thread completed 0 swaps; test is degenerate"
        );
    }

    #[test]
    fn load_rejects_headerless_mpk() {
        use burn::backend::NdArray;
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        let dir = tempfile::tempdir().expect("tempdir");
        let mpk_stem = dir.path().join("test_head");
        let mpk_path = dir.path().join("test_head.mpk");
        let labels_path = dir.path().join("test_labels.txt");
        let device: burn::tensor::Device<NdArray<f32>> = Default::default();
        let head = crate::model::Head::<NdArray<f32>>::new(2, &device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(head.into_record(), mpk_stem)
            .expect("record head");
        // Headerless: recorder raw output, no ACSTHEAD prepend.
        std::fs::write(&labels_path, "alpha\nbeta\n").expect("write labels");
        let err = HotHead::load(&mpk_path, &labels_path, HeadId::new())
            .expect_err("headerless mpk must reject");
        assert!(
            matches!(err, HeadError::SchemaTooOld { .. }),
            "expected SchemaTooOld, got {err:?}",
        );
    }

    #[test]
    fn load_round_trips_through_header_prepended_mpk() {
        use burn::backend::NdArray;
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        let dir = tempfile::tempdir().expect("tempdir");
        let raw_stem = dir.path().join("test_head_raw");
        let raw_mpk = dir.path().join("test_head_raw.mpk");
        let mpk_path = dir.path().join("test_head.mpk");
        let labels_path = dir.path().join("test_labels.txt");
        let device: burn::tensor::Device<NdArray<f32>> = Default::default();
        let head = crate::model::Head::<NdArray<f32>>::new(2, &device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(head.into_record(), raw_stem)
            .expect("record head");
        let payload = std::fs::read(&raw_mpk).expect("read raw mpk");
        let mut composed = std::fs::File::create(&mpk_path).expect("create mpk");
        crate::common::head_header::write_with_payload(
            &mut composed,
            BACKBONE_FEATURE_DIM as u32,
            2,
            &payload,
        )
        .expect("write head with header");
        drop(composed);
        std::fs::write(&labels_path, "alpha\nbeta\n").expect("write labels");

        let h = HotHead::load(&mpk_path, &labels_path, HeadId::new())
            .expect("load header-prepended mpk");
        let snap = h.snapshot();
        assert_eq!(snap.n_classes, 2);
        assert_eq!(snap.labels, vec!["alpha".to_string(), "beta".to_string()]);
    }

    /// `load_inner` consumes the payload in memory: 100 swaps leave no `.head-load-*` sibling.
    #[test]
    fn load_inner_does_not_create_sibling_tempfile() {
        use burn::backend::NdArray;
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        let dir = tempfile::tempdir().expect("tempdir");
        let raw_stem = dir.path().join("test_head_raw");
        let raw_mpk = dir.path().join("test_head_raw.mpk");
        let mpk_path = dir.path().join("test_head.mpk");
        let labels_path = dir.path().join("test_labels.txt");
        let device: burn::tensor::Device<NdArray<f32>> = Default::default();
        let head = crate::model::Head::<NdArray<f32>>::new(2, &device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(head.into_record(), raw_stem)
            .expect("record head");
        let payload = std::fs::read(&raw_mpk).expect("read raw mpk");
        let mut composed = std::fs::File::create(&mpk_path).expect("create mpk");
        crate::common::head_header::write_with_payload(
            &mut composed,
            BACKBONE_FEATURE_DIM as u32,
            2,
            &payload,
        )
        .expect("write head with header");
        drop(composed);
        std::fs::write(&labels_path, "alpha\nbeta\n").expect("write labels");

        let count_sibling_tempfiles = || -> usize {
            std::fs::read_dir(dir.path())
                .expect("read tempdir")
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name();
                    let s = name.to_string_lossy();
                    s.starts_with(".head-load-")
                })
                .count()
        };
        assert_eq!(count_sibling_tempfiles(), 0, "tempdir starts clean");

        let h = HotHead::load(&mpk_path, &labels_path, HeadId::new()).expect("first load");
        for i in 0..100 {
            let new_id = HeadId::new();
            let receipt = h
                .swap(&mpk_path, &labels_path, new_id)
                .unwrap_or_else(|e| panic!("swap #{i} failed: {e:?}"));
            assert!(receipt.version.get() > 0, "receipt version is monotonic");
            assert_eq!(
                count_sibling_tempfiles(),
                0,
                "swap #{i} left a `.head-load-*` artifact in the workspace dir",
            );
        }
        assert_eq!(
            count_sibling_tempfiles(),
            0,
            "100 swaps left a `.head-load-*` artifact in the workspace dir",
        );
    }

    /// A flipped `feature_dim` byte trips the CRC -> `HeaderCorrupt`, not `SchemaTooOld`
    /// (magic intact).
    #[test]
    fn load_rejects_corrupt_header_via_crc() {
        use burn::backend::NdArray;
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
        let dir = tempfile::tempdir().expect("tempdir");
        let raw_stem = dir.path().join("test_head_raw");
        let raw_mpk = dir.path().join("test_head_raw.mpk");
        let mpk_path = dir.path().join("test_head.mpk");
        let labels_path = dir.path().join("test_labels.txt");
        let device: burn::tensor::Device<NdArray<f32>> = Default::default();
        let head = crate::model::Head::<NdArray<f32>>::new(2, &device);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .record(head.into_record(), raw_stem)
            .expect("record");
        let payload = std::fs::read(&raw_mpk).expect("read raw mpk");
        let mut bytes: Vec<u8> = Vec::new();
        let mut composed = std::io::Cursor::new(&mut bytes);
        crate::common::head_header::write_with_payload(
            &mut composed,
            BACKBONE_FEATURE_DIM as u32,
            2,
            &payload,
        )
        .expect("write");
        // Flip a feature_dim byte without recomputing the CRC.
        bytes[12] ^= 0xFF;
        std::fs::write(&mpk_path, &bytes).expect("write tampered");
        std::fs::write(&labels_path, "alpha\nbeta\n").expect("write labels");
        let err = HotHead::load(&mpk_path, &labels_path, HeadId::new())
            .expect_err("corrupt header must reject");
        assert!(
            matches!(err, HeadError::HeaderCorrupt { .. }),
            "expected HeaderCorrupt, got {err:?}",
        );
    }

    /// Bundled default head loads with class count matching the shipped `labels.txt`.
    #[test]
    #[ignore = "depends on bundled fixture assets; --include-ignored"]
    fn reference_head_loads() {
        let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
        let head_mpk = crate_root.join("misc/heads/default/head.mpk");
        let labels = crate_root.join("misc/heads/default/labels.txt");
        let h = HotHead::load(&head_mpk, &labels, HeadId::new()).expect("load reference head");
        let snap = h.snapshot();
        assert!(snap.n_classes >= 1);
        assert_eq!(snap.weight.len(), BACKBONE_FEATURE_DIM * snap.n_classes);
        assert_eq!(snap.bias.len(), snap.n_classes);
        assert_eq!(snap.labels.len(), snap.n_classes);
    }

    #[test]
    fn try_from_inner_rejects_non_finite_weight() {
        let id = HeadId::new();
        let mut inner = synth_inner(3, id);
        inner.weight[7] = f32::NAN;
        let err = HotHead::try_from_inner(inner).expect_err("NaN weight must reject");
        assert!(
            matches!(err, HeadError::NonFiniteWeight { idx: 7, .. }),
            "expected NonFiniteWeight, got {err:?}",
        );

        let id = HeadId::new();
        let mut inner = synth_inner(3, id);
        inner.weight[2] = f32::INFINITY;
        let err = HotHead::try_from_inner(inner).expect_err("+Inf weight must reject");
        assert!(
            matches!(err, HeadError::NonFiniteWeight { idx: 2, .. }),
            "expected NonFiniteWeight, got {err:?}",
        );
    }

    #[test]
    fn try_from_inner_rejects_non_finite_bias() {
        let id = HeadId::new();
        let mut inner = synth_inner(3, id);
        inner.bias[1] = f32::NAN;
        let err = HotHead::try_from_inner(inner).expect_err("NaN bias must reject");
        assert!(
            matches!(err, HeadError::NonFiniteBias { idx: 1, .. }),
            "expected NonFiniteBias, got {err:?}",
        );

        let id = HeadId::new();
        let mut inner = synth_inner(3, id);
        inner.bias[0] = f32::NEG_INFINITY;
        let err = HotHead::try_from_inner(inner).expect_err("-Inf bias must reject");
        assert!(
            matches!(err, HeadError::NonFiniteBias { idx: 0, .. }),
            "expected NonFiniteBias, got {err:?}",
        );
    }

    /// `try_from_inner` accepts a well-formed inner; guards against a false-positive reject.
    #[test]
    fn try_from_inner_accepts_well_shaped() {
        let id = HeadId::new();
        let h = HotHead::try_from_inner(synth_inner(5, id)).expect("well-shaped accepts");
        let snap = h.snapshot();
        assert_eq!(snap.n_classes, 5);
        assert_eq!(snap.head_id, id);
    }

    #[test]
    fn store_inner_rejects_non_finite_weight() {
        let id = HeadId::new();
        let h = HotHead::from_inner(synth_inner(3, id));
        let mut bad = synth_inner(3, HeadId::new());
        bad.weight[0] = f32::NAN;
        let err = h.store_inner(bad).expect_err("NaN weight must reject");
        assert!(
            matches!(err, HeadError::NonFiniteWeight { idx: 0, .. }),
            "expected NonFiniteWeight, got {err:?}",
        );
    }

    /// Each category routes distinctly; a regression would silently re-collapse every
    /// failure to a 400.
    #[test]
    fn classify_head_error_maps_each_category() {
        let nf_mpk = HeadError::ReadHeadMpk {
            path: "/missing.mpk".into(),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        match classify_head_error(nf_mpk) {
            HeadStoreError::NotFound { path } => assert_eq!(path, "/missing.mpk"),
            other => panic!("expected NotFound, got {other:?}"),
        }
        let nf_lbl = HeadError::ReadLabels {
            path: "/missing.txt".into(),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        match classify_head_error(nf_lbl) {
            HeadStoreError::NotFound { path } => assert_eq!(path, "/missing.txt"),
            other => panic!("expected NotFound, got {other:?}"),
        }

        let denied = HeadError::ReadHeadMpk {
            path: "/locked.mpk".into(),
            source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        };
        assert!(
            matches!(
                classify_head_error(denied),
                HeadStoreError::LoadFailed { .. }
            ),
            "non-NotFound I/O must route to LoadFailed",
        );

        let drift = HeadError::TensorIntoVec("shape mismatch".into());
        assert!(
            matches!(classify_head_error(drift), HeadStoreError::Internal { .. }),
            "TensorIntoVec must route to Internal",
        );

        // Header / shape / finiteness failures all -> InvalidContent.
        let representative = [
            HeadError::SchemaTooOld {
                path: "/old.mpk".into(),
            },
            HeadError::HeaderCorrupt {
                path: "/c.mpk".into(),
                reason: "bad crc".into(),
            },
            HeadError::HeaderFeatureDimMismatch {
                path: "/d.mpk".into(),
                got: 7,
                expected: BACKBONE_FEATURE_DIM as u32,
            },
            HeadError::WeightShape {
                got: 1,
                expected: 2,
            },
            HeadError::BiasShape {
                got: 1,
                expected: 2,
            },
            HeadError::InputDim { got: 7 },
            HeadError::BadClassCount { got: 0, max: 100 },
            HeadError::LabelCountMismatch {
                path: "/l.txt".into(),
                got: 1,
                n_classes: 2,
            },
            HeadError::LabelShape {
                got: 1,
                n_classes: 2,
            },
            HeadError::NonFiniteWeight {
                idx: 0,
                value: f32::NAN,
            },
            HeadError::NonFiniteBias {
                idx: 0,
                value: f32::NAN,
            },
            HeadError::LoadMpk {
                path: "/x.mpk".into(),
                message: "burn parse failed".into(),
            },
        ];
        for err in representative {
            let formatted = format!("{err}");
            let mapped = classify_head_error(err);
            assert!(
                matches!(mapped, HeadStoreError::InvalidContent { .. }),
                "{formatted} must classify as InvalidContent, got {mapped:?}",
            );
        }
    }
}
