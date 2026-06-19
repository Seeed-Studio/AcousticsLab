//! Burn fp32 acoustic-model definitions: 4-conv backbone (`[N,1,43,232]` NCHW
//! in -> 2000-dim features) + `Linear` head emitting pre-softmax logits, plus
//! the network's Burn `.mpk` mapping.

use crate::common::dims::BackboneFeatureDim;
use crate::common::head_header::write_with_payload;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, NamedMpkFileRecorder, Recorder};
use std::io::Write;
use std::path::Path;
use thiserror::Error;

/// Errors from the Burn `.mpk` mapping helpers. Recorder errors aren't always
/// `'static + Send + Sync`, so they are stringified rather than `#[source]`d.
#[derive(Debug, Error)]
pub enum Error {
    #[error("load .mpk {path}: {message}")]
    Load { path: String, message: String },
    #[error("save .mpk {path}: {message}")]
    Save { path: String, message: String },
    #[error("invalid head: n_classes = {got} (must be in 1..={max})")]
    BadClassCount { got: usize, max: usize },
}

fn load_err(path: impl std::fmt::Display, message: impl Into<String>) -> Error {
    Error::Load {
        path: path.to_string(),
        message: message.into(),
    }
}

fn save_err(path: impl std::fmt::Display, message: impl Into<String>) -> Error {
    Error::Save {
        path: path.to_string(),
        message: message.into(),
    }
}

/// Re-exported so [`Head::try_new`] enforces the same class ceiling as the
/// inference cold-path validator.
pub use crate::common::dims::MAX_N_CLASSES;

/// Single [`FullPrecisionSettings`] pin so file and in-memory paths can't drift
/// to different precision.
#[inline]
fn recorder() -> NamedMpkFileRecorder<FullPrecisionSettings> {
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
}

#[inline]
fn bytes_recorder() -> NamedMpkBytesRecorder<FullPrecisionSettings> {
    NamedMpkBytesRecorder::<FullPrecisionSettings>::new()
}

/// Frozen embedding backbone: 4 conv2d+maxpool stages then a 2000-dim ReLU dense
/// projection, emitting the feature vector consumed by [`Head`].
#[derive(Module, Debug)]
pub struct Backbone<B: Backend> {
    pub conv1: Conv2d<B>,
    pub pool1: MaxPool2d,
    pub conv2: Conv2d<B>,
    pub pool2: MaxPool2d,
    pub conv3: Conv2d<B>,
    pub pool3: MaxPool2d,
    pub conv4: Conv2d<B>,
    pub pool4: MaxPool2d,
    pub dense1: Linear<B>,
    pub relu: Relu,
}

impl<B: Backend> Backbone<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([1, 8], [2, 8]).init(device),
            pool1: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv2: Conv2dConfig::new([8, 32], [2, 4]).init(device),
            pool2: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv3: Conv2dConfig::new([32, 32], [2, 4]).init(device),
            pool3: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            conv4: Conv2dConfig::new([32, 32], [2, 4]).init(device),
            pool4: MaxPool2dConfig::new([2, 2]).with_strides([1, 2]).init(),
            dense1: LinearConfig::new(704, BackboneFeatureDim::USIZE).init(device),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.relu.forward(self.conv1.forward(x));
        let x = self.pool1.forward(x);
        let x = self.relu.forward(self.conv2.forward(x));
        let x = self.pool2.forward(x);
        let x = self.relu.forward(self.conv3.forward(x));
        let x = self.pool3.forward(x);
        let x = self.relu.forward(self.conv4.forward(x));
        let x = self.pool4.forward(x);
        // Keras channels-last flatten: NCHW -> NHWC -> [N, 704].
        let [n, c, h, w] = x.dims();
        debug_assert_eq!([c, h, w], [32, 2, 11]);
        let x = x.permute([0, 2, 3, 1]).reshape([n, h * w * c]);
        self.relu.forward(self.dense1.forward(x))
    }

    /// Blocking file I/O: call from [`tokio::task::spawn_blocking`].
    pub fn load_mpk(path: &Path, device: &B::Device) -> Result<Self, Error> {
        let record: BackboneRecord<B> = recorder()
            .load(path.to_path_buf(), device)
            .map_err(|e| load_err(path.display(), format!("{e}")))?;
        Ok(Self::new(device).load_record(record))
    }
}

/// Hot-swappable classifier head: one `Linear[BackboneFeatureDim, n_classes]`
/// producing pre-softmax logits.
#[derive(Module, Debug)]
pub struct Head<B: Backend> {
    pub linear: Linear<B>,
}

impl<B: Backend> Head<B> {
    /// Trusted-input constructor: panics inside `LinearConfig::init` on
    /// `n_classes == 0`; use [`Self::try_new`] for untrusted/config inputs.
    pub fn new(n_classes: usize, device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(BackboneFeatureDim::USIZE, n_classes).init(device),
        }
    }

    /// Rejects out-of-range `n_classes` before it reaches Burn's allocator.
    pub fn try_new(n_classes: usize, device: &B::Device) -> Result<Self, Error> {
        if n_classes == 0 || n_classes > MAX_N_CLASSES {
            return Err(Error::BadClassCount {
                got: n_classes,
                max: MAX_N_CLASSES,
            });
        }
        Ok(Self::new(n_classes, device))
    }

    /// Pre-softmax logits; softmax applied externally.
    pub fn forward(&self, feat: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(feat)
    }

    /// `load_record` swaps the whole `Linear`, so the file's recorded `n_classes`
    /// overrides the placeholder shape; callers wanting a specific count must check
    /// `head.linear.weight.val().dims()[1]` themselves. Blocking I/O: call from
    /// [`tokio::task::spawn_blocking`].
    pub fn load_mpk(path: &Path, device: &B::Device) -> Result<Self, Error> {
        let record: HeadRecord<B> = recorder()
            .load(path.to_path_buf(), device)
            .map_err(|e| load_err(path.display(), format!("{e}")))?;
        Ok(Self::new(1, device).load_record(record))
    }

    /// In-memory [`Self::load_mpk`]; `payload` is consumed (Burn may write through the
    /// buffer during decode), clone first to keep the original.
    pub fn load_mpk_bytes(payload: Vec<u8>, device: &B::Device) -> Result<Self, Error> {
        let recorder = bytes_recorder();
        let record: HeadRecord<B> = recorder
            .load(payload, device)
            .map_err(|e| load_err("<bytes>", format!("{e}")))?;
        Ok(Self::new(1, device).load_record(record))
    }

    /// Persist head weights as RAW Burn `.mpk` (no `ACSTHEAD` wrapper), read back
    /// only via [`Self::load_mpk`]; published artifacts use [`Self::save_mpk_atomic`].
    /// Atomic via [`write_blob_atomically`] so a mid-write failure on a later epoch
    /// leaves `path` UNCHANGED, preserving the prior good snapshot (a truncating
    /// `File::create` would destroy it). Blocking I/O + fsyncs: call from
    /// [`tokio::task::spawn_blocking`].
    pub(crate) fn save_mpk(self, path: &Path) -> Result<(), Error> {
        let payload = bytes_recorder()
            .record(self.into_record(), ())
            .map_err(|e| save_err(path.display(), format!("{e}")))?;
        write_blob_atomically(path, &payload)
    }

    /// Persist head weights to `final_path` as the `ACSTHEAD`-wrapped Burn `.mpk`,
    /// byte-for-byte the format `inference::head::load_inner` reads. Consumes `self`.
    /// On-disk: 32-byte header (magic + feature_dim + n_classes + payload_len + CRC32)
    /// then the `NamedMpkBytesRecorder` payload. Atomic (tempfile -> fsync -> rename
    /// -> fsync parent): leaves `final_path` unchanged on any failure. Blocking I/O +
    /// fsyncs: call from [`tokio::task::spawn_blocking`].
    pub fn save_mpk_atomic(self, final_path: &Path) -> Result<(), Error> {
        // n_classes from the live tensor before into_record() consumes self; matches
        // converter's header so training- and converter-published heads are byte-identical.
        let n_classes = self.linear.weight.val().dims()[1] as u32;
        let recorder = bytes_recorder();
        let payload = recorder
            .record(self.into_record(), ())
            .map_err(|e| save_err(final_path.display(), format!("{e}")))?;
        let mut header_blob: Vec<u8> =
            Vec::with_capacity(crate::common::head_header::HEAD_HEADER_SIZE + payload.len());
        write_with_payload(
            &mut header_blob,
            BackboneFeatureDim::USIZE as u32,
            n_classes,
            &payload,
        )
        .map_err(|e| save_err(final_path.display(), format!("compose ACSTHEAD: {e}")))?;
        write_blob_atomically(final_path, &header_blob)
    }
}

/// Atomic write: tempfile -> fsync(file) -> rename -> fsync(parent); leaves `path`
/// UNCHANGED on any failure. Inlined (not `file_mgr::fs_atomic::put_atomic`) because
/// the dep-edge guard forbids `model -> file_mgr`.
fn write_blob_atomically(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let parent = path
        .parent()
        .ok_or_else(|| save_err(path.display(), "atomic write: path has no parent directory"))?;
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .map_err(|e| save_err(parent.display(), format!("create tempfile: {e}")))?;
    tmp.write_all(bytes)
        .map_err(|e| save_err(path.display(), format!("write tempfile: {e}")))?;
    tmp.flush()
        .map_err(|e| save_err(path.display(), format!("flush tempfile: {e}")))?;
    // Durability barrier: else rename can become visible before data hits stable storage.
    tmp.as_file()
        .sync_all()
        .map_err(|e| save_err(path.display(), format!("fsync tempfile: {e}")))?;
    tmp.persist(path)
        .map_err(|e| save_err(path.display(), format!("persist (rename): {e}")))?;
    // fsync parent so the rename's directory entry survives a power loss after `persist`.
    std::fs::File::open(parent)
        .and_then(|f| f.sync_all())
        .map_err(|e| save_err(parent.display(), format!("fsync parent dir: {e}")))?;
    Ok(())
}

/// Composed [`Backbone`] + [`Head`] for training and the parity reference; the
/// streaming inference engine drives the two stages independently.
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub backbone: Backbone<B>,
    pub head: Head<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(n_classes: usize, device: &B::Device) -> Self {
        Self {
            backbone: Backbone::new(device),
            head: Head::new(n_classes, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        self.head.forward(self.backbone.forward(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestB = NdArray<f32>;

    #[test]
    fn forward_shape_smoke() {
        let device: burn::tensor::Device<TestB> = Default::default();
        let model = Model::<TestB>::new(3, &device);
        let x = Tensor::<TestB, 4>::zeros([2, 1, 43, 232], &device);
        let y = model.forward(x);
        assert_eq!(y.dims(), [2, 3]);
    }

    /// `save_mpk_atomic` writes an ACSTHEAD header + payload that round-trips
    /// through `load_mpk_bytes` to the original weights.
    #[test]
    fn head_save_atomic_round_trip() {
        use crate::common::head_header::{HEAD_HEADER_SIZE, parse_header};

        let device: burn::tensor::Device<TestB> = Default::default();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rt_head.mpk");

        // Non-default n_classes so load can't trivially "succeed" via the n=1 placeholder.
        const N: usize = 7;
        let saved = Head::<TestB>::new(N, &device);
        let saved_w = saved.linear.weight.val();
        let saved_dims = saved_w.dims();
        let saved_data: Vec<f32> = saved_w.into_data().to_vec().expect("to_vec");
        saved.save_mpk_atomic(&path).expect("save_mpk_atomic");

        let bytes = std::fs::read(&path).expect("read saved");
        assert!(
            bytes.len() >= HEAD_HEADER_SIZE,
            "file too short for header: {} bytes",
            bytes.len()
        );
        let header = parse_header(&bytes[..HEAD_HEADER_SIZE]).expect("parse header");
        assert_eq!(
            header.feature_dim as usize,
            BackboneFeatureDim::USIZE,
            "header feature_dim mismatch",
        );
        assert_eq!(
            header.num_classes as usize, N,
            "header num_classes must self-describe",
        );
        assert_eq!(
            bytes.len() - HEAD_HEADER_SIZE,
            header.payload_len as usize,
            "header.payload_len disagrees with file tail length",
        );

        let payload = bytes[HEAD_HEADER_SIZE..].to_vec();
        let loaded = Head::<TestB>::load_mpk_bytes(payload, &device).expect("load_mpk_bytes");
        let loaded_w = loaded.linear.weight.val();
        let loaded_dims = loaded_w.dims();
        let loaded_data: Vec<f32> = loaded_w.into_data().to_vec().expect("to_vec");

        assert_eq!(saved_dims, loaded_dims, "shape drift across round-trip");
        assert_eq!(
            loaded_dims,
            [BackboneFeatureDim::USIZE, N],
            "loaded n_classes != saved n_classes"
        );
        assert_eq!(
            saved_data.len(),
            loaded_data.len(),
            "weight buffer length drift",
        );
        for (i, (a, b)) in saved_data.iter().zip(loaded_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "weight drift at idx {i}: saved={a}, loaded={b}",
            );
        }
    }

    /// `try_new` rejects `n=0` and `n > MAX_N_CLASSES`; an in-bounds n succeeds.
    #[test]
    fn head_try_new_rejects_pathological_class_counts() {
        let device: burn::tensor::Device<TestB> = Default::default();

        let err_zero = Head::<TestB>::try_new(0, &device).expect_err("n=0 must reject");
        assert!(
            matches!(
                err_zero,
                Error::BadClassCount {
                    got: 0,
                    max: MAX_N_CLASSES
                }
            ),
            "expected BadClassCount, got {err_zero:?}",
        );

        let err_huge =
            Head::<TestB>::try_new(MAX_N_CLASSES + 1, &device).expect_err("n>MAX must reject");
        assert!(
            matches!(
                err_huge,
                Error::BadClassCount { got, max }
                    if got == MAX_N_CLASSES + 1 && max == MAX_N_CLASSES
            ),
            "expected BadClassCount, got {err_huge:?}",
        );

        let h = Head::<TestB>::try_new(3, &device).expect("n=3 must accept");
        let dims = h.linear.weight.val().dims();
        assert_eq!(dims, [BackboneFeatureDim::USIZE, 3]);
    }

    /// `save_mpk` writes RAW Burn bytes (no ACSTHEAD prefix) that round-trip through
    /// `load_mpk`. Delegating to `save_mpk_atomic` would corrupt every best-epoch
    /// snapshot, which is reloaded via `load_mpk`.
    #[test]
    fn head_save_mpk_writes_raw_payload_no_header() {
        use crate::common::head_header::HEAD_MAGIC;

        let device: burn::tensor::Device<TestB> = Default::default();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("snapshot.mpk");

        const N: usize = 5;
        let saved = Head::<TestB>::new(N, &device);
        let saved_w = saved.linear.weight.val();
        let saved_data: Vec<f32> = saved_w.into_data().to_vec().expect("to_vec");
        // The assert_eq! round-trip below breaks if Burn init ever emits NaN (NaN != NaN),
        // so fail here as a clear init regression rather than a spurious MPK mismatch.
        assert!(
            saved_data.iter().all(|v| v.is_finite()),
            "Head::new must produce finite Xavier weights for bitwise round-trip equality",
        );
        saved.save_mpk(&path).expect("save_mpk");

        let bytes = std::fs::read(&path).expect("read saved");
        assert!(
            bytes.len() >= HEAD_MAGIC.len(),
            "file too short to compare against magic: {} bytes",
            bytes.len(),
        );
        assert_ne!(
            &bytes[..HEAD_MAGIC.len()],
            HEAD_MAGIC,
            "save_mpk MUST NOT prepend the ACSTHEAD header -- best-epoch \
             snapshots are reloaded through Head::load_mpk which expects \
             raw Burn bytes, not the ACSTHEAD-wrapped form save_mpk_atomic \
             produces",
        );

        let loaded = Head::<TestB>::load_mpk(&path, &device).expect("load_mpk");
        let loaded_data: Vec<f32> = loaded
            .linear
            .weight
            .val()
            .into_data()
            .to_vec()
            .expect("to_vec");
        assert_eq!(saved_data, loaded_data, "save_mpk -> load_mpk round-trip");
    }

    /// `load_mpk` on a missing path returns [`Error::Load`] (naming the path), not a panic.
    #[test]
    fn head_load_mpk_missing_file_returns_err() {
        let device: burn::tensor::Device<TestB> = Default::default();
        let bad = std::path::Path::new("/nonexistent/.acousticslab/missing-head.mpk");
        let err = Head::<TestB>::load_mpk(bad, &device).expect_err("must fail");
        match err {
            Error::Load { path, .. } => {
                assert!(
                    path.contains("missing-head"),
                    "diagnostic should name the missing path: {path}",
                );
            }
            Error::Save { .. } => panic!("got Save variant on a load error"),
            Error::BadClassCount { .. } => panic!("got BadClassCount variant on a load error"),
        }
    }
}
