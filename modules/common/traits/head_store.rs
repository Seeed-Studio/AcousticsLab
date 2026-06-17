//! Narrow trait letting `api` reach the active head without depending on
//! `inference`; weights/labels stay in `inference::HeadInner` (prod impl
//! `inference::HotHead`, an `ArcSwap`-backed
//! [`crate::common::version::VersionedSwap`]).

use crate::common::dims::BackboneFeatureDim;
use crate::common::ids::HeadId;
use crate::common::version::{ResourceVersion, SwapReceipt};
use std::path::PathBuf;
use std::sync::Arc;

/// Read-shape view of the active head; source path is omitted (lives in
/// `config` as `head_active.head_mpk`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeadView {
    pub head_id: HeadId,
    pub feature_dim: BackboneFeatureDim,
    pub num_classes: u32,
}

/// Write-shape for [`HeadStore::try_swap`]: carries paths (not weights) so the
/// impl runs the I/O. `head_id` is stamped on every emitted
/// `InferenceFrame.head_id`.
#[derive(Clone, Debug)]
pub struct HeadCandidate {
    pub head_mpk: PathBuf,
    pub labels: PathBuf,
    pub head_id: HeadId,
}

/// [`HeadStore::try_swap`] errors, categorized so the API maps each to an HTTP
/// status without re-parsing the typed source.
#[derive(Debug, thiserror::Error)]
pub enum HeadStoreError {
    #[error("head not found: {path}")]
    NotFound { path: String },
    /// Bytes read but failed validation: magic/CRC/`feature_dim` mismatch, schema
    /// too old, shape mismatch, or non-finite tensor entries.
    #[error("head content invalid: {source}")]
    InvalidContent {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Non-`NotFound` I/O failure on an operator-supplied path; retryable.
    #[error("head load failed: {source}")]
    LoadFailed {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("head store internal: {source}")]
    Internal {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Operation not exposed by this impl (default
    /// [`HeadStore::install_prevalidated`]).
    #[error("head store: operation not supported")]
    Unsupported,
}

impl crate::common::error::Categorized for HeadStoreError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            HeadStoreError::NotFound { .. } => NotFound,
            // Caller-supplied input: route returns 400.
            HeadStoreError::InvalidContent { .. } | HeadStoreError::LoadFailed { .. } => UserInput,
            HeadStoreError::Internal { .. } | HeadStoreError::Unsupported => Internal,
        }
    }
}

/// Read + swap surface for the active classifier head. `Send + Sync + 'static`
/// so `Arc<dyn HeadStore>` can sit on `api::AppState` and flow through axum
/// handlers; all methods `&self`, mutation confined to [`Self::try_swap`] and
/// serialised by [`crate::common::version::VersionedSwap`]'s writer mutex.
pub trait HeadStore: Send + Sync + 'static {
    /// Wait-free read of the current head view.
    fn snapshot(&self) -> Arc<HeadView>;

    /// Current version; drives `?min_version=N` read-your-write filtering.
    fn version(&self) -> ResourceVersion;

    /// Atomic `(snapshot, version)`. The default reads sequentially, so a swap
    /// between the two reads can return an inconsistent pair; `VersionedSwap`
    /// impls should override with `load_full`.
    fn snapshot_with_version(&self) -> (Arc<HeadView>, ResourceVersion) {
        (self.snapshot(), self.version())
    }

    /// Atomic load-and-install of a new head; returns the post-mutation
    /// [`SwapReceipt`] for read-your-write echo. Blocking (~5 ms `.mpk` parse +
    /// label read); async callers should `tokio::task::spawn_blocking`.
    fn try_swap(&self, candidate: HeadCandidate) -> Result<SwapReceipt, HeadStoreError>;

    /// Install a prevalidated candidate (durable-activation flow installs into
    /// `HotHead` only after `current.json` is durable). `Box<dyn Any>` carries
    /// impl-specific state (prod: `inference::HeadInner`), sidestepping the layer
    /// cycle since this trait lives in `common` while `HeadInner` is downstream;
    /// default and type-mismatch both yield [`HeadStoreError::Unsupported`].
    fn install_prevalidated(
        &self,
        _candidate: Box<dyn std::any::Any + Send>,
    ) -> Result<SwapReceipt, HeadStoreError> {
        Err(HeadStoreError::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::error::{Categorized, ErrorKind};

    #[derive(Debug, thiserror::Error)]
    #[error("synthetic")]
    struct Synthetic;

    #[test]
    fn categorized_maps_each_variant_to_intended_http_class() {
        let cases: [(HeadStoreError, ErrorKind, u16); 5] = [
            (
                HeadStoreError::NotFound {
                    path: "/missing".into(),
                },
                ErrorKind::NotFound,
                404,
            ),
            (
                HeadStoreError::InvalidContent {
                    source: Box::new(Synthetic),
                },
                ErrorKind::UserInput,
                400,
            ),
            (
                HeadStoreError::LoadFailed {
                    source: Box::new(Synthetic),
                },
                ErrorKind::UserInput,
                400,
            ),
            (
                HeadStoreError::Internal {
                    source: Box::new(Synthetic),
                },
                ErrorKind::Internal,
                500,
            ),
            (HeadStoreError::Unsupported, ErrorKind::Internal, 500),
        ];
        for (err, expected_kind, expected_status) in cases {
            assert_eq!(err.kind(), expected_kind, "variant kind mismatch for {err}",);
            assert_eq!(
                err.kind().http_status_code(),
                expected_status,
                "variant http status mismatch for {err}",
            );
        }
    }
}
