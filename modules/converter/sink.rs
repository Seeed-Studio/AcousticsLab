//! `ArtifactSink` trait + production [`MpkSink`] emitting Burn `head.mpk`,
//! `labels.txt`, and `metadata.json`, each built in memory then written via
//! one [`crate::file_mgr::FsService::put_atomic`].

use crate::converter::source::LoadedSource;
use crate::converter::{ConvertError, HeadArtifacts};
use crate::file_mgr::FsService;
use std::path::Path;
use std::sync::Arc;

/// Publisher of converted head artifacts; owns the on-disk layout decision.
pub trait ArtifactSink: Send + Sync + std::fmt::Debug {
    /// Write all artifacts under `dst_dir` via `fs.put_atomic`, with
    /// `metadata.json` LAST so its presence is the consistency marker: a
    /// crash before it leaves a workspace loaders treat as not-yet-converted.
    fn publish(
        &self,
        loaded: &LoadedSource,
        labels: &[String],
        dst_dir: &Path,
        source_kind: crate::converter::SourceKind,
        fs: &Arc<dyn FsService>,
    ) -> Result<HeadArtifacts, ConvertError>;
}

/// Sink delegating to [`crate::converter::write_head_artifacts`].
#[derive(Clone, Copy, Debug, Default)]
pub struct MpkSink;

#[cfg(test)]
const _: fn() = || {
    fn assert_obj_safe<T: ?Sized>() {}
    assert_obj_safe::<dyn ArtifactSink>();
    assert_obj_safe::<dyn crate::converter::source::SourceModel>();
};

impl ArtifactSink for MpkSink {
    fn publish(
        &self,
        loaded: &LoadedSource,
        labels: &[String],
        dst_dir: &Path,
        source_kind: crate::converter::SourceKind,
        fs: &Arc<dyn FsService>,
    ) -> Result<HeadArtifacts, ConvertError> {
        crate::converter::write_head_artifacts(
            &loaded.weights,
            labels,
            dst_dir,
            source_kind,
            loaded.source_sha256.clone(),
            fs.as_ref(),
        )
    }
}
