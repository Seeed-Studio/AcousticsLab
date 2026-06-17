//! Composes a [`crate::converter::source::SourceModel`] with an
//! [`crate::converter::sink::ArtifactSink`]; the forwarded `Arc<dyn FsService>`
//! makes the sink route writes through [`crate::file_mgr::FsService::put_atomic`].

use crate::converter::sink::ArtifactSink;
use crate::converter::source::SourceModel;
use crate::converter::{ConvertError, HeadArtifacts};
use std::path::Path;
use std::sync::Arc;

pub struct Pipeline<S: SourceModel, K: ArtifactSink> {
    source: S,
    sink: K,
    fs: Arc<dyn crate::file_mgr::FsService>,
}

impl<S: SourceModel, K: ArtifactSink> std::fmt::Debug for Pipeline<S, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("source", &self.source)
            .field("sink", &self.sink)
            .finish_non_exhaustive()
    }
}

impl<S: SourceModel, K: ArtifactSink> Pipeline<S, K> {
    pub fn new(source: S, sink: K, fs: Arc<dyn crate::file_mgr::FsService>) -> Self {
        Self { source, sink, fs }
    }

    /// Synchronous (blocking I/O); api callers wrap in `spawn_blocking`.
    pub fn run(
        &self,
        src: &Path,
        labels: &[String],
        dst_dir: &Path,
    ) -> Result<HeadArtifacts, ConvertError> {
        let loaded = self.source.load(src, labels)?;
        self.sink
            .publish(&loaded, labels, dst_dir, self.source.kind(), &self.fs)
    }
}
