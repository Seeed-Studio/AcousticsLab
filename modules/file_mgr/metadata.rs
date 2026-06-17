//! `metadata.json` schema + the `MetadataStore` aggregate of [`WorkspaceMgr`].

use std::sync::Arc;

use crate::common::ids::{AssetId, WorkspaceId};
use serde::{Deserialize, Serialize};

use crate::file_mgr::WorkspaceMgr;
use crate::file_mgr::error::{FileError, io_err, metadata_parse_err};

/// Asset kind, mapping to a `<workspace>/<subdir>/` prefix and allowed exts.
/// `snake_case` pins the wire shape (`kind: "head_mpk"`).
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AssetKind {
    Dataset,
    BackboneMpk,
    BackboneRknn,
    HeadMpk,
    HeadLabels,
    Metadata,
}

impl AssetKind {
    pub(crate) fn subdir(&self) -> &'static str {
        match self {
            AssetKind::Dataset => "datasets",
            AssetKind::BackboneMpk | AssetKind::BackboneRknn | AssetKind::HeadMpk => "weights",
            AssetKind::HeadLabels => "labels",
            AssetKind::Metadata => ".",
        }
    }

    pub(crate) fn allowed_ext(&self) -> &[&'static str] {
        match self {
            // json/bin admit TFJS source models (`model.json` + `.bin` shards).
            AssetKind::Dataset => &["tar.gz", "tgz", "zip", "json", "bin"],
            AssetKind::BackboneMpk | AssetKind::HeadMpk => &["mpk"],
            AssetKind::BackboneRknn => &["rknn"],
            AssetKind::HeadLabels => &["txt"],
            AssetKind::Metadata => &["json"],
        }
    }
}

/// Per-asset record. Records alias one on-disk file iff they share `(subdir, name)`
/// because `kind` can't disambiguate (`BackboneMpk`/`BackboneRknn` both under `weights/`),
/// so lookups key on `(subdir, name)`. `name`'s serde via [`AssetId::parse`] rejects
/// traversal names (`"../etc/passwd"`) loudly.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AssetRecord {
    pub kind: AssetKind,
    pub name: AssetId,
    /// SHA-256 of the file, lowercase hex (consumers compare as hex strings).
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceMetadata {
    pub schema_version: u32,
    pub id: WorkspaceId,
    pub name: String,
    /// RFC3339 as `String` so it round-trips byte-identically, unpinned to `time`.
    pub created_at: String,
    pub assets: Vec<AssetRecord>,
}

impl WorkspaceMetadata {
    /// Latest schema this build writes; bump on any non-backward-readable shape change
    /// so an older daemon hits [`FileError::SchemaTooNew`] instead of parsing missing
    /// fields and corrupting state on rewrite.
    pub const CURRENT: u32 = 1;

    /// Oldest schema this build accepts; raise only when dropping older support.
    pub const MIN_COMPATIBLE: u32 = 1;

    pub fn new(id: WorkspaceId, name: String) -> Self {
        Self {
            schema_version: Self::CURRENT,
            id,
            name,
            created_at: crate::file_mgr::time_util::now_rfc3339(),
            assets: Vec::new(),
        }
    }

    /// Index of the record aliasing `(kind, name)`, keyed `(subdir, name)` per [`AssetRecord`].
    pub fn find_index(&self, kind: AssetKind, name: &str) -> Option<usize> {
        let subdir = kind.subdir();
        self.assets
            .iter()
            .position(|a| a.kind.subdir() == subdir && a.name == name)
    }

    /// First asset in `kind`'s subdir matching `name` case-insensitively; guards
    /// case-insensitive filesystems (HFS+, NTFS) where `Foo.mpk` then `foo.mpk` overwrite
    /// bytes yet leave distinct records. ASCII-only fold (non-ASCII collisions slip through).
    pub fn find_case_insensitive(&self, kind: AssetKind, name: &str) -> Option<&AssetRecord> {
        let subdir = kind.subdir();
        self.assets
            .iter()
            .find(|a| a.kind.subdir() == subdir && a.name.as_str().eq_ignore_ascii_case(name))
    }
}

impl WorkspaceMgr {
    /// Per-workspace metadata lock, lazily allocated. Callers MUST pre-check `workspace.json`
    /// existence (else the lazy insert strands a lock on a never-created id) and MUST NOT
    /// hold the guard across `.await` (`parking_lot::Mutex` never yields to the runtime).
    pub(crate) fn metadata_lock(&self, id: &WorkspaceId) -> Arc<parking_lot::Mutex<()>> {
        self.metadata_locks
            .entry(*id)
            .or_insert_with(|| Arc::new(parking_lot::Mutex::new(())))
            .clone()
    }

    /// Load + parse a workspace's `metadata.json`, rejecting out-of-range schema
    /// versions per [`WorkspaceMetadata::CURRENT`]'s downgrade-write hazard.
    pub fn read_metadata(&self, id: &WorkspaceId) -> Result<WorkspaceMetadata, FileError> {
        let path = self.workspace_dir(id).join("metadata.json");
        let bytes = crate::file_mgr::schema::read_capped(
            &path,
            crate::file_mgr::schema::MAX_WORKSPACE_METADATA_BYTES,
        )?;
        let meta: WorkspaceMetadata =
            serde_json::from_slice(&bytes).map_err(|e| metadata_parse_err(path.display(), e))?;
        if meta.schema_version > WorkspaceMetadata::CURRENT {
            return Err(FileError::SchemaTooNew {
                path: path.display().to_string(),
                found: meta.schema_version,
                max: WorkspaceMetadata::CURRENT,
            });
        }
        if meta.schema_version < WorkspaceMetadata::MIN_COMPATIBLE {
            return Err(FileError::SchemaTooOld {
                path: path.display().to_string(),
                found: meta.schema_version,
                min: WorkspaceMetadata::MIN_COMPATIBLE,
            });
        }
        Ok(meta)
    }

    pub fn write_metadata(
        &self,
        id: &WorkspaceId,
        meta: &WorkspaceMetadata,
    ) -> Result<(), FileError> {
        let ws = self.workspace_dir(id);
        std::fs::create_dir_all(&ws).map_err(|e| io_err(ws.display(), e))?;
        let bytes = serde_json::to_vec_pretty(meta)?;
        crate::file_mgr::fs_atomic::put_atomic(&ws.join("metadata.json"), &bytes)
    }
}
