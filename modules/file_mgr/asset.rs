//! Kind-aware asset reads + full workspace-walk validation for [`WorkspaceMgr`].

use std::path::PathBuf;

use crate::common::ids::{AssetId, WorkspaceId};

use crate::file_mgr::WorkspaceMgr;
use crate::file_mgr::WorkspaceReport;
use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::metadata::{AssetKind, AssetRecord};
use crate::file_mgr::validate::{sha256_file_streaming, validate_asset_name};

impl WorkspaceMgr {
    /// Resolve an asset path (no existence check). Panics every build on an
    /// invalid `name` to block a `..` escape outside the workspace; callers must
    /// pre-validate via [`validate_asset_name`] or hold an [`AssetId`].
    pub fn asset_path(&self, id: &WorkspaceId, kind: AssetKind, name: &str) -> PathBuf {
        validate_asset_name(name)
            .unwrap_or_else(|e| panic!("asset_path called with unvalidated name {name:?} ({e})"));
        self.workspace_dir(id).join(kind.subdir()).join(name)
    }

    /// Typed sibling of [`Self::asset_path`]; skips the name re-check since
    /// [`AssetId`] already enforces the allowlist on construction.
    pub fn asset_path_typed(&self, id: &WorkspaceId, kind: AssetKind, name: &AssetId) -> PathBuf {
        self.workspace_dir(id)
            .join(kind.subdir())
            .join(name.as_str())
    }

    /// Assets of a kind from cached `metadata.json` (no filesystem walk).
    pub fn list_assets(
        &self,
        id: &WorkspaceId,
        kind: AssetKind,
    ) -> Result<Vec<AssetRecord>, FileError> {
        let meta = self.read_metadata(id)?;
        Ok(meta.assets.into_iter().filter(|a| a.kind == kind).collect())
    }

    /// Recompute sha256 per declared asset: `missing` (declared, absent),
    /// `corrupt` (hash differs), `extra` (on disk, undeclared). Scoped to the
    /// `metadata.json`-tracked trees only (`datasets/`, `weights/`, `labels/`);
    /// the rest have their own validation or no canonical expected-set. Hashing
    /// streams (`sha256_file_streaming`) so multi-GB assets never load into RAM.
    pub fn validate(&self, id: &WorkspaceId) -> Result<WorkspaceReport, FileError> {
        let meta = self.read_metadata(id)?;
        let ws = self.workspace_dir(id);
        let mut missing = Vec::new();
        let mut corrupt = Vec::new();

        for a in &meta.assets {
            let p = self.asset_path(id, a.kind, a.name.as_str());
            if !p.exists() {
                missing.push((a.kind, a.name.as_str().to_string()));
                continue;
            }
            let digest = sha256_file_streaming(&p)?;
            if digest != a.sha256 {
                corrupt.push((a.kind, a.name.as_str().to_string()));
            }
        }

        let mut extra: Vec<(PathBuf, String)> = Vec::new();
        for subdir in ["datasets", "weights", "labels"] {
            let dir = ws.join(subdir);
            if !dir.exists() {
                continue;
            }
            for e in std::fs::read_dir(&dir).map_err(|err| io_err(dir.display(), err))? {
                let e = e.map_err(|err| io_err(dir.display(), err))?;
                if !e.file_type().is_ok_and(|t| t.is_file()) {
                    continue;
                }
                let name = e.file_name().to_string_lossy().into_owned();
                let known = meta
                    .assets
                    .iter()
                    .any(|a| a.kind.subdir() == subdir && a.name.as_str() == name);
                if !known {
                    extra.push((e.path(), name));
                }
            }
        }

        Ok(WorkspaceReport {
            ok: missing.is_empty() && corrupt.is_empty(),
            missing,
            corrupt,
            extra,
        })
    }
}
