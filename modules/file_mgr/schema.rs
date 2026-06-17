//! Byte-level read/write helpers for `workspace.json`, `heads.json`, and
//! per-head `<head_id>.json` (shapes in `common::workspace`).
//!
//! No locking here: callers hold the per-workspace mutation mutex across
//! read-modify-write. All writes go through [`put_atomic`] so a partial
//! write never appears under the published name.

use crate::common::ids::HeadId;
use crate::common::workspace::{HeadIndex, HeadManifest, MAX_HEADS_PER_WORKSPACE, WorkspaceCore};
use crate::file_mgr::error::{FileError, io_err, metadata_parse_err};
use crate::file_mgr::fs_atomic::put_atomic;
use std::path::{Path, PathBuf};

pub const WORKSPACE_CORE_FILENAME: &str = "workspace.json";
pub const HEAD_INDEX_FILENAME: &str = "heads.json";
pub const HEADS_DIR_NAME: &str = "heads";
/// Raw head weights (Burn `.mpk` artefact).
pub const HEAD_ARTIFACT_EXTENSION: &str = "mpk";
pub const HEAD_MANIFEST_EXTENSION: &str = "json";
/// Defends the eager-cache resident set from operator tampering (nominal core < 1 KiB).
pub const MAX_WORKSPACE_CORE_BYTES: u64 = 64 * 1024;
/// Typical ~2 KiB; 1 MiB keeps a corrupt file from OOMing the boot path.
pub const MAX_HEAD_INDEX_BYTES: u64 = 1024 * 1024;
/// `labels: Vec<String>` worst case (`MAX_N_CLASSES` × ~64-byte labels) ~6 MiB; 8 MiB absorbs it.
pub const MAX_HEAD_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;
/// `current.json` is ~64 bytes.
pub const MAX_ACTIVE_CURRENT_BYTES: u64 = 4 * 1024;
/// Same labels-vector worst case as [`MAX_HEAD_MANIFEST_BYTES`].
pub const MAX_ACTIVE_MANIFEST_BYTES: u64 = 8 * 1024 * 1024;
/// Legacy workspace `metadata.json` body (a list of
/// [`crate::file_mgr::metadata::AssetRecord`]).
pub const MAX_WORKSPACE_METADATA_BYTES: u64 = 4 * 1024 * 1024;
pub const MAX_TOMBSTONE_BYTES: u64 = 4 * 1024;

pub const WORKSPACES_DIR_NAME: &str = "workspaces";
/// Root-scoped staged delete payloads; boot recovery resumes the drain.
pub const ROOT_TMP_DIR_NAME: &str = ".tmp";
pub const ACTIVE_DIR_NAME: &str = "active";
pub const ACTIVE_GENERATIONS_DIR_NAME: &str = "generations";
/// Activation staging under `<root>/active/`.
pub const ACTIVE_TMP_DIR_NAME: &str = ".tmp";
pub const ACTIVE_CURRENT_FILENAME: &str = "current.json";
pub const ACTIVE_MANIFEST_FILENAME: &str = "manifest.json";
pub const ACTIVE_HEAD_FILENAME: &str = "head.mpk";
pub const ACTIVE_LABELS_FILENAME: &str = "labels.txt";

#[inline]
pub fn workspace_core_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(WORKSPACE_CORE_FILENAME)
}

#[inline]
pub fn head_index_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(HEAD_INDEX_FILENAME)
}

#[inline]
pub fn heads_dir(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(HEADS_DIR_NAME)
}

#[inline]
pub fn head_manifest_path(workspace_dir: &Path, head_id: HeadId) -> PathBuf {
    heads_dir(workspace_dir).join(format!("{head_id}.{HEAD_MANIFEST_EXTENSION}"))
}

#[inline]
pub fn head_artifact_path(workspace_dir: &Path, head_id: HeadId) -> PathBuf {
    heads_dir(workspace_dir).join(format!("{head_id}.{HEAD_ARTIFACT_EXTENSION}"))
}

#[inline]
pub fn workspaces_dir(root: &Path) -> PathBuf {
    root.join(WORKSPACES_DIR_NAME)
}

#[inline]
pub fn workspace_dir_for(root: &Path, id: &crate::common::ids::WorkspaceId) -> PathBuf {
    workspaces_dir(root).join(id.to_string())
}

#[inline]
pub fn root_tmp_dir(root: &Path) -> PathBuf {
    root.join(ROOT_TMP_DIR_NAME)
}

#[inline]
pub fn active_dir(root: &Path) -> PathBuf {
    root.join(ACTIVE_DIR_NAME)
}

/// Absent until first activation.
#[inline]
pub fn active_current_path(root: &Path) -> PathBuf {
    active_dir(root).join(ACTIVE_CURRENT_FILENAME)
}

/// Retains current + previous; older pruned once `current.json` is durable.
#[inline]
pub fn active_generations_dir(root: &Path) -> PathBuf {
    active_dir(root).join(ACTIVE_GENERATIONS_DIR_NAME)
}

#[inline]
pub fn active_generation_dir(root: &Path, activation_id: &str) -> PathBuf {
    active_generations_dir(root).join(activation_id)
}

#[inline]
pub fn active_staging_dir(root: &Path) -> PathBuf {
    active_dir(root).join(ACTIVE_TMP_DIR_NAME)
}

/// On-disk shape of `<root>/active/current.json`. `activation_id` is a
/// generation directory name (not a `HeadId`: the `misc/heads/default/`
/// fixture is a dir name), gated on deserialize by
/// [`validate_activation_id`] so `..` cannot escape the active root.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveCurrentPointer {
    #[serde(deserialize_with = "deserialize_activation_id")]
    pub activation_id: String,
}

fn deserialize_activation_id<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let s = String::deserialize(deserializer)?;
    validate_activation_id(&s).map_err(serde::de::Error::custom)?;
    Ok(s)
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ActivationIdError {
    #[error("activation_id is empty")]
    Empty,
    #[error("activation_id length {len} > 64")]
    TooLong { len: usize },
    #[error("activation_id must not start with '.'")]
    LeadingDot,
    #[error(
        "activation_id byte 0x{byte:02x} at index {index} is forbidden \
         (allowed: [A-Za-z0-9._-])"
    )]
    BadByte { index: usize, byte: u8 },
}

/// Reject values that would escape the active root when joined into a path:
/// allows one `AssetPath` component's byte set (`[A-Za-z0-9._-]`, no leading
/// `.`, non-empty, len <= 64); rejects separators, NUL, control, non-ASCII.
pub fn validate_activation_id(s: &str) -> Result<(), ActivationIdError> {
    if s.is_empty() {
        return Err(ActivationIdError::Empty);
    }
    if s.len() > 64 {
        return Err(ActivationIdError::TooLong { len: s.len() });
    }
    if s.as_bytes()[0] == b'.' {
        return Err(ActivationIdError::LeadingDot);
    }
    for (i, &b) in s.as_bytes().iter().enumerate() {
        let ok = b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_');
        if !ok {
            return Err(ActivationIdError::BadByte { index: i, byte: b });
        }
    }
    Ok(())
}

/// A missing pointer surfaces as `FileError::Io` (`NotFound`), distinct
/// from corrupt JSON (`MetadataParse`).
pub fn read_active_current(root: &Path) -> Result<ActiveCurrentPointer, FileError> {
    let path = active_current_path(root);
    let bytes = read_capped(&path, MAX_ACTIVE_CURRENT_BYTES)?;
    serde_json::from_slice(&bytes).map_err(|source| metadata_parse_err(path.display(), source))
}

/// Atomically rewrite the pointer: it reflects either the prior or the new
/// generation, never an in-between.
pub fn write_active_current(root: &Path, pointer: &ActiveCurrentPointer) -> Result<(), FileError> {
    // Re-validate at the write boundary: a caller building the pointer from
    // operator input could otherwise write a hostile id to `current.json`.
    validate_activation_id(&pointer.activation_id)
        .map_err(|e| FileError::InvalidName(e.to_string()))?;
    let path = active_current_path(root);
    let bytes = serde_json::to_vec(pointer)?;
    put_atomic(&path, &bytes)
}

/// Caller must invoke [`crate::common::workspace::ActiveHeadManifest::validate`]
/// before trusting the result (serde misses the structural invariants).
/// `activation_id` is gated against path traversal.
pub fn read_active_manifest(
    root: &Path,
    activation_id: &str,
) -> Result<crate::common::workspace::ActiveHeadManifest, FileError> {
    validate_activation_id(activation_id).map_err(|e| FileError::InvalidName(e.to_string()))?;
    let path = active_generation_dir(root, activation_id).join(ACTIVE_MANIFEST_FILENAME);
    let bytes = read_capped(&path, MAX_ACTIVE_MANIFEST_BYTES)?;
    serde_json::from_slice(&bytes).map_err(|source| metadata_parse_err(path.display(), source))
}

/// Atomically rewrite a generation's `manifest.json`; `activation_id` is
/// gated against path traversal.
pub fn write_active_manifest(
    root: &Path,
    activation_id: &str,
    manifest: &crate::common::workspace::ActiveHeadManifest,
) -> Result<(), FileError> {
    validate_activation_id(activation_id).map_err(|e| FileError::InvalidName(e.to_string()))?;
    let path = active_generation_dir(root, activation_id).join(ACTIVE_MANIFEST_FILENAME);
    let bytes = serde_json::to_vec(manifest)?;
    put_atomic(&path, &bytes)
}

/// Absent file surfaces as `Io` (callers inspect the inner `io::ErrorKind`
/// for absent-vs-corrupt); oversize fails [`FileError::MetadataTooLarge`].
pub fn read_workspace_core(workspace_dir: &Path) -> Result<WorkspaceCore, FileError> {
    let path = workspace_core_path(workspace_dir);
    let bytes = read_capped(&path, MAX_WORKSPACE_CORE_BYTES)?;
    serde_json::from_slice(&bytes).map_err(|source| metadata_parse_err(path.display(), source))
}

/// Refuses an oversize body (hot-path cache budget) before touching disk;
/// the cap/serialize short-circuits skip the write-duration metric, which
/// records on both the success and failure disk arms.
pub fn write_workspace_core(workspace_dir: &Path, core: &WorkspaceCore) -> Result<(), FileError> {
    let path = workspace_core_path(workspace_dir);
    let bytes = serde_json::to_vec(core)?;
    if (bytes.len() as u64) > MAX_WORKSPACE_CORE_BYTES {
        return Err(FileError::MetadataTooLarge {
            path: path.display().to_string(),
            observed: bytes.len() as u64,
            max: MAX_WORKSPACE_CORE_BYTES,
        });
    }
    put_atomic_timed(
        &path,
        &bytes,
        crate::file_mgr::metrics_hooks::emit_workspace_core_write,
    )
}

/// Enforces the `MAX_HEADS_PER_WORKSPACE` cap on read so a corrupt over-cap
/// file fails closed rather than feeding stale heads into activation;
/// over-cap surfaces as a synthesized `MetadataParse`, uniform with real
/// parse failures.
pub fn read_head_index(workspace_dir: &Path) -> Result<HeadIndex, FileError> {
    let path = head_index_path(workspace_dir);
    let bytes = read_capped(&path, MAX_HEAD_INDEX_BYTES)?;
    let index: HeadIndex = serde_json::from_slice(&bytes)
        .map_err(|source| metadata_parse_err(path.display(), source))?;
    if index.heads.len() > MAX_HEADS_PER_WORKSPACE {
        return Err(over_cap_metadata_err(
            &path,
            format!(
                "heads.json has {} entries; max is {}",
                index.heads.len(),
                MAX_HEADS_PER_WORKSPACE
            ),
        ));
    }
    Ok(index)
}

/// Refuses more than `MAX_HEADS_PER_WORKSPACE` entries (symmetric with the
/// read cap); records the write-duration metric on both disk arms.
pub fn write_head_index(workspace_dir: &Path, index: &HeadIndex) -> Result<(), FileError> {
    let path = head_index_path(workspace_dir);
    if index.heads.len() > MAX_HEADS_PER_WORKSPACE {
        return Err(over_cap_metadata_err(
            &path,
            format!(
                "refusing to publish {} entries; max is {}",
                index.heads.len(),
                MAX_HEADS_PER_WORKSPACE
            ),
        ));
    }
    let bytes = serde_json::to_vec(index)?;
    put_atomic_timed(
        &path,
        &bytes,
        crate::file_mgr::metrics_hooks::emit_head_index_write,
    )
}

/// Runs [`HeadManifest::validate`] after parse so a mismatched
/// `n_classes`/`labels.len()` fails closed before activation.
pub fn read_head_manifest(
    workspace_dir: &Path,
    head_id: HeadId,
) -> Result<HeadManifest, FileError> {
    let path = head_manifest_path(workspace_dir, head_id);
    let bytes = read_capped(&path, MAX_HEAD_MANIFEST_BYTES)?;
    let manifest: HeadManifest = serde_json::from_slice(&bytes)
        .map_err(|source| metadata_parse_err(path.display(), source))?;
    manifest.validate().map_err(|e| {
        metadata_parse_err(
            path.display(),
            serde_json::Error::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("HeadManifest validation: {e}"),
            )),
        )
    })?;
    Ok(manifest)
}

/// Atomically rewrite a per-head manifest. Trusts `manifest.head_id`
/// for the filename; does not cross-check it against the body.
pub fn write_head_manifest(workspace_dir: &Path, manifest: &HeadManifest) -> Result<(), FileError> {
    let path = head_manifest_path(workspace_dir, manifest.head_id);
    let bytes = serde_json::to_vec(manifest)?;
    put_atomic(&path, &bytes)
}

/// `put_atomic` timed, emitting on both arms (a slow fsync that then errored
/// still belongs in the latency histogram).
fn put_atomic_timed(
    path: &Path,
    bytes: &[u8],
    emit: impl FnOnce(std::time::Duration),
) -> Result<(), FileError> {
    let start = std::time::Instant::now();
    let res = put_atomic(path, bytes);
    emit(start.elapsed());
    res
}

/// Synthesize a `MetadataParse` from an over-cap/corruption message so the
/// wire-error category stays uniform with real serde failures.
fn over_cap_metadata_err(path: &Path, msg: String) -> FileError {
    metadata_parse_err(
        path.display(),
        serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, msg)),
    )
}

/// Read `path` into memory, rejecting any file larger than `cap`. The
/// `metadata().len()` precheck stops an attacker forcing a giant allocation
/// before the cap fires. `pub(crate)` so every JSON read shares this gate.
pub(crate) fn read_capped(path: &Path, cap: u64) -> Result<Vec<u8>, FileError> {
    use std::io::Read;
    let f = std::fs::File::open(path).map_err(|source| io_err(path.display(), source))?;
    let metadata = f
        .metadata()
        .map_err(|source| io_err(path.display(), source))?;
    if metadata.len() > cap {
        return Err(FileError::MetadataTooLarge {
            path: path.display().to_string(),
            observed: metadata.len(),
            max: cap,
        });
    }
    // `take(cap)` re-caps against a torn write growing the file after the
    // stat; clamping the hint to `min(len, cap)` keeps the `as usize` cast
    // sound on 32-bit hosts where `cap` could exceed `usize::MAX`.
    let cap_for_hint = std::cmp::min(metadata.len(), cap);
    let mut buf = Vec::with_capacity(cap_for_hint as usize);
    let mut limited = f.take(cap);
    limited
        .read_to_end(&mut buf)
        .map_err(|source| io_err(path.display(), source))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // corruption fixtures use raw writes

    use super::*;
    use crate::common::ids::WorkspaceId;
    use crate::common::workspace::{
        HeadIndex, HeadManifest, HeadRecord, MAX_HEADS_PER_WORKSPACE, WorkspaceCore,
        WorkspaceRevision,
    };

    fn ws_id() -> WorkspaceId {
        WorkspaceId::parse("11111111-2222-4333-8444-555555555555").unwrap()
    }

    fn head_id() -> HeadId {
        HeadId::parse("11111111-2222-4333-8444-555555555556").unwrap()
    }

    fn rev(id: u64) -> WorkspaceRevision {
        WorkspaceRevision {
            id,
            at: "2026-05-07T12:00:00Z".to_string(),
        }
    }

    fn sample_core() -> WorkspaceCore {
        WorkspaceCore {
            id: ws_id(),
            name: "main".to_string(),
            tags: Vec::new(),
            created_at: "2026-05-07T12:34:56Z".to_string(),
            workspace_revision: rev(5),
            head_count: 1,
        }
    }

    fn sample_head_record() -> HeadRecord {
        HeadRecord {
            head_id: head_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 12,
            size_bytes: 4096,
            created_at: "2026-05-07T12:34:56Z".to_string(),
        }
    }

    fn sample_manifest() -> HeadManifest {
        HeadManifest {
            head_id: head_id(),
            workspace_id: ws_id(),
            workspace_revision: rev(5),
            sha256: "def".to_string(),
            n_classes: 2,
            size_bytes: 4096,
            created_at: "2026-05-07T12:34:56Z".to_string(),
            labels: vec!["cat".to_string(), "dog".to_string()],
        }
    }

    #[test]
    fn path_helpers_join_redesign_filenames() {
        let ws = Path::new("/tmp/ws");
        assert_eq!(workspace_core_path(ws), Path::new("/tmp/ws/workspace.json"));
        assert_eq!(head_index_path(ws), Path::new("/tmp/ws/heads.json"));
        assert_eq!(heads_dir(ws), Path::new("/tmp/ws/heads"));
        let id = head_id();
        assert_eq!(
            head_manifest_path(ws, id),
            Path::new("/tmp/ws/heads/11111111-2222-4333-8444-555555555556.json")
        );
        assert_eq!(
            head_artifact_path(ws, id),
            Path::new("/tmp/ws/heads/11111111-2222-4333-8444-555555555556.mpk")
        );
    }

    #[test]
    fn workspace_core_write_then_read_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path();
        let core = sample_core();
        write_workspace_core(ws, &core).unwrap();
        // Parented directly under workspace_dir (registry adds `workspaces/`).
        assert!(ws.join("workspace.json").is_file());
        let read = read_workspace_core(ws).unwrap();
        assert_eq!(read, core);
    }

    #[test]
    fn workspace_core_read_missing_file_surfaces_io_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let res = read_workspace_core(tmp.path());
        match res {
            Err(FileError::Io { source, .. }) => {
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected FileError::Io NotFound, got {other:?}"),
        }
    }

    /// Corrupt JSON is `MetadataParse`, not `Io` (malformed vs missing).
    #[test]
    fn workspace_core_read_rejects_corrupt_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(WORKSPACE_CORE_FILENAME), b"{ not json").unwrap();
        let res = read_workspace_core(tmp.path());
        assert!(matches!(res, Err(FileError::MetadataParse { .. })));
    }

    /// Read-side cap is defensive; the writer enforces it too.
    #[test]
    fn workspace_core_read_rejects_oversize_file() {
        let tmp = tempfile::tempdir().unwrap();
        let big = vec![b'a'; (MAX_WORKSPACE_CORE_BYTES + 1) as usize];
        std::fs::write(tmp.path().join(WORKSPACE_CORE_FILENAME), &big).unwrap();
        let res = read_workspace_core(tmp.path());
        assert!(matches!(
            res,
            Err(FileError::MetadataTooLarge { observed, max, .. })
                if observed == MAX_WORKSPACE_CORE_BYTES + 1 && max == MAX_WORKSPACE_CORE_BYTES
        ));
    }

    /// Writer cap fires before any disk write (no partial file left behind).
    #[test]
    fn workspace_core_write_rejects_oversize_body() {
        let tmp = tempfile::tempdir().unwrap();
        let mut core = sample_core();
        core.name = "x".repeat(MAX_WORKSPACE_CORE_BYTES as usize);
        let res = write_workspace_core(tmp.path(), &core);
        assert!(matches!(res, Err(FileError::MetadataTooLarge { .. })));
        assert!(!tmp.path().join(WORKSPACE_CORE_FILENAME).exists());
    }

    /// temp+rename makes each write appear wholly-new, never partial.
    #[test]
    fn workspace_core_write_is_atomic_via_put_atomic() {
        let tmp = tempfile::tempdir().unwrap();
        let core = sample_core();
        write_workspace_core(tmp.path(), &core).unwrap();
        let mut core2 = core.clone();
        core2.workspace_revision = rev(6);
        write_workspace_core(tmp.path(), &core2).unwrap();
        let read = read_workspace_core(tmp.path()).unwrap();
        assert_eq!(read.workspace_revision, rev(6));
    }

    #[test]
    fn head_index_round_trip_at_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let mut index = HeadIndex::default();
        // `{:02x}` suffix gives distinct UUIDs for caps up to 255.
        for i in 0..MAX_HEADS_PER_WORKSPACE {
            let mut rec = sample_head_record();
            rec.head_id =
                HeadId::parse(&format!("11111111-2222-4333-8444-5555555555{:02x}", i + 1)).unwrap();
            rec.workspace_revision = rev((i + 3) as u64);
            index.heads.push(rec);
        }
        assert_eq!(index.heads.len(), MAX_HEADS_PER_WORKSPACE);
        write_head_index(tmp.path(), &index).unwrap();
        let read = read_head_index(tmp.path()).unwrap();
        assert_eq!(read, index);
    }

    #[test]
    fn head_index_read_missing_file_surfaces_io_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        match read_head_index(tmp.path()) {
            Err(FileError::Io { source, .. }) => {
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected FileError::Io NotFound, got {other:?}"),
        }
    }

    #[test]
    fn head_index_default_is_empty_and_writeable() {
        let tmp = tempfile::tempdir().unwrap();
        let empty = HeadIndex::default();
        write_head_index(tmp.path(), &empty).unwrap();
        let read = read_head_index(tmp.path()).unwrap();
        assert_eq!(read, empty);
        assert!(read.heads.is_empty());
    }

    /// Reader rejects over-cap entries (defends hand-edited `heads.json`).
    #[test]
    fn head_index_read_rejects_over_cap_entries() {
        let tmp = tempfile::tempdir().unwrap();
        // Write directly, bypassing the writer cap.
        let mut over = HeadIndex::default();
        for i in 0..(MAX_HEADS_PER_WORKSPACE + 1) {
            let mut rec = sample_head_record();
            rec.head_id =
                HeadId::parse(&format!("11111111-2222-4333-8444-5555555555{:02x}", i + 1)).unwrap();
            over.heads.push(rec);
        }
        let bytes = serde_json::to_vec(&over).unwrap();
        std::fs::write(tmp.path().join(HEAD_INDEX_FILENAME), &bytes).unwrap();
        let res = read_head_index(tmp.path());
        assert!(matches!(res, Err(FileError::MetadataParse { .. })));
    }

    /// Writer-side cap refuses over-cap entries from a buggy caller.
    #[test]
    fn head_index_write_rejects_over_cap_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let mut over = HeadIndex::default();
        for i in 0..(MAX_HEADS_PER_WORKSPACE + 1) {
            let mut rec = sample_head_record();
            rec.head_id =
                HeadId::parse(&format!("11111111-2222-4333-8444-5555555555{:02x}", i + 1)).unwrap();
            over.heads.push(rec);
        }
        let res = write_head_index(tmp.path(), &over);
        assert!(matches!(res, Err(FileError::MetadataParse { .. })));
        assert!(!tmp.path().join(HEAD_INDEX_FILENAME).exists());
    }

    #[test]
    fn head_manifest_round_trips_under_heads_dir() {
        let tmp = tempfile::tempdir().unwrap();
        // heads/ must exist before writing a manifest (lifecycle code makes it).
        std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();
        let manifest = sample_manifest();
        write_head_manifest(tmp.path(), &manifest).unwrap();
        let expected =
            heads_dir(tmp.path()).join(format!("{}.{}", manifest.head_id, HEAD_MANIFEST_EXTENSION));
        assert!(expected.is_file());
        let read = read_head_manifest(tmp.path(), manifest.head_id).unwrap();
        assert_eq!(read, manifest);
    }

    #[test]
    fn head_manifest_read_rejects_corrupt_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();
        let path = head_manifest_path(tmp.path(), head_id());
        std::fs::write(&path, b"{ not json").unwrap();
        let res = read_head_manifest(tmp.path(), head_id());
        assert!(matches!(res, Err(FileError::MetadataParse { .. })));
    }

    /// Read runs `validate`, so mismatched `n_classes` vs `labels.len()` fails closed.
    #[test]
    fn head_manifest_read_rejects_n_classes_labels_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();
        let mut bad = sample_manifest();
        bad.n_classes = 5; // labels.len() == 2
        let path = head_manifest_path(tmp.path(), bad.head_id);
        std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        let res = read_head_manifest(tmp.path(), bad.head_id);
        match res {
            Err(FileError::MetadataParse { source, .. }) => {
                assert!(
                    source.to_string().contains("n_classes"),
                    "diagnostic must mention n_classes; got {source}",
                );
            }
            other => panic!("expected MetadataParse on n_classes mismatch; got {other:?}"),
        }
    }

    /// Same for the `n_classes = 0` corner case.
    #[test]
    fn head_manifest_read_rejects_zero_classes() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(heads_dir(tmp.path())).unwrap();
        let mut bad = sample_manifest();
        bad.n_classes = 0;
        bad.labels.clear();
        let path = head_manifest_path(tmp.path(), bad.head_id);
        std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        let res = read_head_manifest(tmp.path(), bad.head_id);
        assert!(
            matches!(res, Err(FileError::MetadataParse { .. })),
            "expected MetadataParse on zero classes; got {res:?}",
        );
    }
}
