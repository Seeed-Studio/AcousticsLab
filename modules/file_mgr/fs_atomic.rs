//! Atomic "write bytes to a file" primitive backing every workspace
//! metadata + converter/training artifact write. Staging the tempfile in
//! `final_path`'s parent keeps the publishing rename intra-FS, hence
//! POSIX-atomic. Step order is durability-critical: write + `sync_all`
//! -> `persist` (atomic rename) -> fsync parent; see the inline barriers.
//!
//! macOS/APFS: `sync_all` lowers to `fsync(2)` which does NOT trigger a
//! drive-level flush (data only kernel-durable), whereas `fsync_dir`
//! attempts `F_FULLFSYNC` to make the directory entry drive-durable
//! (best-effort: a failure, e.g. on a network mount, is logged not
//! propagated, degrading to plain fsync); on Linux both steps are
//! kernel-durable.

use crate::file_mgr::error::{FileError, io_err};
use crate::file_mgr::validate::fsync_dir;
use std::path::Path;

/// Atomically write `bytes` to `final_path`; its parent dir must exist
/// (the staging tempfile lives there for the intra-FS rename). On failure
/// `final_path` is unchanged: the tempfile is dropped, never partial bytes
/// under the final name.
pub fn put_atomic(final_path: &Path, bytes: &[u8]) -> Result<(), FileError> {
    use std::io::Write;
    let parent = final_path.parent().ok_or_else(|| {
        io_err(
            final_path.display(),
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "put_atomic: path has no parent directory",
            ),
        )
    })?;
    let mut tmp =
        tempfile::NamedTempFile::new_in(parent).map_err(|e| io_err(parent.display(), e))?;
    tmp.write_all(bytes)
        .map_err(|e| io_err(final_path.display(), e))?;
    tmp.flush().map_err(|e| io_err(final_path.display(), e))?;
    // `flush` only drains Rust-side buffering; `sync_all` is the barrier
    // that gets data to stable storage before the rename can publish it.
    tmp.as_file()
        .sync_all()
        .map_err(|e| io_err(final_path.display(), e))?;
    tmp.persist(final_path)?;
    // fsync the parent so the rename's directory-entry update survives a
    // post-`persist` power loss (which would otherwise revert the rename).
    fsync_dir(parent).map_err(|e| io_err(parent.display(), e))?;
    Ok(())
}
