//! Asset-name + extension validation and small filesystem helpers
//! (sha256 streaming, directory fsync).

use crate::common::ids::AssetId;
use crate::file_mgr::error::{FileError, io_err};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Validate a single asset filename without allocating an [`AssetId`]; use
/// [`AssetId::parse`] when the structured [`crate::common::ids::IdError`] is
/// needed.
pub fn validate_asset_name(name: &str) -> Result<(), FileError> {
    AssetId::parse(name)
        .map(|_| ())
        .map_err(|e| FileError::InvalidName(e.to_string()))
}

pub(crate) fn validate_extension(name: &str, allowed: &[&'static str]) -> Result<(), FileError> {
    // `Path::extension` sees only the last component (`.gz` of `.tar.gz`) but we
    // accept whole suffixes; compare on bytes (allowed exts are ASCII) since a
    // str slice at `len() - need` could land mid-UTF-8-codepoint and panic.
    let name_bytes = name.as_bytes();
    for ext in allowed {
        let ext_bytes = ext.as_bytes();
        let need = ext_bytes.len() + 1;
        if name_bytes.len() < need {
            continue;
        }
        let tail = &name_bytes[name_bytes.len() - need..];
        if tail[0] == b'.' && tail[1..].eq_ignore_ascii_case(ext_bytes) {
            return Ok(());
        }
    }
    let got = name
        .find('.')
        .map(|i| &name[i + 1..])
        .unwrap_or(name)
        .to_string();
    Err(FileError::InvalidExtension {
        got,
        expected: allowed.to_vec(),
    })
}

// In `crate::common::hex` so the inference backbone (barred from depending on
// `file_mgr` by the layer guard) can share the encoder.
pub(crate) use crate::common::hex::hex_lowercase;

/// SHA-256 a file in 64 KiB chunks (lowercase-hex digest) so multi-GB uploads
/// aren't slurped into RAM. No `BufReader`: the chunk size is the I/O unit.
pub(crate) fn sha256_file_streaming(path: &Path) -> Result<String, FileError> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| io_err(path.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| io_err(path.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lowercase(&hasher.finalize()))
}

/// Fsync `path`'s parent directory so a rename into it is durable: `rename(2)`
/// is atomic, but the directory-entry update only survives power loss once the
/// directory inode is fsynced, else the entry can name a gone tempfile.
///
/// macOS: `sync_all`/`fsync(2)` only flushes to the APFS device write cache, so
/// follow with `F_FULLFSYNC` (waits for stable media). Its failure (e.g.
/// network mounts) is logged not propagated since `sync_all` already ran; first
/// failure per process at `warn!`, rest at `debug!` to avoid spam.
pub(crate) fn fsync_dir(path: &std::path::Path) -> std::io::Result<()> {
    let f = std::fs::File::open(path)?;
    f.sync_all()?;
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::io::AsRawFd;
        use std::sync::atomic::{AtomicBool, Ordering};
        static FIRST_FULLFSYNC_FAILURE_WARNED: AtomicBool = AtomicBool::new(false);
        // SAFETY: `f` owns the live fd and outlives this synchronous call.
        let rc = unsafe { libc::fcntl(f.as_raw_fd(), libc::F_FULLFSYNC) };
        if rc == -1 {
            let err = std::io::Error::last_os_error();
            if !FIRST_FULLFSYNC_FAILURE_WARNED.swap(true, Ordering::AcqRel) {
                tracing::warn!(
                    target: "file_mgr",
                    path = %path.display(),
                    err = %err,
                    "F_FULLFSYNC not honored (likely a network mount or unsupported FS); \
                     power-loss durability degrades to plain fsync; further failures \
                     logged at debug",
                );
            } else {
                tracing::debug!(
                    target: "file_mgr",
                    path = %path.display(),
                    err = %err,
                    "F_FULLFSYNC not honored (warning already emitted for this process)",
                );
            }
        }
    }
    Ok(())
}
