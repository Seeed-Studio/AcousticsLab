//! Domain config sub-sections; each carries its own `validate()` aggregated by `LaunchConfig::load`.

use crate::stream_io::TransportPolicy;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Default UDS mode 0o666 (world R/W); 0o660 is too tight for non-root tools.
fn default_uds_mode() -> u32 {
    0o666
}

/// 64 slots ~= 1.3 s of 20 ms Opus before a lagging client trips WS 1011.
fn default_broadcast_capacity() -> usize {
    64
}

/// `[output.inference]` -- raw-UDS push socket emitting the length-prefixed protobuf `Envelope`
/// inference stream; conn-capped by the hardcoded `stream_io::INFERENCE_UDS_MAX_CONNS` (no knob:
/// the decoder's per-conn pre-alloc needs a non-zero cap operators can't disable).
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OutputInferenceCfg {
    pub uds_path: PathBuf,
    #[serde(default = "default_uds_mode")]
    pub uds_mode: u32,
}

impl OutputInferenceCfg {
    pub fn validate(&self) -> Result<(), String> {
        validate_uds_path(&self.uds_path, "output.inference.uds_path")?;
        validate_uds_mode(self.uds_mode, "output.inference.uds_mode")?;
        Ok(())
    }
}

/// `[output]` table wrapper.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct OutputCfg {
    #[serde(default)]
    pub inference: Option<OutputInferenceCfg>,
}

impl OutputCfg {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(inference) = &self.inference {
            inference.validate()?;
        }
        Ok(())
    }
}

/// `[api]` -- HTTP API + WebSocket listener; startup-only, not hot-reloaded. `validate` requires
/// at least 1 of `tcp_bind`/`uds_path` (browser UI needs TCP) as the config-time gate;
/// `stream_io::bind_*` re-checks race-safely at bind time -- both fail closed.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ApiCfg {
    /// `--tcp-bind` CLI flag overrides at boot (test-harness escape hatch for ephemeral ports).
    #[serde(default)]
    pub tcp_bind: Option<String>,
    #[serde(default)]
    pub uds_path: Option<PathBuf>,
    #[serde(default = "default_uds_mode")]
    pub uds_mode: u32,
    #[serde(default = "default_broadcast_capacity")]
    pub broadcast_capacity: usize,
    #[serde(default = "default_tcp_policy")]
    pub tcp_policy: TransportPolicy,
    /// UDS relaxes the subprotocol gate: UDS is already gated by filesystem perms.
    #[serde(default = "default_uds_policy")]
    pub uds_policy: TransportPolicy,
}

impl Default for ApiCfg {
    fn default() -> Self {
        Self {
            // Loopback-only is the safest first-boot default.
            tcp_bind: Some("127.0.0.1:8787".into()),
            uds_path: None,
            uds_mode: default_uds_mode(),
            broadcast_capacity: default_broadcast_capacity(),
            tcp_policy: default_tcp_policy(),
            uds_policy: default_uds_policy(),
        }
    }
}

fn default_tcp_policy() -> TransportPolicy {
    TransportPolicy::capped()
}

fn default_uds_policy() -> TransportPolicy {
    TransportPolicy {
        require_subprotocol: false,
        ..TransportPolicy::capped()
    }
}

impl ApiCfg {
    /// Inclusive cap; `tokio::broadcast::channel` eagerly allocates the full ring, so an oversized
    /// `broadcast_capacity` OOM-kills at boot.
    pub const MAX_BROADCAST_CAPACITY: usize = 1024;

    pub fn validate(&self) -> Result<(), String> {
        if self.tcp_bind.is_none() && self.uds_path.is_none() {
            return Err("api: configure at least one listener -- set `tcp_bind` \
                 (browsers/LAN) and/or `uds_path` (local tools)"
                .into());
        }
        if let Some(tcp_bind) = &self.tcp_bind {
            validate_tcp_bind(tcp_bind)?;
        }
        if let Some(uds_path) = &self.uds_path {
            validate_uds_path(uds_path, "api.uds_path")?;
            validate_uds_mode(self.uds_mode, "api.uds_mode")?;
        }
        if self.broadcast_capacity == 0 {
            return Err("api.broadcast_capacity must be > 0".into());
        }
        if self.broadcast_capacity > Self::MAX_BROADCAST_CAPACITY {
            return Err(format!(
                "api.broadcast_capacity {} exceeds the {}-slot maximum",
                self.broadcast_capacity,
                Self::MAX_BROADCAST_CAPACITY,
            ));
        }
        Ok(())
    }
}

fn validate_uds_mode(mode: u32, label: &str) -> Result<(), String> {
    if mode > 0o7777 {
        return Err(format!(
            "{label} {mode:#o} exceeds the 12-bit POSIX mode range (max 0o7777)"
        ));
    }
    Ok(())
}

/// Shape-only validation (any host accepted, reject only unbindable shapes); auth/exposure is the
/// operator's reverse-proxy responsibility.
fn validate_tcp_bind(tcp_bind: &str) -> Result<(), String> {
    if tcp_bind.is_empty() {
        return Err("api.tcp_bind must be non-empty".into());
    }
    // Split on the LAST colon so IPv6 bracketed forms (`[::1]:8787`) parse.
    let (host_raw, port) = tcp_bind
        .rsplit_once(':')
        .ok_or_else(|| format!("api.tcp_bind {tcp_bind:?} must be host:port (missing colon)"))?;
    port.parse::<u16>().map_err(|_| {
        format!("api.tcp_bind {tcp_bind:?} port component must parse as u16; got {port:?}")
    })?;
    // Strip IPv6 brackets so `[]:8787` and `:8787` both hit the empty-host check.
    let host = host_raw
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(host_raw);
    if host.is_empty() {
        return Err(format!(
            "api.tcp_bind {tcp_bind:?} has empty host (use e.g. 127.0.0.1, 0.0.0.0, or [::])"
        ));
    }
    Ok(())
}

/// Static config-time gate (`stream_io::bind_uds` owns the race-safe bind-time delete + fd-handover):
/// require an explicit existing parent dir (bare filename => undefined CWD; auto-create masks typos)
/// and an existing path be a Unix socket. All stats use `symlink_metadata` to hard-reject symlinks
/// rather than follow them, since following at the bind-time unlink is a TOCTOU race. A
/// world-writable-no-sticky parent is deferred to `bind_uds` (warns and binds; operator owns it).
fn validate_uds_path(uds_path: &Path, label: &str) -> Result<(), String> {
    if uds_path.as_os_str().is_empty() {
        return Err(format!("{label} must be non-empty"));
    }
    // `parent()` is `None` for root/empty; bare `foo.sock` yields `Some("")` (CWD) -- same mistake.
    let no_parent_err = || {
        format!(
            "{label} {:?} has no parent directory; specify a full path \
             (e.g. /run/acoustics_lab.sock)",
            uds_path.display()
        )
    };
    let parent = uds_path.parent().ok_or_else(no_parent_err)?;
    if parent.as_os_str().is_empty() {
        return Err(no_parent_err());
    }
    // `symlink_metadata` (not `Path::exists`) distinguishes absent from EACCES.
    match std::fs::symlink_metadata(parent) {
        Ok(md) => {
            if md.file_type().is_symlink() {
                return Err(format!(
                    "{label} {:?}: parent {} is a symlink; refuse to bind \
                     (/var/run is a symlink to /run on systemd distros -- \
                     use /run/acoustics_lab.sock instead)",
                    uds_path.display(),
                    parent.display(),
                ));
            }
            if !md.file_type().is_dir() {
                return Err(format!(
                    "{label} {:?}: parent {} exists but is not a directory",
                    uds_path.display(),
                    parent.display(),
                ));
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "{label} {:?}: parent directory {} does not exist; \
                 create it (e.g. systemd-tmpfiles) or pick an existing path",
                uds_path.display(),
                parent.display(),
            ));
        }
        Err(e) => {
            return Err(format!(
                "{label} {:?}: stat parent {} failed: {e}",
                uds_path.display(),
                parent.display(),
            ));
        }
    }
    // `symlink_metadata` not `metadata`: following a symlink lets a parent-dir attacker aim the
    // bind-time unlink at e.g. `/etc/passwd`.
    match std::fs::symlink_metadata(uds_path) {
        Ok(md) => {
            let ft = md.file_type();
            if ft.is_symlink() {
                return Err(format!(
                    "{label} {:?} is a symlink; refuse to bind",
                    uds_path.display()
                ));
            }
            if ft.is_file() {
                return Err(format!(
                    "{label} {:?} is a regular file; refuse to bind",
                    uds_path.display()
                ));
            }
            if ft.is_dir() {
                return Err(format!(
                    "{label} {:?} is a directory; refuse to bind",
                    uds_path.display()
                ));
            }
            // Reject FIFO/device so the bind-time unlink can't touch them.
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileTypeExt;
                if !ft.is_socket() {
                    return Err(format!(
                        "{label} {:?} is a {} (not a unix socket); \
                         refuse to bind",
                        uds_path.display(),
                        describe_unix_file_type(&ft),
                    ));
                }
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Fresh path -- bind will create it.
        }
        Err(e) => {
            return Err(format!(
                "{label} {:?}: stat failed: {e}",
                uds_path.display()
            ));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn describe_unix_file_type(ft: &std::fs::FileType) -> &'static str {
    use std::os::unix::fs::FileTypeExt;
    if ft.is_fifo() {
        "fifo"
    } else if ft.is_block_device() {
        "block device"
    } else if ft.is_char_device() {
        "char device"
    } else {
        "non-socket file"
    }
}

/// Default training hyperparameters; per-job invocations can override.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingDefaults {
    pub epochs: u32,
    pub batch_size: u32,
    /// micro-lr (100 = 1e-4); int for clean TOML round-trip.
    pub learning_rate_e6: u32,
}

impl Default for TrainingDefaults {
    fn default() -> Self {
        Self {
            epochs: 50,
            batch_size: 16,
            learning_rate_e6: 100, // 1e-4
        }
    }
}

impl TrainingDefaults {
    /// Reject defaults that fail every job (zeros) or OOM (millions) at boot, not first
    /// `POST /train`. Bounds are typo-guards, not hardware limits (override per-job); the
    /// `learning_rate_e6` cap keeps lr at or below 1.0.
    pub fn validate(&self) -> Result<(), String> {
        const MAX_EPOCHS: u32 = 10_000;
        const MAX_BATCH: u32 = 4_096;
        const MAX_LR_E6: u32 = 1_000_000;

        if self.epochs == 0 {
            return Err("training_defaults.epochs must be >= 1".into());
        }
        if self.epochs > MAX_EPOCHS {
            return Err(format!(
                "training_defaults.epochs {} exceeds {} (override per-job if intended)",
                self.epochs, MAX_EPOCHS
            ));
        }
        if self.batch_size == 0 {
            return Err("training_defaults.batch_size must be >= 1".into());
        }
        if self.batch_size > MAX_BATCH {
            return Err(format!(
                "training_defaults.batch_size {} exceeds {} (override per-job if intended)",
                self.batch_size, MAX_BATCH
            ));
        }
        if self.learning_rate_e6 == 0 {
            return Err("training_defaults.learning_rate_e6 must be >= 1".into());
        }
        if self.learning_rate_e6 > MAX_LR_E6 {
            return Err(format!(
                "training_defaults.learning_rate_e6 {} exceeds {} (~1.0 lr)",
                self.learning_rate_e6, MAX_LR_E6
            ));
        }
        Ok(())
    }
}

/// File-service admission caps; mirrors `file_mgr::AdmissionCfg` but lives in `config` for TOML override.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FileCfg {
    pub max_upload_bytes: u64,
    /// `usize` for the `tokio::sync::Semaphore`; bridges to `AdmissionCfg`'s `u32` at the boundary.
    pub max_concurrent_uploads: usize,
}

impl Default for FileCfg {
    fn default() -> Self {
        Self {
            max_upload_bytes: 256 * 1024 * 1024, // 256 MiB
            max_concurrent_uploads: 4,
        }
    }
}

impl FileCfg {
    /// Zero on either field refuses every upload; reject at boot.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_upload_bytes == 0 {
            return Err("file.max_upload_bytes must be > 0".into());
        }
        if self.max_concurrent_uploads == 0 {
            return Err("file.max_concurrent_uploads must be > 0".into());
        }
        Ok(())
    }
}
