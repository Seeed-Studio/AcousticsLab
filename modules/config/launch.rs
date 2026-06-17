//! Launch-time deployment manifest (immutable at runtime) + cross-validation
//! helpers between user-pref `Config` and the launch catalogues.

use crate::audio_io::mic_arbitrator::{
    CandidateSource, MicCandidate, MicCatalogue, MicPolicy, PolicyValidationError,
};
use crate::audio_io::mock::Waveform;
use crate::common::ids::MicId;
use crate::config::domain::{ApiCfg, FileCfg, OutputCfg, TrainingDefaults};
use crate::config::error::{ConfigError, parse_err, read_err, write_err};
use crate::inference::BackboneCatalogue;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct HeadLaunchConfig {
    /// Deployment-bundled default head; when omitted, boot recovery and
    /// `POST /active { default: true }` have no classifier to fall back to.
    #[serde(default)]
    pub default: Option<DefaultHeadRef>,
}

impl HeadLaunchConfig {
    fn validate(&self) -> Result<(), String> {
        if let Some(default) = &self.default {
            default.validate()?;
        }
        Ok(())
    }
}

/// Independent `.mpk` + labels paths so the daemon assumes no filename or layout
/// convention for the bundled default head.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DefaultHeadRef {
    pub path: PathBuf,
    pub labels_path: PathBuf,
}

impl DefaultHeadRef {
    fn validate(&self) -> Result<(), String> {
        if self.path.as_os_str().is_empty() {
            return Err("head.default.path must be non-empty".into());
        }
        if self.labels_path.as_os_str().is_empty() {
            return Err("head.default.labels_path must be non-empty".into());
        }
        Ok(())
    }
}

/// Launch-time deployment manifest. Read once at daemon boot; never mutated by
/// API, never hot-reloaded - operators edit the file and restart to apply changes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct LaunchConfig {
    pub mic: MicCatalogue,
    /// Ordered backbone candidates (empty = daemon runs without inference);
    /// `load_first_supported` skips kinds the build lacks (e.g. rknn off cfg) so one
    /// launch.toml ships on both macOS host-dev and aarch64.
    #[serde(default)]
    pub backbone: BackboneCatalogue,
    pub api: ApiCfg,
    #[serde(default)]
    pub output: OutputCfg,
    #[serde(default)]
    pub head: HeadLaunchConfig,
    #[serde(default)]
    pub training_defaults: TrainingDefaults,
    #[serde(default)]
    pub file: FileCfg,
}

impl LaunchConfig {
    /// First-boot defaults: a mock candidate so a fresh dev workstation produces
    /// audio without editing the TOML; empty backbone forces deployments to name
    /// concrete artifacts rather than inherit hardcoded paths.
    pub fn default_for() -> Self {
        Self {
            mic: MicCatalogue {
                candidates: vec![MicCandidate {
                    id: MicId::from_static("default-mock"),
                    source: CandidateSource::Mock {
                        waveforms: vec![Waveform::Sine {
                            freq_hz: 1_000.0,
                            amplitude: 0.25,
                        }],
                        period_size: 512,
                        sample_rate: crate::common::dims::SampleRate::VALUE,
                    },
                    channels: vec![0],
                }],
            },
            backbone: BackboneCatalogue::default(),
            api: ApiCfg::default(),
            output: OutputCfg::default(),
            head: HeadLaunchConfig::default(),
            training_defaults: TrainingDefaults::default(),
            file: FileCfg::default(),
        }
    }

    /// Load + validate from `path`. A missing file yields `ConfigError::Read` whose
    /// inner `io::Error::kind() == NotFound`, the bootstrap signal to materialize
    /// [`Self::default_for`].
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let invalid = |section: &'static str, err: String| ConfigError::Invalid {
            path: path.display().to_string(),
            msg: format!("launch {section}: {err}"),
        };
        let text = std::fs::read_to_string(path).map_err(|e| read_err(path.display(), e))?;
        // Probe the retired `[stream]` table before the structured parse so the
        // rename hint wins over serde's deny_unknown_fields error.
        if let Ok(value) = toml::from_str::<toml::Value>(&text)
            && value.get("stream").is_some()
        {
            return Err(invalid(
                "stream",
                "the `[stream]` table was retired; rename it to `[api]` \
                 (tcp_bind / uds_path / uds_mode / broadcast_capacity / \
                 tcp_policy / uds_policy), and for the raw inference push \
                 socket add an optional `[output.inference]` \
                 (uds_path / uds_mode). See docs/BUILD.md"
                    .to_string(),
            ));
        }
        let cfg: LaunchConfig = toml::from_str(&text).map_err(|e| parse_err(path.display(), e))?;
        if let Err((id, err)) = cfg.mic.validate() {
            return Err(invalid("mic catalogue", format!("candidate {id}: {err}")));
        }
        if let Err((idx, err)) = cfg.backbone.validate() {
            return Err(invalid(
                "backbone catalogue",
                format!("candidate[{idx}]: {err}"),
            ));
        }
        cfg.api.validate().map_err(|err| invalid("api", err))?;
        cfg.output
            .validate()
            .map_err(|err| invalid("output", err))?;
        // `[api]` UDS and `[output.inference]` must bind distinct paths: a shared path
        // silently unlinks+orphans whichever bound first with no runtime diagnostic.
        if let Some(api_uds) = &cfg.api.uds_path
            && let Some(out) = &cfg.output.inference
            && *api_uds == out.uds_path
        {
            return Err(invalid(
                "output",
                format!(
                    "[output.inference].uds_path {} collides with [api].uds_path; \
                     bind distinct paths",
                    out.uds_path.display()
                ),
            ));
        }
        cfg.head.validate().map_err(|err| invalid("head", err))?;
        // Validate at boot so a typo (epochs = 0) surfaces in the systemd log, not
        // at the first POST /train.
        cfg.training_defaults
            .validate()
            .map_err(|err| invalid("training_defaults", err))?;
        cfg.file.validate().map_err(|err| invalid("file", err))?;
        Ok(cfg)
    }

    /// Persist via tempfile + atomic rename; used by bootstrap when no TOML exists.
    pub fn persist(&self, path: &Path) -> Result<(), ConfigError> {
        write_launch_toml_atomically(path, self)
    }
}

/// Validate a hot-reloaded [`MicPolicy`] against the immutable [`MicCatalogue`],
/// surfacing failures as `ConfigError::Invalid`.
pub fn validate_policy_against_catalogue(
    policy: &MicPolicy,
    catalogue: &MicCatalogue,
    path_for_diag: &Path,
) -> Result<(), ConfigError> {
    policy
        .validate_against(catalogue)
        .map_err(|e: PolicyValidationError| ConfigError::Invalid {
            path: path_for_diag.display().to_string(),
            msg: format!("mic policy: {e}"),
        })
}

fn write_launch_toml_atomically(path: &Path, cfg: &LaunchConfig) -> Result<(), ConfigError> {
    let text = toml::to_string_pretty(cfg)?;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    if !dir.exists() {
        std::fs::create_dir_all(dir).map_err(|e| write_err(dir.display(), e))?;
    }
    crate::file_mgr::fs_atomic::put_atomic(path, text.as_bytes())
        .map_err(crate::config::watcher::file_to_config_err)
}
