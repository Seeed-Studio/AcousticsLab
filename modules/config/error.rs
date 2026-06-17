use thiserror::Error;

/// Typed per-section validation failure for [`crate::config::Config`], so the hot-reload
/// callback matches on category not log text. Boot-only validators bypass this into `Invalid`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConfigValidationError {
    #[error("inference: {0}")]
    Inference(String),
    /// User reload callback rejected (typically cross-validation against the launch catalogue).
    #[error("rejected by reload callback: {0}")]
    Callback(String),
}

/// Failure shapes from config load / mutate / persist; mapped to HTTP statuses via `Categorized`.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("write {path}: {source}")]
    Write {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parse {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: toml::de::Error,
    },
    #[error("serialize: {0}")]
    Serialize(#[from] toml::ser::Error),
    #[error("watcher: {0}")]
    Watcher(#[from] notify::Error),
    #[error("persist: {0}")]
    Persist(#[from] tempfile::PersistError),
    /// A sub-section's `validate()` rejected; raised wherever a config is materialized (boot `from_value`/`load`, runtime `validate_persist_swap` behind `mutate_then`/`CellGuard::commit`, launch `load` and mic-policy cross-validation) so an invalid config fails loud instead of clamping silently.
    #[error("invalid config {path}: {msg}")]
    Invalid { path: String, msg: String },
    /// Re-entered `mutate_then` on the same thread; surfaced explicitly since the non-reentrant `parking_lot::Mutex` would otherwise silently deadlock.
    #[error("re-entrant config mutate")]
    ReentrantMutate,
    /// `watch()` debounce-thread spawn failed; distinct from a read error to avoid mis-attribution.
    #[error("spawn config-reload thread for {path}: {source}")]
    ThreadSpawn {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

impl crate::common::error::Categorized for ConfigError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            // Programmer error, not operator-fixable.
            ConfigError::ReentrantMutate => Internal,
            // Operator-fixable: edited TOML failed parse or `validate()`.
            ConfigError::Parse { .. } | ConfigError::Invalid { .. } => UserInput,
            ConfigError::Read { .. }
            | ConfigError::Write { .. }
            | ConfigError::Serialize(_)
            | ConfigError::Watcher(_)
            | ConfigError::Persist(_)
            | ConfigError::ThreadSpawn { .. } => Internal,
        }
    }
}

/// `path` is `impl Display` so callers pass `Path::display()` without an intermediate `String`.
pub(crate) fn read_err(path: impl std::fmt::Display, source: std::io::Error) -> ConfigError {
    ConfigError::Read {
        path: path.to_string(),
        source,
    }
}

pub(crate) fn write_err(path: impl std::fmt::Display, source: std::io::Error) -> ConfigError {
    ConfigError::Write {
        path: path.to_string(),
        source,
    }
}

pub(crate) fn parse_err(path: impl std::fmt::Display, source: toml::de::Error) -> ConfigError {
    ConfigError::Parse {
        path: path.to_string(),
        source,
    }
}
