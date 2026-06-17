//! API error type + IntoResponse plumbing. Re-exported by [`crate`].

use crate::common::error::Categorized;
use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use thiserror::Error;
use tokio::task;

/// Top-level API failure shape; every domain error maps to one variant before HTTP rendering.
#[derive(Debug, Error)]
pub enum ApiError {
    #[error("invalid request: {0}")]
    Bad(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("config: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("mic: {0}")]
    Mic(#[from] crate::config::MicError),
    #[error("head load: {0}")]
    Head(#[from] crate::inference::HeadError),
    #[error("head swap: {0}")]
    HeadStore(#[from] crate::common::traits::head_store::HeadStoreError),
    #[error("file: {0}")]
    File(#[from] crate::file_mgr::FileError),
    /// Type-erased `FsService` error; `Categorized::kind` survives the boxing so status mapping is unchanged.
    #[error("fs: {0}")]
    Fs(#[from] crate::file_mgr::FsError),
    #[error("invalid identifier: {0}")]
    Id(#[from] crate::common::ids::IdError),
    #[error("convert: {0}")]
    Convert(#[from] crate::converter::ConvertError),
    #[error("training: {0}")]
    Training(#[from] crate::training::TrainingError),
    #[error("activation: {0}")]
    Activation(#[from] crate::file_mgr::ActivationError),
    #[error("spawn_blocking join: {0}")]
    Join(#[from] task::JoinError),
    #[error("not implemented (Phase {phase})")]
    NotImplemented { phase: &'static str },
    /// Read-your-writes `?min_version=N` with `current < requested`: 425, non-blocking, caller retries.
    #[error("requested min_version={requested}, current={current}: retry after the write settles")]
    TooEarly { requested: u64, current: u64 },
    /// 405 envelope is a per-variant override outside the `ErrorKind` taxonomy.
    #[error("method not allowed: {method} {path}")]
    MethodNotAllowed { method: String, path: String },
}

impl crate::common::error::Categorized for ApiError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        use crate::common::error::ErrorKind::*;
        match self {
            ApiError::Bad(_) => UserInput,
            ApiError::NotFound(_) => NotFound,
            ApiError::NotImplemented { .. } => NotImplemented,
            ApiError::Id(e) => e.kind(),
            ApiError::Config(e) => e.kind(),
            ApiError::Mic(e) => e.kind(),
            ApiError::Head(e) => e.kind(),
            ApiError::HeadStore(e) => e.kind(),
            ApiError::File(e) => e.kind(),
            ApiError::Fs(e) => e.kind(),
            ApiError::Convert(e) => e.kind(),
            ApiError::Training(e) => e.kind(),
            ApiError::Activation(e) => e.kind(),
            ApiError::Join(_) => Internal,
            // status overridden in http_status(); closest kind() fit so consumers don't panic.
            ApiError::TooEarly { .. } => Conflict,
            ApiError::MethodNotAllowed { .. } => UserInput,
        }
    }
}

impl ApiError {
    fn http_status(&self) -> StatusCode {
        // Per-variant overrides; all else routes through kind()->http_status_code() (canonical statuses).
        match self {
            ApiError::TooEarly { .. } => StatusCode::TOO_EARLY,
            ApiError::MethodNotAllowed { .. } => StatusCode::METHOD_NOT_ALLOWED,
            _ => StatusCode::from_u16(self.kind().http_status_code())
                .expect("ErrorKind::http_status_code returns canonical HTTP statuses"),
        }
    }

    fn code(&self) -> &'static str {
        // AnotherTrainRunning gets a dedicated code (vs generic `conflict`), reached via any of three carriers.
        match self {
            ApiError::TooEarly { .. } => "too_early",
            ApiError::MethodNotAllowed { .. } => "method_not_allowed",
            ApiError::File(crate::file_mgr::FileError::AnotherTrainRunning) => {
                "another_train_running"
            }
            ApiError::Training(crate::training::TrainingError::File(
                crate::file_mgr::FileError::AnotherTrainRunning,
            )) => "another_train_running",
            ApiError::Fs(e)
                if matches!(
                    std::error::Error::source(e)
                        .and_then(|s| s.downcast_ref::<crate::file_mgr::FileError>()),
                    Some(crate::file_mgr::FileError::AnotherTrainRunning),
                ) =>
            {
                "another_train_running"
            }
            // `.alpkg` import collision (divergent sha256 for an existing head_id), via File or Convert.
            ApiError::File(crate::file_mgr::FileError::HeadIdCollision { .. }) => {
                "head_id_collision"
            }
            ApiError::Convert(crate::converter::ConvertError::HeadIdCollision(_)) => {
                "head_id_collision"
            }
            _ => self.kind().code_str(),
        }
    }
}

#[derive(Serialize)]
struct ApiErrorBody {
    error: String,
    code: &'static str,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.http_status();
        let code = self.code();
        // Sanitize 5xx bodies (chain leaks fs paths / OS messages) to a generic copy; 4xx full-fidelity; full chain always logged.
        let error = if status.is_server_error() {
            tracing::warn!(
                target: "api",
                code,
                status = status.as_u16(),
                err = %self,
                "5xx response: full error chain logged here; sanitized body sent to client",
            );
            format!("internal error ({code})")
        } else {
            self.to_string()
        };
        let body = ApiErrorBody { error, code };
        (status, Json(body)).into_response()
    }
}
