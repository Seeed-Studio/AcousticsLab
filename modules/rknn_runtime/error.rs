//! Every fallible `rknn_*` C call is checked against `RKNN_SUCC` and wrapped in
//! [`Error::Rknn`] with function name, raw code, and context -- no silent
//! failures, diagnosable from on-device logs. The lone exception is
//! `rknn_destroy` on `Session`'s `Drop` path, which is still success-checked but
//! logs a `tracing::warn!` rather than returning, since `Drop` cannot yield a
//! `Result`.

use crate::rknn_runtime::ffi::LoadError;
use std::ffi::c_int;

pub type Result<T> = std::result::Result<T, Error>;

#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// Library open failed: missing file, wrong arch, or broken ELF.
    LibraryLoad {
        path: std::path::PathBuf,
        source: libloading::Error,
    },

    /// Symbol missing -- usually wrong library variant (`librknnrt` vs `librknnmrt`) or older SDK.
    SymbolNotFound {
        name: &'static str,
        source: libloading::Error,
    },

    /// A C call returned a non-success `RKNN_ERR_*` code.
    Rknn {
        name: &'static str,
        code: c_int,
        context: &'static str,
    },

    /// Buffer length precondition violation (distinct from C-layer `Rknn`).
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    /// Tensor/version string not valid UTF-8; preserved instead of panicking.
    Utf8(std::str::Utf8Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::LibraryLoad { path, source } => {
                write!(f, "load library {}: {source}", path.display())
            }
            Error::SymbolNotFound { name, source } => {
                write!(f, "resolve symbol {name}: {source}")
            }
            Error::Rknn {
                name,
                code,
                context,
            } => {
                write!(
                    f,
                    "{name} ({context}) failed: {code} ({})",
                    rknn_error_name(*code),
                )
            }
            Error::ShapeMismatch {
                what,
                expected,
                got,
            } => {
                write!(f, "{what} shape mismatch: expected {expected}, got {got}")
            }
            Error::Utf8(e) => write!(f, "utf-8 decode: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::LibraryLoad { source, .. } | Error::SymbolNotFound { source, .. } => {
                Some(source)
            }
            Error::Utf8(e) => Some(e),
            _ => None,
        }
    }
}

impl From<LoadError> for Error {
    fn from(e: LoadError) -> Self {
        match e {
            LoadError::Library { path, source } => Error::LibraryLoad { path, source },
            LoadError::Symbol { name, source } => Error::SymbolNotFound { name, source },
        }
    }
}

impl From<std::str::Utf8Error> for Error {
    fn from(e: std::str::Utf8Error) -> Self {
        Error::Utf8(e)
    }
}

/// Map a known `RKNN_ERR_*` code to its header name, else `"UNKNOWN"`.
pub(crate) fn rknn_error_name(code: c_int) -> &'static str {
    use crate::rknn_runtime::sys;
    match code {
        x if x == sys::RKNN_SUCC as c_int => "RKNN_SUCC",
        sys::RKNN_ERR_FAIL => "RKNN_ERR_FAIL",
        sys::RKNN_ERR_TIMEOUT => "RKNN_ERR_TIMEOUT",
        sys::RKNN_ERR_DEVICE_UNAVAILABLE => "RKNN_ERR_DEVICE_UNAVAILABLE",
        sys::RKNN_ERR_MALLOC_FAIL => "RKNN_ERR_MALLOC_FAIL",
        sys::RKNN_ERR_PARAM_INVALID => "RKNN_ERR_PARAM_INVALID",
        sys::RKNN_ERR_MODEL_INVALID => "RKNN_ERR_MODEL_INVALID",
        sys::RKNN_ERR_CTX_INVALID => "RKNN_ERR_CTX_INVALID",
        sys::RKNN_ERR_INPUT_INVALID => "RKNN_ERR_INPUT_INVALID",
        sys::RKNN_ERR_OUTPUT_INVALID => "RKNN_ERR_OUTPUT_INVALID",
        sys::RKNN_ERR_DEVICE_UNMATCH => "RKNN_ERR_DEVICE_UNMATCH",
        sys::RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL => "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL",
        sys::RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION => {
            "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION"
        }
        sys::RKNN_ERR_TARGET_PLATFORM_UNMATCH => "RKNN_ERR_TARGET_PLATFORM_UNMATCH",
        _ => "UNKNOWN",
    }
}

/// `Err(Error::Rknn)` unless `code == RKNN_SUCC`; `context` describes the
/// operation, e.g. `"query input 0 attr"`.
#[inline]
pub(crate) fn check(name: &'static str, context: &'static str, code: c_int) -> Result<()> {
    if code == crate::rknn_runtime::sys::RKNN_SUCC as c_int {
        Ok(())
    } else {
        Err(Error::Rknn {
            name,
            code,
            context,
        })
    }
}
