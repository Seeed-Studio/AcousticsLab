//! Shared error taxonomy: domain modules impl [`Categorized`] so the API layer maps to HTTP via a
//! single match on [`ErrorKind`], adding no per-module arms.

use std::fmt;

use serde::Serialize;

/// Coarse category of a domain error. Variant order is load-bearing: ascending severity (UserInput
/// recoverable, Internal not) so [`Ord`] lets handlers prefer the most severe of chained errors.
#[derive(Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorKind {
    UserInput,
    NotFound,
    Conflict,
    NotImplemented,
    Unavailable,
    Internal,
}

impl ErrorKind {
    /// HTTP status code; `u16` (not an http type) keeps `common` dep-free, and every value is a
    /// canonical status so the API layer's `StatusCode::from_u16(..)` cannot panic.
    pub const fn http_status_code(self) -> u16 {
        match self {
            ErrorKind::UserInput => 400,
            ErrorKind::NotFound => 404,
            ErrorKind::Conflict => 409,
            ErrorKind::NotImplemented => 501,
            ErrorKind::Unavailable => 503,
            ErrorKind::Internal => 500,
        }
    }

    /// Stable wire identifier for the API response `code` field, independent of the source module.
    pub const fn code_str(self) -> &'static str {
        match self {
            ErrorKind::UserInput => "bad_request",
            ErrorKind::NotFound => "not_found",
            ErrorKind::Conflict => "conflict",
            ErrorKind::NotImplemented => "not_implemented",
            ErrorKind::Unavailable => "unavailable",
            ErrorKind::Internal => "internal",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code_str())
    }
}

pub trait Categorized {
    fn kind(&self) -> ErrorKind;
}

/// Operator-vs-internal axis on terminal job-failure events, derived from [`ErrorKind`] so
/// frontends colour the failure card off this enum instead of parsing free-form error strings.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    OperatorFixable,
    Internal,
}

impl From<ErrorKind> for Severity {
    fn from(kind: ErrorKind) -> Self {
        // No `_` arm: a new ErrorKind must be classified explicitly.
        match kind {
            ErrorKind::UserInput => Severity::OperatorFixable,
            ErrorKind::NotFound
            | ErrorKind::Conflict
            | ErrorKind::NotImplemented
            | ErrorKind::Unavailable
            | ErrorKind::Internal => Severity::Internal,
        }
    }
}

/// Stringify a `catch_unwind` panic payload (`&'static str`/`String`, else a fixed marker).
pub fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_status_codes_are_canonical() {
        assert_eq!(ErrorKind::UserInput.http_status_code(), 400);
        assert_eq!(ErrorKind::NotFound.http_status_code(), 404);
        assert_eq!(ErrorKind::Conflict.http_status_code(), 409);
        assert_eq!(ErrorKind::NotImplemented.http_status_code(), 501);
        assert_eq!(ErrorKind::Unavailable.http_status_code(), 503);
        assert_eq!(ErrorKind::Internal.http_status_code(), 500);
    }

    #[test]
    fn code_str_round_trips_through_display() {
        for k in [
            ErrorKind::UserInput,
            ErrorKind::NotFound,
            ErrorKind::Conflict,
            ErrorKind::NotImplemented,
            ErrorKind::Unavailable,
            ErrorKind::Internal,
        ] {
            assert_eq!(format!("{k}"), k.code_str());
        }
    }

    /// Guards the ascending-severity ordering callers rely on via `.max_by_key(.kind())`.
    #[test]
    fn ord_reflects_severity() {
        assert!(ErrorKind::UserInput < ErrorKind::Internal);
        assert!(ErrorKind::NotFound < ErrorKind::Internal);
        assert!(ErrorKind::Conflict < ErrorKind::Unavailable);
    }

    /// Pins the two-tone collapse the job-failure wire schemas bake in; a regression silently
    /// downgrades frontend hint-card colouring.
    #[test]
    fn severity_from_errorkind_collapses_to_two_tones() {
        assert_eq!(
            Severity::from(ErrorKind::UserInput),
            Severity::OperatorFixable
        );
        for k in [
            ErrorKind::NotFound,
            ErrorKind::Conflict,
            ErrorKind::NotImplemented,
            ErrorKind::Unavailable,
            ErrorKind::Internal,
        ] {
            assert_eq!(
                Severity::from(k),
                Severity::Internal,
                "kind {k:?} must collapse to Internal",
            );
        }
    }

    /// Pins the snake_case wire form the frontend matches on.
    #[test]
    fn severity_serializes_snake_case() {
        let v = serde_json::to_value(Severity::OperatorFixable).unwrap();
        assert_eq!(v, serde_json::json!("operator_fixable"));
        let v = serde_json::to_value(Severity::Internal).unwrap();
        assert_eq!(v, serde_json::json!("internal"));
    }
}
