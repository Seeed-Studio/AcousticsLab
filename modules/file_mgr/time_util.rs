//! Centralized RFC3339 wall-clock helper so every workspace write stamps the
//! same format; sentinel fallbacks are defensive and never expected in production.

use std::time::SystemTime;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

pub(crate) const RFC3339_SENTINEL: &str = "1970-01-01T00:00:00Z";

fn format_rfc3339(t: OffsetDateTime) -> String {
    t.format(&Rfc3339)
        .unwrap_or_else(|_| String::from(RFC3339_SENTINEL))
}

/// Current UTC wall-clock formatted as RFC3339. Never panics.
pub fn now_rfc3339() -> String {
    format_rfc3339(OffsetDateTime::now_utc())
}

/// Parse an RFC3339 timestamp into its instant for chronological comparison:
/// `Rfc3339`'s variable-width fractional seconds make a lexical compare
/// non-monotonic within a second (`'Z'` > `'.'` > digits), so compare instants.
/// Unparseable yields [`OffsetDateTime::UNIX_EPOCH`] (ranks oldest); callers
/// that can collide on equal/unparseable instants add their own mtime/id
/// tiebreak (see `recovery.rs`).
pub fn parse_rfc3339_or_epoch(s: &str) -> OffsetDateTime {
    OffsetDateTime::parse(s, &Rfc3339).unwrap_or(OffsetDateTime::UNIX_EPOCH)
}

/// Format a [`SystemTime`] as an RFC3339 UTC string; out-of-range timestamps
/// fall back to `RFC3339_SENTINEL` (filesystem nonsense, not a daemon bug).
pub fn rfc3339_from(t: SystemTime) -> String {
    // `OffsetDateTime::from(SystemTime)` panics beyond year ±9999, so a garbage
    // mtime would abort the handler instead of hitting the sentinel; go through
    // the checked `from_unix_timestamp_nanos` (nanos fit i128 for any SystemTime).
    let nanos: i128 = match t.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => d.as_nanos() as i128,
        Err(e) => -(e.duration().as_nanos() as i128),
    };
    match OffsetDateTime::from_unix_timestamp_nanos(nanos) {
        Ok(dt) => format_rfc3339(dt),
        Err(_) => String::from(RFC3339_SENTINEL),
    }
}

/// [`rfc3339_from`] for a `Result<SystemTime, _>` (e.g. `Metadata::modified`); Err maps to the sentinel.
pub(crate) fn rfc3339_from_result<E>(t: Result<SystemTime, E>) -> String {
    match t {
        Ok(t) => rfc3339_from(t),
        Err(_) => String::from(RFC3339_SENTINEL),
    }
}
