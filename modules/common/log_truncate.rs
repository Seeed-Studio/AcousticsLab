//! Bounded UTF-8 truncation for operator-supplied log lines; distinct from
//! the job registry's `append_log` cap/suffix (`" ... [truncated]"`).

/// Per-event log-line cap in bytes; past this a JSONL line stops being
/// hand-scannable with `head`/`jq`.
pub const MAX_LOG_LINE_BYTES: usize = 8 * 1024;

/// Truncate `m` to at most [`MAX_LOG_LINE_BYTES`], snapping down to a UTF-8 char
/// boundary so a straddling codepoint is dropped not corrupted; appends
/// `"...[truncated]"` on truncation.
pub fn truncate_log_message(m: &str) -> String {
    if m.len() <= MAX_LOG_LINE_BYTES {
        return m.to_string();
    }
    let mut idx = MAX_LOG_LINE_BYTES;
    while idx > 0 && !m.is_char_boundary(idx) {
        idx -= 1;
    }
    let mut s = m[..idx].to_string();
    s.push_str("...[truncated]");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_message_is_unchanged() {
        let m = "hello world";
        assert_eq!(truncate_log_message(m), "hello world");
    }

    #[test]
    fn exact_cap_is_unchanged() {
        let m: String = std::iter::repeat_n('a', MAX_LOG_LINE_BYTES).collect();
        let out = truncate_log_message(&m);
        assert_eq!(out, m);
        assert!(!out.contains("...[truncated]"));
    }

    /// Naive `&str[..cap]` panics when the cap lands inside a 4-byte codepoint.
    #[test]
    fn four_byte_codepoint_at_boundary_snaps_down() {
        let mut m = String::with_capacity(MAX_LOG_LINE_BYTES + 4);
        for _ in 0..(MAX_LOG_LINE_BYTES - 1) {
            m.push('a');
        }
        m.push('\u{1F600}');
        let out = truncate_log_message(&m);
        assert!(
            out.ends_with("...[truncated]"),
            "expected truncation marker; got len={}",
            out.len(),
        );
        let body = out.trim_end_matches("...[truncated]");
        assert!(
            body.is_char_boundary(body.len()),
            "body must end on a char boundary",
        );
        assert!(
            body.len() <= MAX_LOG_LINE_BYTES,
            "body must not exceed cap; got {}",
            body.len(),
        );
    }

    /// Leading ASCII byte forces the even cap to land mid-`é`; without it every
    /// even offset in an all-`é` string is a boundary and the snap-down never runs.
    #[test]
    fn two_byte_codepoint_at_boundary_snaps_down() {
        let mut s = String::from("a");
        while s.len() < MAX_LOG_LINE_BYTES + 1024 {
            s.push('é');
        }
        let truncated = truncate_log_message(&s);
        let body = truncated.trim_end_matches("...[truncated]");
        assert_eq!(body.len(), MAX_LOG_LINE_BYTES - 1);
        assert!(
            body.is_char_boundary(body.len()),
            "body must end on a char boundary",
        );
        assert!(
            truncated.len() <= MAX_LOG_LINE_BYTES + b"...[truncated]".len(),
            "truncated len {} > cap + suffix",
            truncated.len(),
        );
        assert!(truncated.ends_with("...[truncated]"));
        let _ = truncated.chars().count();
    }
}
