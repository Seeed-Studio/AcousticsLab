//! Forward-only JSONL log paging: full re-scan returning a bounded page of
//! [`LogEvent`]s with `seq > after_seq` (per-job files stay ~10-15 events; no
//! index). Malformed/truncated lines are silently skipped without advancing
//! the cursor (the broadcast channel is authoritative; JSONL is a backstop).
//! Missing file returns an empty page echoing `after_seq` (no 404).

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

/// Per-call default page size when the caller omits `?limit=`.
pub const DEFAULT_LOG_PAGE_LIMIT: usize = 200;

/// Hard ceiling on `?limit=`; bounds buffering in the blocking scan.
pub const MAX_LOG_PAGE_LIMIT: usize = 1000;

/// Per-line byte cap: producers enforce none, so an unterminated multi-GiB
/// line would OOM the pager; oversize lines are skipped like malformed JSON.
/// 1 MiB is well above the largest field emitted (~256 KiB).
pub const MAX_LOG_LINE_BYTES: usize = 1 << 20;

/// One JSONL line, deserialised forgivingly: only `seq`/`at` are typed; unknown
/// producer fields land in `payload` rather than failing the parse.
#[derive(Debug, Serialize, Deserialize)]
pub struct LogEvent {
    pub seq: u64,
    pub at: String,
    #[serde(flatten)]
    pub payload: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct LogPageResp {
    pub events: Vec<LogEvent>,
    /// Next `?after_seq=`: the last event's `seq`, else echoes input.
    pub next_after_seq: u64,
}

/// Read one bounded page from `path`; `limit` is silently clamped to `[1, MAX_LOG_PAGE_LIMIT]`.
pub fn read_jsonl_page(path: &Path, after_seq: u64, limit: usize) -> io::Result<LogPageResp> {
    use std::io::{BufRead, BufReader, Read};
    let limit = limit.clamp(1, MAX_LOG_PAGE_LIMIT);
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Ok(LogPageResp {
                events: Vec::new(),
                next_after_seq: after_seq,
            });
        }
        Err(e) => return Err(e),
    };
    let mut events = Vec::with_capacity(limit.min(64));
    let mut reader = BufReader::new(file);
    let mut next_after_seq = after_seq;
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    loop {
        buf.clear();
        // Read-bound at cap+1 (+1 captures an exactly-at-cap line's newline):
        // bare `read_until` would buffer the whole unterminated line before any
        // length check, the OOM the cap exists to prevent.
        let read = {
            let mut limited = (&mut reader).take(MAX_LOG_LINE_BYTES as u64 + 1);
            limited.read_until(b'\n', &mut buf)?
        };
        if read == 0 {
            break;
        }
        // Ceiling hit without a newline => line exceeds the cap; drain the rest
        // (cursor advances past it, so later events on this pass still read).
        if read > MAX_LOG_LINE_BYTES && buf.last() != Some(&b'\n') {
            skip_rest_of_line(&mut reader)?;
            continue;
        }
        while buf.last().is_some_and(|c| matches!(c, b'\n' | b'\r')) {
            buf.pop();
        }
        if buf.is_empty() {
            continue;
        }
        let evt: LogEvent = match serde_json::from_slice(&buf) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if evt.seq <= after_seq {
            continue;
        }
        next_after_seq = evt.seq;
        events.push(evt);
        if events.len() >= limit {
            break;
        }
    }
    Ok(LogPageResp {
        events,
        next_after_seq,
    })
}

/// Discard the rest of the current line (through the next `\n` or EOF) without
/// buffering, via `fill_buf`/`consume`, so the drain cannot re-introduce the
/// OOM the read-side cap guards against.
fn skip_rest_of_line<R: std::io::BufRead>(reader: &mut R) -> io::Result<()> {
    loop {
        let (found_newline, consume) = {
            let available = reader.fill_buf()?;
            if available.is_empty() {
                return Ok(()); // EOF mid-line.
            }
            match available.iter().position(|&b| b == b'\n') {
                Some(i) => (true, i + 1),
                None => (false, available.len()),
            }
        };
        reader.consume(consume);
        if found_newline {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_file(path: &Path, bytes: &[u8]) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    /// Oversize line skipped without OOM; events before AND after it returned.
    #[test]
    fn oversize_line_skipped_and_following_events_still_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(br#"{"seq":1,"at":"t1"}"#);
        bytes.push(b'\n');
        bytes.extend_from_slice(&vec![b'x'; MAX_LOG_LINE_BYTES + 1024]);
        bytes.push(b'\n');
        bytes.extend_from_slice(br#"{"seq":2,"at":"t2"}"#);
        bytes.push(b'\n');
        write_file(&path, &bytes);

        let resp = read_jsonl_page(&path, 0, 100).unwrap();
        let seqs: Vec<u64> = resp.events.iter().map(|e| e.seq).collect();
        assert_eq!(seqs, vec![1, 2], "oversize line skipped; real events read");
        assert_eq!(resp.next_after_seq, 2);
    }

    /// A line exactly at the cap is kept, not treated as oversize.
    #[test]
    fn at_cap_line_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        let prefix = br#"{"seq":7,"at":"t","pad":""#;
        let suffix = br#""}"#;
        let pad = MAX_LOG_LINE_BYTES - prefix.len() - suffix.len();
        let mut line = Vec::new();
        line.extend_from_slice(prefix);
        line.extend_from_slice(&vec![b'a'; pad]);
        line.extend_from_slice(suffix);
        assert_eq!(line.len(), MAX_LOG_LINE_BYTES);
        line.push(b'\n');
        write_file(&path, &line);

        let resp = read_jsonl_page(&path, 0, 100).unwrap();
        assert_eq!(resp.events.len(), 1);
        assert_eq!(resp.events[0].seq, 7);
    }
}
