//! Producer-side per-job JSONL writer over `E: Serialize`, backing
//! `<workspace>/<subtree>/<job_id>.jsonl` (monotonic per-file `seq`),
//! round-trippable by [`super::log_page`].
//!
//! Per-line `fsync` is skipped (10x cost): a crash loses at most the
//! kernel-writeback window of trailing events; tolerable only because
//! `open` fsyncs the subtree dir, making the file's existence durable.

use std::io;
use std::marker::PhantomData;
use std::path::Path;

use serde::Serialize;

use crate::common::ids::JobId;

/// Per-job JSONL writer over the producer's event type. Not thread-safe;
/// callers wrap in `Arc<Mutex<_>>` to share a log.
///
/// `_marker` is `PhantomData<fn(&E)>` not `PhantomData<E>`: the struct only
/// borrows `E`, so the fn-pointer form avoids inheriting `E`'s
/// `Send`/`Sync`/drop-check.
#[derive(Debug)]
pub struct JsonlEventLog<E: Serialize> {
    file: std::fs::File,
    seq: u64,
    /// Bytes successfully appended; seeded from file size at `open` and
    /// cached so the truncate path avoids a per-event `fstat(2)`.
    written_len: u64,
    /// Sticky: set when `emit`'s truncate-on-Err `set_len` itself fails, so
    /// the kernel EOF has drifted past `written_len` and a later
    /// `set_len(written_len)` would silently truncate committed events.
    /// Thereafter every `emit` returns Err and on-disk JSONL is left intact;
    /// per-handle, so the next `open` re-reads `written_len` from `metadata()`.
    poisoned: bool,
    _marker: PhantomData<fn(&E)>,
}

#[derive(Debug, Serialize)]
struct Envelope<'a, E: Serialize> {
    seq: u64,
    at: String,
    #[serde(flatten)]
    event: &'a E,
}

impl<E: Serialize> JsonlEventLog<E> {
    /// Open the job's `.jsonl` for append (creating the subtree dir),
    /// then enforce [`super::LOG_RETENTION_KEEP_COUNT`] over the dir's
    /// `.jsonl` siblings.
    pub fn open(workspace_dir: &Path, subtree: &str, job_id: JobId) -> io::Result<Self> {
        let dir = workspace_dir.join(subtree);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{job_id}.jsonl"));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        // Exempt the just-opened path so sub-ms clock-skew / fs-tiebreak
        // races can't pick the producer's own log for deletion.
        super::log_retention::enforce_keep_last_n_excluding(
            &dir,
            super::LOG_RETENTION_KEEP_COUNT,
            Some(&path),
        );
        // fsync the parent dir so the new dirent and the sweep's unlinks are
        // durable; else a power loss loses the whole log, not just its tail.
        super::validate::fsync_dir(&dir)?;
        // Seed from on-disk size to cover a log that survived a restart.
        let written_len = file.metadata()?.len();
        Ok(Self {
            file,
            seq: 0,
            written_len,
            poisoned: false,
            _marker: PhantomData,
        })
    }

    /// Append one envelope line. `seq` commits only after serialise,
    /// `write_all`, AND `flush` succeed, so on failure `self.seq` is unchanged
    /// and stays in lockstep with the caller's. A partial `write_all` leaves
    /// kernel-appended bytes that, since append never overwrites, the next
    /// emit would write past and duplicate the `seq`; `set_len(pre_write_len)`
    /// chops the torn tail so the retry reuses the seq against a clean EOF.
    /// If that `set_len` fails, `poisoned` is set (see field doc).
    pub fn emit(&mut self, event: &E) -> io::Result<()> {
        use std::io::Write as _;
        if self.poisoned {
            return Err(io::Error::other(
                "jsonl_event_log: handle poisoned by a prior truncate-on-Err failure; \
                 the on-disk EOF has drifted from the cached written_len, so further \
                 emits would risk silently truncating previously-committed events. \
                 Re-open the log to recover (typically via daemon restart).",
            ));
        }
        let candidate_seq = self.seq.saturating_add(1);
        let line = Envelope {
            seq: candidate_seq,
            at: super::now_rfc3339(),
            event,
        };
        let mut bytes = serde_json::to_vec(&line).map_err(io::Error::other)?;
        bytes.push(b'\n');
        let pre_write_len = self.written_len;
        if let Err(e) = self.file.write_all(&bytes) {
            if self.file.set_len(pre_write_len).is_err() {
                self.poisoned = true;
            }
            return Err(e);
        }
        if let Err(e) = self.file.flush() {
            // No-op on std::fs::File today; truncate symmetrically so a
            // future buffered wrapper's partial flush stays consistent.
            if self.file.set_len(pre_write_len).is_err() {
                self.poisoned = true;
            }
            return Err(e);
        }
        self.seq = candidate_seq;
        self.written_len = self.written_len.saturating_add(bytes.len() as u64);
        Ok(())
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    #[cfg(test)]
    pub(crate) fn current_seq(&self) -> u64 {
        self.seq
    }
}

#[cfg(test)]
mod tests {
    // Raw `std::fs::*` fixtures: atomic-write guards don't apply to
    // append-only log files.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use serde::Serialize;

    #[derive(Debug, Serialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Probe {
        First,
        Second { value: u32 },
    }

    /// Pins on-disk shape: one line per `emit`, `seq` from 1, RFC3339 `at`,
    /// event fields flattened into the same object.
    #[test]
    fn emit_writes_envelope_with_seq_at_and_flattened_event() {
        let tmp = tempfile::tempdir().unwrap();
        let job_id = JobId::new();
        let mut log =
            JsonlEventLog::<Probe>::open(tmp.path(), "test_logs", job_id).expect("open log");
        log.emit(&Probe::First).expect("first emit");
        log.emit(&Probe::Second { value: 42 }).expect("second emit");
        assert_eq!(log.current_seq(), 2);
        drop(log);

        let path = tmp.path().join("test_logs").join(format!("{job_id}.jsonl"));
        let body = std::fs::read_to_string(&path).expect("read log");
        let lines: Vec<_> = body.lines().collect();
        assert_eq!(lines.len(), 2, "one JSONL line per emit() call");

        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["seq"], 1);
        assert_eq!(first["kind"], "first");
        assert!(
            first["at"].as_str().unwrap().ends_with('Z'),
            "RFC3339 with Z suffix",
        );

        let second: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["seq"], 2);
        assert_eq!(second["kind"], "second");
        assert_eq!(second["value"], 42);
        assert!(second["at"].as_str().is_some());
    }

    #[test]
    fn open_creates_subtree_dir_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let subtree_dir = tmp.path().join("nested_logs");
        assert!(!subtree_dir.exists(), "subtree dir absent pre-open");
        let job_id = JobId::new();
        let _log = JsonlEventLog::<Probe>::open(tmp.path(), "nested_logs", job_id)
            .expect("open creates dir");
        assert!(subtree_dir.is_dir(), "subtree dir materialised on open");
    }

    /// Open runs retention: stale `.jsonl` siblings beyond the cap are
    /// unlinked while the new freshest log survives.
    #[test]
    fn open_enforces_keep_last_n() {
        let tmp = tempfile::tempdir().unwrap();
        let workspace_dir = tmp.path();
        let dir = workspace_dir.join("test_logs");
        std::fs::create_dir_all(&dir).unwrap();
        let cap = super::super::LOG_RETENTION_KEEP_COUNT;
        // cap+1 stale logs, mtime-backdated so the new log out-freshes all.
        let mut stale_paths = Vec::with_capacity(cap + 1);
        for i in 0..=cap {
            let p = dir.join(format!("00000000-0000-4000-8000-{i:012x}.jsonl"));
            std::fs::write(&p, b"{}\n").unwrap();
            let backdate = std::time::SystemTime::now()
                .checked_sub(std::time::Duration::from_secs((1000 - i as u64) * 60))
                .expect("backdate");
            let secs = backdate
                .duration_since(std::time::UNIX_EPOCH)
                .expect("post-epoch")
                .as_secs();
            let ft = filetime::FileTime::from_unix_time(secs as i64, 0);
            filetime::set_file_mtime(&p, ft).expect("set mtime");
            stale_paths.push(p);
        }
        let job_id = JobId::new();
        let _log =
            JsonlEventLog::<Probe>::open(workspace_dir, "test_logs", job_id).expect("open log");

        let new_path = dir.join(format!("{job_id}.jsonl"));
        assert!(new_path.is_file(), "new log survives");
        // cap+1 stale + 1 new = cap+2; keeping cap unlinks the 2 oldest.
        assert!(!stale_paths[0].exists(), "oldest stale unlinked");
        assert!(!stale_paths[1].exists(), "second-oldest stale unlinked");
        assert!(stale_paths[2].exists(), "third-oldest survives the cap");
        let remaining = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .ok()
                    .and_then(|e| e.file_name().into_string().ok())
                    .is_some_and(|n| n.ends_with(".jsonl"))
            })
            .count();
        assert_eq!(remaining, cap, "exactly `cap` jsonl files remain");
    }
}
