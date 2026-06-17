//! Broadcast-lag observability trait letting the API read WS-receiver lag
//! counters without depending on `stream_io`; [`BroadcastLagSnapshot`] lives in
//! `common` (no workspace deps) rather than `status`, which re-exports it.

/// Cumulative broadcast-lag counters: monotonic u64s reset only on daemon
/// restart, counting dropped MESSAGES not lag events (each
/// `RecvError::Lagged(n)` adds `n`, not 1).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BroadcastLagSnapshot {
    pub audio_messages_dropped: u64,
    pub inference_messages_dropped: u64,
}

/// Read-side view of per-stream broadcast-lag counters; [`Self::snapshot`] is
/// wait-free (atomic loads). `Send + Sync + 'static` lets an
/// `Arc<dyn LagSource>` be shared across `api::AppState` handler tasks.
pub trait LagSource: Send + Sync + 'static {
    fn snapshot(&self) -> BroadcastLagSnapshot;
}
