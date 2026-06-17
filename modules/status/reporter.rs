//! `StatusReporter` lets the api crate hold `Arc<dyn StatusReporter>` without the
//! concrete monitor; daemon-only `register()` stays on [`crate::status::StatusMonitor`].

use crate::status::{BroadcastLagSnapshot, StatusMonitor, StatusSnapshot};

pub trait StatusReporter: Send + Sync + std::fmt::Debug {
    /// Wait-free: reads the `ArcSwap<MetricsSnapshot>` the background sampler
    /// publishes, so the request path never touches sysinfo nor `spawn_blocking`.
    fn snapshot(&self, broadcast_lags: BroadcastLagSnapshot) -> StatusSnapshot;
}

impl StatusReporter for StatusMonitor {
    fn snapshot(&self, broadcast_lags: BroadcastLagSnapshot) -> StatusSnapshot {
        StatusMonitor::snapshot(self, broadcast_lags)
    }
}

#[cfg(test)]
const _: fn() = || {
    fn assert_obj_safe<T: ?Sized>() {}
    assert_obj_safe::<dyn StatusReporter>();
};
