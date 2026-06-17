//! Lifecycle Row 5 (PLACEHOLDER): when repeated opus encoder failures (allocator
//! pressure, malformed input surviving the NaN scan, libopus hiccup) exhaust the
//! per-task restart budget, the daemon MUST deterministically either mark the `opus`
//! subsystem degraded (rest of daemon up, the `audio` WebSocket stream stops, inference/mic stay
//! healthy) or exit non-zero; the hard-fail-vs-degrade policy is still unspecified.
//!
//! Stub because the supervisor `RestartPolicy` machinery doesn't exist yet, so there's
//! no per-task budget to exhaust: the opus task is a single `tokio::spawn` of `run` with
//! no restart wrapper, and a panic inside `run` reaches the drain registry as a JoinError
//! without synthesizing a degraded opus heartbeat. One `#[ignore]`d test keeps the gap
//! grep-able and the contract visible until the supervisor + policy decision land.

#![allow(clippy::disallowed_methods)]

#[test]
#[ignore = "supervisor RestartPolicy + opus restart-budget not yet implemented; see file-level doc"]
fn lifecycle_row5_opus_task_policy_exhausted_transitions_subsystem() {
    // Intent once the supervisor lands: inject restart-budget-many consecutive opus encoder
    // failures via the reserved `panic-inject` Cargo-feature hook, assert the `opus` heartbeat
    // goes `degraded` (or non-zero exit per the policy) within the budget window, and assert
    // other subsystems stay healthy -- the load-bearing non-cascading claim.
    panic!(
        "Row 5 test is a placeholder; remove #[ignore] when supervisor RestartPolicy + opus \
         restart budget land"
    );
}
