//! Lifecycle Row 4 (PLACEHOLDER): an inference-engine-task panic MUST be observed
//! by the daemon's supervisor, which SHOULD then either log + exit non-zero (external
//! supervisor restarts) or respawn the task; the daemon MUST NOT silently continue
//! with a dead inference loop.
//!
//! Stubbed because asserting this needs the supervisor's per-task `RestartPolicy`,
//! which does not yet exist: `catch_unwind` only wraps the audio-capture closure, so
//! an engine-task panic bubbles to the join handle untyped. The one `#[ignore]`'d
//! test keeps the gap grep-able and pins the contract for a future implementer.

#![allow(clippy::disallowed_methods)]

#[test]
#[ignore = "supervisor RestartPolicy not yet implemented; see file-level doc"]
fn lifecycle_row4_inference_task_panic_triggers_supervised_restart() {
    // When the supervisor lands: boot long-running, inject an engine panic (reserved
    // `panic-inject` feature), then assert the daemon exits non-zero within the restart
    // budget (logging `panic`+inference) or respawns and resumes -- never a silent dead
    // loop (no heartbeat past T+5s).
    panic!("Row 4 test is a placeholder; remove #[ignore] when supervisor RestartPolicy lands");
}
