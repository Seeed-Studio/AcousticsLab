//! Boot with an empty backbone catalogue (working mock mic intact) so the
//! inference subsystem takes the involuntary-skip path and reports
//! `degraded_reason="no_backbone"` while staying `healthy=true` (degraded and
//! unhealthy are orthogonal axes). Asserts `--check` exits 0, inference is
//! degraded with that reason, and no other subsystem cascade-degrades.
//!
//! Distinct from `--no-inference`, the VOLUNTARY skip that still reports plain
//! healthy; Row 2 is specifically the involuntary case.

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, launch_check_mode};

/// Inline launch fixture: any working mock mic plus an empty
/// `[backbone].candidates` to trigger the empty-backbone boot branch.
const NO_BACKBONE_LAUNCH_TOML: &str = r#"
[[mic.candidates]]
id = "default-mock"
channels = [0]
source = { kind = "mock", period_size = 512, sample_rate = 44100, waveforms = [{ kind = "sine", freq_hz = 1000.0, amplitude = 0.25 }] }

[backbone]
candidates = []

[api]
tcp_bind = "127.0.0.1:0"
broadcast_capacity = 64
"#;

/// Spawns the daemon WITHOUT `--no-inference` so the empty backbone forces the
/// involuntary-skip path, then asserts that degraded contract.
#[tokio::test]
async fn lifecycle_row2_no_backbone_inference_degraded() {
    let profile = CheckProfile {
        check_seconds: 3,
        mock_audio: false, // use the launch.toml mic catalogue
        no_inference: false,
        timeout: Duration::from_secs(15),
        tcp_bind: "127.0.0.1:0".into(),
        launch_toml_override: Some(NO_BACKBONE_LAUNCH_TOML.into()),
        extra_args: Vec::new(),
        cwd_override: None,
    };

    let run = launch_check_mode(profile)
        .await
        .expect("acousticslabd --check launch must succeed");

    if run.exit_code != 0 || run.snapshot.is_none() {
        panic!(
            "acousticslabd --check failed (exit={}, elapsed={:?})\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            run.exit_code, run.elapsed, run.stdout, run.stderr,
        );
    }

    let snap = run
        .snapshot
        .as_ref()
        .expect("snapshot non-None per check above");
    let subsystems = snap
        .get("subsystems")
        .and_then(|v| v.as_object())
        .unwrap_or_else(|| {
            panic!(
                "snapshot missing 'subsystems' object; full snapshot:\n{}",
                serde_json::to_string_pretty(snap).unwrap_or_default(),
            )
        });

    let inference = subsystems.get("inference").unwrap_or_else(|| {
        panic!(
            "snapshot missing 'inference' subsystem; got keys {:?}\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            subsystems.keys().collect::<Vec<_>>(),
            run.stdout,
            run.stderr,
        )
    });

    let healthy = inference
        .get("healthy")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let degraded_reason = inference.get("degraded_reason").and_then(|v| v.as_str());
    assert!(
        healthy,
        "inference subsystem must stay healthy under Row 2 (degraded != \
         unhealthy per B.3.1 orthogonal-axes): {inference}\n\
         ===== STDOUT =====\n{}\n\
         ===== STDERR =====\n{}",
        run.stdout, run.stderr,
    );
    assert_eq!(
        degraded_reason,
        Some("no_backbone"),
        "inference subsystem must carry degraded_reason=no_backbone \
         on Row 2: {inference}\n\
         ===== STDOUT =====\n{}\n\
         ===== STDERR =====\n{}",
        run.stdout,
        run.stderr,
    );

    // Audio/opus/stream/training are independent of the inference engine, so a
    // missing backbone must not cascade-degrade them; any flip here is a wiring
    // regression.
    for (name, view) in subsystems {
        if name == "inference" {
            continue;
        }
        let degraded = view
            .get("degraded_reason")
            .and_then(|v| v.as_str())
            .is_some();
        assert!(
            !degraded,
            "subsystem {name:?} cascade-degraded on Row 2 (inference \
             missing a backbone shouldn't affect {name}): {view}\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            run.stdout, run.stderr,
        );
    }
}
