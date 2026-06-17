//! Lifecycle Row 3: no mic device available.
//!
//! Empty `[[mic.candidates]]` makes `audio_capture` report
//! `degraded(reason="no_device")` both at registration and on every 1 Hz pump
//! tick: the pump keeps degraded rather than flipping to unhealthy, since a
//! missing device is an operator config gap, not a transient outage. Daemon
//! `--check` exits 0 because degraded != unhealthy (healthy stays true on a
//! config gap). An empty backbone catalogue degrades `inference`
//! independently (`no_backbone`): two paths in one boot prove they don't
//! cascade-trigger each other.

#[path = "daemon_helpers/mod.rs"]
mod daemon_helpers;

use std::time::Duration;

use daemon_helpers::{CheckProfile, launch_check_mode};

/// Fixture with empty mic + backbone catalogues; the validator only warns on
/// empties, so the daemon still boots. `[api]` is mandatory (no field default,
/// and `ApiCfg::validate` demands >=1 listener) so a TCP bind is supplied (the
/// helper's `--tcp-bind` overrides it anyway).
const NO_MIC_NO_BACKBONE_LAUNCH_TOML: &str = r#"
[mic]
candidates = []

[backbone]
candidates = []

[api]
tcp_bind = "127.0.0.1:0"
broadcast_capacity = 64
"#;

/// Spawn WITHOUT `--mock-audio` (operator wants real audio but the config
/// supplies no candidates) and assert exit 0 with `audio_capture` degraded
/// `"no_device"`.
#[tokio::test]
async fn lifecycle_row3_no_mic_audio_capture_degraded() {
    let profile = CheckProfile {
        check_seconds: 3,
        mock_audio: false,
        no_inference: false,
        timeout: Duration::from_secs(15),
        tcp_bind: "127.0.0.1:0".into(),
        launch_toml_override: Some(NO_MIC_NO_BACKBONE_LAUNCH_TOML.into()),
        extra_args: Vec::new(),
        cwd_override: None,
    };

    let run = launch_check_mode(profile)
        .await
        .expect("acousticsd --check launch must succeed");

    if run.exit_code != 0 || run.snapshot.is_none() {
        panic!(
            "acousticsd --check failed (exit={}, elapsed={:?})\n\
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

    let audio = subsystems.get("audio_capture").unwrap_or_else(|| {
        panic!(
            "snapshot missing 'audio_capture' subsystem; got keys {:?}\n\
             ===== STDOUT =====\n{}\n\
             ===== STDERR =====\n{}",
            subsystems.keys().collect::<Vec<_>>(),
            run.stdout,
            run.stderr,
        )
    });

    let healthy = audio
        .get("healthy")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let degraded_reason = audio.get("degraded_reason").and_then(|v| v.as_str());
    assert!(
        healthy,
        "audio_capture must stay healthy under Row 3 (degraded != \
         unhealthy on operator misconfig per B.3.1 orthogonal-axes): {audio}\n\
         ===== STDOUT =====\n{}\n\
         ===== STDERR =====\n{}",
        run.stdout, run.stderr,
    );
    assert_eq!(
        degraded_reason,
        Some("no_device"),
        "audio_capture must carry degraded_reason=no_device on Row 3: {audio}\n\
         ===== STDOUT =====\n{}\n\
         ===== STDERR =====\n{}",
        run.stdout,
        run.stderr,
    );

    let inference = subsystems.get("inference").expect("inference registered");
    assert_eq!(
        inference.get("degraded_reason").and_then(|v| v.as_str()),
        Some("no_backbone"),
        "inference must report no_backbone independently in Row 3 \
         (defends non-cascading independence): {inference}",
    );

    for name in ["opus_stream", "stream_io", "training"] {
        let view = subsystems
            .get(name)
            .unwrap_or_else(|| panic!("{name} registered"));
        assert!(
            view.get("degraded_reason")
                .and_then(|v| v.as_str())
                .is_none(),
            "subsystem {name:?} should NOT cascade-degrade on Row 3: {view}",
        );
    }
}
