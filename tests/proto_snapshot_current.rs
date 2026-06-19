// Test scaffolding writes .bin fixtures via `std::fs::write`; clippy.toml's atomic-writer rule is production-only.
#![allow(clippy::disallowed_methods)]

//! Wire-format snapshot: encodes a canonical instance of each streamed `acousticslab::proto::*`
//! frame type (`AudioFrame`, `InferenceFrame`, `TopK`) and asserts the bytes match a checked-in
//! fixture, so any accidental wire change is a
//! PR-visible byte diff. `UPDATE_SNAPSHOTS=1` rewrites fixtures in place to carry an intentional
//! delta into review. Single-version protocol (clients are this daemon's own consumers only); a
//! future re-version is a fresh-start replacement regenerating this fixture.

use acousticslab::proto::{AudioFrame, InferenceFrame, TopK};
use prost::Message;
use std::path::{Path, PathBuf};

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("proto_fixtures")
        .join(format!("{name}.bin"))
}

/// Compares `actual` against the fixture, writing it only under `UPDATE_SNAPSHOTS=1`. A missing
/// fixture is a failure, never an auto-write: auto-writing would let a fresh checkout pass by
/// capturing whatever the encoder currently produces, defeating the regression class this guards.
fn assert_snapshot(name: &str, actual: &[u8]) {
    let path = fixture_path(name);
    let update = std::env::var_os("UPDATE_SNAPSHOTS").is_some();

    if update {
        std::fs::create_dir_all(path.parent().unwrap()).expect("mkdir fixtures");
        std::fs::write(&path, actual).expect("write fixture");
        eprintln!(
            "snapshot {} written ({} bytes) -- re-run without UPDATE_SNAPSHOTS to verify",
            path.display(),
            actual.len()
        );
        return;
    }

    if !path.exists() {
        panic!(
            "snapshot fixture missing: {}\n\
             Run `UPDATE_SNAPSHOTS=1 cargo test --test proto_snapshot_current` to seed it. \
             A missing fixture must NOT silently auto-write because that would let a fresh \
             checkout pass the wire-format snapshot test by capturing whatever the encoder \
             currently produces -- exactly the regression class this test is meant to catch.",
            path.display(),
        );
    }

    let expected =
        std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    assert_eq!(
        actual,
        expected.as_slice(),
        "wire-format snapshot mismatch for `{name}` (encoded {} bytes, fixture {} bytes). \
         If this change is intentional, re-run with UPDATE_SNAPSHOTS=1 and \
         document the wire delta in the PR.",
        actual.len(),
        expected.len(),
    );
}

fn canonical_audio_frame() -> AudioFrame {
    use acousticslab::proto::audio_frame::Codec;
    AudioFrame {
        seq: 0xDEAD_BEEF_CAFE_F00D,
        // Both clock stamps present so the snapshot exercises both proto field tags.
        t_us_capture_monotonic: Some(123_456_789),
        t_us_publish_unix: Some(1_700_000_000_000_000),
        sample_rate: Some(48_000),
        frame_duration_ms: Some(20),
        codec: Some(Codec::Opus((0..=127u8).collect::<Vec<u8>>().into())),
    }
}

fn canonical_inference_frame() -> InferenceFrame {
    InferenceFrame {
        seq: 7,
        t_us_capture_monotonic: Some(123_456_789),
        t_us_publish_unix: Some(1_700_000_000_000_000),
        top_k: vec![
            TopK {
                class_idx: 1,
                label: "yes".into(),
                prob: 0.91,
            },
            TopK {
                class_idx: 2,
                label: "no".into(),
                prob: 0.07,
            },
            TopK {
                class_idx: 0,
                label: "bg".into(),
                prob: 0.02,
            },
        ],
        head_id: Some("00000000-0000-0000-0000-000000000001".into()),
        // Always present in production frames (snapshotted atomically with the head).
        head_version: Some(42),
    }
}

fn canonical_top_k() -> TopK {
    TopK {
        class_idx: 42,
        label: "alpha".into(),
        prob: 0.5,
    }
}

#[test]
fn audio_frame_snapshot() {
    assert_snapshot("audio_frame", &canonical_audio_frame().encode_to_vec());
}

#[test]
fn inference_frame_snapshot() {
    assert_snapshot(
        "inference_frame",
        &canonical_inference_frame().encode_to_vec(),
    );
}

#[test]
fn top_k_snapshot() {
    assert_snapshot("top_k", &canonical_top_k().encode_to_vec());
}

/// Encode/decode round-trip: orthogonal to the byte snapshot, guards decoder symmetry.
#[test]
fn snapshots_round_trip() {
    let orig = canonical_audio_frame();
    let bytes = orig.encode_to_vec();
    assert_eq!(AudioFrame::decode(bytes.as_slice()).unwrap(), orig);

    let orig = canonical_inference_frame();
    let bytes = orig.encode_to_vec();
    assert_eq!(InferenceFrame::decode(bytes.as_slice()).unwrap(), orig);
}
