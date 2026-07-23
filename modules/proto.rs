//! Wire-format types. Every streaming write decodes the `Envelope` wrapper first
//! to dispatch on its `payload` oneof. Single-version (clients are only the
//! daemon's own consumers, so re-versioning is a fresh-start replacement). The
//! only wire id is `InferenceFrame.head_id`, a `Display` UUID-v4 string;
//! validated id newtypes live in [`crate::common::ids`].

#![forbid(unsafe_code)]
// prost copies `.proto` comments verbatim into `///` docs; their indented
// bullets trip these doc lints.
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

include!(concat!(env!("OUT_DIR"), "/acousticslab.v1.rs"));

// Sync framing decoder lives here so producers avoid `stream_io`'s tokio::io
// dep tree.
pub mod framing;

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    /// Catches field-number drift across an encode/decode round-trip.
    #[test]
    fn audio_frame_round_trip() {
        use audio_frame::Codec;
        let f = AudioFrame {
            seq: 12345,
            t_us_capture_monotonic: Some(123_456_789),
            t_us_publish_unix: Some(1_700_000_000_000_000),
            sample_rate: Some(48_000),
            frame_duration_ms: Some(20),
            codec: Some(Codec::Opus(bytes::Bytes::from_static(&[
                0xDE, 0xAD, 0xBE, 0xEF,
            ]))),
        };
        let bytes = f.encode_to_vec();
        let back = AudioFrame::decode(bytes.as_slice()).expect("decode");
        assert_eq!(f, back);
    }

    #[test]
    fn inference_frame_round_trip() {
        let f = InferenceFrame {
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
            head_id: Some("00000000-0000-0000-0000-000000000000".into()),
            head_version: Some(42),
        };
        let bytes = f.encode_to_vec();
        let back = InferenceFrame::decode(bytes.as_slice()).expect("decode");
        assert_eq!(f, back);
    }
}
