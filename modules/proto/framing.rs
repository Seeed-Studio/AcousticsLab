//! Sync framing: wrap a payload in an [`crate::proto::Envelope`], prost-encoded to
//! `bytes::Bytes` for broadcast. Std-only so producers avoid the async `stream_io` dep.

use crate::proto::{AudioFrame, Envelope, InferenceFrame, envelope::Payload as EnvelopePayload};
use bytes::{Bytes, BytesMut};
use prost::Message;
use thiserror::Error;

/// WS handlers refuse upgrades whose `Sec-WebSocket-Protocol` lacks this exact value.
pub const WS_SUBPROTOCOL: &str = "acousticslab.v1";

/// Reader cap on UDS frame length (~16x headroom); a larger prefix is closed
/// without parsing, defending against a giant-prefix DoS.
pub const MAX_UDS_FRAME_BYTES: u32 = 64 * 1024;

/// Wrap an [`AudioFrame`] in an [`Envelope`] and prost-encode; hot paths should
/// prefer the alloc-reusing [`wrap_audio_into`].
#[must_use = "envelope bytes are produced for a broadcast send -- ignoring them drops the frame"]
pub fn wrap_audio(frame: AudioFrame) -> Bytes {
    let env = Envelope {
        payload: Some(EnvelopePayload::Audio(frame)),
    };
    Bytes::from(env.encode_to_vec())
}

/// Wrap an [`InferenceFrame`] in an [`Envelope`] and prost-encode; hot paths
/// should prefer the alloc-reusing [`wrap_inference_into`].
#[must_use = "envelope bytes are produced for a broadcast send -- ignoring them drops the frame"]
pub fn wrap_inference(frame: InferenceFrame) -> Bytes {
    let env = Envelope {
        payload: Some(EnvelopePayload::Inference(frame)),
    };
    Bytes::from(env.encode_to_vec())
}

/// [`wrap_audio`] reusing a loop-held scratch `buf` (zero steady-state alloc),
/// split as Arc-backed [`Bytes`] for zero-copy fan-out.
#[must_use = "envelope bytes are produced for a broadcast send -- ignoring them drops the frame"]
pub fn wrap_audio_into(buf: &mut BytesMut, frame: AudioFrame) -> Bytes {
    buf.clear();
    let env = Envelope {
        payload: Some(EnvelopePayload::Audio(frame)),
    };
    // Abort-by-design: encode only fails on allocator failure growing a few-KB
    // buffer, which has no daemon recovery path -- do NOT make this a Result.
    env.encode(buf).expect("BytesMut grows on demand");
    buf.split().freeze()
}

/// Allocation-reusing [`wrap_inference`]; see [`wrap_audio_into`] for the
/// reuse and abort-by-design contract.
#[must_use = "envelope bytes are produced for a broadcast send -- ignoring them drops the frame"]
pub fn wrap_inference_into(buf: &mut BytesMut, frame: InferenceFrame) -> Bytes {
    buf.clear();
    let env = Envelope {
        payload: Some(EnvelopePayload::Inference(frame)),
    };
    env.encode(buf).expect("BytesMut grows on demand");
    buf.split().freeze()
}

#[derive(Debug, Error)]
pub enum FramingEncodeError {
    /// `observed` is `usize` to preserve true lengths above `u32::MAX`.
    #[error("payload too large: {observed} bytes > {max}")]
    PayloadTooLarge { observed: usize, max: u32 },
}

impl crate::common::error::Categorized for FramingEncodeError {
    /// Hitting the cap is a producer-side bug, not operator input.
    fn kind(&self) -> crate::common::error::ErrorKind {
        crate::common::error::ErrorKind::Internal
    }
}

/// Encode `payload` as a 4-byte LE length prefix then the bytes. The cap mirrors
/// the decoder (never emit a frame readers reject) and keeps the `as u32` prefix
/// cast lossless, not silently truncated.
#[must_use = "the encoded frame must be sent or it is dropped"]
pub fn try_encode_length_prefixed(payload: &[u8]) -> Result<Bytes, FramingEncodeError> {
    if payload.len() > MAX_UDS_FRAME_BYTES as usize {
        return Err(FramingEncodeError::PayloadTooLarge {
            observed: payload.len(),
            max: MAX_UDS_FRAME_BYTES,
        });
    }
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(payload);
    Ok(Bytes::from(buf))
}

/// Failures from [`decode_envelope`]. Each is terminal: the length prefix is the
/// sync point and resync is undefined, so callers MUST close the connection.
#[derive(Debug, Error)]
pub enum ProtoDecodeError {
    /// Not a valid prost-encoded [`Envelope`] (hostile peer or wire drift).
    #[error("envelope decode: {source}")]
    Decode {
        #[source]
        source: prost::DecodeError,
    },
    /// No `payload` variant; a payload-less endpoint must add a sibling, not relax this.
    #[error("envelope missing payload variant")]
    MissingPayload,
}

impl crate::common::error::Categorized for ProtoDecodeError {
    /// Peer-sourced data, so `UserInput` makes a wrapping API render 400 not 500.
    fn kind(&self) -> crate::common::error::ErrorKind {
        crate::common::error::ErrorKind::UserInput
    }
}

/// Decode an [`Envelope`] enforcing payload-presence; every receiver MUST use this
/// (not [`Envelope::decode`]) to centralise the [`ProtoDecodeError`] close policy.
pub fn decode_envelope(bytes: &[u8]) -> Result<Envelope, ProtoDecodeError> {
    let env = Envelope::decode(bytes).map_err(|source| ProtoDecodeError::Decode { source })?;
    if env.payload.is_none() {
        return Err(ProtoDecodeError::MissingPayload);
    }
    Ok(env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{TopK, audio_frame::Codec};

    #[test]
    fn wrap_audio_decodes_via_envelope() {
        let frame = AudioFrame {
            seq: 7,
            t_us_capture_monotonic: Some(123),
            t_us_publish_unix: Some(456),
            sample_rate: Some(48_000),
            frame_duration_ms: Some(20),
            codec: Some(Codec::Opus(Bytes::from_static(b"\xDE\xAD\xBE\xEF"))),
        };
        let wire = wrap_audio(frame.clone());
        let env = Envelope::decode(wire.as_ref()).expect("decode envelope");
        match env.payload {
            Some(EnvelopePayload::Audio(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Audio payload, got {other:?}"),
        }
    }

    #[test]
    fn wrap_inference_decodes_via_envelope() {
        let frame = InferenceFrame {
            seq: 11,
            t_us_capture_monotonic: Some(123),
            t_us_publish_unix: Some(456),
            top_k: vec![TopK {
                class_idx: 0,
                label: "bg".into(),
                prob: 1.0,
            }],
            head_id: Some("00000000-0000-0000-0000-000000000000".into()),
            head_version: Some(1),
        };
        let wire = wrap_inference(frame.clone());
        let env = Envelope::decode(wire.as_ref()).expect("decode envelope");
        match env.payload {
            Some(EnvelopePayload::Inference(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Inference payload, got {other:?}"),
        }
    }

    #[test]
    fn wrap_audio_into_decodes_via_envelope() {
        let frame = AudioFrame {
            seq: 7,
            t_us_capture_monotonic: Some(123),
            t_us_publish_unix: Some(456),
            sample_rate: Some(48_000),
            frame_duration_ms: Some(20),
            codec: Some(Codec::Opus(Bytes::from_static(b"\xDE\xAD\xBE\xEF"))),
        };
        let mut buf = BytesMut::with_capacity(64);
        let wire = wrap_audio_into(&mut buf, frame.clone());
        let env = Envelope::decode(wire.as_ref()).expect("decode envelope");
        match env.payload {
            Some(EnvelopePayload::Audio(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Audio payload, got {other:?}"),
        }
    }

    #[test]
    fn wrap_inference_into_decodes_via_envelope() {
        let frame = InferenceFrame {
            seq: 11,
            t_us_capture_monotonic: Some(123),
            t_us_publish_unix: Some(456),
            top_k: vec![TopK {
                class_idx: 0,
                label: "bg".into(),
                prob: 1.0,
            }],
            head_id: Some("00000000-0000-0000-0000-000000000000".into()),
            head_version: Some(1),
        };
        let mut buf = BytesMut::with_capacity(64);
        let wire = wrap_inference_into(&mut buf, frame.clone());
        let env = Envelope::decode(wire.as_ref()).expect("decode envelope");
        match env.payload {
            Some(EnvelopePayload::Inference(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Inference payload, got {other:?}"),
        }
    }

    /// Pin the WS token: admission is case-insensitive, but axum's `protocols()`
    /// echo is case-SENSITIVE, so only this lowercase literal yields a 101 the
    /// browser accepts. `.v1` is the wire-break negotiation surface (a v2 daemon
    /// lists both); the web client's SUBPROTOCOL literal moves in lockstep.
    #[test]
    fn ws_subprotocol_token_is_lowercase_versioned() {
        assert_eq!(WS_SUBPROTOCOL, "acousticslab.v1");
        assert!(
            WS_SUBPROTOCOL
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'.')
        );
    }

    /// Scratch capacity stays bounded across many `_into` calls; a split leaking
    /// each Bytes via an Arc cycle would grow it unbounded.
    #[test]
    fn wrap_inference_into_steady_state_capacity_is_bounded() {
        let frame = InferenceFrame {
            seq: 1,
            t_us_capture_monotonic: Some(123),
            t_us_publish_unix: Some(456),
            top_k: vec![TopK {
                class_idx: 0,
                label: "bg".into(),
                prob: 1.0,
            }],
            head_id: Some("00000000-0000-0000-0000-000000000000".into()),
            head_version: Some(1),
        };
        let mut buf = BytesMut::with_capacity(4096);
        for _ in 0..1000 {
            let _wire = wrap_inference_into(&mut buf, frame.clone());
        }
        assert!(
            buf.capacity() < 64 * 1024,
            "scratch capacity unexpectedly grew unbounded (cap={})",
            buf.capacity(),
        );
    }

    #[test]
    fn try_encode_length_prefixed_round_trip() {
        let payload = b"\x01\x02\x03\x04\x05".to_vec();
        let framed = try_encode_length_prefixed(&payload).expect("under cap");
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&framed[..4]);
        assert_eq!(u32::from_le_bytes(len_bytes), payload.len() as u32);
        assert_eq!(&framed[4..], payload.as_slice());
    }

    /// At-cap accepted (inclusive), cap+1 rejected; pins the boundary off-by-one.
    #[test]
    fn try_encode_length_prefixed_enforces_cap() {
        let at_cap = vec![0xABu8; MAX_UDS_FRAME_BYTES as usize];
        assert!(
            try_encode_length_prefixed(&at_cap).is_ok(),
            "at-cap payload must be accepted",
        );

        let over_cap = vec![0xCDu8; MAX_UDS_FRAME_BYTES as usize + 1];
        let err = try_encode_length_prefixed(&over_cap)
            .expect_err("over-cap payload must be rejected, not silently truncated");
        match err {
            FramingEncodeError::PayloadTooLarge { observed, max } => {
                assert_eq!(observed, MAX_UDS_FRAME_BYTES as usize + 1);
                assert_eq!(max, MAX_UDS_FRAME_BYTES);
            }
        }
    }

    #[test]
    fn decode_envelope_accepts_audio() {
        let frame = AudioFrame {
            seq: 1,
            t_us_capture_monotonic: Some(1),
            t_us_publish_unix: Some(2),
            sample_rate: Some(48_000),
            frame_duration_ms: Some(20),
            codec: Some(Codec::Opus(Bytes::from_static(b"\x01\x02"))),
        };
        let wire = wrap_audio(frame.clone());
        let env = decode_envelope(wire.as_ref()).expect("happy decode");
        match env.payload {
            Some(EnvelopePayload::Audio(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Audio payload, got {other:?}"),
        }
    }

    #[test]
    fn decode_envelope_accepts_inference() {
        let frame = InferenceFrame {
            seq: 9,
            t_us_capture_monotonic: Some(1),
            t_us_publish_unix: Some(2),
            top_k: vec![TopK {
                class_idx: 0,
                label: "bg".into(),
                prob: 1.0,
            }],
            head_id: Some("00000000-0000-0000-0000-000000000000".into()),
            head_version: Some(1),
        };
        let wire = wrap_inference(frame.clone());
        let env = decode_envelope(wire.as_ref()).expect("happy decode");
        match env.payload {
            Some(EnvelopePayload::Inference(decoded)) => assert_eq!(decoded, frame),
            other => panic!("expected Inference payload, got {other:?}"),
        }
    }

    #[test]
    fn decode_envelope_rejects_missing_payload() {
        let env = Envelope { payload: None };
        let wire = env.encode_to_vec();
        let err = decode_envelope(&wire).expect_err("must reject");
        assert!(matches!(err, ProtoDecodeError::MissingPayload));
    }

    #[test]
    fn decode_envelope_rejects_garbage_bytes() {
        let garbage = b"\xff\xff\xff\xff\xff\xff\xff\xff";
        let err = decode_envelope(garbage).expect_err("must reject");
        assert!(matches!(err, ProtoDecodeError::Decode { .. }));
    }
}
