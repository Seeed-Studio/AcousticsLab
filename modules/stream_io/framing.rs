//! Async UDS framing: the `tokio::io::AsyncRead` reader counterpart to the
//! sync wrap helpers in [`crate::proto::framing`]. Wire format per envelope:
//! `[u32 LE payload_len] [payload_len bytes: Envelope-encoded]`, produced by
//! [`crate::stream_io::serve_inference_uds`]. The prefix is the only sync point
//! and re-sync is undefined, so readers MUST close on any framing error.

use bytes::Bytes;

pub use crate::proto::framing::{
    FramingEncodeError, MAX_UDS_FRAME_BYTES, ProtoDecodeError, WS_SUBPROTOCOL, decode_envelope,
    try_encode_length_prefixed, wrap_audio, wrap_inference,
};

/// Framing errors from [`decode_length_prefixed`]; every variant obligates the caller to close.
#[derive(Debug, thiserror::Error)]
pub enum FramingError {
    #[error("length prefix {observed} exceeds max {max}; close connection")]
    OversizedPrefix { observed: u32, max: u32 },
    #[error("payload truncated: declared {declared} bytes, never completed before EOF")]
    Truncated { declared: u32 },
    #[error("io: {source}")]
    Io {
        #[source]
        source: std::io::Error,
    },
}

/// Decode one length-prefixed frame, returning the prefix-stripped payload; any error obligates close.
///
/// Pre-allocates the declared payload before the body arrives, so a stalling peer
/// pins up to `MAX_UDS_FRAME_BYTES` per connection: server readers MUST cap
/// concurrency (cf. [`crate::stream_io::INFERENCE_UDS_MAX_CONNS`]).
pub async fn decode_length_prefixed<R>(reader: &mut R) -> Result<Bytes, FramingError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let mut len_buf = [0u8; 4];
    reader
        .read_exact(&mut len_buf)
        .await
        .map_err(|source| FramingError::Io { source })?;
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_UDS_FRAME_BYTES {
        return Err(FramingError::OversizedPrefix {
            observed: len,
            max: MAX_UDS_FRAME_BYTES,
        });
    }
    let mut payload = vec![0u8; len as usize];
    match reader.read_exact(&mut payload).await {
        Ok(_) => Ok(Bytes::from(payload)),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            Err(FramingError::Truncated { declared: len })
        }
        Err(source) => Err(FramingError::Io { source }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn length_prefix_round_trip() {
        let payload = b"\x01\x02\x03\x04\x05".to_vec();
        let framed = try_encode_length_prefixed(&payload).expect("under cap");
        let mut cursor = std::io::Cursor::new(framed.to_vec());
        let decoded = decode_length_prefixed(&mut cursor).await.expect("decode");
        assert_eq!(decoded.as_ref(), payload.as_slice());
    }

    #[tokio::test]
    async fn length_prefix_rejects_oversized() {
        let mut prefix = Vec::with_capacity(4);
        prefix.extend_from_slice(&(MAX_UDS_FRAME_BYTES + 1).to_le_bytes());
        let mut cursor = std::io::Cursor::new(prefix);
        let err = decode_length_prefixed(&mut cursor)
            .await
            .expect_err("oversized prefix must reject");
        assert!(matches!(err, FramingError::OversizedPrefix { .. }));
    }

    #[tokio::test]
    async fn length_prefix_rejects_truncated_payload() {
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(b"abcd");
        let mut cursor = std::io::Cursor::new(buf);
        let err = decode_length_prefixed(&mut cursor)
            .await
            .expect_err("truncated payload must reject");
        assert!(matches!(err, FramingError::Truncated { .. }));
    }
}
