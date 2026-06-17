//! `.mpk` head-artifact 32-byte little-endian header `[magic:8 |
//! header_version:u32 | feature_dim:u32 | num_classes:u32 | reserved:u32=0 |
//! payload_len:u32 | crc32:u32]`. CRC32-IEEE over `[0..28)` catches
//! truncation/corruption before the prost decoder; loaders assert
//! `feature_dim == BackboneFeatureDim::USIZE` before decoding the payload, then
//! cross-check `header.num_classes` against the payload-decoded class count (and
//! `labels.len()` against that count). Pre-header `.mpk` is not auto-detected:
//! it fails [`HeadHeaderError::BadMagic`].

use std::io::Write;

pub const HEAD_MAGIC: &[u8; 8] = b"ACSTHEAD";

/// Bumped only when the header layout changes.
pub const HEAD_HEADER_VERSION: u32 = 1;

pub const HEAD_HEADER_SIZE: usize = 32;

/// `payload_len` bounds the prost payload read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeadHeader {
    pub header_version: u32,
    pub feature_dim: u32,
    pub num_classes: u32,
    pub payload_len: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum HeadHeaderError {
    /// Shorter than [`HEAD_HEADER_SIZE`]; fixed-offset indexing would panic.
    #[error("header too short: got {got} bytes, need at least {min}")]
    TooShort { got: usize, min: usize },
    /// Missing `ACSTHEAD` magic, typically a pre-header `.mpk`.
    #[error("bad magic: expected {:?}, got {got:?}", HEAD_MAGIC)]
    BadMagic { got: [u8; 8] },
    #[error("schema too new: header_version={found} > supported max {supported}")]
    SchemaTooNew { found: u32, supported: u32 },
    /// Truncation/corruption; loader MUST refuse the payload.
    #[error("header CRC mismatch: stored=0x{stored:08x}, computed=0x{computed:08x}")]
    BadCrc { stored: u32, computed: u32 },
}

/// CRC32 covers bytes `[0..28)` and is written at offset 28.
pub fn serialize_header(
    feature_dim: u32,
    num_classes: u32,
    payload_len: u32,
) -> [u8; HEAD_HEADER_SIZE] {
    let mut buf = [0u8; HEAD_HEADER_SIZE];
    buf[0..8].copy_from_slice(HEAD_MAGIC);
    buf[8..12].copy_from_slice(&HEAD_HEADER_VERSION.to_le_bytes());
    buf[12..16].copy_from_slice(&feature_dim.to_le_bytes());
    buf[16..20].copy_from_slice(&num_classes.to_le_bytes());
    // Reserved [20..24] stays zero.
    buf[24..28].copy_from_slice(&payload_len.to_le_bytes());
    let crc = crc32_ieee(&buf[..28]);
    buf[28..32].copy_from_slice(&crc.to_le_bytes());
    buf
}

/// Parse a 32-byte header, panic-free on any slice (short input -> `TooShort`);
/// the v1 `reserved` field `[20..24]` is ignored on read.
pub fn parse_header(bytes: &[u8]) -> Result<HeadHeader, HeadHeaderError> {
    if bytes.len() < HEAD_HEADER_SIZE {
        return Err(HeadHeaderError::TooShort {
            got: bytes.len(),
            min: HEAD_HEADER_SIZE,
        });
    }
    let magic: [u8; 8] = bytes[0..8].try_into().expect("8-byte slice");
    if &magic != HEAD_MAGIC {
        return Err(HeadHeaderError::BadMagic { got: magic });
    }
    // CRC MUST precede the version check: the version field is CRC-covered, so a
    // flipped bit there would otherwise mis-report corruption as SchemaTooNew.
    let stored_crc = u32::from_le_bytes(bytes[28..32].try_into().expect("4-byte slice"));
    let computed_crc = crc32_ieee(&bytes[..28]);
    if stored_crc != computed_crc {
        return Err(HeadHeaderError::BadCrc {
            stored: stored_crc,
            computed: computed_crc,
        });
    }
    let header_version = u32::from_le_bytes(bytes[8..12].try_into().expect("4-byte slice"));
    if header_version > HEAD_HEADER_VERSION {
        return Err(HeadHeaderError::SchemaTooNew {
            found: header_version,
            supported: HEAD_HEADER_VERSION,
        });
    }
    let feature_dim = u32::from_le_bytes(bytes[12..16].try_into().expect("4-byte slice"));
    let num_classes = u32::from_le_bytes(bytes[16..20].try_into().expect("4-byte slice"));
    let payload_len = u32::from_le_bytes(bytes[24..28].try_into().expect("4-byte slice"));
    Ok(HeadHeader {
        header_version,
        feature_dim,
        num_classes,
        payload_len,
    })
}

/// Stream-write header then `payload`; `payload.len() > u32::MAX` returns
/// [`std::io::ErrorKind::InvalidInput`] without writing, since a narrowing cast
/// would corrupt the 4-byte length slot.
pub fn write_with_payload<W: Write>(
    writer: &mut W,
    feature_dim: u32,
    num_classes: u32,
    payload: &[u8],
) -> std::io::Result<()> {
    let payload_len = payload_len_or_err(payload.len())?;
    let header = serialize_header(feature_dim, num_classes, payload_len);
    writer.write_all(&header)?;
    writer.write_all(payload)?;
    Ok(())
}

fn payload_len_or_err(len: usize) -> std::io::Result<u32> {
    u32::try_from(len).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "head payload too large: {len} bytes > u32::MAX ({})",
                u32::MAX
            ),
        )
    })
}

const CRC32_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut c = i;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB88320u32 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i as usize] = c;
        i += 1;
    }
    t
};

/// CRC-32 IEEE, reflected polynomial `0xEDB88320` (zip/gzip checksum);
/// hand-rolled table to avoid a crate dependency.
fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc = CRC32_TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_header() {
        let feature_dim = 2000;
        let num_classes = 17;
        let payload_len = 12_345;
        let bytes = serialize_header(feature_dim, num_classes, payload_len);
        assert_eq!(bytes.len(), HEAD_HEADER_SIZE);
        let parsed = parse_header(&bytes).expect("parse");
        assert_eq!(parsed.header_version, HEAD_HEADER_VERSION);
        assert_eq!(parsed.feature_dim, feature_dim);
        assert_eq!(parsed.num_classes, num_classes);
        assert_eq!(parsed.payload_len, payload_len);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = serialize_header(2000, 2, 100);
        bytes[0] = b'Z';
        // Magic is checked before CRC, so flipping bytes[0] yields BadMagic
        // regardless; recompute the CRC anyway for parity with sibling tests.
        let crc = crc32_ieee(&bytes[..28]);
        bytes[28..32].copy_from_slice(&crc.to_le_bytes());
        let err = parse_header(&bytes).expect_err("bad magic must reject");
        assert!(matches!(err, HeadHeaderError::BadMagic { .. }));
    }

    #[test]
    fn rejects_schema_too_new() {
        let mut bytes = serialize_header(2000, 2, 100);
        bytes[8..12].copy_from_slice(&(HEAD_HEADER_VERSION + 1).to_le_bytes());
        let crc = crc32_ieee(&bytes[..28]);
        bytes[28..32].copy_from_slice(&crc.to_le_bytes());
        let err = parse_header(&bytes).expect_err("future schema must reject");
        assert!(matches!(err, HeadHeaderError::SchemaTooNew { .. }));
    }

    #[test]
    fn rejects_bad_crc() {
        let mut bytes = serialize_header(2000, 2, 100);
        // Flip a byte without updating the CRC.
        bytes[12] ^= 0xFF;
        let err = parse_header(&bytes).expect_err("bad CRC must reject");
        assert!(matches!(err, HeadHeaderError::BadCrc { .. }));
    }

    /// Canonical vector: `b"123456789"` -> `0xCBF43926`.
    #[test]
    fn crc32_ieee_matches_canonical_test_vector() {
        assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn write_with_payload_round_trips() {
        let payload = b"\x01\x02\x03\x04\x05\x06\x07\x08";
        let mut buf = Vec::new();
        write_with_payload(&mut buf, 2000, 2, payload).expect("write");
        assert_eq!(buf.len(), HEAD_HEADER_SIZE + payload.len());
        let header = parse_header(&buf[..HEAD_HEADER_SIZE]).expect("parse header");
        assert_eq!(header.payload_len, payload.len() as u32);
        assert_eq!(&buf[HEAD_HEADER_SIZE..], payload);
    }

    #[test]
    fn parse_header_rejects_empty_input() {
        let err = parse_header(&[]).expect_err("empty input must reject");
        assert!(matches!(
            err,
            HeadHeaderError::TooShort {
                got: 0,
                min: HEAD_HEADER_SIZE
            }
        ));
    }

    /// One byte short must reject, not out-of-bounds index.
    #[test]
    fn parse_header_rejects_31_byte_input() {
        let bytes = [0u8; HEAD_HEADER_SIZE - 1];
        let err = parse_header(&bytes).expect_err("31 bytes must reject");
        assert!(matches!(
            err,
            HeadHeaderError::TooShort {
                got: 31,
                min: HEAD_HEADER_SIZE
            }
        ));
    }

    /// Tested via the helper to avoid a >4 GiB allocation.
    #[test]
    fn write_with_payload_rejects_oversized_length() {
        assert_eq!(payload_len_or_err(0).unwrap(), 0);
        assert_eq!(payload_len_or_err(u32::MAX as usize).unwrap(), u32::MAX);

        // `u32::MAX + 1` is only representable on 64-bit targets.
        let too_big: usize = (u32::MAX as usize) + 1;
        let err = payload_len_or_err(too_big).expect_err("must reject");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("u32::MAX"),
            "diagnostic must name the cap, got {err}"
        );

        let err2 = payload_len_or_err(usize::MAX).expect_err("must reject");
        assert_eq!(err2.kind(), std::io::ErrorKind::InvalidInput);
    }
}
