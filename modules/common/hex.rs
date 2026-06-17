//! Lowercase-hex byte encoding; in `common` because the layer guard forbids
//! the `inference -> file_mgr` edge that digest-stamping layers would need.

/// Encode `bytes` as lowercase ASCII hex; direct nibble lookup is ~5x faster
/// than `format!("{b:02x}")` on the upload-digest hot path.
pub fn hex_lowercase(bytes: &[u8]) -> String {
    static HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = vec![0u8; bytes.len() * 2];
    for (i, &b) in bytes.iter().enumerate() {
        out[2 * i] = HEX[(b >> 4) as usize];
        out[2 * i + 1] = HEX[(b & 0x0f) as usize];
    }
    String::from_utf8(out).expect("ascii hex is utf8")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_empty_string() {
        assert_eq!(hex_lowercase(&[]), "");
    }

    #[test]
    fn known_vector() {
        assert_eq!(hex_lowercase(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    }

    #[test]
    fn output_length_is_double_input() {
        let payload: Vec<u8> = (0..=255u8).collect();
        assert_eq!(hex_lowercase(&payload).len(), payload.len() * 2);
    }
}
