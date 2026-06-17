//! Validated identifier newtypes: validators run at the deserialize/handler
//! boundary so downstream code cannot hold an unvalidated string.
//!
//! UUID-strict ([`WorkspaceId`], [`JobId`], [`HeadId`]): canonical lowercase
//! v4 form is a safe-filename subset, so the type can never escape a workspace
//! root; no `Default`, which would fabricate identities at `unwrap_or_default`.
//! [`AssetId`] is a filename; [`MicId`] is `Arc<str>`-backed for refcount-cheap
//! per-frame clones.

use crate::common::error::{Categorized, ErrorKind};
use std::fmt;
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;

// MARK: Errors

/// Renders [`IdError::BadUuidByte`] with a lowercase-hint for uppercase hex;
/// the `#[error]` derive can't switch on the byte, so the renderer lives here.
fn bad_uuid_byte_msg(index: usize, byte: u8) -> String {
    if matches!(byte, b'A'..=b'F') {
        format!(
            "invalid byte 0x{byte:02x} ('{}') at position {index}; try lowercasing the input",
            byte as char,
        )
    } else {
        format!(
            "invalid byte 0x{byte:02x} at position {index} \
             (UUID-v4 expects lowercase hex digits with `-` at positions 8/13/18/23)"
        )
    }
}

/// Validation failure constructing an identifier; `Display` is operator-facing,
/// the structured fields are for programmatic discrimination.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum IdError {
    #[error("identifier is empty")]
    Empty,
    #[error("expected UUID-v4 (36 chars hex + dashes); got {got} chars")]
    BadUuidLength { got: usize },
    /// Uppercase hex rejected: keeps the wire/filesystem representation unique.
    #[error("{}", bad_uuid_byte_msg(*index, *byte))]
    BadUuidByte { index: usize, byte: u8 },
    #[error("expected UUID-v4 version nibble '4' at position 14, got 0x{byte:02x}")]
    BadUuidVersion { byte: u8 },
    #[error("expected RFC 4122 variant nibble in '8'..='b' at position 19, got 0x{byte:02x}")]
    BadUuidVariant { byte: u8 },
    #[error("{kind} too long: {got} chars > {max}")]
    TooLong {
        kind: &'static str,
        got: usize,
        max: usize,
    },
    #[error("{kind} contains forbidden byte 0x{byte:02x} at position {index}")]
    BadChar {
        kind: &'static str,
        index: usize,
        byte: u8,
    },
    /// Leading `.` rejected: covers path-traversal `.`/`..` and unix hidden files.
    #[error("{kind} must not begin with `.`")]
    LeadingDot { kind: &'static str },
}

impl Categorized for IdError {
    fn kind(&self) -> ErrorKind {
        ErrorKind::UserInput
    }
}

// MARK: UUID-strict identifier macro

/// Validate canonical lowercase UUID-v4 (36 chars, `-` at 8/13/18/23, version
/// `'4'` at byte 14, variant `'8'..='b'` at byte 19); lowercase pins one form.
fn validate_uuid_v4_str(s: &str) -> Result<(), IdError> {
    if s.is_empty() {
        return Err(IdError::Empty);
    }
    if s.len() != 36 {
        return Err(IdError::BadUuidLength { got: s.len() });
    }
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        let ok = if matches!(i, 8 | 13 | 18 | 23) {
            b == b'-'
        } else {
            b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
        };
        if !ok {
            return Err(IdError::BadUuidByte { index: i, byte: b });
        }
    }
    if bytes[14] != b'4' {
        return Err(IdError::BadUuidVersion { byte: bytes[14] });
    }
    // RFC 4122 variant: high two bits `10` => hex '8'/'9'/'a'/'b'.
    if !matches!(bytes[19], b'8' | b'9' | b'a' | b'b') {
        return Err(IdError::BadUuidVariant { byte: bytes[19] });
    }
    Ok(())
}

// A macro, not a generic `Uuid<Tag>`, so each id is a distinct nominal type
// (the compiler rejects passing a `WorkspaceId` where a `JobId` is expected).
macro_rules! uuid_id {
    ($(#[$attr:meta])* $name:ident) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        #[derive(serde::Serialize, serde::Deserialize)]
        // Wire serde routes through `parse`: a non-v4-strict literal is rejected, not wrapped.
        #[serde(try_from = "String", into = "String")]
        pub struct $name(Uuid);

        // No `Default`: see module docs (fresh-UUID default fabricates ids).
        #[allow(clippy::new_without_default)]
        impl $name {
            /// Generate a fresh random UUID-v4 (daemon-created ids).
            #[inline]
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            /// Validate operator input; canonical lowercase UUID-v4 only.
            pub fn parse(s: &str) -> Result<Self, IdError> {
                validate_uuid_v4_str(s)?;
                Ok(Self(Uuid::parse_str(s).expect("validated UUID-v4 shape")))
            }

            #[inline]
            pub fn as_uuid(&self) -> &Uuid {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }

        impl std::str::FromStr for $name {
            type Err = IdError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::parse(s)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = IdError;
            fn try_from(s: &str) -> Result<Self, Self::Error> {
                Self::parse(s)
            }
        }

        impl TryFrom<String> for $name {
            type Error = IdError;
            fn try_from(s: String) -> Result<Self, Self::Error> {
                Self::parse(&s)
            }
        }

        impl From<$name> for String {
            fn from(id: $name) -> Self {
                id.0.to_string()
            }
        }
    };
}

uuid_id! {
    /// Workspace identifier, naming a per-workspace directory under the file
    /// root. v4-strict => the path component is provably safe (no `..`,
    /// separators, or case-sensitivity tricks).
    WorkspaceId
}

uuid_id! {
    /// Background-job identifier.
    JobId
}

uuid_id! {
    /// Inference-head identifier; stamped at publish time onto every inference
    /// frame's `head_id` so consumers know which weights produced it.
    HeadId
}

// MARK: Default runtime HeadId

/// Canonical UUID-v4 of the bundled default head; pinned so the default
/// runtime-head id is reproducible across deploys.
pub const DEFAULT_RUNTIME_HEAD_ID_STR: &str = "00000000-0000-4000-8000-000000000000";

/// Parsed [`HeadId`] for the bundled default head; parsed (not literal) so the
/// v4-strict validator runs at every call.
pub fn default_runtime_head_id() -> HeadId {
    HeadId::parse(DEFAULT_RUNTIME_HEAD_ID_STR)
        .expect("default runtime head id is a hard-coded valid UUID-v4")
}

// MARK: AssetId

const ASSET_ID_MAX: usize = 255;

/// Asset identifier: a single file basename inside a workspace. Charset
/// `[A-Za-z0-9._-]`, non-empty, <=255, no leading `.`; rejecting `/` keeps the
/// type from escaping the workspace root. Serde routes through `parse`,
/// serializing as a bare string.
#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct AssetId(String);

impl From<AssetId> for String {
    fn from(id: AssetId) -> Self {
        id.0
    }
}

impl AssetId {
    /// 255 is the `NAME_MAX` floor; leading `.` is rejected but interior or
    /// trailing `.` is fine.
    pub fn parse(s: &str) -> Result<Self, IdError> {
        if s.is_empty() {
            return Err(IdError::Empty);
        }
        if s.len() > ASSET_ID_MAX {
            return Err(IdError::TooLong {
                kind: "asset id",
                got: s.len(),
                max: ASSET_ID_MAX,
            });
        }
        // Dedicated leading-`.` variant (vs `BadChar`) keeps the diagnostic distinct.
        if s.as_bytes()[0] == b'.' {
            return Err(IdError::LeadingDot { kind: "asset id" });
        }
        for (i, &b) in s.as_bytes().iter().enumerate() {
            let ok = b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_');
            if !ok {
                return Err(IdError::BadChar {
                    kind: "asset id",
                    index: i,
                    byte: b,
                });
            }
        }
        Ok(Self(s.to_string()))
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AssetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for AssetId {
    type Err = IdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for AssetId {
    type Error = IdError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s)
    }
}

impl TryFrom<String> for AssetId {
    type Error = IdError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse(&s)
    }
}

impl PartialEq<str> for AssetId {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for AssetId {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<AssetId> for str {
    fn eq(&self, other: &AssetId) -> bool {
        self == other.0
    }
}

impl PartialEq<AssetId> for &str {
    fn eq(&self, other: &AssetId) -> bool {
        *self == other.0
    }
}

// MARK: MicId

const MIC_ID_MAX: usize = 128;

/// Microphone identifier (operator-set via TOML, e.g. `"hw:1,0"`), a per-source
/// policy key. `Arc<str>`-backed for refcount-cheap per-frame clones. Charset
/// alphanumerics + `: , = _ . -` only (excludes shell metacharacters), <=128;
/// serde routes through `parse`, serializing as a bare string.
#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct MicId(Arc<str>);

impl From<MicId> for String {
    fn from(id: MicId) -> Self {
        id.0.as_ref().to_owned()
    }
}

impl MicId {
    /// Construct from a known-valid `&'static str` literal (fixtures, mock
    /// catalogue); panics on invalid. Operator input must go through
    /// [`Self::parse`].
    pub fn from_static(s: &'static str) -> Self {
        Self::parse(s)
            .unwrap_or_else(|e| panic!("MicId::from_static({s:?}) -- invalid literal: {e}"))
    }

    pub fn parse(s: &str) -> Result<Self, IdError> {
        if s.is_empty() {
            return Err(IdError::Empty);
        }
        if s.len() > MIC_ID_MAX {
            return Err(IdError::TooLong {
                kind: "mic id",
                got: s.len(),
                max: MIC_ID_MAX,
            });
        }
        for (i, &b) in s.as_bytes().iter().enumerate() {
            // Punctuation real mic ids use (ALSA `plughw:CARD=USB`), shell-safe.
            let ok =
                b.is_ascii_alphanumeric() || matches!(b, b':' | b',' | b'=' | b'_' | b'.' | b'-');
            if !ok {
                return Err(IdError::BadChar {
                    kind: "mic id",
                    index: i,
                    byte: b,
                });
            }
        }
        Ok(Self(Arc::from(s)))
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MicId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for MicId {
    type Err = IdError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for MicId {
    type Error = IdError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s)
    }
}

impl TryFrom<String> for MicId {
    type Error = IdError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse(&s)
    }
}

// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn good_uuid() -> &'static str {
        "11111111-2222-4333-8444-555555555555"
    }

    #[test]
    fn uuid_id_parse_accepts_canonical() {
        let id = WorkspaceId::parse(good_uuid()).expect("valid uuid");
        assert_eq!(id.to_string(), good_uuid());
    }

    #[test]
    fn uuid_id_parse_rejects_empty() {
        assert_eq!(WorkspaceId::parse(""), Err(IdError::Empty));
    }

    #[test]
    fn uuid_id_parse_rejects_wrong_length() {
        let res = JobId::parse("abc");
        assert!(matches!(res, Err(IdError::BadUuidLength { got: 3 })));
    }

    #[test]
    fn uuid_id_parse_rejects_invalid_chars() {
        let bad = "X1111111-2222-3333-4444-555555555555";
        let res = HeadId::parse(bad);
        assert!(matches!(
            res,
            Err(IdError::BadUuidByte {
                index: 0,
                byte: b'X'
            })
        ));
    }

    #[test]
    fn uuid_id_parse_rejects_missing_dash() {
        // 35 chars => length check fires first; accept either failure.
        let bad = "111111111222-3333-4444-555555555555";
        let res = WorkspaceId::parse(bad);
        assert!(matches!(
            res,
            Err(IdError::BadUuidLength { .. }) | Err(IdError::BadUuidByte { index: 8, .. })
        ));
    }

    #[test]
    fn uuid_id_new_round_trips_through_parse() {
        for _ in 0..16 {
            let id = JobId::new();
            let s = id.to_string();
            let parsed = JobId::parse(&s).expect("self-round-trip");
            assert_eq!(id, parsed);
        }
    }

    #[test]
    fn uuid_id_try_from_works() {
        let _: WorkspaceId = good_uuid().try_into().unwrap();
        let _: WorkspaceId = good_uuid().to_string().try_into().unwrap();
        let res: Result<WorkspaceId, _> = "nope".try_into();
        assert!(res.is_err());
    }

    #[test]
    fn uuid_id_rejects_uppercase_hex() {
        let bad = "11111111-2222-4333-8444-55555555555A";
        let res = WorkspaceId::parse(bad);
        assert!(matches!(
            res,
            Err(IdError::BadUuidByte {
                index: 35,
                byte: b'A'
            })
        ));
    }

    #[test]
    fn uuid_id_rejects_non_v4_version_nibble() {
        let bad = "11111111-2222-1333-8444-555555555555";
        let res = HeadId::parse(bad);
        assert!(matches!(res, Err(IdError::BadUuidVersion { byte: b'1' })));
    }

    #[test]
    fn uuid_id_rejects_bad_variant_nibble() {
        let bad = "11111111-2222-4333-c444-555555555555";
        let res = JobId::parse(bad);
        assert!(matches!(res, Err(IdError::BadUuidVariant { byte: b'c' })));
    }

    #[test]
    fn default_runtime_head_id_parses_and_round_trips() {
        let id = default_runtime_head_id();
        assert_eq!(id.to_string(), DEFAULT_RUNTIME_HEAD_ID_STR);
        let parsed = HeadId::parse(DEFAULT_RUNTIME_HEAD_ID_STR).unwrap();
        assert_eq!(id, parsed);
    }

    /// Wire serde routes UUID newtypes through `parse`: rejects v4-strict
    /// failures, accepts the canonical form.
    #[test]
    fn uuid_id_serde_round_trip_validates() {
        let id = WorkspaceId::parse(good_uuid()).unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, format!("\"{}\"", good_uuid()), "transparent wire");
        let round: WorkspaceId = serde_json::from_str(&json).unwrap();
        assert_eq!(round, id);

        for bad in [
            "\"\"",                                     // empty
            "\"abc\"",                                  // length
            "\"11111111-2222-3333-8444-555555555555\"", // v3
            "\"11111111-2222-4333-c444-555555555555\"", // bad variant
            "\"11111111-2222-4333-8444-55555555555A\"", // uppercase
            "\"11111111_2222_4333_8444_555555555555\"", // wrong separators
        ] {
            let res: Result<WorkspaceId, _> = serde_json::from_str(bad);
            assert!(
                res.is_err(),
                "wire input {bad:?} must be rejected by serde shim"
            );
        }
    }

    #[test]
    fn asset_id_accepts_filename_shape() {
        for s in &[
            "head_v3.mpk",
            "labels.txt",
            "dataset-2026-05-01.zip",
            "tfjs_model.json",
            "a",
        ] {
            assert!(AssetId::parse(s).is_ok(), "should accept {s:?}");
        }
    }

    #[test]
    fn asset_id_rejects_empty() {
        assert_eq!(AssetId::parse(""), Err(IdError::Empty));
    }

    #[test]
    fn asset_id_rejects_path_separators() {
        // Leading-dot => `LeadingDot`; other forbidden bytes => `BadChar`.
        for s in &["weights/head.mpk", "a\\b"] {
            assert!(
                matches!(AssetId::parse(s), Err(IdError::BadChar { .. })),
                "should reject {s:?} with BadChar",
            );
        }
        for s in &["..", "../etc/passwd", ".hidden"] {
            assert!(
                matches!(AssetId::parse(s), Err(IdError::LeadingDot { .. })),
                "should reject {s:?} with LeadingDot",
            );
        }
    }

    #[test]
    fn asset_id_rejects_oversized() {
        let s = "a".repeat(256);
        assert!(matches!(
            AssetId::parse(&s),
            Err(IdError::TooLong {
                max: 255,
                got: 256,
                ..
            })
        ));
    }

    /// Wire-supplied traversal/leading-dot/non-allowlisted chars must be
    /// rejected at deserialize time, not silently wrapped.
    #[test]
    fn asset_id_serde_round_trip_validates() {
        let id = AssetId::parse("head_v3.mpk").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"head_v3.mpk\"", "transparent wire shape");
        let round: AssetId = serde_json::from_str(&json).unwrap();
        assert_eq!(round, id);

        for bad in ["\"..\"", "\"weights/head.mpk\"", "\".hidden\"", "\"a b\""] {
            let res: Result<AssetId, _> = serde_json::from_str(bad);
            assert!(
                res.is_err(),
                "wire input {bad:?} must be rejected by serde shim",
            );
        }
    }

    #[test]
    fn mic_id_accepts_canonical_shapes() {
        for s in &["hw:1,0", "mock:0", "default", "plughw:CARD=USB"] {
            assert!(MicId::parse(s).is_ok(), "should accept {s:?}");
        }
    }

    #[test]
    fn mic_id_rejects_whitespace() {
        for s in &["hw 1,0", " hw:1,0", "hw:1,0\n"] {
            assert!(
                matches!(MicId::parse(s), Err(IdError::BadChar { .. })),
                "should reject {s:?}"
            );
        }
    }

    #[test]
    fn mic_id_rejects_non_ascii() {
        assert!(matches!(
            MicId::parse("mic\u{2014}1"),
            Err(IdError::BadChar { .. })
        ));
    }

    #[test]
    fn mic_id_rejects_shell_metacharacters() {
        for s in &[
            "hw'1", "hw\"1", "hw\\1", "hw;1", "hw|1", "hw>1", "hw<1", "hw&1", "hw(1)", "hw*1",
            "hw?1", "hw#1", "hw$1", "hw%1", "hw/1", "hw+1", "hw@1", "hw!1",
        ] {
            assert!(
                matches!(MicId::parse(s), Err(IdError::BadChar { .. })),
                "should reject {s:?}",
            );
        }
    }

    #[test]
    fn mic_id_rejects_oversized() {
        let s = "x".repeat(129);
        assert!(matches!(
            MicId::parse(&s),
            Err(IdError::TooLong {
                max: 128,
                got: 129,
                ..
            })
        ));
    }

    /// Clone shares the allocation, proving the `Arc<str>` rationale.
    #[test]
    fn mic_id_clone_is_refcount_cheap() {
        let original = MicId::parse("hw:1,0").unwrap();
        let cheap = original.clone();
        assert_eq!(original, cheap);
        assert!(Arc::ptr_eq(&original.0, &cheap.0));
    }

    /// Wire-supplied control/whitespace/non-ASCII must be rejected at
    /// deserialize time so a TOML literal can't smuggle bad bytes through.
    #[test]
    fn mic_id_serde_rejects_invalid_strings() {
        for bad in [
            "\"\"",
            "\"hw 1,0\"",
            "\"hw:1,0\\n\"",
            "\" hw:1,0\"",
            "\"mic\\u2014\"", // em-dash => 3-byte UTF-8
        ] {
            let res: Result<MicId, _> = serde_json::from_str(bad);
            assert!(
                res.is_err(),
                "wire input {bad:?} must be rejected by serde shim"
            );
        }
    }

    #[test]
    fn mic_id_serde_round_trip_validates() {
        let id = MicId::parse("hw:1,0").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"hw:1,0\"", "transparent wire shape");
        let round: MicId = serde_json::from_str(&json).unwrap();
        assert_eq!(round, id);
    }
}
