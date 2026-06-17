//! Workspace asset-tree path that cannot escape the workspace root, validated
//! once at the deserialize/handler boundary.
//!
//! `/`-joined components, each non-empty `[A-Za-z0-9._-]`, no leading `.`;
//! total <= 256 bytes, per-component <= 255 (FS `NAME_MAX`), depth <= 8. The
//! serde `try_from = "String"` shape routes wire input through
//! [`AssetPath::parse`], rejecting failing literals at deserialize. Because the
//! route layer URL-decodes first, both raw (`%2E%2E%2F` -> `BadByte` on `%`)
//! and decoded (`../` -> leading-`.`) traversal fail closed.

use crate::common::error::{Categorized, ErrorKind};
use std::fmt;
use thiserror::Error;

#[inline]
fn is_allowed_component_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_')
}

pub const MAX_TOTAL_BYTES: usize = 256;
/// Per-component cap is the filesystem `NAME_MAX` floor.
pub const MAX_COMPONENT_BYTES: usize = 255;
pub const MAX_DEPTH: usize = 8;

/// Validation failure for [`AssetPath::parse`]; `Display` is the API-response
/// rendering, structured fields stay machine-readable.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AssetPathError {
    #[error("asset path is empty")]
    Empty,
    #[error("asset path component {index} is empty (no leading/trailing/double `/`)")]
    EmptyComponent { index: usize },
    #[error("asset path too long: {got} bytes > {max}")]
    TotalTooLong { got: usize, max: usize },
    #[error("asset path component {index} too long: {got} bytes > {max}")]
    ComponentTooLong {
        index: usize,
        got: usize,
        max: usize,
    },
    #[error("asset path too deep: {got} components > {max}")]
    TooDeep { got: usize, max: usize },
    /// Component starting with `.`, ruling out `.`/`..`/`.hidden`.
    #[error("asset path component {index} starts with `.` (forbidden)")]
    LeadingDot { index: usize },
    /// Defense-in-depth so a future unquoted shell-out can't read a leading-`-`
    /// component as a flag (`-rf`/`-i`).
    #[error("asset path component {index} starts with `-` (forbidden)")]
    LeadingHyphen { index: usize },
    /// Byte outside `[A-Za-z0-9._-]`, closing `\\`, NUL, control, non-ASCII, and
    /// URL-encoded `%` introducers.
    #[error(
        "asset path component {component_index} byte {byte_index} \
         is forbidden (0x{byte:02x}); allowed: [A-Za-z0-9._-]"
    )]
    BadByte {
        component_index: usize,
        byte_index: usize,
        byte: u8,
    },
}

impl Categorized for AssetPathError {
    /// All variants are operator-input failures (`400 Bad Request`).
    fn kind(&self) -> ErrorKind {
        ErrorKind::UserInput
    }
}

/// Operator-supplied path identifying a file/dir inside a workspace's
/// daemon-owned `datasets/` tree (validation contract in module docs).
#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct AssetPath(String);

impl AssetPath {
    /// Validate operator input against the module-doc rule set. Stores the input
    /// byte-identically (no normalisation), so equality/hash/serde round-trip are
    /// the identity.
    pub fn parse(s: &str) -> Result<Self, AssetPathError> {
        if s.is_empty() {
            return Err(AssetPathError::Empty);
        }
        if s.len() > MAX_TOTAL_BYTES {
            return Err(AssetPathError::TotalTooLong {
                got: s.len(),
                max: MAX_TOTAL_BYTES,
            });
        }
        // `split('/')` yields empty slots for leading/trailing/double `/`,
        // which the is_empty check rejects.
        let mut depth = 0usize;
        for (component_index, component) in s.split('/').enumerate() {
            depth += 1;
            if depth > MAX_DEPTH {
                return Err(AssetPathError::TooDeep {
                    got: depth,
                    max: MAX_DEPTH,
                });
            }
            if component.is_empty() {
                return Err(AssetPathError::EmptyComponent {
                    index: component_index,
                });
            }
            if component.len() > MAX_COMPONENT_BYTES {
                return Err(AssetPathError::ComponentTooLong {
                    index: component_index,
                    got: component.len(),
                    max: MAX_COMPONENT_BYTES,
                });
            }
            // Leading `.` rules out `.`/`..`/`.hidden`; interior `.` stays valid.
            if component.starts_with('.') {
                return Err(AssetPathError::LeadingDot {
                    index: component_index,
                });
            }
            // Leading `-` defense-in-depth against shell-flag injection.
            if component.starts_with('-') {
                return Err(AssetPathError::LeadingHyphen {
                    index: component_index,
                });
            }
            for (byte_index, &b) in component.as_bytes().iter().enumerate() {
                if !is_allowed_component_byte(b) {
                    return Err(AssetPathError::BadByte {
                        component_index,
                        byte_index,
                        byte: b,
                    });
                }
            }
        }
        Ok(Self(s.to_string()))
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Number of `/`-separated components; always >= 1 for a valid value.
    pub fn depth(&self) -> usize {
        self.0.split('/').count()
    }

    /// Iterator over the `/`-separated components, each already validated.
    pub fn components(&self) -> impl Iterator<Item = &str> {
        self.0.split('/')
    }
}

impl fmt::Display for AssetPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::str::FromStr for AssetPath {
    type Err = AssetPathError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for AssetPath {
    type Error = AssetPathError;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::parse(s)
    }
}

impl TryFrom<String> for AssetPath {
    type Error = AssetPathError;
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::parse(&s)
    }
}

impl From<AssetPath> for String {
    fn from(p: AssetPath) -> Self {
        p.0
    }
}

impl PartialEq<str> for AssetPath {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<&str> for AssetPath {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_canonical_shapes() {
        for s in &[
            "audio_dataset",
            "audio_dataset/cat",
            "audio_dataset/cat/sample.wav",
            "labels.txt",
            "model/manifest.json",
            "a",
            "a-b_c.d",
            "a/b/c/d/e/f/g/h", // depth = MAX_DEPTH
        ] {
            assert!(AssetPath::parse(s).is_ok(), "should accept {s:?}");
        }
    }

    #[test]
    fn rejects_empty() {
        assert_eq!(AssetPath::parse(""), Err(AssetPathError::Empty));
    }

    #[test]
    fn rejects_dot_and_dotdot_components() {
        for s in &[
            ".",
            "..",
            "../etc",
            "foo/.",
            "foo/..",
            "foo/../bar",
            ".hidden",
        ] {
            assert!(
                matches!(AssetPath::parse(s), Err(AssetPathError::LeadingDot { .. })),
                "should reject {s:?}"
            );
        }
    }

    #[test]
    fn rejects_leading_hyphen_components() {
        for s in &["-rf", "-i", "foo/-rf", "-rf/foo", "-"] {
            assert!(
                matches!(
                    AssetPath::parse(s),
                    Err(AssetPathError::LeadingHyphen { .. })
                ),
                "should reject {s:?}"
            );
        }
        // Only leading hyphens are rejected; interior/trailing validate.
        for s in &["my-asset.wav", "a-b/c-d", "trailing-"] {
            AssetPath::parse(s).unwrap_or_else(|e| panic!("{s:?} should validate; got {e:?}"));
        }
    }

    /// Both decoded (`../` -> `LeadingDot`) and raw (`%` -> `BadByte`) traversal
    /// must fail closed.
    #[test]
    fn rejects_url_decoded_traversal() {
        assert!(matches!(
            AssetPath::parse("../etc/passwd"),
            Err(AssetPathError::LeadingDot { index: 0 })
        ));
        let res = AssetPath::parse("%2E%2E%2Fetc");
        assert!(matches!(
            res,
            Err(AssetPathError::BadByte { byte: b'%', .. })
        ));
    }

    #[test]
    fn rejects_backslash() {
        let res = AssetPath::parse("foo\\bar");
        assert!(matches!(
            res,
            Err(AssetPathError::BadByte { byte: b'\\', .. })
        ));
    }

    #[test]
    fn rejects_nul_and_control_bytes() {
        let res_nul = AssetPath::parse("foo\0bar");
        assert!(matches!(
            res_nul,
            Err(AssetPathError::BadByte { byte: 0x00, .. })
        ));
        let res_lf = AssetPath::parse("foo\nbar");
        assert!(matches!(
            res_lf,
            Err(AssetPathError::BadByte { byte: b'\n', .. })
        ));
        let res_tab = AssetPath::parse("foo\tbar");
        assert!(matches!(
            res_tab,
            Err(AssetPathError::BadByte { byte: b'\t', .. })
        ));
    }

    #[test]
    fn rejects_non_ascii_bytes() {
        let res = AssetPath::parse("caf\u{00e9}/foo");
        assert!(matches!(res, Err(AssetPathError::BadByte { .. })));
        let res = AssetPath::parse("\u{6f22}");
        assert!(matches!(res, Err(AssetPathError::BadByte { .. })));
    }

    #[test]
    fn rejects_leading_trailing_double_slash() {
        assert!(matches!(
            AssetPath::parse("/foo"),
            Err(AssetPathError::EmptyComponent { index: 0 })
        ));
        assert!(matches!(
            AssetPath::parse("foo/"),
            Err(AssetPathError::EmptyComponent { index: 1 })
        ));
        assert!(matches!(
            AssetPath::parse("foo//bar"),
            Err(AssetPathError::EmptyComponent { index: 1 })
        ));
    }

    #[test]
    fn rejects_total_length_exceeded() {
        let s = "a".repeat(MAX_TOTAL_BYTES + 1);
        assert!(matches!(
            AssetPath::parse(&s),
            Err(AssetPathError::TotalTooLong { got: 257, max: 256 })
        ));
    }

    #[test]
    fn accepts_total_length_at_cap() {
        // "a/" + 254 bytes = 256 total, with no component over its own cap.
        let s = format!("a/{}", "b".repeat(MAX_TOTAL_BYTES - 2));
        assert_eq!(s.len(), MAX_TOTAL_BYTES);
        assert!(AssetPath::parse(&s).is_ok());
    }

    #[test]
    fn rejects_component_length_exceeded_when_total_fits() {
        // 256-byte single component: at the total cap (==256), over the per-component cap.
        let s = "a".repeat(MAX_COMPONENT_BYTES + 1);
        let res = AssetPath::parse(&s);
        assert!(matches!(
            res,
            Err(AssetPathError::ComponentTooLong {
                index: 0,
                got: 256,
                max: 255
            })
        ));
    }

    #[test]
    fn accepts_component_at_max() {
        let s = "a".repeat(MAX_COMPONENT_BYTES);
        assert!(AssetPath::parse(&s).is_ok());
    }

    #[test]
    fn rejects_depth_exceeded() {
        // 9 components, 17 bytes total (under the byte caps) so depth is what fires.
        let s = std::iter::repeat_n("a", MAX_DEPTH + 1)
            .collect::<Vec<_>>()
            .join("/");
        let res = AssetPath::parse(&s);
        assert!(matches!(
            res,
            Err(AssetPathError::TooDeep { got: 9, max: 8 })
        ));
    }

    #[test]
    fn accepts_depth_at_cap() {
        let s = std::iter::repeat_n("a", MAX_DEPTH)
            .collect::<Vec<_>>()
            .join("/");
        let p = AssetPath::parse(&s).expect("depth 8 accepted");
        assert_eq!(p.depth(), MAX_DEPTH);
    }

    #[test]
    fn try_from_works() {
        let _: AssetPath = "audio/dataset.wav".try_into().unwrap();
        let _: AssetPath = "audio/dataset.wav".to_string().try_into().unwrap();
        let res: Result<AssetPath, _> = "..".try_into();
        assert!(res.is_err());
    }

    /// Wire-supplied traversal/leading-dot/non-allowlisted chars must be rejected
    /// at deserialize, not silently wrapped.
    #[test]
    fn serde_round_trip_validates() {
        let p = AssetPath::parse("audio_dataset/cat/sample.wav").unwrap();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(
            json, "\"audio_dataset/cat/sample.wav\"",
            "transparent wire shape"
        );
        let round: AssetPath = serde_json::from_str(&json).unwrap();
        assert_eq!(round, p);

        for bad in [
            "\"\"",
            "\"..\"",
            "\"foo/../bar\"",
            "\"foo\\\\bar\"",
            "\"foo bar\"",
            "\"%2E%2E%2F\"",
            "\"\\u0000\"",
            "\"caf\\u00e9\"",
            "\"/foo\"",
            "\"foo/\"",
            "\"foo//bar\"",
        ] {
            let res: Result<AssetPath, _> = serde_json::from_str(bad);
            assert!(
                res.is_err(),
                "wire input {bad:?} must be rejected by serde shim"
            );
        }
    }

    #[test]
    fn components_iterator_matches_depth() {
        let p = AssetPath::parse("a/b/c").unwrap();
        let comps: Vec<_> = p.components().collect();
        assert_eq!(comps, vec!["a", "b", "c"]);
        assert_eq!(p.depth(), 3);
    }

    #[test]
    fn display_matches_input() {
        let p = AssetPath::parse("audio/dataset.wav").unwrap();
        assert_eq!(p.to_string(), "audio/dataset.wav");
        assert_eq!(p.as_str(), "audio/dataset.wav");
    }

    #[test]
    fn error_kind_classification_is_user_input() {
        let err = AssetPath::parse("..").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UserInput);
        let err = AssetPath::parse("").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UserInput);
    }
}
