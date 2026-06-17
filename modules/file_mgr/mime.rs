//! Extension-only MIME mapping for asset GET responses: never sniffs bytes and is not an admission allowlist.

use std::path::Path;

const OCTET_STREAM: &str = "application/octet-stream";

/// Case-insensitive; multi-component extensions (`.tar.gz`) win over the trailing component.
pub fn content_type_from_path(path: &Path) -> &'static str {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
        return "application/gzip";
    }
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .as_deref()
    {
        Some("json") => "application/json",
        Some("txt") => "text/plain; charset=utf-8",
        Some("zip") => "application/zip",
        Some("wav") => "audio/wav",
        Some("mpk") => OCTET_STREAM,
        Some("rknn") => OCTET_STREAM,
        Some("bin") => OCTET_STREAM,
        _ => OCTET_STREAM,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_extension() {
        assert_eq!(
            content_type_from_path(Path::new("foo.json")),
            "application/json"
        );
    }

    #[test]
    fn txt_extension() {
        assert_eq!(
            content_type_from_path(Path::new("labels.txt")),
            "text/plain; charset=utf-8"
        );
    }

    #[test]
    fn tar_gz_takes_precedence_over_gz() {
        assert_eq!(
            content_type_from_path(Path::new("a.tar.gz")),
            "application/gzip"
        );
        assert_eq!(
            content_type_from_path(Path::new("a.tgz")),
            "application/gzip"
        );
    }

    #[test]
    fn zip_extension() {
        assert_eq!(
            content_type_from_path(Path::new("bundle.zip")),
            "application/zip"
        );
    }

    #[test]
    fn wav_extension() {
        assert_eq!(content_type_from_path(Path::new("clip.wav")), "audio/wav");
    }

    #[test]
    fn binary_artifacts_collapse_to_octet_stream() {
        for f in ["head.mpk", "model.rknn", "blob.bin"] {
            assert_eq!(content_type_from_path(Path::new(f)), OCTET_STREAM);
        }
    }

    #[test]
    fn unknown_extension_falls_back() {
        assert_eq!(
            content_type_from_path(Path::new("foo.unknown")),
            OCTET_STREAM
        );
        assert_eq!(content_type_from_path(Path::new("noext")), OCTET_STREAM);
    }

    #[test]
    fn extension_match_is_case_insensitive() {
        assert_eq!(
            content_type_from_path(Path::new("CAPS.JSON")),
            "application/json"
        );
        assert_eq!(
            content_type_from_path(Path::new("MIXED.Tar.Gz")),
            "application/gzip"
        );
    }
}
