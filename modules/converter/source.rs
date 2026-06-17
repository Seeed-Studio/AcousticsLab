//! `SourceModel` trait + production loader (`TfjsSource`), the sole supported
//! source format. [`read_head_bytes_streaming`] copies only the kernel+bias
//! ranges so peak heap is one shard (~50 MB), not the concatenated payload.

use crate::converter::{
    ConvertError, ConvertLimits, HeadWeights, SourceKind, TfjsManifest, TfjsManifestEntry,
};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Computed once by the loader so the sink emits metadata without re-reading.
#[derive(Clone, Debug)]
pub struct LoadedSource {
    /// Head weights in Burn's `[in_dim, n_classes]` orientation.
    pub weights: HeadWeights,
    /// SHA-256 of source bytes (model.json + shards in manifest order), lowercase hex.
    pub source_sha256: String,
}

/// Loader for one source-model format. Stateless (`&self`).
pub trait SourceModel: Send + Sync + std::fmt::Debug {
    /// Discriminator surfaced into the persisted [`crate::converter::ConversionMetadata`].
    fn kind(&self) -> SourceKind;

    /// `labels` is forward-compat for formats that embed labels to cross-validate; current impls ignore it.
    fn load(&self, src: &Path, labels: &[String]) -> Result<LoadedSource, ConvertError>;
}

/// TFJS Layers-Model directory loader (`model.json` + shards), using
/// [`ConvertLimits::default`] caps. Use [`TfjsSourceLimited`] to override.
#[derive(Clone, Copy, Debug, Default)]
pub struct TfjsSource;

impl SourceModel for TfjsSource {
    fn kind(&self) -> SourceKind {
        SourceKind::Tfjs
    }
    fn load(&self, tfjs_dir: &Path, labels: &[String]) -> Result<LoadedSource, ConvertError> {
        TfjsSourceLimited::new(ConvertLimits::default()).load(tfjs_dir, labels)
    }
}

/// TFJS source with explicit [`ConvertLimits`] overriding the defaults.
#[derive(Clone, Copy, Debug)]
pub struct TfjsSourceLimited {
    limits: ConvertLimits,
}

impl TfjsSourceLimited {
    pub fn new(limits: ConvertLimits) -> Self {
        Self { limits }
    }
}

impl SourceModel for TfjsSourceLimited {
    fn kind(&self) -> SourceKind {
        SourceKind::Tfjs
    }
    fn load(&self, tfjs_dir: &Path, _labels: &[String]) -> Result<LoadedSource, ConvertError> {
        let model_json = tfjs_dir.join("model.json");
        let json_bytes = std::fs::read(&model_json)
            .map_err(|e| crate::converter::convert_read_err(model_json.display(), e))?;
        let manifest =
            crate::converter::parse_tfjs_manifest_with_limits(&json_bytes, &self.limits)?;
        let (k_entry, b_entry) = crate::converter::pick_tfjs_head_entries(&manifest)?;
        let mut hasher = Sha256::new();
        hasher.update(&json_bytes);
        let (kernel_bytes, bias_bytes) =
            read_head_bytes_streaming(tfjs_dir, &manifest, k_entry, b_entry, Some(&mut hasher))?;
        let source_sha256 = crate::common::hex::hex_lowercase(&hasher.finalize());
        let weights = crate::converter::head_weights_from_head_byte_ranges(
            &manifest,
            k_entry,
            b_entry,
            &kernel_bytes,
            &bias_bytes,
        )?;
        Ok(LoadedSource {
            weights,
            source_sha256,
        })
    }
}

/// Stream-read each shard once, feeding bytes through `hasher` (when supplied)
/// and copying only the head's kernel+bias byte ranges; peak is one shard's
/// `Vec<u8>` (<= 50 MB), returned buffers total ~24 KB.
///
/// `hasher` is optional (`extract_head_from_tfjs*` predate the SHA contract and
/// pass `None`); callers needing `source_sha256` feed `model.json` first and finalize after.
///
/// `manifest.shards` order MUST match parse-time math (`offset_bytes` is absolute
/// into the manifest-order concatenated set); the cumulative==declared-total
/// check below guards a manifest/shard-set mismatch that would yield zeroed kernel bytes.
pub(crate) fn read_head_bytes_streaming(
    tfjs_dir: &Path,
    manifest: &TfjsManifest,
    k: &TfjsManifestEntry,
    b: &TfjsManifestEntry,
    mut hasher: Option<&mut Sha256>,
) -> Result<(Vec<u8>, Vec<u8>), ConvertError> {
    // `checked_add`: this `pub(crate)` fn accepts a caller-supplied manifest
    // that need not have gone through parse-time validation (tests build
    // entries by hand), so don't trust the entries to sum within usize::MAX.
    let declared_total: usize = manifest.entries.iter().try_fold(0usize, |acc, e| {
        acc.checked_add(e.len_bytes)
            .ok_or_else(|| ConvertError::TfjsShapeOverflow {
                name: format!("<cumulative len_bytes after `{}`>", e.name),
                shape: e.shape.clone(),
            })
    })?;
    let mut kernel_bytes = vec![0u8; k.len_bytes];
    let mut bias_bytes = vec![0u8; b.len_bytes];
    let mut cumulative: usize = 0;

    // TRUST-MODEL (open gap): manifest declares shard PATHS but no per-shard
    // LENGTH, so a size-preserving redistribution (truncated middle + oversized
    // final) keeps cumulative==declared_total yet copies kernel/bias from WRONG
    // offsets, silently corrupting the head; the equality gate catches only a
    // wrong TOTAL, and production passes hasher=None so never compares the SHA.
    // Bounded (shards are daemon-staged from an authenticated upload under
    // max_upload_bytes; corruption surfaces as bad val accuracy). Fix would be a
    // client-declared source hash compared before extraction.
    for shard in &manifest.shards {
        let path = tfjs_dir.join(shard);
        // Size-check against declared_total (per-shard ceiling) BEFORE fs::read,
        // else an oversize shard pays the OOM peak before the bottom gate rejects
        // it. u64 compare: usize-truncating an untrusted length on a 32-bit host
        // drops high bits and defeats this for a >4 GiB shard.
        let shard_size = std::fs::metadata(&path)
            .map_err(|e| crate::converter::convert_read_err(path.display(), e))?
            .len();
        if shard_size > declared_total as u64 {
            return Err(ConvertError::TfjsParse {
                what: "shard size",
                // Manifest-relative shard NAME, never path.display(): this lifts
                // into the convert job's unsanitized SSE/JSONL stream, leaking the staging path.
                msg: format!(
                    "shard `{shard}` on-disk size {shard_size} bytes exceeds \
                     manifest's declared total {declared_total} bytes",
                ),
            });
        }
        let bytes = std::fs::read(&path)
            .map_err(|e| crate::converter::convert_read_err(path.display(), e))?;
        if let Some(h) = hasher.as_deref_mut() {
            h.update(&bytes);
        }
        let shard_start = cumulative;
        let shard_end =
            cumulative
                .checked_add(bytes.len())
                .ok_or_else(|| ConvertError::TfjsShapeOverflow {
                    name: "<cumulative shard offset>".to_string(),
                    shape: vec![],
                })?;

        // Copy each entry's half-open overlap [max(starts), min(ends)) with this shard.
        for (entry, out) in [(k, &mut kernel_bytes), (b, &mut bias_bytes)] {
            let entry_end = entry.offset_bytes.saturating_add(entry.len_bytes);
            let overlap_start = entry.offset_bytes.max(shard_start);
            let overlap_end = entry_end.min(shard_end);
            if overlap_start >= overlap_end {
                continue;
            }
            let in_shard = overlap_start - shard_start;
            let in_buffer = overlap_start - entry.offset_bytes;
            let copy_len = overlap_end - overlap_start;
            out[in_buffer..in_buffer + copy_len]
                .copy_from_slice(&bytes[in_shard..in_shard + copy_len]);
        }

        cumulative = shard_end;
    }

    if cumulative != declared_total {
        return Err(ConvertError::TfjsBlobLength {
            have: cumulative,
            declared: declared_total,
        });
    }

    Ok((kernel_bytes, bias_bytes))
}

#[cfg(test)]
mod tests {
    // Fixtures use direct `std::fs::write`; atomic-writer constraint is production-only.
    #![allow(clippy::disallowed_methods)]
    use super::*;
    use crate::converter::TfjsManifestEntry;

    fn entry(name: &str, shape: Vec<usize>, offset: usize) -> TfjsManifestEntry {
        let len_bytes: usize = shape.iter().product::<usize>() * 4;
        TfjsManifestEntry {
            name: name.to_string(),
            shape,
            offset_bytes: offset,
            len_bytes,
        }
    }

    /// Pins the byte-range overlap math when a tensor straddles shard boundaries.
    #[test]
    fn streaming_read_handles_tensor_straddling_shards() {
        let dir = tempfile::tempdir().expect("tempdir");

        // Entries cover all 128 bytes contiguously (cumulative-length check passes);
        // kernel 32..104 straddles both shard boundaries.
        let mut blob = Vec::with_capacity(128);
        for i in 0..128u8 {
            blob.push(i);
        }
        std::fs::write(dir.path().join("s0.bin"), &blob[0..40]).unwrap();
        std::fs::write(dir.path().join("s1.bin"), &blob[40..80]).unwrap();
        std::fs::write(dir.path().join("s2.bin"), &blob[80..128]).unwrap();

        let k = TfjsManifestEntry {
            name: "k".into(),
            shape: vec![18],
            offset_bytes: 32,
            len_bytes: 72,
        };
        let b = TfjsManifestEntry {
            name: "b".into(),
            shape: vec![3],
            offset_bytes: 108,
            len_bytes: 12,
        };
        let manifest = TfjsManifest {
            entries: vec![
                entry("pre_k", vec![8], 0),
                k.clone(),
                entry("between", vec![1], 104),
                b.clone(),
                entry("tail", vec![2], 120),
            ],
            shards: vec!["s0.bin".into(), "s1.bin".into(), "s2.bin".into()],
        };

        let (kernel_bytes, bias_bytes) =
            read_head_bytes_streaming(dir.path(), &manifest, &k, &b, None).expect("stream");
        assert_eq!(kernel_bytes, blob[32..104]);
        assert_eq!(bias_bytes, blob[108..120]);
    }

    /// Single-shard layout (the most-common path), no straddling.
    #[test]
    fn streaming_read_single_shard() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut blob = vec![0u8; 200];
        for (i, b) in blob.iter_mut().enumerate() {
            *b = (i & 0xff) as u8;
        }
        std::fs::write(dir.path().join("only.bin"), &blob).unwrap();

        let k = TfjsManifestEntry {
            name: "k".into(),
            shape: vec![2, 5],
            offset_bytes: 16,
            len_bytes: 40,
        };
        let b = TfjsManifestEntry {
            name: "b".into(),
            shape: vec![5],
            offset_bytes: 56,
            len_bytes: 20,
        };
        let manifest = TfjsManifest {
            entries: vec![
                entry("pre", vec![4], 0),
                k.clone(),
                b.clone(),
                entry("post", vec![31], 76),
            ],
            shards: vec!["only.bin".into()],
        };

        let (kb, bb) =
            read_head_bytes_streaming(dir.path(), &manifest, &k, &b, None).expect("stream");
        assert_eq!(kb, blob[16..56]);
        assert_eq!(bb, blob[56..76]);
    }

    /// Streaming hash (json ++ shards) is byte-equal to a single concat hash;
    /// `source_sha256` is a persisted content-addressable key, so this must hold.
    #[test]
    fn streaming_sha256_matches_concat_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let model_json = b"{\"weightsManifest\": []}";
        let s0 = vec![0xAAu8; 64];
        let s1 = vec![0xBBu8; 96];
        std::fs::write(dir.path().join("s0.bin"), &s0).unwrap();
        std::fs::write(dir.path().join("s1.bin"), &s1).unwrap();

        let mut reference_blob = Vec::new();
        reference_blob.extend_from_slice(model_json);
        reference_blob.extend_from_slice(&s0);
        reference_blob.extend_from_slice(&s1);
        let mut h_ref = Sha256::new();
        h_ref.update(&reference_blob);
        let ref_digest = h_ref.finalize();

        let manifest = TfjsManifest {
            entries: vec![entry("only", vec![64 / 4 + 96 / 4], 0)],
            shards: vec!["s0.bin".into(), "s1.bin".into()],
        };
        // Dummy entry: exercises the hasher pass only, not the copy.
        let dummy = TfjsManifestEntry {
            name: "dummy".into(),
            shape: vec![1],
            offset_bytes: 0,
            len_bytes: 4,
        };
        let mut h_stream = Sha256::new();
        h_stream.update(model_json);
        let _ =
            read_head_bytes_streaming(dir.path(), &manifest, &dummy, &dummy, Some(&mut h_stream))
                .expect("stream");
        let stream_digest = h_stream.finalize();

        assert_eq!(stream_digest.as_slice(), ref_digest.as_slice());
    }

    /// Cumulative shard length disagreeing with the declared total (dropped or
    /// truncated shard) surfaces `TfjsBlobLength`, not silently zeroed bytes.
    #[test]
    fn streaming_blob_length_mismatch_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let s0 = vec![0u8; 30];
        std::fs::write(dir.path().join("s0.bin"), &s0).unwrap();
        let k = TfjsManifestEntry {
            name: "k".into(),
            shape: vec![10],
            offset_bytes: 0,
            len_bytes: 40,
        };
        let b = TfjsManifestEntry {
            name: "b".into(),
            shape: vec![1],
            offset_bytes: 36,
            len_bytes: 4,
        };
        let manifest = TfjsManifest {
            entries: vec![k.clone(), b.clone()],
            shards: vec!["s0.bin".into()],
        };
        let err = read_head_bytes_streaming(dir.path(), &manifest, &k, &b, None).unwrap_err();
        assert!(
            matches!(
                err,
                ConvertError::TfjsBlobLength {
                    have: 30,
                    declared: 44
                }
            ),
            "{err:?}",
        );
    }
}
