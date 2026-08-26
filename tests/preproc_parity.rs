//! Parity test of `Preproc::spectrogram` vs bundled reference tensors in misc/npys/.
//!
//! Gate p99|D| < 1e-4 and mean|D| < 2e-5 catches a regression that drops the
//! `(x - mean) / (sqrt(var) + 1e-4)` z-norm epsilon (re-inflates p99 to
//! ~1.4e-4..3.8e-4); max|D| < 2e-2 stays loose for the rustfft-vs-TF FFT floor.

use acousticslab::common::dims::{NBins, NFrames, WaveformLen};
use acousticslab::preproc::Preproc;
use std::io::Read;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// Read a float32 .npy file into (shape, data); handles only f4 dtype, C-order.
fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let mut f =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {}", path.display(), e));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    assert_eq!(
        &buf[0..6],
        b"\x93NUMPY",
        "{} not an npy file",
        path.display()
    );
    let major = buf[6];
    let header_len = match major {
        1 => u16::from_le_bytes([buf[8], buf[9]]) as usize,
        2 | 3 => u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
        v => panic!("unsupported npy major version {v}"),
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&buf[header_start..header_start + header_len]).unwrap();
    let descr = extract_str(header, "'descr'");
    assert!(
        descr == "<f4" || descr == "|f4" || descr == "=f4",
        "only f32 npy supported, got descr={descr:?} in {}",
        path.display()
    );
    let fortran = extract_str(header, "'fortran_order'");
    assert_eq!(fortran, "False", "fortran-order npy not supported");
    let shape_str = extract_tuple(header, "'shape'");
    let shape: Vec<usize> = shape_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let n: usize = shape.iter().product();
    let data_start = header_start + header_len;
    let raw = &buf[data_start..data_start + n * 4];
    let mut out = Vec::with_capacity(n);
    for &chunk in raw.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(chunk));
    }
    (shape, out)
}

fn extract_str<'a>(header: &'a str, key: &str) -> &'a str {
    let i = header
        .find(key)
        .unwrap_or_else(|| panic!("npy header missing {key}: {header}"));
    let after = &header[i + key.len()..];
    let colon = after.find(':').unwrap();
    let rest = after[colon + 1..].trim_start();
    if let Some(quote) = rest.chars().next().filter(|c| *c == '\'' || *c == '"') {
        let s = &rest[1..];
        let end = s.find(quote).unwrap();
        &s[..end]
    } else {
        let end = rest.find([',', '}']).unwrap();
        rest[..end].trim()
    }
}

fn extract_tuple<'a>(header: &'a str, key: &str) -> &'a str {
    let i = header.find(key).unwrap();
    let after = &header[i + key.len()..];
    let colon = after.find(':').unwrap();
    let rest = &after[colon + 1..];
    let open = rest.find('(').unwrap();
    let close = rest.find(')').unwrap();
    &rest[open + 1..close]
}

fn load_waveform(idx: usize) -> Box<[f32; WaveformLen::USIZE]> {
    let (shape, data) = read_npy_f32(&crate_root().join(format!("misc/npys/waveform_{idx}.npy")));
    assert_eq!(shape, vec![WaveformLen::USIZE]);
    let mut arr = Box::new([0f32; WaveformLen::USIZE]);
    arr.copy_from_slice(&data);
    arr
}

fn load_spectrogram_ref(idx: usize) -> Vec<f32> {
    let (shape, data) =
        read_npy_f32(&crate_root().join(format!("misc/npys/spectrogram_{idx}.npy")));
    assert_eq!(shape, vec![NFrames::USIZE, NBins::USIZE, 1]);
    data
}

struct Stats {
    max: f32,
    p99: f32,
    mean: f32,
}

fn compare(tag: &str, mine: &[[f32; NBins::USIZE]; NFrames::USIZE], reference: &[f32]) -> Stats {
    assert_eq!(reference.len(), NFrames::USIZE * NBins::USIZE);
    let mut diffs: Vec<f32> = Vec::with_capacity(NFrames::USIZE * NBins::USIZE);
    let mut sum_abs = 0.0f64;
    for t in 0..NFrames::USIZE {
        for k in 0..NBins::USIZE {
            let d = (mine[t][k] - reference[t * NBins::USIZE + k]).abs();
            diffs.push(d);
            sum_abs += d as f64;
        }
    }
    diffs.sort_by(f32::total_cmp);
    let mean = (sum_abs / diffs.len() as f64) as f32;
    let p99 = diffs[(diffs.len() as f64 * 0.99) as usize];
    let max = *diffs.last().unwrap();
    eprintln!("  [{tag}] max|D|={max:.3e}  p99|D|={p99:.3e}  mean|D|={mean:.3e}");
    Stats { max, p99, mean }
}

#[test]
fn spectrogram_parity_all_5() {
    let mut p = Preproc::new();
    // MAX_TOL stays loose: near-zero-magnitude bins (notably DC on near-zero-mean
    // frames) get amplified by `ln` when rustfft's accumulation order differs from
    // TF's by a few ULPs; downstream-negligible since the CNN integrates across bins.
    const P99_TOL: f32 = 1e-4;
    const MEAN_TOL: f32 = 2e-5;
    const MAX_TOL: f32 = 2e-2;
    for idx in 0..5 {
        let wf = load_waveform(idx);
        let spec_ref = load_spectrogram_ref(idx);
        let mine = p.spectrogram(&wf);
        let s = compare(&format!("idx={idx}"), &mine, &spec_ref);
        assert!(
            s.p99 < P99_TOL,
            "sample {idx}: p99|D|={} exceeds {P99_TOL}",
            s.p99
        );
        assert!(
            s.mean < MEAN_TOL,
            "sample {idx}: mean|D|={} exceeds {MEAN_TOL}",
            s.mean
        );
        assert!(
            s.max < MAX_TOL,
            "sample {idx}: max|D|={} exceeds {MAX_TOL}",
            s.max
        );
    }
}

/// `spectrogram` and `spectrogram_into` must be bit-identical (shared impl, the
/// former wraps the latter) regardless of `buf`'s prior contents. Guards the
/// engine's buffer-reuse: it reuses one box across frames, so `spectrogram_into`
/// sees the previous frame's values; a read-before-write of any cell would
/// silently corrupt the output. Exercises both seeded-NaN and leftover states.
#[test]
fn spectrogram_into_matches_spectrogram() {
    let mut p1 = Preproc::new();
    let mut p2 = Preproc::new();
    let mut buf = Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);

    for idx in 0..5 {
        let wf = load_waveform(idx);

        let owned = p1.spectrogram(&wf);

        // idx 0 seeds NaN sentinels to prove prior contents are ignored; idx >= 1
        // relies on the leftover previous-frame values already in `buf`.
        if idx == 0 {
            for row in buf.iter_mut() {
                row.fill(f32::NAN);
            }
        }
        p2.spectrogram_into(&wf, &mut buf);

        for t in 0..NFrames::USIZE {
            for k in 0..NBins::USIZE {
                let a = owned[t][k];
                let b = buf[t][k];
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "idx={idx} t={t} k={k}: spectrogram={a:e} spectrogram_into={b:e}",
                );
            }
        }
    }
}
