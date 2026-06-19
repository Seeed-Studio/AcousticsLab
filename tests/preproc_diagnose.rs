//! Diagnostic: locate where preproc spectrogram error vs reference concentrates.
use acousticslab::common::dims::{NBins, NFrames, WaveformLen};
use acousticslab::preproc::Preproc;
use std::io::Read;
use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    assert_eq!(&buf[0..6], b"\x93NUMPY");
    let (header_start, header_len) = if buf[6] == 1 {
        (10, u16::from_le_bytes([buf[8], buf[9]]) as usize)
    } else {
        (
            12,
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
        )
    };
    let header = std::str::from_utf8(&buf[header_start..header_start + header_len]).unwrap();
    let shape_open = header.find('(').unwrap();
    let shape_close = header.find(')').unwrap();
    let shape: Vec<usize> = header[shape_open + 1..shape_close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let n: usize = shape.iter().product();
    let data = &buf[header_start + header_len..];
    let mut out = Vec::with_capacity(n);
    for c in data.chunks_exact(4).take(n) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    (shape, out)
}

#[test]
#[ignore] // run explicitly with: cargo test --release -- --ignored diagnose
fn diagnose_sample_2() {
    let mut p = Preproc::new();
    let (_, wf) = read_npy_f32(&crate_root().join("misc/npys/waveform_2.npy"));
    let mut arr = Box::new([0f32; WaveformLen::USIZE]);
    arr.copy_from_slice(&wf);
    let mine = p.spectrogram(&arr);
    let (_, r) = read_npy_f32(&crate_root().join("misc/npys/spectrogram_2.npy"));

    let mut diffs: Vec<(f32, usize, usize, f32, f32)> = Vec::new();
    for t in 0..NFrames::USIZE {
        for k in 0..NBins::USIZE {
            let rv = r[t * NBins::USIZE + k];
            let m = mine[t][k];
            diffs.push(((m - rv).abs(), t, k, m, rv));
        }
    }
    diffs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!("worst 15 bins for sample 2:");
    for (d, t, k, m, r) in diffs.iter().take(15) {
        eprintln!("  t={t:2} k={k:3}  D={d:.3e}  mine={m:+.6}  ref={r:+.6}");
    }
}
