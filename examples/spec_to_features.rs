//! Dev tool: read a raw LE-f32 row-major `NFrames`x`NBins` spectrogram, run the
//! frozen backbone, write the LE-f32 feature vector. Measures whether a
//! Rust-vs-official-TF (`sc_preproc_model`) preprocessing divergence propagates
//! into the backbone's features.
//!
//!   cargo run --release --example spec_to_features -- <in_spec.f32> <out_feat.f32> [backbone.mpk]

#![allow(clippy::disallowed_methods)]

use std::io::Read;
use std::path::{Path, PathBuf};

use acousticslab::common::dims::{BackboneFeatureDim, NBins, NFrames};
use acousticslab::inference::BurnBackbone;

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: spec_to_features <in_spec.f32> <out_feat.f32> [backbone.mpk]");
        return std::process::ExitCode::FAILURE;
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let backbone_mpk = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("misc/backbones/backbone.mpk"));

    let mut bytes = Vec::new();
    std::fs::File::open(&args[0])
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .unwrap_or_else(|e| panic!("read {}: {e}", args[0]));
    assert_eq!(
        bytes.len(),
        NFrames::USIZE * NBins::USIZE * 4,
        "expected {}x{} f32 spectrogram",
        NFrames::USIZE,
        NBins::USIZE
    );
    let mut spec: Box<[[f32; NBins::USIZE]; NFrames::USIZE]> =
        Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
    let mut it = bytes.as_chunks::<4>().0.iter();
    for row in spec.iter_mut() {
        for v in row.iter_mut() {
            let c = it.next().unwrap();
            *v = f32::from_le_bytes(*c);
        }
    }

    let mut backbone =
        BurnBackbone::load(&backbone_mpk).unwrap_or_else(|e| panic!("load backbone: {e:?}"));
    let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
    backbone
        .infer(&spec, &mut features)
        .unwrap_or_else(|e| panic!("backbone infer: {e:?}"));

    let mut out = Vec::with_capacity(BackboneFeatureDim::USIZE * 4);
    for &v in features.iter() {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&args[1], &out).unwrap_or_else(|e| panic!("write {}: {e}", args[1]));
    std::process::ExitCode::SUCCESS
}
