//! Dev tool: `Preproc` an arbitrary `WaveformLen` raw LE-f32 waveform to a
//! headerless 43x232 (frame-major, the slowest axis) row-major LE-f32
//! spectrogram, for parity checks vs the official TF model beyond the 5
//! bundled real-speech npys.
//!
//!   cargo run --release --example dump_spectrogram -- <in_wav.f32> <out_spec.f32>

// Dev tool: plain `std::fs` is fine (no daemon write atomicity needed).
#![allow(clippy::disallowed_methods)]

use std::io::Read;

use acousticslab::common::dims::{NBins, NFrames, WaveformLen};
use acousticslab::preproc::Preproc;

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: dump_spectrogram <in_waveform.f32> <out_spectrogram.f32>");
        return std::process::ExitCode::FAILURE;
    }
    let mut bytes = Vec::new();
    match std::fs::File::open(&args[0]).and_then(|mut f| f.read_to_end(&mut bytes)) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("read {}: {e}", args[0]);
            return std::process::ExitCode::FAILURE;
        }
    }
    if bytes.len() != WaveformLen::USIZE * 4 {
        eprintln!(
            "expected {} f32 samples ({} bytes), got {} bytes",
            WaveformLen::USIZE,
            WaveformLen::USIZE * 4,
            bytes.len()
        );
        return std::process::ExitCode::FAILURE;
    }
    let mut pcm = Box::new([0.0f32; WaveformLen::USIZE]);
    for (i, &c) in bytes.as_chunks::<4>().0.iter().enumerate() {
        pcm[i] = f32::from_le_bytes(c);
    }
    let mut preproc = Preproc::new();
    let spec = preproc.spectrogram(&pcm);
    let mut out = Vec::with_capacity(NFrames::USIZE * NBins::USIZE * 4);
    for row in spec.iter() {
        for &v in row.iter() {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    if let Err(e) = std::fs::write(&args[1], &out) {
        eprintln!("write {}: {e}", args[1]);
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}
