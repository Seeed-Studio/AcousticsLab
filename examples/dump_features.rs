//! Dump frozen-backbone features for a labeled WAV dataset to a raw f32 file
//! for offline analysis. Path is `read_wav_mono` -> `to_waveform` -> `Preproc`
//! -> `BurnBackbone::infer`; this matches the fine-tune feature path only for
//! single-window clips (<= one ~1 s `WaveformLen`). Fine-tune uses
//! `to_waveform_windows` (up to `MAX_WINDOWS_PER_FILE` windows/recording), so a
//! long recording dumps just its first window where training sees several.
//!
//! `dump_features <dataset_dir> <out_prefix> [per_class_cap] [class_a,...] [backbone.mpk]`
//! writes `<out_prefix>.f32` (row-major LE f32 [N, 2000]),
//! `.labels.txt` (per-row class index), `.classes.txt` (class names, index order).

// Plain `std::fs` writes: dev tool, no daemon atomicity needed.
#![allow(clippy::disallowed_methods)]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use acousticslab::common::dims::{BackboneFeatureDim, NBins, NFrames};
use acousticslab::inference::BurnBackbone;
use acousticslab::preproc::Preproc;
use acousticslab::preproc::wav_io::{self, ResamplerCache};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return Err(
            "usage: dump_features <dataset_dir> <out_prefix> [per_class_cap] [class_a,class_b,...] [backbone.mpk]"
                .into(),
        );
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dataset_dir = PathBuf::from(&args[0]);
    let out_prefix = PathBuf::from(&args[1]);
    let cap: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let class_filter: Option<Vec<String>> = args
        .get(3)
        .filter(|s| !s.is_empty())
        .map(|s| s.split(',').map(|c| c.trim().to_string()).collect());
    let backbone_mpk = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("misc/backbones/backbone.mpk"));

    let mut backbone =
        BurnBackbone::load(&backbone_mpk).map_err(|e| format!("load backbone: {e:?}"))?;
    let mut preproc = Preproc::new();

    let mut class_dirs: Vec<(String, PathBuf)> = std::fs::read_dir(&dataset_dir)
        .map_err(|e| format!("read {}: {e}", dataset_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| (e.file_name().to_string_lossy().into_owned(), e.path()))
        .filter(|(n, _)| !n.starts_with('.'))
        .filter(|(n, _)| class_filter.as_ref().is_none_or(|f| f.contains(n)))
        .collect();
    class_dirs.sort_by(|a, b| a.0.cmp(&b.0));
    if class_dirs.is_empty() {
        return Err("no class folders matched".into());
    }

    let mut feat_file = std::fs::File::create(format!("{}.f32", out_prefix.display()))
        .map_err(|e| format!("create .f32: {e}"))?;
    let mut labels: Vec<usize> = Vec::new();
    let mut classes: Vec<String> = Vec::new();
    let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
    let mut spec: Box<[[f32; NBins::USIZE]; NFrames::USIZE]> =
        Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);

    for (class_idx, (name, dir)) in class_dirs.iter().enumerate() {
        classes.push(name.clone());
        let mut wavs: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("read {}: {e}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "wav"))
            .collect();
        wavs.sort();
        let mut kept = 0usize;
        for wav in wavs.iter().take(cap) {
            let mut cache = ResamplerCache::empty();
            let pcm = match wav_io::read_wav_mono(wav)
                .and_then(|(sr, mono)| wav_io::to_waveform(sr, mono, &mut cache))
            {
                Ok(p) => p,
                Err(_) => continue,
            };
            preproc.spectrogram_into(&pcm, &mut spec);
            // Silence dumps (finite plane); only faulty non-finite planes skip.
            if spec
                .as_slice()
                .as_flattened()
                .iter()
                .any(|v| !v.is_finite())
            {
                continue;
            }
            if backbone.infer(&spec, &mut features).is_err() {
                continue;
            }
            let mut bytes = Vec::with_capacity(BackboneFeatureDim::USIZE * 4);
            for &v in features.iter() {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            feat_file
                .write_all(&bytes)
                .map_err(|e| format!("write .f32: {e}"))?;
            labels.push(class_idx);
            kept += 1;
        }
        println!("class [{class_idx}] {name:<24} kept {kept}");
    }
    feat_file.flush().map_err(|e| format!("flush .f32: {e}"))?;

    let labels_txt = labels
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(format!("{}.labels.txt", out_prefix.display()), labels_txt)
        .map_err(|e| format!("write labels: {e}"))?;
    std::fs::write(
        format!("{}.classes.txt", out_prefix.display()),
        classes.join("\n"),
    )
    .map_err(|e| format!("write classes: {e}"))?;
    println!(
        "wrote {} rows x {} dims -> {}.f32",
        labels.len(),
        BackboneFeatureDim::USIZE,
        out_prefix.display()
    );
    Ok(())
}
