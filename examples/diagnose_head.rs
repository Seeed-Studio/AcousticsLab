//! Dev diagnostic for heads that classify "randomly" after deploy: runs a head + the Burn
//! fp32 backbone over a labeled WAV dataset through the REAL inference kernels (no RKNN),
//! replicating the fine-tune feature path (`wav_io` decode/resample + `Preproc` + backbone)
//! over the FIRST 1 s window only (via `to_waveform`), then driving the deployed
//! `head_forward`. Byte-identical to training for sub-1 s clips, but training snippet-chops
//! longer recordings into multiple windows (`to_waveform_windows`) whereas this scores only
//! the first. A divergence from training's reported accuracy is therefore a genuine
//! train<->serve bug on Burn, independent of RKNN/fp16.
//!
//! Dataset: `<dataset_dir>/<class_name>/*.wav` (recursive; class_name must match a labels.txt
//! line). Run: `cargo run --release --example diagnose_head -- <dataset_dir> [head.mpk]
//! [labels.txt] [backbone.mpk]`; defaults head/labels=misc/heads/default/*,
//! backbone=misc/backbones/backbone.mpk.
//!
//! Interpreting it (see the runtime VERDICT HINT): run FIRST on the TRAINING WAVs. HIGH there
//! => head+Burn path consistent with training, so "random" deploy comes from the deploy
//! environment (RKNN fp16 / live-mic windowing) -- re-test through RKNN. ~CHANCE there =>
//! genuine train<->serve bug unrelated to RKNN (feature path / head load / label order), and
//! the confidence stats discriminate (near-uniform softmax = dead head; confident-but-wrong =
//! a permutation). HIGH on train but CHANCE on a HELD-OUT set => overfit (validation_split=0.0
//! publishes the last unregularized epoch and reports train-only accuracy).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use acousticslab::common::dims::{BackboneFeatureDim, NBins, NFrames};
use acousticslab::common::ids::HeadId;
use acousticslab::inference::{BurnBackbone, HotHead, head_forward, softmax_into};
use acousticslab::preproc::Preproc;
use acousticslab::preproc::wav_io::{self, ResamplerCache};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        eprintln!(
            "usage: diagnose_head <dataset_dir> [head.mpk] [labels.txt] [backbone.mpk]\n\
             \n  <dataset_dir>/<class_name>/*.wav  (class_name must match a labels.txt line)\n\
             \ndefaults: head/labels = misc/heads/default/*, backbone = misc/backbones/backbone.mpk\n\
             see the file header for how to interpret the output."
        );
        return ExitCode::FAILURE;
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dataset_dir = PathBuf::from(&args[0]);
    let head_mpk = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("misc/heads/default/head.mpk"));
    let labels_txt = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("misc/heads/default/labels.txt"));
    let backbone_mpk = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("misc/backbones/backbone.mpk"));

    for (what, p) in [
        ("dataset dir", &dataset_dir),
        ("head.mpk", &head_mpk),
        ("labels.txt", &labels_txt),
        ("backbone.mpk", &backbone_mpk),
    ] {
        if !p.exists() {
            return Err(format!("{what} not found: {}", p.display()));
        }
    }

    // Load head + backbone via the SAME public paths the daemon uses.
    let head = HotHead::load(&head_mpk, &labels_txt, HeadId::new())
        .map_err(|e| format!("load head ({}): {e:?}", head_mpk.display()))?;
    let snap = head.snapshot();
    let n_classes = snap.n_classes;
    let labels: &[String] = &snap.labels;
    println!(
        "head: {} classes, labels = {:?}\nbackbone: {}\n",
        n_classes,
        labels,
        backbone_mpk.display()
    );

    let mut backbone =
        BurnBackbone::load(&backbone_mpk).map_err(|e| format!("load backbone: {e:?}"))?;
    let mut preproc = Preproc::new();

    // Match class folders to head class index by NAME; a folder absent from labels.txt is
    // itself a finding (label/folder mismatch) -- reported and skipped.
    let class_dirs = read_class_dirs(&dataset_dir)?;
    let mut examples: Vec<(PathBuf, usize, String)> = Vec::new();
    let mut unmatched: Vec<String> = Vec::new();
    for (name, dir) in &class_dirs {
        match labels.iter().position(|l| l == name) {
            Some(idx) => {
                let mut wavs = Vec::new();
                collect_wavs(dir, &mut wavs)?;
                for w in wavs {
                    examples.push((w, idx, name.clone()));
                }
            }
            None => unmatched.push(name.clone()),
        }
    }
    if !unmatched.is_empty() {
        println!(
            "WARNING: {} dataset folder(s) have no matching label and are SKIPPED: {:?}\n\
             (a folder<->label name mismatch alone makes every such class look 'random')\n",
            unmatched.len(),
            unmatched
        );
    }
    if examples.is_empty() {
        return Err("no usable examples (no class folders matched the head's labels)".into());
    }

    let mut total = 0usize;
    let mut correct = 0usize;
    let mut dropped = 0usize;
    let mut per_class_total = vec![0usize; n_classes];
    let mut per_class_correct = vec![0usize; n_classes];
    let mut pred_histogram = vec![0usize; n_classes];
    let mut sum_top1_prob = 0.0f64;
    let mut sum_margin = 0.0f64;

    let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
    let mut spec: Box<[[f32; NBins::USIZE]; NFrames::USIZE]> =
        Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
    let mut logits = vec![0.0f32; n_classes];
    let mut probs = vec![0.0f32; n_classes];

    for (wav, true_idx, _name) in &examples {
        // Fine-tune feature path for the first 1 s window only (read_wav_mono -> to_waveform ->
        // Preproc); training snippet-chops longer recordings via to_waveform_windows.
        let mut cache = ResamplerCache::empty();
        let pcm = match wav_io::read_wav_mono(wav)
            .and_then(|(sr, mono)| wav_io::to_waveform(sr, mono, &mut cache))
        {
            Ok(p) => p,
            Err(_) => {
                dropped += 1;
                continue;
            }
        };
        preproc.spectrogram_into(&pcm, &mut spec);
        if spec
            .as_slice()
            .as_flattened()
            .iter()
            .any(|v| !v.is_finite())
        {
            // Fault gate, as in the engine; silence is finite and scored.
            dropped += 1;
            continue;
        }
        if backbone.infer(&spec, &mut features).is_err() {
            dropped += 1;
            continue;
        }
        head_forward(&features[..], &snap.weight, &snap.bias, &mut logits);
        softmax_into(&logits, &mut probs);

        let pred = argmax(&probs);
        let (top1, top2) = top1_top2(&probs);
        total += 1;
        per_class_total[*true_idx] += 1;
        pred_histogram[pred] += 1;
        sum_top1_prob += top1 as f64;
        sum_margin += (top1 - top2) as f64;
        if pred == *true_idx {
            correct += 1;
            per_class_correct[*true_idx] += 1;
        }
    }

    if total == 0 {
        return Err(format!(
            "every example was dropped ({dropped} decode/non-finite drops); nothing to score"
        ));
    }

    let acc = correct as f64 / total as f64;
    let chance = 1.0 / n_classes as f64;
    println!("=== RESULTS (Burn backbone, deployed head kernel) ===");
    println!("scored {total} examples ({dropped} dropped to decode/non-finite input)");
    println!(
        "overall top-1 accuracy: {:.1}%   (chance = {:.1}% for {n_classes} classes)",
        acc * 100.0,
        chance * 100.0
    );
    println!(
        "mean top-1 prob: {:.3}   mean (top1-top2) margin: {:.3}",
        sum_top1_prob / total as f64,
        sum_margin / total as f64
    );
    println!("\nper-class accuracy:");
    for c in 0..n_classes {
        let t = per_class_total[c];
        let acc_c = if t == 0 {
            f64::NAN
        } else {
            per_class_correct[c] as f64 / t as f64
        };
        println!(
            "  [{c:>2}] {:<24} {:>5}/{:<5} = {:>5.1}%   (predicted {} times)",
            labels[c],
            per_class_correct[c],
            t,
            acc_c * 100.0,
            pred_histogram[c],
        );
    }

    let dominant = pred_histogram
        .iter()
        .enumerate()
        .max_by_key(|&(_, &n)| n)
        .map(|(i, &n)| (i, n))
        .unwrap();
    let dominant_frac = dominant.1 as f64 / total as f64;
    let mean_top1 = sum_top1_prob / total as f64;

    println!("\n=== VERDICT HINT ===");
    if acc >= 0.8 {
        println!(
            "Accuracy is HIGH on this set. The head + Burn deploy path are CONSISTENT with training.\n\
             => If real deploy is random, the cause is the DEPLOY ENVIRONMENT (RKNN fp16 backbone\n\
                or live-mic windowing), not the head. Re-run this through the RKNN backbone next."
        );
    } else if acc <= chance * 1.5 {
        if mean_top1 < chance * 2.0 {
            println!(
                "Accuracy is ~CHANCE and softmax is NEAR-UNIFORM (mean top-1 prob {:.3} vs chance {:.3}).\n\
                 => The head is effectively DEAD: untrained / zeroed / NaN-collapsed / wrong feature scale.\n\
                    Check that training actually converged and that the published head is the one evaluate() scored.",
                mean_top1, chance
            );
        } else if dominant_frac > 0.6 {
            println!(
                "Accuracy is ~CHANCE but predictions COLLAPSE onto class [{}] {:?} ({:.0}% of all preds)\n\
                 with confident softmax (mean top-1 prob {:.3}).\n\
                 => Mode collapse: the head learned a constant. Suspect a degenerate/imbalanced split\n\
                    or that only train accuracy (val_split=0) was reported.",
                dominant.0,
                labels[dominant.0],
                dominant_frac * 100.0,
                mean_top1
            );
        } else {
            println!(
                "Accuracy is ~CHANCE but softmax is CONFIDENT (mean top-1 prob {:.3}).\n\
                 => Confidently-WRONG: a permutation. Suspect feature-basis / label-order /\n\
                    weight-orientation mismatch between train and serve. Since this is the Burn path,\n\
                    RKNN is NOT involved -- the bug is in the feature path, head load, or label order.\n\
                 NOTE: if you ran this on a HELD-OUT set, this can instead be plain OVERFIT.\n\
                    Re-run on the TRAINING WAVs: still chance there => real bug; high there => overfit.",
                mean_top1
            );
        }
    } else {
        println!(
            "Accuracy is BETWEEN chance and good ({:.1}%). Likely a covariate shift (e.g. train/serve\n\
             windowing, sample-rate, or gain) and/or a weakly-regularized head -- degraded, not random.\n\
             Re-run on a held-out set to separate overfit from a genuine generalization gap.",
            acc * 100.0
        );
    }

    Ok(())
}

/// Sorted (name, path) of `dir`'s immediate non-hidden subdirectories, mirroring the
/// fine-tuner's class-folder discovery.
fn read_class_dirs(dir: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let mut out = Vec::new();
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("read dataset dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read entry: {e}"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            out.push((name, path));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Recursively collect non-hidden `.wav` files under `dir`.
fn collect_wavs(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("read class dir {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read entry: {e}"))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            collect_wavs(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "wav") {
            out.push(path);
        }
    }
    Ok(())
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Largest and second-largest values in `xs`.
fn top1_top2(xs: &[f32]) -> (f32, f32) {
    let mut t1 = f32::NEG_INFINITY;
    let mut t2 = f32::NEG_INFINITY;
    for &v in xs {
        if v > t1 {
            t2 = t1;
            t1 = v;
        } else if v > t2 {
            t2 = v;
        }
    }
    if t2 == f32::NEG_INFINITY {
        t2 = 0.0;
    }
    (t1, t2)
}
