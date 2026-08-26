//! Dev tool: regenerate the in-tree fixture set under `misc/`.
//!
//! Lives outside the test harness because it rewrites tracked files;
//! running it under `cargo test --include-ignored` would mutate
//! fixtures and churn git status with non-deterministic Burn `ParamId`
//! drift on every run. Backbone-vs-upstream parity is instead verified
//! by a separate `#[ignore]` test that is read-only.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use acousticslab::common::dims::{BackboneFeatureDim, WaveformLen};
use acousticslab::common::ids::HeadId;
use acousticslab::converter::{convert_tfjs, read_tfjs_labels};
use acousticslab::file_mgr::FsServiceImpl;
use acousticslab::inference::{BurnBackbone, HotHead, head_forward, softmax_into};
use acousticslab::preproc::Preproc;

fn crate_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("");
    let result = match cmd {
        "default-head" => regen_default_head(),
        "npys" => regen_npys(),
        "all" => regen_default_head().and_then(|_| regen_npys()),
        "" | "-h" | "--help" | "help" => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        other => {
            eprintln!("unknown subcommand {other:?}\n");
            print_usage();
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("regen failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    eprintln!(
        "usage: cargo run --release --example regen_fixtures -- <subcommand>\n\
         \n\
         subcommands:\n\
           default-head    regenerate misc/heads/default/{{head.mpk,labels.txt}}\n\
           npys            regenerate misc/npys/{{logits,probs}}_{{0..4}}.npy\n\
           all             run default-head then npys"
    );
}

fn regen_default_head() -> Result<(), String> {
    let root = crate_root();
    let src = root.join("misc/models");
    let dst_dir = root.join("misc/heads/default");

    require_exists(
        &src.join("model.json"),
        Some("run misc/models/get_tfjs_sc_model.sh"),
    )?;
    require_exists(&dst_dir, None)?;

    let labels =
        read_tfjs_labels(&src.join("metadata.json")).map_err(|e| format!("read labels: {e}"))?;

    let mpk_bytes = convert_to_mpk_bytes(&src, &labels)?;
    let labels_blob = labels.join("\n") + "\n";
    publish_atomic(&dst_dir.join("head.mpk"), &mpk_bytes)?;
    publish_atomic(&dst_dir.join("labels.txt"), labels_blob.as_bytes())?;

    println!(
        "default-head: {} ({} bytes, {} classes) + labels.txt ({} bytes)",
        dst_dir.join("head.mpk").display(),
        mpk_bytes.len(),
        labels.len(),
        labels_blob.len(),
    );
    Ok(())
}

fn convert_to_mpk_bytes(src_tfjs: &Path, labels: &[String]) -> Result<Vec<u8>, String> {
    let staging = tempfile::tempdir().map_err(|e| format!("staging tempdir: {e}"))?;
    let fs_root = tempfile::tempdir().map_err(|e| format!("fs root tempdir: {e}"))?;
    let fs = FsServiceImpl::new(fs_root.path().to_path_buf());
    let arts = convert_tfjs(src_tfjs, labels, staging.path(), &fs)
        .map_err(|e| format!("convert_tfjs: {e}"))?;
    if arts.n_classes != labels.len() {
        return Err(format!(
            "converter n_classes={} != labels.len()={}",
            arts.n_classes,
            labels.len(),
        ));
    }
    std::fs::read(&arts.head_mpk).map_err(|e| format!("read converter head.mpk: {e}"))
}

fn regen_npys() -> Result<(), String> {
    let root = crate_root();
    let backbone_path = root.join("misc/backbones/backbone.mpk");
    let head_path = root.join("misc/heads/default/head.mpk");
    let labels_path = root.join("misc/heads/default/labels.txt");

    for p in [&backbone_path, &head_path, &labels_path] {
        require_exists(p, None)?;
    }

    // Load head + backbone once (each costs ~5-200 ms); the loop reuses both handles.
    let head = HotHead::load(&head_path, &labels_path, HeadId::new())
        .map_err(|e| format!("load head: {e:?}"))?;
    let n_classes = head.snapshot().n_classes;
    println!("npys: regenerating against {n_classes}-class head");

    let mut backbone =
        BurnBackbone::load(&backbone_path).map_err(|e| format!("load backbone: {e:?}"))?;

    for sample_idx in 0..5 {
        let waveform_path = root.join(format!("misc/npys/waveform_{sample_idx}.npy"));
        let (_, pcm_vec) = read_npy_f32(&waveform_path);
        if pcm_vec.len() != WaveformLen::USIZE {
            return Err(format!(
                "{}: expected {} samples, got {}",
                waveform_path.display(),
                WaveformLen::USIZE,
                pcm_vec.len(),
            ));
        }
        let pcm: &[f32; WaveformLen::USIZE] = pcm_vec
            .as_slice()
            .try_into()
            .expect("pcm length matches WaveformLen");

        let mut preproc = Preproc::new();
        let spec = preproc.spectrogram(pcm);

        let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
        backbone
            .infer(&spec, &mut features)
            .map_err(|e| format!("backbone infer (sample {sample_idx}): {e:?}"))?;

        let snap = head.snapshot();
        let mut logits = vec![0.0f32; snap.n_classes];
        head_forward(&features[..], &snap.weight, &snap.bias, &mut logits);

        let mut probs = vec![0.0f32; snap.n_classes];
        softmax_into(&logits, &mut probs);

        write_npy_f32(
            &root.join(format!("misc/npys/logits_{sample_idx}.npy")),
            &[snap.n_classes],
            &logits,
        )?;
        write_npy_f32(
            &root.join(format!("misc/npys/probs_{sample_idx}.npy")),
            &[snap.n_classes],
            &probs,
        )?;

        let (top_idx, top_p) = probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));
        let top_label = snap.labels.get(top_idx).map(String::as_str).unwrap_or("?");
        println!("  sample {sample_idx}: top-1={top_label} (idx {top_idx}, p={top_p:.4})");
    }
    Ok(())
}

fn require_exists(path: &Path, hint: Option<&str>) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    match hint {
        Some(h) => Err(format!("missing {}: {h}", path.display())),
        None => Err(format!("missing {}", path.display())),
    }
}

/// Crash-safe publish via fsync'd tempfile-in-dest-dir + rename + chmod 0o644:
/// a `Ctrl-C` mid-write leaves the destination fully old or fully new, never torn.
fn publish_atomic(dst: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let dir = dst
        .parent()
        .ok_or_else(|| format!("{} has no parent", dst.display()))?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir)
        .map_err(|e| format!("tempfile in {}: {e}", dir.display()))?;
    tmp.write_all(bytes)
        .map_err(|e| format!("write tmp: {e}"))?;
    tmp.flush().map_err(|e| format!("flush tmp: {e}"))?;
    tmp.as_file()
        .sync_all()
        .map_err(|e| format!("fsync tmp: {e}"))?;
    tmp.persist(dst)
        .map_err(|e| format!("persist {}: {e}", dst.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(dst)
            .map_err(|e| format!("stat {}: {e}", dst.display()))?
            .permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(dst, perms)
            .map_err(|e| format!("chmod {}: {e}", dst.display()))?;
    }
    Ok(())
}

/// Minimal NPY f32 reader, inlined to avoid exposing `crate::inference::npy` as public API.
fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
        .read_to_end(&mut buf)
        .unwrap();
    assert_eq!(
        &buf[0..6],
        b"\x93NUMPY",
        "{} not an npy file",
        path.display()
    );
    let (header_start, header_len) = match buf[6] {
        1 => (10, u16::from_le_bytes([buf[8], buf[9]]) as usize),
        2 | 3 => (
            12,
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
        ),
        v => panic!("unsupported npy major version {v}"),
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
    for &c in data.as_chunks::<4>().0.iter().take(n) {
        out.push(f32::from_le_bytes(c));
    }
    (shape, out)
}

/// Minimal NPY f32 writer; pads the header so `preamble + header_len` is a 64-byte
/// multiple (numpy's alignment convention) and publishes crash-safely via [`publish_atomic`].
fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> Result<(), String> {
    let n: usize = shape.iter().product();
    if data.len() != n {
        return Err(format!(
            "data length {} does not match shape product {n}",
            data.len(),
        ));
    }
    let shape_str: String = shape.iter().map(|s| format!("{s}, ")).collect();
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({shape_str}), }}");
    let preamble = 10usize; // magic(6) + major(1) + minor(1) + len(2)
    let mut header_len = dict.len() + 1; // dict + trailing '\n'
    while !(preamble + header_len).is_multiple_of(64) {
        header_len += 1;
    }
    let pad = header_len - dict.len() - 1;
    let header = format!("{dict}{}\n", " ".repeat(pad));
    debug_assert_eq!(header.len(), header_len);

    let mut buf: Vec<u8> = Vec::with_capacity(preamble + header_len + n * 4);
    buf.extend_from_slice(b"\x93NUMPY");
    buf.push(1);
    buf.push(0);
    buf.extend_from_slice(&(header_len as u16).to_le_bytes());
    buf.extend_from_slice(header.as_bytes());
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    publish_atomic(path, &buf)
}
