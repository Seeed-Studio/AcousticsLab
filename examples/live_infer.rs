//! Live microphone inference test (macOS): runs the REAL inference kernels
//! (`Preproc` -> `BurnBackbone` -> `head_forward` -> softmax) on a sliding 1 s
//! window of mic audio, to check whether real mic audio (vs training WAVs)
//! reproduces "random" behavior using the daemon's compute. Mirrors only the
//! inference content path, not the daemon's capture subsystem.
//!
//! Capture uses a `sox -d` / `ffmpeg -f avfoundation` subprocess (raw f32le
//! mono PCM) rather than a Rust crate on purpose: `cpal` pulls an `alsa-sys`
//! major conflicting with the pinned `alsa =0.11`, perturbing production capture.
//!
//! Prereq: `brew install sox` (or `ffmpeg`) + grant terminal Mic permission.
//! Run: `cargo run --release --example live_infer -- [head.mpk] [labels.txt] [backbone.mpk]`
//! (defaults under misc/; labels = one per line in the head's class order).
//! Override capture with LIVE_INFER_CAPTURE_CMD="<cmd emitting f32le mono 44100>".

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

// `parking_lot::Mutex`: std::sync::Mutex is disallowed by clippy.toml; also
// non-poisoning, so lock sites stay infallible.
use parking_lot::Mutex;

use acoustics_lab::common::dims::{BackboneFeatureDim, NBins, NFrames, WaveformLen};
use acoustics_lab::common::ids::HeadId;
use acoustics_lab::inference::{
    BurnBackbone, HotHead, head_forward, softmax_into, top_k_indices_into,
};
use acoustics_lab::preproc::Preproc;

/// Capture subprocess emits PCM at exactly this rate, so the external tool
/// (not us) does the resampling.
const TARGET_SR: u32 = 44_100;
const WINDOW: usize = WaveformLen::USIZE;
/// Poll/emit cadence (~4 Hz), faster than the daemon's 1 Hz hop for responsiveness.
const POLL: Duration = Duration::from_millis(250);

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "-h" || a == "--help") {
        eprintln!(
            "usage: live_infer [head.mpk] [labels.txt] [backbone.mpk]\n\
             needs `sox` or `ffmpeg` on PATH (brew install sox); grant Mic permission; Ctrl-C to stop."
        );
        return Ok(());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let head_mpk = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("misc/heads/default/head.mpk"));
    let labels_txt = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("misc/heads/default/labels.txt"));
    let backbone_mpk = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("misc/backbones/backbone.mpk"));

    let head = HotHead::load(&head_mpk, &labels_txt, HeadId::new())
        .map_err(|e| format!("load head ({}): {e:?}", head_mpk.display()))?;
    let snap = head.snapshot();
    let n_classes = snap.n_classes;
    println!("head: {n_classes} classes {:?}", snap.labels);
    let mut backbone =
        BurnBackbone::load(&backbone_mpk).map_err(|e| format!("load backbone: {e:?}"))?;
    let mut preproc = Preproc::new();

    let (mut child, how) = spawn_capture()?;
    println!(
        "capture: {how}  ->  {TARGET_SR} Hz mono f32\n\
         speak / play audio near the mic; Ctrl-C to stop.\n"
    );
    let mono: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let stdout = child
        .stdout
        .take()
        .ok_or("capture subprocess has no stdout")?;
    let reader_sink = mono.clone();
    std::thread::spawn(move || read_f32_stream(stdout, reader_sink));
    let _guard = ChildGuard(child);

    let mut acc: Vec<f32> = Vec::with_capacity(WINDOW * 2);
    let mut window: Box<[f32; WaveformLen::USIZE]> = Box::new([0.0; WaveformLen::USIZE]);
    let mut spec: Box<[[f32; NBins::USIZE]; NFrames::USIZE]> =
        Box::new([[0.0; NBins::USIZE]; NFrames::USIZE]);
    let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
    let mut logits = vec![0.0f32; n_classes];
    let mut probs = vec![0.0f32; n_classes];
    let mut top = Vec::with_capacity(n_classes);

    loop {
        std::thread::sleep(POLL);
        let chunk = {
            let mut g = mono.lock();
            std::mem::take(&mut *g)
        };
        if chunk.is_empty() {
            continue;
        }
        acc.extend_from_slice(&chunk);
        if acc.len() < WINDOW {
            continue;
        }
        // Classify the LATEST window and drop older samples so polls slide
        // forward without backlog.
        let start = acc.len() - WINDOW;
        window.copy_from_slice(&acc[start..]);
        if start > 0 {
            acc.drain(..start);
        }

        preproc.spectrogram_into(&window, &mut spec);
        if spec
            .as_slice()
            .as_flattened()
            .iter()
            .any(|v| !v.is_finite())
        {
            println!("  (silence / constant input -- frame dropped, as the engine does)");
            continue;
        }
        if let Err(e) = backbone.infer(&spec, &mut features) {
            println!("  backbone error: {e:?}");
            continue;
        }
        head_forward(&features[..], &snap.weight, &snap.bias, &mut logits);
        softmax_into(&logits, &mut probs);
        top_k_indices_into(&probs, 3.min(n_classes), &mut top);

        let line: Vec<String> = top
            .iter()
            .map(|&i| format!("{}={:.2}", snap.labels[i], probs[i]))
            .collect();
        println!("  {}", line.join("  "));
    }
}

/// Spawn a capture subprocess emitting raw f32le mono PCM at [`TARGET_SR`] on
/// stdout: honors `LIVE_INFER_CAPTURE_CMD`, else tries `sox` then `ffmpeg`.
fn spawn_capture() -> Result<(Child, String), String> {
    if let Some(cmd) = std::env::var_os("LIVE_INFER_CAPTURE_CMD") {
        let cmd = cmd.to_string_lossy().into_owned();
        let child = Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawn LIVE_INFER_CAPTURE_CMD: {e}"))?;
        return Ok((child, format!("LIVE_INFER_CAPTURE_CMD: {cmd}")));
    }

    // sox `-d` = default input device.
    let sox = Command::new("sox")
        .args([
            "-q",
            "-d",
            "-t",
            "raw",
            "-e",
            "floating-point",
            "-b",
            "32",
            "-L",
            "-r",
        ])
        .arg(TARGET_SR.to_string())
        .args(["-c", "1", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn();
    match sox {
        Ok(child) => return Ok((child, "sox -d (default input)".into())),
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
            return Err(format!("spawn sox: {e}"));
        }
        Err(_) => { /* sox not installed; fall through to ffmpeg */ }
    }

    // ffmpeg avfoundation `:default` = default audio input, no video.
    let ffmpeg = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-i",
            ":default",
            "-ac",
            "1",
            "-ar",
        ])
        .arg(TARGET_SR.to_string())
        .args(["-f", "f32le", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn();
    match ffmpeg {
        Ok(child) => Ok((child, "ffmpeg -f avfoundation -i :default".into())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Err("neither `sox` nor `ffmpeg` found on PATH.\n\
             install one (e.g. `brew install sox`) or set LIVE_INFER_CAPTURE_CMD to a command \
             that writes raw f32le mono 44100 Hz PCM to stdout."
                .into())
        }
        Err(e) => Err(format!("spawn ffmpeg: {e}")),
    }
}

/// Append f32le samples from the capture child's stdout to the shared buffer,
/// carrying any partial (<4-byte) tail across reads so samples never tear at a
/// read boundary.
fn read_f32_stream(mut stdout: std::process::ChildStdout, sink: Arc<Mutex<Vec<f32>>>) {
    let mut buf = [0u8; 16 * 1024];
    let mut carry: Vec<u8> = Vec::with_capacity(4);
    loop {
        match stdout.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                carry.extend_from_slice(&buf[..n]);
                let full = carry.len() - (carry.len() % 4);
                if full > 0 {
                    {
                        let mut g = sink.lock();
                        for c in carry[..full].chunks_exact(4) {
                            g.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                        }
                    }
                    carry.drain(..full);
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
}

/// Kills + reaps the capture subprocess on drop (error / early-return path) so
/// the mic is released and no orphaned `sox`/`ffmpeg` lingers. On Ctrl-C the
/// drop does NOT run (no SIGINT handler, so no unwind); the child instead dies
/// via the shell's foreground process-group SIGINT.
struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
