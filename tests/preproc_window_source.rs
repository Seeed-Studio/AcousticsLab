//! Ignored diagnostic: for each waveform prints the max/mean abs diff against the
//! reference spectrogram for two paths -- the production `Preproc` (window from the
//! bundled `.bin`) and a standalone reimplementation `PreprocWithWindow` fed the
//! `.npy`-sourced window -- to probe which window source moves the DC-bin outlier.

use acousticslab::common::dims::{NBins, NFrames};
use acousticslab::preproc::{FRAME_LEN, Preproc};
use rustfft::{FftPlanner, num_complex::Complex32};

struct PreprocWithWindow {
    fft: std::sync::Arc<dyn rustfft::Fft<f32>>,
    window: [f32; FRAME_LEN],
}

impl PreprocWithWindow {
    fn new(window: [f32; FRAME_LEN]) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self {
            fft: planner.plan_fft_forward(FRAME_LEN),
            window,
        }
    }

    // Standalone spectrogram parameterized on the window. Output-affecting deltas vs
    // Preproc::spectrogram: divides by bare std (no Z_NORM_EPSILON), accumulates mean/var
    // in f64, takes norm().ln() not 0.5*norm_sqr().ln(), and uses a full complex FFT.
    fn spectrogram(&self, pcm: &[f32; 44032]) -> Box<[[f32; NBins::USIZE]; NFrames::USIZE]> {
        use acousticslab::common::dims::HopSamples;
        use acousticslab::preproc::FRONT_PAD;
        let mut buf = [Complex32::new(0.0, 0.0); FRAME_LEN];
        let mut out = Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
        for t in 0..NFrames::USIZE {
            let start_virt = t * HopSamples::USIZE;
            // Index-style mirrors the numerical parity spec; clearer than the iter rewrite clippy proposes.
            #[allow(clippy::needless_range_loop)]
            for j in 0..FRAME_LEN {
                let virt_idx = start_virt + j;
                let sample = if virt_idx < FRONT_PAD {
                    0.0
                } else {
                    pcm[virt_idx - FRONT_PAD]
                };
                buf[j] = Complex32::new(sample * self.window[j], 0.0);
            }
            self.fft.process(&mut buf);
            let row = &mut out[t];
            for k in 0..NBins::USIZE {
                row[k] = buf[k].norm().ln();
            }
        }
        let count = (NFrames::USIZE * NBins::USIZE) as f64;
        let mut sum = 0.0f64;
        for row in out.iter() {
            for &v in row.iter() {
                sum += v as f64;
            }
        }
        let mean = sum / count;
        let mut sq = 0.0f64;
        for row in out.iter() {
            for &v in row.iter() {
                let d = v as f64 - mean;
                sq += d * d;
            }
        }
        let std = (sq / count).sqrt();
        let mean = mean as f32;
        let std = std as f32;
        for row in out.iter_mut() {
            for v in row.iter_mut() {
                *v = (*v - mean) / std;
            }
        }
        out
    }
}

fn read_npy_f32(path: &std::path::Path) -> (Vec<usize>, Vec<f32>) {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let (hs, hl) = if buf[6] == 1 {
        (10, u16::from_le_bytes([buf[8], buf[9]]) as usize)
    } else {
        (
            12,
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
        )
    };
    let header = std::str::from_utf8(&buf[hs..hs + hl]).unwrap();
    let lp = header.find('(').unwrap();
    let rp = header.find(')').unwrap();
    let shape: Vec<usize> = header[lp + 1..rp]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();
    let n: usize = shape.iter().product();
    let data = &buf[hs + hl..];
    let mut out = Vec::with_capacity(n);
    for &c in data.as_chunks::<4>().0.iter().take(n) {
        out.push(f32::from_le_bytes(c));
    }
    (shape, out)
}

#[test]
#[ignore]
fn bundled_window_vs_analytic() {
    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();

    let (shape, bundled) = read_npy_f32(&crate_root.join("misc/npys/window_blackman_2048.npy"));
    assert_eq!(shape, vec![2048]);
    let mut bundled_arr = [0f32; FRAME_LEN];
    bundled_arr.copy_from_slice(&bundled);

    let mut analytic_preproc = Preproc::new();
    let bundled_preproc = PreprocWithWindow::new(bundled_arr);

    for idx in 0..5 {
        let (_, w) = read_npy_f32(&crate_root.join(format!("misc/npys/waveform_{idx}.npy")));
        let mut wav = Box::new([0f32; 44032]);
        wav.copy_from_slice(&w);
        let (_, r) = read_npy_f32(&crate_root.join(format!("misc/npys/spectrogram_{idx}.npy")));

        let a = analytic_preproc.spectrogram(&wav);
        let b = bundled_preproc.spectrogram(&wav);

        let mut max_a = 0f32;
        let mut max_b = 0f32;
        let mut sum_a = 0f64;
        let mut sum_b = 0f64;
        for t in 0..NFrames::USIZE {
            for k in 0..NBins::USIZE {
                let rv = r[t * NBins::USIZE + k];
                let da = (a[t][k] - rv).abs();
                let db = (b[t][k] - rv).abs();
                if da > max_a {
                    max_a = da;
                }
                if db > max_b {
                    max_b = db;
                }
                sum_a += da as f64;
                sum_b += db as f64;
            }
        }
        let n = (NFrames::USIZE * NBins::USIZE) as f64;
        eprintln!(
            "[idx={idx}]  analytic: max|D|={max_a:.3e} mean|D|={:.3e}   bundled: max|D|={max_b:.3e} mean|D|={:.3e}",
            (sum_a / n) as f32,
            (sum_b / n) as f32
        );
    }
}
