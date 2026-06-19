//! Per-window Burn fp32 (CPU) inference latency: PCM -> spectrogram -> backbone
//! -> head logits, mirroring the engine. Skips peek_into, softmax/top-k, and
//! proto encode/broadcast to isolate hot compute. Random Burn-init weights give
//! meaningless logits but production matmul/conv shapes, so wall-clock is
//! representative (no asserts). RKNN not benched (needs Linux+aarch64+NPU).

use acousticslab::common::dims::{BackboneFeatureDim, NBins, NFrames, WaveformLen};
use acousticslab::inference::head_forward;
use acousticslab::model::Backbone;
use acousticslab::preproc::Preproc;
use burn::backend::ndarray::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

const N_CLASSES: usize = 10;

type B = NdArray<f32>;

fn bench_run_window_burn(c: &mut Criterion) {
    // Sine+harmonic excites non-trivial FFT/conv bins; silence would be an
    // unrealistic best case.
    let pcm: Box<[f32; WaveformLen::USIZE]> = {
        let mut buf = Box::new([0.0f32; WaveformLen::USIZE]);
        for (i, s) in buf.iter_mut().enumerate() {
            let t = i as f32 / 44_100.0;
            *s = 0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 2_500.0 * t).sin();
        }
        buf
    };

    let mut preproc = Preproc::new();
    let device = NdArrayDevice::default();
    let backbone = Backbone::<B>::new(&device);

    // Scratch matching the engine's per-loop buffers (no per-window alloc).
    let mut spec = Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
    let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
    let head_weight: Vec<f32> = vec![0.001; BackboneFeatureDim::USIZE * N_CLASSES];
    let head_bias: Vec<f32> = vec![0.0; N_CLASSES];
    let mut logits: Vec<f32> = vec![0.0; N_CLASSES];

    let mut group = c.benchmark_group("inference/run_window_burn");
    group.throughput(Throughput::Elements(1));
    group.bench_function("preproc+backbone+head", |b| {
        b.iter(|| {
            preproc.spectrogram_into(black_box(&pcm), &mut spec);

            // The per-window Vec alloc is unavoidable: Burn's TensorData
            // consumes its Vec, same as the engine sees in production.
            let flat: Vec<f32> = spec.as_slice().as_flattened().to_vec();
            let input = burn::tensor::Tensor::<B, 4>::from_data(
                burn::tensor::TensorData::new(flat, [1, 1, NFrames::USIZE, NBins::USIZE]),
                &device,
            );
            let output = backbone.forward(input);
            let data = output.into_data();
            let slice = data.as_slice::<f32>().expect("burn slice");
            features.copy_from_slice(slice);

            head_forward(&features[..], &head_weight, &head_bias, &mut logits);
            black_box(&logits[0]);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_run_window_burn);
criterion_main!(benches);
