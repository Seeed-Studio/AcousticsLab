//! Opus encode wall time, feeding one `PCM_PULL_CHUNK` per call like the
//! daemon's per-period rate; resamples 44.1 -> 48 kHz, emitting ~1.15
//! 20 ms packets/feed (count varies with resampler residue). Production
//! encoder defaults; criterion's 3 s warm_up_time covers predictor warm-up.

use acousticslab::opus_stream::{OpusEngine, PCM_PULL_CHUNK};
use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_encode_one_frame(c: &mut Criterion) {
    // Sine is a reproducible middle-ground (Opus time is content-dependent).
    let pcm: Vec<f32> = (0..PCM_PULL_CHUNK)
        .map(|i| {
            let t = i as f32 / 44_100.0;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.5
        })
        .collect();

    let mut engine = OpusEngine::new().expect("opus encoder init");
    let mut out: Vec<Bytes> = Vec::with_capacity(2);

    // Converge resampler + encoder predictor state before measuring.
    for _ in 0..16 {
        out.clear();
        engine.process_pcm(&pcm, &mut out).expect("warmup encode");
    }

    let mut group = c.benchmark_group("opus_stream/encode_frame");
    group.throughput(Throughput::Elements(PCM_PULL_CHUNK as u64));
    group.bench_function("1024_samples_in_per_call", |b| {
        b.iter(|| {
            out.clear();
            let n = engine
                .process_pcm(black_box(&pcm), &mut out)
                .expect("encode");
            black_box(n)
        });
    });
    group.finish();
}

criterion_group!(benches, bench_encode_one_frame);
criterion_main!(benches);
