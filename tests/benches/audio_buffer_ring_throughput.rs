//! Seqlock ring throughput, mirroring the daemon hot path: `push_period`
//! measures write-side seqlock publish of one ALSA period; `peek_window`
//! measures read-side seqlock double-read of one inference window.
//! Production capacity keeps wrap-around frequency realistic.

use acoustics_lab::audio_buffer::AudioBuffer;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

const CAPACITY: usize = 262_144;
const PERIOD: usize = 1_024; // ALSA period
const WINDOW: usize = 44_032; // inference window = WaveformLen

fn bench_push_period(c: &mut Criterion) {
    let buf = AudioBuffer::new(CAPACITY);
    let mut writer = buf.take_writer();
    let samples: Vec<f32> = (0..PERIOD).map(|i| (i as f32).sin()).collect();

    let mut group = c.benchmark_group("audio_buffer/push_period");
    group.throughput(Throughput::Elements(PERIOD as u64));
    group.bench_function("1024_samples", |b| {
        b.iter(|| {
            writer.push(black_box(&samples));
        });
    });
    group.finish();
}

fn bench_peek_window(c: &mut Criterion) {
    let buf = AudioBuffer::new(CAPACITY);
    let mut writer = buf.take_writer();
    let samples: Vec<f32> = (0..PERIOD).map(|i| (i as f32).sin()).collect();
    // Pre-warm head past peek-distance + window so `peek_into` lands in the
    // Ready (lag = 0) path, not a cold/short-buffer branch.
    let warmup_periods = (WINDOW * 2).div_ceil(PERIOD) + 4;
    for _ in 0..warmup_periods {
        writer.push(&samples);
    }

    let reader = buf.reader_at(WINDOW);
    let mut out = vec![0.0f32; WINDOW];

    let mut group = c.benchmark_group("audio_buffer/peek_window");
    group.throughput(Throughput::Elements(WINDOW as u64));
    group.bench_function("44032_samples", |b| {
        b.iter(|| {
            let _status = reader.peek_into(black_box(&mut out));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_push_period, bench_peek_window);
criterion_main!(benches);
