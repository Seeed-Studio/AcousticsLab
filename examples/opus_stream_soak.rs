//! Soak gate for the streaming encoder: encodes N seconds of synthetic
//! 44.1 kHz audio through `OpusEngine`, decodes every packet via libopus,
//! and asserts no decode errors and an exact sample-count round-trip
//! (encoder pinned to 20 ms / 48 kHz / mono, so each packet decodes to
//! exactly `FRAME_SAMPLES` = 960). A CLI example rather than a `#[test]`
//! because 60 s is too slow for routine `cargo test`; CI covers the same
//! path at 5 s.

use acousticslab::opus_stream::{
    FRAME_SAMPLES, IN_RATE_HZ, OUT_RATE_HZ, OpusEngine, PCM_PULL_CHUNK,
};
use bytes::Bytes;
use opus::{Channels, Decoder};
use std::process::ExitCode;
use std::time::Instant;

fn parse_args() -> (u64, &'static str) {
    let mut seconds: u64 = 60;
    let mut signal: &'static str = "noise";
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--seconds" => {
                seconds = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| usage());
            }
            "--signal" => {
                let v = match it.next() {
                    Some(v) => v,
                    None => usage(),
                };
                signal = match v.as_str() {
                    "noise" => "noise",
                    "tone" => "tone",
                    "silence" => "silence",
                    _ => usage(),
                };
            }
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    (seconds, signal)
}

fn usage() -> ! {
    eprintln!(
        "usage: soak [--seconds N] [--signal noise|tone|silence]\n\
         \n\
         Encodes N seconds of synthetic audio through OpusEngine, \
         decodes every packet, asserts no errors and matched length."
    );
    std::process::exit(2);
}

fn main() -> ExitCode {
    let (seconds, signal) = parse_args();
    let n_in = (IN_RATE_HZ as usize) * (seconds as usize);
    let pcm = generate(signal, n_in);

    eprintln!(
        "soak: signal={signal} seconds={seconds} samples={n_in} ({:.1} MB f32)",
        (n_in * 4) as f64 / 1024.0 / 1024.0,
    );

    let mut engine = match OpusEngine::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: OpusEngine init: {e}");
            return ExitCode::from(3);
        }
    };

    let t_enc_start = Instant::now();
    let mut packets: Vec<Bytes> = Vec::with_capacity(n_in / 882 + 1);
    for chunk in pcm.chunks(PCM_PULL_CHUNK) {
        if let Err(e) = engine.process_pcm(chunk, &mut packets) {
            eprintln!("error: encode failed mid-stream: {e}");
            return ExitCode::from(4);
        }
    }
    let t_enc = t_enc_start.elapsed();

    let mut decoder = match Decoder::new(OUT_RATE_HZ, Channels::Mono) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: decoder init: {e}");
            return ExitCode::from(5);
        }
    };

    let t_dec_start = Instant::now();
    let mut frame_buf = vec![0f32; FRAME_SAMPLES];
    let mut decoded_total = 0usize;
    for (i, pkt) in packets.iter().enumerate() {
        match decoder.decode_float(pkt.as_ref(), &mut frame_buf, false) {
            Ok(n) => decoded_total += n,
            Err(e) => {
                eprintln!("error: packet {i} decode: {e}");
                return ExitCode::from(7);
            }
        }
    }
    let t_dec = t_dec_start.elapsed();

    let expected_decoded = packets.len() * FRAME_SAMPLES;
    if decoded_total != expected_decoded {
        eprintln!(
            "error: decoded sample mismatch: got {decoded_total}, expected {expected_decoded}",
        );
        return ExitCode::from(8);
    }

    let avg_pkt_bytes: f64 =
        packets.iter().map(|p| p.len() as f64).sum::<f64>() / packets.len().max(1) as f64;
    let kbps = (avg_pkt_bytes * 8.0) / 0.020 / 1000.0; // 20 ms per packet

    println!("soak: ok");
    println!("  packets:        {}", packets.len());
    println!("  decoded samples:{decoded_total}");
    println!("  avg packet:     {avg_pkt_bytes:.0} B  ({kbps:.0} kbps)");
    println!(
        "  encode time:    {:.2?}  ({:.1}x real-time)",
        t_enc,
        seconds as f64 / t_enc.as_secs_f64(),
    );
    println!(
        "  decode time:    {:.2?}  ({:.1}x real-time)",
        t_dec,
        seconds as f64 / t_dec.as_secs_f64(),
    );

    ExitCode::SUCCESS
}

fn generate(kind: &str, n: usize) -> Vec<f32> {
    match kind {
        "tone" => (0..n)
            .map(|i| {
                0.5 * (2.0 * std::f32::consts::PI * 1000.0 * (i as f32 / IN_RATE_HZ as f32)).sin()
            })
            .collect(),
        "silence" => vec![0.0; n],
        // LCG white noise matching the unit test so soak exercises the same audio character.
        _ => {
            let mut s: u32 = 0xdeadbeef;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((s >> 8) as f32 / 0xFFFFFF as f32) * 0.5 - 0.25
                })
                .collect()
        }
    }
}
