//! Continuous Opus encoder pipeline: 44.1 kHz mono PCM -> streaming
//! resampler -> 20 ms Opus packets -> broadcast to UDS/WS subscribers.
//!
//! Subscriber-driven pause: at 0 subscribers the task parks on
//! `subscribers.changed()` consuming nothing; on resume it rebuilds
//! encoder+resampler and `seek_latest(BACKLOG_SAMPLES)` to drop backlog.
//!
//! Resampler reset clicks the FIR mid-stream, so NEVER reset in steady
//! state; DO reset after `Lagged` (the glitch masks the transient),
//! always paired with `seek_latest()`.

#![warn(missing_debug_implementations)]

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use bytes::Bytes;
use opus::{Application, Bitrate, Channels, Encoder};
use thiserror::Error;
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;

use crate::audio_buffer::{ReadStatus, Reader};
use crate::dsp::resample::Streaming;

/// Output rate; Opus accepts only 8/12/16/24/48 kHz.
pub const OUT_RATE_HZ: u32 = 48_000;

/// Source sample rate of the [`crate::audio_buffer::AudioBuffer`] feed.
pub const IN_RATE_HZ: u32 = 44_100;

pub const FRAME_MS: u32 = 20;

pub const FRAME_SAMPLES: usize = (OUT_RATE_HZ as usize) * (FRAME_MS as usize) / 1000;

/// Per-call output cap; Opus RFC limit is 1275 B/packet. 4000 (libopus
/// repacketize rec) absorbs any VBR spike.
pub const MAX_PACKET_BYTES: usize = 4000;

/// 32 kbps: above Opus transparent-speech threshold (~24 kbps), ~half the
/// auto default; peaks stay below `MAX_PACKET_BYTES`.
pub const BITRATE_BPS: i32 = 32_000;

/// Complexity (0..=10); 5 vs libopus default 9-10 is ~30% less encode CPU,
/// audibly indistinguishable here.
pub const COMPLEXITY: i32 = 5;

/// Per-pull chunk ~23 ms at 44.1 kHz, matching resampler
/// `input_frames_next()`; one chunk -> one or two 20 ms windows.
pub const PCM_PULL_CHUNK: usize = 1024;

/// Resume pre-roll: seek ~100 ms behind the live edge so the first packet
/// is full, not silence-padded.
pub const BACKLOG_SAMPLES: usize = 4096;

/// Active-state wait timeout: react to subscriber changes without spinning.
const ACTIVE_TICK: Duration = Duration::from_millis(10);

/// libopus / resampler pipeline failures. Each, propagated, ENDS the sole
/// encoder task for ALL subscribers. `BadPcm` is unreachable on
/// arbitrator-validated input; `EncoderInternal`/`EmptyEncode`/
/// `ResamplerInternal` are FFI/internal invariant faults, not bad input.
/// `opus::Error` is boxed so an `opus` version bump is not a SemVer break;
/// `stage` keeps log-grep.
#[derive(Debug, Error)]
pub enum OpusError {
    /// Libopus FFI error; `stage` is `"init"`, `"encode"`, or `"reset"`.
    #[error("opus encoder ({stage}) failed: {source}")]
    EncoderInternal {
        stage: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("opus encode produced 0 bytes")]
    EmptyEncode,
    /// Non-finite PCM at `index`; whole slice rejected (libopus
    /// `encode_float` undefined on non-finite, sinc FIR would smear it
    /// forward). Unreachable: `mic_arbitrator` clamps to 0.0 first.
    #[error("non-finite PCM sample at index {index}")]
    BadPcm { index: usize },
    #[error("opus resampler internal: {source}")]
    ResamplerInternal {
        #[from]
        source: crate::dsp::resample::StreamingResampleError,
    },
}

impl OpusError {
    fn encoder_internal(
        stage: &'static str,
        e: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        OpusError::EncoderInternal {
            stage,
            source: Box::new(e),
        }
    }
}

impl crate::common::error::Categorized for OpusError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        crate::common::error::ErrorKind::Internal
    }
}

/// 44.1 kHz f32 PCM in -> 20 ms Opus packets out. Callers rebuild a fresh
/// `OpusEngine` after a discontinuity to drop stale FIR / predictor state.
pub struct OpusEngine {
    encoder: Encoder,
    resampler: Streaming,
    /// One-packet scratch; prefix copied out as owned `Bytes` since
    /// receivers outlive it.
    encode_scratch: Vec<u8>,
    /// Resampler output; encoder reads `FRAME_SAMPLES` from the front, then
    /// `drain`s the consumed prefix to avoid a per-packet frame copy.
    out_pcm: Vec<f32>,
}

impl std::fmt::Debug for OpusEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpusEngine")
            .field("resampler", &self.resampler)
            .field("out_pcm_len", &self.out_pcm.len())
            .finish_non_exhaustive()
    }
}

impl OpusEngine {
    pub fn new() -> Result<Self, OpusError> {
        let mut encoder = Encoder::new(OUT_RATE_HZ, Channels::Mono, Application::Audio)
            .map_err(|e| OpusError::encoder_internal("init", e))?;
        encoder
            .set_bitrate(Bitrate::Bits(BITRATE_BPS))
            .map_err(|e| OpusError::encoder_internal("init", e))?;
        encoder
            .set_complexity(COMPLEXITY)
            .map_err(|e| OpusError::encoder_internal("init", e))?;
        Ok(Self {
            encoder,
            resampler: Streaming::new(IN_RATE_HZ, OUT_RATE_HZ),
            encode_scratch: vec![0u8; MAX_PACKET_BYTES],
            out_pcm: Vec::with_capacity(FRAME_SAMPLES * 4),
        })
    }

    /// Feed `pcm_44k_in` and drain every now-ready 20 ms packet into `out`,
    /// returning the count pushed. One exactly-sized `Bytes` per packet; all
    /// other buffers reuse capacity.
    pub fn process_pcm(
        &mut self,
        pcm_44k_in: &[f32],
        out: &mut Vec<Bytes>,
    ) -> Result<usize, OpusError> {
        // Reject non-finite up front (libopus undefined; sinc FIR smears).
        if let Some(index) = pcm_44k_in.iter().position(|s| !s.is_finite()) {
            return Err(OpusError::BadPcm { index });
        }
        self.resampler.process(pcm_44k_in)?;
        self.resampler.drain_output_into(&mut self.out_pcm);

        let mut emitted = 0usize;
        // Cursor-then-single-drain: multi-frame emissions cost O(tail), not
        // O(N^2) from per-frame `drain(..FRAME_SAMPLES)` tail-shifts.
        let mut offset = 0usize;
        while self.out_pcm.len() - offset >= FRAME_SAMPLES {
            let n = self
                .encoder
                .encode_float(
                    &self.out_pcm[offset..offset + FRAME_SAMPLES],
                    &mut self.encode_scratch,
                )
                .map_err(|e| OpusError::encoder_internal("encode", e))?;
            if n == 0 {
                return Err(OpusError::EmptyEncode);
            }
            // Owned `Bytes`: receivers outlive `encode_scratch`.
            out.push(Bytes::copy_from_slice(&self.encode_scratch[..n]));
            offset += FRAME_SAMPLES;
            emitted += 1;
        }
        if offset > 0 {
            self.out_pcm.drain(..offset);
        }
        Ok(emitted)
    }

    /// Reset FIR + clear SILK/CELT predictor after a `Lagged` (the glitch
    /// masks the FIR transient), without rebuilding the engine.
    pub fn reset_after_discontinuity(&mut self) -> Result<(), OpusError> {
        self.resampler.reset_after_discontinuity();
        self.out_pcm.clear();
        self.encoder
            .reset_state()
            .map_err(|e| OpusError::encoder_internal("reset", e))?;
        Ok(())
    }
}

/// Async encoder run loop. Paused at `subscribers == 0` (engine dropped,
/// reader held); Active at `> 0` rebuilds the engine, seeks
/// `BACKLOG_SAMPLES` behind the edge, then pumps/encodes/broadcasts.
///
/// `packets_encoded` (Relaxed) counts encoder progress per packet, NOT
/// delivery (increments even with zero subscribers); the heartbeat reads it
/// to detect a stall.
///
/// `timing_anchor`, when present, projects each frame's
/// `t_us_capture_monotonic` to the chunk's first-sample capture time via
/// [`crate::common::time::capture_us_for`]; `None` falls back to a
/// publish-time `CaptureTime::now()` stamp.
///
/// A propagated `Err` ends streaming for all subscribers until an external
/// supervisor restarts the daemon (no self-restart, no per-subscriber 1011
/// close); only the heartbeat signals it.
pub async fn run(
    mut reader: Reader,
    mut subscribers: watch::Receiver<usize>,
    out: broadcast::Sender<Bytes>,
    shutdown: CancellationToken,
    packets_encoded: Arc<AtomicU64>,
    timing_anchor: Option<crate::common::time::SharedTimingAnchor>,
) -> Result<(), OpusError> {
    let mut packet_scratch: Vec<Bytes> = Vec::with_capacity(4);
    let mut pcm_scratch = vec![0.0f32; PCM_PULL_CHUNK];
    // `split().freeze()` yields Arc-backed `Bytes` (zero-copy fan-out) and
    // keeps capacity, so this 1 KiB (packet + envelope) is reused after warm-up.
    let mut encode_buf: bytes::BytesMut = bytes::BytesMut::with_capacity(1024);
    // Pre-increment so the first frame carries seq = 1; restarts per process.
    let mut audio_seq: u64 = 0;
    // Gates `sleep(ACTIVE_TICK)` to only `Wait`, so steady audio pumps at once
    // and idle reaches deeper cpuidle C-states. The `borrow() == 0` checks are
    // snapshots, so the count can drop to 0 mid-iteration -> a brief encode-
    // while-no-subscribers window; acceptable (send's SendError(_) discarded,
    // next re-check pauses). Best-effort: do NOT tighten into a per-step gate
    // without re-checking wake-rate cost.
    let mut last_status = ReadStatus::Wait;

    loop {
        if shutdown.is_cancelled() {
            return Ok(());
        }

        // Paused. `borrow()` does NOT mark-seen, so a subscriber arriving in
        // the pre-await gap still wakes us, and it reads the latest value so a
        // 0->1->0 toggle re-parks correctly.
        if *subscribers.borrow() == 0 {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => return Ok(()),
                changed = subscribers.changed() => {
                    // Err = watch closed, subscribers gone forever.
                    if changed.is_err() {
                        return Ok(());
                    }
                }
            }
            continue;
        }

        let mut engine = OpusEngine::new()?;
        reader.seek_latest(BACKLOG_SAMPLES);
        tracing::info!(
            target: "opus_stream",
            subscribers = *subscribers.borrow(),
            "audio stream resumed; encoder + resampler rebuilt",
        );

        loop {
            if shutdown.is_cancelled() {
                return Ok(());
            }
            // `changed()` only fires on `Wait`, so the steady-Ready path needs
            // this non-blocking sender-drop check (Err iff all dropped = EOL).
            if subscribers.has_changed().is_err() {
                return Ok(());
            }
            if *subscribers.borrow() == 0 {
                tracing::info!(target: "opus_stream", "audio stream paused; dropping encoder");
                break;
            }

            // All arms cancel-safe. The loop-top count re-check is load-
            // bearing: below the select it would miss a 0->1->0 within a tick.
            if matches!(last_status, ReadStatus::Wait) {
                tokio::select! {
                    biased;
                    _ = shutdown.cancelled() => return Ok(()),
                    changed = subscribers.changed() => {
                        if changed.is_err() {
                            return Ok(());
                        }
                        continue;
                    }
                    _ = tokio::time::sleep(ACTIVE_TICK) => {}
                }
            }

            last_status = reader.peek_into(&mut pcm_scratch);
            match last_status {
                ReadStatus::Wait => continue,
                ReadStatus::Lagged { by } => {
                    tracing::warn!(
                        target: "opus_stream",
                        by_samples = by,
                        "audio reader lagged in encoder; resync + reset",
                    );
                    reader.seek_latest(BACKLOG_SAMPLES);
                    engine.reset_after_discontinuity()?;
                    // Synthetic Wait forces the next select pass, bounding a
                    // re-lapping writer's reset+log tight-spin to ACTIVE_TICK.
                    last_status = ReadStatus::Wait;
                    continue;
                }
                ReadStatus::Ready => {}
            }
            // Snapshot tail BEFORE advance: the stamp references the chunk's
            // first sample; a >1-packet pull shares this tail, so later packets
            // drift one frame.
            let pull_start_tail = reader.tail();
            reader.advance(PCM_PULL_CHUNK);

            engine.process_pcm(&pcm_scratch, &mut packet_scratch)?;
            for pkt in packet_scratch.drain(..) {
                // No proto epoch field -> seq is NOT monotonic across process
                // lifetimes (benign: WS clients reconnect with fresh dedup).
                audio_seq = audio_seq.wrapping_add(1);
                let t_us_capture_monotonic = match timing_anchor.as_ref() {
                    Some(cell) => {
                        let anchor = **cell.load();
                        crate::common::time::capture_us_for(anchor, pull_start_tail)
                    }
                    None => crate::common::time::CaptureTime::now().as_micros(),
                };
                let frame = crate::proto::AudioFrame {
                    seq: audio_seq,
                    t_us_capture_monotonic: Some(t_us_capture_monotonic),
                    t_us_publish_unix: crate::common::time::WallTime::now().map(|w| w.as_micros()),
                    // From module constants so a bump flows into the wire shape.
                    sample_rate: Some(OUT_RATE_HZ),
                    frame_duration_ms: Some(FRAME_MS),
                    codec: Some(crate::proto::audio_frame::Codec::Opus(pkt)),
                };
                // Arc-backed Bytes: broadcast `send` clones by Arc bump, no copy.
                let envelope_bytes = crate::proto::framing::wrap_audio_into(&mut encode_buf, frame);
                // SendError(_) (no receivers) is fine.
                let _ = out.send(envelope_bytes);
                packets_encoded.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_samples_is_960() {
        assert_eq!(FRAME_SAMPLES, 960);
    }

    #[test]
    fn opus_engine_constructs() {
        let _e = OpusEngine::new().expect("engine init");
    }

    /// Pins bitrate + complexity CTLs: a dropped `set_*` silently reverts to
    /// libopus auto defaults (~2x wire bandwidth, more CPU).
    #[test]
    fn encoder_pins_bitrate_and_complexity() {
        let mut e = OpusEngine::new().expect("engine init");
        match e.encoder.get_bitrate().expect("get_bitrate") {
            Bitrate::Bits(n) => assert_eq!(n, BITRATE_BPS, "bitrate not pinned"),
            other => panic!("expected Bits({BITRATE_BPS}), got {other:?}"),
        }
        assert_eq!(
            e.encoder.get_complexity().expect("get_complexity"),
            COMPLEXITY,
            "complexity not pinned",
        );
    }

    /// 1 kHz sine round-trip checked spectrally (RMS within 5%, >=95% energy
    /// in the 1 kHz bin); sample-domain RMSE is too phase-brittle for a pure
    /// tone through a sinc resampler.
    #[test]
    fn round_trip_1khz_tone_spectral_energy_at_1khz() {
        use opus::{Channels, Decoder};

        const TONE_HZ: f32 = 1000.0;
        const TONE_AMPL: f32 = 0.5;

        let n_in = IN_RATE_HZ as usize;
        let pcm: Vec<f32> = (0..n_in)
            .map(|i| {
                let t = i as f32 / IN_RATE_HZ as f32;
                TONE_AMPL * (2.0 * std::f32::consts::PI * TONE_HZ * t).sin()
            })
            .collect();

        let mut engine = OpusEngine::new().expect("engine");
        let mut packets: Vec<Bytes> = Vec::new();
        for chunk in pcm.chunks(PCM_PULL_CHUNK) {
            engine.process_pcm(chunk, &mut packets).expect("encode");
        }
        assert!(packets.len() > 30, "too few packets: {}", packets.len());

        let mut decoder = Decoder::new(OUT_RATE_HZ, Channels::Mono).expect("decoder");
        let mut decoded = Vec::with_capacity(packets.len() * FRAME_SAMPLES);
        let mut frame_buf = vec![0f32; FRAME_SAMPLES];
        for pkt in &packets {
            let n = decoder
                .decode_float(pkt.as_ref(), &mut frame_buf, false)
                .expect("decode");
            // > FRAME_SAMPLES means `frame_buf` was undersized.
            assert!(
                n <= FRAME_SAMPLES,
                "decoded {n} > FRAME_SAMPLES ({FRAME_SAMPLES})"
            );
            decoded.extend_from_slice(&frame_buf[..n]);
        }
        assert!(
            decoded.len() >= FRAME_SAMPLES * 30,
            "decoded too short: {}",
            decoded.len(),
        );

        // Skip 100 ms head (encoder + sinc lookahead) and a short tail.
        let skip_head = (OUT_RATE_HZ as usize) * 100 / 1000;
        let skip_tail = (OUT_RATE_HZ as usize) * 20 / 1000;
        let body = &decoded[skip_head..decoded.len() - skip_tail];
        assert!(
            body.len() > 4 * FRAME_SAMPLES,
            "comparison window too small ({})",
            body.len(),
        );

        // Goertzel-style 1 kHz energy via sin/cos correlation.
        let omega = 2.0_f64 * std::f64::consts::PI * (TONE_HZ as f64) / (OUT_RATE_HZ as f64);
        let mut acc_cos = 0.0f64;
        let mut acc_sin = 0.0f64;
        let mut total_sq = 0.0f64;
        for (i, &v) in body.iter().enumerate() {
            let phase = omega * i as f64;
            acc_cos += v as f64 * phase.cos();
            acc_sin += v as f64 * phase.sin();
            total_sq += (v as f64).powi(2);
        }
        let n = body.len() as f64;
        // Pure A*sin(omega t): sin-corr ~= N*A/2 -> power = A^2/4; x2 expresses
        // it as mean v^2 for that component.
        let target_power = 2.0 * (acc_cos.powi(2) + acc_sin.powi(2)) / (n * n);
        let total_power = total_sq / n;
        let body_rms = total_power.sqrt() as f32;
        let in_band_frac = target_power / total_power;

        let ref_rms = TONE_AMPL / 2.0_f32.sqrt();
        let amp_drift = ((body_rms - ref_rms) / ref_rms).abs();

        eprintln!(
            "round_trip_1khz: body_rms={body_rms:.4} ref_rms={ref_rms:.4} \
             amp_drift={:.2}% in_band@1kHz={:.2}% ({:.0} samples)",
            amp_drift * 100.0,
            in_band_frac * 100.0,
            n,
        );

        assert!(
            amp_drift < 0.05,
            "amplitude drift {:.2}% > 5% (body_rms={body_rms} vs ref_rms={ref_rms})",
            amp_drift * 100.0,
        );
        assert!(
            in_band_frac > 0.95,
            "only {:.1}% of body energy at 1 kHz; expected >= 95%",
            in_band_frac * 100.0,
        );
    }

    /// Soak: 5 s of white noise, every packet decodes cleanly.
    #[test]
    fn five_second_white_noise_all_packets_decode() {
        use opus::{Channels, Decoder};

        let n_in = (IN_RATE_HZ as usize) * 5;
        let mut s: u32 = 0xdeadbeef; // deterministic LCG
        let pcm: Vec<f32> = (0..n_in)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / 0xFFFFFF as f32) * 0.5 - 0.25
            })
            .collect();

        let mut engine = OpusEngine::new().expect("engine");
        let mut packets: Vec<Bytes> = Vec::new();
        for chunk in pcm.chunks(PCM_PULL_CHUNK) {
            engine.process_pcm(chunk, &mut packets).expect("encode");
        }

        let mut decoder = Decoder::new(OUT_RATE_HZ, Channels::Mono).expect("decoder");
        let mut frame_buf = vec![0f32; FRAME_SAMPLES];
        let mut total_decoded = 0usize;
        for (i, pkt) in packets.iter().enumerate() {
            let n = decoder
                .decode_float(pkt.as_ref(), &mut frame_buf, false)
                .unwrap_or_else(|e| panic!("packet {i} decode failed: {e}"));
            assert_eq!(
                n, FRAME_SAMPLES,
                "packet {i} decoded {n} samples (expected 960)"
            );
            total_decoded += n;
        }
        assert!(
            packets.len() >= 240,
            "expected >=240 packets in 5 s, got {}",
            packets.len()
        );
        assert!(total_decoded >= 240 * FRAME_SAMPLES);
    }

    /// Every packet (silence, sine, noise, square) must fit
    /// `MAX_PACKET_BYTES`; the WS frame-cap pins that bound, so a codec swap
    /// violating it fails here pre-ship.
    #[test]
    fn every_packet_fits_max_packet_bytes_across_input_shapes() {
        // 0.5 s per case (~25 packets, amortizes VBR steady state).
        let n_in = (IN_RATE_HZ as usize) / 2;

        let silence = vec![0.0f32; n_in];

        let sine: Vec<f32> = (0..n_in)
            .map(|i| {
                let t = i as f32 / IN_RATE_HZ as f32;
                0.5 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
            })
            .collect();

        let mut s: u32 = 0x1234_5678; // deterministic LCG
        let noise: Vec<f32> = (0..n_in)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / 0xFF_FFFF as f32) * 1.8 - 0.9
            })
            .collect();

        // Saturated 200 Hz square: VBR worst case (spike every edge).
        let period = (IN_RATE_HZ as usize) / 200;
        let square: Vec<f32> = (0..n_in)
            .map(|i| {
                if (i / (period / 2)).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        for (label, pcm) in [
            ("silence", silence),
            ("sine_1k", sine),
            ("white_noise", noise),
            ("square_200", square),
        ] {
            let mut engine = OpusEngine::new().expect("engine");
            let mut packets: Vec<Bytes> = Vec::new();
            for chunk in pcm.chunks(PCM_PULL_CHUNK) {
                engine.process_pcm(chunk, &mut packets).expect("encode");
            }
            assert!(
                !packets.is_empty(),
                "{label}: no packets emitted for {} input samples",
                pcm.len()
            );
            for (i, pkt) in packets.iter().enumerate() {
                assert!(
                    pkt.len() <= MAX_PACKET_BYTES,
                    "{label}: packet {i} = {} bytes > MAX_PACKET_BYTES ({})",
                    pkt.len(),
                    MAX_PACKET_BYTES,
                );
                // Guards a future path bypassing EmptyEncode.
                assert!(
                    !pkt.is_empty(),
                    "{label}: packet {i} is empty (encoder corrupt?)",
                );
            }
        }
    }

    /// NaN PCM rejected with `BadPcm { index }` before resample/encode, and
    /// the engine stays usable for valid input.
    #[test]
    fn process_pcm_rejects_nan_with_index_and_does_not_emit() {
        let mut engine = OpusEngine::new().expect("engine");
        let mut pcm = vec![0.1f32; PCM_PULL_CHUNK];
        pcm[7] = f32::NAN; // non-zero index proves the reported index is faithful
        let mut packets: Vec<Bytes> = Vec::new();
        let err = engine
            .process_pcm(&pcm, &mut packets)
            .expect_err("non-finite PCM must surface as Err");
        match err {
            OpusError::BadPcm { index } => assert_eq!(index, 7, "BadPcm index mismatch"),
            other => panic!("expected BadPcm {{ index: 7 }}, got {other:?}"),
        }
        assert!(
            packets.is_empty(),
            "no packets must be emitted when the input slice is rejected (got {})",
            packets.len(),
        );

        // Rejected slice never reached the resampler, so a clean slice still
        // encodes; 2x chunk clears one full frame past FIR warmup.
        let clean = vec![0.1f32; PCM_PULL_CHUNK * 2];
        let n = engine
            .process_pcm(&clean, &mut packets)
            .expect("encode after rejected slice");
        assert!(n > 0, "engine remained usable: expected >0 packets, got 0");
    }

    /// Both +Inf and -Inf rejected (not just NaN), reporting the FIRST
    /// non-finite index.
    #[test]
    fn process_pcm_rejects_pos_and_neg_inf() {
        for (label, bad) in [("+inf", f32::INFINITY), ("-inf", f32::NEG_INFINITY)] {
            let mut engine = OpusEngine::new().expect("engine");
            let mut pcm = vec![0.0f32; PCM_PULL_CHUNK];
            pcm[42] = bad;
            pcm[100] = f32::NAN; // second bad value confirms the FIRST index wins
            let mut packets: Vec<Bytes> = Vec::new();
            let err = engine
                .process_pcm(&pcm, &mut packets)
                .expect_err("non-finite PCM must surface as Err");
            match err {
                OpusError::BadPcm { index } => assert_eq!(
                    index, 42,
                    "{label}: expected first non-finite index 42, got {index}"
                ),
                other => panic!("{label}: expected BadPcm, got {other:?}"),
            }
            assert!(packets.is_empty(), "{label}: must not emit packets");
        }
    }

    /// `reset_after_discontinuity` clears state; encoding continues after.
    #[test]
    fn reset_then_continue_works() {
        let mut engine = OpusEngine::new().expect("engine");
        let pcm = vec![0.1f32; PCM_PULL_CHUNK * 2];
        let mut packets: Vec<Bytes> = Vec::new();
        engine.process_pcm(&pcm, &mut packets).expect("encode");
        assert!(!packets.is_empty());

        engine.reset_after_discontinuity().expect("reset");
        packets.clear();
        engine
            .process_pcm(&pcm, &mut packets)
            .expect("encode after reset");
        assert!(!packets.is_empty(), "no packets emitted after reset");
    }
}
