//! Integration smoke test for the async `run` loop: subscriber-driven pause/resume, broadcast wiring, lag handling.

use acoustics_lab::audio_buffer::AudioBuffer;
use acoustics_lab::opus_stream::{IN_RATE_HZ, run};
use bytes::Bytes;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;

fn fill_writer(writer: &mut acoustics_lab::audio_buffer::Writer, seconds: f32) {
    let n = (IN_RATE_HZ as f32 * seconds) as usize;
    let pcm: Vec<f32> = (0..n)
        .map(|i| {
            0.5 * (2.0_f32 * std::f32::consts::PI * 1000.0 * (i as f32 / IN_RATE_HZ as f32)).sin()
        })
        .collect();
    for chunk in pcm.chunks(1024) {
        writer.push(chunk);
    }
}

/// Active stream: 2 s pushed, encoded, collected, then cancelled.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn active_stream_emits_packets() {
    // 262_144 = 2^18 (~5.94 s) = next pow2 above 5 x IN_RATE_HZ.
    let buf = AudioBuffer::new(262_144);
    let mut writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let (sub_tx, sub_rx) = watch::channel(1usize); // 1 subscriber = active from the start
    let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(256);
    let token = CancellationToken::new();
    let token_run = token.clone();
    let packets_encoded = Arc::new(AtomicU64::new(0));
    let packets = packets_encoded.clone();

    fill_writer(&mut writer, 2.0);

    let run_handle =
        tokio::spawn(async move { run(reader, sub_rx, out_tx, token_run, packets, None).await });

    // Refill on each Empty so the encoder's reader keeps getting new data (no Wait stall).
    let mut packets = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline && packets.len() < 30 {
        match out_rx.try_recv() {
            Ok(b) => packets.push(b),
            Err(broadcast::error::TryRecvError::Empty) => {
                tokio::time::sleep(Duration::from_millis(20)).await;
                fill_writer(&mut writer, 0.020);
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(broadcast::error::TryRecvError::Closed) => break,
        }
    }

    token.cancel();
    let _ = sub_tx.send(0);
    let _ = run_handle.await.expect("run task panicked");
    assert!(
        packets.len() >= 30,
        "expected >=30 packets, got {}",
        packets.len()
    );

    for (i, p) in packets.iter().enumerate() {
        assert!(!p.is_empty(), "packet {i} empty");
        assert!(p.len() <= 4000, "packet {i} too big: {} B", p.len());
    }

    // Counter bumps per `out.send`; receiver may skip via Lagged, so collected len is a lower bound, not equal.
    let counted = packets_encoded.load(Ordering::Relaxed);
    assert!(
        counted >= packets.len() as u64,
        "packets_encoded={counted} < collected={}; counter must include packets the receiver missed",
        packets.len(),
    );
    assert!(
        counted >= 30,
        "packets_encoded={counted} < 30; encoder did not bump counter for emitted packets",
    );
}

/// Paused stream emits no packets; resume kicks them in.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pause_resume_state_machine() {
    // 262_144 = 2^18 (~5.94 s) = next pow2 above 5 x IN_RATE_HZ.
    let buf = AudioBuffer::new(262_144);
    let mut writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let (sub_tx, sub_rx) = watch::channel(0usize); // 0 subscribers = paused
    let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(256);
    let token = CancellationToken::new();
    let token_run = token.clone();
    let packets = Arc::new(AtomicU64::new(0));

    fill_writer(&mut writer, 1.0);

    let run_handle =
        tokio::spawn(async move { run(reader, sub_rx, out_tx, token_run, packets, None).await });

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        matches!(
            out_rx.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ),
        "paused state emitted packets",
    );

    // Resume: 1 subscriber rebuilds the engine and starts encoding.
    sub_tx.send(1).expect("send subscribers=1");

    let fill_done = tokio::time::Instant::now() + Duration::from_millis(800);
    let mut got = 0usize;
    while tokio::time::Instant::now() < fill_done {
        fill_writer(&mut writer, 0.020);
        tokio::time::sleep(Duration::from_millis(20)).await;
        while let Ok(_b) = out_rx.try_recv() {
            got += 1;
        }
    }
    assert!(got > 5, "got only {got} packets after resume");

    // Pause again: after draining, queue length stays flat (no new emissions).
    sub_tx.send(0).expect("send subscribers=0");
    tokio::time::sleep(Duration::from_millis(100)).await;
    while out_rx.try_recv().is_ok() {}
    let pre = out_rx.len();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let post = out_rx.len();
    assert_eq!(pre, post, "packets emitted after pause: {pre} -> {post}");

    token.cancel();
    let _ = sub_tx.send(0);
    let _ = run_handle.await.expect("run task panicked");
}

/// Guards end-to-end capture-timestamp projection: each emitted
/// `AudioFrame.t_us_capture_monotonic` equals its first 44.1 kHz sample
/// projected through the producer-side anchor, within +/-1 ms (not sub-us:
/// inverting the projection here is a us<->sample integer round-trip, and
/// back-mapping a packet to its first input sample is itself approximate --
/// the resampler carries partial input plus a sinc-FIR group delay -- so 1 ms
/// is the generous tolerance that covers the rounding).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn timing_anchor_drives_capture_us_within_one_ms() {
    use acoustics_lab::common::time::{
        BufferTimingAnchor, CaptureTime, capture_us_for, shared_timing_anchor,
    };
    use acoustics_lab::proto::envelope::Payload;
    use acoustics_lab::proto::framing::decode_envelope;

    let buf = AudioBuffer::new(262_144);
    let mut writer = buf.take_writer();
    let reader = buf.reader_at(0);

    // Stage anchor before audio flows so sample N maps to capture_us = 1e9 + N*1e6/44_100
    // (head_pos=0, captured_at=1e9). `shared_timing_anchor()` returns a FRESH cell per
    // call despite the name, so this mutation is test-local (no cross-test leakage).
    let anchor_cell = shared_timing_anchor();
    anchor_cell.store(Arc::new(BufferTimingAnchor {
        head_pos: 0,
        captured_at: CaptureTime::from_micros(1_000_000_000),
        sample_rate_hz: 44_100,
    }));

    let (_sub_tx, sub_rx) = watch::channel(1usize);
    let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(256);
    let token = CancellationToken::new();
    let token_run = token.clone();
    let packets_encoded = Arc::new(AtomicU64::new(0));
    let packets = packets_encoded.clone();
    let anchor_for_run = anchor_cell.clone();

    fill_writer(&mut writer, 2.0);

    let run_handle = tokio::spawn(async move {
        run(
            reader,
            sub_rx,
            out_tx,
            token_run,
            packets,
            Some(anchor_for_run),
        )
        .await
    });

    // One decoded packet validates the projection; batch a few against transient lag.
    let mut packet_bytes: Vec<Bytes> = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline && packet_bytes.len() < 5 {
        match out_rx.try_recv() {
            Ok(b) => packet_bytes.push(b),
            Err(broadcast::error::TryRecvError::Empty) => {
                tokio::time::sleep(Duration::from_millis(20)).await;
                fill_writer(&mut writer, 0.020);
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            Err(broadcast::error::TryRecvError::Closed) => break,
        }
    }
    token.cancel();
    let _ = run_handle.await.expect("run task panicked");

    assert!(
        !packet_bytes.is_empty(),
        "encoder produced no packets within deadline",
    );

    // Projected stamp of the first sample (N >= 0) lies in [1e9, 1e9 + 2e6] (N = 2 s at 44.1 kHz).
    let env = decode_envelope(&packet_bytes[0]).expect("decode envelope");
    let payload = env.payload.expect("envelope payload present");
    let frame = match payload {
        Payload::Audio(f) => f,
        other => panic!("expected Payload::Audio, got {other:?}"),
    };
    // `run` bumps audio_seq (init 0) BEFORE building the frame, so the first packet is seq=1;
    // gates a refactor moving the bump after construction.
    assert_eq!(
        frame.seq, 1,
        "first emitted AudioFrame must carry seq=1 (audio_seq increments BEFORE frame construction)",
    );
    let stamp = frame
        .t_us_capture_monotonic
        .expect("anchor-plumbed encoder must populate t_us_capture_monotonic");

    assert!(
        stamp >= 1_000_000_000,
        "stamp {stamp} below anchor captured_at; \
         anchor projection went backwards from sample 0",
    );
    assert!(
        stamp <= 1_000_000_000 + 2_000_000 + 1_000,
        "stamp {stamp} above expected upper bound; \
         encoder produced a packet covering audio outside the 2 s fill window",
    );

    // Invert the projection to recover sample N (no need for the encoder's private
    // BACKLOG_SAMPLES/PCM_PULL_CHUNK), then assert N in [0, head_now] (no un-pushed
    // reads) and round-trip residual <= 1 ms (dominated by us<->sample rounding).
    let anchor = **anchor_cell.load();
    let head_now = writer.head_pos();
    let stamp_delta_us = stamp.saturating_sub(anchor.captured_at.as_micros());
    let n_recovered = (stamp_delta_us as u128 * anchor.sample_rate_hz as u128 / 1_000_000) as u64
        + anchor.head_pos;
    assert!(
        n_recovered <= head_now,
        "recovered sample position {n_recovered} > head_now {head_now}; \
         encoder stamped a future sample (anchor projection inverted)",
    );
    let projected = capture_us_for(anchor, n_recovered);
    let residual = projected.abs_diff(stamp);
    assert!(
        residual <= 1_000,
        "anchor round-trip residual {residual} us > 1 ms tolerance; \
         stamp {stamp}, recovered N {n_recovered}, projected {projected}",
    );

    // Stamp must be in the anchor's image: captured_at is 1e9 us (~17 min from boot),
    // whereas the publish-time fallback stamps `CaptureTime::now()` < 1e9 on a fresh process.
    assert!(
        (999_900_000..=1_000_900_000 + 2_000_000).contains(&stamp),
        "stamp {stamp} not in the anchor's projection image; \
         the publish-time fallback path leaked through",
    );
}

/// Cancellation tears down the loop quickly even from paused state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_terminates_paused_loop() {
    // 131_072 = 2^17 (~2.97 s) = next pow2 above 2 x IN_RATE_HZ.
    let buf = AudioBuffer::new(131_072);
    let _writer = buf.take_writer();
    let reader = buf.reader_at(0);

    let (_sub_tx, sub_rx) = watch::channel(0usize);
    let (out_tx, _out_rx) = broadcast::channel::<Bytes>(16);
    let token = CancellationToken::new();
    let token_run = token.clone();
    let packets = Arc::new(AtomicU64::new(0));

    let run_handle =
        tokio::spawn(async move { run(reader, sub_rx, out_tx, token_run, packets, None).await });

    token.cancel();
    let res = tokio::time::timeout(Duration::from_millis(500), run_handle).await;
    let inner = res.expect("run task did not exit within 500 ms");
    let _ = inner.expect("run task panicked");
}
