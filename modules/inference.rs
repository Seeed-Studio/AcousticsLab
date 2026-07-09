//! Streaming hybrid inference engine: audio_buffer -> Preproc (CPU) ->
//! backbone (NPU/CPU) -> head_forward+softmax -> top-k -> broadcast to subscribers.
//!
//! Single `spawn_blocking` worker; cross-thread state is the `HotHead` ArcSwap
//! (atomic), the optional `SharedTimingAnchor` ArcSwap, the `Heartbeat` watch
//! channel, and the broadcast output sender.

#![warn(missing_debug_implementations)]

// Re-exported below, so pub(crate) keeps the surface off the public API.
pub(crate) mod backbone;
pub(crate) mod engine;
pub(crate) mod head;
pub(crate) mod kernel;

#[cfg(test)]
mod npy;

#[cfg(all(target_os = "linux", feature = "rknpu"))]
pub use backbone::RknnBackbone;
pub use backbone::{
    Backbone, BackboneCatalogue, BackboneError, BackboneKind, BackbonePipeline, BackboneRef,
    BurnBackbone,
};
pub use engine::{
    EngineError, EngineState, Heartbeat, InferenceCfg, InferenceEngine, SAMPLE_RATE_HZ,
    WAVEFORM_DURATION_NS,
};
pub use head::{HeadError, HeadInner, HotHead, MAX_N_CLASSES};
pub use kernel::{
    head_forward, softmax_into, top_k_indices_into, transpose_frame_major_to_bin_major,
};

#[cfg(test)]
mod parity_tests {
    #![allow(clippy::disallowed_methods)] // tests load fixtures from misc/ directly, bypassing fs_atomic
    //! Parity tests for the CPU pipeline against reference assets under `misc/`.
    //! All `#[ignore]`'d (path-fragile + slow).

    use super::*;
    use crate::common::dims::BackboneFeatureDim;
    use crate::inference::npy;
    use std::path::PathBuf;

    fn crate_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    /// Pins our softmax against upstream tf-style `probs_*.npy`.
    #[test]
    #[ignore = "depends on repo-root reference assets; --include-ignored"]
    fn softmax_parity_against_reference_probs() {
        let root = crate_root();
        for i in 0..5 {
            let logits_path = root.join(format!("misc/npys/logits_{i}.npy"));
            let probs_path = root.join(format!("misc/npys/probs_{i}.npy"));
            let (_, logits) = npy::read_f32(&logits_path);
            let (_, ref_probs) = npy::read_f32(&probs_path);
            assert_eq!(logits.len(), ref_probs.len(), "shape mismatch in {i}");
            let mut probs = vec![0.0f32; logits.len()];
            softmax_into(&logits, &mut probs);
            for (j, (a, b)) in probs.iter().zip(ref_probs.iter()).enumerate() {
                let drift = (a - b).abs();
                assert!(
                    drift < 1e-6,
                    "softmax drift at sample {i} class {j}: ours={a}, ref={b}, |D|={drift}",
                );
            }
        }
    }

    /// Full Burn pipeline (Preproc -> BurnBackbone -> head_forward) vs reference
    /// logits. Fixtures were captured against this same pipeline, so this guards
    /// bit-stability, not cross-implementation parity.
    #[test]
    #[ignore = "depends on bundled fixtures; --include-ignored"]
    fn burn_backbone_parity_against_reference_logits() {
        let root = crate_root();
        let backbone_path = root.join("misc/backbones/backbone.mpk");
        let head_path = root.join("misc/heads/default/head.mpk");
        let labels_path = root.join("misc/heads/default/labels.txt");
        for p in [&backbone_path, &head_path, &labels_path] {
            assert!(p.exists(), "missing test asset: {}", p.display());
        }

        // Load once, reuse across samples: backbone load is ~200ms-1s.
        let mut backbone = BurnBackbone::load(&backbone_path).expect("load Burn backbone");
        let head = HotHead::load(&head_path, &labels_path, crate::common::ids::HeadId::new())
            .expect("load head");
        let snap = head.snapshot();
        let mut preproc_inst = crate::preproc::Preproc::new();

        for sample_idx in 0..5 {
            let waveform_path = root.join(format!("misc/npys/waveform_{sample_idx}.npy"));
            let logits_ref_path = root.join(format!("misc/npys/logits_{sample_idx}.npy"));
            let (_, pcm_vec) = npy::read_f32(&waveform_path);
            assert_eq!(pcm_vec.len(), crate::common::dims::WaveformLen::USIZE);
            let pcm: &[f32; crate::common::dims::WaveformLen::USIZE] =
                pcm_vec.as_slice().try_into().expect("pcm length match");

            let spec = preproc_inst.spectrogram(pcm);

            let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
            backbone
                .infer(&spec, &mut features)
                .expect("backbone infer");

            let mut logits = vec![0.0f32; snap.n_classes];
            head_forward(&features[..], &snap.weight, &snap.bias, &mut logits);

            let (_, ref_logits) = npy::read_f32(&logits_ref_path);
            assert_eq!(
                ref_logits.len(),
                logits.len(),
                "sample {sample_idx}: logit count mismatch",
            );
            // Burn fp32 is deterministic; only drift is cross-version rounding (~1e-5).
            for (i, (&a, &b)) in logits.iter().zip(ref_logits.iter()).enumerate() {
                let drift = (a - b).abs();
                assert!(
                    drift < 1e-3,
                    "sample {sample_idx} logit {i}: ours={a}, ref={b}, |D|={drift}",
                );
            }
        }
    }

    /// On-device train/serve feature-basis parity: RKNN fp16 features vs the
    /// canonical Burn fp32 features on the bundled reference waveforms.
    /// Training extracts through this same `RknnBackbone`, so this pins the
    /// basis drift the linear head must absorb; the 1% normalized-MAE gate is
    /// ~10x expected fp16 drift -- failure means the `.rknn` is no longer a
    /// faithful conversion of the `.mpk`.
    #[cfg(all(target_os = "linux", feature = "rknpu"))]
    #[test]
    #[ignore = "on-device only; requires NPU + librknnrt + reference assets"]
    fn rknn_burn_feature_mae_within_tolerance() {
        use crate::common::dims::BackboneFeatureDim;

        let root = crate_root();
        let rknn_path = root.join("misc/backbones/backbone.rknn");
        let mpk_path = root.join("misc/backbones/backbone.mpk");
        for p in [&rknn_path, &mpk_path] {
            assert!(p.exists(), "missing test asset: {}", p.display());
        }

        let mut rknn = RknnBackbone::load(&rknn_path).expect("load rknn backbone");
        let mut burn = BurnBackbone::load(&mpk_path).expect("load burn backbone");
        let mut preproc_inst = crate::preproc::Preproc::new();

        for sample_idx in 0..5 {
            let waveform_path = root.join(format!("misc/npys/waveform_{sample_idx}.npy"));
            let (_, pcm_vec) = npy::read_f32(&waveform_path);
            assert_eq!(pcm_vec.len(), crate::common::dims::WaveformLen::USIZE);
            let pcm: &[f32; crate::common::dims::WaveformLen::USIZE] =
                pcm_vec.as_slice().try_into().expect("pcm length match");

            let spec = preproc_inst.spectrogram(pcm);
            let mut feats_rknn = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
            let mut feats_burn = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
            rknn.infer(&spec, &mut feats_rknn).expect("rknn infer");
            burn.infer(&spec, &mut feats_burn).expect("burn infer");

            let mut abs_err_sum = 0.0f64;
            let mut max_abs_err = 0.0f32;
            let mut ref_abs_sum = 0.0f64;
            for (&a, &b) in feats_rknn.iter().zip(feats_burn.iter()) {
                assert!(
                    a.is_finite(),
                    "sample {sample_idx}: non-finite rknn feature"
                );
                assert!(
                    b.is_finite(),
                    "sample {sample_idx}: non-finite burn feature"
                );
                let d = (a - b).abs();
                abs_err_sum += f64::from(d);
                max_abs_err = max_abs_err.max(d);
                ref_abs_sum += f64::from(b.abs());
            }
            let n = BackboneFeatureDim::USIZE as f64;
            let mae = abs_err_sum / n;
            let ref_mean_abs = ref_abs_sum / n;
            assert!(
                ref_mean_abs > 0.0,
                "sample {sample_idx}: burn features are all-zero; fixture unusable",
            );
            let normalized_mae = mae / ref_mean_abs;
            eprintln!(
                "rknn_burn_feature_mae sample {sample_idx}: mae={mae:.6} \
                 normalized={normalized_mae:.6} max_abs={max_abs_err:.6} \
                 ref_mean_abs={ref_mean_abs:.6}",
            );
            assert!(
                normalized_mae < 0.01,
                "sample {sample_idx}: normalized feature MAE {normalized_mae:.6} \
                 exceeds 1% of mean |burn| ({ref_mean_abs:.6}); the .rknn no \
                 longer matches the .mpk basis -- re-convert and re-verify",
            );
        }
    }

    /// top_k must equal argmax-by-descending of the recorded probs.
    #[test]
    #[ignore = "depends on repo-root reference assets; --include-ignored"]
    fn top_k_parity_against_argsort_of_reference() {
        let root = crate_root();
        let (_, ref_probs) = npy::read_f32(&root.join("misc/npys/probs_0.npy"));
        let mut top = Vec::with_capacity(8);
        top_k_indices_into(&ref_probs, 3, &mut top);
        assert_eq!(top.len(), 3.min(ref_probs.len()));
        for w in top.windows(2) {
            let (a, b) = (w[0], w[1]);
            assert!(
                ref_probs[a] >= ref_probs[b],
                "top_k not descending: idx {a} (p={}) before idx {b} (p={})",
                ref_probs[a],
                ref_probs[b],
            );
        }
    }
}

#[cfg(test)]
mod stream_e2e {
    #![allow(clippy::disallowed_methods)] // tests load fixtures from misc/ directly, bypassing fs_atomic
    //! End-to-end gate: drive the engine from a stitched waveform_0.npy and verify
    //! per-frame top-1 matches the reference. RKNN variant cfg-gated to linux+rknpu
    //! (NPU-only), Burn variant runs on host; both `#[ignore]`'d.

    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use crate::audio_buffer::AudioBuffer;
    use crate::common::dims::WaveformLen;
    use arc_swap::ArcSwap;
    use bytes::Bytes;
    use prost::Message;
    use tokio::sync::{broadcast, watch};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::inference::npy;
    use crate::proto::envelope::Payload as EnvelopePayload;
    use crate::proto::{Envelope, InferenceFrame};

    /// Decode the engine's envelope bytes into the inner `InferenceFrame`.
    /// Must unwrap the envelope: decoding raw bytes as `InferenceFrame` silently
    /// yields defaults because prost ignores unknown fields.
    fn decode_inference_envelope(bytes: &[u8]) -> InferenceFrame {
        let env = Envelope::decode(bytes).expect("decode envelope");
        match env.payload.expect("envelope.payload") {
            EnvelopePayload::Inference(f) => f,
            other => panic!("unexpected envelope payload variant: {other:?}"),
        }
    }

    fn crate_root() -> PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
    }

    #[cfg(all(target_os = "linux", feature = "rknpu"))]
    #[test]
    #[ignore = "on-device only; requires NPU + librknnrt + reference assets"]
    fn stream_parity_e2e() {
        let root = crate_root();
        let backbone = root.join("misc/backbones/backbone.rknn");
        let head_mpk = root.join("misc/heads/default/head.mpk");
        let labels_path = root.join("misc/heads/default/labels.txt");
        let waveform_path = root.join("misc/npys/waveform_0.npy");
        for p in [&backbone, &head_mpk, &labels_path, &waveform_path] {
            assert!(p.exists(), "missing test asset: {}", p.display());
        }

        let buf = AudioBuffer::new(262_144);
        let mut writer = buf.take_writer();
        let reader = buf.reader_at(0);

        let head = HotHead::load(&head_mpk, &labels_path, crate::common::ids::HeadId::new())
            .expect("load head");
        let cfg = Arc::new(ArcSwap::from_pointee(InferenceCfg {
            hop_samples: 11_025,
            top_k: 3,
        }));
        let (mon_tx, _mon_rx) = watch::channel(Heartbeat::default());
        let pipeline = BackbonePipeline::Rknn(Box::new(
            RknnBackbone::load(&backbone).expect("load rknn backbone"),
        ));
        let engine = InferenceEngine::new(pipeline.into_boxed(), head, cfg, mon_tx, None);

        let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(64);
        let token = CancellationToken::new();
        let token_engine = token.clone();

        let engine_handle =
            std::thread::spawn(move || engine.run_blocking(reader, out_tx, token_engine));

        // Stitch waveform_0 10x at ~real-time cadence (1024 samples ~= 23 ms).
        let (_, pcm) = npy::read_f32(&waveform_path);
        assert_eq!(
            pcm.len(),
            WaveformLen::USIZE,
            "reference waveform must be 44032 samples"
        );
        let writer_handle = std::thread::spawn(move || {
            for _ in 0..10 {
                for chunk in pcm.chunks(1024) {
                    writer.push(chunk);
                    std::thread::sleep(Duration::from_millis(23));
                }
            }
        });

        let mut frames: Vec<InferenceFrame> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(12);
        while Instant::now() < deadline && frames.len() < 8 {
            match out_rx.try_recv() {
                Ok(bytes) => {
                    let f = decode_inference_envelope(bytes.as_ref());
                    frames.push(f);
                }
                Err(broadcast::error::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(broadcast::error::TryRecvError::Closed) => break,
                Err(broadcast::error::TryRecvError::Lagged(_)) => {
                    continue;
                }
            }
        }

        token.cancel();
        writer_handle.join().expect("writer thread panicked");
        engine_handle
            .join()
            .expect("engine thread panicked")
            .expect("engine returned an error");

        assert!(!frames.is_empty(), "no inference frames produced");
        for w in frames.windows(2) {
            assert!(
                w[1].seq > w[0].seq,
                "seq not monotonic: {} -> {}",
                w[0].seq,
                w[1].seq
            );
        }
        for f in &frames {
            assert!(!f.top_k.is_empty(), "top_k empty in frame {}", f.seq);
            assert!(
                f.t_us_capture_monotonic.is_some(),
                "capture timestamp absent in frame {}",
                f.seq,
            );
            assert!(
                f.t_us_publish_unix.is_some(),
                "publish timestamp absent in frame {}",
                f.seq
            );
            // head_id is proto-optional but always populated in production frames.
            assert!(
                f.head_id.as_deref().is_some_and(|s| !s.is_empty()),
                "head_id absent or empty in frame {}",
                f.seq,
            );
            for tk in &f.top_k {
                assert!(
                    (0.0..=1.0).contains(&tk.prob),
                    "prob out of range in frame {}: {}",
                    f.seq,
                    tk.prob
                );
                assert!(!tk.label.is_empty(), "label empty in frame {}", f.seq);
            }
        }

        // waveform_0 has a deterministic top-1; frames past the warmup transient
        // (rubato/realfft converging from 0-init) must agree on it.
        let steady: Vec<_> = frames.iter().skip(1).collect();
        if !steady.is_empty() {
            let first_top1 = steady[0].top_k[0].class_idx;
            for f in &steady {
                assert_eq!(
                    f.top_k[0].class_idx, first_top1,
                    "top-1 drift after warm-up: frame {} -> idx {}, expected {}",
                    f.seq, f.top_k[0].class_idx, first_top1,
                );
            }
            eprintln!(
                "stream_parity_e2e: {} frames; steady-state top-1 idx={} label={:?} p={:.3}",
                frames.len(),
                steady[0].top_k[0].class_idx,
                steady[0].top_k[0].label,
                steady[0].top_k[0].prob,
            );
        }
    }

    /// Burn (CPU) backbone variant of `stream_parity_e2e`; runs on host.
    #[test]
    #[ignore = "depends on repo-root reference assets; --include-ignored"]
    fn stream_parity_e2e_burn() {
        let root = crate_root();
        let backbone_path = root.join("misc/backbones/backbone.mpk");
        let head_mpk = root.join("misc/heads/default/head.mpk");
        let labels_path = root.join("misc/heads/default/labels.txt");
        let waveform_path = root.join("misc/npys/waveform_0.npy");
        for p in [&backbone_path, &head_mpk, &labels_path, &waveform_path] {
            assert!(p.exists(), "missing test asset: {}", p.display());
        }

        let buf = AudioBuffer::new(262_144);
        let mut writer = buf.take_writer();
        let reader = buf.reader_at(0);

        let head = HotHead::load(&head_mpk, &labels_path, crate::common::ids::HeadId::new())
            .expect("load head");
        let cfg = Arc::new(ArcSwap::from_pointee(InferenceCfg {
            hop_samples: 11_025,
            top_k: 3,
        }));
        let (mon_tx, _mon_rx) = watch::channel(Heartbeat::default());
        let pipeline = BackbonePipeline::Burn(Box::new(
            BurnBackbone::load(&backbone_path).expect("load burn backbone"),
        ));
        let engine = InferenceEngine::new(pipeline.into_boxed(), head, cfg, mon_tx, None);

        let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(64);
        let token = CancellationToken::new();
        let token_engine = token.clone();
        let engine_handle =
            std::thread::spawn(move || engine.run_blocking(reader, out_tx, token_engine));

        let (_, pcm) = npy::read_f32(&waveform_path);
        assert_eq!(pcm.len(), WaveformLen::USIZE);
        // 3 copies (vs RKNN's 10) clears warmup; Burn forward is slow.
        let writer_handle = std::thread::spawn(move || {
            for _ in 0..3 {
                for chunk in pcm.chunks(1024) {
                    writer.push(chunk);
                    std::thread::sleep(Duration::from_millis(23));
                }
            }
        });

        let mut frames: Vec<InferenceFrame> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline && frames.len() < 4 {
            match out_rx.try_recv() {
                Ok(bytes) => {
                    let f = decode_inference_envelope(bytes.as_ref());
                    frames.push(f);
                }
                Err(broadcast::error::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(broadcast::error::TryRecvError::Closed) => break,
                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            }
        }

        token.cancel();
        writer_handle.join().expect("writer thread panicked");
        engine_handle
            .join()
            .expect("engine thread panicked")
            .expect("engine returned an error");

        assert!(!frames.is_empty(), "no inference frames produced");
        for f in &frames {
            assert!(!f.top_k.is_empty(), "top_k empty in frame {}", f.seq);
            // head_id is proto-optional but always populated in production frames.
            assert!(
                f.head_id.as_deref().is_some_and(|s| !s.is_empty()),
                "head_id absent or empty in frame {}",
                f.seq,
            );
            for tk in &f.top_k {
                assert!(
                    (0.0..=1.0).contains(&tk.prob),
                    "prob out of range in frame {}: {}",
                    f.seq,
                    tk.prob
                );
            }
        }
        eprintln!(
            "stream_parity_e2e_burn: {} frames; first top-1 idx={} prob={:.3}",
            frames.len(),
            frames[0].top_k[0].class_idx,
            frames[0].top_k[0].prob,
        );
    }

    /// With a plumbed `SharedTimingAnchor`, each frame's `t_us_capture_monotonic`
    /// must be the window-start 44.1 kHz sample position projected through the
    /// anchor; back-projection round-trip residual must be within the 1 ms contract.
    #[test]
    #[ignore = "depends on repo-root reference assets; --include-ignored"]
    fn timing_anchor_drives_inference_capture_us_within_one_ms() {
        use crate::common::time::{
            BufferTimingAnchor, CaptureTime, capture_us_for, shared_timing_anchor,
        };

        let root = crate_root();
        let backbone_path = root.join("misc/backbones/backbone.mpk");
        let head_mpk = root.join("misc/heads/default/head.mpk");
        let labels_path = root.join("misc/heads/default/labels.txt");
        let waveform_path = root.join("misc/npys/waveform_0.npy");
        for p in [&backbone_path, &head_mpk, &labels_path, &waveform_path] {
            assert!(p.exists(), "missing test asset: {}", p.display());
        }

        let buf = AudioBuffer::new(262_144);
        let mut writer = buf.take_writer();
        let reader = buf.reader_at(0);

        // Stage anchor before audio flows: head_pos=0, captured_at=1e9 us pin
        // capture_us(N) = 1e9 + N * 1e6 / 44_100.
        let anchor_cell = shared_timing_anchor();
        anchor_cell.store(Arc::new(BufferTimingAnchor {
            head_pos: 0,
            captured_at: CaptureTime::from_micros(1_000_000_000),
            sample_rate_hz: 44_100,
        }));
        let anchor_for_writer = anchor_cell.clone();

        let head = HotHead::load(&head_mpk, &labels_path, crate::common::ids::HeadId::new())
            .expect("load head");
        let cfg = Arc::new(ArcSwap::from_pointee(InferenceCfg {
            hop_samples: 11_025,
            top_k: 3,
        }));
        let (mon_tx, _mon_rx) = watch::channel(Heartbeat::default());
        let pipeline = BackbonePipeline::Burn(Box::new(
            BurnBackbone::load(&backbone_path).expect("load burn backbone"),
        ));
        let engine = InferenceEngine::new(
            pipeline.into_boxed(),
            head,
            cfg,
            mon_tx,
            Some(anchor_cell.clone()),
        );

        let (out_tx, mut out_rx) = broadcast::channel::<Bytes>(64);
        let token = CancellationToken::new();
        let token_engine = token.clone();
        let engine_handle =
            std::thread::spawn(move || engine.run_blocking(reader, out_tx, token_engine));

        // Anchor stays at head_pos=0 so we verify the engine projects against the
        // staged anchor, not a producer-side update.
        let (_, pcm) = npy::read_f32(&waveform_path);
        assert_eq!(pcm.len(), WaveformLen::USIZE);
        let writer_handle = std::thread::spawn(move || {
            for _ in 0..3 {
                for chunk in pcm.chunks(1024) {
                    writer.push(chunk);
                    std::thread::sleep(Duration::from_millis(23));
                }
            }
            anchor_for_writer
        });

        // One frame suffices as proof of contract.
        let mut frames: Vec<InferenceFrame> = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline && frames.is_empty() {
            match out_rx.try_recv() {
                Ok(bytes) => {
                    let f = decode_inference_envelope(bytes.as_ref());
                    frames.push(f);
                }
                Err(broadcast::error::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(broadcast::error::TryRecvError::Closed) => break,
                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
            }
        }

        token.cancel();
        let _final_anchor = writer_handle.join().expect("writer thread panicked");
        engine_handle
            .join()
            .expect("engine thread panicked")
            .expect("engine returned an error");

        assert!(!frames.is_empty(), "no inference frames produced");
        let frame = &frames[0];
        let stamp = frame
            .t_us_capture_monotonic
            .expect("anchor-plumbed engine must populate t_us_capture_monotonic");

        // stamp must lie in the projection image for some N>=0, bounded above by
        // ~3 s of pushed audio (132_096 samples = 3 x WaveformLen) plus slack.
        assert!(
            stamp >= 999_900_000,
            "stamp {stamp} below anchor captured_at; \
             projection went backwards from sample 0",
        );
        assert!(
            stamp <= 1_000_000_000 + 3_500_000,
            "stamp {stamp} above expected upper bound (~3 s of audio)",
        );

        // Back-project N via the inverse of `capture_us_for`; round-trip residual
        // must be <= 1 ms.
        let anchor = **anchor_cell.load();
        let stamp_delta_us = stamp.saturating_sub(anchor.captured_at.as_micros());
        let n_recovered = (stamp_delta_us as u128 * anchor.sample_rate_hz as u128 / 1_000_000)
            as u64
            + anchor.head_pos;
        let projected = capture_us_for(anchor, n_recovered);
        let residual = projected.abs_diff(stamp);
        assert!(
            residual <= 1_000,
            "anchor round-trip residual {residual} us > 1 ms tolerance; \
             stamp {stamp}, recovered N {n_recovered}, projected {projected}",
        );

        eprintln!(
            "timing_anchor_drives_inference_capture_us_within_one_ms: \
             stamp={} us, recovered N={}, residual={} us",
            stamp, n_recovered, residual,
        );
    }
}
