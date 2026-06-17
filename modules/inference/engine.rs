//! Streaming inference engine: PCM ring -> preproc -> backbone -> head ->
//! `InferenceFrame` broadcast. `InferenceEngine` is `Send + !Sync` (the
//! `rknn_runtime::Session` is `!Sync`); moved into one `spawn_blocking` worker,
//! fully sync, scratch heap-allocated once pre-loop. Cancellation polled per
//! outer iter (cancel -> `Ok(())`). Backpressure: `Wait` -> sleep, `Lagged` ->
//! `seek_latest` + skip.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use bytes::Bytes;
use thiserror::Error;
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;

use crate::audio_buffer::{ReadStatus, Reader};
use crate::common::dims::{BackboneFeatureDim, NBins, NFrames, WaveformLen};
use crate::preproc::Preproc;
use crate::proto::{InferenceFrame, TopK};

use crate::inference::backbone::{Backbone, BackboneError};
use crate::inference::head::HotHead;
use crate::inference::kernel::{head_forward, softmax_into, top_k_indices_into};

/// Buffer-channel sample rate; the mic arbitrator always normalizes to 44.1 kHz.
pub const SAMPLE_RATE_HZ: u64 = 44_100;

/// Integer-ns duration of one full [`WaveformLen`] window at [`SAMPLE_RATE_HZ`],
/// for external callers (no internal consumer).
pub const WAVEFORM_DURATION_NS: u64 = WaveformLen::VALUE as u64 * 1_000_000_000 / SAMPLE_RATE_HZ;

/// Initial reservation for top-k / logits / probs vectors; heads up to 32
/// classes never realloc, larger ones realloc once on first frame.
const INITIAL_CLASSES_HINT: usize = 32;

/// Wall-clock (not failure-count, so retry-pacing can't shift it) budget
/// absorbing back-to-back backbone failures before surfacing.
const BACKBONE_FAILURE_BUDGET: Duration = Duration::from_secs(5);

/// Gap beyond which the failure streak re-anchors, so the budget spans only
/// CONTIGUOUS backbone work: a wedge fails every exercised frame (never
/// re-anchors, still trips), but two isolated failures bracketing a quiet room
/// (backbone unexercised between) must not share one anchor. NOT applied to the
/// NaN streak, which must survive idle (see [`EngineError::SustainedNanDrops`]).
const BACKBONE_FAILURE_GAP_RESET: Duration = Duration::from_secs(2);

/// Wall-clock budget dropping NaN/Inf-bearing frames before surfacing the wedge:
/// a backbone returning `Ok(())` with non-finite features slips past the
/// `Err`-only [`BACKBONE_FAILURE_BUDGET`] and would starve forever, and the
/// watchdog can't distinguish `frames_emitted == 0` from "no audio yet".
const NAN_FAILURE_BUDGET: Duration = Duration::from_secs(5);

/// Sleep after each failed backbone call so a fast-failing backbone (µs-return
/// "device not ready") can't tight-loop; 100 ms paces retries to ~10 Hz, above
/// the typical 4 Hz cadence.
const BACKBONE_FAILURE_BACKOFF: Duration = Duration::from_millis(100);

/// Upper bound on the per-iteration `Wait` sleep; 50 ms = 1/8 of the 4 Hz
/// cadence so emission isn't delayed past one hop at the busy end.
const MAX_WAIT_SLEEP: Duration = Duration::from_millis(50);

/// Lower floor for the per-iteration `Wait` sleep; 1 ms avoids both wake-up
/// latency dominance and a tight poll loop on just-barely-not-enough samples.
const MIN_WAIT_SLEEP: Duration = Duration::from_millis(1);

/// Cadence cap on re-asserting the idle `Waiting` heartbeat (the `Wait`/catchup
/// arms spin ~20x/sec, each beat waking the async hb-pump for no state change).
/// INVARIANT: must stay well under the supervisor's 5 s heartbeat-receipt abort
/// (only real engine beats refresh that timer); 500 ms keeps a 10x margin.
const WAITING_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(500);

/// Failure shapes from the streaming inference engine.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("backbone: {0}")]
    Backbone(#[from] BackboneError),
    #[error("encode InferenceFrame: {0}")]
    Encode(#[from] prost::EncodeError),
    /// Non-finite BACKBONE-OUTPUT features dropped past `NAN_FAILURE_BUDGET`, so
    /// the supervisor restarts rather than spin. Spectrogram-NaN is excluded: an
    /// all-NaN spectrogram is the intentional digital-silence signal (preproc's
    /// bare `ln` on zero-magnitude bins gives `-inf` log-mags, NaN-ing mean/std
    /// and the whole plane), survivable forever so a muted mic doesn't trip this.
    #[error("non-finite frames dropped for {budget_ms} ms (streak {streak_count} frames)")]
    SustainedNanDrops { budget_ms: u64, streak_count: u64 },
}

/// Tunable inference cadence, held via `Arc<ArcSwap<InferenceCfg>>` so the API
/// can change it mid-stream without touching the engine.
#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InferenceCfg {
    /// Stride between successive windows, in samples. First inference fires once
    /// `WaveformLen::USIZE` is buffered, then advances by this. Valid range
    /// [`Self::MIN_HOP_SAMPLES`]..=[`Self::MAX_HOP_SAMPLES`]; smaller hops cost
    /// CPU but cut per-frame latency.
    pub hop_samples: usize,

    pub top_k: usize,
}

impl Default for InferenceCfg {
    fn default() -> Self {
        Self {
            // Relaxed end (1 s hop, 1 Hz): Speech-Commands low-power baseline.
            hop_samples: Self::MAX_HOP_SAMPLES,
            top_k: 20,
        }
    }
}

impl InferenceCfg {
    /// Largest fraction of the window successive hops may overlap; 75% =
    /// re-classify the same audio 4x/window, the practical ceiling before
    /// per-frame work wastes CPU. Drives [`Self::MIN_HOP_SAMPLES`].
    pub const MAX_OVERLAP_RATIO: f32 = 0.75;

    /// Smallest accepted `hop_samples` = `sample_rate * (1 - MAX_OVERLAP_RATIO)`
    /// = 11_025 at canonical (44_100, 0.75) (exact in f32; `as usize` floors).
    /// Smaller hops overlap >75% and burn CPU on the same audio.
    pub const MIN_HOP_SAMPLES: usize =
        ((SAMPLE_RATE_HZ as f32) * (1.0 - Self::MAX_OVERLAP_RATIO)) as usize;

    /// Largest accepted `hop_samples` = `sample_rate` (1 inference/sec); exceeds
    /// [`WaveformLen::USIZE`] (44_032) by 68 samples (~1.5 ms), so non-overlapping
    /// windows wait on `Reader::available` before `advance` to keep the buffer's
    /// `tail <= head` invariant.
    pub const MAX_HOP_SAMPLES: usize = SAMPLE_RATE_HZ as usize;

    /// Hard cap on `top_k`, matching the fixed buffer + protobuf footprint.
    pub const MAX_TOP_K: usize = 64;

    /// Explicit-reject hook for API + config loaders; the hot loop additionally
    /// clamps `hop_samples` against validator-bypass paths.
    pub fn validate(&self) -> Result<(), String> {
        if self.hop_samples < Self::MIN_HOP_SAMPLES || self.hop_samples > Self::MAX_HOP_SAMPLES {
            return Err(format!(
                "hop_samples must be {}..={}; got {}",
                Self::MIN_HOP_SAMPLES,
                Self::MAX_HOP_SAMPLES,
                self.hop_samples
            ));
        }
        if self.top_k == 0 || self.top_k > Self::MAX_TOP_K {
            return Err(format!(
                "top_k must be 1..={}; got {}",
                Self::MAX_TOP_K,
                self.top_k
            ));
        }
        Ok(())
    }
}

/// Running totals across loop iterations, bundled so heartbeat-emit helpers take
/// one reference, not 4 reorder-prone positional `u64`s; `Heartbeat` mirrors
/// these for the public watch channel.
#[derive(Debug, Clone, Copy, Default)]
struct EngineCounters {
    last_seq: u64,
    frames_emitted: u64,
    frames_dropped_nan: u64,
    frames_dropped_lag: u64,
}

/// Per-iteration liveness signal via `watch::Sender<Heartbeat>` so the status
/// crate sees engine liveness without polling the broadcast channel. `Default`
/// is constructible before start so the daemon builds the channel up-front.
#[derive(Debug, Clone, Copy)]
pub struct Heartbeat {
    pub at: Instant,
    pub state: EngineState,
    /// Most recent emitted seq, or 0 if no frame yet.
    pub last_seq: u64,
    pub frames_emitted: u64,
    /// Frames dropped on non-finite values, from both sites: a NaN/Inf
    /// spectrogram (intentional digital silence, survivable forever) and NaN/Inf
    /// backbone-output features (a fault gated by `NAN_FAILURE_BUDGET`).
    pub frames_dropped_nan: u64,
    /// Inference windows skipped on `Lagged`, `ceil(by_samples / hop_samples)`
    /// per event: missed cadence ticks, not `Lagged` events (one event commonly
    /// increments by many).
    pub frames_dropped_lag: u64,
}

impl Default for Heartbeat {
    fn default() -> Self {
        Self {
            at: Instant::now(),
            state: EngineState::Starting,
            last_seq: 0,
            frames_emitted: 0,
            frames_dropped_nan: 0,
            frames_dropped_lag: 0,
        }
    }
}

/// Coarse health state of the engine, surfaced via the [`Heartbeat`] feed.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EngineState {
    /// Pre-loop init.
    Starting,
    Running,
    /// Buffer underrun (`ReadStatus::Wait`).
    Waiting,
    /// Buffer lap (`ReadStatus::Lagged`).
    Lagged,
    /// Loop exited cleanly via cancellation.
    Stopped,
    /// Loop exited via error (returned to caller).
    Failed,
}

/// Lowercase display matching the daemon's lowercase-prefix convention.
impl std::fmt::Display for EngineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Starting => "starting",
            Self::Running => "running",
            Self::Waiting => "waiting",
            Self::Lagged => "lagged",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
        })
    }
}

/// Owning engine: constructed once, moved into `spawn_blocking`, never cloned.
/// `backbone` is a trait object so tests substitute mocks without going through
/// `BackbonePipeline`'s cfg-gated arms.
pub struct InferenceEngine {
    preproc: Preproc,
    backbone: Box<dyn Backbone>,
    head: HotHead,
    cfg: Arc<ArcSwap<InferenceCfg>>,
    monitor: watch::Sender<Heartbeat>,
    /// When `Some`, `InferenceFrame.t_us_capture_monotonic` is projected from the
    /// window's FIRST 44.1 kHz sample; `None` stays `None` on the wire (tests pass
    /// `None`, production always supplies one).
    timing_anchor: Option<crate::common::time::SharedTimingAnchor>,
}

impl std::fmt::Debug for InferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("backbone", &self.backbone)
            .field("head", &self.head)
            .field("cfg", &self.cfg.load())
            .finish_non_exhaustive()
    }
}

impl InferenceEngine {
    /// Initialize with a pre-constructed backbone; the caller chooses + loads it
    /// so the RKNN-first / Burn-fallback policy lives in one place. Infallible:
    /// validation is the backbone constructor's job.
    pub fn new(
        backbone: Box<dyn Backbone>,
        head: HotHead,
        cfg: Arc<ArcSwap<InferenceCfg>>,
        monitor: watch::Sender<Heartbeat>,
        timing_anchor: Option<crate::common::time::SharedTimingAnchor>,
    ) -> Self {
        Self {
            preproc: Preproc::new(),
            backbone,
            head,
            cfg,
            monitor,
            timing_anchor,
        }
    }

    /// Run forever. Consumes self (Session owns runtime resources that must not
    /// outlive this call). `Ok(())` on clean shutdown, `Err` on mid-stream
    /// backbone failure (supervisor restarts the task).
    pub fn run_blocking(
        mut self,
        mut reader: Reader,
        out: broadcast::Sender<Bytes>,
        shutdown: CancellationToken,
    ) -> Result<(), EngineError> {
        let result = self.run_blocking_inner(&mut reader, &out, &shutdown);
        let final_state = if result.is_err() {
            EngineState::Failed
        } else {
            EngineState::Stopped
        };
        self.send_heartbeat(|hb| {
            hb.at = Instant::now();
            hb.state = final_state;
        });
        result
    }

    fn run_blocking_inner(
        &mut self,
        reader: &mut Reader,
        out: &broadcast::Sender<Bytes>,
        shutdown: &CancellationToken,
    ) -> Result<(), EngineError> {
        // Pre-loop scratch (allocated once); `*_into` variants overwrite every
        // cell, so initial state is immaterial.
        let mut pcm = Box::new([0.0f32; WaveformLen::USIZE]);
        let mut spec_buf: Box<[[f32; NBins::USIZE]; NFrames::USIZE]> =
            Box::new([[0.0f32; NBins::USIZE]; NFrames::USIZE]);
        let mut features = Box::new([0.0f32; BackboneFeatureDim::USIZE]);
        let mut logits: Vec<f32> = Vec::with_capacity(INITIAL_CLASSES_HINT);
        let mut probs: Vec<f32> = Vec::with_capacity(INITIAL_CLASSES_HINT);
        let mut top_idx: Vec<usize> = Vec::with_capacity(INITIAL_CLASSES_HINT);
        // Cached prior-frame n_classes so the three vectors reserve once per
        // head-load; 0 makes the first `n > reserved_n_classes` true (validate
        // guarantees `n >= 1`).
        let mut reserved_n_classes: usize = 0;
        // `BytesMut::split().freeze()` yields Arc-backed `Bytes` (zero-copy
        // fan-out) reusing residual capacity; 4 KiB fits one typical envelope.
        let mut encode_buf: bytes::BytesMut = bytes::BytesMut::with_capacity(4096);

        // Local mirror of the heartbeat counters to avoid load-then-store per frame.
        let mut counters = EngineCounters::default();

        // `consecutive_failures` drives only log throttling (power-of-two thinning);
        // the hard-exit budget is wall-clock via `failure_streak_start`, re-anchored
        // when `last_failure_at` shows a gap (see `BACKBONE_FAILURE_GAP_RESET`).
        let mut consecutive_failures: u32 = 0;
        let mut failure_streak_start: Option<Instant> = None;
        let mut last_failure_at: Option<Instant> = None;

        // NaN-streak budget; reset on any finite emit, anchored at the FIRST drop
        // so retry pacing can't shift it. Surfaces as `SustainedNanDrops`.
        let mut nan_streak_count: u64 = 0;
        let mut nan_streak_start: Option<Instant> = None;

        // Warn once per distinct out-of-range hop, not per iteration.
        let mut last_warned_hop_clamp: Option<usize> = None;

        // `None` fires immediately so entry into Waiting shows promptly, then
        // throttled to `WAITING_HEARTBEAT_INTERVAL`.
        let mut last_waiting_beat: Option<Instant> = None;

        // One pre-loop Running beat so a fast receiver `await` sees liveness.
        self.send_heartbeat_state(EngineState::Running, &counters);

        'engine_main: loop {
            if shutdown.is_cancelled() {
                return Ok(());
            }

            let cfg_snap = self.cfg.load_full();
            // Clamp both ends against a validator-bypass swap: an over-large hop
            // would spin `available() < hop_samples` forever, a hop of 1 would run
            // ~44 kHz (one inference/sample), DoSing the backbone.
            let raw_hop_samples = cfg_snap.hop_samples;
            let hop_samples =
                raw_hop_samples.clamp(InferenceCfg::MIN_HOP_SAMPLES, InferenceCfg::MAX_HOP_SAMPLES);
            if raw_hop_samples != hop_samples && last_warned_hop_clamp != Some(raw_hop_samples) {
                tracing::warn!(
                    target: "inference",
                    raw = raw_hop_samples,
                    clamped = hop_samples,
                    min = InferenceCfg::MIN_HOP_SAMPLES,
                    max = InferenceCfg::MAX_HOP_SAMPLES,
                    "InferenceCfg.hop_samples out of range; clamping to published bound",
                );
                last_warned_hop_clamp = Some(raw_hop_samples);
            }
            // Integer ns to avoid f64 noise on small hops.
            let hop_ns = hop_samples as u64 * 1_000_000_000 / SAMPLE_RATE_HZ;
            // Wake 4x per hop, clamped to [MIN_WAIT_SLEEP, MAX_WAIT_SLEEP].
            let wait_ns = (hop_ns / 4).min(MAX_WAIT_SLEEP.as_nanos() as u64);
            let wait_sleep = Duration::from_nanos(wait_ns.max(MIN_WAIT_SLEEP.as_nanos() as u64));

            match reader.peek_into(&mut pcm[..]) {
                ReadStatus::Wait => {
                    // Must NOT reset the feature-NaN streak: the SustainedNanDrops
                    // budget must survive idle gaps so a wedge interleaved with
                    // inter-hop waits still trips.
                    self.send_waiting_heartbeat_throttled(&counters, &mut last_waiting_beat);
                    std::thread::sleep(wait_sleep);
                    continue;
                }
                ReadStatus::Lagged { by } => {
                    let dropped_windows = by.div_ceil(hop_samples as u64);
                    tracing::warn!(
                        target: "inference",
                        by_samples = by,
                        dropped_windows = dropped_windows,
                        "audio reader lagged; resyncing to latest window",
                    );
                    reader.seek_latest(WaveformLen::USIZE);
                    counters.frames_dropped_lag =
                        counters.frames_dropped_lag.saturating_add(dropped_windows);
                    // Must NOT reset the feature-NaN streak (see Wait path).
                    self.send_heartbeat_state(EngineState::Lagged, &counters);
                    continue;
                }
                ReadStatus::Ready => {}
            }
            // Snapshot tail BEFORE advance: the anchor-derived capture stamp refers
            // to the window's FIRST sample; after advance tail points at the next hop.
            let window_start_tail = reader.tail();
            // When `hop_samples > WaveformLen::USIZE` (1 Hz) peek-Ready guarantees
            // only `head - tail >= WaveformLen`, so a direct `advance` would break
            // the `tail <= head` invariant; spin on `Reader::available` (~1.5 ms /
            // 68-sample window). Lag is detected INSIDE the spin so a writer surge
            // past `safe_peek_window` isn't uncounted, routed through the same Lagged
            // handling; pcm is locally owned, so a missed lap only undercounts
            // `frames_dropped_lag`.
            let safe_window = reader.safe_peek_window() as u64;
            let mut wait_observed_lag: Option<u64> = None;
            while (reader.available() as usize) < hop_samples {
                if shutdown.is_cancelled() {
                    return Ok(());
                }
                let avail = reader.available();
                if avail > safe_window {
                    wait_observed_lag = Some(avail);
                    break;
                }
                self.send_waiting_heartbeat_throttled(&counters, &mut last_waiting_beat);
                // Sleep sized to the remaining-samples gap, floored at `MIN_WAIT_SLEEP`
                // (no busy-spin) and clamped at `wait_sleep` (a buggy `available()`
                // can't sleep multi-second).
                let remaining = (hop_samples as u64).saturating_sub(avail);
                let remaining_ns = remaining
                    .saturating_mul(1_000_000_000)
                    .saturating_div(SAMPLE_RATE_HZ);
                let catchup_sleep =
                    Duration::from_nanos(remaining_ns.max(MIN_WAIT_SLEEP.as_nanos() as u64));
                std::thread::sleep(catchup_sleep.min(wait_sleep));
            }
            if let Some(by) = wait_observed_lag {
                let dropped_windows = by.div_ceil(hop_samples as u64);
                tracing::warn!(
                    target: "inference",
                    by_samples = by,
                    dropped_windows = dropped_windows,
                    "audio reader lagged during hop-catchup wait loop; resyncing to latest window",
                );
                reader.seek_latest(WaveformLen::USIZE);
                counters.frames_dropped_lag =
                    counters.frames_dropped_lag.saturating_add(dropped_windows);
                // Must NOT reset the feature-NaN streak (see Wait path).
                self.send_heartbeat_state(EngineState::Lagged, &counters);
                continue 'engine_main;
            }
            reader.advance(hop_samples);

            // Capture-monotonic stamp of the window's FIRST sample; no anchor -> None.
            let t_us_capture_monotonic = self.timing_anchor.as_ref().map(|cell| {
                let anchor = **cell.load();
                crate::common::time::capture_us_for(anchor, window_start_tail)
            });

            // Flat-slice scan auto-vectorizes; `iter().flatten()` inhibits the
            // SIMD `is_finite`.
            self.preproc.spectrogram_into(&pcm, &mut spec_buf);
            if spec_buf
                .as_slice()
                .as_flattened()
                .iter()
                .any(|v| !v.is_finite())
            {
                tracing::warn!(
                    target: "inference",
                    seq = counters.last_seq + 1,
                    "frame dropped: NaN/Inf in spec",
                );
                counters.frames_dropped_nan = counters.frames_dropped_nan.saturating_add(1);
                // Spectrogram-NaN is intentional silence-suppression: does NOT feed
                // `bump_nan_streak` (silence survives forever); only the post-backbone
                // feature-NaN gate counts toward the wedge.
                self.send_heartbeat_state(EngineState::Running, &counters);
                continue;
            }

            match self.backbone.infer(&spec_buf, &mut features) {
                Ok(()) => {
                    if consecutive_failures > 0 {
                        tracing::info!(
                            target: "inference",
                            backbone = self.backbone.description(),
                            consecutive_failures,
                            "backbone recovered after failure streak",
                        );
                        consecutive_failures = 0;
                        failure_streak_start = None;
                    }
                }
                Err(e) => {
                    let now = Instant::now();
                    // Re-anchor on a gap (backbone unexercised between).
                    if let Some(last) = last_failure_at
                        && now.saturating_duration_since(last) > BACKBONE_FAILURE_GAP_RESET
                    {
                        consecutive_failures = 0;
                        failure_streak_start = None;
                    }
                    last_failure_at = Some(now);
                    consecutive_failures = consecutive_failures.saturating_add(1);
                    let streak_started_at = *failure_streak_start.get_or_insert(now);
                    let streak_elapsed = now.saturating_duration_since(streak_started_at);
                    if consecutive_failures == 1 {
                        tracing::error!(
                            target: "inference",
                            err = %e,
                            backbone = self.backbone.description(),
                            "backbone failure; engine will retry",
                        );
                    } else if consecutive_failures.is_power_of_two() {
                        tracing::warn!(
                            target: "inference",
                            err = %e,
                            backbone = self.backbone.description(),
                            consecutive_failures,
                            streak_ms = streak_elapsed.as_millis() as u64,
                            "backbone failure (continued)",
                        );
                    }
                    if streak_elapsed >= BACKBONE_FAILURE_BUDGET {
                        tracing::error!(
                            target: "inference",
                            err = %e,
                            backbone = self.backbone.description(),
                            consecutive_failures,
                            streak_ms = streak_elapsed.as_millis() as u64,
                            budget_ms = BACKBONE_FAILURE_BUDGET.as_millis() as u64,
                            "backbone failure streak exceeded budget; engine giving up",
                        );
                        return Err(e.into());
                    }
                    // Clear the NaN-drop streak: the two budgets measure distinct
                    // failure shapes, else a NaN drop bracketing a long backbone-Err
                    // window would falsely surface `SustainedNanDrops`.
                    nan_streak_count = 0;
                    nan_streak_start = None;
                    // Heartbeat required, else a silent Err arm could race the
                    // watchdog into `process::abort()` before `BACKBONE_FAILURE_BUDGET`
                    // (the intended sole authority) elapses.
                    self.send_heartbeat_state(EngineState::Running, &counters);
                    std::thread::sleep(BACKBONE_FAILURE_BACKOFF);
                    continue;
                }
            }

            // A misbehaving backbone (driver hiccup, fp16 underflow) can emit
            // NaN/Inf features; unchecked, `head_forward`'s matmul propagates NaN
            // and `softmax_into` emits a meaningless uniform `1/n`.
            if features.iter().any(|v| !v.is_finite()) {
                tracing::warn!(
                    target: "inference",
                    seq = counters.last_seq + 1,
                    backbone = self.backbone.description(),
                    "frame dropped: NaN/Inf in backbone output features",
                );
                counters.frames_dropped_nan = counters.frames_dropped_nan.saturating_add(1);
                bump_nan_streak(
                    &mut nan_streak_count,
                    &mut nan_streak_start,
                    Instant::now(),
                    NAN_FAILURE_BUDGET,
                )?;
                self.send_heartbeat_state(EngineState::Running, &counters);
                continue;
            }
            // Past both NaN gates -- reset so the budget spans only CONSECUTIVE drops.
            nan_streak_count = 0;
            nan_streak_start = None;

            // Atomic `(snapshot, version)`: stamping version separately would race
            // a swap landing between the reads, yielding logits from version N
            // stamped N+1.
            let (snap, head_version) = self.head.snapshot_with_version();
            let n = snap.n_classes;
            // Reserve once per head-load when n grows so the resize below never
            // reallocs (head_forward overwrites every entry).
            if n > reserved_n_classes {
                logits.reserve(n.saturating_sub(logits.len()));
                probs.reserve(n.saturating_sub(probs.len()));
                top_idx.reserve(n.saturating_sub(top_idx.len()));
                reserved_n_classes = n;
            }
            logits.resize(n, 0.0);
            probs.resize(n, 0.0);
            head_forward(&features[..], &snap.weight, &snap.bias, &mut logits);
            softmax_into(&logits, &mut probs);
            top_k_indices_into(&probs, cfg_snap.top_k, &mut top_idx);

            let next_seq = counters.last_seq.wrapping_add(1);
            // Sampled just BEFORE broadcast (at advance time it would bias ~30-100 ms
            // early); `None` (pre-epoch clock) stays `None`, not a misleading zero.
            let t_us_publish_unix = crate::common::time::WallTime::now().map(|w| w.as_micros());
            let frame = InferenceFrame {
                seq: next_seq,
                t_us_capture_monotonic,
                t_us_publish_unix,
                head_id: Some(snap.head_id.to_string()),
                head_version: Some(head_version.get()),
                top_k: top_idx
                    .iter()
                    .map(|&i| TopK {
                        class_idx: i as u32,
                        label: snap.labels[i].clone(),
                        prob: probs[i],
                    })
                    .collect(),
            };

            // Ignore `SendError`: no subscribers is steady state (no UI connected).
            let payload = crate::proto::framing::wrap_inference_into(&mut encode_buf, frame);
            let _ = out.send(payload);

            counters.last_seq = next_seq;
            counters.frames_emitted = counters.frames_emitted.saturating_add(1);
            self.send_heartbeat_state(EngineState::Running, &counters);
        }
    }

    fn send_heartbeat<F: FnOnce(&mut Heartbeat)>(&self, edit: F) {
        self.monitor.send_modify(edit);
    }

    /// Throttled `Waiting` beat for the idle `Wait`/catchup arms; every other
    /// state beats unconditionally so a real transition or drop is never delayed.
    fn send_waiting_heartbeat_throttled(
        &self,
        counters: &EngineCounters,
        last_waiting_beat: &mut Option<Instant>,
    ) {
        let now = Instant::now();
        if waiting_beat_due(*last_waiting_beat, now, WAITING_HEARTBEAT_INTERVAL) {
            self.send_heartbeat_state(EngineState::Waiting, counters);
            *last_waiting_beat = Some(now);
        }
    }

    fn send_heartbeat_state(&self, state: EngineState, counters: &EngineCounters) {
        // Copy out so the closure doesn't borrow self.monitor.
        let snap = *counters;
        self.send_heartbeat(|hb| {
            hb.at = Instant::now();
            hb.state = state;
            hb.last_seq = snap.last_seq;
            hb.frames_emitted = snap.frames_emitted;
            hb.frames_dropped_nan = snap.frames_dropped_nan;
            hb.frames_dropped_lag = snap.frames_dropped_lag;
        });
    }
}

/// Whether a throttled `Waiting` heartbeat is due: on first entry (`None`) or
/// once `interval` elapsed since the last beat. `now`/`interval` explicit so
/// tests hit the boundary without sleeping.
fn waiting_beat_due(last_sent: Option<Instant>, now: Instant, interval: Duration) -> bool {
    match last_sent {
        None => true,
        Some(prev) => now.saturating_duration_since(prev) >= interval,
    }
}

/// Bump the NaN-drop streak count and check the wall-clock budget. `start`
/// anchors at the FIRST call so the budget measures total streak duration, not
/// inter-call gap; the caller resets (`count = 0; start = None;`) on a clean frame
/// or backbone-Err transition, else the anchor conflates with backbone-Err windows.
/// Called ONLY from the post-backbone feature-NaN gate (spectrogram-NaN is
/// intentional silence, not fed here).
fn bump_nan_streak(
    count: &mut u64,
    start: &mut Option<Instant>,
    now: Instant,
    budget: Duration,
) -> Result<(), EngineError> {
    *count = count.saturating_add(1);
    let started_at = *start.get_or_insert(now);
    if now.saturating_duration_since(started_at) >= budget {
        return Err(EngineError::SustainedNanDrops {
            budget_ms: budget.as_millis() as u64,
            streak_count: *count,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waveform_duration_ns_is_998_458_049() {
        assert_eq!(WAVEFORM_DURATION_NS, 998_458_049);
    }

    #[test]
    fn default_cfg_is_1hz_hop() {
        let c = InferenceCfg::default();
        assert_eq!(c.hop_samples, 44_100);
        assert_eq!(c.top_k, 20);
    }

    #[test]
    fn waiting_beat_due_fires_on_entry_then_only_after_interval() {
        let t0 = Instant::now();
        let interval = Duration::from_millis(500);
        assert!(waiting_beat_due(None, t0, interval));
        assert!(!waiting_beat_due(Some(t0), t0, interval));
        assert!(!waiting_beat_due(
            Some(t0),
            t0 + Duration::from_millis(499),
            interval
        ));
        // Boundary inclusive.
        assert!(waiting_beat_due(Some(t0), t0 + interval, interval));
        assert!(waiting_beat_due(
            Some(t0),
            t0 + Duration::from_millis(750),
            interval
        ));
    }

    /// Guards the throttle-vs-watchdog invariant: interval must stay well under
    /// the supervisor's 5 s heartbeat-receipt abort.
    #[test]
    fn waiting_heartbeat_interval_well_under_abort_window() {
        assert!(WAITING_HEARTBEAT_INTERVAL <= Duration::from_secs(1));
    }

    /// Pin the bound literals so a `MAX_OVERLAP_RATIO`/`SAMPLE_RATE_HZ` retune
    /// shows up as a deliberate edit, not accidental drift.
    #[test]
    fn hop_samples_bounds_match_sample_rate_policy() {
        assert_eq!(InferenceCfg::MIN_HOP_SAMPLES, 11_025);
        assert_eq!(InferenceCfg::MAX_HOP_SAMPLES, 44_100);
        // Default sits at the relaxed end: operators can only narrow the hop.
        assert_eq!(
            InferenceCfg::default().hop_samples,
            InferenceCfg::MAX_HOP_SAMPLES,
        );
        // Upper bound exceeds WaveformLen (drives the spin).
        const _: () = assert!(InferenceCfg::MAX_HOP_SAMPLES > WaveformLen::USIZE);
    }

    #[test]
    fn validate_rejects_out_of_range_hop() {
        let base = InferenceCfg::default();

        let mut c = base;
        c.hop_samples = InferenceCfg::MIN_HOP_SAMPLES - 1;
        let err = c.validate().expect_err("below min must reject");
        assert!(err.contains("hop_samples"), "{err}");

        let mut c = base;
        c.hop_samples = 0;
        c.validate().expect_err("zero must reject");

        let mut c = base;
        c.hop_samples = InferenceCfg::MAX_HOP_SAMPLES + 1;
        c.validate().expect_err("above max must reject");

        let mut c = base;
        c.hop_samples = InferenceCfg::MIN_HOP_SAMPLES;
        c.validate().expect("min must accept");
        c.hop_samples = InferenceCfg::MAX_HOP_SAMPLES;
        c.validate().expect("max must accept");
    }

    #[test]
    fn heartbeat_default_state_is_starting() {
        let hb = Heartbeat::default();
        assert_eq!(hb.state, EngineState::Starting);
        assert_eq!(hb.last_seq, 0);
        assert_eq!(hb.frames_emitted, 0);
    }

    /// Pin the lowercase form so a variant rename can't silently change
    /// user-facing dashboard output.
    #[test]
    fn engine_state_display_is_lowercase() {
        assert_eq!(EngineState::Starting.to_string(), "starting");
        assert_eq!(EngineState::Running.to_string(), "running");
        assert_eq!(EngineState::Waiting.to_string(), "waiting");
        assert_eq!(EngineState::Lagged.to_string(), "lagged");
        assert_eq!(EngineState::Stopped.to_string(), "stopped");
        assert_eq!(EngineState::Failed.to_string(), "failed");
    }

    #[test]
    fn wall_clock_now_post_epoch() {
        let t = crate::common::time::WallTime::now()
            .expect("wall clock post-epoch")
            .as_micros();
        // After 2001-09-09 (~10^15 us).
        assert!(t > 1_000_000_000_000_000, "wall clock < 2001? t={t}");
    }

    /// First bump anchors, mid-budget bump holds the anchor, at-budget bump Errs.
    #[test]
    fn bump_nan_streak_anchors_at_first_call_and_fires_at_budget() {
        let budget = Duration::from_secs(5);
        let t0 = Instant::now();
        let mut count: u64 = 0;
        let mut start: Option<Instant> = None;

        bump_nan_streak(&mut count, &mut start, t0, budget).expect("first bump in-budget");
        assert_eq!(count, 1);
        assert_eq!(start, Some(t0));

        let t_mid = t0 + Duration::from_secs(4);
        bump_nan_streak(&mut count, &mut start, t_mid, budget).expect("mid bump in-budget");
        assert_eq!(count, 2);
        assert_eq!(start, Some(t0), "anchor must not slide on subsequent bumps");

        let t_edge = t0 + Duration::from_secs(5);
        let err = bump_nan_streak(&mut count, &mut start, t_edge, budget)
            .expect_err("at-budget bump must fire");
        match err {
            EngineError::SustainedNanDrops {
                budget_ms,
                streak_count,
            } => {
                assert_eq!(budget_ms, 5_000);
                assert_eq!(streak_count, 3, "count must reflect this final bump");
            }
            other => panic!("expected SustainedNanDrops; got {other:?}"),
        }
    }

    /// After a backbone-Err reset (`count = 0; start = None;`) a fresh streak
    /// re-anchors at the first post-reset bump and must NOT fire past the
    /// original budget.
    #[test]
    fn bump_nan_streak_reset_restarts_budget_anchor() {
        let budget = Duration::from_secs(5);
        let t0 = Instant::now();
        let mut count: u64 = 0;
        let mut start: Option<Instant> = None;

        bump_nan_streak(&mut count, &mut start, t0, budget).unwrap();
        bump_nan_streak(&mut count, &mut start, t0 + Duration::from_secs(1), budget).unwrap();
        assert_eq!(count, 2);
        assert_eq!(start, Some(t0));

        // Mirror the engine's backbone-Err reset.
        count = 0;
        start = None;

        // Past original anchor + budget: a t0-anchored bump would fire, but the
        // reset re-anchored at `t_resume`, so this must be Ok.
        let t_resume = t0 + Duration::from_secs(10);
        bump_nan_streak(&mut count, &mut start, t_resume, budget)
            .expect("post-reset bump must Ok: anchor restarted from t_resume");
        assert_eq!(count, 1, "reset must zero the count");
        assert_eq!(
            start,
            Some(t_resume),
            "anchor must restart at the post-reset bump's now",
        );

        bump_nan_streak(
            &mut count,
            &mut start,
            t_resume + Duration::from_secs(2),
            budget,
        )
        .expect("second post-reset bump still in fresh budget");
        assert_eq!(count, 2);
        assert_eq!(start, Some(t_resume), "anchor must hold after the reset");
    }

    /// Pin the power-of-two log-thinning algebra so a refactor can't silently
    /// change the cadence. `STREAK_CAP = 50` ~ worst-case retry count at 100 ms
    /// backoff under the 5 s wall-clock budget (the real hard exit).
    #[test]
    fn backbone_failure_log_thinning_is_logarithmic() {
        const STREAK_CAP: u32 = 50;
        let mut log_lines: Vec<u32> = Vec::new();
        for streak in 1..=STREAK_CAP {
            if streak == 1 || streak.is_power_of_two() {
                log_lines.push(streak);
            }
        }
        // The final hard-exit line is unconditional at the call site, not this
        // throttling formula.
        assert_eq!(
            log_lines,
            vec![1, 2, 4, 8, 16, 32],
            "log thinning shifted; expected 6 power-of-two boundaries in [1, 50]",
        );
    }
}
