//! Mic arbitrator: operator-set/first-available mic selection + RMS-arbitrated
//! intra-mic channel switching (no cross-mic RMS comparison). Immutable
//! catalogue (launch-TOML) and hot-reloadable policy bundle into one
//! [`MicSettings`] so the hot loop reads both with a single `ArcSwap` load.
//!
//! One std thread (`mic-arbitrator`); per period: snapshot settings, resolve +
//! open the desired mic, read a period, demux + per-slot EMA RMS, pick a slot,
//! then write the active slot directly (native rate) or feed every slot's
//! resampler (all FIR states kept current for glitch-free switching) and drain
//! only the active slot's output.

mod store;
#[cfg(test)]
mod tests;
mod types;

pub use store::{ArcSwapStore, MicSettingsStore};
pub use types::{
    CandidateError, CandidateSource, ChannelSelection, MAX_BUFFER_FRAMES, MAX_CHANNEL_INDEX,
    MAX_CHANNELS, MAX_NEGOTIABLE_CHANNELS, MAX_PERIOD_FRAMES, MAX_SAMPLE_RATE, MIN_SAMPLE_RATE,
    MicCandidate, MicCatalogue, MicPolicy, MicSelection, MicSettings, PolicyValidationError,
};

use crate::audio_buffer::Writer;
use crate::audio_io::source::{ActiveSource, OpenError, open_source};
use crate::common::dims::SampleRate;
use crate::common::ids::MicId;
use crate::dsp::resample::Streaming;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Tunables for the mic-arbitrator's run loop.  Validate via [`Self::validate`]
/// before constructing a [`MicArbitrator`].
#[derive(Clone, Debug)]
pub struct MicArbitratorConfig {
    /// dB margin a candidate channel must beat the active channel by to switch.
    pub hysteresis_db: f32,
    /// Minimum hold time before another channel switch.
    pub dwell: Duration,
    /// EMA time constant for per-channel RMS.
    pub rms_window: Duration,
    /// `FirstAvailable`: silence-before-failover-walk threshold.  `Fixed` ignores.
    pub mic_failover_after: Duration,
    /// `FirstAvailable`: sleep between retry-walks when no candidate opens.
    pub failover_retry_interval: Duration,
    /// Pin the thread to a CPU core.  `None` = kernel default.  Best-effort.
    pub sched_pin: Option<usize>,
    /// `SCHED_FIFO` at this priority (needs `CAP_SYS_NICE`); `None` keeps
    /// `SCHED_OTHER` (best-effort; failure -> occasional ALSA underruns).
    pub sched_priority: Option<i32>,
    /// When set, publish a [`crate::common::time::BufferTimingAnchor`] after each
    /// `Writer::push` so consumers project sample-position to capture time without
    /// the producer; `None` = consumers stamp `CaptureTime::now()` at emit (looser).
    pub timing_anchor: Option<crate::common::time::SharedTimingAnchor>,
}

impl Default for MicArbitratorConfig {
    fn default() -> Self {
        Self {
            hysteresis_db: 3.0,
            dwell: Duration::from_millis(250),
            rms_window: Duration::from_millis(100),
            mic_failover_after: Duration::from_secs(2),
            failover_retry_interval: Duration::from_secs(1),
            // No-pin/realtime/anchor so tests + macOS dev hosts work; production overrides.
            sched_pin: None,
            sched_priority: None,
            timing_anchor: None,
        }
    }
}

impl MicArbitratorConfig {
    /// Linear RMS ratio a candidate must beat the active channel by to switch.
    /// Factor 20 (not 10) because RMS is amplitude, not power.  Called once per
    /// arbitrator lifetime, keeping `powf` out of the hot path.
    pub fn hysteresis_linear(&self) -> f32 {
        10f32.powf(self.hysteresis_db / 20.0)
    }

    /// Sanity-check operator tunables.  Rejects negative `hysteresis_db` (inverts
    /// the comparison, oscillates), zero `rms_window` (div-by-zero EMA), and zero
    /// `mic_failover_after`/`failover_retry_interval` (tight-loop the gated
    /// branches); `dwell` zero is legal.
    pub fn validate(&self) -> Result<(), String> {
        // Soft cap catching unit typos (60 s where 1 s meant): an hour-long
        // interval/dwell would delay hot-unplug recovery or latch the boot channel.
        const MAX_DURATION: Duration = Duration::from_secs(60);

        if !(self.hysteresis_db.is_finite() && self.hysteresis_db >= 0.0) {
            return Err(format!(
                "mic_arbitrator: hysteresis_db must be finite and >= 0.0; got {}",
                self.hysteresis_db
            ));
        }
        if self.rms_window.is_zero() {
            return Err("mic_arbitrator: rms_window must be > 0".into());
        }
        if self.rms_window > MAX_DURATION {
            return Err(format!(
                "mic_arbitrator: rms_window {:?} exceeds the {MAX_DURATION:?} sanity cap",
                self.rms_window,
            ));
        }
        if self.dwell > MAX_DURATION {
            return Err(format!(
                "mic_arbitrator: dwell {:?} exceeds the {MAX_DURATION:?} sanity cap",
                self.dwell,
            ));
        }
        if self.mic_failover_after.is_zero() {
            return Err("mic_arbitrator: mic_failover_after must be > 0".into());
        }
        if self.mic_failover_after > MAX_DURATION {
            return Err(format!(
                "mic_arbitrator: mic_failover_after {:?} exceeds the {MAX_DURATION:?} sanity cap",
                self.mic_failover_after,
            ));
        }
        if self.failover_retry_interval.is_zero() {
            return Err("mic_arbitrator: failover_retry_interval must be > 0".into());
        }
        if self.failover_retry_interval > MAX_DURATION {
            return Err(format!(
                "mic_arbitrator: failover_retry_interval {:?} exceeds the {MAX_DURATION:?} sanity cap",
                self.failover_retry_interval,
            ));
        }
        Ok(())
    }
}

/// Per-block RMS over a sample slice (tests verifying captured-signal energy);
/// f64 accumulator preserves precision for long low-amplitude blocks.
#[cfg(test)]
fn block_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
}

/// EMA alpha `1 - exp(-block_dur / window)`.  `window == 0` disables smoothing
/// (alpha = 1).
fn ema_alpha(block_dur: Duration, window: Duration) -> f32 {
    if window.is_zero() {
        return 1.0;
    }
    let ratio = block_dur.as_secs_f32() / window.as_secs_f32();
    1.0 - (-ratio).exp()
}

/// Min interval between rate-limited warnings.  A single shared `last` cell
/// covers ALL warn categories (one warn locks out the others for the interval),
/// keeping `journalctl` readable through a long USB unplug.
const WARN_INTERVAL: Duration = Duration::from_secs(30);

/// Per-effective-whitelist-slot RMS state (slot, not device-channel: whitelist
/// `[0, 2]` maps slot 0 -> channel 0, slot 1 -> channel 2).
#[derive(Clone, Copy, Debug, Default)]
struct SlotState {
    rms: f32,
}

/// Mutable run-loop state.  Scratch buffers are pre-allocated per source so the
/// hot path is alloc-free.
struct ArbitratorState {
    active: Option<ActiveSource>,
    /// Cached at `boot()`, cleared by `tear_down()` to spare per-period borrows through `state.active`.
    cached_whitelist: Vec<u16>,
    cached_n_channels: usize,
    cached_rate: u32,
    /// `active_src.period_size()`; ALSA can return fewer on hot-unplug/partial reads (hence frame-accurate alpha).
    cached_period_frames: usize,
    /// Full-period EMA alpha; short reads recompute from their frame count to avoid over-weighting the EMA.
    cached_alpha: f32,
    /// `len == cached_whitelist.len()`.
    per_slot: Vec<SlotState>,
    /// `Some` iff source rate != [`SampleRate::VALUE`].  ALL `Some` slots run
    /// each period (FIR stays current); only the active slot's output drains.
    resamplers: Vec<Option<Streaming>>,
    active_slot: Option<usize>,
    /// Drives the dwell check.
    last_switch_at: Option<Instant>,
    /// Drives the `FirstAvailable` mic-failover-after timer.
    last_data_at: Instant,
    /// `period_size * channels` floats; re-sized each `boot`, cap persists.
    interleaved_scratch: Vec<f32>,
    /// Flat per-slot demuxed scratch, stride `cached_period_frames`; flat layout avoids the `Vec<Vec<f32>>` pointer chase.
    slot_scratch_flat: Vec<f32>,
    /// Per-slot sum-of-squares (`len == per_slot.len()`); f32 precision rationale in [`single_pass_demux_and_rms`].
    sum_sq_scratch: Vec<f32>,
    /// Active slot's resampler-output drain buffer; reused per period.
    out_scratch: Vec<f32>,
}

impl ArbitratorState {
    fn new() -> Self {
        Self {
            active: None,
            cached_whitelist: Vec::new(),
            cached_n_channels: 0,
            cached_rate: 0,
            cached_period_frames: 0,
            cached_alpha: 1.0,
            per_slot: Vec::new(),
            resamplers: Vec::new(),
            active_slot: None,
            last_switch_at: None,
            last_data_at: Instant::now(),
            interleaved_scratch: Vec::new(),
            slot_scratch_flat: Vec::new(),
            sum_sq_scratch: Vec::new(),
            out_scratch: Vec::new(),
        }
    }

    fn active_id(&self) -> Option<&MicId> {
        self.active.as_ref().map(|s| s.id())
    }

    /// Wire up state for a freshly-opened source: allocate per-slot scratch +
    /// (conditionally) resamplers, cache the static-for-this-source values so the
    /// hot path reads struct fields, not `state.active`.
    fn boot(&mut self, src: ActiveSource, cfg: &MicArbitratorConfig) {
        let n_channels = src.channels() as usize;
        let period = src.period_size();
        let rate = src.rate();
        let needs_resample = rate != SampleRate::VALUE;

        self.cached_whitelist.clear();
        self.cached_whitelist
            .extend_from_slice(src.effective_whitelist());
        self.cached_n_channels = n_channels;
        self.cached_rate = rate;
        self.cached_period_frames = period;
        let nominal_block = Duration::from_secs_f64(period as f64 / rate as f64);
        self.cached_alpha = ema_alpha(nominal_block, cfg.rms_window);
        let n_slots = self.cached_whitelist.len();

        self.per_slot.clear();
        self.per_slot.resize(n_slots, SlotState::default());
        self.resamplers.clear();
        self.resamplers.extend((0..n_slots).map(|_| {
            if needs_resample {
                Some(Streaming::new(rate, SampleRate::VALUE))
            } else {
                None
            }
        }));
        // Stride is the constant `period` so per-slot offsets stay loop-invariant.
        self.slot_scratch_flat.clear();
        self.slot_scratch_flat.resize(n_slots * period, 0.0);
        self.sum_sq_scratch.clear();
        self.sum_sq_scratch.resize(n_slots, 0.0);
        self.interleaved_scratch.clear();
        self.interleaved_scratch.resize(period * n_channels, 0.0);
        self.out_scratch.clear();
        self.active_slot = None;
        self.last_switch_at = None;
        self.last_data_at = Instant::now();
        self.active = Some(src);
    }

    fn tear_down(&mut self) {
        // Scratch capacity persists for the next `boot` (no realloc churn on
        // back-to-back hot-plug); cached fields reset to uphold
        // `state.active.is_some() == cached_*-populated`.  INVARIANT: `last_data_at`
        // is deliberately NOT reset -- the failover gate reads it only while
        // `state.active.is_some()` (false here) and `boot()` refreshes it; a
        // refactor loosening that gate MUST reset it here or failover fires stale.
        self.active = None;
        self.cached_whitelist.clear();
        self.cached_n_channels = 0;
        self.cached_rate = 0;
        self.cached_period_frames = 0;
        self.cached_alpha = 1.0;
        self.per_slot.clear();
        self.resamplers.clear();
        self.active_slot = None;
        self.last_switch_at = None;
    }
}

/// Top-of-loop policy -> desired-candidate-index.  `FirstAvailable` is sticky
/// (keep the active source if still listed, else `candidates[0]`) to avoid
/// flap-tearing the working source when the operator names a new first candidate.
fn resolve_desired_idx(
    policy: &MicSelection,
    candidates: &[MicCandidate],
    current_active: Option<&MicId>,
) -> Option<usize> {
    match policy {
        MicSelection::Fixed { id } => candidates.iter().position(|c| &c.id == id),
        MicSelection::FirstAvailable => {
            if let Some(active) = current_active
                && let Some(idx) = candidates.iter().position(|c| &c.id == active)
            {
                return Some(idx);
            }
            if candidates.is_empty() { None } else { Some(0) }
        }
    }
}

/// On `FirstAvailable`, walk candidates from `start_idx` and log + boot the
/// first that opens.  No-op for `Fixed` or when nothing opens.
#[allow(clippy::too_many_arguments)]
fn try_first_available_walk(
    policy: &MicSelection,
    candidates: &[MicCandidate],
    start_idx: usize,
    stop: &Arc<AtomicBool>,
    cfg: &MicArbitratorConfig,
    state: &mut ArbitratorState,
    last_warn_at: &mut Option<Instant>,
    last_opened_id: &mut Option<MicId>,
) {
    if matches!(policy, MicSelection::FirstAvailable)
        && let Some((j, src)) =
            open_starting_from(candidates, start_idx, stop.clone(), last_warn_at)
    {
        log_active_mic_open(last_opened_id, &candidates[j].id);
        state.boot(src, cfg);
    }
}

/// Walk candidates from `start_idx`, returning the first that opens; logs each
/// failed open at warn (rate-limited).
fn open_starting_from(
    candidates: &[MicCandidate],
    start_idx: usize,
    stop: Arc<AtomicBool>,
    last_warn_at: &mut Option<Instant>,
) -> Option<(usize, ActiveSource)> {
    for (i, cand) in candidates.iter().enumerate().skip(start_idx) {
        match open_source(cand, stop.clone()) {
            Ok(src) => return Some((i, src)),
            Err(e) => rate_limited_warn(last_warn_at, "open failed", &cand.id, &e),
        }
    }
    None
}

/// Log "active mic opened" + advance the dedup baseline ONLY on an actual id
/// change: a wedged-but-openable device re-opens the same id every failover
/// cycle and would otherwise flood the journal.
fn log_active_mic_open(last_opened_id: &mut Option<MicId>, id: &MicId) {
    if last_opened_id.as_ref() != Some(id) {
        tracing::info!(
            target: "audio_io.mic_arbitrator",
            id = %id,
            "active mic opened",
        );
        *last_opened_id = Some(id.clone());
    }
}

fn rate_limited_warn(
    last: &mut Option<Instant>,
    what: &'static str,
    id: &MicId,
    reason: &dyn std::fmt::Display,
) {
    let now = Instant::now();
    let should = last.is_none_or(|t| now.duration_since(t) > WARN_INTERVAL);
    if should {
        tracing::warn!(
            target: "audio_io.mic_arbitrator",
            id = %id,
            reason = %reason,
            "{what}",
        );
        *last = Some(now);
    }
}

/// Reset per-channel resampler FIR history after a successful ALSA `try_recover`
/// (xrun/suspend): the next `readi` is a fresh capture, so skipping the reset
/// convolves new input against a phantom pre-xrun tail (~sinc_len/2).  EMA RMS +
/// active-slot pick are deliberately PRESERVED -- they survive the xrun, and
/// resetting them would drop the dwell timer and risk a channel switch piled on the xrun click.
#[cfg_attr(not(all(target_os = "linux", feature = "alsa-real")), allow(dead_code))]
fn reset_per_channel_fir(resamplers: &mut [Option<Streaming>]) {
    for r in resamplers.iter_mut().flatten() {
        r.reset_after_discontinuity();
    }
}

/// Single-pass demux + per-slot sum-of-squares: each whitelisted-channel sample
/// reads once into the `stride`-sized window of `slot_scratch_flat`;
/// `sum_sq_scratch[slot]` is overwritten (no pre-clear).  Non-finite samples
/// clamp to 0.0 so a NaN can't poison the EMA RMS forever (defensive vs buggy
/// drivers, ~free on finite real sources).  Outer-slots/inner-frames loop order
/// keeps `slot` loop-invariant (`dst` borrowed once, `sum_sq` in an f32 register
/// that is NEON-eligible where f64 is not) and the strided `interleaved` read
/// L1-resident.
#[inline]
fn single_pass_demux_and_rms(
    interleaved: &[f32],
    n_channels: usize,
    whitelist: &[u16],
    slot_scratch_flat: &mut [f32],
    sum_sq_scratch: &mut [f32],
    frames: usize,
    stride: usize,
) {
    debug_assert!(frames <= stride);
    debug_assert_eq!(slot_scratch_flat.len(), whitelist.len() * stride);
    debug_assert_eq!(sum_sq_scratch.len(), whitelist.len());
    debug_assert_eq!(interleaved.len(), frames * n_channels);
    for (slot_idx, &ch) in whitelist.iter().enumerate() {
        let offset = slot_idx * stride;
        let dst = &mut slot_scratch_flat[offset..offset + frames];
        let ch = ch as usize;
        let mut sum_sq: f32 = 0.0;
        for f in 0..frames {
            // `ch < n_channels` (enforced at source open); bounds check kept, negligible vs resample.
            let raw = interleaved[f * n_channels + ch];
            let s = if raw.is_finite() { raw } else { 0.0 };
            dst[f] = s;
            sum_sq += s * s;
        }
        sum_sq_scratch[slot_idx] = sum_sq;
    }
}

/// EMA alpha for this read: full-period reads reuse the cached alpha; short
/// reads recompute from their frame count to keep the time constant wall-accurate.
#[inline]
fn alpha_for_frames(
    frames: usize,
    nominal_period_frames: usize,
    rate: u32,
    cached_nominal_alpha: f32,
    window: Duration,
) -> f32 {
    debug_assert!(frames > 0, "process_period only handles non-zero reads");
    debug_assert!(nominal_period_frames > 0, "source period must be non-zero");
    debug_assert!(rate > 0, "source rate must be non-zero");
    if frames == nominal_period_frames {
        cached_nominal_alpha
    } else {
        let actual_block = Duration::from_secs_f64(frames as f64 / rate as f64);
        ema_alpha(actual_block, window)
    }
}

/// Choose the active slot from per-slot RMS, channel policy, hysteresis, and
/// dwell.  Pure for unit-testability (hence the argument count).
#[allow(clippy::too_many_arguments)]
fn pick_slot(
    policy: &ChannelSelection,
    per_slot: &[SlotState],
    whitelist: &[u16],
    current_slot: Option<usize>,
    last_switched_at: Option<Instant>,
    now: Instant,
    hysteresis_linear: f32,
    dwell: Duration,
) -> Option<usize> {
    if per_slot.is_empty() {
        return None;
    }
    if let ChannelSelection::Fixed { channel } = policy
        && let Some(idx) = whitelist.iter().position(|&w| w == *channel)
    {
        return Some(idx);
    }
    // Auto, OR Fixed-but-channel-not-in-whitelist: fall back to Auto, not silence.
    let (loudest_idx, loudest_rms) = per_slot
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.rms))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty above");

    match current_slot {
        None => Some(loudest_idx),
        Some(active_idx) if active_idx == loudest_idx => Some(active_idx),
        Some(active_idx) => {
            let active_rms = per_slot[active_idx].rms;
            let above_threshold = loudest_rms > active_rms * hysteresis_linear;
            let dwell_satisfied = match last_switched_at {
                None => true,
                Some(t) => now.saturating_duration_since(t) >= dwell,
            };
            if above_threshold && dwell_satisfied {
                Some(loudest_idx)
            } else {
                Some(active_idx)
            }
        }
    }
}

/// Process one successfully-read period (non-zero frame count): demux + RMS +
/// pick slot + resample if needed + write the active slot to the audio buffer.
fn process_period(
    state: &mut ArbitratorState,
    writer: &mut Writer,
    frames: usize,
    channel_policy: &ChannelSelection,
    cfg: &MicArbitratorConfig,
    hysteresis_linear: f32,
) {
    // `state.active` is NOT borrowed here: an outer `&state.active` borrow would
    // block the interleaving mutable scratch-field borrows below.
    let n_channels = state.cached_n_channels;
    let n_slots = state.cached_whitelist.len();
    let stride = state.cached_period_frames;
    let interleaved = &state.interleaved_scratch[..frames * n_channels];

    single_pass_demux_and_rms(
        interleaved,
        n_channels,
        &state.cached_whitelist,
        &mut state.slot_scratch_flat,
        &mut state.sum_sq_scratch,
        frames,
        stride,
    );

    let alpha = alpha_for_frames(
        frames,
        state.cached_period_frames,
        state.cached_rate,
        state.cached_alpha,
        cfg.rms_window,
    );
    for slot in 0..n_slots {
        // Sum-of-squares can overflow to +inf; clamp non-finite before the EMA so a glitch can't poison the slot.
        let mut block_rms = (state.sum_sq_scratch[slot] / frames as f32).sqrt();
        if !block_rms.is_finite() {
            block_rms = 0.0;
        }
        state.per_slot[slot].rms = alpha * block_rms + (1.0 - alpha) * state.per_slot[slot].rms;
    }

    let now = Instant::now();
    let prev_slot = state.active_slot;
    let new_slot = pick_slot(
        channel_policy,
        &state.per_slot,
        &state.cached_whitelist,
        prev_slot,
        state.last_switch_at,
        now,
        hysteresis_linear,
        cfg.dwell,
    );
    if new_slot != prev_slot {
        state.last_switch_at = Some(now);
        if let (Some(p), Some(n)) = (prev_slot, new_slot) {
            tracing::info!(
                target: "audio_io.mic_arbitrator",
                from_slot = p,
                from_channel = state.cached_whitelist[p],
                to_slot = n,
                to_channel = state.cached_whitelist[n],
                "channel switched",
            );
        } else if let Some(n) = new_slot {
            tracing::info!(
                target: "audio_io.mic_arbitrator",
                to_slot = n,
                to_channel = state.cached_whitelist[n],
                "channel selected (initial)",
            );
        }
    }
    state.active_slot = new_slot;

    let Some(active_idx) = state.active_slot else {
        return; // unreachable for a non-empty whitelist
    };

    let needs_resample = state.resamplers[active_idx].is_some();
    if !needs_resample {
        // Snapshot `captured_at` BEFORE push so it reflects sample-ready time,
        // not the push's memcpy/atomic-store (forward drift on contended cores).
        let active_offset = active_idx * stride;
        let captured_at = crate::common::time::CaptureTime::now();
        writer.push(&state.slot_scratch_flat[active_offset..active_offset + frames]);
        publish_timing_anchor(writer, cfg, captured_at);
    } else {
        // Auto (slot may change period-to-period): feed every resampler so all FIR
        // states stay current and a switch is glitch-free across the group delay.
        // Truly-Fixed locks `active_idx`, so skip the others but RESET them:
        // `Streaming`'s partial-chunk `accum` (when `period_size != chunk_size`)
        // would glue pre-skip audio onto the first chunk if the policy flips Fixed->Auto.
        let truly_fixed = match channel_policy {
            ChannelSelection::Fixed { channel } => state.cached_whitelist.contains(channel),
            ChannelSelection::Auto => false,
        };
        for slot in 0..n_slots {
            if truly_fixed && slot != active_idx {
                let r = state.resamplers[slot].as_mut().expect("alloc'd in boot");
                r.reset_after_discontinuity();
                continue;
            }
            let r = state.resamplers[slot].as_mut().expect("alloc'd in boot");
            let slot_offset = slot * stride;
            // expect() panics into the catch_unwind-then-abort wrapper in `start`.
            r.process(&state.slot_scratch_flat[slot_offset..slot_offset + frames])
                .expect("Streaming::process invariant break -- see StreamingResampleError docs");
            if slot == active_idx {
                state.out_scratch.clear();
                r.drain_output_into(&mut state.out_scratch);
                if !state.out_scratch.is_empty() {
                    // Snapshot BEFORE the chunk loop (as in the native-rate branch).
                    let captured_at = crate::common::time::CaptureTime::now();
                    // Chunk by `max_push_len()` to honour the writer's safety margin under a future ring resize.
                    let max_push = writer.max_push_len();
                    for chunk in state.out_scratch.chunks(max_push) {
                        writer.push(chunk);
                    }
                    // Publish ONCE per drain (last write wins); no per-chunk Arc.
                    publish_timing_anchor(writer, cfg, captured_at);
                }
            } else {
                r.drop_output();
            }
        }
    }
}

/// Publish a [`crate::common::time::BufferTimingAnchor`] when a shared anchor
/// cell is configured (else no-op).  `head_pos` loaded AFTER the push (Acquire),
/// `captured_at` sampled BEFORE it, pinning the post-push ring against the
/// pre-push clock so consumers projecting `captured_at - (head_pos - sample_idx)
/// / rate` get the tightest per-sample bound.  `sample_rate_hz` is canonical
/// (audio is already resampled here).
#[inline]
fn publish_timing_anchor(
    writer: &Writer,
    cfg: &MicArbitratorConfig,
    captured_at: crate::common::time::CaptureTime,
) {
    let Some(cell) = cfg.timing_anchor.as_ref() else {
        return;
    };
    let anchor = crate::common::time::BufferTimingAnchor {
        head_pos: writer.head_pos(),
        captured_at,
        sample_rate_hz: SampleRate::VALUE,
    };
    cell.store(std::sync::Arc::new(anchor));
}

/// Running mic arbitrator thread.  Drop or call [`MicArbitrator::stop`] to
/// terminate.  Dropping the [`crate::audio_buffer::AudioBuffer`] handle after
/// starting is safe; [`Writer`] keeps the ring alive.
#[derive(Debug)]
pub struct MicArbitrator {
    handle: Option<thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl MicArbitrator {
    /// Spin up the arbitrator thread.  `settings` is read via `snapshot()` once
    /// per loop, so an API mutation becomes visible within ~one period.  Panics if
    /// `cfg` fails [`MicArbitratorConfig::validate`] -- the single validation gate
    /// no call site can bypass.
    pub fn start(
        writer: Writer,
        settings: Arc<dyn MicSettingsStore>,
        cfg: MicArbitratorConfig,
    ) -> Self {
        cfg.validate().unwrap_or_else(|e| {
            panic!("MicArbitrator::start: invalid MicArbitratorConfig: {e}");
        });
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();
        // `run_loop` consumes `cfg`; copy sched fields out before the move.
        let sched_pin = cfg.sched_pin;
        let sched_priority = cfg.sched_priority;
        let handle = thread::Builder::new()
            .name("mic-arbitrator".into())
            .spawn(move || {
                // Pin BEFORE set_realtime so placement is deterministic (an
                // unpinned RT thread can be kernel-rebalanced).  Both best-effort.
                if let Some(core) = sched_pin
                    && let Err(e) = crate::sched::pin_to_core(core)
                {
                    tracing::warn!(
                        target: "audio_io",
                        err = %e,
                        core = core,
                        "mic-arbitrator pin_to_core failed; continuing on default placement",
                    );
                }
                if let Some(prio) = sched_priority
                    && let Err(e) = crate::sched::set_realtime(prio)
                {
                    tracing::warn!(
                        target: "audio_io",
                        err = %e,
                        priority = prio,
                        "mic-arbitrator set_realtime failed (likely missing CAP_SYS_NICE); \
                         continuing at SCHED_OTHER",
                    );
                }
                // Fatal-thread policy: catch panic, log, abort for supervisor
                // restart.  This thread produces all downstream audio timing, so
                // without abort the buffer head freezes and stale inference frames
                // flow past the silent failure.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_loop(writer, settings, cfg, stop_clone)
                }));
                if result.is_err() {
                    tracing::error!(
                        target: "audio_io",
                        "mic-arbitrator panicked; aborting process so operator can restart",
                    );
                    // Non-blocking tracing appender may buffer the abort log; mirror to stderr + brief flush.
                    eprintln!(
                        "acousticslabd: ABORT -- mic-arbitrator panicked; \
                         supervisor must restart"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    std::process::abort();
                }
            })
            .unwrap_or_else(|e| {
                // Boot-time rlimit exhaustion or similar: abort with a structured
                // diagnostic, matching the in-thread panic policy.
                tracing::error!(
                    target: "audio_io",
                    err = %e,
                    "MicArbitrator::start: failed to spawn mic-arbitrator thread; \
                     aborting process (rlimit exhaustion or similar resource pressure)",
                );
                eprintln!(
                    "acousticslabd: ABORT -- mic-arbitrator thread spawn failed \
                     (err={e}); supervisor must restart"
                );
                std::thread::sleep(std::time::Duration::from_millis(50));
                std::process::abort();
            });
        Self {
            handle: Some(handle),
            stop,
        }
    }

    /// Signal stop without waiting.  Idempotent.  Used in shutdown to silence
    /// the producer BEFORE draining consumers: the loop exits within ~one capture
    /// period, so consumers drain into a quiet pipeline.
    pub fn signal_stop(&self) {
        signal_only(&self.stop, self.handle.as_ref());
    }

    /// Signal stop and join.  Idempotent with `signal_stop`: callers may
    /// `signal_stop()` early and `stop()` later to join.
    pub fn stop(mut self) {
        signal_and_join(&self.stop, &mut self.handle);
    }
}

impl Drop for MicArbitrator {
    fn drop(&mut self) {
        // Signal + unpark but DO NOT synchronously join: a Drop-time join would
        // hang runtime drop (wedging a tokio worker on the unwind path) if the
        // audio thread is stuck on a wedged ALSA syscall.  The thread sees the
        // Release-store of `stop` within one capture period and exits; the OS
        // reclaims the detached thread at process exit.  Drop is the unwind
        // fallback to the bounded join in `stop(self)`, idempotent via `as_ref()`
        // -> `None` once `stop()` took `self.handle`.
        signal_only(&self.stop, self.handle.as_ref());
    }
}

/// Release-store the stop flag and wake any park; does not join.
fn signal_only(stop: &AtomicBool, handle: Option<&thread::JoinHandle<()>>) {
    stop.store(true, Ordering::Release);
    if let Some(h) = handle {
        h.thread().unpark();
    }
}

/// Signal stop, wake any park, and join.  The `unpark` keeps shutdown prompt
/// when the loop is in the no-source `park_timeout` branch (else teardown blocks
/// up to `failover_retry_interval`).
fn signal_and_join(stop: &AtomicBool, handle: &mut Option<thread::JoinHandle<()>>) {
    stop.store(true, Ordering::Release);
    if let Some(h) = handle.take() {
        h.thread().unpark();
        let _ = h.join();
    }
}

fn run_loop(
    mut writer: Writer,
    settings: Arc<dyn MicSettingsStore>,
    cfg: MicArbitratorConfig,
    stop: Arc<AtomicBool>,
) {
    let hysteresis_linear = cfg.hysteresis_linear();
    let mut state = ArbitratorState::new();
    let mut last_warn_at: Option<Instant> = None;
    let mut last_opened_id: Option<MicId> = None;

    while !stop.load(Ordering::Acquire) {
        let snap = settings.snapshot();

        let desired_idx = resolve_desired_idx(
            &snap.policy.mic,
            &snap.catalogue.candidates,
            state.active_id(),
        );

        let desired_id = desired_idx.map(|d| &snap.catalogue.candidates[d].id);
        let need_switch = state.active_id() != desired_id;

        if need_switch {
            state.tear_down();
            if let Some(idx) = desired_idx {
                let cand = &snap.catalogue.candidates[idx];
                match open_source(cand, stop.clone()) {
                    Ok(src) => {
                        log_active_mic_open(&mut last_opened_id, &cand.id);
                        state.boot(src, &cfg);
                    }
                    Err(OpenError::AlsaNotCompiledIn(id)) => {
                        rate_limited_warn(
                            &mut last_warn_at,
                            "alsa-real not compiled in",
                            &id,
                            &"feature not supported in this build",
                        );
                        try_first_available_walk(
                            &snap.policy.mic,
                            &snap.catalogue.candidates,
                            idx + 1,
                            &stop,
                            &cfg,
                            &mut state,
                            &mut last_warn_at,
                            &mut last_opened_id,
                        );
                    }
                    Err(e) => {
                        rate_limited_warn(&mut last_warn_at, "open failed", e.mic_id(), &e);
                        try_first_available_walk(
                            &snap.policy.mic,
                            &snap.catalogue.candidates,
                            idx + 1,
                            &stop,
                            &cfg,
                            &mut state,
                            &mut last_warn_at,
                            &mut last_opened_id,
                        );
                    }
                }
            }
        }

        // FirstAvailable mic-level failover: source silent past threshold ->
        // tear down and re-resolve next iteration.  Fixed ignores this.
        if state.active.is_some()
            && matches!(&snap.policy.mic, MicSelection::FirstAvailable)
            && Instant::now().saturating_duration_since(state.last_data_at) > cfg.mic_failover_after
        {
            // Bind id BEFORE `tear_down` clears `state.active`.  Shared throttle cell
            // so a wedged-but-openable device (never advancing `last_data_at`) doesn't warn each cycle.
            if let Some(id) = state.active_id() {
                rate_limited_warn(
                    &mut last_warn_at,
                    "no fresh data within mic_failover_after; failing over",
                    id,
                    &"check the source",
                );
            }
            state.tear_down();
            // No sleep -- next iter opens another candidate immediately.
            continue;
        }

        // No source open: park and retry.
        let Some(active_src) = state.active.as_mut() else {
            // A `Fixed { id }` missing from the catalogue resolves to `None`, opens
            // nothing, and parks here every iteration -- silent without this warn
            // (other routes here log their own above).
            if desired_idx.is_none()
                && let MicSelection::Fixed { id } = &snap.policy.mic
            {
                rate_limited_warn(
                    &mut last_warn_at,
                    "fixed mic id not in catalogue; staying inert",
                    id,
                    &"add the candidate or change policy",
                );
            }
            // `park_timeout` (not `sleep`) so `stop()`/`Drop` `unpark` for prompt
            // teardown instead of blocking up to `failover_retry_interval`.  The
            // unpark token persists, so a `stop()` racing the park still wakes us;
            // spurious wakeups just round-trip the loop top.
            thread::park_timeout(cfg.failover_retry_interval);
            continue;
        };

        // ALSA recovery stays a concrete `ActiveSource` arm (mock has no
        // analogue); the bounded poll timeout lives in `read_interleaved`.
        use crate::audio_io::source::{MicSource as _, ReadError, ReadOutcome};
        let frames_read = match active_src.read_interleaved(&mut state.interleaved_scratch) {
            Ok(ReadOutcome::Frames(n)) => n.get(),
            Ok(ReadOutcome::Timeout) | Ok(ReadOutcome::StopRequested) => {
                // Top-of-loop re-checks `stop`, else re-reads the same source.
                continue;
            }
            Ok(ReadOutcome::EndOfStream) => {
                // ALSA `Frames(0)`: tear down rather than spin on a dead PCM.
                tracing::warn!(
                    target: "audio_io.mic_arbitrator",
                    id = %active_src.id(),
                    "source returned EndOfStream; tearing down",
                );
                state.tear_down();
                continue;
            }
            Err(read_err) => match read_err {
                #[cfg(all(target_os = "linux", feature = "alsa-real"))]
                ReadError::Alsa(e) => {
                    let id = active_src.id().clone();
                    tracing::warn!(
                        target: "audio_io.mic_arbitrator",
                        id = %id,
                        err = %e,
                        "ALSA read error; attempting recovery",
                    );
                    if let ActiveSource::Alsa(a) = active_src {
                        if a.try_recover(e).is_err() {
                            state.tear_down();
                        } else {
                            reset_per_channel_fir(&mut state.resamplers);
                        }
                    }
                    continue;
                }
                ReadError::Mock(infallible) => match infallible {},
            },
        };

        state.last_data_at = Instant::now();

        process_period(
            &mut state,
            &mut writer,
            frames_read,
            &snap.policy.channel,
            &cfg,
            hysteresis_linear,
        );
    }

    state.tear_down();
}
