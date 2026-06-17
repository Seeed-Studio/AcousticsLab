//! Two clock-domain newtypes forcing consumers to pick monotonic-vs-wallclock
//! explicitly: [`CaptureTime`] (µs since boot, intra-process) and [`WallTime`]
//! (µs since Unix epoch, jumps on NTP, cross-host), mapping to proto
//! `t_us_capture_monotonic` / `t_us_publish_unix`. µs (not ns) keeps `u64`
//! good for >580k years; intervals go through [`std::time::Duration`].

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;

/// Microseconds since process boot (monotonic); intra-process only.
#[derive(
    Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct CaptureTime(u64);

impl CaptureTime {
    /// Sample the monotonic clock, anchored to the first call's `Instant`, so
    /// skew from true boot is the `main()`-to-first-`now()` latency (<100 ms).
    pub fn now() -> Self {
        Self(elapsed_us_since_boot_anchor())
    }

    #[inline]
    pub const fn from_micros(us: u64) -> Self {
        Self(us)
    }

    #[inline]
    pub const fn as_micros(self) -> u64 {
        self.0
    }

    /// Difference as a [`Duration`], or `None` if `self < other`.
    pub const fn since(self, other: Self) -> Option<Duration> {
        if self.0 < other.0 {
            None
        } else {
            Some(Duration::from_micros(self.0 - other.0))
        }
    }
}

/// Microseconds since the Unix epoch (wall-clock); jumps on NTP; cross-host.
#[derive(
    Copy, Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct WallTime(u64);

impl WallTime {
    /// Sample the wall-clock; `None` only if it predates the Unix epoch. First
    /// `None` per process logs `warn!` (later `debug!`) so `unwrap_or(default)`
    /// fallbacks don't silently stamp the 1970 sentinel.
    pub fn now() -> Option<Self> {
        let result = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|d| u64::try_from(d.as_micros()).ok())
            .map(Self);
        if result.is_none() {
            use std::sync::atomic::{AtomicBool, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::AcqRel) {
                tracing::warn!(
                    target: "common::time",
                    "WallTime::now() returned None (clock set before Unix epoch); callers \
                     will see the default fallback",
                );
            } else {
                tracing::debug!(
                    target: "common::time",
                    "WallTime::now() returned None (warning already emitted this process)",
                );
            }
        }
        result
    }

    #[inline]
    pub const fn from_micros(us: u64) -> Self {
        Self(us)
    }

    #[inline]
    pub const fn as_micros(self) -> u64 {
        self.0
    }
}

/// Producer *capture* time (not engine encode/publish) snapshotted at an
/// absolute ring head position, refreshed after each `Writer::push` so a
/// consumer's sample cursor converts via [`capture_us_for`] in one hop.
/// `captured_at` is the producer's pre-push (sample-ready) instant, snapshotted
/// before the ring write to avoid push-memcpy forward drift, shared by all
/// samples in that push and interpolated linearly at `sample_rate_hz`; mic-swap
/// boundaries
/// leave an uncorrected residual across the overlap. `Copy` so consumers deref
/// the `ArcSwap` once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BufferTimingAnchor {
    /// Same u64 sample-counter space as `AudioBuffer::head` / `Reader::tail`.
    pub head_pos: u64,
    pub captured_at: CaptureTime,
    /// Per-anchor so a future variable-rate producer needs no second lookup.
    pub sample_rate_hz: u32,
}

impl BufferTimingAnchor {
    /// Self-consistent linear projection before the producer's first push
    /// (which overwrites it), not uninitialised state.
    pub const fn boot_placeholder() -> Self {
        Self {
            head_pos: 0,
            captured_at: CaptureTime::from_micros(0),
            sample_rate_hz: 44_100,
        }
    }
}

/// Wait-free cell: producer publishes a fresh anchor lock-free; consumers load
/// it in their hot loop.
pub type SharedTimingAnchor = Arc<ArcSwap<BufferTimingAnchor>>;

/// Fresh [`SharedTimingAnchor`] seeded with [`BufferTimingAnchor::boot_placeholder`].
pub fn shared_timing_anchor() -> SharedTimingAnchor {
    Arc::new(ArcSwap::from_pointee(BufferTimingAnchor::boot_placeholder()))
}

/// Project absolute `read_pos` (signed-relative to `anchor.head_pos`) to
/// capture monotonic time via the latest anchor; underflow saturates at 0
/// ("older than process boot"), correct for pre-anchor samples.
pub fn capture_us_for(anchor: BufferTimingAnchor, read_pos: u64) -> u64 {
    let head_pos = anchor.head_pos as i128;
    let read = read_pos as i128;
    let sr = anchor.sample_rate_hz as i128;
    if sr == 0 {
        return anchor.captured_at.as_micros(); // malformed anchor: avoid div-by-zero
    }
    let delta_samples = read - head_pos;
    // i128 keeps the product in range at extreme sample positions.
    let delta_us = delta_samples * 1_000_000 / sr;
    let projected = anchor.captured_at.as_micros() as i128 + delta_us;
    // Saturate both ends rather than wrap through a naked `as u64`.
    u64::try_from(projected.max(0)).unwrap_or(u64::MAX)
}

fn elapsed_us_since_boot_anchor() -> u64 {
    use std::sync::OnceLock;
    static ANCHOR: OnceLock<Instant> = OnceLock::new();
    let anchor = ANCHOR.get_or_init(Instant::now);
    u64::try_from(anchor.elapsed().as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn capture_time_now_is_monotonic() {
        let a = CaptureTime::now();
        thread::sleep(Duration::from_micros(100));
        let b = CaptureTime::now();
        assert!(b.as_micros() >= a.as_micros());
    }

    #[test]
    fn capture_time_since_returns_duration() {
        let a = CaptureTime::from_micros(1_000);
        let b = CaptureTime::from_micros(3_500);
        assert_eq!(b.since(a), Some(Duration::from_micros(2_500)));
        // Reversed direction is None, not a saturating sub.
        assert_eq!(a.since(b), None);
    }

    #[test]
    fn wall_time_now_post_dates_2000() {
        let now = WallTime::now().expect("clock past unix epoch");
        const Y2000_US: u64 = 946_684_800_000_000;
        assert!(now.as_micros() > Y2000_US);
    }

    #[test]
    fn from_micros_round_trips() {
        let c = CaptureTime::from_micros(42);
        let w = WallTime::from_micros(42);
        assert_eq!(c.as_micros(), 42);
        assert_eq!(w.as_micros(), 42);
    }

    /// No shared `From`/`Into`: the wire format must explicitly pick one when
    /// emitting a timestamp.
    #[test]
    fn types_are_distinct_at_compile_time() {
        fn _accepts_capture(_: CaptureTime) {}
        fn _accepts_wall(_: WallTime) {}
        // _accepts_capture(WallTime::from_micros(0));   // <-- compile error
        // _accepts_wall(CaptureTime::from_micros(0));   // <-- compile error
    }

    #[test]
    fn capture_us_for_at_anchor_head_returns_captured_at() {
        let anchor = BufferTimingAnchor {
            head_pos: 44_100,
            captured_at: CaptureTime::from_micros(1_000_000),
            sample_rate_hz: 44_100,
        };
        assert_eq!(capture_us_for(anchor, 44_100), 1_000_000);
    }

    #[test]
    fn capture_us_for_behind_anchor_interpolates_back() {
        let anchor = BufferTimingAnchor {
            head_pos: 88_200,
            captured_at: CaptureTime::from_micros(2_000_000),
            sample_rate_hz: 44_100,
        };
        assert_eq!(capture_us_for(anchor, 44_100), 1_000_000); // 1 s back
        assert_eq!(capture_us_for(anchor, 66_150), 1_500_000); // 0.5 s back
    }

    #[test]
    fn capture_us_for_ahead_of_anchor_extrapolates_forward() {
        let anchor = BufferTimingAnchor {
            head_pos: 44_100,
            captured_at: CaptureTime::from_micros(1_000_000),
            sample_rate_hz: 44_100,
        };
        assert_eq!(capture_us_for(anchor, 66_150), 1_500_000); // 0.5 s ahead
    }

    #[test]
    fn capture_us_for_saturates_on_underflow() {
        let anchor = BufferTimingAnchor {
            head_pos: 44_100,
            captured_at: CaptureTime::from_micros(500_000),
            sample_rate_hz: 44_100,
        };
        assert_eq!(capture_us_for(anchor, 0), 0);
    }

    #[test]
    fn capture_us_for_respects_anchor_sample_rate() {
        let half = BufferTimingAnchor {
            head_pos: 22_050,
            captured_at: CaptureTime::from_micros(1_000_000),
            sample_rate_hz: 22_050,
        };
        assert_eq!(capture_us_for(half, 0), 0); // 22_050 samples back at 22.05 kHz = 1 s

        let high = BufferTimingAnchor {
            head_pos: 48_000,
            captured_at: CaptureTime::from_micros(2_000_000),
            sample_rate_hz: 48_000,
        };
        assert_eq!(capture_us_for(high, 24_000), 1_500_000); // 24_000 back at 48 kHz = 0.5 s
    }

    #[test]
    fn capture_us_for_zero_sample_rate_returns_captured_at() {
        let bad = BufferTimingAnchor {
            head_pos: 100,
            captured_at: CaptureTime::from_micros(7_777_777),
            sample_rate_hz: 0,
        };
        assert_eq!(capture_us_for(bad, 0), 7_777_777);
        assert_eq!(capture_us_for(bad, 1_000_000), 7_777_777);
    }

    #[test]
    fn boot_placeholder_is_self_consistent() {
        let p = BufferTimingAnchor::boot_placeholder();
        assert_eq!(p.head_pos, 0);
        assert_eq!(p.captured_at.as_micros(), 0);
        assert_eq!(p.sample_rate_hz, 44_100);
        assert_eq!(capture_us_for(p, 0), 0);
        assert!(capture_us_for(p, 22_050) > 0);
    }

    #[test]
    fn shared_timing_anchor_round_trips() {
        let cell = shared_timing_anchor();
        let initial = **cell.load();
        assert_eq!(initial, BufferTimingAnchor::boot_placeholder());

        let fresh = BufferTimingAnchor {
            head_pos: 12_345,
            captured_at: CaptureTime::from_micros(67_890),
            sample_rate_hz: 48_000,
        };
        cell.store(Arc::new(fresh));
        let observed = **cell.load();
        assert_eq!(observed, fresh);
    }
}
