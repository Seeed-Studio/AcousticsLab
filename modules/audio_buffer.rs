//! Single-writer / multi-reader f32 sample ring buffer: copy-out reads with a
//! seqlock-style recheck guarding against the writer overwriting cells mid-read.
//! Unique [`Writer`] (`Send + !Sync + !Clone`); per-task [`Reader`]s with own
//! `tail` (`Send + !Sync`); [`AudioBuffer`] is `Send + Sync + Clone`.
//!
//! Memory ordering: writer does per-cell `store(Relaxed)` then `head.store(Release)`
//! (plain store not `fetch_add` -- single-writer makes the RMW superfluous);
//! reader does `head.load(Acquire)` -> per-cell `load(Relaxed)` -> recheck head.
//! `head`'s Release/Acquire pair is the happens-before edge; `AtomicU32` cells
//! never tear, so the recheck guards only mid-read overwrite.
//!
//! Soundness rests on two invariants: (1) single-writer, enforced by the
//! `writer_taken` swap/clear (concurrent writers would interleave cell stores
//! into gibberish); (2) safety margin -- peek requires `head - tail <= cap*3/4`
//! and the post-copy recheck reports `Lagged` (discarding `out`) at `avail >
//! cmm`, sound as long as the writer cannot publish `>= cap/4` samples between
//! precheck and recheck (holds under production `SCHED_FIFO` capture + the
//! bounded `push`; under `SCHED_OTHER` a torn `out` is still caught and
//! `must_use` forces the discard). No internal retry (re-peeking without moving
//! tail re-lags), so the caller must `seek_latest` to recover. Readers polling
//! on `Wait` pay up to one sleep interval of lag.

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

use std::cell::UnsafeCell;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// 64 B-aligned wrapper isolating the contended `head` from immutable `Inner`
/// fields so the writer's per-push store doesn't invalidate readers' cached
/// metadata (partial isolation on 128 B-line CPUs).
#[repr(align(64))]
struct CacheLineAligned<T>(T);

impl<T> Deref for CacheLineAligned<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

/// Outcome of a [`Reader::peek_into`] call.
#[must_use = "ignoring ReadStatus risks reading uninitialized samples on Wait/Lagged"]
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum ReadStatus {
    /// `out` was fully written; tail is unchanged.
    Ready,
    /// Not enough new data yet (`head - tail < out.len()`); `out` unchanged.
    Wait,
    /// Reader fell more than `capacity * 3/4` behind head; `out` may be
    /// clobbered. Caller should [`Reader::seek_latest`] to resync.
    Lagged {
        /// `head - tail` in SAMPLES. Unbounded (the writer never throttles for a
        /// stalled reader, so `by` can far exceed `capacity`) -- consumers MUST
        /// use saturating arithmetic; only the inversion fallback clamps to
        /// `cap` as a sentinel.
        by: u64,
    },
}

/// Sample ring + atomic coordination state, shared via `Arc`. All fields atomic
/// or immutable-after-construction (auto-`Sync`); writer<->reader happens-before
/// edge is `head`'s Release/Acquire pair.
struct Inner {
    /// Ring of `capacity` cells, each an f32 bit pattern
    /// (`to_bits`/`from_bits`); `AtomicU32` layout matches `Box<[f32]>`.
    data: Box<[AtomicU32]>,
    capacity: usize,
    /// `capacity - capacity/4`; "safe to read" is `head - tail <= this`.
    capacity_minus_margin: usize,
    /// `capacity/4`: max samples one [`Writer::push`] may publish without
    /// breaking the seqlock recheck.
    safety_margin: usize,
    /// True iff a [`Writer`] currently exists for this buffer.
    writer_taken: AtomicBool,
    /// Total samples ever written; monotonic (2^64/44100 ~= 13 Myr). The only
    /// contended atomic, cache-line isolated via [`CacheLineAligned`].
    head: CacheLineAligned<AtomicU64>,
}

impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Inner")
            .field("capacity", &self.capacity)
            .field("capacity_minus_margin", &self.capacity_minus_margin)
            .field("head", &self.head.load(Ordering::Relaxed))
            .field("writer_taken", &self.writer_taken.load(Ordering::Relaxed))
            .finish()
    }
}

/// Handle to a sample ring buffer; cheap to clone (Arc bump).
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    inner: Arc<Inner>,
}

impl AudioBuffer {
    /// Create a fresh buffer with `capacity` sample slots.
    ///
    /// `capacity` must be a power of two (wrap is `head & (capacity-1)`) and >= 4
    /// (seqlock needs `cap/4 >= 1` to distinguish "not lapped" from "just
    /// lapped"). Size for `>= 2x` the largest peek window (canonical daemon
    /// 262 144 = 2^18, ~5.94 s at 44.1 kHz); capacity governs stall-recovery
    /// margin, not latency.
    ///
    /// # Panics
    /// If `capacity` is not a power of two or `< 4`.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity >= 4 && capacity.is_power_of_two(),
            "AudioBuffer capacity must be a power of two and >= 4 (got {capacity}); \
             round up with `usize::next_power_of_two()`",
        );
        let data: Box<[AtomicU32]> = (0..capacity).map(|_| AtomicU32::new(0)).collect();
        let safety_margin = capacity / 4;
        let capacity_minus_margin = capacity - safety_margin;
        Self {
            inner: Arc::new(Inner {
                data,
                capacity,
                capacity_minus_margin,
                safety_margin,
                writer_taken: AtomicBool::new(false),
                head: CacheLineAligned(AtomicU64::new(0)),
            }),
        }
    }

    /// Total ring capacity in samples.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Largest peek window that can be served (= `capacity - capacity/4`).
    /// [`Reader::peek_into`] panics (every build) if `out.len()` exceeds it.
    #[inline]
    pub fn safe_peek_window(&self) -> usize {
        self.inner.capacity_minus_margin
    }

    /// Maximum samples a single [`Writer::push`] may publish (= `capacity/4`,
    /// the seqlock safety margin); larger batches must be split.
    ///
    /// Cross-module invariant: capture-side
    /// [`crate::audio_io::mic_arbitrator::MAX_PERIOD_FRAMES`] (8 192) must stay
    /// below this margin; shrinking the canonical capacity requires updating
    /// that constant in lockstep or `push` panics.
    #[inline]
    pub fn max_push_len(&self) -> usize {
        self.inner.safety_margin
    }

    /// Current head (samples written so far); for tests/instrumentation.
    #[inline]
    pub fn head(&self) -> u64 {
        self.inner.head.load(Ordering::Acquire)
    }

    /// Acquire the unique [`Writer`]; the flag is cleared on Writer drop.
    ///
    /// # Panics
    /// If a Writer for this buffer already exists.
    pub fn take_writer(&self) -> Writer {
        // AcqRel pairs with `Writer::drop`'s Release: Acquire sees a prior
        // Drop's cleared flag, Release publishes `true`.
        if self.inner.writer_taken.swap(true, Ordering::AcqRel) {
            panic!("a Writer already exists for this AudioBuffer");
        }
        Writer {
            inner: Arc::clone(&self.inner),
            _not_sync: PhantomData,
        }
    }

    /// Spawn a reader at the live edge (`tail = head`); `peek_into` returns
    /// `Wait` until new samples arrive. For readers needing no history.
    pub fn reader(&self) -> Reader {
        let tail = self.inner.head.load(Ordering::Acquire);
        self.reader_with_tail(tail)
    }

    /// Spawn a reader `behind_head` samples behind the live edge
    /// (`tail = head - behind_head`, saturating), for a small startup backlog.
    pub fn reader_at(&self, behind_head: usize) -> Reader {
        let head = self.inner.head.load(Ordering::Acquire);
        self.reader_with_tail(head.saturating_sub(behind_head as u64))
    }

    fn reader_with_tail(&self, tail: u64) -> Reader {
        Reader {
            inner: Arc::clone(&self.inner),
            tail,
            _not_sync: PhantomData,
        }
    }
}

/// Unique sample writer (`Send + !Sync + !Clone`); pushes at head with
/// `Release` ordering so readers see a consistent slice after Acquire.
pub struct Writer {
    inner: Arc<Inner>,
    /// `!Sync` marker enforcing single-writer (no cross-thread sharing).
    _not_sync: PhantomData<UnsafeCell<()>>,
}

impl fmt::Debug for Writer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Writer")
            .field("head", &self.inner.head.load(Ordering::Relaxed))
            .field("capacity", &self.inner.capacity)
            .finish()
    }
}

impl Writer {
    /// Same as [`AudioBuffer::max_push_len`], on the [`Writer`] so producers can
    /// chunk without the [`AudioBuffer`] handle.
    #[inline]
    pub fn max_push_len(&self) -> usize {
        self.inner.safety_margin
    }

    /// Current absolute head position (`Acquire`-load synchronizing-with the
    /// latest `push`); used by [`crate::common::time::BufferTimingAnchor`] to
    /// anchor head to capture time.
    #[inline]
    pub fn head_pos(&self) -> u64 {
        self.inner.head.load(Ordering::Acquire)
    }

    /// Append `samples`; wraps silently, overwriting cells outside
    /// `[head - capacity, head)`.
    ///
    /// # Invariant
    /// `samples.len() <= capacity/4` ([`AudioBuffer::max_push_len`]): a larger
    /// push overwrites a reader's currently-safe window before the recheck sees
    /// head pass `capacity_minus_margin`, tearing the read.
    ///
    /// # Panics
    /// If `samples.len() > capacity/4` -- hard assert in every build.
    #[inline]
    pub fn push(&mut self, samples: &[f32]) {
        let n = samples.len();
        if n == 0 {
            return;
        }
        let cap = self.inner.capacity;
        let max_push = self.inner.safety_margin;
        assert!(
            n <= max_push,
            "samples.len()={n} exceeds max_push_len={max_push} (= {cap}/4); split the batch",
        );
        // Relaxed: only the writer modifies head; the Release store below
        // publishes visibility.
        let head = self.inner.head.load(Ordering::Relaxed);
        let start = (head & (cap as u64 - 1)) as usize;
        // `n <= cap` keeps the head + wrap slices in range and disjoint.
        let first = (cap - start).min(n);
        let (head_part, wrap_part) = samples.split_at(first);
        let data = &self.inner.data;
        for (cell, &s) in data[start..start + first].iter().zip(head_part) {
            cell.store(s.to_bits(), Ordering::Relaxed);
        }
        for (cell, &s) in data[..wrap_part.len()].iter().zip(wrap_part) {
            cell.store(s.to_bits(), Ordering::Relaxed);
        }
        // Release publishes the prior Relaxed cell stores; reader's Acquire-load
        // on `head` synchronizes-with this.
        self.inner
            .head
            .store(head.wrapping_add(n as u64), Ordering::Release);
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.inner.writer_taken.store(false, Ordering::Release);
    }
}

/// Sample reader (`Send + !Sync`); owns its `tail`, copies out into a
/// caller-provided buffer (never lends slices into the ring).
pub struct Reader {
    inner: Arc<Inner>,
    tail: u64,
    _not_sync: PhantomData<UnsafeCell<()>>,
}

impl fmt::Debug for Reader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reader")
            .field("tail", &self.tail)
            .field("head", &self.inner.head.load(Ordering::Relaxed))
            .field("capacity", &self.inner.capacity)
            .finish()
    }
}

impl Reader {
    /// Copy `out.len()` of the oldest unconsumed samples into `out` without
    /// advancing tail. `Ready` on success; `Wait` if short; `Lagged { by }` if
    /// the writer is `> capacity*3/4` ahead.
    ///
    /// Single-iteration seqlock (load head -> bounds-check -> per-cell Relaxed
    /// loads -> reload head): a mid-read overwrite trips the recheck to `Lagged`
    /// and `out` must be discarded (caller `seek_latest`s). No retry loop --
    /// head is monotonic, so a second pass would re-lag immediately.
    #[must_use = "the read may have failed (Wait/Lagged); inspect the status before consuming `out`"]
    #[inline]
    pub fn peek_into(&self, out: &mut [f32]) -> ReadStatus {
        let n = out.len();
        if n == 0 {
            return ReadStatus::Ready;
        }
        // Hard-assert (not debug_assert): else a release build stalls in `Wait`
        // forever on an oversized peek -- with `n > capacity_minus_margin` a
        // non-lagged reader has `avail <= cmm < n`, so `avail < n_u64` is always
        // true and the success path (`avail >= n`) is never reached.
        assert!(
            n <= self.inner.capacity_minus_margin,
            "out.len()={n} > safe_peek_window={} (capacity {})",
            self.inner.capacity_minus_margin,
            self.inner.capacity,
        );

        let cap = self.inner.capacity;
        let mask = (cap - 1) as u64;
        let cmm = self.inner.capacity_minus_margin as u64;
        let n_u64 = n as u64;

        let h0 = self.inner.head.load(Ordering::Acquire);
        // `tail > head` is unreachable (every tail-mutating path upholds `tail
        // <= head`); detected explicitly rather than masked as a permanent `Wait`.
        debug_assert!(
            self.tail <= h0,
            "Reader::peek_into: tail ({}) past head ({}) -- invariant violation",
            self.tail,
            h0,
        );
        if self.tail > h0 {
            // `Lagged` not `Wait` so the caller's `seek_latest` re-pins tail out
            // of the inversion; `by = cap` (not `u64::MAX`) keeps the `by /
            // hop_samples` diagnostic sane (recovery ignores `by`).
            return ReadStatus::Lagged { by: cap as u64 };
        }
        let avail = h0 - self.tail;
        if avail < n_u64 {
            return ReadStatus::Wait;
        }
        if avail > cmm {
            return ReadStatus::Lagged { by: avail };
        }

        // avail <= cmm = cap - cap/4, so [tail, tail+n) lies in the writer's
        // valid past [h0-cap, h0); overwriting it needs >= cap/4 more pushes,
        // caught as Lagged by the recheck below.
        let start = (self.tail & mask) as usize;
        let first = (cap - start).min(n);
        let (out_head, out_wrap) = out.split_at_mut(first);
        let data = &self.inner.data;
        for (slot, cell) in out_head.iter_mut().zip(&data[start..start + first]) {
            *slot = f32::from_bits(cell.load(Ordering::Relaxed));
        }
        let wrap_len = out_wrap.len();
        for (slot, cell) in out_wrap.iter_mut().zip(&data[..wrap_len]) {
            *slot = f32::from_bits(cell.load(Ordering::Relaxed));
        }

        // Full fence (not compiler_fence/Acquire-load): aarch64 LDAR forbids
        // hoisting later loads above it but lets prior Relaxed cell loads sink
        // past it; without this load-load barrier the cell loads could observe
        // writer state newer than `h1` and a lap slip past `avail1 <= cmm`.
        std::sync::atomic::fence(Ordering::Acquire);

        // Recheck for mid-read overwrite. `h1 >= h0 >= tail` (head monotonic) so
        // plain `-` can't underflow; plain over `saturating_sub` so a broken
        // monotonicity invariant trips the debug overflow check rather than
        // misreporting Ready on a tear.
        let h1 = self.inner.head.load(Ordering::Acquire);
        debug_assert!(h1 >= self.tail, "head retreated below tail");
        let avail1 = h1 - self.tail;
        if avail1 <= cmm {
            ReadStatus::Ready
        } else {
            ReadStatus::Lagged { by: avail1 }
        }
    }

    /// Advance tail by `n` samples; hard-asserts `tail <= head` in every build
    /// (over-advance panics, not a silent permanent `Wait`). Head is monotonic,
    /// so `tail <= head_now` implies the at-entry invariant held despite loading
    /// after `saturating_add`.
    #[inline]
    pub fn advance(&mut self, n: usize) {
        self.tail = self.tail.saturating_add(n as u64);
        let head = self.inner.head.load(Ordering::Acquire);
        assert!(
            self.tail <= head,
            "advance({n}): tail={} exceeds head={head}",
            self.tail,
        );
    }

    /// Resync tail to `head - n` (saturating) to drop backlog after `Lagged`.
    #[inline]
    pub fn seek_latest(&mut self, n: usize) {
        let head = self.inner.head.load(Ordering::Acquire);
        self.tail = head.saturating_sub(n as u64);
    }

    /// Current tail (samples consumed so far on this reader).
    #[inline]
    pub fn tail(&self) -> u64 {
        self.tail
    }

    /// Readable sample count `head - tail` (Acquire-synced lower bound). Lets a
    /// consumer poll before [`Reader::advance`] when `hop_samples > WaveformLen`
    /// (advancing past what peek guarantees trips `advance`'s assert);
    /// `saturating_sub` covers the `reader_at` case of tail ahead of head.
    #[inline]
    pub fn available(&self) -> u64 {
        let head = self.inner.head.load(Ordering::Acquire);
        head.saturating_sub(self.tail)
    }

    /// Lag threshold for consumers polling [`Self::available`] without the
    /// [`Self::peek_into`] recheck: `available() > safe_peek_window()` means the
    /// writer overran the margin and the reader must `seek_latest`. Same value
    /// as [`AudioBuffer::safe_peek_window`].
    #[inline]
    pub fn safe_peek_window(&self) -> usize {
        self.inner.capacity_minus_margin
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    #[should_panic(expected = "capacity must be a power of two and >= 4")]
    fn new_zero_capacity_panics() {
        let _ = AudioBuffer::new(0);
    }

    #[test]
    #[should_panic(expected = "capacity must be a power of two and >= 4")]
    fn new_capacity_below_minimum_panics() {
        let _ = AudioBuffer::new(3);
    }

    #[test]
    #[should_panic(expected = "capacity must be a power of two and >= 4")]
    fn new_non_power_of_two_capacity_panics() {
        let _ = AudioBuffer::new(100);
    }

    /// `head` must sit on a 64 B boundary in a full 64 B line, else the writer's
    /// per-push store invalidates readers' cached metadata.
    #[test]
    fn head_is_isolated_on_its_own_cache_line() {
        use std::mem;
        assert_eq!(mem::size_of::<CacheLineAligned<AtomicU64>>(), 64);
        assert_eq!(mem::align_of::<CacheLineAligned<AtomicU64>>(), 64);
        assert_eq!(mem::offset_of!(Inner, head) % 64, 0);
        let head_off = mem::offset_of!(Inner, head);
        let head_line_end = head_off + 64;
        for (name, off) in [
            ("data", mem::offset_of!(Inner, data)),
            ("capacity", mem::offset_of!(Inner, capacity)),
            (
                "capacity_minus_margin",
                mem::offset_of!(Inner, capacity_minus_margin),
            ),
            ("safety_margin", mem::offset_of!(Inner, safety_margin)),
            ("writer_taken", mem::offset_of!(Inner, writer_taken)),
        ] {
            assert!(
                off < head_off || off >= head_line_end,
                "field {name} at offset {off} overlaps head's cache line [{head_off}, {head_line_end})",
            );
        }
    }

    #[test]
    fn cap_4_smallest_valid_capacity_works() {
        let buf = AudioBuffer::new(4);
        assert_eq!(buf.safe_peek_window(), 3);
        assert_eq!(buf.max_push_len(), 1);
        let mut w = buf.take_writer();
        let mut r = buf.reader();
        for s in [1.0_f32, 2.0, 3.0] {
            w.push(&[s]);
        }
        let mut out = [0.0; 3];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [1.0, 2.0, 3.0]);
        r.advance(3);
        for s in [4.0_f32, 5.0] {
            w.push(&[s]); // wraps: cells 3, 0
        }
        let mut out2 = [0.0; 2];
        assert_eq!(r.peek_into(&mut out2), ReadStatus::Ready);
        assert_eq!(out2, [4.0, 5.0]);
    }

    /// Peek of exactly `safe_peek_window` (seqlock boundary) is Ready, not Lagged.
    #[test]
    fn peek_at_exactly_cmm_is_ready() {
        let buf = AudioBuffer::new(16);
        assert_eq!(buf.max_push_len(), 4);
        let mut w = buf.take_writer();
        let r = buf.reader();
        for batch_start in (0..12).step_by(4) {
            let batch: Vec<f32> = (batch_start..batch_start + 4).map(|i| i as f32).collect();
            w.push(&batch);
        }
        let mut out = [0.0; 12];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i as f32);
        }
    }

    #[test]
    #[should_panic(expected = "exceeds max_push_len")]
    fn push_larger_than_capacity_panics() {
        let buf = AudioBuffer::new(8);
        let mut w = buf.take_writer();
        // Prime head off zero so push's wrap-arithmetic is exercised.
        w.push(&[0.0; 2]);
        let big: Vec<f32> = (0..20).map(|i| i as f32).collect();
        w.push(&big);
    }

    /// `safety_margin + 1` panics: strict seqlock bound, not off-by-one tolerant.
    #[test]
    #[should_panic(expected = "exceeds max_push_len")]
    fn push_exceeds_safety_margin_panics() {
        let buf = AudioBuffer::new(64);
        assert_eq!(buf.max_push_len(), 16);
        let mut w = buf.take_writer();
        let just_over: Vec<f32> = (0..17).map(|i| i as f32).collect();
        w.push(&just_over);
    }

    /// Push of exactly `safety_margin` is legal (pins the `<=` bound).
    #[test]
    fn push_at_exactly_safety_margin_succeeds() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let r = buf.reader();
        let exact: Vec<f32> = (0..16).map(|i| i as f32).collect();
        w.push(&exact);
        let mut out = [0.0; 16];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, i as f32);
        }
    }

    #[test]
    #[should_panic(expected = "Writer already exists")]
    fn take_writer_twice_panics() {
        let buf = AudioBuffer::new(64);
        let _w1 = buf.take_writer();
        let _w2 = buf.take_writer();
    }

    #[test]
    fn take_writer_after_drop_ok() {
        let buf = AudioBuffer::new(64);
        {
            let _w1 = buf.take_writer();
        }
        let _w2 = buf.take_writer();
    }

    #[test]
    fn reader_starts_at_live_edge() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        w.push(&[1.0, 2.0, 3.0, 4.0]);
        let r = buf.reader();
        let mut out = [0.0; 2];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Wait);
        w.push(&[5.0, 6.0]);
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [5.0, 6.0]);
    }

    #[test]
    fn reader_at_with_backlog_returns_ready_immediately() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        w.push(&[1.0, 2.0, 3.0, 4.0]);
        let r = buf.reader_at(2);
        let mut out = [0.0; 2];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [3.0, 4.0]);
    }

    #[test]
    fn simple_push_and_peek_round_trip() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let r = buf.reader();
        w.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut out = [0.0; 5];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn peek_does_not_advance() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let r = buf.reader();
        w.push(&[10.0, 20.0, 30.0]);
        let mut out = [0.0; 3];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [10.0, 20.0, 30.0]);
        out.fill(0.0);
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn available_tracks_head_minus_tail() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let mut r = buf.reader();
        assert_eq!(r.available(), 0);
        w.push(&[1.0, 2.0, 3.0]);
        assert_eq!(r.available(), 3);
        let mut out = [0.0; 3];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(r.available(), 3);
        r.advance(2);
        assert_eq!(r.available(), 1);
        w.push(&[4.0, 5.0]);
        assert_eq!(r.available(), 3);
    }

    #[test]
    fn advance_consumes_samples() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let mut r = buf.reader();
        w.push(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut out = [0.0; 2];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [10.0, 20.0]);
        r.advance(2);
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [30.0, 40.0]);
        r.advance(2); // only [50] remains
        let mut last = [0.0; 2];
        assert_eq!(r.peek_into(&mut last), ReadStatus::Wait);
        let mut one = [0.0; 1];
        assert_eq!(r.peek_into(&mut one), ReadStatus::Ready);
        assert_eq!(one, [50.0]);
    }

    #[test]
    fn peek_wait_when_short() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let r = buf.reader();
        w.push(&[1.0, 2.0]);
        let mut out = [0.0; 5];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Wait);
        assert_eq!(out, [0.0; 5]);
    }

    #[test]
    fn peek_lagged_when_writer_far_ahead() {
        let buf = AudioBuffer::new(16);
        let mut w = buf.take_writer();
        let r = buf.reader();
        // head reaches 13 > cmm=12 -> Lagged.
        for batch_start in (0..12).step_by(4) {
            let batch: Vec<f32> = (batch_start..batch_start + 4).map(|i| i as f32).collect();
            w.push(&batch);
        }
        w.push(&[12.0]);
        let mut out = [0.0; 4];
        match r.peek_into(&mut out) {
            ReadStatus::Lagged { by: 13 } => {}
            other => panic!("expected Lagged{{by:13}}, got {other:?}"),
        }
    }

    #[test]
    fn seek_latest_resets_tail_near_head() {
        let buf = AudioBuffer::new(16);
        let mut w = buf.take_writer();
        let mut r = buf.reader();
        for batch_start in (0..12).step_by(4) {
            let batch: Vec<f32> = (batch_start..batch_start + 4).map(|i| i as f32).collect();
            w.push(&batch);
        }
        w.push(&[12.0]);
        let mut out = [0.0; 4];
        assert!(matches!(r.peek_into(&mut out), ReadStatus::Lagged { .. }));
        r.seek_latest(4);
        assert_eq!(r.tail(), 9); // head=13, behind=4 -> tail=9
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out, [9.0, 10.0, 11.0, 12.0]);
    }

    /// Wrap-around correctness across the bitwise-mod boundary.
    #[test]
    fn wrap_around_correctness() {
        let cap = 16;
        let buf = AudioBuffer::new(cap);
        let mut w = buf.take_writer();
        let mut r = buf.reader();
        let first: Vec<f32> = (0..8).map(|i| i as f32).collect();
        for chunk in first.chunks(4) {
            w.push(chunk);
        }
        let mut out = [0.0; 8];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
        assert_eq!(out.to_vec(), first);
        r.advance(8);

        // head 8 -> 18 wraps (cap=16, cells 0..2 overwritten); tail=8 stays in
        // valid range [head-cap, head) = [2, 18).
        let second: Vec<f32> = (100..110).map(|i| i as f32).collect();
        for chunk in second.chunks(4) {
            w.push(chunk);
        }
        let mut out2 = [0.0; 10];
        assert_eq!(r.peek_into(&mut out2), ReadStatus::Ready);
        assert_eq!(
            out2.to_vec(),
            second,
            "reader should see exactly the second batch starting at tail=8",
        );
    }

    /// Two readers progress at independent rates against one writer.
    #[test]
    fn multi_reader_independent_tails() {
        let buf = AudioBuffer::new(64);
        let mut w = buf.take_writer();
        let mut r1 = buf.reader();
        let r2 = buf.reader();
        w.push(&[1.0, 2.0, 3.0, 4.0]);
        let mut o1 = [0.0; 2];
        let mut o2 = [0.0; 4];
        assert_eq!(r1.peek_into(&mut o1), ReadStatus::Ready);
        assert_eq!(o1, [1.0, 2.0]);
        r1.advance(2);
        assert_eq!(r2.peek_into(&mut o2), ReadStatus::Ready);
        assert_eq!(o2, [1.0, 2.0, 3.0, 4.0]); // r2 still at tail=0
        let mut o1b = [0.0; 2];
        assert_eq!(r1.peek_into(&mut o1b), ReadStatus::Ready);
        assert_eq!(o1b, [3.0, 4.0]); // r1 at tail=2
    }

    /// Concurrent writer+reader: pins the memory-ordering edge (fails on torn
    /// read or wrong Acquire/Release). Capacity > TOTAL so it never laps,
    /// isolating ordering from scheduling.
    #[test]
    fn concurrent_writer_reader_ordering() {
        const TOTAL: u64 = 50_000;
        const CHUNK: usize = 256;
        let buf = AudioBuffer::new(131_072);
        let mut r = buf.reader_at(0);

        let buf_w = buf.clone();
        let writer_thread = thread::spawn(move || {
            let mut w = buf_w.take_writer();
            let mut next: u64 = 0;
            while next < TOTAL {
                let upto = (next + CHUNK as u64).min(TOTAL);
                let batch: Vec<f32> = (next..upto).map(|i| i as f32).collect();
                w.push(&batch);
                next = upto;
            }
        });

        let reader_thread = thread::spawn(move || {
            let mut got = 0u64;
            let mut chunk = vec![0f32; CHUNK];
            while got < TOTAL {
                match r.peek_into(&mut chunk) {
                    ReadStatus::Ready => {
                        for (i, &v) in chunk.iter().enumerate() {
                            assert_eq!(
                                v,
                                (got + i as u64) as f32,
                                "drift at sample {} (got={got})",
                                got + i as u64,
                            );
                        }
                        r.advance(CHUNK);
                        got += CHUNK as u64;
                    }
                    ReadStatus::Wait => {
                        let remaining = TOTAL - got;
                        if remaining < CHUNK as u64 && remaining > 0 {
                            let mut tail_chunk = vec![0f32; remaining as usize];
                            if r.peek_into(&mut tail_chunk) == ReadStatus::Ready {
                                for (i, &v) in tail_chunk.iter().enumerate() {
                                    assert_eq!(v, (got + i as u64) as f32);
                                }
                                r.advance(remaining as usize);
                                got += remaining;
                                break;
                            }
                        }
                        thread::sleep(Duration::from_micros(10));
                    }
                    ReadStatus::Lagged { by } => {
                        panic!(
                            "capacity > TOTAL guarantees no Lag -- got by={by}, got={got}; \
                             this is a code bug, not a flake",
                        );
                    }
                }
            }
            got
        });

        writer_thread.join().expect("writer thread");
        let got = reader_thread.join().expect("reader thread");
        assert_eq!(got, TOTAL);
    }

    /// Forces wrap-around + transient Lag, recovers via `seek_latest`; every
    /// observed sample must match the index-as-f32 pattern (Lag-dropped samples
    /// allowed by design).
    #[test]
    fn concurrent_wrap_around_with_lag_recovery() {
        const TOTAL: u64 = 200_000;
        const CHUNK: usize = 256;
        // Small capacity -> forces dozens of wraps + plenty of Lag.
        let buf = AudioBuffer::new(2_048);
        let mut r = buf.reader_at(0);
        let writer_done = Arc::new(AtomicBool::new(false));

        let buf_w = buf.clone();
        let writer_done_w = writer_done.clone();
        let writer_thread = thread::spawn(move || {
            let mut w = buf_w.take_writer();
            let mut next: u64 = 0;
            while next < TOTAL {
                let upto = (next + CHUNK as u64).min(TOTAL);
                let batch: Vec<f32> = (next..upto).map(|i| i as f32).collect();
                w.push(&batch);
                next = upto;
            }
            writer_done_w.store(true, Ordering::Release);
        });

        let writer_done_r = writer_done.clone();
        let reader_thread = thread::spawn(move || {
            let mut chunk = vec![0f32; CHUNK];
            let mut samples_in_pattern = 0u64;
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            loop {
                if std::time::Instant::now() > deadline {
                    panic!("timeout: tail={}", r.tail());
                }
                match r.peek_into(&mut chunk) {
                    ReadStatus::Ready => {
                        let base = r.tail();
                        for (i, &v) in chunk.iter().enumerate() {
                            assert_eq!(
                                v,
                                (base + i as u64) as f32,
                                "pattern drift at sample {}",
                                base + i as u64,
                            );
                        }
                        r.advance(CHUNK);
                        samples_in_pattern += CHUNK as u64;
                    }
                    ReadStatus::Wait => {
                        if writer_done_r.load(Ordering::Acquire) {
                            break;
                        }
                        thread::sleep(Duration::from_micros(20));
                    }
                    ReadStatus::Lagged { by: _ } => {
                        r.seek_latest(CHUNK);
                    }
                }
            }
            samples_in_pattern
        });

        writer_thread.join().expect("writer thread");
        let observed = reader_thread.join().expect("reader thread");
        // Lag expected, so assert only that the pattern check ran, not observed == TOTAL.
        assert!(
            observed > 0,
            "reader observed nothing -- pattern check never ran",
        );
    }

    #[test]
    fn safe_peek_window_matches_capacity_three_quarters() {
        let buf = AudioBuffer::new(128);
        assert_eq!(buf.safe_peek_window(), 96); // 128 - 128/4
    }

    #[test]
    fn peek_into_empty_buffer_is_ready_no_op() {
        let buf = AudioBuffer::new(16);
        let r = buf.reader();
        let mut out: [f32; 0] = [];
        assert_eq!(r.peek_into(&mut out), ReadStatus::Ready);
    }

    #[test]
    fn auto_trait_assertions() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        fn assert_send_static<T: Send + 'static>() {}

        assert_send::<AudioBuffer>();
        assert_sync::<AudioBuffer>();
        assert_send_static::<AudioBuffer>();
        assert_send_static::<Writer>();
        assert_send_static::<Reader>();
    }

    /// One writer + two slow readers, both force-lagged: each must advance its
    /// tail (recovery re-arms) AND see at least one `Lagged` (recovery path ran),
    /// and Ready windows must be ascending (no torn reads) -- the contract
    /// inference and opus_stream share.
    #[test]
    fn concurrent_two_readers_resync_under_writer_pressure() {
        let buf = AudioBuffer::new(1024);
        // Pin reader tails at 0 BEFORE the writer starts: constructing them in
        // the threads risks the writer finishing all pushes first, leaving both
        // readers at head so `lags > 0` spuriously fails.
        let r1 = buf.reader_at(0);
        let r2 = buf.reader_at(0);
        let buf_w = buf.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_r1 = stop.clone();
        let stop_r2 = stop.clone();

        let writer = std::thread::spawn(move || {
            let mut w = buf_w.take_writer();
            let mut counter: f32 = 0.0;
            for _ in 0..1_000 {
                let chunk: Vec<f32> = (0..200).map(|i| counter + i as f32).collect();
                counter += 200.0;
                w.push(&chunk);
            }
        });

        let r1_handle = std::thread::spawn(move || {
            let mut r = r1;
            let mut window = [0.0_f32; 64];
            let mut total_advanced = 0_u64;
            let mut lag_count = 0_u64;
            let mut max_seen: f32 = -1.0;
            while !stop_r1.load(Ordering::Relaxed) {
                match r.peek_into(&mut window) {
                    ReadStatus::Ready => {
                        for w_pair in window.windows(2) {
                            assert!(
                                w_pair[1] > w_pair[0],
                                "torn read: {:?} not ascending",
                                w_pair
                            );
                        }
                        if window[0] >= max_seen {
                            max_seen = window[63];
                        }
                        r.advance(32);
                        total_advanced += 32;
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    ReadStatus::Wait => std::thread::sleep(Duration::from_micros(50)),
                    ReadStatus::Lagged { .. } => {
                        r.seek_latest(64);
                        lag_count += 1;
                    }
                }
            }
            (total_advanced, lag_count, max_seen)
        });

        let r2_handle = std::thread::spawn(move || {
            let mut r = r2;
            let mut window = [0.0_f32; 32];
            let mut total_advanced = 0_u64;
            let mut lag_count = 0_u64;
            let mut max_seen: f32 = -1.0;
            while !stop_r2.load(Ordering::Relaxed) {
                match r.peek_into(&mut window) {
                    ReadStatus::Ready => {
                        for w_pair in window.windows(2) {
                            assert!(w_pair[1] > w_pair[0], "torn read on reader 2: {w_pair:?}");
                        }
                        if window[0] >= max_seen {
                            max_seen = window[31];
                        }
                        r.advance(16);
                        total_advanced += 16;
                    }
                    ReadStatus::Wait => std::thread::sleep(Duration::from_micros(20)),
                    ReadStatus::Lagged { .. } => {
                        r.seek_latest(32);
                        lag_count += 1;
                    }
                }
            }
            (total_advanced, lag_count, max_seen)
        });

        writer.join().expect("writer panicked");
        // Grace for readers to drain the tail.
        std::thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Relaxed);
        let (r1_adv, r1_lags, r1_max) = r1_handle.join().expect("r1 panicked");
        let (r2_adv, r2_lags, r2_max) = r2_handle.join().expect("r2 panicked");

        assert!(r1_adv > 0, "reader 1 never advanced");
        assert!(r2_adv > 0, "reader 2 never advanced");
        assert!(
            r1_lags > 0 || r2_lags > 0,
            "neither reader observed Lagged -- test didn't exercise the contract"
        );
        assert!(r1_max < 200_000.0, "r1 saw out-of-range value {r1_max}");
        assert!(r2_max < 200_000.0, "r2 saw out-of-range value {r2_max}");
    }

    /// `advance` past head panics in every build (debug_assert would silently
    /// cap and stall forever in `Wait`).
    #[test]
    fn reader_advance_past_head_panics() {
        let buf = AudioBuffer::new(64);
        let mut writer = buf.take_writer();
        writer.push(&[0.0, 0.1, 0.2, 0.3]); // head = 4
        let mut r = buf.reader_at(0);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            r.advance(100); // must panic, not saturate
        }));
        assert!(
            result.is_err(),
            "advance past head must panic; got Ok which means saturating_add silently capped",
        );
    }

    /// `peek_into` with `out.len() > safe_peek_window` panics in every build
    /// (debug_assert would silently stall in `Wait`).
    #[test]
    fn reader_peek_into_oversize_panics() {
        let buf = AudioBuffer::new(64);
        let _writer = buf.take_writer();
        let r = buf.reader_at(0);
        // safe_peek_window for cap=64 is 48; 100 is well past it.
        let mut out = vec![0.0f32; 100];
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.peek_into(&mut out)));
        assert!(
            result.is_err(),
            "peek_into past safe_peek_window must panic; got Ok which means \
             release silently accepted the oversized peek",
        );
    }
}
