//! Loom memory-ordering model of the `audio_buffer` seqlock, re-implemented
//! with `loom::sync::atomic` (running Loom on production code would pollute
//! release builds with its atomic shim). Asserts `peek_into`'s invariants over
//! all interleavings: on `Ready` every sample is a writer-published bit
//! pattern forming the contiguous prefix from tail (no torn read); `Lagged` is
//! a true-positive guard whose data is never consumed.
//!
//! Targets aarch64: production stores cells `Relaxed`, publishes `head`
//! `Release`, and fences `Acquire` between cell loads and the recheck head
//! load. x86's strong ordering makes the fence a near-no-op, but aarch64's
//! one-way `LDAR` lets prior `Relaxed` loads sink past it absent the
//! `dmb ishld` the fence emits; Loom enumerates those sinking orders.
//!
//! Run: `RUSTFLAGS="--cfg audio_buffer_loom" cargo test --test
//! audio_buffer_loom --release`. cfg is `audio_buffer_loom` not bare `loom`
//! because dev-dep tokio-tungstenite pulls tokio `net` (`#![cfg(not(loom))]`)
//! -- `--cfg loom` would break the dev-dep graph compile.

#![cfg(audio_buffer_loom)]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};
use loom::thread;

/// Mirrors `audio_buffer::Inner` minus cache-line padding (perf-only).
struct Ring {
    cells: Box<[AtomicU32]>,
    head: AtomicU64,
}

impl Ring {
    /// Smallest legal cap; larger only multiplies Loom paths without adding
    /// distinct seqlock-edge interleavings.
    const CAP: usize = 4;
    const MASK: u64 = (Self::CAP - 1) as u64;
    /// Reader's safe-to-read upper bound on `head - tail` (`capacity_minus_margin`).
    const CMM: u64 = (Self::CAP - Self::CAP / 4) as u64;

    fn new() -> Self {
        let cells = (0..Self::CAP).map(|_| AtomicU32::new(0)).collect();
        Self {
            cells,
            head: AtomicU64::new(0),
        }
    }

    /// Mirrors `Writer::push`: Relaxed cell store, then Release head store.
    fn push_one(&self, sample: f32) {
        let head = self.head.load(Ordering::Relaxed);
        let idx = (head & Self::MASK) as usize;
        self.cells[idx].store(sample.to_bits(), Ordering::Relaxed);
        self.head.store(head.wrapping_add(1), Ordering::Release);
    }
}

/// Mirrors `Reader::peek_into`'s outcome enum (values only).
#[derive(Debug, Eq, PartialEq)]
enum ReadStatus {
    Ready,
    Wait,
    Lagged,
}

/// Mirrors `Reader::peek_into` for `out.len() == n`; on `Ready` writes `out`.
fn peek(ring: &Ring, tail: u64, out: &mut [f32]) -> ReadStatus {
    let n = out.len() as u64;
    let h0 = ring.head.load(Ordering::Acquire);
    let avail = h0.saturating_sub(tail);
    if avail < n {
        return ReadStatus::Wait;
    }
    if avail > Ring::CMM {
        return ReadStatus::Lagged;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let idx = ((tail.wrapping_add(i as u64)) & Ring::MASK) as usize;
        *slot = f32::from_bits(ring.cells[idx].load(Ordering::Relaxed));
    }
    // Load-load barrier pinning the Relaxed cell loads above the recheck on
    // aarch64; Loom honours it like `dmb ishld`.
    fence(Ordering::Acquire);
    let h1 = ring.head.load(Ordering::Acquire);
    let avail1 = h1.saturating_sub(tail);
    if avail1 <= Ring::CMM {
        ReadStatus::Ready
    } else {
        ReadStatus::Lagged
    }
}

/// One-sample peek at tail=0 against a 3-push writer (smallest model of cell
/// Relaxed visibility vs the recheck under writer progress). A stale 0-bit
/// cell reaching `Ready` would prove the fence/recheck unsound.
#[test]
fn loom_seqlock_no_torn_read_one_sample() {
    loom::model(|| {
        let ring = Arc::new(Ring::new());
        let writer_ring = ring.clone();
        let writer = thread::spawn(move || {
            // Nonzero sentinels: 0.0 (cell init) marks an uninitialized read.
            for k in 1..=3u32 {
                writer_ring.push_one(k as f32);
            }
        });

        let reader_ring = ring;
        let reader = thread::spawn(move || {
            let mut out = [0.0_f32];
            let status = peek(&reader_ring, 0, &mut out);
            match status {
                ReadStatus::Ready => {
                    let v = out[0];
                    assert!(
                        v == 1.0 || v == 2.0 || v == 3.0,
                        "torn read: peek returned {v}, expected one of \
                         the sentinel samples 1.0/2.0/3.0",
                    );
                    // h1 saw >=1 push, so the Release/Acquire edge must make
                    // cell[0]=1.0 visible; else torn.
                    assert_eq!(
                        v, 1.0,
                        "Ready returned cell[0]={v} but tail=0; the head \
                         Release-store must publish cell[0]=1.0 first",
                    );
                }
                ReadStatus::Wait | ReadStatus::Lagged => {
                    // Only Wait is reachable (avail < 1 at h0); Lagged needs
                    // avail > 3, impossible since 3 pushes cap avail at 3 = cmm
                    // so the recheck always returns Ready.
                }
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
    });
}

/// Two-sample peek at tail=0 against a 2-push writer: catches observing
/// cell[1] before cell[0]'s Release-publish is visible, which the recheck must
/// convert into `Lagged`/`Wait` rather than a torn `Ready`.
#[test]
fn loom_seqlock_no_torn_read_two_samples() {
    loom::model(|| {
        let ring = Arc::new(Ring::new());
        let writer_ring = ring.clone();
        let writer = thread::spawn(move || {
            writer_ring.push_one(11.0);
            writer_ring.push_one(22.0);
        });

        let reader_ring = ring;
        let reader = thread::spawn(move || {
            let mut out = [0.0_f32; 2];
            let status = peek(&reader_ring, 0, &mut out);
            match status {
                ReadStatus::Ready => {
                    // Out-of-order (cell[1]==22, cell[0]==0) = second store seen
                    // before first; Release/Acquire + recheck must make it
                    // Lagged/Wait, never Ready.
                    assert_eq!(
                        out,
                        [11.0, 22.0],
                        "torn read on 2-sample peek: got {out:?}, expected \
                         [11.0, 22.0]",
                    );
                }
                ReadStatus::Wait | ReadStatus::Lagged => {
                    // Wait legal (avail < 2 at h0); Lagged needs h1-tail > 3,
                    // impossible with 2 pushes -- so it would be a bug.
                    assert_ne!(
                        status,
                        ReadStatus::Lagged,
                        "Lagged is unreachable with 2 pushes and cmm=3; \
                         got Lagged anyway -- check the recheck arithmetic",
                    );
                }
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();
    });
}
