//! Per-thread CPU affinity + SCHED_FIFO realtime priority helpers. Lives outside
//! [`crate::common`] (`#![forbid(unsafe_code)]`) because the syscall wrappers
//! need `unsafe`. Non-Linux gets no-op shims, but [`set_realtime`] still
//! validates priority so dev-host config tests catch bad settings.
//!
//! Failures are non-fatal (call sites log and continue, never `?`): the
//! daemon MUST stay up. Dominant Linux failure is `EPERM` from [`set_realtime`]
//! without `CAP_SYS_NICE` (grant via `setcap cap_sys_nice+ep`); without it the
//! daemon runs `SCHED_OTHER`, with occasional ALSA underruns under load.

#![warn(missing_debug_implementations)]

use std::io;

/// Reject priorities outside `SCHED_FIFO` range `1..=99` or above the daemon
/// [`MAX_DAEMON_REALTIME_PRIORITY`] cap. Runs before cfg dispatch so shim and
/// real-syscall paths return identical `InvalidInput` errors.
fn validate_realtime_priority(priority: i32) -> io::Result<()> {
    if !(1..=99).contains(&priority) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("realtime priority {priority} out of SCHED_FIFO range 1..=99"),
        ));
    }
    if priority > MAX_DAEMON_REALTIME_PRIORITY {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "realtime priority {priority} exceeds the daemon-side cap \
                 {MAX_DAEMON_REALTIME_PRIORITY} (reserved for kernel + operator chains)"
            ),
        ));
    }
    Ok(())
}

/// Upper bound on daemon-requested `SCHED_FIFO` priorities, capped below the
/// kernel max of 99 to leave headroom for kernel housekeeping; daemon uses
/// audio source = 50, inference = 30.
pub const MAX_DAEMON_REALTIME_PRIORITY: i32 = 60;

/// Pin the calling thread to a single CPU core (Linux only; no-op elsewhere).
///
/// # Errors
///
/// `InvalidInput` when `core >= libc::CPU_SETSIZE`; else the syscall errno
/// (`EINVAL` on a single-core host requesting `core > 0`, cgroup/cpuset reject).
pub fn pin_to_core(core: usize) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        linux::pin_to_core(core)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = core;
        Ok(())
    }
}

/// Switch the calling thread to `SCHED_FIFO` with `priority` (Linux only; no-op
/// after validation elsewhere). Valid range `1..=`[`MAX_DAEMON_REALTIME_PRIORITY`].
///
/// # Errors
///
/// `InvalidInput` on every platform (before any syscall) when out of range; on
/// Linux `EPERM` without `CAP_SYS_NICE`, or other syscall errno.
pub fn set_realtime(priority: i32) -> io::Result<()> {
    validate_realtime_priority(priority)?;
    #[cfg(target_os = "linux")]
    {
        linux::set_realtime(priority)
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::io;
    use std::mem::MaybeUninit;

    pub(super) fn pin_to_core(core: usize) -> io::Result<()> {
        // `CPU_SET` has no bounds check; reject out-of-range cores first.
        if core >= libc::CPU_SETSIZE as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "core {core} out of cpu_set_t range (max {})",
                    libc::CPU_SETSIZE - 1
                ),
            ));
        }
        // SAFETY: `CPU_ZERO` writes every byte of `set` before any read; `&mut`
        // is exclusive and the storage outlives the call.
        let mut set = MaybeUninit::<libc::cpu_set_t>::uninit();
        unsafe {
            libc::CPU_ZERO(&mut *set.as_mut_ptr());
            libc::CPU_SET(core, &mut *set.as_mut_ptr());
        }
        // SAFETY: `set` fully initialised above; `pid=0` is the calling thread;
        // kernel reads `cpusetsize == size_of::<cpu_set_t>()` bytes.
        let rc = unsafe {
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), set.as_ptr())
        };
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    pub(super) fn set_realtime(priority: i32) -> io::Result<()> {
        // SAFETY: zeroed `sched_param` is valid; `sched_priority` is the only
        // field `SCHED_FIFO` reads.
        let mut param: libc::sched_param = unsafe { std::mem::zeroed() };
        param.sched_priority = priority;
        // SAFETY: `param` outlives the syscall; `pid=0` is the calling thread.
        let rc = unsafe { libc::sched_setscheduler(0, libc::SCHED_FIFO, &param) };
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn pin_to_core_out_of_range_rejects_locally() {
        let err = pin_to_core(libc::CPU_SETSIZE as usize).expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        let err = pin_to_core(usize::MAX).expect_err("must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn non_linux_pin_is_noop() {
        assert!(pin_to_core(0).is_ok());
        assert!(pin_to_core(usize::MAX).is_ok());
    }

    /// Guards against panic/hang/UB in the syscall path: `pin_to_core(0)` must
    /// succeed or return a recognisable errno.
    #[cfg(target_os = "linux")]
    #[test]
    fn pin_to_core_zero_returns_a_result() {
        match pin_to_core(0) {
            Ok(()) => {}
            Err(e) => {
                let raw = e.raw_os_error();
                assert!(
                    matches!(raw, Some(errno) if (1..=255).contains(&errno)),
                    "unexpected pin_to_core(0) error: {e:?}",
                );
            }
        }
    }

    #[test]
    fn set_realtime_invalid_priority_rejects_locally() {
        for &bad in &[0_i32, -1, 100, 255, i32::MIN, i32::MAX] {
            let err = set_realtime(bad).expect_err("must reject out-of-range priority");
            assert_eq!(
                err.kind(),
                io::ErrorKind::InvalidInput,
                "priority {bad} should be InvalidInput",
            );
        }
    }

    /// The validation gate must not reject in-range priorities (syscall result
    /// is platform-dependent, so only `InvalidInput` is forbidden here).
    #[test]
    fn set_realtime_valid_priority_returns_a_result() {
        for &good in &[1_i32, 30, 50, MAX_DAEMON_REALTIME_PRIORITY] {
            match set_realtime(good) {
                Ok(()) => {}
                Err(e) => {
                    assert_ne!(
                        e.kind(),
                        io::ErrorKind::InvalidInput,
                        "valid priority {good} was rejected as InvalidInput: {e:?}",
                    );
                }
            }
        }
    }

    /// Above-cap priorities reject locally even though the kernel accepts up to
    /// 99: defends against a typo that preempts kernel work.
    #[test]
    fn set_realtime_rejects_above_daemon_cap() {
        for &over in &[MAX_DAEMON_REALTIME_PRIORITY + 1, 80, 99] {
            let err = set_realtime(over).expect_err("above-cap priority must reject");
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        }
    }
}
