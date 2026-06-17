//! Global-allocator release hint via mimalloc's `mi_collect`.
//!
//! Outside [`crate::common`] (`#![forbid(unsafe_code)]`) because the FFI needs `unsafe`.
//! Call after large buffers drop; never from hot paths.

#![warn(missing_debug_implementations)]

/// Hint the allocator to release freed pages to the OS. With `mimalloc`, `mi_collect(true)`
/// drops VmRSS now via Linux `madvise(MADV_DONTNEED)`; otherwise no-op. ~<10 ms, so async
/// callers should `spawn_blocking`.
pub fn release_to_os() {
    #[cfg(feature = "mimalloc")]
    {
        // SAFETY: thread-safe FFI, no pointer args; safe even when mimalloc isn't the active
        // allocator (operates on an empty arena pool).
        unsafe { libmimalloc_sys::mi_collect(true) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_to_os_does_not_panic() {
        release_to_os();
        release_to_os();
    }
}
