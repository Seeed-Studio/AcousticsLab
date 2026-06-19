//! Acoustics Lab daemon library crate; the `acousticslabd` binary calls [`daemon::run`].

// Inner `#![forbid(unsafe_code)]` scopes that guardrail to this subtree only.
pub mod common;
pub mod proto;

pub mod allocator;
pub mod audio_buffer;
pub mod file_mgr;
pub mod sched;

// The sinc resampler lives in `dsp::resample`; `preproc` and `opus_stream`
// reach it there directly, not through this capture module.
pub mod audio_io;
pub mod dsp;
pub mod model;
pub mod preproc;

#[cfg(feature = "rknpu")]
pub mod rknn_runtime;

pub mod inference;
pub mod opus_stream;
pub mod stream_io;

pub mod api;
// `LaunchConfig` read once at boot, `Config` is hot-reloadable via notify + `ArcSwap`.
pub mod config;
pub mod converter;
pub mod daemon;
pub mod status;
pub mod training;
