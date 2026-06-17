//! Neutral DSP utilities shared across capture, preprocessing, and codec.
//!
//! Layering invariant: `dsp` depends on nothing, so capture, `preproc`, and
//! `opus_stream` reaching [`resample`] only through `dsp` gain no upward edge
//! to `audio_io`; holds only items with more than one consumer outside `preproc`.

pub mod resample;
