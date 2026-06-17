//! Audio capture, intra-mic channel arbitration, and streaming resample.
//!
//! [`mic_arbitrator::MicArbitrator`] RMS-arbitrates the loudest whitelisted
//! channel within a single capture source; mic selection is operator-set or
//! first-available failover only (no cross-mic RMS-driven switching).

#![warn(missing_debug_implementations)]

pub mod mic_arbitrator;
pub mod mock;
pub(crate) mod source;
