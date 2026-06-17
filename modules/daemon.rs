//! Acoustics Lab daemon boot and subsystem wiring.
//!
//! Failed subsystems are not self-restarted; an external supervisor (e.g.
//! systemd `Type=notify`) owns restart.

#![warn(missing_debug_implementations)]

pub(crate) mod drain_registry;

mod main_body;

pub use main_body::run;
