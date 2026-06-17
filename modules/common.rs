//! Acoustics Lab contract module: shared vocabulary (dimension newtypes,
//! validated ids, `ErrorKind`/`Categorized`, time/version primitives,
//! object-safe traits) every other module uses.
//!
//! Invariant: a leaf the whole tree may import -- zero heavy deps
//! (no tokio/axum/prost/alsa/opus/burn/rknn) and no unsafe.

#![forbid(unsafe_code)]

pub mod asset_path;
pub mod dims;
pub mod error;
// Pure-bytes `.mpk` head-artifact header spec shared by converter (writes) and inference (reads).
pub mod head_header;
// Hex digest encoder; lives here so digest-stamping layers can import it despite the `inference -> file_mgr` ban.
pub mod hex;
pub mod ids;
pub mod log_truncate;
pub mod time;
pub mod traits;
pub mod version;
pub mod workspace;
