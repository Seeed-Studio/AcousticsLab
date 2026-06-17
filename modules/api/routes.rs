//! Per-domain route modules.
//!
//! Blocking-pool discipline: sync FS I/O (`FsService` / `WorkspaceMgr`) plus the
//! head install/swap chain (`HotHead::install_prevalidated`, an in-memory `ArcSwap`
//! landing after `current.json` is durable) MUST run inside `spawn_blocking` or it
//! starves the tokio worker driving the shared audio + inference broadcast loops.

pub mod active;
pub mod converter;
pub mod dataset;
pub mod heads;
pub mod health;
pub mod inference;
pub mod jobs;
pub mod mic;
pub mod status;
pub mod training;
pub mod workspace;
