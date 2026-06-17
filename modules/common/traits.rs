//! Object-safe traits consumed as `Arc<dyn Trait>` so test mocks substitute without rebuilding dependents.

pub mod head_store;
pub mod lag_source;
