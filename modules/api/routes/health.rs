//! Liveness ping for external process supervisors.

use axum::Router;
use axum::response::{IntoResponse, Json};
use axum::routing::get;

use crate::api::AppState;

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok"}))
}

pub fn router() -> Router<AppState> {
    Router::new().route("/health", get(health))
}
