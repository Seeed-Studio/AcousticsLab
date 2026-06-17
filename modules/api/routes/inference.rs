//! `GET/POST /inference`: live `InferenceCfg` (hop_samples + top_k).

use std::sync::Arc;

use crate::config::ConfigHandle;
use crate::inference::InferenceCfg;
use arc_swap::ArcSwap;
use axum::Router;
use axum::extract::State;
use axum::response::{IntoResponse, Json};
use axum::routing::get;
use serde::{Deserialize, Serialize};

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiJson;

#[derive(Serialize)]
struct InferenceResp {
    cfg: InferenceCfg,
}

async fn get_inference(
    State(inference_cfg): State<Arc<ArcSwap<InferenceCfg>>>,
) -> impl IntoResponse {
    let cfg = **inference_cfg.load();
    Json(InferenceResp { cfg })
}

/// `deny_unknown_fields` so a client typo 400s (via `ApiJson` -> `ApiError::Bad`) instead of being silently kept at the prior value by `unwrap_or`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetInferenceReq {
    /// Stride in samples, `MIN_HOP_SAMPLES..=MAX_HOP_SAMPLES` (75% max overlap .. 1 Hz at 44.1 kHz).
    pub hop_samples: Option<usize>,
    /// TopK entries per frame, `1..=MAX_TOP_K` (64).
    pub top_k: Option<usize>,
}

async fn post_inference(
    State(inference_cfg): State<Arc<ArcSwap<InferenceCfg>>>,
    State(config): State<Arc<dyn ConfigHandle>>,
    ApiJson(req): ApiJson<SetInferenceReq>,
) -> Result<Json<InferenceResp>, ApiError> {
    // Validate before locking so a 400 never holds the config lock; bounds owned by `InferenceCfg::validate`.
    let current = **inference_cfg.load();
    let candidate = InferenceCfg {
        hop_samples: req.hop_samples.unwrap_or(current.hop_samples),
        top_k: req.top_k.unwrap_or(current.top_k),
    };
    candidate.validate().map_err(ApiError::Bad)?;

    // `spawn_blocking`: `commit` holds a parking_lot guard across a multi-ms fsync. Merge runs
    // INSIDE the lock so concurrent partial updates aren't computed from a stale snapshot.
    let inference_cfg_for_after = inference_cfg.clone();
    let next = tokio::task::spawn_blocking(move || -> Result<InferenceCfg, ApiError> {
        let mut guard = config.open_mutation()?;
        let next = {
            let c = guard.config();
            let mut next = c.inference;
            if let Some(h) = req.hop_samples {
                next.hop_samples = h;
            }
            if let Some(k) = req.top_k {
                next.top_k = k;
            }
            c.inference = next;
            next
        };
        // Allocate the ArcSwap value before `commit`: its `after` closure runs post-disk-write, so an
        // Arc::new OOM-panic there would leave disk=NEW, live=OLD; hoisting makes any panic precede every side effect.
        let next_arc = Arc::new(next);
        guard.commit(Box::new(move |_c| {
            inference_cfg_for_after.store(next_arc);
        }))?;
        Ok(next)
    })
    .await??;
    Ok(Json(InferenceResp { cfg: next }))
}

pub fn router() -> Router<AppState> {
    Router::new().route("/inference", get(get_inference).post(post_inference))
}
