//! `GET /mic` + `POST /mic/policy` (+ legacy `POST /mic` alias).
//! Catalogue is launch-immutable; only the policy is mutable here.

use std::sync::Arc;

use crate::audio_io::mic_arbitrator::{MicCatalogue, MicPolicy};
use crate::config::MicSettingsHandle;
use axum::Router;
use axum::extract::State;
use axum::response::Json;
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use tokio::task;

use crate::api::error::ApiError;
use crate::api::extract::{ApiJson, ApiQuery};
use crate::api::{AppState, VersionQuery, check_min_version};

/// `GET /mic` response carrying both layers (catalogue + policy) in one round-trip.
#[derive(Serialize)]
struct MicResp {
    catalogue: MicCatalogue,
    policy: MicPolicy,
    /// Read-your-writes stamp; increments on every successful policy mutation.
    version: u64,
}

async fn get_mic(
    State(mic_settings): State<Arc<dyn MicSettingsHandle>>,
    ApiQuery(q): ApiQuery<VersionQuery>,
) -> Result<Json<MicResp>, ApiError> {
    // Atomic value+version (separate reads tear vs a concurrent `try_set_policy`);
    // `?min_version=N` -> 425 until reached.
    let (live, cur) = mic_settings.snapshot_with_version();
    check_min_version(cur, q.min_version)?;
    Ok(Json(MicResp {
        catalogue: (*live.catalogue).clone(),
        policy: live.policy.clone(),
        version: cur.get(),
    }))
}

/// [`MicPolicy`] layer only, cross-validated against the live catalogue before
/// commit (unknown/out-of-whitelist `Fixed` -> 400).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SetMicPolicyReq {
    policy: MicPolicy,
}

async fn post_mic_policy(
    State(mic_settings): State<Arc<dyn MicSettingsHandle>>,
    ApiJson(req): ApiJson<SetMicPolicyReq>,
) -> Result<Json<MicResp>, ApiError> {
    let new_policy = req.policy;
    // Echo our own policy (not a re-snapshot, which could observe a later writer).
    let echoed_policy = new_policy.clone();

    // `try_set_policy` validate+swap+persist; the TOML write needs `spawn_blocking`.
    let mic_settings_for_spawn = mic_settings.clone();
    let receipt = task::spawn_blocking(
        move || -> Result<crate::common::version::SwapReceipt, ApiError> {
            mic_settings_for_spawn
                .try_set_policy(new_policy)
                .map_err(Into::into)
        },
    )
    .await??;

    // Catalogue is launch-immutable, so any snapshot's Arc is fine.
    let catalogue = (*mic_settings.snapshot().catalogue).clone();
    Ok(Json(MicResp {
        catalogue,
        policy: echoed_policy,
        version: receipt.version.get(),
    }))
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/mic", get(get_mic).post(post_mic_policy))
        .route("/mic/policy", post(post_mic_policy))
}
