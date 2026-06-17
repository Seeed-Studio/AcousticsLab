//! In-memory job snapshot + SSE routes via [`crate::file_mgr::JobRegistry`] (no log files
//! opened); durable JSONL lives at `/workspaces/{id}/assets/{training,converter}_logs/{job_id}.jsonl`.
//! `GET /jobs/{job_id}/events` replays ring events after `after_seq` then follows the broadcast
//! channel until terminal/disconnect; a cursor older than the ring's oldest seq yields 409 `event_gap`.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::get;
use futures_util::stream::{self, Stream};
use serde::{Deserialize, Serialize};

use crate::api::AppState;
use crate::api::error::ApiError;
use crate::api::extract::ApiQuery;
use crate::common::ids::JobId;
use crate::file_mgr::{EventGap, JobEvent, JobRegistry, JobSnapshot};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JobsListQuery {
    /// Clamped server-side to `cfg.max_recent_jobs`.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JobEventsQuery {
    /// Replay strictly after this seq; `0`/absent replays whatever the ring holds.
    #[serde(default)]
    after_seq: Option<u64>,
    /// Include log-line events; defaults to `true`.
    #[serde(default)]
    logs: Option<bool>,
}

/// 409 body for [`EventGap`]; `code` is the literal `"event_gap"`.
#[derive(Serialize)]
struct EventGapBody {
    error: &'static str,
    code: &'static str,
    oldest_seq: u64,
    latest_seq: u64,
}

fn job_event_to_sse(event: JobEvent, include_logs: bool) -> (Event, bool) {
    let terminal = event.state.is_some_and(|state| !state.is_active());
    if !include_logs && event.message.is_some() {
        return (Event::default().comment("log filtered"), terminal);
    }
    let json = serde_json::to_string(&event).unwrap_or_default();
    (Event::default().event("job").data(json), terminal)
}

async fn list_jobs(
    State(jobs): State<Arc<JobRegistry>>,
    ApiQuery(q): ApiQuery<JobsListQuery>,
) -> Json<Vec<JobSnapshot>> {
    let cap = jobs.cfg().max_recent_jobs;
    let limit = q.limit.unwrap_or(cap).min(cap);
    Json(jobs.recent(limit))
}

async fn get_job(
    State(jobs): State<Arc<JobRegistry>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobSnapshot>, ApiError> {
    let job_id = JobId::parse(&job_id)?;
    jobs.snapshot(job_id)
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("job not found: {job_id}")))
}

async fn job_events(
    State(jobs): State<Arc<JobRegistry>>,
    Path(job_id): Path<String>,
    ApiQuery(q): ApiQuery<JobEventsQuery>,
) -> Result<Response, ApiError> {
    let job_id = JobId::parse(&job_id)?;
    if jobs.snapshot(job_id).is_none() {
        return Err(ApiError::NotFound(format!("job not found: {job_id}")));
    }
    let after_seq = q.after_seq.unwrap_or(0);
    let include_logs = q.logs.unwrap_or(true);
    let stream = match jobs.subscribe_events(job_id, after_seq) {
        Ok(s) => s,
        Err(EventGap {
            oldest_seq,
            latest_seq,
        }) => {
            let body = EventGapBody {
                error: "event ring overflow; backfill via /{training,converter}_logs",
                code: "event_gap",
                oldest_seq,
                latest_seq,
            };
            return Ok((StatusCode::CONFLICT, Json(body)).into_response());
        }
    };
    // RAII guard owned by the unfold state so disconnect/terminal/abrupt-drop all decrement
    // `sse_clients_current`; `Option` since the metrics global may be absent in tests.
    let sse_guard: Option<crate::status::SseClientGuard> =
        crate::status::workspace_metrics::global().map(|m| m.sse_client_guard());

    let event_stream = stream::unfold(
        (stream, false, include_logs, sse_guard),
        move |(mut s, terminal_emitted, include_logs, sse_guard)| async move {
            // Replay is registry-filtered already; only logs are filtered here.
            if let Some(e) = s.next_replay() {
                let (event, terminal) = job_event_to_sse(e, include_logs);
                let terminal_emitted = terminal_emitted || terminal;
                return Some((Ok(event), (s, terminal_emitted, include_logs, sse_guard)));
            }
            // Replay drained: close if terminal seen, else await live events.
            if terminal_emitted || s.terminal_seen() {
                drop(sse_guard);
                return None;
            }
            match s.recv().await {
                Ok(e) => {
                    let (event, terminal) = job_event_to_sse(e, include_logs);
                    Some((
                        Ok::<_, Infallible>(event),
                        (s, terminal, include_logs, sse_guard),
                    ))
                }
                Err(_) => {
                    drop(sse_guard);
                    None
                } // Lagged or Closed: end stream.
            }
        },
    );
    let pinned: std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> =
        Box::pin(event_stream);
    let sse = Sse::new(pinned).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
    let mut response = sse.into_response();
    // `no-cache` per SSE spec; `X-Accel-Buffering: no` disables nginx proxy buffering.
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert("X-Accel-Buffering", HeaderValue::from_static("no"));
    Ok(response)
}

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/jobs", get(list_jobs))
        .route("/jobs/{job_id}", get(get_job))
        .route("/jobs/{job_id}/events", get(job_events))
}
