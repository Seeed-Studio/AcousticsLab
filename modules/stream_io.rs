//! WebSocket fan-out for the audio + inference broadcast streams (`[api]`
//! listener: `/stream/audio` Opus, `/stream/infer` protobuf), plus the
//! WS-free raw-UDS inference push socket ([`serve_inference_uds`],
//! length-prefixed `Envelope` frames, capped by [`INFERENCE_UDS_MAX_CONNS`]).
//!
//! `axum::serve` accepts `TcpListener` natively; UDS hand-rolls the accept loop
//! on `hyper-util` ([`serve_tcp`] / [`serve_uds`]). Each WS connection holds a
//! `SubscriberGuard` bumping a `watch::Sender<usize>` so `opus_stream::run`
//! auto-pauses the encoder at zero clients. Lag past channel capacity closes
//! 1011 rather than skipping, avoiding a torn protobuf decode at the receiver.

#![warn(missing_debug_implementations)]

pub mod framing;
pub use framing::{
    FramingEncodeError, FramingError, MAX_UDS_FRAME_BYTES, ProtoDecodeError, WS_SUBPROTOCOL,
    decode_envelope, decode_length_prefixed, try_encode_length_prefixed, wrap_audio,
    wrap_inference,
};

use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::get;
use bytes::Bytes;
use thiserror::Error;
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::sync::{broadcast, watch};
use tokio_util::sync::CancellationToken;

/// Listener bind / permissions / serve-loop failures; only [`Self::Serve`] is
/// operator-visible at runtime (the rest fail boot).
#[derive(Debug, Error)]
pub enum StreamError {
    #[error("uds bind {path}: {source}")]
    UdsBind {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("tcp bind {addr}: {source}")]
    TcpBind {
        addr: String,
        #[source]
        source: std::io::Error,
    },
    #[error("uds permissions {path}: {source}")]
    UdsPerms {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("uds remove stale {path}: {source}")]
    UdsRemove {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// Stat of the existing entry failed (no unlink attempted, typically
    /// `EACCES`); distinct from [`Self::UdsRemove`].
    #[error("uds stat {path}: {source}")]
    UdsStat {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// Existing entry at `path` is not a Unix socket; rejected before any
    /// `unlink` so operator data is never silently destroyed. `kind` labels it.
    #[error("uds path {path} is a {kind}, not a unix socket")]
    UdsPathNotSocket { path: String, kind: &'static str },
    /// Parent dir missing/symlink/non-dir reopens the unlink/swap TOCTOU the
    /// path-based bind+chmod relies on the parent to close (world-writable-no-
    /// sticky parent is the softer hijack risk: warned, not rejected).
    #[error("uds parent dir {parent} for {path} is not safely confined: {detail}")]
    UdsParentInsecure {
        path: String,
        parent: String,
        detail: String,
    },
    /// Serve-loop failure: listener fd is gone, supervisor restarts.
    #[error("{transport} serve loop: {source}")]
    Serve {
        transport: &'static str,
        #[source]
        source: std::io::Error,
    },
}

/// All variants are server-side failures (bind / permissions / serve-loop),
/// never operator input, so all map to `Internal`.
impl crate::common::error::Categorized for StreamError {
    fn kind(&self) -> crate::common::error::ErrorKind {
        crate::common::error::ErrorKind::Internal
    }
}

/// Per-stream broadcast lag counters, surfaced in `/api/v1/status`. `Relaxed`:
/// pure counters, no ordering dependency on the broadcast.
#[derive(Clone, Debug, Default)]
pub struct BroadcastLagCounters {
    audio: Arc<std::sync::atomic::AtomicU64>,
    inference: Arc<std::sync::atomic::AtomicU64>,
}

impl BroadcastLagCounters {
    /// Cumulative messages (not events) dropped on the audio channel.
    pub fn audio_messages_dropped(&self) -> u64 {
        self.audio.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn inference_messages_dropped(&self) -> u64 {
        self.inference.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// So `api` can carry `Arc<dyn LagSource>` without depending on `stream_io`.
impl crate::common::traits::lag_source::LagSource for BroadcastLagCounters {
    fn snapshot(&self) -> crate::common::traits::lag_source::BroadcastLagSnapshot {
        crate::common::traits::lag_source::BroadcastLagSnapshot {
            audio_messages_dropped: self.audio_messages_dropped(),
            inference_messages_dropped: self.inference_messages_dropped(),
        }
    }
}

/// Transport-level admission policy for the WS endpoints, tunable via TOML
/// (`[api.tcp_policy]` / `[api.uds_policy]`). Enforced in the upgrade handlers
/// BEFORE upgrade completes, so a rejected client never reaches the broadcast
/// subscribe step. No auth / `Origin` filtering -- that lives at the proxy.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TransportPolicy {
    /// `0` disables the cap; else the WS handler rejects 429 at `subscribers >= max`.
    pub max_connections_per_stream: u32,
    /// When `true` (default) every WS upgrade MUST list `acoustics` in
    /// `Sec-WebSocket-Protocol` or it is rejected 400. `false` when admission
    /// is gated elsewhere (UDS perms, localhost-only bind).
    pub require_subprotocol: bool,
}

impl Default for TransportPolicy {
    fn default() -> Self {
        Self {
            max_connections_per_stream: 0,
            require_subprotocol: true,
        }
    }
}

impl TransportPolicy {
    /// Production defaults: 32 conns/stream, strict subprotocol check.
    pub fn capped() -> Self {
        Self {
            max_connections_per_stream: 32,
            require_subprotocol: true,
        }
    }
}

/// State threaded into every WS upgrade handler.
#[derive(Clone)]
struct AppState {
    audio_tx: broadcast::Sender<Bytes>,
    infer_tx: broadcast::Sender<Bytes>,
    audio_subs: Arc<watch::Sender<usize>>,
    infer_subs: Arc<watch::Sender<usize>>,
    lag_counters: BroadcastLagCounters,
    policy: Arc<TransportPolicy>,
    /// Into every detached `handle_ws` task so a WS connection winds down within
    /// the drain budget at SIGTERM, not only at runtime drop.
    shutdown: CancellationToken,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("audio_subs", &*self.audio_subs.borrow())
            .field("infer_subs", &*self.infer_subs.borrow())
            .finish_non_exhaustive()
    }
}

/// Owned WS fan-out state. Construct once; spawn one `serve_tcp` + one
/// `serve_uds` task.
#[derive(Debug)]
pub struct StreamRouter {
    audio_tx: broadcast::Sender<Bytes>,
    infer_tx: broadcast::Sender<Bytes>,
    audio_subs_tx: Arc<watch::Sender<usize>>,
    audio_subs_rx: watch::Receiver<usize>,
    infer_subs_tx: Arc<watch::Sender<usize>>,
    infer_subs_rx: watch::Receiver<usize>,
    lag_counters: BroadcastLagCounters,
    policy: Arc<TransportPolicy>,
}

impl StreamRouter {
    /// Default 64 slots/stream: ~1.3 s buffered audio (50 Hz), ~16 s inference
    /// (4 Hz) before a slow client trips `RecvError::Lagged` and is closed 1011.
    pub fn new() -> Self {
        Self::with_capacities(64, 64)
    }

    pub fn with_capacities(audio_cap: usize, infer_cap: usize) -> Self {
        Self::with_capacities_and_policy(audio_cap, infer_cap, TransportPolicy::default())
    }

    pub fn with_capacities_and_policy(
        audio_cap: usize,
        infer_cap: usize,
        policy: TransportPolicy,
    ) -> Self {
        let (audio_tx, _) = broadcast::channel::<Bytes>(audio_cap);
        let (infer_tx, _) = broadcast::channel::<Bytes>(infer_cap);
        let (audio_subs_tx, audio_subs_rx) = watch::channel(0usize);
        let (infer_subs_tx, infer_subs_rx) = watch::channel(0usize);
        Self {
            audio_tx,
            infer_tx,
            audio_subs_tx: Arc::new(audio_subs_tx),
            audio_subs_rx,
            infer_subs_tx: Arc::new(infer_subs_tx),
            infer_subs_rx,
            lag_counters: BroadcastLagCounters::default(),
            policy: Arc::new(policy),
        }
    }

    pub fn lag_counters(&self) -> BroadcastLagCounters {
        self.lag_counters.clone()
    }

    pub fn audio_tx(&self) -> broadcast::Sender<Bytes> {
        self.audio_tx.clone()
    }

    pub fn infer_tx(&self) -> broadcast::Sender<Bytes> {
        self.infer_tx.clone()
    }

    /// Audio-subscriber count; `opus_stream::run` auto-pauses at 0.
    pub fn audio_subscribers(&self) -> watch::Receiver<usize> {
        self.audio_subs_rx.clone()
    }

    /// Inference-subscriber count; the engine never auto-pauses (always runs),
    /// unlike audio capture.
    pub fn infer_subscribers(&self) -> watch::Receiver<usize> {
        self.infer_subs_rx.clone()
    }

    /// Build the axum router with the constructor's [`TransportPolicy`].
    pub fn router(&self) -> Router {
        self.router_with_policy(self.policy.as_ref().clone())
    }

    /// Explicit [`TransportPolicy`], one per listener (distinct TCP vs UDS
    /// admission) while sharing the broadcast channels and subscriber counters.
    pub fn router_with_policy(&self, policy: TransportPolicy) -> Router {
        // Fresh never-cancelled token (tests / non-drain callers): WS shutdown arm inert.
        self.router_with_policy_and_shutdown(policy, CancellationToken::new())
    }

    /// Threads a per-listener `shutdown` token (same one given to `serve_tcp` /
    /// `serve_uds`) into every upgraded WS task: axum spawns `handle_ws`
    /// untracked by the listener's graceful-shutdown, so a SIGTERM drain would
    /// otherwise leave them for runtime drop.
    pub fn router_with_policy_and_shutdown(
        &self,
        policy: TransportPolicy,
        shutdown: CancellationToken,
    ) -> Router {
        let state = AppState {
            audio_tx: self.audio_tx.clone(),
            infer_tx: self.infer_tx.clone(),
            audio_subs: self.audio_subs_tx.clone(),
            infer_subs: self.infer_subs_tx.clone(),
            lag_counters: self.lag_counters.clone(),
            policy: Arc::new(policy),
            shutdown,
        };
        Router::new()
            .route("/stream/audio", get(audio_ws_handler))
            .route("/stream/infer", get(infer_ws_handler))
            .with_state(state)
    }
}

impl Default for StreamRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// `Err(429)` short-circuits the upgrade so a rejected client never subscribes;
/// on admit the caller moves the returned guard into `ws.on_upgrade`.
fn enforce_admission(
    subs_tx: Arc<watch::Sender<usize>>,
    policy: &TransportPolicy,
) -> Result<SubscriberGuard, StatusCode> {
    SubscriberGuard::try_acquire(subs_tx, policy.max_connections_per_stream)
        .ok_or(StatusCode::TOO_MANY_REQUESTS)
}

/// Inbound WS per-frame/message cap, aligned with [`MAX_UDS_FRAME_BYTES`]
/// (64 KiB): streams are producer-only, so clamping replaces axum's 16/64 MiB
/// defaults that would let one slow oversize frame pin tens of MB resident.
const WS_INBOUND_BYTE_CAP: usize = MAX_UDS_FRAME_BYTES as usize;

/// Per-send budget for every outbound `socket.send` in `handle_ws`: without it a
/// peer that handshakes then never reads pins `send().await` forever, holding its
/// `SubscriberGuard` and draining the per-stream cap; also bounds the Lagged/
/// Closed close sends so a close can't wedge behind the lag backpressure.
const WS_SEND_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

/// Per-connection tungstenite read-buffer: axum's 128 KiB default is zero-filled
/// resident on the first inbound read, but producer-only clients send only
/// Close/Ping/Pong control frames (<= 125 B), so 8 KiB saves ~120 KiB/client and
/// still grows on demand to [`WS_INBOUND_BYTE_CAP`].
const WS_READ_BUFFER_BYTES: usize = 8 * 1024;

/// Bake the subprotocol + inbound byte caps into every WS upgrade so no endpoint
/// inherits axum's 16/64 MiB defaults. Callers MUST run `enforce_subprotocol` +
/// `enforce_admission` BEFORE this (those per-handler gates need the topic-
/// specific subscriber `watch::Sender`).
fn configure_ws_upgrade(ws: WebSocketUpgrade) -> WebSocketUpgrade {
    ws.protocols([WS_SUBPROTOCOL])
        .max_frame_size(WS_INBOUND_BYTE_CAP)
        .max_message_size(WS_INBOUND_BYTE_CAP)
        .read_buffer_size(WS_READ_BUFFER_BYTES)
}

async fn audio_ws_handler(
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    State(state): State<AppState>,
) -> axum::response::Response {
    if let Err(s) = enforce_subprotocol(&headers, &state.policy) {
        return s.into_response();
    }
    let guard = match enforce_admission(state.audio_subs.clone(), &state.policy) {
        Ok(g) => g,
        Err(s) => return s.into_response(),
    };
    let rx = state.audio_tx.subscribe();
    let lag = state.lag_counters.audio.clone();
    let shutdown = state.shutdown.clone();
    configure_ws_upgrade(ws)
        .on_upgrade(move |socket| handle_ws(socket, rx, guard, lag, "audio", shutdown))
        .into_response()
}

async fn infer_ws_handler(
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    State(state): State<AppState>,
) -> axum::response::Response {
    if let Err(s) = enforce_subprotocol(&headers, &state.policy) {
        return s.into_response();
    }
    let guard = match enforce_admission(state.infer_subs.clone(), &state.policy) {
        Ok(g) => g,
        Err(s) => return s.into_response(),
    };
    let rx = state.infer_tx.subscribe();
    let lag = state.lag_counters.inference.clone();
    let shutdown = state.shutdown.clone();
    configure_ws_upgrade(ws)
        .on_upgrade(move |socket| handle_ws(socket, rx, guard, lag, "infer", shutdown))
        .into_response()
}

/// Require `Sec-WebSocket-Protocol: acoustics` when
/// [`TransportPolicy::require_subprotocol`] is `true` (default): axum's
/// `protocols([...])` echoes the matched token but does NOT reject clients that
/// omit it (RFC 6455 allows accept-without-echo), so an outdated client could
/// otherwise stream pre-envelope payloads it can't decode. `false` skips it
/// (UDS/localhost-only, where the listener is the auth boundary).
fn enforce_subprotocol(headers: &HeaderMap, policy: &TransportPolicy) -> Result<(), StatusCode> {
    if !policy.require_subprotocol {
        return Ok(());
    }
    let Some(raw) = headers.get(header::SEC_WEBSOCKET_PROTOCOL) else {
        return Err(StatusCode::BAD_REQUEST);
    };
    let s = raw.to_str().map_err(|_| StatusCode::BAD_REQUEST)?;
    // Admission is case-insensitive (RFC 6455) but axum's `protocols()` echo is
    // case-SENSITIVE: a non-lowercase token admitted here yields a 101 with no
    // echo (browsers reject that), so they agree only for the exact lowercase token.
    let listed = s
        .split(',')
        .any(|tok| tok.trim().eq_ignore_ascii_case(WS_SUBPROTOCOL));
    if listed {
        Ok(())
    } else {
        Err(StatusCode::BAD_REQUEST)
    }
}

/// RAII subscriber-count guard: increments on construction, decrements on drop
/// (panic-safe, async-cancellation-safe).
struct SubscriberGuard {
    tx: Arc<watch::Sender<usize>>,
}

impl SubscriberGuard {
    /// Atomic acquire-or-reject: `send_if_modified`'s closure runs under the
    /// watch's internal lock, so observe-and-bump is race-free against concurrent
    /// acquires. `None` at cap; `max_per_stream = 0` means no cap.
    fn try_acquire(tx: Arc<watch::Sender<usize>>, max_per_stream: u32) -> Option<Self> {
        if max_per_stream == 0 {
            tx.send_modify(|c| *c = c.saturating_add(1));
            return Some(Self { tx });
        }
        let admitted = tx.send_if_modified(|c| {
            if *c < max_per_stream as usize {
                *c = c.saturating_add(1);
                true
            } else {
                false
            }
        });
        if admitted { Some(Self { tx }) } else { None }
    }
}

impl Drop for SubscriberGuard {
    fn drop(&mut self) {
        self.tx.send_modify(|c| *c = c.saturating_sub(1));
    }
}

/// Budgeted Close send (bounded by [`WS_SEND_BUDGET`]) so a slow peer can't
/// wedge it. `code` is an RFC 6455 WS close code.
async fn send_close_budgeted(socket: &mut WebSocket, code: u16, reason: &str) {
    let _ = tokio::time::timeout(
        WS_SEND_BUDGET,
        socket.send(Message::Close(Some(CloseFrame {
            code,
            reason: reason.into(),
        }))),
    )
    .await;
}

async fn handle_ws(
    mut socket: WebSocket,
    mut rx: broadcast::Receiver<Bytes>,
    guard: SubscriberGuard,
    lag_counter: Arc<std::sync::atomic::AtomicU64>,
    stream_name: &'static str,
    shutdown: CancellationToken,
) {
    // Drop on any exit decrements the subscriber count.
    let _guard = guard;
    tracing::debug!(target: "stream_io", stream = stream_name, "ws subscribed");

    loop {
        // CANCEL-SAFE: broadcast/WS `recv` are cancel-safe and a mid-send-
        // cancelled `socket.send` is fine -- the only durable invariant is the
        // subscriber-count decrement in `_guard`'s Drop.
        tokio::select! {
            biased;
            // Detached task untracked by listener graceful-shutdown: observe the
            // token directly and close within the drain budget.
            _ = shutdown.cancelled() => {
                send_close_budgeted(&mut socket, 1001, "server shutting down").await;
                break;
            }
            recv = rx.recv() => match recv {
                Ok(payload) => {
                    // Binary takes Bytes directly (no copy); timeout treats a
                    // never-reading peer as gone so it can't pin the slot.
                    match tokio::time::timeout(
                        WS_SEND_BUDGET,
                        socket.send(Message::Binary(payload)),
                    ).await {
                        Ok(Ok(())) => {}
                        Ok(Err(_)) => break,
                        Err(_) => {
                            tracing::warn!(
                                target: "stream_io",
                                stream = stream_name,
                                budget_ms = WS_SEND_BUDGET.as_millis() as u64,
                                "ws send budget exhausted; closing slow subscriber",
                            );
                            break;
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    lag_counter.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                    tracing::warn!(
                        target: "stream_io",
                        stream = stream_name,
                        skipped = n,
                        "ws receiver lagged; closing 1011",
                    );
                    send_close_budgeted(&mut socket, 1011, &format!("lagged {n}")).await;
                    break;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    send_close_budgeted(&mut socket, 1001, "stream closed").await;
                    break;
                }
            },
            client_msg = socket.recv() => match client_msg {
                None => break,
                Some(Ok(Message::Close(_))) => break,
                Some(Ok(Message::Ping(payload))) => {
                    // tungstenite flushes its auto-pong only on the NEXT recv(),
                    // which a ping-only/half-open client never triggers, so flush
                    // explicitly now (coalesced, no dup frame); budgeted so an
                    // unbudgeted pin on this detached task can't leak the slot.
                    match tokio::time::timeout(
                        WS_SEND_BUDGET,
                        socket.send(Message::Pong(payload)),
                    )
                    .await
                    {
                        Ok(Ok(())) => {}
                        Ok(Err(_)) => break,
                        Err(_) => {
                            tracing::warn!(
                                target: "stream_io",
                                stream = stream_name,
                                budget_ms = WS_SEND_BUDGET.as_millis() as u64,
                                "ws pong-send budget exhausted; closing slow subscriber",
                            );
                            break;
                        }
                    }
                }
                Some(Ok(_)) => {
                    // Producer-only; ignore client text/binary.
                }
                Some(Err(e)) => {
                    tracing::debug!(target: "stream_io", err = %e, "ws recv error");
                    break;
                }
            }
        }
    }

    tracing::debug!(target: "stream_io", stream = stream_name, "ws unsubscribed");
}

/// Serve `router` over TCP until `shutdown` fires or the listener errors.
pub async fn serve_tcp(
    listener: TcpListener,
    router: Router,
    shutdown: CancellationToken,
) -> Result<(), StreamError> {
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            shutdown.cancelled().await;
        })
        .await
        .map_err(|source| StreamError::Serve {
            transport: "tcp",
            source,
        })
}

/// Classify a `UnixListener::accept` error as transient (FD pressure / client
/// churn -- must NOT tear down the listener) vs fatal. EMFILE recovers as
/// in-flight connections close; ECONNABORTED is a vanished connect; EAGAIN
/// can't occur here but is classified defensively.
fn is_transient_accept_error(e: &std::io::Error) -> bool {
    matches!(
        e.raw_os_error(),
        Some(libc::EMFILE) | Some(libc::ENFILE) | Some(libc::EAGAIN) | Some(libc::ECONNABORTED)
    )
}

/// Transient-accept back-off: short enough that a one-off FD spike costs few
/// iterations, long enough that a sustained EMFILE storm doesn't spin a CPU on
/// the failing syscall.
const UDS_ACCEPT_BACKOFF: std::time::Duration = std::time::Duration::from_millis(100);

/// Serve `router` over UDS. `axum::serve` can't accept `UnixListener`, so
/// hand-roll the accept loop on hyper's HTTP/1.1 builder (HTTP/1.1 only --
/// axum's WS upgrade requires it). Each connection is its own task so a slow
/// client can't block accept; transient FD-pressure errors back off.
pub async fn serve_uds(
    listener: UnixListener,
    router: Router,
    shutdown: CancellationToken,
) -> Result<(), StreamError> {
    use hyper_util::service::TowerToHyperService;

    // Per-conn tasks: bounded drain at shutdown plus continuous steady-state
    // reaping so the set can't grow unbounded.
    let mut conns: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();
    let mut conn_id = 0u64;
    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                tracing::debug!(
                    target: "stream_io",
                    open_conns = conns.len(),
                    "uds accept loop: shutdown",
                );
                break;
            }
            // Reap continuously; guard against empty-set busy-loop.
            Some(joined) = conns.join_next(), if !conns.is_empty() => {
                if let Err(je) = joined
                    && !je.is_cancelled()
                {
                    // `warn!` so a panicked UDS WS handler survives the default `info` filter.
                    tracing::warn!(
                        target: "stream_io",
                        err = %je,
                        is_panic = je.is_panic(),
                        "uds conn task ended with join error (likely a panicked WS handler)",
                    );
                }
            }
            accept = listener.accept() => {
                let (stream, _addr) = match accept {
                    Ok(s) => s,
                    Err(e) => {
                        if is_transient_accept_error(&e) {
                            tracing::warn!(
                                target: "stream_io",
                                err = %e,
                                errno = ?e.raw_os_error(),
                                "uds accept transient failure; backing off",
                            );
                            // Race back-off against `shutdown` so a SIGTERM during
                            // an EMFILE storm cancels at once, not one backoff per
                            // failed accept.
                            tokio::select! {
                                biased;
                                _ = shutdown.cancelled() => break,
                                _ = tokio::time::sleep(UDS_ACCEPT_BACKOFF) => {}
                            }
                            continue;
                        }
                        return Err(StreamError::Serve {
                            transport: "uds",
                            source: e,
                        });
                    }
                };
                conn_id = conn_id.wrapping_add(1);
                let svc = TowerToHyperService::new(router.clone());
                let shutdown_cloned = shutdown.clone();
                conns.spawn(serve_one_uds_conn(stream, svc, shutdown_cloned, conn_id));
            }
        }
    }

    // Drain: tasks already observe `shutdown.cancelled()`; bound the wait
    // (smaller than the outer DrainRegistry envelope), then abort stragglers.
    const UDS_CONN_DRAIN_BUDGET: std::time::Duration = std::time::Duration::from_millis(500);
    let drain_until = tokio::time::Instant::now() + UDS_CONN_DRAIN_BUDGET;
    while !conns.is_empty() {
        match tokio::time::timeout_at(drain_until, conns.join_next()).await {
            Ok(Some(joined)) => {
                if let Err(je) = joined
                    && !je.is_cancelled()
                {
                    tracing::warn!(
                        target: "stream_io",
                        err = %je,
                        is_panic = je.is_panic(),
                        "uds conn task ended with join error during drain (likely a panicked WS handler)",
                    );
                }
                continue;
            }
            Ok(None) => break,
            Err(_elapsed) => break,
        }
    }
    let remaining = conns.len();
    if remaining > 0 {
        tracing::warn!(
            target: "stream_io",
            remaining,
            budget_ms = UDS_CONN_DRAIN_BUDGET.as_millis() as u64,
            "uds conn drain budget exceeded; aborting outstanding tasks",
        );
        conns.abort_all();
    }
    Ok(())
}

/// Serve one HTTP/1.1 (with WS upgrades) UDS connection until it ends or
/// `shutdown` cancels. `tokio::pin!` required: `select!` re-polls the hyper conn
/// future via `&mut conn`.
async fn serve_one_uds_conn(
    stream: UnixStream,
    svc: hyper_util::service::TowerToHyperService<Router>,
    shutdown: CancellationToken,
    conn_id: u64,
) {
    use hyper::server::conn::http1;
    use hyper_util::rt::TokioIo;
    let builder = http1::Builder::new();
    let conn = builder
        .serve_connection(TokioIo::new(stream), svc)
        .with_upgrades();
    tokio::pin!(conn);
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => {
            // hyper has no direct close; dropping the future closes the connection.
            tracing::trace!(target: "stream_io", conn_id, "uds conn cancelled");
        }
        res = &mut conn => {
            if let Err(e) = res {
                tracing::debug!(target: "stream_io", conn_id, err = %e, "uds conn ended");
            }
        }
    }
}

/// Connection cap for the `[output.inference]` raw-UDS socket: a consumer's
/// decoder pre-allocates up to [`MAX_UDS_FRAME_BYTES`] (64 KiB) per in-flight
/// frame, so concurrency MUST be bounded. Not operator-tunable, since a TOML
/// knob could be set to the forbidden uncapped (`0`) shape.
pub const INFERENCE_UDS_MAX_CONNS: u32 = 16;

/// Send budget for the raw-output writer (mirrors [`WS_SEND_BUDGET`]): a consumer
/// that stops reading must not pin a writer task and its cap slot; on timeout
/// treat the peer as gone and close.
const INFERENCE_UDS_SEND_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

/// Drive the `[output.inference]` raw-UDS push listener: each connection receives
/// the inference broadcast as length-prefixed `Envelope` frames
/// ([`framing::decode_length_prefixed`]), no HTTP/WS layer. Capped at
/// [`INFERENCE_UDS_MAX_CONNS`] (uncapped = memory-amplification DoS). Any
/// framing/IO error closes just that connection -- the length prefix is the only
/// sync point, re-sync undefined. Accept-loop hardening mirrors [`serve_uds`].
pub async fn serve_inference_uds(
    listener: UnixListener,
    infer_tx: broadcast::Sender<Bytes>,
    shutdown: CancellationToken,
) -> Result<(), StreamError> {
    // Subscriber counter backing the cap; no receiver retained (`send_modify` /
    // `send_if_modified` use the sender).
    let subs_tx = Arc::new(watch::channel(0usize).0);
    let mut conns: tokio::task::JoinSet<()> = tokio::task::JoinSet::new();
    let mut conn_id = 0u64;
    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                tracing::debug!(
                    target: "stream_io",
                    open_conns = conns.len(),
                    "inference-uds accept loop: shutdown",
                );
                break;
            }
            // Reap continuously; guard against empty-set busy-loop.
            Some(joined) = conns.join_next(), if !conns.is_empty() => {
                if let Err(je) = joined
                    && !je.is_cancelled()
                {
                    tracing::warn!(
                        target: "stream_io",
                        err = %je,
                        is_panic = je.is_panic(),
                        "inference-uds conn task ended with join error",
                    );
                }
            }
            accept = listener.accept() => {
                let (stream, _addr) = match accept {
                    Ok(s) => s,
                    Err(e) => {
                        if is_transient_accept_error(&e) {
                            tracing::warn!(
                                target: "stream_io",
                                err = %e,
                                errno = ?e.raw_os_error(),
                                "inference-uds accept transient failure; backing off",
                            );
                            tokio::select! {
                                biased;
                                _ = shutdown.cancelled() => break,
                                _ = tokio::time::sleep(UDS_ACCEPT_BACKOFF) => {}
                            }
                            continue;
                        }
                        return Err(StreamError::Serve {
                            transport: "inference-uds",
                            source: e,
                        });
                    }
                };
                conn_id = conn_id.wrapping_add(1);
                // Admit BEFORE subscribing so an at-cap peer never reaches the
                // subscribe step (its decoder never pre-allocates); dropping
                // `stream` closes it.
                let Some(guard) =
                    SubscriberGuard::try_acquire(subs_tx.clone(), INFERENCE_UDS_MAX_CONNS)
                else {
                    // At-cap is expected; `debug!` so a connect-flood can't amplify
                    // into unbounded `warn!` volume.
                    tracing::debug!(
                        target: "stream_io",
                        conn_id,
                        cap = INFERENCE_UDS_MAX_CONNS,
                        "inference-uds connection cap reached; rejecting peer",
                    );
                    continue;
                };
                let rx = infer_tx.subscribe();
                let shutdown_cloned = shutdown.clone();
                conns.spawn(serve_one_inference_conn(stream, rx, guard, shutdown_cloned, conn_id));
            }
        }
    }

    // Drain: bounded teardown of writer tasks (mirror of `serve_uds`); budget
    // smaller than the outer DrainRegistry envelope.
    const INFERENCE_UDS_CONN_DRAIN_BUDGET: std::time::Duration =
        std::time::Duration::from_millis(500);
    let drain_until = tokio::time::Instant::now() + INFERENCE_UDS_CONN_DRAIN_BUDGET;
    while !conns.is_empty() {
        match tokio::time::timeout_at(drain_until, conns.join_next()).await {
            Ok(Some(joined)) => {
                if let Err(je) = joined
                    && !je.is_cancelled()
                {
                    tracing::warn!(
                        target: "stream_io",
                        err = %je,
                        is_panic = je.is_panic(),
                        "inference-uds conn task ended with join error during drain",
                    );
                }
                continue;
            }
            Ok(None) => break,
            Err(_elapsed) => break,
        }
    }
    let remaining = conns.len();
    if remaining > 0 {
        tracing::warn!(
            target: "stream_io",
            remaining,
            budget_ms = INFERENCE_UDS_CONN_DRAIN_BUDGET.as_millis() as u64,
            "inference-uds conn drain budget exceeded; aborting outstanding tasks",
        );
        conns.abort_all();
    }
    Ok(())
}

/// Stream the inference broadcast to one raw-UDS consumer as length-prefixed
/// `Envelope` frames until disconnect, broadcast close/lag, or `shutdown`.
/// Close-on-lag is deliberate: the length prefix is the only sync point, a
/// lagged consumer can't resume.
async fn serve_one_inference_conn(
    mut stream: UnixStream,
    mut rx: broadcast::Receiver<Bytes>,
    guard: SubscriberGuard,
    shutdown: CancellationToken,
    conn_id: u64,
) {
    use tokio::io::AsyncWriteExt;
    // Drop on any exit frees the cap slot.
    let _guard = guard;
    tracing::debug!(target: "stream_io", conn_id, "inference-uds subscribed");
    loop {
        // CANCEL-SAFE: `recv` is cancel-safe and a mid-send-cancelled `write_all`
        // is fine -- the only durable invariant is the cap-slot decrement in
        // `_guard`'s Drop.
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => break,
            recv = rx.recv() => match recv {
                Ok(payload) => {
                    // Payload is already `Envelope`-encoded Bytes; the raw socket
                    // only adds the 4-byte length prefix. An over-cap frame can't
                    // re-sync, so close on error.
                    let framed = match try_encode_length_prefixed(&payload) {
                        Ok(f) => f,
                        Err(e) => {
                            tracing::warn!(
                                target: "stream_io",
                                conn_id,
                                err = %e,
                                "inference-uds frame exceeds cap; closing connection",
                            );
                            break;
                        }
                    };
                    match tokio::time::timeout(
                        INFERENCE_UDS_SEND_BUDGET,
                        stream.write_all(&framed),
                    )
                    .await
                    {
                        Ok(Ok(())) => {}
                        Ok(Err(_)) => break,
                        Err(_) => {
                            tracing::warn!(
                                target: "stream_io",
                                conn_id,
                                budget_ms = INFERENCE_UDS_SEND_BUDGET.as_millis() as u64,
                                "inference-uds send budget exhausted; closing slow consumer",
                            );
                            break;
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(
                        target: "stream_io",
                        conn_id,
                        skipped = n,
                        "inference-uds consumer lagged; closing (re-sync undefined)",
                    );
                    break;
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    }
    tracing::debug!(target: "stream_io", conn_id, "inference-uds unsubscribed");
}

/// Label a non-socket [`std::fs::FileType`] for
/// [`StreamError::UdsPathNotSocket`]. Single source so the security-relevant
/// bind-time and chmod-time gates can't diverge on classification.
#[cfg(unix)]
fn describe_non_socket_file_type(ft: &std::fs::FileType) -> &'static str {
    use std::os::unix::fs::FileTypeExt;
    if ft.is_symlink() {
        "symlink"
    } else if ft.is_file() {
        "regular file"
    } else if ft.is_dir() {
        "directory"
    } else if ft.is_fifo() {
        "fifo"
    } else if ft.is_block_device() {
        "block device"
    } else if ft.is_char_device() {
        "char device"
    } else {
        "unknown file type"
    }
}

/// The single socket gate shared by [`bind_uds`] and [`set_uds_permissions`].
#[cfg(unix)]
fn reject_if_not_socket(path: &std::path::Path, ft: &std::fs::FileType) -> Result<(), StreamError> {
    use std::os::unix::fs::FileTypeExt;
    if ft.is_socket() {
        Ok(())
    } else {
        Err(StreamError::UdsPathNotSocket {
            path: path.display().to_string(),
            kind: describe_non_socket_file_type(ft),
        })
    }
}

/// Bind a UDS, safely removing any stale socket file. Caller chmods via
/// [`set_uds_permissions`]. Bind-time backstop to the config layer's
/// `validate_uds_path`; both fail closed. Safety contract:
///
/// 1. Refuses to unlink non-socket files (`symlink_metadata` first); symlinks
///    rejected even to a socket target -- following at unlink time is the
///    classic TOCTOU vector.
/// 2. Validates the parent is a real (non-symlink) directory; world-writable-no-
///    sticky parent is warned not rejected so the daemon still boots (operator
///    owns that narrow risk).
/// 3. Stale-socket cleanup is best-effort; multi-instance deployments must use
///    distinct paths.
pub async fn bind_uds(path: &std::path::Path) -> Result<UnixListener, StreamError> {
    // Reject unsafe parents before touching the filesystem. A None (`"/"`) or
    // empty (`"foo.sock"`) parent is a typo: the daemon never binds at root/CWD.
    let parent = path
        .parent()
        .ok_or_else(|| StreamError::UdsParentInsecure {
            path: path.display().to_string(),
            parent: String::new(),
            detail: "no parent directory; pick a full path (e.g. /run/acousticslab.sock)".into(),
        })?;
    if parent.as_os_str().is_empty() {
        return Err(StreamError::UdsParentInsecure {
            path: path.display().to_string(),
            parent: String::new(),
            detail: "empty parent directory; pick a full path (e.g. /run/acousticslab.sock)".into(),
        });
    }
    validate_parent_dir_confinement(path, parent)?;

    // `symlink_metadata` (NOT `metadata`) so a symlink is observable, not
    // silently followed.
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(StreamError::UdsStat {
                path: path.display().to_string(),
                source: e,
            });
        }
        Ok(md) => {
            let ft = md.file_type();
            // Only a socket is safe to `remove_file + bind`; everything else
            // (incl. symlink to a socket) is a hard reject.
            #[cfg(unix)]
            reject_if_not_socket(path, &ft)?;
            std::fs::remove_file(path).map_err(|e| StreamError::UdsRemove {
                path: path.display().to_string(),
                source: e,
            })?;
        }
    }
    UnixListener::bind(path).map_err(|e| StreamError::UdsBind {
        path: path.display().to_string(),
        source: e,
    })
}

/// Validate the UDS parent dir. A missing/symlinked/non-dir parent is a hard
/// reject -- it reopens the unlink/swap TOCTOU the path-based bind+chmod relies
/// on the parent to close. A world-writable-no-sticky parent only warns: the
/// post-`bind` `chmod` follows the path, so a user who can enter the parent
/// could swap the socket first (`fchmod(fd)` can't help -- UDS fds reject it,
/// EINVAL), but that is a narrow non-default risk and refusing to boot is worse.
fn validate_parent_dir_confinement(
    path: &std::path::Path,
    parent: &std::path::Path,
) -> Result<(), StreamError> {
    // Parent must already exist (don't auto-create -- absence is a typo).
    // `symlink_metadata` (lstat) rejects a symlinked parent: it reintroduces the
    // unlink/swap TOCTOU surface.
    let md = match std::fs::symlink_metadata(parent) {
        Ok(md) => md,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(StreamError::UdsParentInsecure {
                path: path.display().to_string(),
                parent: parent.display().to_string(),
                detail: format!(
                    "parent directory does not exist (create it with systemd-tmpfiles or pick an existing path); \
                     stat error: {e}"
                ),
            });
        }
        Err(e) => {
            return Err(StreamError::UdsParentInsecure {
                path: path.display().to_string(),
                parent: parent.display().to_string(),
                detail: format!("stat failed: {e}"),
            });
        }
    };
    if md.file_type().is_symlink() {
        return Err(StreamError::UdsParentInsecure {
            path: path.display().to_string(),
            parent: parent.display().to_string(),
            detail: "parent directory is a symlink".into(),
        });
    }
    if !md.is_dir() {
        return Err(StreamError::UdsParentInsecure {
            path: path.display().to_string(),
            parent: parent.display().to_string(),
            detail: "parent path is not a directory".into(),
        });
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = md.permissions().mode();
        // World-writable without the sticky bit lets any local user swap the
        // just-bound socket before the post-`bind` chmod lands; narrow,
        // non-default risk -- warn and proceed rather than refuse to boot.
        if (mode & 0o002) != 0 && (mode & 0o1000) == 0 {
            tracing::warn!(
                target: "stream_io",
                path = %path.display(),
                parent = %parent.display(),
                mode = format!("{:#o}", mode & 0o7777),
                "uds parent dir is world-writable without the sticky bit; \
                 any local user can swap the socket before chmod. Tighten to \
                 0o755/0o750 or set the sticky bit (/tmp-shape 0o1777).",
            );
        }
    }
    Ok(())
}

/// Apply Unix mode bits to the bound socket (call after `bind_uds`). This
/// `chmod(path)` is safe iff [`bind_uds`]'s parent-dir confinement passed
/// (`fchmod(fd)` can't be used -- UDS fds reject it). Re-stats with
/// `symlink_metadata` and refuses non-sockets as defence-in-depth; a
/// socket->socket swap still chmods the attacker's socket, only parent
/// confinement closes that.
pub fn set_uds_permissions(path: &std::path::Path, mode: u32) -> Result<(), StreamError> {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::symlink_metadata(path) {
        Ok(md) => {
            let ft = md.file_type();
            #[cfg(unix)]
            reject_if_not_socket(path, &ft)?;
        }
        Err(e) => {
            return Err(StreamError::UdsPerms {
                path: path.display().to_string(),
                source: e,
            });
        }
    }
    let perms = std::fs::Permissions::from_mode(mode);
    std::fs::set_permissions(path, perms).map_err(|e| StreamError::UdsPerms {
        path: path.display().to_string(),
        source: e,
    })
}

#[cfg(test)]
mod tests {
    // Stages tempfiles via `std::fs::write`; clippy.toml production constraint
    // doesn't apply in tests.
    #![allow(clippy::disallowed_methods)]
    use super::*;

    #[test]
    fn router_construction_smoke() {
        let r = StreamRouter::new();
        let _audio_tx = r.audio_tx();
        let _infer_tx = r.infer_tx();
        assert_eq!(*r.audio_subscribers().borrow(), 0);
        assert_eq!(*r.infer_subscribers().borrow(), 0);
        let _ = r.router();
    }

    #[test]
    fn subscriber_guard_round_trip() {
        let (tx, rx) = watch::channel(0usize);
        let tx = Arc::new(tx);
        {
            let _g1 = SubscriberGuard::try_acquire(tx.clone(), 0).expect("uncapped");
            assert_eq!(*rx.borrow(), 1);
            {
                let _g2 = SubscriberGuard::try_acquire(tx.clone(), 0).expect("uncapped");
                assert_eq!(*rx.borrow(), 2);
            }
            assert_eq!(*rx.borrow(), 1);
        }
        assert_eq!(*rx.borrow(), 0);
    }

    /// Cap enforced atomically; rejected acquire doesn't bump.
    #[test]
    fn subscriber_guard_caps_concurrent() {
        let (tx, rx) = watch::channel(0usize);
        let tx = Arc::new(tx);
        let g1 = SubscriberGuard::try_acquire(tx.clone(), 2).expect("first slot");
        let g2 = SubscriberGuard::try_acquire(tx.clone(), 2).expect("second slot");
        assert_eq!(*rx.borrow(), 2);
        assert!(
            SubscriberGuard::try_acquire(tx.clone(), 2).is_none(),
            "third acquire must reject at cap"
        );
        assert_eq!(*rx.borrow(), 2);
        drop(g1);
        assert_eq!(*rx.borrow(), 1);
        let _g3 = SubscriberGuard::try_acquire(tx.clone(), 2).expect("after free");
        assert_eq!(*rx.borrow(), 2);
        drop(g2);
    }

    /// Strict rejects header-omitting requests, relaxed accepts; both accept a
    /// request that lists the token.
    #[test]
    fn enforce_subprotocol_policy_controlled() {
        let strict = TransportPolicy::default();
        assert!(strict.require_subprotocol, "default must be strict");
        let relaxed = TransportPolicy {
            require_subprotocol: false,
            ..TransportPolicy::default()
        };

        let empty = HeaderMap::new();
        assert_eq!(
            enforce_subprotocol(&empty, &strict),
            Err(StatusCode::BAD_REQUEST),
            "strict policy must reject when no Sec-WebSocket-Protocol",
        );
        assert_eq!(
            enforce_subprotocol(&empty, &relaxed),
            Ok(()),
            "relaxed policy must admit when no Sec-WebSocket-Protocol",
        );

        let mut listed = HeaderMap::new();
        listed.insert(
            header::SEC_WEBSOCKET_PROTOCOL,
            WS_SUBPROTOCOL.parse().expect("hv"),
        );
        assert_eq!(enforce_subprotocol(&listed, &strict), Ok(()));
        assert_eq!(enforce_subprotocol(&listed, &relaxed), Ok(()));

        let mut other = HeaderMap::new();
        other.insert(
            header::SEC_WEBSOCKET_PROTOCOL,
            "acoustics.v0, soap".parse().expect("hv"),
        );
        assert_eq!(
            enforce_subprotocol(&other, &strict),
            Err(StatusCode::BAD_REQUEST),
        );
        assert_eq!(enforce_subprotocol(&other, &relaxed), Ok(()));
    }

    /// The four transient errnos classify transient, everything else (e.g.
    /// EBADF) fatal.
    #[test]
    fn is_transient_accept_error_classifies_correctly() {
        for &errno in &[libc::EMFILE, libc::ENFILE, libc::EAGAIN, libc::ECONNABORTED] {
            let e = std::io::Error::from_raw_os_error(errno);
            assert!(
                is_transient_accept_error(&e),
                "errno {errno} ({e}) must classify as transient",
            );
        }
        for &errno in &[libc::EBADF, libc::EINVAL, libc::ENOTSOCK, libc::EFAULT] {
            let e = std::io::Error::from_raw_os_error(errno);
            assert!(
                !is_transient_accept_error(&e),
                "errno {errno} ({e}) must NOT classify as transient",
            );
        }
        let other = std::io::Error::other("synthetic");
        assert!(
            !is_transient_accept_error(&other),
            "non-OS error must not classify as transient",
        );
    }

    /// Counters start zero and clones share state (api + WS handler views must
    /// agree).
    #[test]
    fn broadcast_lag_counters_start_zero_and_share_state() {
        let r = StreamRouter::new();
        let view_a = r.lag_counters();
        let view_b = r.lag_counters();
        assert_eq!(view_a.audio_messages_dropped(), 0);
        assert_eq!(view_a.inference_messages_dropped(), 0);

        view_a
            .inference
            .fetch_add(7, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(view_b.inference_messages_dropped(), 7);
        assert_eq!(view_b.audio_messages_dropped(), 0);
    }

    /// Applies the requested mode to a real socket (staged via `bind_uds`).
    #[tokio::test(flavor = "current_thread")]
    async fn set_uds_permissions_applies_mode() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.sock");
        let _listener = bind_uds(&path).await.expect("bind");
        set_uds_permissions(&path, 0o666).expect("chmod");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o666, "got mode 0o{mode:o}");
    }

    /// Rejects a regular file (`UdsPathNotSocket`) and leaves it byte-identical
    /// -- never silently destroyed.
    #[tokio::test(flavor = "current_thread")]
    async fn bind_uds_refuses_to_unlink_regular_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("not-a-socket");
        let payload = b"do not delete me";
        std::fs::write(&path, payload).expect("stage regular file");
        let err = bind_uds(&path)
            .await
            .expect_err("bind_uds must refuse a regular file");
        match err {
            StreamError::UdsPathNotSocket { kind, path: p } => {
                assert_eq!(kind, "regular file");
                assert!(p.contains("not-a-socket"), "path in err: {p}");
            }
            other => panic!("wrong error variant: {other:?}"),
        }
        let still_there = std::fs::read(&path).expect("file still exists");
        assert_eq!(still_there, payload, "file contents were modified");
    }

    /// Rejects a symlink even when its target is a socket (following at unlink
    /// time is the classic TOCTOU vector).
    #[tokio::test(flavor = "current_thread")]
    async fn bind_uds_refuses_symlink_at_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let target = dir.path().join("target.sock");
        let _real = std::os::unix::net::UnixListener::bind(&target).expect("stage socket");
        let link = dir.path().join("link.sock");
        std::os::unix::fs::symlink(&target, &link).expect("stage symlink");
        let err = bind_uds(&link)
            .await
            .expect_err("bind_uds must refuse a symlink");
        match err {
            StreamError::UdsPathNotSocket { kind, .. } => {
                assert_eq!(kind, "symlink");
            }
            other => panic!("wrong error variant: {other:?}"),
        }
        let read_link = std::fs::read_link(&link).expect("symlink still present");
        assert_eq!(read_link, target);
        assert!(
            std::fs::symlink_metadata(&target).is_ok(),
            "target file should still exist",
        );
    }

    /// Removes a stale socket (previous-daemon-crash case) and rebinds.
    #[tokio::test(flavor = "current_thread")]
    async fn bind_uds_removes_stale_socket_and_binds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("stale.sock");
        {
            let _stale = std::os::unix::net::UnixListener::bind(&path).expect("stage stale");
        }
        assert!(
            std::fs::symlink_metadata(&path).is_ok(),
            "stale socket file should exist before bind",
        );
        let listener = bind_uds(&path)
            .await
            .expect("bind_uds must remove stale socket and rebind");
        assert!(std::fs::symlink_metadata(&path).is_ok());
        drop(listener);
    }

    /// A world-writable-no-sticky parent is warned, not rejected: bind still
    /// succeeds. Sticky `/tmp`-shape passes the same way.
    #[tokio::test(flavor = "current_thread")]
    async fn bind_uds_accepts_unconfined_parent_dir() {
        use std::os::unix::fs::PermissionsExt;
        let outer = tempfile::tempdir().expect("tempdir");
        // Sub-dir, not the temp root (chmod there would break cleanup).
        let unsafe_parent = outer.path().join("public-rw");
        std::fs::create_dir(&unsafe_parent).expect("mkdir unsafe");
        std::fs::set_permissions(&unsafe_parent, std::fs::Permissions::from_mode(0o777))
            .expect("chmod unsafe");
        let path = unsafe_parent.join("test.sock");
        let listener = bind_uds(&path)
            .await
            .expect("bind_uds must accept (warn) a world-writable parent");
        drop(listener);
        std::fs::remove_file(&path).expect("clear socket between binds");
        std::fs::set_permissions(&unsafe_parent, std::fs::Permissions::from_mode(0o1777))
            .expect("chmod sticky");
        let listener = bind_uds(&path)
            .await
            .expect("bind_uds must accept sticky-bit parent");
        drop(listener);
    }

    /// Refuses to chmod a non-socket file (defence-in-depth).
    #[tokio::test(flavor = "current_thread")]
    async fn set_uds_permissions_refuses_non_socket() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("not-a-socket");
        std::fs::write(&path, b"hello").expect("stage regular file");
        let err = set_uds_permissions(&path, 0o600)
            .expect_err("set_uds_permissions must refuse a regular file");
        match err {
            StreamError::UdsPathNotSocket { kind, .. } => {
                assert_eq!(kind, "regular file");
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    /// TOML round-trips and differs from `default()`; empty falls back to
    /// default. Non-default `require_subprotocol = false` so a deser-skip would
    /// fail the round-trip.
    #[test]
    fn transport_policy_toml_round_trips() {
        let toml_input = r#"
max_connections_per_stream = 16
require_subprotocol = false
"#;
        let parsed: TransportPolicy = toml::from_str(toml_input).expect("toml load");
        assert_eq!(parsed.max_connections_per_stream, 16);
        assert!(!parsed.require_subprotocol);
        assert_ne!(
            parsed,
            TransportPolicy::default(),
            "populated case must not equal default; otherwise a deser regression \
             could silently pass",
        );

        let serialized = toml::to_string(&parsed).expect("serialize");
        let reparsed: TransportPolicy = toml::from_str(&serialized).expect("re-parse");
        assert_eq!(parsed, reparsed, "TOML round-trip must preserve all fields");

        let empty: TransportPolicy = toml::from_str("").expect("empty toml");
        assert_eq!(empty, TransportPolicy::default());
    }

    /// Poll until `receiver_count() >= want`: a broadcast send before any
    /// receiver subscribes is dropped, so tests must wait first.
    async fn wait_for_receiver_count(tx: &broadcast::Sender<Bytes>, want: usize) {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if tx.receiver_count() >= want {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "receiver_count {} never reached {want}",
                tx.receiver_count(),
            );
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }

    /// Each broadcast `Bytes` round-trips byte-for-byte through
    /// `decode_length_prefixed` on a connected consumer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn serve_inference_uds_streams_length_prefixed_frames() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out.sock");
        let listener = bind_uds(&path).await.expect("bind");
        let (infer_tx, _) = broadcast::channel::<Bytes>(16);
        let shutdown = CancellationToken::new();
        let server = tokio::spawn({
            let infer_tx = infer_tx.clone();
            let shutdown = shutdown.clone();
            async move { serve_inference_uds(listener, infer_tx, shutdown).await }
        });

        let mut client = UnixStream::connect(&path).await.expect("connect");
        // Subscribe before publishing, else the broadcast drops it.
        wait_for_receiver_count(&infer_tx, 1).await;

        // Opaque payload: server frames without decoding.
        let payload = Bytes::from_static(b"\x01\x02\x03 envelope-ish bytes");
        infer_tx.send(payload.clone()).expect("send");
        let decoded = decode_length_prefixed(&mut client)
            .await
            .expect("decode one frame");
        assert_eq!(decoded, payload, "frame must round-trip byte-for-byte");

        let payload2 = Bytes::from_static(b"second");
        infer_tx.send(payload2.clone()).expect("send 2");
        let decoded2 = decode_length_prefixed(&mut client).await.expect("decode 2");
        assert_eq!(decoded2, payload2);

        shutdown.cancel();
        let _ = server.await;
    }

    /// Caps consumers at [`INFERENCE_UDS_MAX_CONNS`]; an over-cap connection is
    /// closed without subscribing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn serve_inference_uds_caps_connections() {
        use tokio::io::AsyncReadExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out.sock");
        let listener = bind_uds(&path).await.expect("bind");
        let (infer_tx, _) = broadcast::channel::<Bytes>(16);
        let shutdown = CancellationToken::new();
        let server = tokio::spawn({
            let infer_tx = infer_tx.clone();
            let shutdown = shutdown.clone();
            async move {
                let _ = serve_inference_uds(listener, infer_tx, shutdown).await;
            }
        });

        let cap = INFERENCE_UDS_MAX_CONNS as usize;
        let mut clients = Vec::with_capacity(cap);
        for _ in 0..cap {
            clients.push(UnixStream::connect(&path).await.expect("connect"));
        }
        wait_for_receiver_count(&infer_tx, cap).await;
        assert_eq!(infer_tx.receiver_count(), cap, "all {cap} must subscribe");

        // Over-cap: server accepts then closes without subscribing.
        let mut extra = UnixStream::connect(&path).await.expect("connect extra");
        let mut buf = [0u8; 1];
        let n = tokio::time::timeout(std::time::Duration::from_secs(2), extra.read(&mut buf))
            .await
            .expect("rejected connection should close promptly")
            .expect("read");
        assert_eq!(n, 0, "over-cap connection must be closed (EOF)");
        assert_eq!(
            infer_tx.receiver_count(),
            cap,
            "over-cap connection must not subscribe",
        );

        shutdown.cancel();
        let _ = server.await;
    }
}
