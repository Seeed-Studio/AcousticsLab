//! Shared WS-over-TCP test scaffolding: spawn a router, connect, and wait until
//! the server observes the subscriber so `tx.send(...)` is delivered. UDS WS
//! handshakes stay hand-rolled per-test (`connect_async` can't speak `unix:`).

#![allow(dead_code)]

use acousticslab::stream_io::{StreamRouter, TransportPolicy, serve_tcp};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message as TtMessage;
use tokio_util::sync::CancellationToken;

/// Per-await timeout: loopback WS round-trips settle in ~1 ms, so 5 s catches
/// CI hangs without flaking on a busy runner.
pub const TEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Live `StreamRouter` + server task; cancels and aborts the task on drop.
pub struct RouterHarness {
    pub router: StreamRouter,
    pub local_addr: std::net::SocketAddr,
    pub audio_tx: tokio::sync::broadcast::Sender<Bytes>,
    pub infer_tx: tokio::sync::broadcast::Sender<Bytes>,
    pub audio_subs: watch::Receiver<usize>,
    pub infer_subs: watch::Receiver<usize>,
    cancel: CancellationToken,
    server: Option<JoinHandle<()>>,
}

impl Drop for RouterHarness {
    fn drop(&mut self) {
        self.cancel.cancel();
        // Abort (not block_on) in Drop: releases the listener; runtime reaps it.
        if let Some(h) = self.server.take() {
            h.abort();
        }
    }
}

/// Spawn a router + TCP listener on `127.0.0.1:0` with a relaxed
/// [`TransportPolicy`] (`require_subprotocol = false`), so a bare
/// `connect_async` is admitted; strict-default tests build their own router.
pub async fn spawn_tcp_router() -> RouterHarness {
    let policy = TransportPolicy {
        require_subprotocol: false,
        ..TransportPolicy::default()
    };
    let router = StreamRouter::with_capacities_and_policy(64, 64, policy);
    let audio_tx = router.audio_tx();
    let infer_tx = router.infer_tx();
    let audio_subs = router.audio_subscribers();
    let infer_subs = router.infer_subscribers();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let local_addr = listener.local_addr().expect("local_addr");
    let cancel = CancellationToken::new();
    let token_srv = cancel.clone();
    let app = router.router();
    let server = tokio::spawn(async move {
        let _ = serve_tcp(listener, app, token_srv).await;
    });

    RouterHarness {
        router,
        local_addr,
        audio_tx,
        infer_tx,
        audio_subs,
        infer_subs,
        cancel,
        server: Some(server),
    }
}

/// Connect a WS client and wait until its subscriber count reaches 1, so a
/// later broadcast on the matching `*_tx` reaches the receiver.
pub async fn connect_tcp_ws(
    harness: &RouterHarness,
    path: &str,
) -> WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>> {
    assert!(
        path == "/stream/audio" || path == "/stream/infer",
        "connect_tcp_ws: unknown path `{path}` (use /stream/audio or /stream/infer)"
    );
    let url = format!("ws://{}{}", harness.local_addr, path);
    let (ws, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");

    let mut subs_rx = if path == "/stream/audio" {
        harness.audio_subs.clone()
    } else {
        harness.infer_subs.clone()
    };
    let waited = timeout(TEST_TIMEOUT, async {
        while *subs_rx.borrow_and_update() == 0 {
            subs_rx.changed().await.expect("subs watch closed");
        }
    })
    .await;
    waited.expect("subscriber count should reach 1 within timeout");
    ws
}

/// Read one binary WS message; panics on timeout, clean close, or non-binary frame.
pub async fn recv_binary<S>(ws: &mut WebSocketStream<S>) -> Bytes
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let next = timeout(TEST_TIMEOUT, ws.next())
        .await
        .expect("ws recv timeout")
        .expect("ws stream closed");
    match next.expect("ws msg") {
        TtMessage::Binary(b) => Bytes::from(b.to_vec()),
        other => panic!("expected Binary frame, got {other:?}"),
    }
}

/// Send a binary WS message (rare client->server case for the server-initiated
/// daemon protocol, but useful for negative tests).
pub async fn send_binary<S>(ws: &mut WebSocketStream<S>, bytes: Bytes)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    timeout(TEST_TIMEOUT, ws.send(TtMessage::Binary(bytes)))
        .await
        .expect("ws send timeout")
        .expect("ws send error");
}
